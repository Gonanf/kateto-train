# %% [markdown]
# # Experimento RWKV-8 chico: tiny RWKV-7 desde cero + capa ROSA
#
# Sesión Colab separada (~1.5-2h, T4). Objetivo: medir si la capa ROSA
# (recuperación exacta de strings, estilo RWKV-8) mejora el punto débil de
# RWKV para Kateto: copiar strings exactos (args de tool calls, IDs).
#
# - Celda 1: deps + clonar BlinkDL/RWKV-LM
# - Celda 2: tarea "copiá el argumento" + corpus argentino
# - Celda 3: binidx con el tokenizador world (make_data.py de RWKV-v5)
# - Celda 4: tiny RWKV-7 L6-D512 desde cero (train_temp)
# - Celda 5: capa ROSA soft (ruta rosa_soft/wind_rosa) entrenada igual
# - Celda 6: evaluación exact-recall
#
# Expectativa honesta: el modelo chico NO va a conversar bien (tiene ~1h de
# pretraining vs 332B tokens del G1). Lo que medimos es exact-recall.

# %% [1] Deps + repos
!pip install -q lightning==2.6.5 deepspeed einops triton rwkv-fla rwkv datasets
!test -d /content/RWKV-LM || git clone --depth 1 https://github.com/BlinkDL/RWKV-LM /content/RWKV-LM
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))

# %% [2] Tarea: "copiá el argumento del tool call exactamente"
import json, random
from datasets import load_dataset
random.seed(42)

TOOL_NAMES = ['list_events', 'create_task', 'get_project', 'send_message',
              'advance_workflow', 'assign_task']
IDS = [f't_{random.randint(10**8, 10**9-1)}' for _ in range(400)]
NAMES = ['WorkMatch', 'Kateto', 'Orion', 'Hermes', 'kanban-general', 'sprint-47']

def sample_tool_doc():
    tool = random.choice(TOOL_NAMES)
    arg = random.choice(IDS + NAMES + [f'{random.randint(1,28)}/0{random.randint(1,9)}/2026'])
    prompt = f'Necesito {tool} con {arg}'
    target = f'```json\n{{"name": "{tool}", "arguments": {{"id": "{arg}"}}}}\n```'
    return f'User: {prompt}\n\nAssistant: {target}'

docs = [sample_tool_doc() for _ in range(8000)]
test_docs = [sample_tool_doc() for _ in range(300)]  # held-out

def first_text(ex, cand=('text','content','body','article','news','description','title')):
    for c in cand:
        v = ex.get(c)
        if isinstance(v, str) and len(v.strip()) > 40:
            return v.strip()
    return None

try:
    ds = load_dataset('finiteautomata/news-argentina', split='train')
    fill = [t for t in (first_text(ex) for ex in ds) if t][:3000]
except Exception as e:
    print('corpus fallo ->', e)
    fill = []

with open('/content/rwkv8_train.jsonl', 'w') as f:
    for d in docs + fill:
        f.write(json.dumps({'text': d}, ensure_ascii=False) + '\n')
with open('/content/rwkv8_test.jsonl', 'w') as f:
    for d in test_docs:
        f.write(json.dumps({'text': d}, ensure_ascii=False) + '\n')
print('train:', len(docs) + len(fill), '| test tool-calls:', len(test_docs))

# %% [3] binidx con el tokenizador world (RWKV-v5/make_data.py)
%cd /content/RWKV-LM/RWKV-v5
!python make_data.py /content/rwkv8_train.jsonl 2 1024
import glob
BIN = sorted(glob.glob('/content/RWKV-LM/RWKV-v5/rwkv8_train*.bin'))
assert BIN, 'make_data no genero el .bin'
BINIDX = BIN[0][:-4]
print('binidx:', BINIDX)

# %% [4] Tiny RWKV-7 L6-D512 desde cero (RWKV-v7/train_temp/train.py)
# 4a) inicializar pesos (train_stage 1)
%cd /content/RWKV-LM/RWKV-v7/train_temp
!python train.py --wandb "" --proj_dir /content/out8 \
  --data_file "{BINIDX}" --data_type binidx --vocab_size 65536 --my_testing x070 \
  --ctx_len 1024 --train_stage 1 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
  --weight_decay 0 --head_size 64 --num_nodes 1 --micro_bsz 1 \
  --n_layer 6 --n_embd 512 --accelerator gpu --devices 1 --precision bf16 \
  --grad_cp 1 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10

# 4b) entrenamiento corto (~1h): corpus argentino + tarea de copia
!python train.py --wandb "" --proj_dir /content/out8 \
  --data_file "{BINIDX}" --data_type binidx --vocab_size 65536 --my_testing x070 \
  --ctx_len 1024 --micro_bsz 4 --accumulate_grad_batches 2 \
  --n_layer 6 --n_embd 512 --head_size 64 --num_nodes 1 \
  --epoch_steps 1500 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
  --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 100 --beta1 0.9 --beta2 0.99 \
  --accelerator gpu --devices 1 --precision bf16 --grad_cp 1
BASE8 = sorted(glob.glob('/content/out8/*.pth'))[-1]
print('modelo tiny:', BASE8)

# %% [5] Capa ROSA (soft retrieval, ruta probada de rosa_soft/wind_rosa)
import torch, torch.nn.functional as F
from torch import nn
import sys
sys.path.insert(0, '/content/RWKV-LM/RWKV-v5')
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tk = TRIE_TOKENIZER('/content/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt')

VOCAB = 65536
class RosaSoft(nn.Module):
    """Q/K/V soft retrieval causal sobre el stream: q_i . k_j recupera v_j."""
    def __init__(self, dim=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, VOCAB, bias=False)
    def forward(self, ids):
        e = self.emb(ids)                       # B,T,D
        q, k, v = self.q(e), self.k(e), self.v(e)
        scores = torch.einsum('bid,bjd->bij', q, k) / q.shape[-1] ** 0.5
        T = ids.shape[1]
        mask = torch.tril(torch.ones(T, T, device=ids.device), -1)  # causal
        scores = scores.masked_fill(mask == 0, -1e9)
        ctx = scores.softmax(-1) @ v            # retrieval del contexto previo
        return self.head(ctx + e)               # retrieval + residuo

def encode_pad(texts, T=256):
    out = []
    for t in texts:
        ids = tk.encode(t)[:T]
        ids = ids + [0] * (T - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long).cuda()

def split_doc(doc):
    p, tgt = doc.split('\n\nAssistant: ')
    return p + '\n\nAssistant: ', tgt

rosa = RosaSoft().cuda()
opt = torch.optim.AdamW(rosa.parameters(), lr=3e-4)
B = 8
for step in range(400):
    batch = random.sample(docs, B)
    texts = [split_doc(d)[0] + split_doc(d)[1] for d in batch]
    ids = encode_pad(texts)
    logits = rosa(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), ids[:, 1:].reshape(-1),
                           ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(step, float(loss))

# %% [6] Evaluación exact-recall
def exact_match(model, n=200):
    ok = 0
    for d in test_docs[:n]:
        prompt, target = split_doc(d)
        ids = encode_pad([prompt])
        tgt_ids = tk.encode(target)
        with torch.no_grad():
            seq = ids.clone()
            gen = []
            for _ in range(len(tgt_ids)):
                logits = model(seq[:, -256:])[:, -1]
                nxt = logits.argmax(-1, keepdim=True)
                gen.append(int(nxt))
                seq = torch.cat([seq, nxt], 1)
        ok += int(gen == tgt_ids)
    return ok, n

ok, n = exact_match(rosa)
print(f'ROSA soft — copia exacta: {ok}/{n} = {ok/n:.1%}')
# Comparación: el tiny RWKV del paso 4 se evalúa igual con el demo de
# inferencia de RWKV-v7 (rwkv_v7_demo_fast.py cargando BASE8). Si ROSA gana
# en copia exacta, vale integrarla como capa de retrieval sobre el G1j-2.9B.
