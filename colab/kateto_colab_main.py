# %% [markdown]
# # Kateto — Plan principal (RWKV7-G1j-2.9B via fla + LoRA) en Colab T4
#
# Correr las celdas de arriba hacia abajo. Presupuesto: ~2.5-3h de las 4h.
#
# NO usa Joluck/RWKV-PEFT ni .pth: usa el modelo oficial en formato
# transformers (fla-hub/RWKV7-G1j-2.9B-20260831, espejo RWKV/RWKV7-G1j-2.9B)
# + peft/trl LoRA estándar. El chat template del modelo trae tool-calling
# nativo (```json {"name":..,"arguments":..}```).
#
# - Celda 1: deps
# - Celda 2: carga del modelo
# - Celda 3: subi tu sft_train.jsonl (618 filas)
# - Celda 4: descarga datasets HF + normaliza
# - Celda 5: arma datasets en formato messages
# - Celda 6: Stage 1 — adaptación de dominio (corpus argentino, LoRA)
# - Celda 7: Stage 2 — SFT final (persona + tool calling, va último)
# - Celda 8: test de inferencia
# - Celda 9: export y download
#
# Si Colab se desconecta: reejecutá 1-5 y seguí desde la celda siguiente.

# %% [1] Dependencias (Colab y Kaggle) — solo instala lo que falta/esta viejo
# Si en Kaggle agregaste los wheels precompilados como dependencies, esto NO los toca.
import importlib.metadata as _md, subprocess, sys

def _vtuple(v):
    return tuple(int(x) for x in __import__("re").findall(r"\d+", v)[:3]) or (0,)

def ensure(pkg, minver):
    try:
        have = _md.version(pkg)
        ok = _vtuple(have) >= _vtuple(minver)
    except _md.PackageNotFoundError:
        have, ok = None, False
    print(f'{pkg}: {have or "NO ESTA"}{"" if ok else f" -> instalando >={minver}"}')
    return ok

_missing = [p for p, mv in [
    ('transformers', '5.0'), ('peft', '0.18'), ('trl', '0.13'),
    ('accelerate', '1.0'), ('torchao', '0.16'),
    ('flash-linear-attention', '0'), ('datasets', '2.20'), ('huggingface_hub', '0.24'),
] if not ensure(p, mv)]
if _missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_missing], check=True)
    print('instalado:', *_missing)
else:
    print('todo ok, nada que instalar')
import os
# T4 x2: Trainer activaria DataParallel y los kernels triton de fla explotan en
# las replicas (autotuner -> NoneType is not a mapping). Una sola GPU.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
print('GPUs visibles para torch:', torch.cuda.device_count(), '(queremos 1)')
assert torch.cuda.is_available(), 'Activá GPU: Kaggle: Settings > Accelerator > GPU T4 x2 / Colab: T4'
print(torch.cuda.get_device_name(0))
CAP = torch.cuda.get_device_capability(0)
if CAP < (7, 0):
    raise RuntimeError(
        f'GPU con compute capability {CAP} (ej: P100 sm_60). El torch de Kaggle '
        '(2.10+cu128) NO soporta sm_60: los kernels CUDA fallan. Cambia el '
        'accelerator a GPU T4 x2 (Settings > Accelerator).')
import fla  # verifica que flash-linear-attention anda

# --- T4/Turing: los kernels triton de fla usan configs con num_stages>=3 que piden
# ~96KB de shared memory, pero Turing tiene 64KB -> OutOfResources en backward.
# fla no tiene env var para esto (check_shared_mem solo conoce Ada/Ampere/Hopper).
# Fix: capar TODOS los autotuners de fla a num_stages=1 cuando la GPU no sea Ampere+.
if torch.cuda.get_device_capability(0) < (8, 0):
    import pkgutil, importlib
    import triton
    def _cap_stages(mod_name):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            return 0
        n = 0
        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, 'configs') and hasattr(obj, 'fn'):  # triton Autotuner
                try:
                    obj.configs = [triton.Config(dict(c.kwargs or {}),
                                                 num_warps=c.num_warps, num_stages=1)
                                   for c in obj.configs if (c.num_stages or 1) <= 1]                                   or [triton.Config({}, num_warps=4, num_stages=1)]
                    n += 1
                except Exception:
                    pass
        return n
    _patched = 0
    import fla.ops as _fla_ops
    for _m in pkgutil.walk_packages(_fla_ops.__path__, 'fla.ops.'):
        _patched += _cap_stages(_m.name)
    print(f'fla: {_patched} kernels autotuneados a num_stages=1 (Turing, shared mem 64KB)')
# (Opcional) token HF para rate limits: en Kaggle: Add-ons > Secrets > HF_TOKEN
import os
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        pass
if not os.environ.get('HF_TOKEN'):
    print('AVISO: sin HF_TOKEN. Kaggle comparte IPs -> 429 rate limit casi seguro.')
    print('Add-ons/Secrets: creá HF_TOKEN con un token de huggingface.co/settings/tokens')

# %% [2] Modelo base en formato transformers
# fla-hub = espejo transformers del pth de BlinkDL (mismo modelo, safetensors).
# Plan B: RWKV/RWKV7-G1j-2.9B-20260831 ; plan C liviano: RWKV/RWKV7-G1j-1.5B.
# OJO: RWKV7 NO soporta sdpa -> 'eager'. Y P100 (sm_60) ya no es soportada por el torch de Kaggle.
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = 'fla-hub/RWKV7-G1j-2.9B-20260831'
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, dtype=torch.float16, device_map='cuda', trust_remote_code=True,  # fp16: T4/P100 sin bf16 nativo
    attn_implementation='eager',
)
model.config.use_cache = False
# L2Wrap de fla (penalizacion anti-overconfidence para bf16) devuelve el loss como
# view de una custom Function -> choca con el 'loss *= scale' in-place del Trainer.
# En fp16 no la necesitamos: la apagamos.
model.config.use_l2warp = False
# Template limpio sin tags de thinking y con token EOS al final del turno de assistant:
CLEAN_CHAT_TEMPLATE = (
    "{%- set add_generation_prompt = add_generation_prompt | default(false) -%}"
    "{%- set tools = tools | default([], true) -%}"
    "{%- set ns = namespace(system_prompt='') -%}"
    "{%- for message in messages -%}"
        "{%- if message.role == 'system' -%}"
            "{%- set ns.system_prompt = message.content | trim -%}"
        "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.system_prompt or tools | length > 0 -%}"
        "{{ 'System: ' ~ ns.system_prompt }}"
        "{%- if tools | length > 0 -%}"
            "{%- if ns.system_prompt %}{{ '\\n' }}{%- endif -%}"
            "{{ 'Tools:\\n' ~ (tools | tojson) ~ '\\nWhen using a tool, return only a compact JSON function call in a ```json block, like {\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}. The `name` field must be top-level, never inside `arguments`. Do not copy the tool schema into arguments. Otherwise answer normally.' }}"
        "{%- endif -%}"
        "{{ '\\n\\n' }}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
        "{%- if message.role == 'user' -%}"
            "{{ 'User: ' ~ (message.content | trim) ~ '\\n\\n' }}"
        "{%- elif message.role == 'assistant' -%}"
            "{%- generation -%}"
            "{%- set content = message.content | default('', true) | trim -%}"
            "{{ 'Assistant: ' ~ content ~ (eos_token or '<|endoftext|>') ~ '\\n\\n' }}"
            "{%- endgeneration -%}"
        "{%- elif message.role == 'tool' -%}"
            "{{ 'User: Function output:\\n' ~ (message.content | trim) ~ '\\n\\n' }}"
        "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
        "{{ 'Assistant: ' }}"
    "{%- endif -%}"
)
tok.chat_template = CLEAN_CHAT_TEMPLATE
print('template con tools:', 'Tools:' in (tok.chat_template or ''))
import torch
CAP = torch.cuda.get_device_capability(0)
print('GPU compute capability:', CAP)
if CAP < (7, 0):
    print('OJO: Pascal (sm<7.0). Los kernels triton de fla pueden no estar soportados.')
    print('Si la celda 6 falla en el forward, usa T4 en su lugar.')

# %% [3] sft_train.jsonl (persona argentina + Kateto tool calling)
# Kaggle: agregá el dataset con el jsonl (Input) o subilo como dataset
# 'kateto-sft' -> queda en /kaggle/input/kateto-sft/sft_train.jsonl
# Colab: falla al input y usa el uploader.
import json, os, shutil

CANDIDATES = [
    '/kaggle/input/kateto-sft/sft_train.jsonl',
    '/kaggle/input/sft-train/sft_train.jsonl',
]
# buscar en cualquier input montado de Kaggle
for root in ('/kaggle/input',):
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f == 'sft_train.jsonl':
                    CANDIDATES.append(os.path.join(dirpath, f))
DST = '/kaggle/working/sft_train.jsonl' if os.path.isdir('/kaggle/working') else '/content/sft_train.jsonl'
found = next((c for c in CANDIDATES if os.path.isfile(c)), None)
if found:
    shutil.copy(found, DST)
else:
    if os.path.isdir('/kaggle/working'):
        raise FileNotFoundError('Subí sft_train.jsonl como dataset de Kaggle (Input) y reejecutá')
    from google.colab import files
    print('Subi data/sft_out/sft_train.jsonl (618 filas, campos query/response):')
    up = files.upload()
    src = 'sft_train.jsonl' if 'sft_train.jsonl' in up else list(up.keys())[0]
    shutil.move(src, DST)
n = sum(1 for _ in open(DST))
r0 = json.loads(open(DST).readline())
print(f'{DST}: filas={n} keys={sorted(r0.keys())}')
assert {'query', 'response'} <= set(r0), 'faltan query/response'
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content'
SFT_PATH = DST

# %% [4] Datasets HF -> corpus argentino + sft extra
import re, os, random, glob
from datasets import load_dataset
random.seed(42)

def first_text(ex, cand=('text','content','body','article','full_text','news','description','title','document')):
    for c in cand:
        v = ex.get(c)
        if isinstance(v, str) and len(v.strip()) > 40:
            return v.strip()
    return None

HF_TEXT_DATASETS = [
    'samuelandaudreymedianetwork/che-argentina-travel-article-corpus',
    'Villaitech/argentina-news',
    'finiteautomata/news-argentina',
    'jmibarlucea/mistral-argentina-reddit',
    'juanmoisesdelas/adolescentes-argentina-uso-de-redes-sociales-sueno-y-somnolencia-base-recodifica',
]
def load_any(repo):
    """Carga robusta: split train -> sin verificacion -> otros splits -> archivos sueltos."""
    try:
        return load_dataset(repo, split='train')
    except Exception:
        pass
    try:
        return load_dataset(repo, split='train', verification_mode='no_checks')
    except Exception:
        pass
    try:
        from datasets import get_dataset_split_names
        for s in get_dataset_split_names(repo):
            try:
                return load_dataset(repo, split=s, verification_mode='no_checks')
            except Exception:
                continue
    except Exception:
        pass
    # ultimo recurso: bajar archivos sueltos del repo (jsonl/csv/parquet/txt)
    from huggingface_hub import list_repo_files
    for f in list_repo_files(repo, repo_type='dataset'):
        ext = f.rsplit('.', 1)[-1].lower()
        if ext not in ('jsonl', 'json', 'csv', 'parquet', 'txt'):
            continue
        url = f'https://huggingface.co/datasets/{repo}/resolve/main/{f}'
        try:
            if ext in ('jsonl', 'json'):
                return load_dataset('json', data_files=url, split='train')
            if ext == 'csv':
                return load_dataset('csv', data_files=url, split='train')
            if ext == 'parquet':
                return load_dataset('parquet', data_files=url, split='train')
            return load_dataset('text', data_files=url, split='train')
        except Exception:
            continue
    raise RuntimeError('ningun split ni archivo cargable')

# Espejo local de los datasets (../data/raw en la maquina; en Kaggle subilos como
# dataset "kateto-raw" y quedan en /kaggle/input/kateto-raw/<carpeta>/...)
LOCAL_DATASETS = {
    'samuelandaudreymedianetwork/che-argentina-travel-article-corpus': 'che-argentina-travel',
    'Villaitech/argentina-news': 'argentina-news',
    'finiteautomata/news-argentina': 'news-argentina',
    'jmibarlucea/mistral-argentina-reddit': 'argentina-reddit',
    'bertin-project/alpaca-spanish': 'alpaca-spanish-cleaned',
}
EXTRA_LOCAL = ['spanish-roleplay-4.5k', 'rioplatense-audiobooks']  # bonus locales

def find_local_dir(folder):
    roots = ['/kaggle/input', os.path.expanduser('~/data/raw'), 'data/raw', '../data/raw']
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if os.path.basename(dirpath) == folder:
                return dirpath
            # por si el dataset de Kaggle anida: <input>/<nombre>/<carpeta>
            if folder in dirnames:
                return os.path.join(dirpath, folder)
    return None

def load_local_dir(d):
    import itertools
    groups = {}
    for ext in ('parquet', 'jsonl', 'json', 'csv'):
        fs = sorted(glob.glob(os.path.join(d, f'**/*.{ext}'), recursive=True))
        # ignorar .gz emparejados y splits del alpaca si hubo error de parseo: se reintenta por archivo
        if fs:
            groups[ext] = fs
    parts = []
    for ext, fs in groups.items():
        fmt = {'parquet': 'parquet', 'csv': 'csv'}.get(ext, 'json')
        try:
            parts.append(load_dataset(fmt, data_files=fs, split='train'))
        except Exception:
            # mezcla problematica: cargar archivo por archivo y concatenar lo que entre
            for f in fs:
                try:
                    parts.append(load_dataset(fmt, data_files=[f], split='train'))
                except Exception:
                    pass
    # NO concatenar: los shards suelen tener esquemas distintos -> se devuelve lista
    return parts or None

# fallback a HF: OFF por defecto en Kaggle (IPs compartidas -> 429 con esperas de minutos).
# Pone el folder que falte en tu dataset de Kaggle y re-corre; en Colab si queda ON.
USE_HF_FALLBACK = not os.path.isdir('/kaggle/input')
if os.path.isdir('/kaggle/input'):
    print('Inputs de Kaggle montados:')
    for dirpath, dirnames, _ in os.walk('/kaggle/input'):
        for dn in dirnames:
            print('  -', os.path.join(dirpath, dn))

corpus, errors = [], []
alpaca = []
for repo in HF_TEXT_DATASETS:
    try:
        texts = []
        local = LOCAL_DATASETS.get(repo)
        if local:
            d = find_local_dir(local)
            if d:
                parts = load_local_dir(d)
                if parts:
                    print(f'{repo}: LOCAL {d} ({len(parts)} archivos)')
                    for p in parts:
                        texts += [t for t in (first_text(ex) for ex in p) if t]
        if not texts:
            if not USE_HF_FALLBACK:
                print(f'{repo}: SIN copia local y fallback HF desactivado -> se salta. '
                      f'Agregá la carpeta "{local}" a tu dataset de Kaggle.')
                errors.append(repo)
                continue
            ds = load_any(repo)
            texts = [t for t in (first_text(ex) for ex in ds) if t]
        print(f'{repo}: {len(texts)} docs')
        corpus.extend(texts)
    except Exception as e:
        print(f'{repo}: FALLO -> {e}')
        errors.append(repo)

# datasets locales extra (no estan en HF con este nombre o son bonus)
for folder in EXTRA_LOCAL:
    d = find_local_dir(folder)
    if d:
        try:
            for p in (load_local_dir(d) or []):
                texts = [t for t in (first_text(ex) for ex in p) if t]
                print(f'{folder} (local): {len(texts)} docs')
                corpus.extend(texts)
        except Exception as e:
            print(f'{folder}: FALLO -> {e}')

def clean(t):
    t = re.sub(r'\s+', ' ', t).strip()
    return t if len(t) > 80 else None

corpus = [c for c in map(clean, corpus) if c]
random.shuffle(corpus)
with open(f'{WORK}/corpus.txt', 'w') as f:
    f.write('\n\n'.join(corpus))
print('corpus.txt:', len(corpus), 'docs,', os.path.getsize(f'{WORK}/corpus.txt')/1e6, 'MB')
print('fallaron (se sigue igual):', errors)

# alpaca-spanish -> pares q/a
try:
    alp = None
    d = find_local_dir('alpaca-spanish-cleaned') or find_local_dir('alpaca-spanish')
    if d:
        try:
            alp_parts = load_local_dir(d)
            print('alpaca: LOCAL', d, f'({len(alp_parts or [])} archivos)')
        except Exception:
            alp_parts = None
    else:
        alp_parts = None
    if not alp_parts:
        if not USE_HF_FALLBACK:
            print('alpaca-spanish: SIN copia local y fallback HF desactivado -> sin pares. '
                  'Agregá "alpaca-spanish-cleaned" a tu dataset de Kaggle.')
            alp_parts = []
        else:
            try:
                alp_parts = [load_dataset('bertin-project/alpaca-spanish', split='train')]
            except Exception:
                alp_parts = [load_dataset('bertin-project/alpaca-spanish', split='train',
                                          verification_mode='no_checks')]
    alpaca = []
    for alp in alp_parts:
        for ex in alp:
            q = (ex.get('instruction') or '').strip()
            inp = (ex.get('input') or '').strip()
            a = (ex.get('output') or '').strip()
            if q and a:
                if inp:
                    q = f'{q}\n{inp}'
                if len(q) < 800 and len(a) < 2000:
                    alpaca.append({'query': q, 'response': a})
            if len(alpaca) >= 3000:
                break
        if len(alpaca) >= 3000:
            break
    print('alpaca pares:', len(alpaca))
except Exception as e:
    print('alpaca-spanish fallo ->', e)
    alpaca = []

# %% [5] Formato messages (el template del modelo manda: tools nativos)
SYSTEM = ('Sos Kateto, un asistente de equipo con voz argentina, canchero y humano. '
          'Hacés chistes, referenciás cultura y música argentina, y cuando hace falta '
          'llamás a las herramientas disponibles con un json en un bloque ```json.')

def to_messages(row):
    if isinstance(row.get('openai'), dict):
        msgs = row['openai'].get('messages') or []
        if msgs:
            return [{'role': 'system', 'content': SYSTEM}] + [
                {'role': m['role'], 'content': m.get('content') or ''}
                for m in msgs if m.get('role') in ('user', 'assistant', 'tool')
            ]
    q, a = (row.get('query') or '').strip(), (row.get('response') or '').strip()
    if not q or not a:
        return None
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': q},
            {'role': 'assistant', 'content': a}]

sft_rows = []
for line in open(SFT_PATH):
    m = to_messages(json.loads(line))
    if m:
        sft_rows.append(m)
for row in alpaca:
    m = to_messages(row)
    if m:
        sft_rows.append(m)
random.shuffle(sft_rows)
print('SFT ejemplos:', len(sft_rows))

from datasets import Dataset
sft_ds = Dataset.from_dict({'messages': sft_rows})

chunks, cur, size = [], [], 0
for t in corpus:
    cur.append(t); size += len(t)
    if size > 3500:
        chunks.append('\n\n'.join(cur)); cur, size = [], 0
if cur:
    chunks.append('\n\n'.join(cur))
random.shuffle(chunks)
raw_ds = Dataset.from_dict({'text': chunks})
print('chunks dominio:', len(raw_ds))

_test = tok.apply_chat_template(sft_rows[0], tokenize=False)
print(_test[:400])

# kwargs defensivos: filtra args que la version instalada de transformers no acepta
def kw(cls, **kwargs):
    import inspect
    ok = inspect.signature(cls.__init__).parameters
    dropped = [k for k in kwargs if k not in ok]
    if dropped:
        print('args no soportados por', cls.__name__, '->', dropped)
    return {k: v for k, v in kwargs.items() if k in ok}

# warmup: en transformers v5 warmup_ratio fue ELIMINADO; warmup_steps acepta
# int (steps exactos) o float en [0,1) = ratio del total (verificado en source main).
# En 4.x warmup_steps es int-only, pero float 0.1 no rompe (warmup ~0).
def warmup_kw(cls, ratio=0.1, steps=50):
    import inspect
    p = inspect.signature(cls.__init__).parameters
    if 'warmup_ratio' in p:            # transformers 4.x viejo
        return {'warmup_ratio': ratio}
    return {'warmup_steps': ratio}     # transformers v5: float = ratio

# %% [6] Stage 1 — adaptación de dominio (corpus argentino, texto crudo)
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      task_type='CAUSAL_LM',
                      target_modules=['r_proj', 'k_proj', 'v_proj', 'o_proj', 'key', 'value'])
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

def tok_raw(batch):
    # SIN labels: DataCollatorForLanguageModeling(mlm=False) los crea solo
    # (labels = input_ids). Si los precomputamos, el collator los deja ragged
    # (1024 vs 1023) y explota al armar el batch.
    return tok(batch['text'], truncation=True, max_length=1024)

raw_tok = raw_ds.map(tok_raw, batched=True, remove_columns=['text'])

args1 = TrainingArguments(**kw(TrainingArguments,
    output_dir=f'{WORK}/out-stage1', per_device_train_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=1, max_steps=600,
    learning_rate=2e-4, lr_scheduler_type='cosine', **warmup_kw(TrainingArguments, 0.1, 30),
    fp16=True, gradient_checkpointing=True, logging_steps=10,  # fp16: T4/P100 sin bf16 nativo
    save_strategy='no', report_to=[], optim='adamw_torch_fused',
))
tr1 = Trainer(model=model, args=args1, train_dataset=raw_tok,
              data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
tr1.train()
model.save_pretrained(f'{WORK}/out-stage1')
print('Stage 1 listo')

# %% [7] Stage 2 — SFT final: persona + tool calling (último a propósito)
# TRL usa los marcadores {% generation %} del template del modelo para
# calcular loss solo en la respuesta del assistant (assistant_only_loss).
from trl import SFTTrainer, SFTConfig

sft_cfg = SFTConfig(**kw(SFTConfig,
    output_dir=f'{WORK}/out-stage2', per_device_train_batch_size=2,
    gradient_accumulation_steps=8, num_train_epochs=2,
    learning_rate=1e-4, lr_scheduler_type='cosine', **warmup_kw(SFTConfig, 0.1, 50),
    fp16=True, gradient_checkpointing=True, logging_steps=10,  # fp16: T4/P100 sin bf16 nativo
    save_strategy='no', report_to=[], optim='adamw_torch_fused',
    max_length=1024, assistant_only_loss=True,
))
trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=sft_ds,
                     processing_class=tok)
trainer.train()
model.save_pretrained(f'{WORK}/out-stage2')
print('Stage 2 listo — adapter en '+WORK+'/out-stage2')

# %% [8] Test de inferencia
model.eval()
model.config.use_cache = True

def chat(msgs, tools=None, max_new=200):
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True, tools=tools)
    ids = tok(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=True, temperature=0.8, top_p=0.9)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

print(chat([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': 'Kateto, contame algo del asado del domingo y haceme reír'}]))
print(chat([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '¿Qué eventos hay disponibles en el bus?'}],
           tools=[{'name': 'list_events', 'description': 'Lista eventos del bus',
                   'arguments': {}}]))

# %% [9] Export + download
model.save_pretrained(f'{WORK}/kateto-final')
tok.save_pretrained(f'{WORK}/kateto-final')
import os
os.system(f'cd {WORK} && zip -r kateto-lora.zip kateto-final')
print('Resultado:', os.path.abspath(f'{WORK}/kateto-lora.zip'))
if WORK == '/content':
    from google.colab import files
    files.download(f'{WORK}/kateto-lora.zip')
else:
    print('Kaggle: el zip queda en /kaggle/working/kateto-lora.zip (Output del notebook)')
# El zip trae el modelo con el LoRA mergeado (save_pretrained de peft).
# Cargar local: AutoModelForCausalLM.from_pretrained('/ruta/kateto-final') + fla.
