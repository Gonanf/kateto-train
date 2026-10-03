#!/usr/bin/env python3
"""
Motor de Inferencia de Kateto en RWKV-7 con Swap de Estados Modulados.
Permite alternar entre personalidades al vuelo (0 ms overhead) cargando
el archivo de estado correspondiente sin tocar los pesos del modelo.
"""

import os
import sys
import argparse
import copy
import hashlib
import time
from pathlib import Path

# ROCm pre-import: torch caches cuda availability at import, so the HIP
# override must be in the environment BEFORE `import torch` runs.
# setdefault keeps an explicit start.sh export intact.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "RWKV-PEFT"))

try:
    from rwkv_pipeline._stop_helpers import ends_with, count_consecutive_ngram_repeats
except ImportError:  # invocacion como script directo (python rwkv_pipeline/infer_kateto.py)
    from _stop_helpers import ends_with, count_consecutive_ngram_repeats

# Clave reservada dentro del `.pth` de estado para la metadata de la base sobre
# la que se entreno. Los estados viejos (2026-09-06) no la tienen -> no verificados.
META_KEY = "__kateto_meta__"

# Centinelas de silencio del dataset (turno de la voz). El modelo entrena con
# estos marcadores como respuesta; con allow_no_response=False se convierten en
# el texto de fallback.
SILENCE_SENTINELS = ("<|no_response|>", "<|wait|>")


def base_fingerprint(base_path: "str | None"):
    """Huella corta de la base: sha256(resolve|mtime_ns|size) — barato, sin hashear el .pth.

    El tamaño y el mtime del artefacto base no cambian entre entrenamientos del
    estado (el orquestador no reescribe el .pth base), asi que es suficiente para
    distinguir 'entrenado contra rwkv_kateto_base' de 'entrenado contra
    rwkv_kateto_base_plus_0.4b'.
    """
    if not base_path:
        return None
    p = Path(base_path)
    if not p.exists():
        return None
    st = p.stat()
    digest = hashlib.sha256(f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    return f"base-{digest}"


def get_device() -> str:
    # Mirrors Kateto/kateto/providers/rwkv_rocm.py _get_or_create_engine logic.
    cuda_available = torch.cuda.is_available()
    rocm_active = bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")) or bool(os.environ.get("ROCM_HOME"))
    if cuda_available:
        return "cuda"
    if rocm_active:
        print("AVISO: torch.cuda.is_available()=False pero ROCm env activo — forzando device=cuda")
        return "cuda"
    return "cpu"

# ==============================================================================
# TOKENIZADOR OFICIAL RWKV
# ==============================================================================
class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[:line.index(' ')])
                x = eval(line[line.index(' '):line.rindex(' ')])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(line[line.rindex(' '):])
                sorted_tokens.append(x)
                self.idx2token[idx] = x

        self.token2idx = {v: k for k, v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for s in reversed(sorted_tokens):
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> List[int]:
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i : i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decodeBytes(self, tokens: List[int]) -> bytes:
        return b''.join(map(lambda i: self.idx2token.get(i, b''), tokens))

    def encode(self, src: str) -> List[int]:
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return self.decodeBytes(tokens).decode('utf-8', errors='replace')


# ==============================================================================
# OPERADORES RWKV-7 RNN
# ==============================================================================
def time_mixing__(layer_id: int, H: int, N: int, x, x_prev, v_first, state,
                  x_r, x_w, x_k, x_v, x_a, x_g,
                  w0, w1, w2, a0, a1, a2, v0, v1, v2,
                  g1, g2, k_k, k_a, r_k,
                  kw, vw, rw, ow, ln_w, ln_b):
    xx = x_prev - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = rw @ xr
    w = torch.tanh(xw @ w1) @ w2
    k = kw @ xk
    v = vw @ xv
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = F.normalize(kk.view(H, N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a - 1) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = w0 + w.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w))

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    out = F.group_norm(out.view(1, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H * N)
    out = out + ((r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    return ow @ (out * g), x, state, v_first

def channel_mixing__(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x

time_mixing = torch.jit.script(time_mixing__)
channel_mixing = torch.jit.script(channel_mixing__)

class RWKV_RNN(torch.jit.ScriptModule):
    def __init__(self, model_path: str, n_layer: int, n_embd: int, head_size: int = 64, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.head_size = head_size
        self.dtype = dtype
        self.device = device

        print(f"Cargando pesos de RWKV-7 desde {model_path}...")
        z = torch.load(model_path, map_location=device)
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if k.endswith('att.w0'):
                z[k] = z[k].float()
            else:
                z[k] = z[k].to(dtype=dtype)
            z[k] = z[k].squeeze()
            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']
        self.z = z

    @torch.jit.script_method
    def _step(self, token: int, state: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            z = self.z
            x = z['emb.weight'][token]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln1.weight'], bias=z[bbb + 'ln1.bias'])
                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = time_mixing(
                    i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                    z[att + 'x_r'], z[att + 'x_w'], z[att + 'x_k'], z[att + 'x_v'], z[att + 'x_a'], z[att + 'x_g'],
                    z[att + 'w0'], z[att + 'w1'], z[att + 'w2'], z[att + 'a0'], z[att + 'a1'], z[att + 'a2'],
                    z[att + 'v0'], z[att + 'v1'], z[att + 'v2'], z[att + 'g1'], z[att + 'g2'],
                    z[att + 'k_k'], z[att + 'k_a'], z[att + 'r_k'],
                    z[att + 'key.weight'], z[att + 'value.weight'], z[att + 'receptance.weight'], z[att + 'output.weight'],
                    z[att + 'ln_x.weight'], z[att + 'ln_x.bias']
                )
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln2.weight'], bias=z[bbb + 'ln2.bias'])
                xx, state[i * 3 + 2] = channel_mixing(xx, state[i * 3 + 2], z[ffn + 'x_k'], z[ffn + 'key.weight'], z[ffn + 'value.weight'])
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            logits = z['head.weight'] @ x
            # `x` sale del `layer_norm` final, o sea es la entrada del head: es lo
            # que ROSA necesita para su rama en paralelo.
            return logits, state, x

    @torch.jit.script_method
    def forward(self, token: int, state: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits, state, _ = self._step(token, state)
        return logits, state

def sample_logits(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.7, occurrence: dict = None, alpha_presence: float = 0.5, alpha_frequency: float = 0.4) -> int:
    if occurrence:
        for k, v in occurrence.items():
            logits[k] -= (alpha_presence + v * alpha_frequency)
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits.float(), dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0.0
    return int(torch.multinomial(probs, num_samples=1).item())

# ==============================================================================
# PIPELINE DE INFERENCIA
# ==============================================================================
class KatetoInferenceEngine:
    def __init__(self, model_path: str, vocab_path: str, device="cuda",
                 rosa: bool = False, rosa_retrieval_dim: int = 128,
                 rosa_head: str = ""):
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = RWKV_TOKENIZER(vocab_path)

        checkpoint = torch.load(model_path, map_location="cpu")
        self.n_embd = checkpoint["emb.weight"].shape[1]
        self.vocab_size = checkpoint["head.weight"].shape[0]
        self.n_layer = 0
        while f"blocks.{self.n_layer}.att.r_k" in checkpoint:
            self.n_layer += 1
        self.head_size = checkpoint["blocks.0.att.r_k"].shape[1]
        del checkpoint

        if device == "cuda" and torch.cuda.is_available():
            free_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if self.n_embd >= 2560 and free_vram < 6.0:
                print(f"AVISO: GPU VRAM ({free_vram:.1f} GB) es insuficiente para modelo 2.9B (~6 GB requeridos). Usando CPU automáticamente.")
                device = "cpu"
                self.device = "cpu"
                self.dtype = torch.float32

        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        try:
            self.model = RWKV_RNN(model_path, self.n_layer, self.n_embd, self.head_size, dtype=self.dtype, device=self.device)
        except Exception as e:
            if device == "cuda":
                print(f"AVISO: carga en cuda falló ({type(e).__name__}: {e}) — cayendo a cpu")
                self.device = "cpu"
                self.dtype = torch.float32
                device = "cpu"
                torch.set_num_threads(os.cpu_count() or 4)
                self.model = RWKV_RNN(model_path, self.n_layer, self.n_embd, self.head_size, dtype=self.dtype, device="cpu")
            else:
                raise
        self.states = {}
        self.base_path = str(Path(model_path).resolve())
        self.base_fingerprint = base_fingerprint(model_path)

        # ROSA: rama paralela al head, apagada por defecto. Con `rosa=False` no se
        # importa el modulo ni se instancia nada, y `_forward` es el `forward` crudo.
        self.rosa = None
        self.rosa_memory = None
        if rosa:
            try:
                from rwkv_pipeline.rosa_module import RosaAssociativeMemory
            except ImportError:  # invocacion como script directo (ver _stop_helpers)
                from rosa_module import RosaAssociativeMemory
            self.rosa = RosaAssociativeMemory(
                vocab_size=self.vocab_size, hidden_dim=self.n_embd,
                retrieval_dim=rosa_retrieval_dim).to(
                    device=self.device, dtype=self.dtype).eval()
            for p in self.rosa.parameters():
                p.requires_grad_(False)
            n_rosa = sum(p.numel() for p in self.rosa.parameters())
            if rosa_head:
                # El head se entrena contra una arch y un vocab concretos. Si no
                # matchean, `load_state_dict` falla con una forma rarisima o peor:
                # carga parcial. Chequeo explicito, con el nombre del campo.
                ck = torch.load(rosa_head, map_location="cpu")
                sd = ck["state_dict"] if "state_dict" in ck else ck
                for campo, mio, suyo in (
                        ("n_embd", self.n_embd, ck.get("n_embd")),
                        ("vocab_size", self.vocab_size, ck.get("vocab_size")),
                        ("retrieval_dim", rosa_retrieval_dim, ck.get("retrieval_dim"))):
                    if suyo is not None and suyo != mio:
                        raise SystemExit(
                            f"[Kateto] el head ROSA de '{rosa_head}' fue entrenado con "
                            f"{campo}={suyo} pero el modelo que estas cargando tiene "
                            f"{campo}={mio}. Son de otro tamaño de modelo: midiendo "
                            f"cualquier cosa. No se carga.")
                self.rosa.load_state_dict(sd)
                print(f"[Kateto] ROSA encendida con PESOS ENTRENADOS de '{rosa_head}' "
                      f"(step {ck.get('step', '?')}, +{n_rosa:,} params): la rama suma "
                      f"gate*rosa_logits sobre los logits del head.", file=sys.stderr)
            else:
                print(f"[Kateto] ROSA encendida en paralelo al head: +{n_rosa:,} params "
                      f"inference-only. OJO: pesos RANDOM sin entrenar (no se paso "
                      f"rosa_head) -- la rama suma gate*rosa_logits a los logits, pero "
                      f"todavia no hay pesos que aporten nada, asi que el texto puede "
                      f"salir identico al de la base.", file=sys.stderr)

        if device == "cuda":
            # ponytail: one probe forward; full HIP failure taxonomy if more failure modes appear
            try:
                probe = self.build_initial_state("none")
                self.model.forward(0, probe)
            except Exception as e:
                print(f"AVISO: forward en cuda falló ({type(e).__name__}: {e}) — HIP roto en este box, cayendo a cpu")
                self.device = "cpu"
                self.dtype = torch.float32
                torch.set_num_threads(os.cpu_count() or 4)
                self.model = RWKV_RNN(model_path, self.n_layer, self.n_embd, self.head_size, dtype=self.dtype, device="cpu")

    def load_state(self, state_path: str, name: str = "seco", force: bool = False) -> bool:
        s = torch.load(state_path, map_location=self.device)
        meta = s.get(META_KEY) if isinstance(s, dict) else None
        meta_base = meta.get("base") if isinstance(meta, dict) else None
        cur = self.base_fingerprint
        if meta_base and cur and meta_base != cur:
            if force:
                print(f"[Kateto] estado '{name}' entrenado para base {meta_base}; cargando {cur} — FORZADO por flag.", file=sys.stderr)
            else:
                print(f"[Kateto] estado '{name}' entrenado para base {meta_base}; estas cargando {cur}. "
                      f"Uso --voice none o reentreñá el estado.", file=sys.stderr)
                return False
        elif not meta_base:
            # Estado viejo (2026-09-06) sin metadata: no verificado. Avisa, no rompe.
            print(f"[Kateto] estado '{name}' sin metadata de base (no verificado): se carga igual.", file=sys.stderr)
        # Chequeo de cabezas sobre el primer tensor real (ignora la metadata no-tensor).
        first_t = next((v for v in s.values() if hasattr(v, 'shape')), None)
        if first_t is not None:
            s_heads = first_t.shape[0]
            if s_heads != self.model.n_head:
                print(f"[Kateto] El estado '{state_path}' tiene {s_heads} cabezas pero el modelo base tiene "
                      f"{self.model.n_head} cabezas. Incompatible (pertenece a otro tamaño de modelo, ej: 0.4B vs 2.9B). "
                      f"Omitiendo carga.", file=sys.stderr)
                return False
        print(f"Cargando estado modulado '{name}' desde: {state_path}")
        self.states[name] = s
        return True

    def build_initial_state(self, voice: str = "none") -> List[torch.Tensor]:
        H = self.n_embd // self.head_size
        N = self.head_size
        state = [None] * (self.n_layer * 3)
        loaded_s = self.states.get(voice)

        for i in range(self.n_layer):
            state[i * 3 + 0] = torch.zeros(self.n_embd, dtype=self.dtype, device=self.device)
            key_s0 = f"blocks.{i}.att.time_state"
            if loaded_s and key_s0 in loaded_s:
                s_val = loaded_s[key_s0].to(dtype=torch.float, device=self.device)
                if s_val.ndim == 2:
                    s_val = s_val.view(H, N, N)
                state[i * 3 + 1] = s_val.clone()
            else:
                state[i * 3 + 1] = torch.zeros((H, N, N), dtype=torch.float, device=self.device)
            state[i * 3 + 2] = torch.zeros(self.n_embd, dtype=self.dtype, device=self.device)
        return state

    def format_chat_prompt(self, prompt: str, voice: str = "none", history: "list[tuple[str, str]] | None" = None, max_history: int = 10) -> str:
        # ponytail: text-concat history, re-encoded each turn; carry RNN state across turns if context grows
        voice = "seco" if voice == "none" else voice
        parts = []
        for u, a in (history or [])[-max_history:]:
            parts.append(f"<|im_user|>{u.strip()}<|im_end|>\n<|im_start|>{voice}\n{a.strip()}<|im_end|>")
        parts.append(f"<|im_user|>{prompt.strip()}<|im_end|>\n<|im_start|>{voice}\n")
        return "".join(parts)

    def _forward(self, tok: int, state: List[torch.Tensor]):
        """Un token del RNN, con la rama ROSA mezclada encima si esta encendida.

        Apagado devuelve exactamente lo que devuelve `RWKV_RNN.forward`: mismo
        metodo, mismos tensores. Encendido consulta la memoria KV incremental
        (`rosa_memory`) y suma `gate * rosa_logits` sobre los logits del head.
        """
        if self.rosa is None:
            out, state = self.model.forward(tok, state)
            return out, state
        out, state, hidden = self.model._step(tok, state)
        rosa_logits, gate, self.rosa_memory = self.rosa.step_infer(
            hidden.view(1, 1, -1), self.rosa_memory)
        return out + gate.view(-1) * rosa_logits.view(-1), state

    def generate(self, prompt: str, voice: str = "none", max_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.65, alpha_presence: float = 0.6, alpha_frequency: float = 0.5, history: "list[tuple[str, str]] | None" = None, max_history: int = 4, allow_no_response: bool = False, ngram_n: int = 3, ngram_repeat_max: int = 12) -> str:
        state = self.build_initial_state("seco" if voice == "none" else voice)
        formatted = self.format_chat_prompt(prompt, voice, history, max_history)
        prompt_tokens = self.tokenizer.encode(formatted)

        self.rosa_memory = None
        out = None
        for tok in prompt_tokens:
            out, state = self._forward(tok, state)

        im_end_ids = self.tokenizer.encode("<|im_end|>")
        im_user_ids = self.tokenizer.encode("<|im_user|>")
        generated_tokens = []
        # Inicializar occurrence con tokens del prompt del usuario para desincentivar que repita la frase como eco
        prompt_user_tokens = self.tokenizer.encode(prompt.strip())
        occurrence = {t: 1 for t in prompt_user_tokens if t not in (0, 11, 61)}
        stop_reason = None
        for step in range(max_tokens):
            logits = out.clone()
            # Ojo: el hack viejo (logits[61] -= 15 en step 0) penalizaba el token
            # equivocado. Medicion: <|no_response|> -> [61,125,2073,96,54316,125,63]
            # y <|wait|> -> [61,125,27444,125,63]: son MULTItoken, sin id unico, y
            # el 61 es el '<' compartido por todos los marcadores (<|im_end|>, etc).
            # Penalizar un logit no puede apuntar al centinela -> se elimino; el
            # corte por <|im_end|>/<|im_user|> y la comparacion de SILENCE_SENTINELS
            # abajo ya cubren el caso.
            tok = sample_logits(logits, temperature=temperature, top_p=top_p, occurrence=occurrence, alpha_presence=alpha_presence, alpha_frequency=alpha_frequency)
            if tok == 0:
                stop_reason = "eos"
                break
            generated_tokens.append(tok)
            occurrence[tok] = occurrence.get(tok, 0) + 1
            # Corte por ids: la cola de tokens generados termina en los ids del marcador
            # (mas robusto y mas barato que re-decodear el string en cada paso).
            if ends_with(generated_tokens, im_end_ids) or ends_with(generated_tokens, im_user_ids):
                stop_reason = "id_end"
                break
            # Guarda de repeticion por n-grama: el ultimo n-grama se repitio
            # ngram_repeat_max veces seguidas -> generacion degenerada en loop.
            if count_consecutive_ngram_repeats(generated_tokens, ngram_n) + 1 >= ngram_repeat_max:
                stop_reason = "repetition"
                break
            text_so_far = self.tokenizer.decode(generated_tokens)
            if "<tool_call>" in text_so_far:
                stop_reason = "marker"
                break
            if "\nUser:" in text_so_far or "\n\nUser:" in text_so_far:
                stop_reason = "marker"
                break
            if "\nAssistant:" in text_so_far or "\n\nAssistant:" in text_so_far:
                stop_reason = "marker"
                break
            if "</tool_call>" in text_so_far:
                stop_reason = "marker"
                break
            if "<|endoftext|>" in text_so_far:
                stop_reason = "marker"
                break
            out, state = self._forward(tok, state)
        else:
            stop_reason = "max_tokens"
        if stop_reason is None:
            stop_reason = "max_tokens"
        print(f"[Kateto] generacion cortada: stop={stop_reason} (tokens={len(generated_tokens)})", file=sys.stderr)

        full_output = self.tokenizer.decode(generated_tokens)
        for stop_marker in ["<tool_call>", "<|im_end|>", "<|im_user|>", "\n\nUser:", "\nUser:", "\n\nAssistant:", "\nAssistant:", "</tool_call>", "<|endoftext|>"]:
            if stop_marker in full_output:
                full_output = full_output.split(stop_marker)[0]
        cleaned = full_output.strip()
        if not allow_no_response and (cleaned in SILENCE_SENTINELS or not cleaned):
            return "Che, no te entendí bien, ¿me repetís?"
        return cleaned


def get_latest_checkpoint(files: list) -> "Path | None":
    import re
    def get_epoch(p):
        m = re.search(r'rwkv-(\d+)\.pth', str(p))
        return int(m.group(1)) if m else -1
    return max(files, key=get_epoch) if files else None


def load_voice_state(engine, voice: str, state_file: "str | None" = None, force: bool = False) -> str:
    if voice == "none" and not state_file:
        return "none"
    s_path = state_file
    v_name = voice if voice != "none" else "custom"
    if not s_path and voice != "none":
        s_dir = PROJECT / "out/rwkv_states" / voice
        if s_dir.exists():
            states = list(s_dir.glob("*.pth"))
            best = get_latest_checkpoint(states)
            if best:
                s_path = str(best)
    if s_path and Path(s_path).exists():
        if engine.load_state(s_path, name=v_name, force=force):
            return v_name
    if voice not in ("none", "seco"):
        seco_dir = PROJECT / "out/rwkv_states/seco"
        seco_states = list(seco_dir.glob("*.pth")) if seco_dir.exists() else []
        best_seco = get_latest_checkpoint(seco_states)
        if best_seco and engine.load_state(str(best_seco), name=v_name, force=force):
            print(f"AVISO: estado de '{voice}' ausente, fallback explícito a 'seco' ({best_seco}).", file=sys.stderr)
            return v_name
        else:
            print(f"AVISO: No se encontró estado compatible para '{voice}'. Se usará base neutra.", file=sys.stderr)
    else:
        print(f"AVISO: No se encontró estado compatible para '{voice}'. Se usará base neutra.", file=sys.stderr)
    return "none"


def save_voice_state_with_base(state_dict: dict, out_path: str, base_path: "str | None" = None) -> str:
    """Persiste un estado junto a la identidad de la base sobre la que se entrenó.

    El orquestador de reentrenamiento DEBE llamar a esta función (en vez de
    `torch.save(sd, path)`) para que `load_state`/`load_voice_state` puedan
    verificar que el estado no se está cargando sobre la base equivocada.

    Escribe en un `.tmp` y hace `os.replace` (escritura atomica). Devuelve el
    fingerprint de base persistido ("" si no se pudo calcular).
    """
    payload = dict(state_dict)
    payload[META_KEY] = {
        "base": base_fingerprint(base_path) if base_path else None,
        "base_path": str(Path(base_path).resolve()) if base_path else None,
        "fmt": 1,
    }
    tmp = str(out_path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, str(out_path))
    return payload[META_KEY]["base"] or ""


def chat_loop(engine, voice: str, max_history: int, allow_no_response: bool = False):
    try:
        import readline  # noqa: edición de línea e historial con flechas gratis
    except ImportError:
        pass
    history: "list[tuple[str, str]]" = []
    print("[Kateto] chat interactivo — comandos: /salir /limpiar /voz <nombre>")
    while True:
        try:
            msg = input("\nVos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Kateto] nos vemos!")
            break
        if not msg:
            continue
        if msg in ("/salir", "/exit", "/quit"):
            print("[Kateto] nos vemos!")
            break
        if msg == "/limpiar":
            history.clear()
            print("[Kateto] historial limpio.")
            continue
        if msg.startswith("/voz"):
            parts = msg.split()
            if len(parts) != 2 or parts[1] not in ("none", "seco", "streamer", "jane", "doktor", "whisperer"):
                print("[Kateto] uso: /voz <none|seco|streamer|jane|doktor|whisperer>")
                continue
            voice = load_voice_state(engine, parts[1])
            history.clear()
            print(f"[Kateto] ahora hablo como '{voice}'. Historial reiniciado.")
            continue
        ans = engine.generate(msg, voice=voice, history=history, max_history=max_history, allow_no_response=allow_no_response)
        print(f"[Kateto]: {ans}")
        if ans and ans not in SILENCE_SENTINELS:
            history.append((msg, ans))


def main():
    parser = argparse.ArgumentParser(description="Inferencia Kateto RWKV-7 con State Swap")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo base .pth")
    parser.add_argument("--voice", type=str, default="seco", choices=["none", "seco", "streamer", "jane", "doktor", "whisperer"], help="Voz/Personalidad a cargar")
    parser.add_argument("--state_file", type=str, default=None, help="Ruta al checkpoint .pth de estado")
    parser.add_argument("--force_state", action="store_true", help="Cargar el estado aun si su metadata de base no coincide con la base cargada")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Dispositivo para inferencia (cuda o cpu)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt a evaluar (one-shot). Sin --prompt ni --test_suite entra a chat interactivo.")
    parser.add_argument("--allow_no_response", action="store_true", help="Dejar pasar los centinelas de silencio (<|no_response|>/<|wait|>) como respuesta, en vez de convertirlos en el texto de fallback")
    parser.add_argument("--chat", action="store_true", help="Forzar modo chat interactivo con historial temporal")
    parser.add_argument("--max_history", type=int, default=10, help="Turnos previos a incluir como contexto en chat")
    parser.add_argument("--test_suite", action="store_true", help="Ejecutar batería de preguntas clave")
    parser.add_argument("--rosa", action="store_true", help="Encender ROSA (memoria asociativa) como rama paralela al head. Apagado por defecto: sin el flag la salida es identica a la de siempre.")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        base_dir = PROJECT / "out/rwkv_kateto_base"
        if base_dir.exists():
            models = sorted(list(base_dir.glob("*.pth")), key=os.path.getmtime)
            if models:
                model_path = str(models[-1])
        if not model_path:
            model_path = str(PROJECT / "models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth")

    vocab_path = PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
    if not vocab_path.exists():
        vocab_path = PROJECT / "RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt"

    device = args.device or os.environ.get("INFER_DEVICE") or get_device()
    engine = KatetoInferenceEngine(model_path, str(vocab_path), device=device, rosa=args.rosa)
    print(f"[Kateto] device={engine.device} (pedido: {device}) model={model_path}")

    voice = load_voice_state(engine, args.voice, args.state_file, force=args.force_state)

    if args.test_suite:
        test_prompts = [
            "Requeson",
            "Que es python?",
            "Dime quien eres y que haces",
            "Che boludo, quién sos y qué hacés acá?",
            "Que opinas de Kristina?"
        ]
        print("\n" + "=" * 60)
        print(f">>> EJECUTANDO BATERÍA DE PRUEBAS DE CONTROL [Voz: {voice}] <<<")
        print("=" * 60)
        for p in test_prompts:
            print(f"\n[Prompt]: {p}")
            ans = engine.generate(p, voice=voice)
            print(f"[Kateto]: {ans}")
            print("-" * 40)
    elif args.chat or args.prompt is None:
        chat_loop(engine, voice, args.max_history, allow_no_response=args.allow_no_response)
    else:
        print(f"\n[Prompt]: {args.prompt}")
        ans = engine.generate(args.prompt, voice=voice, allow_no_response=args.allow_no_response)
        print(f"[Kateto]: {ans}\n")

if __name__ == "__main__":
    main()

