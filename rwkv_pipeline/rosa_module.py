#!/usr/bin/env python3
"""
Módulo ROSA (Retrieval On Soft Attention) para RWKV-8 / RWKV-7 Híbrido.
Proporciona memoria asociativa de recuperación causal exacta sobre el contexto
para resolver el talón de Aquiles de las RNNs: nombres de funciones, tokens exactos y sintaxis JSON.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_KV_INIT_CAP = 512


class RosaAssociativeMemory(nn.Module):
    """
    Capa de recuperación asociativa (ROSA) para inferencia y decoding en RWKV.
    """
    def __init__(self, vocab_size=65536, hidden_dim=1024, retrieval_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.retrieval_dim = retrieval_dim
        
        # Proyecciones desde el espacio oculto del modelo hacia el espacio de retrieval
        self.proj_q = nn.Linear(hidden_dim, retrieval_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, retrieval_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, retrieval_dim, bias=False)
        self.out_head = nn.Linear(retrieval_dim, vocab_size, bias=False)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward_train(self, hidden_states):
        """
        Paso forward causal paralelo para entrenamiento.
        hidden_states: (Batch, Seq_Len, hidden_dim)
        """
        B, T, D = hidden_states.shape
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)

        # Matriz de atención causal sobre el stream pasado
        scale = 1.0 / (self.retrieval_dim ** 0.5)
        scores = torch.einsum("bid,bjd->bij", q, k) * scale
        
        # Máscara causal estricta
        mask = torch.tril(torch.ones(T, T, device=hidden_states.device), diagonal=0)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # Recuperación de contexto
        retrieved = torch.einsum("bij,bjd->bid", attn, v)
        rosa_logits = self.out_head(retrieved)
        gate_weight = torch.sigmoid(self.gate(hidden_states))
        
        return rosa_logits, gate_weight

    def step_infer(self, current_hidden, memory=None):
        """
        Paso recurrente token a token con buffer KV incremental preasignado.

        Agregar un token es O(1): el historial se escribe in-place por indice de
        escritura, sin re-copiar el buffer. Queda O(T) por token solo por el scan
        de atencion, inevitable en softmax sobre memoria creciente.

        current_hidden: (B, 1, hidden_dim)
        memory: tupla (K, V, pos) de la llamada previa, o None para empezar.
        Devuelve (rosa_logits, gate_weight, memory).
        """
        q = self.proj_q(current_hidden)
        k = self.proj_k(current_hidden)
        v = self.proj_v(current_hidden)

        if memory is None:
            shape = (current_hidden.shape[0], _KV_INIT_CAP, self.retrieval_dim)
            memory = (k.new_empty(shape), v.new_empty(shape), 0)
        keys, values, pos = memory

        if pos == keys.shape[1]:
            keys = torch.cat([keys, keys.new_empty(keys.shape)], dim=1)
            values = torch.cat([values, values.new_empty(values.shape)], dim=1)
        keys[:, pos:pos + 1] = k
        values[:, pos:pos + 1] = v
        memory = (keys, values, pos + 1)

        # el token actual entra en su propia softmax: es la mascara `diagonal=0` de
        # `forward_train`, asi que la fila t coincide con la fila t de ahi.
        keys, values = keys[:, :pos + 1], values[:, :pos + 1]
        scale = 1.0 / (self.retrieval_dim ** 0.5)
        attn = F.softmax(q @ keys.transpose(-1, -2) * scale, dim=-1)
        retrieved = attn @ values

        rosa_logits = self.out_head(retrieved)
        gate_weight = torch.sigmoid(self.gate(current_hidden))

        return rosa_logits, gate_weight, memory


class RosaHead(nn.Module):
    """ROSA como rama paralela al head de salida: `logits + gate * rosa_logits`.

    Envuelve un `nn.Linear` head existente (no lo reemplaza), asi el camino
    original queda intacto y ROSA suma encima. El gate lo produce el modulo
    mismo (`sigmoid(gate(h))`, un escalar por token).
    """

    def __init__(self, head: nn.Linear, retrieval_dim: int = 128):
        super().__init__()
        self.head = head
        self.rosa = RosaAssociativeMemory(
            vocab_size=head.out_features, hidden_dim=head.in_features,
            retrieval_dim=retrieval_dim)
        # sin este cast, forward_train revienta con un ckpt bf16: los params de
        # ROSA nacen float32 (medido en docs/rosa-diagnostico.md, seccion 1).
        self.rosa.to(device=head.weight.device, dtype=head.weight.dtype)

    def forward(self, x):
        rosa_logits, gate = self.rosa.forward_train(x)
        return self.head(x) + gate * rosa_logits


def attach_rosa(model, retrieval_dim: int = 128) -> RosaHead:
    """Engancha ROSA en `model.head`. Idempotente: si ya esta, no hace nada."""
    existing = getattr(model, "head", None)
    if isinstance(existing, RosaHead):
        return existing
    model.head = RosaHead(existing, retrieval_dim=retrieval_dim)
    return model.head
