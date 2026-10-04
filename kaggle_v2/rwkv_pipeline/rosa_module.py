#!/usr/bin/env python3
"""
Módulo ROSA (Retrieval On Soft Attention) para RWKV-8 / RWKV-7 Híbrido.
Proporciona memoria asociativa de recuperación causal exacta sobre el contexto
para resolver el talón de Aquiles de las RNNs: nombres de funciones, tokens exactos y sintaxis JSON.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def step_infer(self, current_hidden, memory_k=None, memory_v=None):
        """
        Paso recurrente O(1) para inferencia token a token con buffer acumulativo.
        current_hidden: (1, 1, hidden_dim)
        """
        q = self.proj_q(current_hidden) # (1, 1, retrieval_dim)
        k = self.proj_k(current_hidden)
        v = self.proj_v(current_hidden)

        if memory_k is None:
            memory_k = k
            memory_v = v
        else:
            memory_k = torch.cat([memory_k, k], dim=1)
            memory_v = torch.cat([memory_v, v], dim=1)

        scale = 1.0 / (self.retrieval_dim ** 0.5)
        scores = torch.einsum("bid,bjd->bij", q, memory_k) * scale
        attn = F.softmax(scores, dim=-1)
        retrieved = torch.einsum("bij,bjd->bid", attn, memory_v)
        
        rosa_logits = self.out_head(retrieved)
        gate_weight = torch.sigmoid(self.gate(current_hidden))
        
        return rosa_logits, gate_weight, memory_k, memory_v
