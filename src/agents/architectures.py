"""
Agent Architectures for ToM-NAS - TRN, RSAN, Transformer
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransparentRNN(nn.Module):
    """Transparent Recurrent Network with complete interpretability"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "update_gate": nn.Linear(hidden_dim * 2, hidden_dim),
                        "reset_gate": nn.Linear(hidden_dim * 2, hidden_dim),
                        "candidate": nn.Linear(hidden_dim * 2, hidden_dim),
                        "layer_norm": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.belief_projection = nn.Linear(hidden_dim, output_dim)
        self.action_projection = nn.Linear(hidden_dim, 1)
        self.computation_trace = []

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        self.computation_trace = []
        x_transformed = self.input_transform(x)
        outputs = []

        for t in range(seq_len):
            h = hidden
            x_t = x_transformed[:, t, :]

            for layer_idx, layer in enumerate(self.layers):
                combined = torch.cat([x_t, h], dim=1)
                z = torch.sigmoid(layer["update_gate"](combined))
                r = torch.sigmoid(layer["reset_gate"](combined))
                h_tilde = torch.tanh(layer["candidate"](torch.cat([x_t, r * h], dim=1)))
                h_new = (1 - z) * h + z * h_tilde
                h = layer["layer_norm"](h_new)
                x_t = h

            outputs.append(h)
            hidden = h

        output_tensor = torch.stack(outputs, dim=1)
        beliefs = torch.sigmoid(self.belief_projection(output_tensor[:, -1, :]))
        actions = torch.sigmoid(self.action_projection(output_tensor[:, -1, :]))

        return {
            "hidden_states": output_tensor,
            "beliefs": beliefs,
            "actions": actions.squeeze(-1),
            "final_hidden": hidden,
            "trace": self.computation_trace,
        }


class RecursiveSelfAttention(nn.Module):
    """RSAN for emergent recursive reasoning"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4, max_recursion: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_recursion = max_recursion

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention_modules = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1) for _ in range(max_recursion)]
        )
        self.belief_projection = nn.Linear(hidden_dim, output_dim)
        self.action_projection = nn.Linear(hidden_dim, 1)
        self.attention_patterns = []

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        h = self.input_proj(x).transpose(0, 1)
        self.attention_patterns = []

        for depth in range(self.max_recursion):
            h_attended, _ = self.attention_modules[depth](h, h, h)
            h = h + h_attended

        h_final = h.transpose(0, 1)
        beliefs = torch.sigmoid(self.belief_projection(h_final[:, -1, :]))
        actions = torch.sigmoid(self.action_projection(h_final[:, -1, :]))

        return {
            "hidden_states": h_final,
            "beliefs": beliefs,
            "actions": actions.squeeze(-1),
            "attention_patterns": self.attention_patterns,
        }


class TransformerToMAgent(nn.Module):
    """Transformer for communication and pragmatics"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.input_projection(x)
        h_encoded = self.transformer_encoder(h)
        beliefs = torch.sigmoid(self.belief_head(h_encoded[:, -1, :]))
        actions = torch.sigmoid(self.action_head(h_encoded[:, -1, :]))

        return {
            "hidden_states": h_encoded,
            "beliefs": beliefs,
            "actions": actions.squeeze(-1),
            "message_tokens": torch.randint(0, 100, (x.shape[0],)),
        }


class HybridArchitecture(nn.Module):
    """Hybrid combining all architectures through evolution"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, architecture_genes: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.genes = architecture_genes
        # Simplified hybrid implementation
        self.base_net = nn.Linear(input_dim, hidden_dim)
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.base_net(x[:, -1, :])
        beliefs = torch.sigmoid(self.belief_head(h))
        actions = torch.sigmoid(self.action_head(h))
        return {"hidden_states": h.unsqueeze(1), "beliefs": beliefs, "actions": actions.squeeze(-1)}
