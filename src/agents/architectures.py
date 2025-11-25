from typing import Dict, List, Optional
"""
Agent Architectures for ToM-NAS - TRN, RSAN, Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TransparentRNN(nn.Module):
    """Transparent Recurrent Network with complete interpretability"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'update_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'reset_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'candidate': nn.Linear(hidden_dim * 2, hidden_dim),
                'layer_norm': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
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
            step_trace = {'timestep': t, 'layers': []}

            for layer_idx, layer in enumerate(self.layers):
                combined = torch.cat([x_t, h], dim=1)
                z = torch.sigmoid(layer['update_gate'](combined))
                r = torch.sigmoid(layer['reset_gate'](combined))
                h_tilde = torch.tanh(layer['candidate'](torch.cat([x_t, r * h], dim=1)))
                h_new = (1 - z) * h + z * h_tilde
                h = layer['layer_norm'](h_new)

                # Record computation trace for interpretability
                layer_trace = {
                    'layer_idx': layer_idx,
                    'update_gate_mean': z.mean().item(),
                    'reset_gate_mean': r.mean().item(),
                    'hidden_norm': h.norm().item(),
                    'update_gate_active': (z > 0.5).float().mean().item(),
                    'reset_gate_active': (r > 0.5).float().mean().item(),
                }
                step_trace['layers'].append(layer_trace)

                x_t = h

            step_trace['final_hidden_norm'] = h.norm().item()
            self.computation_trace.append(step_trace)
            outputs.append(h)
            hidden = h

        output_tensor = torch.stack(outputs, dim=1)
        beliefs = torch.sigmoid(self.belief_projection(output_tensor[:, -1, :]))
        actions = torch.sigmoid(self.action_projection(output_tensor[:, -1, :]))

        return {
            'hidden_states': output_tensor,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'final_hidden': hidden,
            'trace': self.computation_trace
        }

class RecursiveSelfAttention(nn.Module):
    """RSAN for emergent recursive reasoning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 4, max_recursion: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_recursion = max_recursion
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention_modules = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
            for _ in range(max_recursion)
        ])
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
            'hidden_states': h_final,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'attention_patterns': self.attention_patterns
        }

class TransformerToMAgent(nn.Module):
    """Transformer for communication and pragmatics"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, num_heads: int = 4):
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
            'hidden_states': h_encoded,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'message_tokens': torch.randint(0, 100, (x.shape[0],))
        }

class HybridArchitecture(nn.Module):
    """Hybrid combining architecture components through evolution.

    Dynamically builds a network based on gene configuration:
    - Uses recurrent processing (GRU-style) controlled by use_update_gate/use_reset_gate
    - Uses attention mechanism controlled by num_heads
    - Applies layer normalization and dropout based on gene settings
    - Stacks multiple processing layers based on num_layers
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, architecture_genes: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.genes = architecture_genes

        # Extract gene parameters with defaults
        num_layers = architecture_genes.get('num_layers', 2)
        num_heads = architecture_genes.get('num_heads', 4)
        max_recursion = architecture_genes.get('max_recursion', 3)
        use_layer_norm = architecture_genes.get('use_layer_norm', True)
        use_dropout = architecture_genes.get('use_dropout', True)
        dropout_rate = architecture_genes.get('dropout_rate', 0.1)
        use_update_gate = architecture_genes.get('use_update_gate', True)
        use_reset_gate = architecture_genes.get('use_reset_gate', True)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Build hybrid layers dynamically based on genes
        self.hybrid_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict()

            # Recurrent gating (if enabled)
            if use_update_gate:
                layer['update_gate'] = nn.Linear(hidden_dim * 2, hidden_dim)
            if use_reset_gate:
                layer['reset_gate'] = nn.Linear(hidden_dim * 2, hidden_dim)
            layer['candidate'] = nn.Linear(hidden_dim * 2, hidden_dim)

            # Attention mechanism (multi-head attention for ToM recursive reasoning)
            if num_heads > 0:
                layer['attention'] = nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout_rate if use_dropout else 0.0,
                    batch_first=True
                )

            # Normalization and regularization
            if use_layer_norm:
                layer['norm'] = nn.LayerNorm(hidden_dim)
            if use_dropout:
                layer['dropout'] = nn.Dropout(dropout_rate)

            self.hybrid_layers.append(layer)

        # Recursive attention for deeper ToM reasoning (controlled by max_recursion)
        self.recursion_depth = min(max_recursion, 5)
        if self.recursion_depth > 0 and num_heads > 0:
            self.recursive_attention = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate if use_dropout else 0.0, batch_first=True)
                for _ in range(self.recursion_depth)
            ])
        else:
            self.recursive_attention = None

        # Output heads with configurable depth
        belief_head_layers = architecture_genes.get('belief_head_layers', 1)
        action_head_layers = architecture_genes.get('action_head_layers', 1)

        if belief_head_layers > 1:
            belief_layers = []
            for i in range(belief_head_layers - 1):
                belief_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
                ])
            belief_layers.append(nn.Linear(hidden_dim, output_dim))
            self.belief_head = nn.Sequential(*belief_layers)
        else:
            self.belief_head = nn.Linear(hidden_dim, output_dim)

        if action_head_layers > 1:
            action_layers = []
            for i in range(action_head_layers - 1):
                action_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
                ])
            action_layers.append(nn.Linear(hidden_dim, 1))
            self.action_head = nn.Sequential(*action_layers)
        else:
            self.action_head = nn.Linear(hidden_dim, 1)

        # Store configuration for introspection
        self.use_update_gate = use_update_gate
        self.use_reset_gate = use_reset_gate
        self.use_layer_norm = use_layer_norm
        self.use_dropout = use_dropout
        self.attention_patterns = []

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        self.attention_patterns = []

        # Input projection
        h = self.input_proj(x)

        # Process through hybrid layers
        hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = h[:, t, :]

            for layer in self.hybrid_layers:
                combined = torch.cat([x_t, hidden], dim=1)

                # Gated update (GRU-style)
                if self.use_update_gate and 'update_gate' in layer:
                    z = torch.sigmoid(layer['update_gate'](combined))
                else:
                    z = torch.ones(batch_size, self.hidden_dim, device=x.device) * 0.5

                if self.use_reset_gate and 'reset_gate' in layer:
                    r = torch.sigmoid(layer['reset_gate'](combined))
                else:
                    r = torch.ones(batch_size, self.hidden_dim, device=x.device)

                h_tilde = torch.tanh(layer['candidate'](torch.cat([x_t, r * hidden], dim=1)))
                hidden_new = (1 - z) * hidden + z * h_tilde

                # Apply normalization
                if self.use_layer_norm and 'norm' in layer:
                    hidden_new = layer['norm'](hidden_new)

                # Apply dropout
                if self.use_dropout and 'dropout' in layer:
                    hidden_new = layer['dropout'](hidden_new)

                hidden = hidden_new
                x_t = hidden

            outputs.append(hidden)

        output_tensor = torch.stack(outputs, dim=1)

        # Apply recursive attention for ToM depth
        if self.recursive_attention is not None:
            h_attended = output_tensor
            for attn_layer in self.recursive_attention:
                attn_out, attn_weights = attn_layer(h_attended, h_attended, h_attended)
                h_attended = h_attended + attn_out  # Residual connection
                self.attention_patterns.append(attn_weights.detach())
            output_tensor = h_attended

        # Output heads
        final_hidden = output_tensor[:, -1, :]
        beliefs = torch.sigmoid(self.belief_head(final_hidden))
        actions = torch.sigmoid(self.action_head(final_hidden))

        return {
            'hidden_states': output_tensor,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'final_hidden': final_hidden,
            'attention_patterns': self.attention_patterns,
            'gene_config': self.genes
        }
