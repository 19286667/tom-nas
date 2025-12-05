"""
Agent Architectures for ToM-NAS - TRN, RSAN, Transformer, and Hybrids
Complete implementation with transparency tools and evolved combinations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ComputationTrace:
    """Records computation steps for transparency."""
    layer_name: str
    input_state: torch.Tensor
    output_state: torch.Tensor
    gates: Optional[Dict[str, torch.Tensor]] = None
    attention_weights: Optional[torch.Tensor] = None
    timestamp: int = 0


class TransparentRNN(nn.Module):
    """
    Transparent Recurrent Network with complete interpretability.

    Specializes in:
    - Sequential state tracking
    - Belief revision over time
    - Interpretable gate operations

    Key transparency features:
    - All gate values logged
    - Hidden state evolution tracked
    - Belief formation traced
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.arch_type = 'TRN'

        # Input projection
        self.input_transform = nn.Linear(input_dim, hidden_dim)

        # GRU-style layers with transparent gates
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'update_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'reset_gate': nn.Linear(hidden_dim * 2, hidden_dim),
                'candidate': nn.Linear(hidden_dim * 2, hidden_dim),
                'layer_norm': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

        # Output projections
        self.belief_projection = nn.Linear(hidden_dim, output_dim)
        self.action_projection = nn.Linear(hidden_dim, 1)

        # Transparency tracking
        self.computation_trace: List[ComputationTrace] = []
        self.record_trace = True

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None,
                return_trace: bool = False) -> Dict[str, Any]:
        """
        Forward pass with optional trace recording.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            hidden: Initial hidden state [batch, hidden_dim]
            return_trace: Whether to include computation trace

        Returns:
            Dict with beliefs, actions, hidden states, and optional trace
        """
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        if self.record_trace:
            self.computation_trace = []

        # Transform input
        x_transformed = self.input_transform(x)
        outputs = []
        all_gates = []

        for t in range(seq_len):
            h = hidden
            x_t = x_transformed[:, t, :]
            step_gates = {}

            for layer_idx, layer in enumerate(self.layers):
                # Concatenate input and hidden
                combined = torch.cat([x_t, h], dim=1)

                # Compute gates (transparent - we can inspect these)
                z = torch.sigmoid(layer['update_gate'](combined))
                r = torch.sigmoid(layer['reset_gate'](combined))

                # Candidate hidden state
                reset_hidden = torch.cat([x_t, r * h], dim=1)
                h_tilde = torch.tanh(layer['candidate'](reset_hidden))

                # Update hidden state
                h_new = (1 - z) * h + z * h_tilde
                h = layer['layer_norm'](h_new)
                h = layer['dropout'](h)

                # Store gate values for transparency
                step_gates[f'layer_{layer_idx}'] = {
                    'update_gate': z.detach(),
                    'reset_gate': r.detach(),
                    'candidate': h_tilde.detach()
                }

                x_t = h

                # Record trace
                if self.record_trace:
                    self.computation_trace.append(ComputationTrace(
                        layer_name=f'layer_{layer_idx}_t{t}',
                        input_state=combined.detach(),
                        output_state=h.detach(),
                        gates={'z': z.detach(), 'r': r.detach()}
                    ))

            outputs.append(h)
            all_gates.append(step_gates)
            hidden = h

        # Stack outputs
        output_tensor = torch.stack(outputs, dim=1)

        # Compute beliefs and actions from final hidden state
        final_h = output_tensor[:, -1, :]
        beliefs = torch.sigmoid(self.belief_projection(final_h))
        actions = torch.sigmoid(self.action_projection(final_h))

        result = {
            'hidden_states': output_tensor,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'final_hidden': hidden,
            'gates': all_gates if return_trace else None
        }

        if return_trace:
            result['trace'] = self.computation_trace

        return result

    def get_interpretable_state(self, hidden: torch.Tensor) -> Dict[str, float]:
        """Extract interpretable features from hidden state."""
        with torch.no_grad():
            beliefs = torch.sigmoid(self.belief_projection(hidden))
            return {
                'mean_belief': beliefs.mean().item(),
                'max_belief': beliefs.max().item(),
                'belief_entropy': -(beliefs * torch.log(beliefs + 1e-8)).sum().item()
            }


class RecursiveSelfAttention(nn.Module):
    """
    Recursive Self-Attention Network for emergent recursive reasoning.

    Specializes in:
    - Nested belief modeling (I think you think I think...)
    - Self-referential reasoning
    - Attention-based state tracking

    Key features:
    - Multiple recursion depths (up to max_recursion)
    - Attention pattern extraction for analysis
    - Halting mechanism for adaptive computation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 4, max_recursion: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.max_recursion = max_recursion
        self.arch_type = 'RSAN'

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recursive attention modules
        self.attention_modules = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(max_recursion)
        ])

        # Layer norms for each recursion level
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(max_recursion)
        ])

        # Feed-forward networks
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(max_recursion)
        ])

        # Halting mechanism (learned)
        self.halting_layer = nn.Linear(hidden_dim, 1)

        # Output projections
        self.belief_projection = nn.Linear(hidden_dim, output_dim)
        self.action_projection = nn.Linear(hidden_dim, 1)

        # Recursion depth projection (for explicit ToM depth)
        self.recursion_embedding = nn.Embedding(max_recursion + 1, hidden_dim)

        # Attention pattern storage
        self.attention_patterns: List[torch.Tensor] = []
        self.record_attention = True

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, Any]:
        """
        Forward pass with recursive self-attention.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_attention: Whether to return attention patterns

        Returns:
            Dict with beliefs, actions, hidden states, recursion info
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        h = self.input_proj(x)

        if self.record_attention:
            self.attention_patterns = []

        # Track cumulative halt probabilities
        halt_probs = []
        recursion_outputs = []
        actual_depth = self.max_recursion

        # Recursive attention processing
        for depth in range(self.max_recursion):
            # Add recursion depth embedding
            depth_emb = self.recursion_embedding(
                torch.tensor([depth], device=x.device)
            ).unsqueeze(0).expand(batch_size, seq_len, -1)
            h_with_depth = h + depth_emb

            # Self-attention
            h_attended, attn_weights = self.attention_modules[depth](
                h_with_depth, h_with_depth, h_with_depth
            )

            # Store attention patterns
            if self.record_attention:
                self.attention_patterns.append(attn_weights.detach())

            # Residual connection and norm
            h = self.layer_norms[depth](h + h_attended)

            # Feed-forward
            h = h + self.ffn[depth](h)

            recursion_outputs.append(h.clone())

            # Compute halt probability
            halt_logit = self.halting_layer(h[:, -1, :])
            halt_prob = torch.sigmoid(halt_logit)
            halt_probs.append(halt_prob)

            # Adaptive halting (during inference)
            if not self.training and halt_prob.mean() > 0.9:
                actual_depth = depth + 1
                break

        # Final output
        h_final = h
        beliefs = torch.sigmoid(self.belief_projection(h_final[:, -1, :]))
        actions = torch.sigmoid(self.action_projection(h_final[:, -1, :]))

        result = {
            'hidden_states': h_final,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'recursion_depth': actual_depth,
            'halt_probabilities': torch.stack(halt_probs, dim=1) if halt_probs else None,
            'recursion_outputs': recursion_outputs
        }

        if return_attention:
            result['attention_patterns'] = self.attention_patterns

        return result

    def get_recursion_analysis(self) -> Dict[str, Any]:
        """Analyze recursion patterns for ToM depth assessment."""
        if not self.attention_patterns:
            return {}

        return {
            'num_recursion_steps': len(self.attention_patterns),
            'attention_entropy_by_depth': [
                -(p * torch.log(p + 1e-8)).sum(dim=-1).mean().item()
                for p in self.attention_patterns
            ]
        }


class TransformerToMAgent(nn.Module):
    """
    Transformer architecture for communication and pragmatics.

    Specializes in:
    - Language/communication processing
    - Pragmatic inference
    - Context-dependent reasoning

    Key features:
    - Full transformer encoder
    - Communication token generation
    - Contextual belief formation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.arch_type = 'Transformer'

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output heads
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, 1)

        # Communication head (for generating messages)
        self.comm_head = nn.Linear(hidden_dim, hidden_dim)
        self.comm_vocab_size = 100
        self.comm_projection = nn.Linear(hidden_dim, self.comm_vocab_size)

        # Context aggregation
        self.context_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                generate_message: bool = False) -> Dict[str, Any]:
        """
        Forward pass with optional context and message generation.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            context: Optional context tensor [batch, context_len, hidden_dim]
            generate_message: Whether to generate communication tokens

        Returns:
            Dict with beliefs, actions, hidden states, optional message
        """
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        h = self.input_projection(x)
        h = h + self.pos_encoding[:, :seq_len, :].to(h.device)

        # Transform
        h_encoded = self.transformer_encoder(h)

        # Context integration (if provided)
        if context is not None:
            h_contextualized, _ = self.context_attention(
                h_encoded, context, context
            )
            h_encoded = h_encoded + h_contextualized

        # Extract final representation
        final_h = h_encoded[:, -1, :]

        # Compute outputs
        beliefs = torch.sigmoid(self.belief_head(final_h))
        actions = torch.sigmoid(self.action_head(final_h))

        result = {
            'hidden_states': h_encoded,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'final_hidden': final_h
        }

        # Generate message tokens if requested
        if generate_message:
            comm_hidden = torch.tanh(self.comm_head(final_h))
            message_logits = self.comm_projection(comm_hidden)
            message_tokens = torch.argmax(message_logits, dim=-1)
            result['message_tokens'] = message_tokens
            result['message_logits'] = message_logits

        return result


class HybridArchitecture(nn.Module):
    """
    Hybrid architecture combining multiple base architectures.

    Supports evolved combinations:
    - TRN + Transformer: Sequential + contextual
    - RSAN + TRN: Recursive + sequential
    - RSAN + Transformer: Recursive + contextual
    - All three combined

    Architecture is determined by genes dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 architecture_genes: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.genes = architecture_genes
        self.arch_type = 'Hybrid'

        # Parse genes for architecture configuration
        self.use_trn = architecture_genes.get('use_trn', True)
        self.use_rsan = architecture_genes.get('use_rsan', True)
        self.use_transformer = architecture_genes.get('use_transformer', False)

        # Mixing weights (can be evolved)
        self.trn_weight = architecture_genes.get('trn_weight', 0.4)
        self.rsan_weight = architecture_genes.get('rsan_weight', 0.4)
        self.transformer_weight = architecture_genes.get('transformer_weight', 0.2)

        # Component architectures
        num_layers = architecture_genes.get('num_layers', 2)
        num_heads = architecture_genes.get('num_heads', 4)
        max_recursion = architecture_genes.get('max_recursion', 5)
        dropout = architecture_genes.get('dropout_rate', 0.1)

        if self.use_trn:
            self.trn = TransparentRNN(
                input_dim, hidden_dim, hidden_dim,
                num_layers=num_layers, dropout=dropout
            )

        if self.use_rsan:
            self.rsan = RecursiveSelfAttention(
                input_dim, hidden_dim, hidden_dim,
                num_heads=num_heads, max_recursion=max_recursion, dropout=dropout
            )

        if self.use_transformer:
            self.transformer = TransformerToMAgent(
                input_dim, hidden_dim, hidden_dim,
                num_layers=num_layers, num_heads=num_heads, dropout=dropout
            )

        # Fusion mechanism
        fusion_type = architecture_genes.get('fusion_type', 'weighted')
        self.fusion_type = fusion_type

        if fusion_type == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)

        elif fusion_type == 'gated':
            num_components = sum([self.use_trn, self.use_rsan, self.use_transformer])
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * num_components, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_components),
                nn.Softmax(dim=-1)
            )

        # Final output layers
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, 1)

        # Component tracking
        self.component_outputs: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through hybrid architecture.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Dict with beliefs, actions, hidden states, component info
        """
        batch_size, seq_len, _ = x.shape
        component_hiddens = []
        self.component_outputs = {}

        # Process through each component
        if self.use_trn:
            trn_out = self.trn(x)
            trn_h = trn_out['final_hidden']
            component_hiddens.append(('trn', trn_h, self.trn_weight))
            self.component_outputs['trn'] = trn_out

        if self.use_rsan:
            rsan_out = self.rsan(x)
            rsan_h = rsan_out['hidden_states'][:, -1, :]
            component_hiddens.append(('rsan', rsan_h, self.rsan_weight))
            self.component_outputs['rsan'] = rsan_out

        if self.use_transformer:
            trans_out = self.transformer(x)
            trans_h = trans_out['final_hidden']
            component_hiddens.append(('transformer', trans_h, self.transformer_weight))
            self.component_outputs['transformer'] = trans_out

        # Fuse component outputs
        if len(component_hiddens) == 0:
            # Fallback: simple linear
            h_fused = F.relu(nn.Linear(self.input_dim, self.hidden_dim).to(x.device)(x[:, -1, :]))

        elif len(component_hiddens) == 1:
            h_fused = component_hiddens[0][1]

        else:
            h_fused = self._fuse_components(component_hiddens)

        # Compute outputs
        beliefs = torch.sigmoid(self.belief_head(h_fused))
        actions = torch.sigmoid(self.action_head(h_fused))

        return {
            'hidden_states': h_fused.unsqueeze(1),
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'component_outputs': self.component_outputs,
            'fusion_type': self.fusion_type
        }

    def _fuse_components(self, component_hiddens: List[Tuple[str, torch.Tensor, float]]) -> torch.Tensor:
        """Fuse outputs from multiple components."""

        if self.fusion_type == 'weighted':
            # Simple weighted average
            total_weight = sum(w for _, _, w in component_hiddens)
            h_fused = sum(h * (w / total_weight) for _, h, w in component_hiddens)

        elif self.fusion_type == 'attention':
            # Stack and attend
            h_stack = torch.stack([h for _, h, _ in component_hiddens], dim=1)
            h_attended, _ = self.fusion_attention(h_stack, h_stack, h_stack)
            h_fused = self.fusion_norm(h_attended.mean(dim=1))

        elif self.fusion_type == 'gated':
            # Learned gating
            h_concat = torch.cat([h for _, h, _ in component_hiddens], dim=-1)
            gates = self.gate_network(h_concat)
            h_stack = torch.stack([h for _, h, _ in component_hiddens], dim=1)
            h_fused = (h_stack * gates.unsqueeze(-1)).sum(dim=1)

        elif self.fusion_type == 'concat':
            # Concatenate and project
            h_concat = torch.cat([h for _, h, _ in component_hiddens], dim=-1)
            h_fused = F.relu(nn.Linear(h_concat.shape[-1], self.hidden_dim).to(h_concat.device)(h_concat))

        else:
            # Default: mean
            h_fused = torch.stack([h for _, h, _ in component_hiddens], dim=0).mean(dim=0)

        return h_fused

    def get_component_contributions(self) -> Dict[str, float]:
        """Get relative contribution of each component."""
        contributions = {}
        if self.use_trn:
            contributions['trn'] = self.trn_weight
        if self.use_rsan:
            contributions['rsan'] = self.rsan_weight
        if self.use_transformer:
            contributions['transformer'] = self.transformer_weight

        total = sum(contributions.values())
        return {k: v / total for k, v in contributions.items()}


class TRNTransformerHybrid(HybridArchitecture):
    """
    Specialized TRN + Transformer hybrid.

    Combines:
    - TRN's sequential state tracking
    - Transformer's contextual processing
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4):
        genes = {
            'use_trn': True,
            'use_rsan': False,
            'use_transformer': True,
            'trn_weight': 0.5,
            'transformer_weight': 0.5,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'fusion_type': 'gated'
        }
        super().__init__(input_dim, hidden_dim, output_dim, genes)
        self.arch_type = 'TRN_Transformer'


class RSANTRNHybrid(HybridArchitecture):
    """
    Specialized RSAN + TRN hybrid.

    Combines:
    - RSAN's recursive reasoning
    - TRN's sequential tracking
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, max_recursion: int = 5):
        genes = {
            'use_trn': True,
            'use_rsan': True,
            'use_transformer': False,
            'trn_weight': 0.5,
            'rsan_weight': 0.5,
            'num_layers': num_layers,
            'max_recursion': max_recursion,
            'fusion_type': 'attention'
        }
        super().__init__(input_dim, hidden_dim, output_dim, genes)
        self.arch_type = 'RSAN_TRN'


class FullHybrid(HybridArchitecture):
    """
    Full hybrid combining all three architectures.

    Combines:
    - TRN's sequential state tracking
    - RSAN's recursive reasoning
    - Transformer's contextual processing
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4, max_recursion: int = 5):
        genes = {
            'use_trn': True,
            'use_rsan': True,
            'use_transformer': True,
            'trn_weight': 0.33,
            'rsan_weight': 0.34,
            'transformer_weight': 0.33,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_recursion': max_recursion,
            'fusion_type': 'gated'
        }
        super().__init__(input_dim, hidden_dim, output_dim, genes)
        self.arch_type = 'Full_Hybrid'


def create_architecture(arch_type: str, input_dim: int, hidden_dim: int,
                       output_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create architecture by type.

    Args:
        arch_type: One of 'TRN', 'RSAN', 'Transformer', 'Hybrid',
                   'TRN_Transformer', 'RSAN_TRN', 'Full_Hybrid'
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        **kwargs: Additional architecture-specific parameters

    Returns:
        Instantiated architecture module
    """
    if arch_type == 'TRN':
        return TransparentRNN(
            input_dim, hidden_dim, output_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )

    elif arch_type == 'RSAN':
        return RecursiveSelfAttention(
            input_dim, hidden_dim, output_dim,
            num_heads=kwargs.get('num_heads', 4),
            max_recursion=kwargs.get('max_recursion', 5),
            dropout=kwargs.get('dropout', 0.1)
        )

    elif arch_type == 'Transformer':
        return TransformerToMAgent(
            input_dim, hidden_dim, output_dim,
            num_layers=kwargs.get('num_layers', 3),
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1)
        )

    elif arch_type == 'TRN_Transformer':
        return TRNTransformerHybrid(
            input_dim, hidden_dim, output_dim,
            num_layers=kwargs.get('num_layers', 2),
            num_heads=kwargs.get('num_heads', 4)
        )

    elif arch_type == 'RSAN_TRN':
        return RSANTRNHybrid(
            input_dim, hidden_dim, output_dim,
            num_layers=kwargs.get('num_layers', 2),
            max_recursion=kwargs.get('max_recursion', 5)
        )

    elif arch_type == 'Full_Hybrid':
        return FullHybrid(
            input_dim, hidden_dim, output_dim,
            num_layers=kwargs.get('num_layers', 2),
            num_heads=kwargs.get('num_heads', 4),
            max_recursion=kwargs.get('max_recursion', 5)
        )

    elif arch_type == 'Hybrid':
        return HybridArchitecture(
            input_dim, hidden_dim, output_dim,
            kwargs.get('architecture_genes', {})
        )

    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def get_architecture_info() -> Dict[str, str]:
    """Get descriptions of all available architectures."""
    return {
        'TRN': 'Transparent Recurrent Network - sequential state tracking with interpretable gates',
        'RSAN': 'Recursive Self-Attention Network - nested belief modeling with adaptive depth',
        'Transformer': 'Transformer - contextual processing for communication and pragmatics',
        'TRN_Transformer': 'TRN + Transformer hybrid - sequential + contextual',
        'RSAN_TRN': 'RSAN + TRN hybrid - recursive + sequential',
        'Full_Hybrid': 'All three architectures combined with gated fusion',
        'Hybrid': 'Configurable hybrid with evolved architecture genes'
    }
