"""
Supernet Architecture for Efficient Neural Architecture Search

This module implements a supernet that encompasses multiple architecture configurations
as subnetworks with shared weights. Training the supernet once allows evaluating
any subnetwork by inheriting weights, reducing evaluation time from hours to minutes.

Key concepts:
- Elastic depth: Different numbers of layers can be selected
- Elastic width: Different hidden dimensions can be selected
- Weight sharing: Subnetworks inherit weights from the supernet
- Once-for-All: Train once, evaluate many architectures

Based on the Once-for-All methodology (MIT) and DyNAS-T framework (Intel Labs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import copy


@dataclass
class SubnetConfig:
    """Configuration for a subnetwork extracted from the supernet."""
    arch_type: str  # 'trn', 'rsan', 'transformer'
    num_layers: int
    hidden_dim: int
    num_heads: int = 4  # For attention-based architectures
    dropout: float = 0.1
    use_skip_connections: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'arch_type': self.arch_type,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'use_skip_connections': self.use_skip_connections
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubnetConfig':
        return cls(**data)


class ElasticLinear(nn.Module):
    """
    Linear layer that supports elastic width.

    The full layer has max_in_features and max_out_features.
    At runtime, a subset of features can be used by specifying active dimensions.
    """

    def __init__(
        self,
        max_in_features: int,
        max_out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features

        self.weight = nn.Parameter(torch.Tensor(max_out_features, max_in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(max_out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Active dimensions (default to full)
        self.active_in = max_in_features
        self.active_out = max_out_features

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_active_dims(self, in_features: int, out_features: int):
        """Set active dimensions for elastic evaluation."""
        self.active_in = min(in_features, self.max_in_features)
        self.active_out = min(out_features, self.max_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use only active dimensions
        weight = self.weight[:self.active_out, :self.active_in]
        bias = self.bias[:self.active_out] if self.bias is not None else None

        # Handle input dimension mismatch
        if x.shape[-1] > self.active_in:
            x = x[..., :self.active_in]
        elif x.shape[-1] < self.active_in:
            # Pad with zeros
            padding = torch.zeros(*x.shape[:-1], self.active_in - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)

        return F.linear(x, weight, bias)


class ElasticMultiheadAttention(nn.Module):
    """
    Multi-head attention that supports elastic width and number of heads.
    """

    def __init__(
        self,
        max_embed_dim: int,
        max_num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_embed_dim = max_embed_dim
        self.max_num_heads = max_num_heads
        self.dropout = dropout

        # Projections
        self.q_proj = ElasticLinear(max_embed_dim, max_embed_dim)
        self.k_proj = ElasticLinear(max_embed_dim, max_embed_dim)
        self.v_proj = ElasticLinear(max_embed_dim, max_embed_dim)
        self.out_proj = ElasticLinear(max_embed_dim, max_embed_dim)

        # Active dimensions
        self.active_embed_dim = max_embed_dim
        self.active_num_heads = max_num_heads

    def set_active_dims(self, embed_dim: int, num_heads: int):
        """Set active dimensions."""
        self.active_embed_dim = embed_dim
        self.active_num_heads = num_heads

        self.q_proj.set_active_dims(embed_dim, embed_dim)
        self.k_proj.set_active_dims(embed_dim, embed_dim)
        self.v_proj.set_active_dims(embed_dim, embed_dim)
        self.out_proj.set_active_dims(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        embed_dim = self.active_embed_dim
        num_heads = self.active_num_heads
        head_dim = embed_dim // num_heads

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights


class ElasticRecurrentCell(nn.Module):
    """
    GRU-style recurrent cell with elastic hidden dimension.
    """

    def __init__(self, max_input_dim: int, max_hidden_dim: int):
        super().__init__()
        self.max_input_dim = max_input_dim
        self.max_hidden_dim = max_hidden_dim

        # GRU gates
        self.update_gate = ElasticLinear(max_input_dim + max_hidden_dim, max_hidden_dim)
        self.reset_gate = ElasticLinear(max_input_dim + max_hidden_dim, max_hidden_dim)
        self.candidate = ElasticLinear(max_input_dim + max_hidden_dim, max_hidden_dim)
        self.layer_norm = nn.LayerNorm(max_hidden_dim)

        self.active_input_dim = max_input_dim
        self.active_hidden_dim = max_hidden_dim

    def set_active_dims(self, input_dim: int, hidden_dim: int):
        """Set active dimensions."""
        self.active_input_dim = input_dim
        self.active_hidden_dim = hidden_dim

        combined_dim = input_dim + hidden_dim
        self.update_gate.set_active_dims(combined_dim, hidden_dim)
        self.reset_gate.set_active_dims(combined_dim, hidden_dim)
        self.candidate.set_active_dims(combined_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Truncate inputs to active dimensions
        if x.shape[-1] > self.active_input_dim:
            x = x[..., :self.active_input_dim]
        if h.shape[-1] > self.active_hidden_dim:
            h = h[..., :self.active_hidden_dim]

        combined = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))

        combined_reset = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.candidate(combined_reset))

        h_new = (1 - z) * h + z * h_tilde

        # Apply layer norm only to active dimensions
        h_new = self.layer_norm(h_new)

        return h_new


class ToMSupernet(nn.Module):
    """
    Supernet for Theory of Mind architectures.

    This supernet encompasses:
    - TRN: Transparent Recurrent Networks with elastic depth and width
    - RSAN: Recursive Self-Attention Networks with elastic attention
    - Transformer: Standard transformer with elastic configuration

    All subnetworks share weights, enabling efficient architecture evaluation.
    """

    # Search space bounds
    MAX_LAYERS = 5
    MAX_HIDDEN_DIM = 256
    MAX_HEADS = 8
    MIN_LAYERS = 1
    MIN_HIDDEN_DIM = 64
    MIN_HEADS = 2

    def __init__(
        self,
        input_dim: int = 181,
        output_dim: int = 181,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Input projection (shared)
        self.input_proj = ElasticLinear(input_dim, self.MAX_HIDDEN_DIM)

        # TRN components
        self.trn_cells = nn.ModuleList([
            ElasticRecurrentCell(self.MAX_HIDDEN_DIM, self.MAX_HIDDEN_DIM)
            for _ in range(self.MAX_LAYERS)
        ])

        # RSAN/Transformer attention layers
        self.attention_layers = nn.ModuleList([
            ElasticMultiheadAttention(self.MAX_HIDDEN_DIM, self.MAX_HEADS, dropout)
            for _ in range(self.MAX_LAYERS)
        ])

        # Feedforward layers for transformer
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                ElasticLinear(self.MAX_HIDDEN_DIM, self.MAX_HIDDEN_DIM * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                ElasticLinear(self.MAX_HIDDEN_DIM * 4, self.MAX_HIDDEN_DIM)
            )
            for _ in range(self.MAX_LAYERS)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.MAX_HIDDEN_DIM)
            for _ in range(self.MAX_LAYERS * 2)  # Two per layer for pre-norm
        ])

        # Output heads
        self.belief_head = ElasticLinear(self.MAX_HIDDEN_DIM, output_dim)
        self.action_head = ElasticLinear(self.MAX_HIDDEN_DIM, 1)

        # Current configuration
        self.active_config: Optional[SubnetConfig] = None

    def set_active_config(self, config: SubnetConfig):
        """Set the active subnet configuration."""
        self.active_config = config

        # Configure all elastic components
        hidden_dim = config.hidden_dim
        num_heads = config.num_heads

        self.input_proj.set_active_dims(self.input_dim, hidden_dim)

        for cell in self.trn_cells:
            cell.set_active_dims(hidden_dim, hidden_dim)

        for attn in self.attention_layers:
            attn.set_active_dims(hidden_dim, num_heads)

        for ff in self.ff_layers:
            ff[0].set_active_dims(hidden_dim, hidden_dim * 4)
            ff[3].set_active_dims(hidden_dim * 4, hidden_dim)

        self.belief_head.set_active_dims(hidden_dim, self.output_dim)
        self.action_head.set_active_dims(hidden_dim, 1)

    def forward_trn(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for TRN subnet."""
        batch_size, seq_len, _ = x.shape
        hidden_dim = self.active_config.hidden_dim
        num_layers = self.active_config.num_layers

        h = self.input_proj(x)
        hidden = torch.zeros(batch_size, hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = h[:, t, :]

            for layer_idx in range(num_layers):
                hidden = self.trn_cells[layer_idx](x_t, hidden)
                if self.active_config.use_skip_connections and layer_idx > 0:
                    hidden = hidden + x_t  # Skip connection
                x_t = hidden

            outputs.append(hidden)

        output_tensor = torch.stack(outputs, dim=1)
        beliefs = torch.sigmoid(self.belief_head(output_tensor[:, -1, :]))
        actions = torch.sigmoid(self.action_head(output_tensor[:, -1, :]))

        return {
            'hidden_states': output_tensor,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'final_hidden': hidden
        }

    def forward_rsan(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for RSAN subnet."""
        batch_size, seq_len, _ = x.shape
        num_layers = self.active_config.num_layers

        h = self.input_proj(x)
        attention_patterns = []

        for depth in range(num_layers):
            h_attn, attn_weights = self.attention_layers[depth](h, h, h)
            h = h + F.dropout(h_attn, p=self.dropout, training=self.training)
            attention_patterns.append(attn_weights)

        beliefs = torch.sigmoid(self.belief_head(h[:, -1, :]))
        actions = torch.sigmoid(self.action_head(h[:, -1, :]))

        return {
            'hidden_states': h,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1),
            'attention_patterns': attention_patterns
        }

    def forward_transformer(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for Transformer subnet."""
        batch_size, seq_len, _ = x.shape
        num_layers = self.active_config.num_layers

        h = self.input_proj(x)

        for layer_idx in range(num_layers):
            # Self-attention with pre-norm
            norm_idx = layer_idx * 2
            h_norm = self.layer_norms[norm_idx](h)
            h_attn, _ = self.attention_layers[layer_idx](h_norm, h_norm, h_norm)
            h = h + F.dropout(h_attn, p=self.dropout, training=self.training)

            # Feedforward with pre-norm
            h_norm = self.layer_norms[norm_idx + 1](h)
            h_ff = self.ff_layers[layer_idx][0](h_norm)  # First linear
            h_ff = F.relu(h_ff)
            h_ff = F.dropout(h_ff, p=self.dropout, training=self.training)
            h_ff = self.ff_layers[layer_idx][3](h_ff)  # Second linear
            h = h + F.dropout(h_ff, p=self.dropout, training=self.training)

        beliefs = torch.sigmoid(self.belief_head(h[:, -1, :]))
        actions = torch.sigmoid(self.action_head(h[:, -1, :]))

        return {
            'hidden_states': h,
            'beliefs': beliefs,
            'actions': actions.squeeze(-1)
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using current active configuration."""
        if self.active_config is None:
            # Default to small transformer
            self.set_active_config(SubnetConfig(
                arch_type='transformer',
                num_layers=2,
                hidden_dim=128,
                num_heads=4
            ))

        arch_type = self.active_config.arch_type.lower()

        if arch_type == 'trn':
            return self.forward_trn(x)
        elif arch_type == 'rsan':
            return self.forward_rsan(x)
        elif arch_type == 'transformer':
            return self.forward_transformer(x)
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    def extract_subnet(self, config: SubnetConfig) -> nn.Module:
        """
        Extract a standalone subnet with inherited weights.

        Args:
            config: Configuration for the subnet

        Returns:
            A standalone nn.Module with copied weights
        """
        self.set_active_config(config)

        # Create a lightweight wrapper that captures the current config
        class ExtractedSubnet(nn.Module):
            def __init__(self, supernet, config):
                super().__init__()
                self.config = config
                self.supernet = supernet

            def forward(self, x):
                self.supernet.set_active_config(self.config)
                return self.supernet(x)

        return ExtractedSubnet(self, config)


class SupernetTrainer:
    """
    Trainer for the ToM Supernet using progressive shrinking.

    Progressive shrinking trains the full supernet first, then gradually
    trains smaller subnets while maintaining shared weight quality.
    """

    def __init__(
        self,
        supernet: ToMSupernet,
        device: str = 'cpu',
        lr: float = 1e-3
    ):
        self.supernet = supernet.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(supernet.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        # Training history
        self.losses: List[float] = []
        self.configs_trained: List[SubnetConfig] = []

    def sample_config(self) -> SubnetConfig:
        """Sample a random valid subnet configuration."""
        arch_types = ['trn', 'rsan', 'transformer']

        return SubnetConfig(
            arch_type=np.random.choice(arch_types),
            num_layers=np.random.randint(
                ToMSupernet.MIN_LAYERS,
                ToMSupernet.MAX_LAYERS + 1
            ),
            hidden_dim=np.random.choice([64, 96, 128, 160, 192, 224, 256]),
            num_heads=np.random.choice([2, 4, 6, 8]),
            dropout=np.random.choice([0.0, 0.1, 0.2]),
            use_skip_connections=np.random.random() > 0.3
        )

    def train_step(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        config: Optional[SubnetConfig] = None
    ) -> float:
        """
        Single training step with a sampled or specified configuration.

        Args:
            batch_inputs: Input tensor
            batch_targets: Target tensor for beliefs
            config: Optional config (sampled if not provided)

        Returns:
            Loss value
        """
        self.supernet.train()

        if config is None:
            config = self.sample_config()

        self.supernet.set_active_config(config)

        self.optimizer.zero_grad()

        output = self.supernet(batch_inputs.to(self.device))
        beliefs = output['beliefs']

        # Compute loss
        loss = F.binary_cross_entropy(
            beliefs,
            batch_targets.to(self.device)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.supernet.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.configs_trained.append(config)

        return loss_val

    def train_epoch(
        self,
        data_loader,
        configs_per_batch: int = 4
    ) -> float:
        """
        Train for one epoch, sampling multiple configs per batch.

        Args:
            data_loader: DataLoader providing (inputs, targets)
            configs_per_batch: Number of configurations to sample per batch

        Returns:
            Average loss for the epoch
        """
        epoch_losses = []

        for batch_inputs, batch_targets in data_loader:
            batch_losses = []
            for _ in range(configs_per_batch):
                loss = self.train_step(batch_inputs, batch_targets)
                batch_losses.append(loss)
            epoch_losses.extend(batch_losses)

        self.scheduler.step()
        return np.mean(epoch_losses)

    def progressive_shrinking(
        self,
        data_loader,
        num_epochs: int = 20,
        phases: int = 4
    ) -> Dict[str, Any]:
        """
        Progressive shrinking training strategy.

        Trains full network first, then progressively smaller networks.

        Args:
            data_loader: DataLoader for training
            num_epochs: Total epochs
            phases: Number of shrinking phases

        Returns:
            Training statistics
        """
        epochs_per_phase = num_epochs // phases

        for phase in range(phases):
            # Determine size range for this phase
            # Start with large, progressively include smaller
            min_layer_mult = 1 - (phase / phases)
            min_hidden_mult = 1 - (phase / phases)

            for epoch in range(epochs_per_phase):
                # Sample config biased by phase
                config = self._sample_config_for_phase(min_layer_mult, min_hidden_mult)

                for batch_inputs, batch_targets in data_loader:
                    self.train_step(batch_inputs, batch_targets, config)

                if epoch % 5 == 0:
                    avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else np.mean(self.losses)
                    print(f"Phase {phase+1}/{phases}, Epoch {epoch+1}/{epochs_per_phase}, Loss: {avg_loss:.4f}")

        return {
            'final_loss': np.mean(self.losses[-100:]),
            'total_configs_trained': len(self.configs_trained),
            'loss_history': self.losses
        }

    def _sample_config_for_phase(
        self,
        min_layer_mult: float,
        min_hidden_mult: float
    ) -> SubnetConfig:
        """Sample config appropriate for training phase."""
        min_layers = max(1, int(ToMSupernet.MAX_LAYERS * min_layer_mult))
        min_hidden = max(64, int(ToMSupernet.MAX_HIDDEN_DIM * min_hidden_mult))

        return SubnetConfig(
            arch_type=np.random.choice(['trn', 'rsan', 'transformer']),
            num_layers=np.random.randint(min_layers, ToMSupernet.MAX_LAYERS + 1),
            hidden_dim=np.random.choice([d for d in [64, 96, 128, 160, 192, 224, 256] if d >= min_hidden]),
            num_heads=np.random.choice([2, 4, 6, 8]),
            dropout=np.random.choice([0.0, 0.1, 0.2]),
            use_skip_connections=np.random.random() > 0.3
        )


class SupernetEvaluator:
    """
    Evaluates subnet configurations using inherited supernet weights.
    """

    def __init__(
        self,
        supernet: ToMSupernet,
        device: str = 'cpu'
    ):
        self.supernet = supernet.to(device)
        self.device = device
        self.supernet.eval()

    def evaluate_config(
        self,
        config: SubnetConfig,
        eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
        fine_tune_epochs: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate a subnet configuration.

        Args:
            config: Configuration to evaluate
            eval_data: List of (input, target) tensors
            fine_tune_epochs: Optional fine-tuning epochs

        Returns:
            Dictionary with accuracy and other metrics
        """
        self.supernet.set_active_config(config)

        if fine_tune_epochs > 0:
            self._fine_tune(config, eval_data, fine_tune_epochs)

        # Evaluate
        self.supernet.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in eval_data:
                output = self.supernet(inputs.to(self.device))
                beliefs = output['beliefs']

                # Binary accuracy for belief predictions
                predictions = (beliefs > 0.5).float()
                target_binary = (targets.to(self.device) > 0.5).float()

                correct += (predictions == target_binary).sum().item()
                total += predictions.numel()

        accuracy = correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'config': config.to_dict(),
            'param_count': self._count_active_params(config)
        }

    def _fine_tune(
        self,
        config: SubnetConfig,
        data: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int
    ):
        """Brief fine-tuning of subnet."""
        self.supernet.train()
        self.supernet.set_active_config(config)

        # Use smaller learning rate for fine-tuning
        optimizer = torch.optim.Adam(self.supernet.parameters(), lr=1e-4)

        for _ in range(epochs):
            for inputs, targets in data:
                optimizer.zero_grad()
                output = self.supernet(inputs.to(self.device))
                loss = F.binary_cross_entropy(output['beliefs'], targets.to(self.device))
                loss.backward()
                optimizer.step()

    def _count_active_params(self, config: SubnetConfig) -> int:
        """Count parameters for active configuration."""
        # Approximation based on config
        hidden = config.hidden_dim
        layers = config.num_layers
        input_dim = self.supernet.input_dim
        output_dim = self.supernet.output_dim

        # Input projection
        params = input_dim * hidden + hidden

        if config.arch_type.lower() == 'trn':
            # GRU cells per layer
            params += layers * (3 * (hidden + hidden) * hidden + 3 * hidden)
        else:
            # Attention + FF per layer
            params += layers * (4 * hidden * hidden + hidden * hidden * 4 * 2)

        # Output heads
        params += hidden * output_dim + output_dim + hidden + 1

        return params


def test_supernet():
    """Test supernet functionality."""
    print("=" * 60)
    print("SUPERNET TEST")
    print("=" * 60)

    # Create supernet
    supernet = ToMSupernet(input_dim=181, output_dim=181)

    # Test different configurations
    configs = [
        SubnetConfig('trn', num_layers=2, hidden_dim=128, num_heads=4),
        SubnetConfig('rsan', num_layers=3, hidden_dim=96, num_heads=4),
        SubnetConfig('transformer', num_layers=2, hidden_dim=128, num_heads=4),
    ]

    # Generate test input
    x = torch.randn(4, 10, 181)

    for config in configs:
        print(f"\n--- {config.arch_type.upper()} ---")
        supernet.set_active_config(config)
        output = supernet(x)

        print(f"  Config: layers={config.num_layers}, hidden={config.hidden_dim}")
        print(f"  Beliefs shape: {output['beliefs'].shape}")
        print(f"  Actions shape: {output['actions'].shape}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_supernet()
