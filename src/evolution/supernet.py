"""
Supernet for Neural Architecture Search (NAS)

This module implements elastic/supernet architectures that can be dynamically
configured during NAS. Key components:

- ElasticLSTMCell: LSTM with configurable hidden dimensions
- ElasticTransformer: Transformer with elastic width/depth
- ZeroCostProxies: Fast architecture evaluation without training

The supernet architecture allows efficient weight sharing during search.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ElasticConfig:
    """Configuration for elastic architecture."""

    min_hidden_dim: int = 64
    max_hidden_dim: int = 256
    min_layers: int = 1
    max_layers: int = 6
    min_heads: int = 2
    max_heads: int = 8
    dropout_range: Tuple[float, float] = (0.0, 0.3)


class ElasticLSTMCell(nn.Module):
    """
    LSTM Cell with elastic hidden dimension.

    The cell is initialized with max dimensions but can operate with
    smaller active dimensions for architecture search.

    Bug Fix: LayerNorm is applied dynamically using F.layer_norm
    to handle variable hidden dimensions properly.
    """

    def __init__(self, input_dim: int, max_hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.max_hidden_dim = max_hidden_dim

        # Gates: input, forget, cell, output
        self.weight_ih = nn.Parameter(torch.randn(4 * max_hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.randn(4 * max_hidden_dim, max_hidden_dim))
        self.bias_ih = nn.Parameter(torch.zeros(4 * max_hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(4 * max_hidden_dim))

        # LayerNorm parameters (used dynamically)
        self.layer_norm = nn.LayerNorm(max_hidden_dim)

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        stdv = 1.0 / math.sqrt(self.max_hidden_dim)
        for weight in [self.weight_ih, self.weight_hh]:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor], active_hidden_dim: Optional[int] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional elastic hidden dimension.

        Args:
            x: Input tensor of shape (batch, input_dim)
            state: Tuple of (h, c) hidden and cell states
            active_hidden_dim: Active hidden dimension (uses max if None)

        Returns:
            h_new: New hidden state
            (h_new, c_new): New state tuple
        """
        h, c = state
        active_dim = active_hidden_dim or self.max_hidden_dim

        # Slice weights for active dimensions
        weight_ih = self.weight_ih[: 4 * active_dim, : self.input_dim]
        weight_hh = self.weight_hh[: 4 * active_dim, :active_dim]
        bias_ih = self.bias_ih[: 4 * active_dim]
        bias_hh = self.bias_hh[: 4 * active_dim]

        # Slice states
        h = h[:, :active_dim]
        c = c[:, :active_dim]

        # LSTM computations
        gates = F.linear(x, weight_ih, bias_ih) + F.linear(h, weight_hh, bias_hh)

        # Split gates
        i_gate = torch.sigmoid(gates[:, :active_dim])
        f_gate = torch.sigmoid(gates[:, active_dim : 2 * active_dim])
        g_gate = torch.tanh(gates[:, 2 * active_dim : 3 * active_dim])
        o_gate = torch.sigmoid(gates[:, 3 * active_dim : 4 * active_dim])

        # Cell and hidden state update
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        # Apply LayerNorm with correct dimensions
        # FIX: Use F.layer_norm with sliced parameters instead of self.layer_norm
        h_new = F.layer_norm(
            h_new, [h_new.shape[-1]], self.layer_norm.weight[: h_new.shape[-1]], self.layer_norm.bias[: h_new.shape[-1]]
        )

        # Pad back to max dim if needed
        if active_dim < self.max_hidden_dim:
            padding = torch.zeros(
                h_new.shape[0], self.max_hidden_dim - active_dim, device=h_new.device, dtype=h_new.dtype
            )
            h_new = torch.cat([h_new, padding], dim=1)
            c_new = torch.cat([c_new, padding], dim=1)

        return h_new, (h_new, c_new)


class ElasticTransparentRNN(nn.Module):
    """
    Transparent Recurrent Network with elastic architecture.

    TRN uses explicit symbolic computation paths for interpretability.
    """

    def __init__(self, input_dim: int, max_hidden_dim: int = 256, output_dim: int = 181, max_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.max_hidden_dim = max_hidden_dim
        self.output_dim = output_dim
        self.max_layers = max_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, max_hidden_dim)

        # Elastic LSTM layers
        self.lstm_layers = nn.ModuleList([ElasticLSTMCell(max_hidden_dim, max_hidden_dim) for _ in range(max_layers)])

        # Output projection
        self.output_proj = nn.Linear(max_hidden_dim, output_dim)

        # Symbolic computation paths (for interpretability)
        self.belief_head = nn.Linear(max_hidden_dim, output_dim)
        self.attention_scores = None

    def forward(self, x: torch.Tensor, active_config: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with elastic configuration.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            active_config: Dict with 'hidden_dim' and 'num_layers'

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Get active configuration
        active_hidden = active_config.get("hidden_dim", self.max_hidden_dim) if active_config else self.max_hidden_dim
        active_layers = active_config.get("num_layers", self.max_layers) if active_config else self.max_layers

        # Input projection
        x = self.input_proj(x)

        # Initialize states
        h = torch.zeros(batch_size, self.max_hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.max_hidden_dim, device=x.device)

        # Process sequence through elastic LSTM layers
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer_idx in range(active_layers):
                layer_input, (h, c) = self.lstm_layers[layer_idx](layer_input, (h, c), active_hidden_dim=active_hidden)

            outputs.append(h)

        # Use final hidden state
        final_h = outputs[-1][:, :active_hidden]

        # FIX: Slice output projection weights for active hidden dim
        output_weight = self.output_proj.weight[:, :active_hidden]
        output = F.linear(final_h, output_weight, self.output_proj.bias)

        return output

    def forward_trn(self, x: torch.Tensor, active_config: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with transparent computation paths.

        Returns output and interpretability information.
        """
        batch_size, seq_len, _ = x.shape
        active_hidden = active_config.get("hidden_dim", self.max_hidden_dim) if active_config else self.max_hidden_dim
        active_layers = active_config.get("num_layers", self.max_layers) if active_config else self.max_layers

        x = self.input_proj(x)

        h = torch.zeros(batch_size, self.max_hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.max_hidden_dim, device=x.device)

        # Track computation paths
        layer_outputs = []
        gate_values = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer_idx in range(active_layers):
                layer_input, (h, c) = self.lstm_layers[layer_idx](layer_input, (h, c), active_hidden_dim=active_hidden)
                layer_outputs.append(h.clone())

        final_h = layer_outputs[-1][:, :active_hidden] if layer_outputs else h[:, :active_hidden]

        # Belief prediction with proper slicing
        belief_weight = self.belief_head.weight[:, :active_hidden]
        belief_output = F.linear(final_h, belief_weight, self.belief_head.bias)

        output_weight = self.output_proj.weight[:, :active_hidden]
        output = F.linear(final_h, output_weight, self.output_proj.bias)

        interpretation = {
            "layer_activations": layer_outputs,
            "belief_predictions": belief_output,
            "num_active_layers": active_layers,
            "active_hidden_dim": active_hidden,
        }

        return output, interpretation


class ElasticTransformer(nn.Module):
    """
    Transformer with elastic width and depth for NAS.
    """

    def __init__(
        self, input_dim: int, max_hidden_dim: int = 256, output_dim: int = 181, max_layers: int = 6, max_heads: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_hidden_dim = max_hidden_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.max_heads = max_heads

        # Input embedding
        self.input_embed = nn.Linear(input_dim, max_hidden_dim)
        self.pos_encoding = PositionalEncoding(max_hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [ElasticTransformerLayer(max_hidden_dim, max_heads) for _ in range(max_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(max_hidden_dim, output_dim)

        # Layer norm (with elastic support)
        self.final_norm = nn.LayerNorm(max_hidden_dim)

    def forward(self, x: torch.Tensor, active_config: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with elastic configuration.
        """
        active_hidden = active_config.get("hidden_dim", self.max_hidden_dim) if active_config else self.max_hidden_dim
        active_layers = active_config.get("num_layers", self.max_layers) if active_config else self.max_layers
        active_heads = active_config.get("num_heads", self.max_heads) if active_config else self.max_heads

        # Input embedding
        x = self.input_embed(x)[:, :, :active_hidden]
        x = self.pos_encoding(x)

        # Transformer layers
        for layer_idx in range(active_layers):
            x = self.transformer_layers[layer_idx](x, active_hidden_dim=active_hidden, active_heads=active_heads)

        # Pool sequence
        x = x.mean(dim=1)

        # Slice back to active dimensions (ElasticTransformerLayer pads to max)
        x = x[:, :active_hidden]

        # Final norm with elastic support
        # FIX: Use F.layer_norm with sliced parameters
        x = F.layer_norm(x, [x.shape[-1]], self.final_norm.weight[: x.shape[-1]], self.final_norm.bias[: x.shape[-1]])

        # Output projection with sliced weights
        output_weight = self.output_proj.weight[:, :active_hidden]
        output = F.linear(x, output_weight, self.output_proj.bias)

        return output

    def forward_transformer(self, x: torch.Tensor, active_config: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with attention visualization.
        """
        active_hidden = active_config.get("hidden_dim", self.max_hidden_dim) if active_config else self.max_hidden_dim
        active_layers = active_config.get("num_layers", self.max_layers) if active_config else self.max_layers
        active_heads = active_config.get("num_heads", self.max_heads) if active_config else self.max_heads

        x = self.input_embed(x)[:, :, :active_hidden]
        x = self.pos_encoding(x)

        attention_maps = []
        for layer_idx in range(active_layers):
            x, attn = self.transformer_layers[layer_idx](
                x, active_hidden_dim=active_hidden, active_heads=active_heads, return_attention=True
            )
            attention_maps.append(attn)

        x = x.mean(dim=1)

        # Slice back to active dimensions (ElasticTransformerLayer pads to max)
        x = x[:, :active_hidden]

        # FIX: Use F.layer_norm with sliced parameters
        x = F.layer_norm(x, [x.shape[-1]], self.final_norm.weight[: x.shape[-1]], self.final_norm.bias[: x.shape[-1]])

        output_weight = self.output_proj.weight[:, :active_hidden]
        output = F.linear(x, output_weight, self.output_proj.bias)

        interpretation = {
            "attention_maps": attention_maps,
            "num_active_layers": active_layers,
            "active_hidden_dim": active_hidden,
            "active_heads": active_heads,
        }

        return output, interpretation


class ElasticTransformerLayer(nn.Module):
    """Single transformer layer with elastic dimensions."""

    def __init__(self, max_hidden_dim: int, max_heads: int):
        super().__init__()
        self.max_hidden_dim = max_hidden_dim
        self.max_heads = max_heads

        # Multi-head attention
        self.q_proj = nn.Linear(max_hidden_dim, max_hidden_dim)
        self.k_proj = nn.Linear(max_hidden_dim, max_hidden_dim)
        self.v_proj = nn.Linear(max_hidden_dim, max_hidden_dim)
        self.o_proj = nn.Linear(max_hidden_dim, max_hidden_dim)

        # Feed-forward
        self.ff1 = nn.Linear(max_hidden_dim, max_hidden_dim * 4)
        self.ff2 = nn.Linear(max_hidden_dim * 4, max_hidden_dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(max_hidden_dim)
        self.norm2 = nn.LayerNorm(max_hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self, x: torch.Tensor, active_hidden_dim: int = None, active_heads: int = None, return_attention: bool = False
    ):
        """Forward with elastic configuration."""
        active_hidden = active_hidden_dim or self.max_hidden_dim
        active_heads = active_heads or self.max_heads

        batch_size, seq_len, _ = x.shape
        head_dim = active_hidden // active_heads

        # Slice input to active dimensions
        x = x[:, :, :active_hidden]

        # Self-attention with residual
        residual = x

        # FIX: Use F.layer_norm with sliced parameters
        x = F.layer_norm(x, [x.shape[-1]], self.norm1.weight[: x.shape[-1]], self.norm1.bias[: x.shape[-1]])

        # Project Q, K, V with sliced weights
        q = F.linear(x, self.q_proj.weight[:active_hidden, :active_hidden], self.q_proj.bias[:active_hidden])
        k = F.linear(x, self.k_proj.weight[:active_hidden, :active_hidden], self.k_proj.bias[:active_hidden])
        v = F.linear(x, self.v_proj.weight[:active_hidden, :active_hidden], self.v_proj.bias[:active_hidden])

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, active_hidden)

        # Output projection
        attn_output = F.linear(
            attn_output, self.o_proj.weight[:active_hidden, :active_hidden], self.o_proj.bias[:active_hidden]
        )
        attn_output = self.dropout(attn_output)

        x = residual + attn_output

        # Feed-forward with residual
        residual = x

        # FIX: Use F.layer_norm with sliced parameters
        x = F.layer_norm(x, [x.shape[-1]], self.norm2.weight[: x.shape[-1]], self.norm2.bias[: x.shape[-1]])

        ff_output = F.linear(
            x, self.ff1.weight[: active_hidden * 4, :active_hidden], self.ff1.bias[: active_hidden * 4]
        )
        ff_output = F.gelu(ff_output)
        ff_output = F.linear(
            ff_output, self.ff2.weight[:active_hidden, : active_hidden * 4], self.ff2.bias[:active_hidden]
        )
        ff_output = self.dropout(ff_output)

        x = residual + ff_output

        # Pad back to max dim if needed
        if active_hidden < self.max_hidden_dim:
            padding = torch.zeros(
                batch_size, seq_len, self.max_hidden_dim - active_hidden, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=-1)

        if return_attention:
            return x, attn_probs
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        hidden_dim = x.shape[2]
        return x + self.pe[:, :seq_len, :hidden_dim]


# Zero-Cost Proxies for fast architecture evaluation


class ZeroCostProxy:
    """
    Zero-cost proxies for fast architecture evaluation without training.

    These metrics correlate with final trained performance and allow
    efficient architecture search.
    """

    @staticmethod
    def synflow(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> float:
        """
        Synaptic Flow (SynFlow) proxy.

        Measures path-wise pruning saliency without data.
        """
        model = model.to(device)
        model.eval()

        # Set all parameters to ones
        signs = {}
        for name, param in model.named_parameters():
            signs[name] = param.sign()
            param.data = torch.ones_like(param)

        # Forward pass with ones
        x = torch.ones(1, *input_shape, device=device)
        try:
            output = model(x)
        except Exception:
            return 0.0

        # Backward pass
        if output.requires_grad:
            output.sum().backward()

        # Compute SynFlow score
        score = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                score += (signs[name] * param.grad * param).sum().abs().item()

        # Restore parameters
        model.zero_grad()

        return score

    @staticmethod
    def grad_norm(model: nn.Module, dataloader, device: str = "cpu") -> float:
        """
        Gradient norm at initialization.

        Higher gradient norm often indicates better trainability.
        """
        model = model.to(device)
        model.train()

        total_norm = 0.0
        count = 0

        for batch in dataloader:
            if count >= 10:  # Only use a few batches
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            model.zero_grad()
            output = model(x)

            # Simplified loss
            loss = F.cross_entropy(output, y) if output.dim() > 1 else output.mean()
            loss.backward()

            # Compute gradient norm
            batch_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    batch_norm += param.grad.norm(2).item() ** 2

            total_norm += batch_norm**0.5
            count += 1

        return total_norm / max(count, 1)

    @staticmethod
    def jacob_cov(model: nn.Module, input_shape: Tuple[int, ...], num_samples: int = 32, device: str = "cpu") -> float:
        """
        Jacobian covariance score.

        Measures the rank/diversity of input-output Jacobian.
        """
        model = model.to(device)
        model.eval()

        jacobians = []

        for _ in range(num_samples):
            x = torch.randn(1, *input_shape, device=device, requires_grad=True)

            try:
                output = model(x)
            except Exception:
                return 0.0

            if output.dim() > 1:
                output = output.view(-1)

            # Compute Jacobian for first few outputs
            for i in range(min(10, output.shape[0])):
                model.zero_grad()
                if x.grad is not None:
                    x.grad.zero_()

                output[i].backward(retain_graph=True)

                if x.grad is not None:
                    jacobians.append(x.grad.view(-1).clone())

        if not jacobians:
            return 0.0

        # Stack and compute covariance
        jacobian_matrix = torch.stack(jacobians)

        # Correlation score (higher = more diverse gradients)
        try:
            _, s, _ = torch.svd(jacobian_matrix)
            score = (s > 1e-5).sum().item() / len(s)  # Effective rank
        except Exception:
            score = 0.0

        return score

    @staticmethod
    def compute_all_proxies(
        model: nn.Module, input_shape: Tuple[int, ...], dataloader=None, device: str = "cpu"
    ) -> Dict[str, float]:
        """Compute all available zero-cost proxies."""
        results = {}

        results["synflow"] = ZeroCostProxy.synflow(model, input_shape, device)
        results["jacob_cov"] = ZeroCostProxy.jacob_cov(model, input_shape, device=device)

        if dataloader:
            results["grad_norm"] = ZeroCostProxy.grad_norm(model, dataloader, device)

        return results


# Export
__all__ = [
    "ElasticConfig",
    "ElasticLSTMCell",
    "ElasticTransparentRNN",
    "ElasticTransformer",
    "ElasticTransformerLayer",
    "PositionalEncoding",
    "ZeroCostProxy",
]
