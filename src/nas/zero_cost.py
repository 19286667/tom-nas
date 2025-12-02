"""
Zero-cost proxy evaluation for Neural Architecture Search.

Zero-cost proxies estimate architecture quality without training,
enabling rapid exploration of the architecture space.

Implemented proxies:
- SynFlow: Product of absolute weights along paths
- NASWOT: Jacobian covariance (input distinguishability)
- GradNorm: Gradient norm at initialization
- ParamCount: Total parameter count
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class ZeroCostProxies:
    """
    Zero-cost architecture evaluation proxies.

    These metrics estimate architecture quality without training,
    allowing rapid evaluation of thousands of architectures.
    """

    def __init__(self, input_dim: int = 181, seq_len: int = 10):
        """
        Initialize proxies.

        Args:
            input_dim: Input feature dimension
            seq_len: Sequence length for test inputs
        """
        self.input_dim = input_dim
        self.seq_len = seq_len

    def compute_all(self, model: nn.Module,
                    batch_size: int = 32) -> Dict[str, float]:
        """
        Compute all zero-cost proxies for a model.

        Args:
            model: PyTorch model to evaluate
            batch_size: Batch size for proxy computation

        Returns:
            Dict with proxy scores and combined score
        """
        # Create random input batch
        x = torch.randn(batch_size, self.seq_len, self.input_dim)

        scores = {}
        scores['synflow'] = self._synflow(model, x)
        scores['naswot'] = self._naswot(model, x)
        scores['grad_norm'] = self._grad_norm(model, x)
        scores['params'] = self._param_count(model)
        scores['jacob_cov'] = self._jacob_cov(model, x)

        # Combined score (empirically weighted)
        # Use log-scaled normalization for large values
        # Higher synflow and naswot are better
        # Lower param count is better (efficiency)
        scores['combined'] = (
            0.3 * self._log_normalize(scores['synflow']) +
            0.3 * self._log_normalize(scores['naswot'], signed=True) +
            0.2 * self._log_normalize(scores['jacob_cov']) +
            0.1 * self._log_normalize(scores['grad_norm']) +
            0.1 * (1 - self._log_normalize(scores['params']))  # Prefer smaller
        )

        return scores

    def _synflow(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        SynFlow: Product of absolute weights along paths.

        From "Pruning Neural Networks at Initialization: Why Are We Missing the Mark?"
        Measures the flow of signal through the network.

        Higher score = better information flow = potentially better architecture.
        """
        # Clone model to avoid modifying original
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Set all parameters to absolute value
        @torch.no_grad()
        def linearize(m):
            for param in m.parameters():
                param.abs_()

        linearize(model_copy)

        # Forward pass with ones
        ones = torch.ones_like(x)
        model_copy.zero_grad()

        try:
            output = self._get_output_tensor(model_copy, ones)

            # Sum of outputs
            score = output.sum()

            # Enable gradients for backward pass
            for param in model_copy.parameters():
                param.requires_grad_(True)

            score.backward()

            # SynFlow score = sum of (param * grad)
            synflow = 0.0
            for param in model_copy.parameters():
                if param.grad is not None:
                    synflow += (param * param.grad).abs().sum().item()

            return synflow
        except Exception as e:
            return 0.0

    def _naswot(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        NASWOT: Jacobian covariance - how well inputs are distinguished.

        From "Neural Architecture Search Without Training"
        Measures how differently the network treats different inputs.

        Higher score = better input discrimination = potentially better architecture.
        """
        model.eval()

        # Get activations for each input
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Flatten and store
                activations.append(output.detach().view(output.size(0), -1))

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ReLU, nn.GELU)):
                hooks.append(module.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                _ = self._get_output_tensor(model, x)

            if not activations:
                return 0.0

            # Compute binary activation patterns
            # Concatenate activations (limit size for memory)
            K = torch.cat([a[:, :min(100, a.size(1))] for a in activations[:5]], dim=1)

            # Binary patterns (which neurons are active)
            K_binary = (K > 0).float()

            # Hamming similarity matrix
            n = K_binary.size(0)
            K_matrix = K_binary @ K_binary.T

            # Normalize by number of features
            K_matrix = K_matrix / (K_binary.size(1) + 1e-5)

            # Add small identity for numerical stability
            K_matrix = K_matrix + 1e-4 * torch.eye(n)

            # Score = log determinant (diversity of patterns)
            try:
                sign, logdet = torch.linalg.slogdet(K_matrix)
                if sign > 0:
                    return logdet.item()
                else:
                    return -abs(logdet.item())
            except:
                return 0.0

        finally:
            for h in hooks:
                h.remove()

    def _jacob_cov(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Jacobian Covariance: How varied are the gradients across inputs?

        Measures expressivity by looking at gradient diversity.
        """
        model.eval()

        # Use smaller batch for memory efficiency
        x_small = x[:min(16, x.size(0))]
        x_small.requires_grad_(True)

        try:
            output = self._get_output_tensor(model, x_small)

            # Get gradients w.r.t. input for each output dimension
            gradients = []
            for i in range(min(10, output.size(-1))):
                model.zero_grad()
                if x_small.grad is not None:
                    x_small.grad.zero_()

                output_slice = output[..., i].sum()
                output_slice.backward(retain_graph=True)

                if x_small.grad is not None:
                    gradients.append(x_small.grad.view(x_small.size(0), -1).clone())

            if not gradients:
                return 0.0

            # Stack gradients
            G = torch.stack(gradients, dim=-1)  # (batch, features, outputs)
            G = G.view(G.size(0), -1)  # (batch, features * outputs)

            # Compute covariance
            G_centered = G - G.mean(dim=0)
            cov = G_centered.T @ G_centered / (G.size(0) - 1)

            # Return trace of covariance (variance captured)
            return cov.trace().item()

        except Exception as e:
            return 0.0

    def _grad_norm(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Gradient norm at initialization.

        Measures how trainable the network is at initialization.
        Very low or very high values indicate potential training issues.
        """
        model.train()
        model.zero_grad()

        try:
            output = self._get_output_tensor(model, x)

            # Use random target
            target = torch.randn_like(output)
            loss = ((output - target) ** 2).mean()
            loss.backward()

            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2

            return np.sqrt(total_norm)
        except:
            return 0.0

    def _param_count(self, model: nn.Module) -> float:
        """Total parameter count."""
        return sum(p.numel() for p in model.parameters())

    def _normalize(self, value: float, eps: float = 1e-8) -> float:
        """
        Normalize to [0, 1] range using sigmoid.

        Maps any real value to (0, 1) for combining scores.
        """
        if abs(value) < eps:
            return 0.5
        return 1 / (1 + np.exp(-value / (abs(value) + eps)))

    def _log_normalize(self, value: float, signed: bool = False,
                       scale: float = 10.0) -> float:
        """
        Normalize using log scaling for better discrimination of large values.

        Args:
            value: Value to normalize
            signed: If True, preserve sign (for values that can be negative like naswot)
            scale: Scaling factor for sigmoid

        Returns:
            Normalized value in [0, 1] range
        """
        if value == 0:
            return 0.5

        if signed:
            # For signed values like NASWOT that can be negative
            sign = 1 if value > 0 else -1
            log_val = sign * np.log1p(abs(value))
            # Use sigmoid with scale to get better spread
            return 1 / (1 + np.exp(-log_val / scale))
        else:
            # For positive values, use log directly
            if value <= 0:
                return 0.0
            log_val = np.log1p(value)
            # Normalize to roughly [0, 1] - synflow values around 1e5 -> log ~12
            return min(1.0, log_val / 20.0)

    def _get_output_tensor(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Get output tensor from model, handling various output formats.

        Supports dict, tuple, and tensor outputs.
        """
        output = model(x)

        if isinstance(output, dict):
            # Try common keys
            for key in ['beliefs', 'output', 'logits', 'hidden']:
                if key in output:
                    return output[key]
            # Return first value
            return list(output.values())[0]
        elif isinstance(output, tuple):
            return output[0]
        else:
            return output


def rank_architectures(models: List[nn.Module],
                       proxies: ZeroCostProxies,
                       verbose: bool = False) -> List[Tuple[int, float, Dict[str, float]]]:
    """
    Rank architectures by zero-cost proxy scores.

    Args:
        models: List of PyTorch models to rank
        proxies: ZeroCostProxies instance
        verbose: Print progress

    Returns:
        List of (index, combined_score, all_scores) tuples, sorted by combined score
    """
    results = []

    for i, model in enumerate(models):
        if verbose:
            print(f"  Evaluating architecture {i+1}/{len(models)}...", end=' ')

        scores = proxies.compute_all(model)
        results.append((i, scores['combined'], scores))

        if verbose:
            print(f"score={scores['combined']:.4f}")

    # Sort by combined score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def quick_rank(models: List[nn.Module],
               input_dim: int = 181,
               top_k: int = 10) -> List[int]:
    """
    Quick helper to get top-k architecture indices.

    Args:
        models: List of models
        input_dim: Input dimension
        top_k: Number of top architectures to return

    Returns:
        List of top-k model indices
    """
    proxies = ZeroCostProxies(input_dim=input_dim)
    ranked = rank_architectures(models, proxies)
    return [idx for idx, _, _ in ranked[:top_k]]
