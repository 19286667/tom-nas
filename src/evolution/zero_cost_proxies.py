"""
Zero-Cost Proxies for Neural Architecture Search

This module implements zero-cost proxy metrics that estimate architecture quality
without any training. These proxies enable filtering thousands of architectures
in minutes rather than hours.

Key proxies implemented:
- SynFlow: Measures product of absolute weights along paths (gradient flow quality)
- NASWOT: Measures input discrimination at initialization (representation diversity)
- GradNorm: Measures gradient magnitude at initialization (trainability)

Research has shown these proxies achieve ~0.82 correlation with trained accuracy,
sufficient to identify promising architectures without full training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time


@dataclass
class ProxyScore:
    """Container for proxy evaluation results."""
    synflow: float
    naswot: float
    gradnorm: float
    param_count: int
    combined_score: float
    evaluation_time_ms: float


class ZeroCostProxy:
    """
    Computes zero-cost proxy scores for neural network architectures.

    These proxies estimate architecture quality at initialization without training,
    enabling rapid filtering of candidate architectures in NAS.
    """

    def __init__(
        self,
        input_dim: int = 181,
        seq_len: int = 10,
        batch_size: int = 32,
        device: str = 'cpu',
        target_params: int = 500000
    ):
        """
        Initialize the proxy evaluator.

        Args:
            input_dim: Input dimension (181 for Soul Map)
            seq_len: Sequence length for synthetic inputs
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            target_params: Target parameter count for efficiency penalty
        """
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.target_params = target_params

        # Pre-generate inputs for consistency
        self._synthetic_input = None
        self._ones_input = None

    def _get_synthetic_input(self) -> torch.Tensor:
        """Get or create synthetic input for evaluation."""
        if self._synthetic_input is None:
            self._synthetic_input = torch.randn(
                self.batch_size, self.seq_len, self.input_dim,
                device=self.device
            )
        return self._synthetic_input

    def _get_ones_input(self) -> torch.Tensor:
        """Get or create ones input for SynFlow."""
        if self._ones_input is None:
            self._ones_input = torch.ones(
                1, self.seq_len, self.input_dim,
                device=self.device
            )
        return self._ones_input

    def compute_synflow(self, model: nn.Module) -> float:
        """
        Compute SynFlow proxy score.

        SynFlow measures the product of absolute weights along paths through the network.
        Networks with good gradient flow (avoiding vanishing/exploding gradients)
        tend to have higher SynFlow scores and train more effectively.

        Algorithm:
        1. Replace all parameters with their absolute values
        2. Pass a tensor of ones through the network
        3. Backpropagate from the output
        4. Compute sum of (|param| * |grad|) for all parameters

        Args:
            model: Neural network to evaluate

        Returns:
            SynFlow score (higher is better)
        """
        model = model.to(self.device)
        model.eval()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            # Replace with absolute values
            param.data = torch.abs(param.data)

        try:
            # Enable gradients
            for param in model.parameters():
                param.requires_grad_(True)

            # Forward pass with ones
            input_ones = self._get_ones_input()
            input_ones.requires_grad_(False)

            output = model(input_ones)

            # Handle different output formats
            if isinstance(output, dict):
                # Use beliefs output for ToM architectures
                if 'beliefs' in output:
                    out_tensor = output['beliefs']
                elif 'hidden_states' in output:
                    out_tensor = output['hidden_states'][:, -1, :]
                else:
                    out_tensor = list(output.values())[0]
            else:
                out_tensor = output

            # Sum and backpropagate
            loss = out_tensor.sum()
            loss.backward()

            # Compute SynFlow score
            synflow_score = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    synflow_score += (torch.abs(param) * torch.abs(param.grad)).sum().item()

            return synflow_score

        finally:
            # Restore original parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_params:
                        param.data = original_params[name]
                        param.grad = None

    def compute_naswot(self, model: nn.Module, num_samples: int = 32) -> float:
        """
        Compute NASWOT (Neural Architecture Search Without Training) proxy score.

        NASWOT measures how well the network distinguishes different inputs at initialization.
        Networks that produce diverse activation patterns for different inputs
        tend to learn better representations.

        Algorithm:
        1. Pass a batch of random inputs through the network
        2. Collect activation patterns at intermediate layers
        3. Binarize patterns based on neuron activation
        4. Compute similarity matrix via Hamming distance
        5. Return log-determinant as diversity score

        Args:
            model: Neural network to evaluate
            num_samples: Number of random samples to use

        Returns:
            NASWOT score (higher is better)
        """
        model = model.to(self.device)
        model.eval()

        # Generate random inputs
        inputs = torch.randn(
            num_samples, self.seq_len, self.input_dim,
            device=self.device
        )

        # Collect activations
        activations = []
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach())
            elif isinstance(output, tuple) and len(output) > 0:
                activations.append(output[0].detach())

        # Register hooks on linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                hooks.append(module.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                _ = model(inputs)

            if not activations:
                return 0.0

            # Concatenate all activations and flatten
            all_acts = []
            for act in activations:
                if act.dim() > 2:
                    act = act.view(num_samples, -1)
                all_acts.append(act)

            combined = torch.cat(all_acts, dim=1)  # (num_samples, total_features)

            # Binarize: 1 if activation > 0, else 0
            binary = (combined > 0).float()

            # Compute Hamming similarity matrix
            # K[i,j] = number of matching bits / total bits
            num_features = binary.shape[1]
            similarity = torch.mm(binary, binary.t()) + torch.mm(1 - binary, (1 - binary).t())
            similarity = similarity / num_features

            # Add small diagonal for numerical stability
            similarity = similarity + 1e-6 * torch.eye(num_samples, device=self.device)

            # Compute log-determinant
            try:
                sign, logdet = torch.linalg.slogdet(similarity)
                if sign > 0:
                    return logdet.item()
                else:
                    return -1000.0  # Degenerate case
            except:
                return -1000.0

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def compute_gradnorm(self, model: nn.Module) -> float:
        """
        Compute GradNorm proxy score.

        GradNorm measures the magnitude of gradients at initialization.
        Networks with appropriate gradient magnitudes (neither vanishing nor exploding)
        train more effectively.

        Algorithm:
        1. Forward pass with random inputs
        2. Compute loss against random targets
        3. Backpropagate
        4. Measure L2 norm of gradients

        Args:
            model: Neural network to evaluate

        Returns:
            GradNorm score (moderate values are better)
        """
        model = model.to(self.device)
        model.train()  # Enable dropout etc. for realistic gradients

        # Enable gradients
        for param in model.parameters():
            param.requires_grad_(True)

        try:
            # Forward pass
            inputs = self._get_synthetic_input()
            output = model(inputs)

            # Handle different output formats
            if isinstance(output, dict):
                if 'beliefs' in output:
                    out_tensor = output['beliefs']
                elif 'hidden_states' in output:
                    out_tensor = output['hidden_states'][:, -1, :]
                else:
                    out_tensor = list(output.values())[0]
            else:
                out_tensor = output

            # Generate random targets
            targets = torch.rand_like(out_tensor)

            # Compute MSE loss
            loss = nn.functional.mse_loss(out_tensor, targets)
            loss.backward()

            # Compute gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item() ** 2
            grad_norm = np.sqrt(grad_norm)

            return grad_norm

        finally:
            # Clear gradients
            model.zero_grad()
            for param in model.parameters():
                param.grad = None

    def count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate(self, model: nn.Module) -> ProxyScore:
        """
        Compute all proxy scores for a model.

        Args:
            model: Neural network to evaluate

        Returns:
            ProxyScore with all metrics
        """
        start_time = time.time()

        # Compute individual scores
        synflow = self.compute_synflow(model)
        naswot = self.compute_naswot(model)
        gradnorm = self.compute_gradnorm(model)
        param_count = self.count_parameters(model)

        evaluation_time_ms = (time.time() - start_time) * 1000

        # Compute combined score
        combined = self._compute_combined_score(synflow, naswot, gradnorm, param_count)

        return ProxyScore(
            synflow=synflow,
            naswot=naswot,
            gradnorm=gradnorm,
            param_count=param_count,
            combined_score=combined,
            evaluation_time_ms=evaluation_time_ms
        )

    def _compute_combined_score(
        self,
        synflow: float,
        naswot: float,
        gradnorm: float,
        param_count: int
    ) -> float:
        """
        Compute combined proxy score.

        The combined score balances:
        - SynFlow (gradient flow quality)
        - NASWOT (representation diversity)
        - Parameter efficiency (smaller is better)

        GradNorm is not directly included because moderate values are best,
        which is harder to incorporate into a linear combination.

        Args:
            synflow: SynFlow score
            naswot: NASWOT score
            gradnorm: GradNorm score
            param_count: Parameter count

        Returns:
            Combined score (higher is better)
        """
        # Normalize synflow (log scale due to wide range)
        norm_synflow = np.log1p(max(synflow, 0)) / 20.0  # Rough normalization
        norm_synflow = np.clip(norm_synflow, 0, 1)

        # Normalize naswot (already in reasonable range for log-det)
        norm_naswot = (naswot + 50) / 100.0  # Shift and scale
        norm_naswot = np.clip(norm_naswot, 0, 1)

        # Efficiency penalty
        param_ratio = param_count / self.target_params
        efficiency = 1.0 - np.clip(param_ratio - 1, 0, 1)  # Penalty for exceeding target

        # Weighted combination
        combined = 0.4 * norm_synflow + 0.4 * norm_naswot + 0.2 * efficiency

        return combined


class ArchitectureFilter:
    """
    Filters architectures using zero-cost proxies.

    This class manages the first stage of efficient NAS: generating many
    candidate architectures and filtering to the top performers using
    zero-cost proxies.
    """

    def __init__(
        self,
        proxy_evaluator: Optional[ZeroCostProxy] = None,
        param_budget: int = 500000,
        top_k_ratio: float = 0.1
    ):
        """
        Initialize the architecture filter.

        Args:
            proxy_evaluator: ZeroCostProxy instance (created if not provided)
            param_budget: Maximum allowed parameters
            top_k_ratio: Ratio of architectures to keep (e.g., 0.1 = top 10%)
        """
        self.proxy = proxy_evaluator or ZeroCostProxy()
        self.param_budget = param_budget
        self.top_k_ratio = top_k_ratio

        # Statistics
        self.total_evaluated = 0
        self.total_filtered = 0
        self.evaluation_times: List[float] = []

    def filter_architectures(
        self,
        architectures: List[nn.Module],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, nn.Module, ProxyScore]]:
        """
        Filter architectures using zero-cost proxies.

        Args:
            architectures: List of candidate architectures
            top_k: Number to keep (overrides top_k_ratio if provided)

        Returns:
            List of (original_index, model, score) tuples for top architectures
        """
        if top_k is None:
            top_k = max(1, int(len(architectures) * self.top_k_ratio))

        results: List[Tuple[int, nn.Module, ProxyScore]] = []

        for idx, model in enumerate(architectures):
            score = self.proxy.evaluate(model)
            self.evaluation_times.append(score.evaluation_time_ms)
            self.total_evaluated += 1

            # Filter by parameter budget
            if score.param_count <= self.param_budget:
                results.append((idx, model, score))
            else:
                self.total_filtered += 1

        # Sort by combined score (descending)
        results.sort(key=lambda x: x[2].combined_score, reverse=True)

        # Keep top k
        return results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            'total_evaluated': self.total_evaluated,
            'total_filtered_by_params': self.total_filtered,
            'avg_evaluation_time_ms': np.mean(self.evaluation_times) if self.evaluation_times else 0,
            'total_evaluation_time_s': sum(self.evaluation_times) / 1000
        }


class ProxyValidation:
    """
    Validates that zero-cost proxies discriminate effectively in the search space.

    Before relying on proxies, this class tests whether high-proxy architectures
    actually outperform low-proxy architectures after training.
    """

    def __init__(
        self,
        proxy_evaluator: ZeroCostProxy,
        trainer_fn,  # Function that trains a model and returns accuracy
        num_validation_samples: int = 10
    ):
        """
        Initialize proxy validation.

        Args:
            proxy_evaluator: ZeroCostProxy instance
            trainer_fn: Function(model) -> trained_accuracy
            num_validation_samples: Number of models to fully train for validation
        """
        self.proxy = proxy_evaluator
        self.trainer_fn = trainer_fn
        self.num_samples = num_validation_samples

    def validate(
        self,
        architectures: List[nn.Module]
    ) -> Dict[str, Any]:
        """
        Validate proxy effectiveness.

        Args:
            architectures: List of candidate architectures

        Returns:
            Dictionary with validation results including correlation
        """
        # Evaluate all with proxies
        proxy_scores = []
        for model in architectures:
            score = self.proxy.evaluate(model)
            proxy_scores.append((model, score))

        # Sort by combined score
        proxy_scores.sort(key=lambda x: x[1].combined_score, reverse=True)

        # Select top and bottom for full training
        n = self.num_samples // 2
        top_models = proxy_scores[:n]
        bottom_models = proxy_scores[-n:]

        # Train and evaluate
        top_trained_accuracies = []
        bottom_trained_accuracies = []

        for model, proxy_score in top_models:
            accuracy = self.trainer_fn(model)
            top_trained_accuracies.append(accuracy)

        for model, proxy_score in bottom_models:
            accuracy = self.trainer_fn(model)
            bottom_trained_accuracies.append(accuracy)

        # Compute statistics
        avg_top = np.mean(top_trained_accuracies)
        avg_bottom = np.mean(bottom_trained_accuracies)

        # Simple correlation estimate
        all_proxy = [s[1].combined_score for s in top_models + bottom_models]
        all_trained = top_trained_accuracies + bottom_trained_accuracies

        correlation = np.corrcoef(all_proxy, all_trained)[0, 1]

        return {
            'avg_top_accuracy': avg_top,
            'avg_bottom_accuracy': avg_bottom,
            'accuracy_gap': avg_top - avg_bottom,
            'proxy_trained_correlation': correlation,
            'proxies_effective': avg_top > avg_bottom and correlation > 0.5,
            'top_proxy_scores': [s[1].combined_score for s in top_models],
            'bottom_proxy_scores': [s[1].combined_score for s in bottom_models],
            'top_trained_scores': top_trained_accuracies,
            'bottom_trained_scores': bottom_trained_accuracies
        }


def quick_proxy_test():
    """Quick test of proxy computation."""
    from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent

    print("=" * 60)
    print("ZERO-COST PROXY TEST")
    print("=" * 60)

    proxy = ZeroCostProxy(input_dim=181, seq_len=10)

    # Test each architecture type
    architectures = {
        'TRN': TransparentRNN(181, 128, 181, num_layers=2),
        'RSAN': RecursiveSelfAttention(181, 128, 181, num_heads=4, max_recursion=3),
        'Transformer': TransformerToMAgent(181, 128, 181, num_layers=2, num_heads=4)
    }

    for name, model in architectures.items():
        print(f"\n--- {name} ---")
        score = proxy.evaluate(model)
        print(f"  SynFlow: {score.synflow:.4f}")
        print(f"  NASWOT: {score.naswot:.4f}")
        print(f"  GradNorm: {score.gradnorm:.4f}")
        print(f"  Parameters: {score.param_count:,}")
        print(f"  Combined: {score.combined_score:.4f}")
        print(f"  Time: {score.evaluation_time_ms:.1f}ms")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    quick_proxy_test()
