"""
Tiny Recursive Model (TRM) - Neural Transition Function

The TRM is specialized to take a CognitiveBlock, run a state transition,
and output the next CognitiveBlock. It acts as the learned approximation
within RecursiveSimulationNodes.

At deeper recursion levels, instead of running full simulations,
we use the TRM to approximate what agents will do. This makes
N-th order ToM computationally tractable.

Architecture:
- Input: CognitiveBlock tensor representation
- Hidden: Recurrent state (captures context)
- Output: Next CognitiveBlock tensor + action prediction

Theoretical Foundation:
- Predictive Processing (Clark, Friston)
- Learned World Models (Ha & Schmidhuber)
- Neural Turing Machines (Graves)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mentalese import (
    CognitiveBlock,
    PerceptBlock,
)


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model."""

    # Architecture
    input_dim: int = 128  # Dimension of cognitive block tensors
    hidden_dim: int = 256  # Hidden layer dimension
    output_dim: int = 128  # Output dimension
    num_layers: int = 2  # Number of recurrent layers

    # Prediction heads
    num_actions: int = 32  # Number of possible actions
    belief_dim: int = 64  # Dimension of belief prediction

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True

    # Training
    learning_rate: float = 1e-4
    max_sequence_length: int = 32  # Maximum steps to process


@dataclass
class CognitiveTransition:
    """
    A single cognitive transition for training/inference.

    Captures the before/after of a cognitive state change.
    """

    input_block: CognitiveBlock
    context: Dict[str, Any]
    output_block: CognitiveBlock
    action_taken: Optional[str] = None
    reward: float = 0.0

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert transition to training tensors."""
        input_tensor = torch.from_numpy(self.input_block.to_tensor()).float()
        output_tensor = torch.from_numpy(self.output_block.to_tensor()).float()

        # Context embedding (simplified)
        context_features = []
        for key in ["timestep", "recursion_depth", "urgency"]:
            if key in self.context:
                context_features.append(float(self.context[key]))
            else:
                context_features.append(0.0)
        context_tensor = torch.tensor(context_features, dtype=torch.float32)

        return input_tensor, context_tensor, output_tensor


class TinyRecursiveModel(nn.Module):
    """
    The Tiny Recursive Model for cognitive state transitions.

    Takes a cognitive block representation and predicts:
    1. The next cognitive block state
    2. The action the agent will take
    3. Confidence in the prediction

    This model is trained to approximate full recursive simulation,
    enabling efficient deep ToM reasoning.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(8, config.hidden_dim // 4),  # Small context embedding
            nn.ReLU(),
        )

        # Recurrent core (GRU for efficiency)
        self.rnn = nn.GRU(
            input_size=config.hidden_dim + config.hidden_dim // 4,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        # Output heads
        # 1. Next state prediction
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # 2. Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_actions),
        )

        # 3. Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 4. Belief content prediction
        self.belief_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.belief_dim),
        )

        # Initialize hidden state projection
        self.init_hidden = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_dim))

    def forward(
        self, block_tensor: torch.Tensor, context: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TRM.

        Args:
            block_tensor: (batch, seq_len, input_dim) cognitive block tensors
            context: (batch, seq_len, context_dim) context information
            hidden: (num_layers, batch, hidden_dim) previous hidden state

        Returns:
            Tuple of:
                - next_state: (batch, seq_len, output_dim) predicted next state
                - action_logits: (batch, seq_len, num_actions) action predictions
                - confidence: (batch, seq_len, 1) prediction confidence
                - belief_embedding: (batch, seq_len, belief_dim) belief content
                - new_hidden: (num_layers, batch, hidden_dim) updated hidden state
        """
        batch_size = block_tensor.size(0)
        seq_len = block_tensor.size(1)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden.expand(-1, batch_size, -1).contiguous()

        # Project inputs
        block_emb = self.input_proj(block_tensor)
        context_emb = self.context_proj(context)

        # Combine block and context
        combined = torch.cat([block_emb, context_emb], dim=-1)

        # Pass through RNN
        rnn_out, new_hidden = self.rnn(combined, hidden)

        # Generate predictions
        next_state = self.state_head(rnn_out)
        action_logits = self.action_head(rnn_out)
        confidence = self.confidence_head(rnn_out)
        belief_embedding = self.belief_head(rnn_out)

        return next_state, action_logits, confidence, belief_embedding, new_hidden

    def predict_single_step(
        self, block: CognitiveBlock, context: Dict[str, Any], hidden: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        """
        Predict a single cognitive transition.

        Args:
            block: Input cognitive block
            context: Current context
            hidden: Previous hidden state

        Returns:
            Tuple of (next_state_array, action_index, confidence, new_hidden)
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors
            block_tensor = torch.from_numpy(block.to_tensor()).float()
            block_tensor = block_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)

            # Create context tensor
            context_features = [
                context.get("timestep", 0) / 100.0,
                context.get("recursion_depth", 0) / 5.0,
                context.get("urgency", 0.5),
                context.get("social_pressure", 0.5),
                context.get("time_pressure", 0.5),
                context.get("uncertainty", 0.5),
                context.get("stakes", 0.5),
                context.get("familiarity", 0.5),
            ]
            context_tensor = torch.tensor([context_features], dtype=torch.float32)
            context_tensor = context_tensor.unsqueeze(1)  # (1, 1, 8)

            # Forward pass
            next_state, action_logits, confidence, belief_emb, new_hidden = self.forward(
                block_tensor, context_tensor, hidden
            )

            # Extract predictions
            next_state_array = next_state[0, 0].numpy()
            action_probs = F.softmax(action_logits[0, 0], dim=-1)
            action_index = torch.argmax(action_probs).item()
            conf = confidence[0, 0, 0].item()

            return next_state_array, action_index, conf, new_hidden

    def rollout(
        self,
        initial_block: CognitiveBlock,
        context: Dict[str, Any],
        num_steps: int,
        action_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Roll out predictions for multiple steps.

        Args:
            initial_block: Starting cognitive state
            context: Initial context
            num_steps: Number of steps to predict
            action_names: Optional list of action names for decoding

        Returns:
            List of prediction dicts for each step
        """
        self.eval()
        predictions = []
        hidden = None

        current_state = initial_block.to_tensor()

        for step in range(num_steps):
            # Update context with step
            step_context = {**context, "timestep": step}

            # Create mock block from current state
            mock_block = PerceptBlock(
                perceived_entity="simulated",
            )
            # Override tensor representation
            mock_block._tensor_cache = current_state

            # Predict
            next_state, action_idx, confidence, hidden = self.predict_single_step(mock_block, step_context, hidden)

            # Decode action
            if action_names and action_idx < len(action_names):
                action_name = action_names[action_idx]
            else:
                action_name = f"action_{action_idx}"

            predictions.append(
                {
                    "step": step,
                    "action": action_name,
                    "action_index": action_idx,
                    "confidence": confidence,
                    "state": next_state,
                }
            )

            # Update state for next step
            current_state = next_state

            # Apply confidence decay
            context["uncertainty"] = 1.0 - confidence

        return predictions


class TRMTrainer:
    """
    Trainer for the Tiny Recursive Model.

    Trains the TRM on cognitive transition data collected
    from full recursive simulations.
    """

    def __init__(self, model: TinyRecursiveModel, config: TRMConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Loss functions
        self.state_loss = nn.MSELoss()
        self.action_loss = nn.CrossEntropyLoss()

        # Training history
        self.loss_history: List[float] = []

    def train_step(self, transitions: List[CognitiveTransition], action_to_idx: Dict[str, int]) -> Dict[str, float]:
        """
        Train on a batch of transitions.

        Args:
            transitions: List of cognitive transitions
            action_to_idx: Mapping from action names to indices

        Returns:
            Dict of loss values
        """
        self.model.train()

        # Prepare batch
        batch_inputs = []
        batch_contexts = []
        batch_targets = []
        batch_actions = []

        for t in transitions:
            in_t, ctx_t, out_t = t.to_tensors()
            batch_inputs.append(in_t)
            batch_contexts.append(ctx_t)
            batch_targets.append(out_t)

            if t.action_taken and t.action_taken in action_to_idx:
                batch_actions.append(action_to_idx[t.action_taken])
            else:
                batch_actions.append(0)  # Default action

        # Stack into tensors
        input_batch = torch.stack(batch_inputs).unsqueeze(1).to(self.device)
        context_batch = torch.stack(batch_contexts).unsqueeze(1).to(self.device)
        target_batch = torch.stack(batch_targets).to(self.device)
        action_batch = torch.tensor(batch_actions, dtype=torch.long).to(self.device)

        # Pad context to expected size
        if context_batch.size(-1) < 8:
            pad_size = 8 - context_batch.size(-1)
            context_batch = F.pad(context_batch, (0, pad_size))

        # Forward pass
        next_state, action_logits, confidence, belief_emb, _ = self.model(input_batch, context_batch)

        # Compute losses
        state_loss = self.state_loss(next_state.squeeze(1), target_batch)
        action_loss = self.action_loss(action_logits.squeeze(1), action_batch)

        # Total loss
        total_loss = state_loss + 0.5 * action_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        return {
            "total_loss": loss_val,
            "state_loss": state_loss.item(),
            "action_loss": action_loss.item(),
        }

    def evaluate(self, transitions: List[CognitiveTransition], action_to_idx: Dict[str, int]) -> Dict[str, float]:
        """Evaluate model on transitions."""
        self.model.eval()

        correct_actions = 0
        total_state_error = 0.0

        with torch.no_grad():
            for t in transitions:
                next_state, action_idx, confidence, _ = self.model.predict_single_step(
                    t.input_block,
                    t.context,
                )

                # Check action accuracy
                if t.action_taken and t.action_taken in action_to_idx:
                    expected_idx = action_to_idx[t.action_taken]
                    if action_idx == expected_idx:
                        correct_actions += 1

                # Compute state error
                expected_state = t.output_block.to_tensor()
                state_error = np.mean((next_state - expected_state) ** 2)
                total_state_error += state_error

        n = len(transitions)
        return {
            "action_accuracy": correct_actions / n if n > 0 else 0.0,
            "mean_state_error": total_state_error / n if n > 0 else 0.0,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "loss_history": self.loss_history,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint.get("loss_history", [])


class TRMEnsemble:
    """
    Ensemble of TRMs for more robust prediction.

    Uses multiple TRMs trained on different aspects of cognition
    and combines their predictions.
    """

    def __init__(self, models: List[TinyRecursiveModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        assert len(self.weights) == len(self.models)

    def predict(self, block: CognitiveBlock, context: Dict[str, Any]) -> Tuple[np.ndarray, int, float]:
        """
        Ensemble prediction from all models.

        Returns weighted average of predictions.
        """
        all_states = []
        all_action_probs = []
        all_confidences = []

        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                block_tensor = torch.from_numpy(block.to_tensor()).float()
                block_tensor = block_tensor.unsqueeze(0).unsqueeze(0)

                context_features = [
                    context.get("timestep", 0) / 100.0,
                    context.get("recursion_depth", 0) / 5.0,
                    context.get("urgency", 0.5),
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,  # Padding
                ]
                context_tensor = torch.tensor([context_features]).unsqueeze(1)

                next_state, action_logits, confidence, _, _ = model(block_tensor, context_tensor)

                all_states.append(next_state[0, 0].numpy() * weight)
                action_probs = F.softmax(action_logits[0, 0], dim=-1).numpy()
                all_action_probs.append(action_probs * weight)
                all_confidences.append(confidence[0, 0, 0].item() * weight)

        # Combine predictions
        ensemble_state = np.sum(all_states, axis=0)
        ensemble_action_probs = np.sum(all_action_probs, axis=0)
        ensemble_confidence = np.sum(all_confidences)

        action_idx = int(np.argmax(ensemble_action_probs))

        return ensemble_state, action_idx, ensemble_confidence
