"""
Cognitive Extensions for ToM-NAS Agents
Implements working memory, episodic memory, planning, imagination, and curiosity

These modules can be added to any base architecture to extend cognitive capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class MemoryEntry:
    """Single entry in episodic memory."""
    content: torch.Tensor
    timestamp: int
    importance: float
    context: Dict[str, Any] = field(default_factory=dict)
    emotional_valence: float = 0.0
    retrieval_count: int = 0


class WorkingMemory(nn.Module):
    """
    Working Memory System - maintains active representations.

    Features:
    - Fixed capacity (typically 4-7 items)
    - Attention-based gating for what enters
    - Decay over time
    - Central executive for coordination
    """

    def __init__(self, capacity: int = 7, item_dim: int = 128, decay_rate: float = 0.95):
        super().__init__()
        self.capacity = capacity
        self.item_dim = item_dim
        self.decay_rate = decay_rate

        # Memory slots
        self.register_buffer('slots', torch.zeros(capacity, item_dim))
        self.register_buffer('slot_strengths', torch.zeros(capacity))

        # Gating mechanism for encoding
        self.encoding_gate = nn.Sequential(
            nn.Linear(item_dim * 2, item_dim),
            nn.ReLU(),
            nn.Linear(item_dim, 1),
            nn.Sigmoid()
        )

        # Attention for retrieval
        self.query_proj = nn.Linear(item_dim, item_dim)
        self.key_proj = nn.Linear(item_dim, item_dim)
        self.value_proj = nn.Linear(item_dim, item_dim)

        # Central executive
        self.central_exec = nn.GRUCell(item_dim, item_dim)
        self.exec_state = None

    def encode(self, item: torch.Tensor, force: bool = False) -> bool:
        """
        Attempt to encode item into working memory.

        Args:
            item: Tensor [item_dim] to encode
            force: If True, always encode (replacing weakest)

        Returns:
            True if encoded, False if rejected
        """
        item = item.detach()

        # Find weakest slot
        weakest_idx = torch.argmin(self.slot_strengths).item()

        if force or self.slot_strengths[weakest_idx] < 0.5:
            # Check encoding gate
            if self.exec_state is None:
                self.exec_state = torch.zeros(1, self.item_dim, device=item.device)

            gate_input = torch.cat([item.unsqueeze(0), self.exec_state], dim=-1)
            gate_value = self.encoding_gate(gate_input).item()

            if force or gate_value > 0.5:
                self.slots[weakest_idx] = item
                self.slot_strengths[weakest_idx] = 1.0
                return True

        return False

    def retrieve(self, query: torch.Tensor, top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve items from working memory based on query.

        Returns:
            Tuple of (retrieved items, attention weights)
        """
        # Project query and keys
        q = self.query_proj(query.unsqueeze(0))  # [1, item_dim]
        k = self.key_proj(self.slots)  # [capacity, item_dim]
        v = self.value_proj(self.slots)  # [capacity, item_dim]

        # Attention scores
        scores = torch.matmul(q, k.T) / (self.item_dim ** 0.5)  # [1, capacity]
        scores = scores * self.slot_strengths.unsqueeze(0)  # Weight by strength
        weights = F.softmax(scores, dim=-1)

        # Weighted retrieval
        retrieved = torch.matmul(weights, v)  # [1, item_dim]

        return retrieved.squeeze(0), weights.squeeze(0)

    def decay(self):
        """Apply decay to all slot strengths."""
        self.slot_strengths = self.slot_strengths * self.decay_rate

    def update_executive(self, input_state: torch.Tensor):
        """Update central executive state."""
        if self.exec_state is None:
            self.exec_state = torch.zeros(1, self.item_dim, device=input_state.device)
        self.exec_state = self.central_exec(input_state.unsqueeze(0), self.exec_state)

    def get_contents(self) -> List[Tuple[torch.Tensor, float]]:
        """Get current memory contents with strengths."""
        return [(self.slots[i], self.slot_strengths[i].item())
                for i in range(self.capacity)]

    def clear(self):
        """Clear all memory slots."""
        self.slots.zero_()
        self.slot_strengths.zero_()
        self.exec_state = None


class EpisodicMemory(nn.Module):
    """
    Episodic Memory System - stores experiences.

    Features:
    - Unlimited capacity (with compression/forgetting)
    - Content-addressable retrieval
    - Temporal organization
    - Emotional tagging
    - Consolidation (important memories strengthened)
    """

    def __init__(self, item_dim: int = 128, max_memories: int = 1000):
        super().__init__()
        self.item_dim = item_dim
        self.max_memories = max_memories

        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.current_time = 0

        # Encoding networks
        self.encoder = nn.Sequential(
            nn.Linear(item_dim, item_dim),
            nn.ReLU(),
            nn.Linear(item_dim, item_dim)
        )

        # Importance estimator
        self.importance_net = nn.Sequential(
            nn.Linear(item_dim, item_dim // 2),
            nn.ReLU(),
            nn.Linear(item_dim // 2, 1),
            nn.Sigmoid()
        )

        # Retrieval networks
        self.query_net = nn.Linear(item_dim, item_dim)

    def store(self, experience: torch.Tensor, context: Dict = None,
             emotional_valence: float = 0.0) -> int:
        """
        Store an experience in episodic memory.

        Args:
            experience: Tensor [item_dim] representing the experience
            context: Optional context information
            emotional_valence: Emotional tag (-1 to 1)

        Returns:
            Index of stored memory
        """
        experience = experience.detach()

        # Encode
        encoded = self.encoder(experience.unsqueeze(0)).squeeze(0)

        # Estimate importance
        importance = self.importance_net(experience.unsqueeze(0)).item()

        # Create entry
        entry = MemoryEntry(
            content=encoded,
            timestamp=self.current_time,
            importance=importance,
            context=context or {},
            emotional_valence=emotional_valence
        )

        self.memories.append(entry)
        self.current_time += 1

        # Consolidation - forget low importance memories if over capacity
        if len(self.memories) > self.max_memories:
            self._consolidate()

        return len(self.memories) - 1

    def retrieve(self, query: torch.Tensor, k: int = 5,
                recency_weight: float = 0.3) -> List[MemoryEntry]:
        """
        Retrieve memories similar to query.

        Args:
            query: Query tensor
            k: Number of memories to retrieve
            recency_weight: How much to weight recent memories

        Returns:
            List of top-k most relevant memories
        """
        if not self.memories:
            return []

        query_vec = self.query_net(query.unsqueeze(0)).squeeze(0)

        # Score all memories
        scored_memories = []
        for idx, mem in enumerate(self.memories):
            # Similarity score
            similarity = F.cosine_similarity(
                query_vec.unsqueeze(0),
                mem.content.unsqueeze(0)
            ).item()

            # Recency score (more recent = higher)
            recency = (mem.timestamp + 1) / (self.current_time + 1)

            # Combined score
            score = (1 - recency_weight) * similarity + recency_weight * recency
            score *= (0.5 + 0.5 * mem.importance)  # Boost important memories

            scored_memories.append((score, idx, mem))

        # Sort and return top k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m for _, _, m in scored_memories[:k]]

    def retrieve_by_emotion(self, valence: float, k: int = 5) -> List[MemoryEntry]:
        """Retrieve memories with similar emotional valence."""
        if not self.memories:
            return []

        scored = [(abs(m.emotional_valence - valence), m) for m in self.memories]
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:k]]

    def _consolidate(self):
        """Remove low-importance memories to stay under capacity."""
        # Sort by importance (keep important ones)
        self.memories.sort(key=lambda m: m.importance, reverse=True)
        # Keep top max_memories
        self.memories = self.memories[:self.max_memories]

    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        if not self.memories:
            return {'count': 0}

        importances = [m.importance for m in self.memories]
        valences = [m.emotional_valence for m in self.memories]

        return {
            'count': len(self.memories),
            'mean_importance': np.mean(importances),
            'mean_valence': np.mean(valences),
            'time_span': self.current_time
        }


class PlanningModule(nn.Module):
    """
    Planning System - forward simulation and goal pursuit.

    Features:
    - Goal representation
    - Action sequence generation
    - Forward model for simulation
    - Plan evaluation
    - Replanning on failure
    """

    def __init__(self, state_dim: int = 128, action_dim: int = 32,
                 max_plan_length: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_plan_length = max_plan_length

        # Goal encoder
        self.goal_encoder = nn.Linear(state_dim, state_dim)

        # Forward model (predict next state given state and action)
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )

        # Policy (generate action given state and goal)
        self.policy = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, action_dim),
            nn.Tanh()
        )

        # Value estimator (how good is state for achieving goal)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1),
            nn.Sigmoid()
        )

        # Current plan storage
        self.current_goal: Optional[torch.Tensor] = None
        self.current_plan: List[torch.Tensor] = []
        self.plan_step = 0

    def set_goal(self, goal: torch.Tensor):
        """Set a new goal and clear current plan."""
        self.current_goal = self.goal_encoder(goal.unsqueeze(0)).squeeze(0)
        self.current_plan = []
        self.plan_step = 0

    def generate_plan(self, current_state: torch.Tensor,
                     horizon: int = None) -> List[torch.Tensor]:
        """
        Generate action plan to reach goal from current state.

        Uses forward model to simulate trajectory.
        """
        if self.current_goal is None:
            return []

        horizon = horizon or self.max_plan_length
        plan = []
        state = current_state.clone()

        for _ in range(horizon):
            # Generate action
            policy_input = torch.cat([state, self.current_goal], dim=-1)
            action = self.policy(policy_input.unsqueeze(0)).squeeze(0)
            plan.append(action)

            # Simulate next state
            forward_input = torch.cat([state, action], dim=-1)
            state = self.forward_model(forward_input.unsqueeze(0)).squeeze(0)

            # Check if goal reached
            value_input = torch.cat([state, self.current_goal], dim=-1)
            value = self.value_net(value_input.unsqueeze(0)).item()
            if value > 0.9:
                break

        self.current_plan = plan
        self.plan_step = 0
        return plan

    def get_next_action(self) -> Optional[torch.Tensor]:
        """Get next action from current plan."""
        if self.plan_step >= len(self.current_plan):
            return None
        action = self.current_plan[self.plan_step]
        self.plan_step += 1
        return action

    def evaluate_plan(self, start_state: torch.Tensor,
                     plan: List[torch.Tensor]) -> float:
        """Evaluate quality of a plan."""
        if self.current_goal is None or not plan:
            return 0.0

        state = start_state.clone()
        for action in plan:
            forward_input = torch.cat([state, action], dim=-1)
            state = self.forward_model(forward_input.unsqueeze(0)).squeeze(0)

        value_input = torch.cat([state, self.current_goal], dim=-1)
        return self.value_net(value_input.unsqueeze(0)).item()

    def needs_replanning(self, current_state: torch.Tensor,
                        threshold: float = 0.3) -> bool:
        """Check if current plan needs revision."""
        if not self.current_plan or self.current_goal is None:
            return True

        # Evaluate remaining plan from current state
        remaining_plan = self.current_plan[self.plan_step:]
        if not remaining_plan:
            return True

        plan_value = self.evaluate_plan(current_state, remaining_plan)
        return plan_value < threshold


class ImaginationModule(nn.Module):
    """
    Imagination/Simulation System.

    Features:
    - Counterfactual simulation ("what if...")
    - Mental rehearsal
    - Creative recombination of experiences
    """

    def __init__(self, state_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim

        # World model for simulation
        self.world_model = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )

        # Intervention model (for counterfactuals)
        self.intervention = nn.Linear(state_dim, state_dim)

        # Combination network (for creative recombination)
        self.combiner = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )

    def simulate(self, initial_state: torch.Tensor,
                action_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """Simulate trajectory given action sequence."""
        states = [initial_state]
        state = initial_state

        for action in action_sequence:
            input_vec = torch.cat([state, action], dim=-1)
            next_state = self.world_model(input_vec.unsqueeze(0)).squeeze(0)
            states.append(next_state)
            state = next_state

        return states

    def counterfactual(self, actual_state: torch.Tensor,
                      intervention: torch.Tensor) -> torch.Tensor:
        """
        Generate counterfactual: "What if intervention had occurred?"
        """
        modified = self.intervention(intervention.unsqueeze(0)).squeeze(0)
        combined = torch.cat([actual_state, modified], dim=-1)
        return self.world_model(combined.unsqueeze(0)).squeeze(0)

    def creative_combine(self, state_a: torch.Tensor,
                        state_b: torch.Tensor,
                        blend: float = 0.5) -> torch.Tensor:
        """Creatively combine two states/experiences."""
        combined = torch.cat([state_a, state_b], dim=-1)
        raw = self.combiner(combined.unsqueeze(0)).squeeze(0)
        # Blend between combination and original
        return blend * raw + (1 - blend) * (0.5 * state_a + 0.5 * state_b)


class CuriosityModule(nn.Module):
    """
    Curiosity/Intrinsic Motivation System.

    Features:
    - Prediction error as curiosity signal
    - Novelty detection
    - Information gain estimation
    """

    def __init__(self, state_dim: int = 128, action_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Forward predictor (predicts next state)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )

        # Inverse model (predicts action from state transition)
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, action_dim)
        )

        # Feature extractor (for computing intrinsic reward)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU()
        )

        # Novelty memory (for detecting new situations)
        self.seen_features: List[torch.Tensor] = []
        self.max_novelty_memory = 100

    def compute_curiosity(self, state: torch.Tensor, action: torch.Tensor,
                         next_state: torch.Tensor) -> Dict[str, float]:
        """
        Compute curiosity reward based on prediction error.

        Returns dict with curiosity metrics.
        """
        # Predict next state
        pred_input = torch.cat([state, action], dim=-1)
        predicted_next = self.predictor(pred_input.unsqueeze(0)).squeeze(0)

        # Prediction error (curiosity signal)
        features_actual = self.feature_extractor(next_state.unsqueeze(0)).squeeze(0)
        features_pred = self.feature_extractor(predicted_next.unsqueeze(0)).squeeze(0)
        pred_error = F.mse_loss(features_pred, features_actual.detach()).item()

        # Novelty (distance to nearest seen feature)
        novelty = self._compute_novelty(features_actual)

        # Update novelty memory
        self._update_novelty_memory(features_actual)

        return {
            'prediction_error': pred_error,
            'novelty': novelty,
            'curiosity_reward': pred_error * 0.5 + novelty * 0.5
        }

    def _compute_novelty(self, features: torch.Tensor) -> float:
        """Compute novelty as distance to nearest seen feature."""
        if not self.seen_features:
            return 1.0

        min_dist = float('inf')
        for seen in self.seen_features:
            dist = torch.norm(features - seen).item()
            min_dist = min(min_dist, dist)

        # Normalize to [0, 1]
        return min(1.0, min_dist / 10.0)

    def _update_novelty_memory(self, features: torch.Tensor):
        """Update novelty memory with new features."""
        self.seen_features.append(features.detach().clone())
        if len(self.seen_features) > self.max_novelty_memory:
            # Remove oldest
            self.seen_features.pop(0)

    def get_exploration_bonus(self, state: torch.Tensor) -> float:
        """Get exploration bonus for a state (encourages novelty)."""
        features = self.feature_extractor(state.unsqueeze(0)).squeeze(0)
        return self._compute_novelty(features)


class CognitiveAgent(nn.Module):
    """
    Agent with full cognitive extension suite.

    Combines:
    - Base architecture (TRN, RSAN, or Transformer)
    - Working memory
    - Episodic memory
    - Planning
    - Imagination
    - Curiosity
    """

    def __init__(self, base_architecture: nn.Module, hidden_dim: int = 128):
        super().__init__()
        self.base = base_architecture
        self.hidden_dim = hidden_dim

        # Cognitive modules
        self.working_memory = WorkingMemory(capacity=7, item_dim=hidden_dim)
        self.episodic_memory = EpisodicMemory(item_dim=hidden_dim)
        self.planning = PlanningModule(state_dim=hidden_dim, action_dim=32)
        self.imagination = ImaginationModule(state_dim=hidden_dim)
        self.curiosity = CuriosityModule(state_dim=hidden_dim, action_dim=32)

        # Integration layers
        self.memory_integration = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, x: torch.Tensor, store_experience: bool = True) -> Dict[str, Any]:
        """
        Forward pass with cognitive extensions.
        """
        # Base architecture processing
        base_output = self.base(x)
        hidden = base_output.get('final_hidden', base_output.get('hidden_states', x)[:, -1, :])

        if hidden.dim() > 2:
            hidden = hidden[:, -1, :]

        batch_size = hidden.shape[0]

        # Working memory retrieval
        wm_retrieved, wm_weights = self.working_memory.retrieve(hidden[0])

        # Episodic memory retrieval
        em_memories = self.episodic_memory.retrieve(hidden[0], k=3)
        if em_memories:
            em_content = torch.stack([m.content for m in em_memories]).mean(0)
        else:
            em_content = torch.zeros(self.hidden_dim, device=hidden.device)

        # Integrate memories with current state
        memory_concat = torch.cat([hidden[0], wm_retrieved, em_content], dim=-1)
        integrated = self.memory_integration(memory_concat.unsqueeze(0))

        # Store in working memory
        self.working_memory.encode(hidden[0])

        # Store in episodic memory if requested
        if store_experience:
            beliefs = base_output.get('beliefs', torch.zeros_like(hidden[0]))
            emotional = beliefs[15:38].mean().item() if len(beliefs) > 38 else 0.0
            self.episodic_memory.store(hidden[0], emotional_valence=emotional)

        # Update working memory decay
        self.working_memory.decay()

        # Add cognitive info to output
        output = base_output.copy()
        output['integrated_state'] = integrated
        output['working_memory_weights'] = wm_weights
        output['episodic_retrievals'] = len(em_memories)

        return output

    def plan_action(self, current_state: torch.Tensor,
                   goal: torch.Tensor) -> torch.Tensor:
        """Generate action to achieve goal."""
        self.planning.set_goal(goal)
        plan = self.planning.generate_plan(current_state)
        if plan:
            return plan[0]
        return torch.zeros(32, device=current_state.device)

    def imagine_outcome(self, state: torch.Tensor,
                       action: torch.Tensor) -> torch.Tensor:
        """Imagine outcome of action."""
        return self.imagination.simulate(state, [action])[-1]

    def get_curiosity_reward(self, state: torch.Tensor, action: torch.Tensor,
                            next_state: torch.Tensor) -> float:
        """Get curiosity-based intrinsic reward."""
        metrics = self.curiosity.compute_curiosity(state, action, next_state)
        return metrics['curiosity_reward']
