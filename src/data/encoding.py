"""
Encode scenarios to tensors for neural networks.

This module provides ScenarioEncoder which converts Event sequences
into tensor representations compatible with the Soul Map ontology.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from .events import Event, Scenario


class ScenarioEncoder:
    """
    Encode scenarios to tensors compatible with Soul Map ontology.

    The encoder maps events to vectors in the ontology space, preserving
    information about actors, actions, objects, locations, and observers.

    Dimension allocation:
    - 0-9: Agent encoding (one-hot)
    - 10-19: Action encoding (one-hot)
    - 20-34: Object encoding (one-hot)
    - 35-49: Location encoding (one-hot)
    - 50-59: Observer encoding (multi-hot)
    - 60-69: Source location (for moves)
    - 70-79: Timestamp embedding
    - 80-169: Reserved for future use / ontology alignment
    - 170-180: Question type encoding
    """

    def __init__(self, ontology_dim: int = 181):
        """
        Initialize encoder.

        Args:
            ontology_dim: Dimension of output vectors (matches ontology)
        """
        self.ontology_dim = ontology_dim

        # Vocabularies
        self.agents = ['Sally', 'Anne', 'Emma', 'Jack', 'Mark', 'Lisa',
                      'Tom', 'Kate', 'Ben', 'Lily', 'Observer', 'Unknown']
        self.actions = ['enter', 'leave', 'put', 'move', 'take', 'say', 'none']
        self.objects = ['marble', 'ball', 'apple', 'orange', 'book', 'keys',
                       'toy', 'phone', 'hat', 'bag', 'none']
        self.locations = ['basket', 'box', 'container', 'drawer', 'cupboard',
                         'kitchen', 'bathroom', 'bedroom', 'garden', 'pantry',
                         'room', 'house', 'none']

        # Dimension allocations within ontology
        self.agent_start = 0
        self.agent_end = 12
        self.action_start = 12
        self.action_end = 19
        self.object_start = 19
        self.object_end = 30
        self.location_start = 30
        self.location_end = 43
        self.observer_start = 43
        self.observer_end = 55
        self.source_loc_start = 55
        self.source_loc_end = 68
        self.timestamp_start = 68
        self.timestamp_end = 78
        self.question_start = 170
        self.question_end = 181

    def encode_event(self, event: Event, max_timestamp: int = 10) -> torch.Tensor:
        """
        Encode single event to ontology-dimensional vector.

        Args:
            event: Event to encode
            max_timestamp: Maximum timestamp for normalization

        Returns:
            Tensor of shape (ontology_dim,)
        """
        vec = torch.zeros(self.ontology_dim)

        # Actor (one-hot)
        if event.actor in self.agents:
            idx = self.agents.index(event.actor)
            vec[self.agent_start + idx] = 1.0

        # Action (one-hot)
        if event.action in self.actions:
            idx = self.actions.index(event.action)
            vec[self.action_start + idx] = 1.0

        # Object (one-hot)
        if event.object and event.object in self.objects:
            idx = self.objects.index(event.object)
            vec[self.object_start + idx] = 1.0
        elif not event.object:
            # 'none' object
            vec[self.object_start + len(self.objects) - 1] = 1.0

        # Target location (one-hot)
        if event.target_location:
            loc = event.target_location
            if loc in self.locations:
                idx = self.locations.index(loc)
                vec[self.location_start + idx] = 1.0

        # Source location (one-hot, separate from target)
        if event.source_location:
            loc = event.source_location
            if loc in self.locations:
                idx = self.locations.index(loc)
                vec[self.source_loc_start + idx] = 1.0

        # Observers (multi-hot - multiple agents can observe)
        for observer in event.observed_by:
            if observer in self.agents:
                idx = self.agents.index(observer)
                vec[self.observer_start + idx] = 1.0

        # Timestamp (normalized position encoding)
        t = event.timestamp / max(max_timestamp, 1)
        vec[self.timestamp_start] = t
        # Sinusoidal position encoding for additional temporal info
        for i in range(1, min(10, self.timestamp_end - self.timestamp_start)):
            freq = 2 ** i
            vec[self.timestamp_start + i] = torch.sin(torch.tensor(t * freq * 3.14159))

        return vec

    def encode_question(self, scenario: Scenario) -> torch.Tensor:
        """
        Encode question as a vector.

        The question vector includes:
        - Question type (one-hot) at question_start (3 dims: 170-172)
        - Target agent encoded in agent slot (one-hot) at agent_start
        - Target object encoded in object slot (one-hot) at object_start

        We reuse the agent and object slots rather than creating new indices
        that would exceed the ontology dimension.
        """
        vec = torch.zeros(self.ontology_dim)

        # Question type encoding (indices 170, 171, 172)
        q_types = ['reality', 'first_order_belief', 'second_order_belief']
        if scenario.question_type in q_types:
            idx = q_types.index(scenario.question_type)
            vec[self.question_start + idx] = 1.0

        # Target agent - use agent slot (we're asking about this agent)
        if scenario.question_target_agent in self.agents:
            idx = self.agents.index(scenario.question_target_agent)
            # Mark in agent slot with value 1.0
            vec[self.agent_start + idx] = 1.0

        # Target object - mark in object slot
        if scenario.question_target_object in self.objects:
            idx = self.objects.index(scenario.question_target_object)
            vec[self.object_start + idx] = 1.0

        return vec

    def encode_scenario(self, scenario: Scenario,
                        max_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Encode full scenario including question.

        Args:
            scenario: Scenario to encode
            max_seq_len: Optional max sequence length (pads/truncates if set)

        Returns:
            Tensor of shape (seq_len, ontology_dim)
        """
        max_ts = max(e.timestamp for e in scenario.events) if scenario.events else 1

        # Encode events
        event_vecs = [self.encode_event(e, max_ts) for e in scenario.events]

        # Create question vector as final "event"
        question_vec = self.encode_question(scenario)
        event_vecs.append(question_vec)

        # Stack into sequence tensor
        result = torch.stack(event_vecs)  # Shape: (seq_len, ontology_dim)

        # Pad or truncate if max_seq_len specified
        if max_seq_len is not None:
            if result.size(0) < max_seq_len:
                padding = torch.zeros(max_seq_len - result.size(0), self.ontology_dim)
                result = torch.cat([result, padding], dim=0)
            elif result.size(0) > max_seq_len:
                result = result[:max_seq_len]

        return result

    def encode_batch(self, scenarios: List[Scenario]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode batch of scenarios with padding.

        Args:
            scenarios: List of scenarios to encode

        Returns:
            Tuple of:
            - inputs: Tensor of shape (batch_size, max_seq_len, ontology_dim)
            - targets: Tensor of shape (batch_size,) with location indices
        """
        # Find max sequence length
        max_len = max(len(s.events) + 1 for s in scenarios)  # +1 for question

        inputs = []
        targets = []

        for scenario in scenarios:
            encoded = self.encode_scenario(scenario, max_seq_len=max_len)
            inputs.append(encoded)

            # Target is location index
            target_idx = self.get_location_index(scenario.ground_truth_answer)
            targets.append(target_idx)

        return torch.stack(inputs), torch.tensor(targets)

    def decode_location(self, output: torch.Tensor) -> str:
        """
        Decode network output to location prediction.

        Args:
            output: Model output vector of shape (ontology_dim,)

        Returns:
            Predicted location name
        """
        # Extract location portion of output
        loc_logits = output[self.location_start:self.location_end]
        idx = loc_logits.argmax().item()

        if idx < len(self.locations):
            return self.locations[idx]
        return 'unknown'

    def get_location_index(self, location: str) -> int:
        """
        Get index for location (for computing loss).

        Args:
            location: Location name

        Returns:
            Index in locations vocabulary, or -1 if not found
        """
        if location in self.locations:
            return self.locations.index(location)
        return -1

    def get_num_locations(self) -> int:
        """Get number of possible location classes."""
        return len(self.locations)

    def create_answer_mask(self) -> torch.Tensor:
        """
        Create mask for valid answer positions in output.

        Useful for focusing loss on location prediction portion.
        """
        mask = torch.zeros(self.ontology_dim)
        mask[self.location_start:self.location_end] = 1.0
        return mask


class DataCollator:
    """
    Collate function for batching scenarios.

    Handles variable-length sequences through padding.
    """

    def __init__(self, encoder: ScenarioEncoder, max_seq_len: int = 20):
        self.encoder = encoder
        self.max_seq_len = max_seq_len

    def __call__(self, scenarios: List[Scenario]) -> Dict[str, torch.Tensor]:
        """
        Collate scenarios into batched tensors.

        Returns:
            Dict with:
            - 'input': (batch, seq_len, dim)
            - 'target': (batch,)
            - 'mask': (batch, seq_len) - attention mask
        """
        batch_inputs = []
        batch_targets = []
        batch_masks = []

        for scenario in scenarios:
            encoded = self.encoder.encode_scenario(scenario, self.max_seq_len)
            batch_inputs.append(encoded)

            target_idx = self.encoder.get_location_index(scenario.ground_truth_answer)
            batch_targets.append(target_idx)

            # Create attention mask (1 for real tokens, 0 for padding)
            real_len = len(scenario.events) + 1  # +1 for question
            mask = torch.zeros(self.max_seq_len)
            mask[:min(real_len, self.max_seq_len)] = 1.0
            batch_masks.append(mask)

        return {
            'input': torch.stack(batch_inputs),
            'target': torch.tensor(batch_targets),
            'mask': torch.stack(batch_masks)
        }
