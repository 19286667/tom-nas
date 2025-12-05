"""
ToMi (Theory of Mind Inventory) Benchmark Data Loader

This module loads and processes ToMi benchmark data for Theory of Mind evaluation.
ToMi contains text narratives with questions about agent beliefs, requiring
models to track information asymmetry and false beliefs.

Key features:
- Parses ToMi narratives into Event sequences with observation tracking
- Computes ground truth answers for belief questions
- Supports reality, first-order, and second-order belief questions
- Generates control (non-ToM) variants for comparison
"""

import torch
import numpy as np
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import random

from ..core.events import (
    Event, AgentBeliefs, Question, EventEncoder, ScenarioEncoder,
    AnswerDecoder, compute_ground_truth, QuestionType
)


@dataclass
class ToMiExample:
    """A single ToMi example with narrative, events, and questions."""
    narrative: str
    events: List[Event]
    questions: List[Question]
    story_id: str
    difficulty: str  # 'easy', 'medium', 'hard'


class ToMiParser:
    """
    Parser that converts ToMi narrative text into Event sequences.

    The parser handles standard ToMi patterns:
    - "[Agent] entered the [location]."
    - "[Agent] exited the [location]."
    - "[Agent] moved the [object] to the [location]."
    - "[Agent] put the [object] in the [container]."
    """

    # Common ToMi patterns
    PATTERNS = {
        'enter': re.compile(r'(\w+) entered the (\w+)\.?', re.IGNORECASE),
        'exit': re.compile(r'(\w+) exited the (\w+)\.?', re.IGNORECASE),
        'moved_to': re.compile(r'(\w+) moved the (\w+) to the (\w+)\.?', re.IGNORECASE),
        'put_in': re.compile(r'(\w+) put the (\w+) in the (\w+)\.?', re.IGNORECASE),
        'put_on': re.compile(r'(\w+) put the (\w+) on the (\w+)\.?', re.IGNORECASE),
        'is_in': re.compile(r'The (\w+) is in the (\w+)\.?', re.IGNORECASE),
        'left': re.compile(r'(\w+) left\.?', re.IGNORECASE),
    }

    # Question patterns
    QUESTION_PATTERNS = {
        'where_reality': re.compile(r'Where is the (\w+)\??', re.IGNORECASE),
        'where_think': re.compile(r'Where does (\w+) think the (\w+) is\??', re.IGNORECASE),
        'where_think_think': re.compile(r'Where does (\w+) think (\w+) thinks the (\w+) is\??', re.IGNORECASE),
        'where_really': re.compile(r'Where is the (\w+) really\??', re.IGNORECASE),
        'where_was': re.compile(r'Where was the (\w+) at the beginning\??', re.IGNORECASE),
    }

    def __init__(self):
        self.agent_locations: Dict[str, str] = {}
        self.object_locations: Dict[str, str] = {}
        self.current_room: Optional[str] = None
        self.timestamp = 0

    def reset(self):
        """Reset parser state."""
        self.agent_locations = {}
        self.object_locations = {}
        self.current_room = None
        self.timestamp = 0

    def _get_observers(self, location: Optional[str] = None) -> Set[str]:
        """Get all agents who can observe events at the given location."""
        if location is None:
            location = self.current_room

        observers = {'Observer'}  # Omniscient observer always sees

        for agent, loc in self.agent_locations.items():
            if loc == location:
                observers.add(agent)

        return observers

    def parse_sentence(self, sentence: str) -> Optional[Event]:
        """Parse a single sentence into an Event."""
        sentence = sentence.strip()
        if not sentence:
            return None

        self.timestamp += 1

        # Try enter pattern
        match = self.PATTERNS['enter'].match(sentence)
        if match:
            agent, location = match.groups()
            self.agent_locations[agent] = location
            self.current_room = location

            # Observers: everyone already in the room + the entering agent
            observers = self._get_observers(location)
            observers.add(agent)

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='enter',
                target_location=location,
                observed_by=observers
            )

        # Try exit pattern
        match = self.PATTERNS['exit'].match(sentence)
        if match:
            agent, location = match.groups()
            observers = self._get_observers(location)

            # Agent leaves, so they're removed from location
            if agent in self.agent_locations:
                del self.agent_locations[agent]

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='leave',
                source_location=location,
                observed_by=observers
            )

        # Try left pattern (simpler exit)
        match = self.PATTERNS['left'].match(sentence)
        if match:
            agent = match.group(1)
            location = self.agent_locations.get(agent, self.current_room)
            observers = self._get_observers(location)

            if agent in self.agent_locations:
                del self.agent_locations[agent]

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='leave',
                source_location=location,
                observed_by=observers
            )

        # Try moved_to pattern
        match = self.PATTERNS['moved_to'].match(sentence)
        if match:
            agent, obj, location = match.groups()
            old_location = self.object_locations.get(obj)
            self.object_locations[obj] = location

            # Observers are those in the same room as the agent
            agent_location = self.agent_locations.get(agent, self.current_room)
            observers = self._get_observers(agent_location)

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='move',
                object=obj,
                source_location=old_location,
                target_location=location,
                observed_by=observers
            )

        # Try put_in pattern
        match = self.PATTERNS['put_in'].match(sentence)
        if match:
            agent, obj, container = match.groups()
            self.object_locations[obj] = container

            agent_location = self.agent_locations.get(agent, self.current_room)
            observers = self._get_observers(agent_location)

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='put',
                object=obj,
                target_location=container,
                observed_by=observers
            )

        # Try put_on pattern
        match = self.PATTERNS['put_on'].match(sentence)
        if match:
            agent, obj, surface = match.groups()
            self.object_locations[obj] = surface

            agent_location = self.agent_locations.get(agent, self.current_room)
            observers = self._get_observers(agent_location)

            return Event(
                timestamp=self.timestamp,
                actor=agent,
                action='put',
                object=obj,
                target_location=surface,
                observed_by=observers
            )

        # Try is_in pattern (initial state)
        match = self.PATTERNS['is_in'].match(sentence)
        if match:
            obj, container = match.groups()
            self.object_locations[obj] = container

            return Event(
                timestamp=self.timestamp,
                actor='Narrator',
                action='put',
                object=obj,
                target_location=container,
                observed_by={'Observer'}  # Initial state known only to observer
            )

        return None

    def parse_question(self, question_text: str, locations: List[str]) -> Optional[Question]:
        """Parse a question string into a Question object."""
        question_text = question_text.strip()

        # Second-order: Where does X think Y thinks the Z is?
        match = self.QUESTION_PATTERNS['where_think_think'].match(question_text)
        if match:
            agent1, agent2, obj = match.groups()
            return Question(
                question_type='second_order',
                target_agent=agent1,
                secondary_agent=agent2,
                target_object=obj,
                answer_choices=locations
            )

        # First-order: Where does X think the Y is?
        match = self.QUESTION_PATTERNS['where_think'].match(question_text)
        if match:
            agent, obj = match.groups()
            return Question(
                question_type='first_order',
                target_agent=agent,
                target_object=obj,
                answer_choices=locations
            )

        # Reality: Where is the X?
        match = self.QUESTION_PATTERNS['where_reality'].match(question_text)
        if match:
            obj = match.group(1)
            return Question(
                question_type='reality',
                target_object=obj,
                answer_choices=locations
            )

        # Reality variant: Where is the X really?
        match = self.QUESTION_PATTERNS['where_really'].match(question_text)
        if match:
            obj = match.group(1)
            return Question(
                question_type='reality',
                target_object=obj,
                answer_choices=locations
            )

        # Memory: Where was the X at the beginning?
        match = self.QUESTION_PATTERNS['where_was'].match(question_text)
        if match:
            obj = match.group(1)
            return Question(
                question_type='memory',
                target_object=obj,
                answer_choices=locations
            )

        return None

    def parse_narrative(self, narrative: str) -> List[Event]:
        """Parse a complete narrative into a list of Events."""
        self.reset()

        sentences = re.split(r'[.!?]+', narrative)
        events = []

        for sentence in sentences:
            event = self.parse_sentence(sentence)
            if event:
                events.append(event)

        return events


class ToMiDataset:
    """
    Dataset class for ToMi Theory of Mind benchmark.

    Handles loading from files or generating synthetic examples.
    """

    def __init__(
        self,
        encoder: Optional[EventEncoder] = None,
        scenario_encoder: Optional[ScenarioEncoder] = None
    ):
        self.parser = ToMiParser()
        self.encoder = encoder or EventEncoder()
        self.scenario_encoder = scenario_encoder or ScenarioEncoder(self.encoder)

        self.examples: List[ToMiExample] = []
        self.train_examples: List[ToMiExample] = []
        self.val_examples: List[ToMiExample] = []
        self.test_examples: List[ToMiExample] = []

    def load_from_file(self, path: str):
        """
        Load ToMi data from a JSON file.

        Expected format:
        [
            {
                "story": "narrative text...",
                "questions": [
                    {"question": "Where is the X?", "answer": "location", "type": "reality"},
                    ...
                ],
                "id": "story_001"
            },
            ...
        ]
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ToMi data file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        for item in data:
            events = self.parser.parse_narrative(item['story'])

            # Extract locations from events
            locations = set()
            for event in events:
                if event.target_location:
                    locations.add(event.target_location)
                if event.source_location:
                    locations.add(event.source_location)
            locations = list(locations)

            questions = []
            for q in item.get('questions', []):
                question = self.parser.parse_question(q['question'], locations)
                if question:
                    question.correct_answer = q.get('answer')
                    questions.append(question)

            example = ToMiExample(
                narrative=item['story'],
                events=events,
                questions=questions,
                story_id=item.get('id', f'story_{len(self.examples)}'),
                difficulty=item.get('difficulty', 'medium')
            )
            self.examples.append(example)

    def generate_synthetic(self, num_examples: int = 1000):
        """
        Generate synthetic ToMi-style examples.

        Creates scenarios with information asymmetry for false belief testing.
        """
        agents = ['Sally', 'Anne', 'Tom', 'Mary', 'John']
        objects = ['marble', 'ball', 'apple', 'book', 'key']
        containers = ['basket', 'box', 'drawer', 'bag', 'cupboard']
        rooms = ['kitchen', 'bedroom', 'garden', 'bathroom', 'living_room']

        for i in range(num_examples):
            # Select characters and items
            agent1, agent2 = random.sample(agents, 2)
            obj = random.choice(objects)
            container1, container2 = random.sample(containers, 2)
            room = random.choice(rooms)

            # Build scenario with false belief
            events = []
            timestamp = 0

            # Both agents enter
            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent1,
                action='enter',
                target_location=room,
                observed_by={agent1, agent2, 'Observer'}
            ))

            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent2,
                action='enter',
                target_location=room,
                observed_by={agent1, agent2, 'Observer'}
            ))

            # Agent1 puts object in container1
            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent1,
                action='put',
                object=obj,
                target_location=container1,
                observed_by={agent1, agent2, 'Observer'}
            ))

            # Agent1 leaves
            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent1,
                action='leave',
                source_location=room,
                observed_by={agent2, 'Observer'}  # Agent1 doesn't observe after leaving
            ))

            # Agent2 moves object (agent1 doesn't see this)
            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent2,
                action='move',
                object=obj,
                source_location=container1,
                target_location=container2,
                observed_by={agent2, 'Observer'}  # Agent1 is gone
            ))

            # Agent1 returns
            timestamp += 1
            events.append(Event(
                timestamp=timestamp,
                actor=agent1,
                action='enter',
                target_location=room,
                observed_by={agent1, agent2, 'Observer'}
            ))

            # Build narrative
            narrative = (
                f"{agent1} entered the {room}. "
                f"{agent2} entered the {room}. "
                f"{agent1} put the {obj} in the {container1}. "
                f"{agent1} exited the {room}. "
                f"{agent2} moved the {obj} to the {container2}. "
                f"{agent1} entered the {room}."
            )

            # Create questions
            locations = [container1, container2]
            questions = [
                Question(
                    question_type='reality',
                    target_object=obj,
                    answer_choices=locations,
                    correct_answer=container2  # Reality: object is in container2
                ),
                Question(
                    question_type='first_order',
                    target_agent=agent1,
                    target_object=obj,
                    answer_choices=locations,
                    correct_answer=container1  # False belief: agent1 thinks container1
                ),
                Question(
                    question_type='first_order',
                    target_agent=agent2,
                    target_object=obj,
                    answer_choices=locations,
                    correct_answer=container2  # True belief: agent2 knows container2
                ),
                Question(
                    question_type='second_order',
                    target_agent=agent2,
                    secondary_agent=agent1,
                    target_object=obj,
                    answer_choices=locations,
                    correct_answer=container1  # Agent2 knows agent1 has false belief
                )
            ]

            # Compute ground truth for each question
            for question in questions:
                gt = compute_ground_truth(events, question)
                if gt != question.correct_answer:
                    # Verify our ground truth computation matches expected
                    print(f"Warning: Ground truth mismatch for {question.question_type}")
                    question.correct_answer = gt

            example = ToMiExample(
                narrative=narrative,
                events=events,
                questions=questions,
                story_id=f'synthetic_{i:04d}',
                difficulty='medium'
            )
            self.examples.append(example)

    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split examples into train/val/test sets."""
        random.shuffle(self.examples)

        n = len(self.examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        self.train_examples = self.examples[:train_end]
        self.val_examples = self.examples[train_end:val_end]
        self.test_examples = self.examples[val_end:]

    def encode_example(
        self,
        example: ToMiExample,
        question_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encode an example for model input.

        Returns:
            (input_tensor, target_tensor, correct_answer_idx)
        """
        if question_idx >= len(example.questions):
            question_idx = 0

        question = example.questions[question_idx]

        # Register entities
        self.encoder.register_from_events(example.events)
        for choice in question.answer_choices:
            self.encoder.register_location(choice)

        # Encode scenario
        input_tensor = self.scenario_encoder.encode_scenario(example.events, question)

        # Create target (correct answer as one-hot over locations)
        target = torch.zeros(len(self.encoder.locations_list))
        if question.correct_answer in self.encoder.locations_vocab:
            correct_idx = self.encoder.locations_vocab[question.correct_answer]
            target[correct_idx] = 1.0
        else:
            correct_idx = 0

        return input_tensor, target, correct_idx

    def get_batch(
        self,
        batch_size: int,
        split: str = 'train',
        question_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Get a batch of encoded examples.

        Args:
            batch_size: Number of examples
            split: 'train', 'val', or 'test'
            question_type: Optional filter for question type

        Returns:
            (inputs, targets, correct_indices)
        """
        if split == 'train':
            examples = self.train_examples
        elif split == 'val':
            examples = self.val_examples
        else:
            examples = self.test_examples

        if not examples:
            examples = self.examples

        # Filter by question type if specified
        valid_pairs = []
        for example in examples:
            for q_idx, question in enumerate(example.questions):
                if question_type is None or question.question_type == question_type:
                    valid_pairs.append((example, q_idx))

        if not valid_pairs:
            raise ValueError(f"No examples found for question_type={question_type}")

        # Sample batch
        sampled = random.choices(valid_pairs, k=batch_size)

        inputs = []
        targets = []
        correct_indices = []

        for example, q_idx in sampled:
            inp, tgt, idx = self.encode_example(example, q_idx)
            inputs.append(inp)
            targets.append(tgt)
            correct_indices.append(idx)

        # Pad inputs to same length
        max_len = max(inp.shape[0] for inp in inputs)
        padded_inputs = []
        for inp in inputs:
            if inp.shape[0] < max_len:
                padding = torch.zeros(max_len - inp.shape[0], inp.shape[1])
                inp = torch.cat([inp, padding], dim=0)
            padded_inputs.append(inp)

        # Pad targets to same length
        max_tgt_len = max(tgt.shape[0] for tgt in targets)
        padded_targets = []
        for tgt in targets:
            if tgt.shape[0] < max_tgt_len:
                padding = torch.zeros(max_tgt_len - tgt.shape[0])
                tgt = torch.cat([tgt, padding], dim=0)
            padded_targets.append(tgt)

        return (
            torch.stack(padded_inputs),
            torch.stack(padded_targets),
            correct_indices
        )


class ToMiEvaluator:
    """
    Evaluator for Theory of Mind performance on ToMi benchmark.
    """

    def __init__(self, dataset: ToMiDataset):
        self.dataset = dataset
        self.decoder = AnswerDecoder(dataset.encoder)

    def evaluate(
        self,
        model,
        split: str = 'test',
        num_examples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model on ToMi.

        Returns accuracy broken down by question type.
        """
        examples = getattr(self.dataset, f'{split}_examples', self.dataset.examples)
        if num_examples:
            examples = examples[:num_examples]

        results = {
            'reality': {'correct': 0, 'total': 0},
            'first_order': {'correct': 0, 'total': 0},
            'second_order': {'correct': 0, 'total': 0},
            'memory': {'correct': 0, 'total': 0}
        }

        model.eval()
        with torch.no_grad():
            for example in examples:
                for q_idx, question in enumerate(example.questions):
                    input_tensor, _, correct_idx = self.dataset.encode_example(example, q_idx)

                    # Model forward pass
                    output = model(input_tensor.unsqueeze(0))

                    if isinstance(output, dict):
                        beliefs = output.get('beliefs', output.get('hidden_states', None))
                        if beliefs is not None and beliefs.dim() > 1:
                            beliefs = beliefs[:, -1, :] if beliefs.dim() == 3 else beliefs
                    else:
                        beliefs = output

                    # Decode prediction
                    pred_location = self.decoder.decode_location(beliefs.squeeze())

                    # Check correctness
                    q_type = question.question_type
                    if q_type in results:
                        results[q_type]['total'] += 1
                        if pred_location == question.correct_answer:
                            results[q_type]['correct'] += 1

        # Compute accuracies
        accuracies = {}
        for q_type, counts in results.items():
            if counts['total'] > 0:
                accuracies[f'{q_type}_accuracy'] = counts['correct'] / counts['total']
            else:
                accuracies[f'{q_type}_accuracy'] = 0.0

        # Overall ToM accuracy (first + second order)
        tom_correct = results['first_order']['correct'] + results['second_order']['correct']
        tom_total = results['first_order']['total'] + results['second_order']['total']
        accuracies['tom_accuracy'] = tom_correct / tom_total if tom_total > 0 else 0.0

        # Control accuracy (reality + memory)
        ctrl_correct = results['reality']['correct'] + results['memory']['correct']
        ctrl_total = results['reality']['total'] + results['memory']['total']
        accuracies['control_accuracy'] = ctrl_correct / ctrl_total if ctrl_total > 0 else 0.0

        # ToM specificity
        accuracies['tom_specificity'] = accuracies['tom_accuracy'] - accuracies['control_accuracy']

        return accuracies


def test_tomi_loader():
    """Test ToMi loader functionality."""
    print("=" * 60)
    print("TOMI LOADER TEST")
    print("=" * 60)

    dataset = ToMiDataset()

    print("\nGenerating synthetic examples...")
    dataset.generate_synthetic(num_examples=100)
    print(f"Generated {len(dataset.examples)} examples")

    print("\nSplitting dataset...")
    dataset.split()
    print(f"Train: {len(dataset.train_examples)}, "
          f"Val: {len(dataset.val_examples)}, "
          f"Test: {len(dataset.test_examples)}")

    print("\nExample scenario:")
    example = dataset.examples[0]
    print(f"  Narrative: {example.narrative}")
    print(f"  Events: {len(example.events)}")
    for event in example.events:
        print(f"    t={event.timestamp}: {event.actor} {event.action} "
              f"observed_by={event.observed_by}")

    print(f"\n  Questions:")
    for q in example.questions:
        print(f"    {q.question_type}: {q.target_object} -> {q.correct_answer}")

    print("\nEncoding example...")
    inp, tgt, idx = dataset.encode_example(example, 0)
    print(f"  Input shape: {inp.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Correct index: {idx}")

    print("\nGetting batch...")
    inputs, targets, indices = dataset.get_batch(8, question_type='first_order')
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_tomi_loader()
