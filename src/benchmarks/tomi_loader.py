"""
ToMi Dataset Loader for Theory of Mind Evaluation

The ToMi (Theory of Mind Inventory) dataset contains Sally-Anne style
false belief scenarios for evaluating ToM capabilities.

This module provides:
- ToMi dataset parsing and loading
- Batch generation for training
- Evaluation metrics for ToM accuracy
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ToMiQuestion:
    """A question about an agent's belief or knowledge."""

    question_text: str
    question_type: str  # 'first_order', 'second_order', 'reality', 'memory'
    target_agent: str
    correct_answer: str
    answer_choices: List[str]
    requires_tom: bool = True


@dataclass
class ToMiExample:
    """A complete ToMi scenario with context and questions."""

    story_id: str
    story_text: str
    events: List[Dict[str, Any]]
    questions: List[ToMiQuestion]
    agents: List[str]
    locations: List[str]
    objects: List[str]

    # Metadata
    num_agents: int = 2
    has_false_belief: bool = True
    belief_order: int = 1


@dataclass
class EventEncoder:
    """Encodes ToMi events for neural network processing."""

    # Vocabulary mappings
    action_vocab: Dict[str, int] = field(default_factory=dict)
    agent_vocab: Dict[str, int] = field(default_factory=dict)
    object_vocab: Dict[str, int] = field(default_factory=dict)
    locations_vocab: Dict[str, int] = field(default_factory=dict)

    embedding_dim: int = 64

    def __post_init__(self):
        """Initialize default vocabularies."""
        if not self.action_vocab:
            self.action_vocab = {
                "enter": 0,
                "exit": 1,
                "move": 2,
                "put": 3,
                "take": 4,
                "look": 5,
                "say": 6,
                "think": 7,
                "see": 8,
                "hide": 9,
            }
        if not self.agent_vocab:
            self.agent_vocab = {
                "Sally": 0,
                "Anne": 1,
                "Observer": 2,
                "Mary": 3,
                "John": 4,
                "Alice": 5,
                "Bob": 6,
                "Charlie": 7,
                "<PAD>": 8,
                "<UNK>": 9,
            }
        if not self.object_vocab:
            self.object_vocab = {
                "marble": 0,
                "ball": 1,
                "toy": 2,
                "book": 3,
                "key": 4,
                "apple": 5,
                "coin": 6,
                "<PAD>": 7,
                "<UNK>": 8,
            }
        if not self.locations_vocab:
            self.locations_vocab = {
                "basket": 0,
                "box": 1,
                "drawer": 2,
                "room": 3,
                "table": 4,
                "shelf": 5,
                "pocket": 6,
                "bag": 7,
                "<PAD>": 8,
                "<UNK>": 9,
            }

    def encode_event(self, event: Dict[str, Any]) -> torch.Tensor:
        """Encode a single event as a tensor."""
        action_idx = self.action_vocab.get(event.get("action", ""), len(self.action_vocab) - 1)
        agent_idx = self.agent_vocab.get(event.get("agent", ""), len(self.agent_vocab) - 1)
        obj_idx = self.object_vocab.get(event.get("object", ""), len(self.object_vocab) - 1)
        loc_from = self.locations_vocab.get(event.get("from_location", ""), len(self.locations_vocab) - 1)
        loc_to = self.locations_vocab.get(event.get("to_location", ""), len(self.locations_vocab) - 1)

        # Create one-hot encoded tensor
        encoding = torch.zeros(self.embedding_dim)
        encoding[action_idx] = 1.0
        encoding[10 + agent_idx] = 1.0
        encoding[20 + obj_idx] = 1.0
        encoding[30 + loc_from] = 1.0
        encoding[40 + loc_to] = 1.0

        # Add observer information
        observers = event.get("observed_by", [])
        for i, agent in enumerate(self.agent_vocab.keys()):
            if agent in observers:
                encoding[50 + i] = 1.0

        return encoding

    def encode_sequence(self, events: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode a sequence of events."""
        encodings = [self.encode_event(e) for e in events]
        return torch.stack(encodings) if encodings else torch.zeros(1, self.embedding_dim)


class ToMiParser:
    """Parser for ToMi dataset files."""

    def __init__(self):
        self.examples: List[ToMiExample] = []

    def parse_file(self, filepath: str) -> List[ToMiExample]:
        """Parse a ToMi format file."""
        examples = []
        path = Path(filepath)

        if path.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
                for item in data:
                    example = self._parse_json_item(item)
                    examples.append(example)
        elif path.suffix == ".txt":
            examples = self._parse_text_file(filepath)

        return examples

    def _parse_json_item(self, item: Dict) -> ToMiExample:
        """Parse a JSON format item."""
        questions = []
        for q in item.get("questions", []):
            question = ToMiQuestion(
                question_text=q.get("question", ""),
                question_type=q.get("type", "first_order"),
                target_agent=q.get("target_agent", ""),
                correct_answer=q.get("answer", ""),
                answer_choices=q.get("choices", []),
                requires_tom=q.get("requires_tom", True),
            )
            questions.append(question)

        return ToMiExample(
            story_id=item.get("id", str(random.randint(0, 10000))),
            story_text=item.get("story", ""),
            events=item.get("events", []),
            questions=questions,
            agents=item.get("agents", ["Sally", "Anne"]),
            locations=item.get("locations", ["basket", "box"]),
            objects=item.get("objects", ["marble"]),
            num_agents=len(item.get("agents", ["Sally", "Anne"])),
            has_false_belief=item.get("has_false_belief", True),
            belief_order=item.get("belief_order", 1),
        )

    def _parse_text_file(self, filepath: str) -> List[ToMiExample]:
        """Parse a text format file (original ToMi format)."""
        examples = []
        current_story = []
        current_questions = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_story:
                        example = self._create_example_from_text(current_story, current_questions)
                        examples.append(example)
                        current_story = []
                        current_questions = []
                elif "?" in line:
                    current_questions.append(line)
                else:
                    current_story.append(line)

        # Handle last example
        if current_story:
            example = self._create_example_from_text(current_story, current_questions)
            examples.append(example)

        return examples

    def _create_example_from_text(self, story_lines: List[str], question_lines: List[str]) -> ToMiExample:
        """Create a ToMiExample from parsed text lines."""
        story_text = " ".join(story_lines)
        events = self._extract_events_from_text(story_lines)

        questions = []
        for q_line in question_lines:
            # Parse question format: "Question? Answer"
            parts = q_line.split("\t") if "\t" in q_line else [q_line, ""]
            question = ToMiQuestion(
                question_text=parts[0],
                question_type="first_order",  # Default
                target_agent="Sally",  # Default
                correct_answer=parts[1] if len(parts) > 1 else "",
                answer_choices=["basket", "box"],  # Default
                requires_tom=True,
            )
            questions.append(question)

        return ToMiExample(
            story_id=str(random.randint(0, 10000)),
            story_text=story_text,
            events=events,
            questions=questions,
            agents=["Sally", "Anne"],
            locations=["basket", "box"],
            objects=["marble"],
        )

    def _extract_events_from_text(self, story_lines: List[str]) -> List[Dict]:
        """Extract structured events from story text."""
        events = []
        # Simplified parsing - in production would use NLP
        for line in story_lines:
            event = {"raw_text": line, "observed_by": ["Observer"]}

            # Simple keyword extraction
            if "put" in line.lower() or "place" in line.lower():
                event["action"] = "put"
            elif "move" in line.lower():
                event["action"] = "move"
            elif "enter" in line.lower():
                event["action"] = "enter"
            elif "exit" in line.lower() or "leave" in line.lower():
                event["action"] = "exit"

            events.append(event)

        return events


class ToMiDataset:
    """
    Complete ToMi dataset handler with batch generation and evaluation.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.examples: List[ToMiExample] = []
        self.encoder = EventEncoder()
        self.parser = ToMiParser()

        if data_dir:
            self.load_from_directory(data_dir)
        else:
            # Generate synthetic examples for testing
            self.generate_synthetic_examples()

    def load_from_directory(self, data_dir: str):
        """Load ToMi examples from a directory."""
        path = Path(data_dir)
        for file in path.glob("*.json"):
            examples = self.parser.parse_file(str(file))
            self.examples.extend(examples)
        for file in path.glob("*.txt"):
            examples = self.parser.parse_file(str(file))
            self.examples.extend(examples)

    def generate_synthetic_examples(self, num_examples: int = 100):
        """Generate synthetic Sally-Anne style examples."""
        agent_pairs = [("Sally", "Anne"), ("Mary", "John"), ("Alice", "Bob"), ("Charlie", "Diana"), ("Eve", "Frank")]
        objects = ["marble", "ball", "toy", "book", "key"]
        location_pairs = [("basket", "box"), ("drawer", "shelf"), ("bag", "pocket"), ("table", "cupboard")]

        for i in range(num_examples):
            agent1, agent2 = random.choice(agent_pairs)
            obj = random.choice(objects)
            loc1, loc2 = random.choice(location_pairs)

            # Create Sally-Anne scenario
            events = [
                {"action": "enter", "agent": agent1, "to_location": "room", "observed_by": [agent1, "Observer"]},
                {
                    "action": "enter",
                    "agent": agent2,
                    "to_location": "room",
                    "observed_by": [agent1, agent2, "Observer"],
                },
                {
                    "action": "put",
                    "agent": agent1,
                    "object": obj,
                    "to_location": loc1,
                    "observed_by": [agent1, agent2, "Observer"],
                },
                {"action": "exit", "agent": agent1, "from_location": "room", "observed_by": [agent2, "Observer"]},
                {
                    "action": "move",
                    "agent": agent2,
                    "object": obj,
                    "from_location": loc1,
                    "to_location": loc2,
                    "observed_by": [agent2, "Observer"],
                },  # Sally doesn't see this!
                {
                    "action": "enter",
                    "agent": agent1,
                    "to_location": "room",
                    "observed_by": [agent1, agent2, "Observer"],
                },
            ]

            story_text = (
                f"{agent1} enters the room. {agent2} is also in the room. "
                f"{agent1} puts the {obj} in the {loc1}. {agent1} leaves the room. "
                f"{agent2} moves the {obj} from the {loc1} to the {loc2}. "
                f"{agent1} returns to the room."
            )

            # First-order false belief question
            questions = [
                ToMiQuestion(
                    question_text=f"Where will {agent1} look for the {obj}?",
                    question_type="first_order",
                    target_agent=agent1,
                    correct_answer=loc1,  # False belief - they think it's still there
                    answer_choices=[loc1, loc2],
                    requires_tom=True,
                ),
                ToMiQuestion(
                    question_text=f"Where is the {obj} really?",
                    question_type="reality",
                    target_agent="Observer",
                    correct_answer=loc2,
                    answer_choices=[loc1, loc2],
                    requires_tom=False,
                ),
            ]

            # Add second-order belief for some examples
            if random.random() > 0.5:
                questions.append(
                    ToMiQuestion(
                        question_text=f"Where does {agent2} think {agent1} will look?",
                        question_type="second_order",
                        target_agent=agent2,
                        correct_answer=loc1,  # agent2 knows agent1 has false belief
                        answer_choices=[loc1, loc2],
                        requires_tom=True,
                    )
                )

            example = ToMiExample(
                story_id=f"synthetic_{i}",
                story_text=story_text,
                events=events,
                questions=questions,
                agents=[agent1, agent2, "Observer"],
                locations=[loc1, loc2, "room"],
                objects=[obj],
                has_false_belief=True,
                belief_order=2 if len(questions) > 2 else 1,
            )

            self.examples.append(example)

    def get_batch(
        self, batch_size: int, require_tom: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[ToMiExample]]:
        """
        Get a batch of examples for training.

        Returns:
            event_sequences: Tensor of shape (batch_size, max_seq_len, embedding_dim)
            targets: Tensor of shape (batch_size,) with target location indices
            examples: List of ToMiExample for reference
        """
        # Filter examples
        filtered = [ex for ex in self.examples if not require_tom or ex.has_false_belief]

        if len(filtered) < batch_size:
            batch_indices = list(range(len(filtered)))
        else:
            batch_indices = random.sample(range(len(filtered)), batch_size)

        batch_examples = [filtered[i] for i in batch_indices]

        # Encode sequences
        sequences = []
        max_len = max(len(ex.events) for ex in batch_examples) if batch_examples else 1

        for example in batch_examples:
            seq = self.encoder.encode_sequence(example.events)
            # Pad to max length
            if seq.shape[0] < max_len:
                padding = torch.zeros(max_len - seq.shape[0], self.encoder.embedding_dim)
                seq = torch.cat([seq, padding], dim=0)
            sequences.append(seq)

        event_sequences = torch.stack(sequences) if sequences else torch.zeros(1, 1, self.encoder.embedding_dim)

        # Create targets using location indices
        targets = []
        for idx in batch_indices:
            example = filtered[idx]
            correct_answer = example.questions[0].correct_answer
            target_idx = self.encoder.locations_vocab.get(correct_answer, 0)
            targets.append(target_idx)
        targets = torch.tensor(targets, dtype=torch.long)

        return event_sequences, targets, batch_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ToMiExample:
        return self.examples[idx]


class ToMiEvaluator:
    """Evaluator for ToM performance on ToMi dataset."""

    def __init__(self, dataset: ToMiDataset):
        self.dataset = dataset

    def evaluate(self, model, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate a model on ToMi examples.

        Returns dictionary with:
        - tom_accuracy: Accuracy on questions requiring ToM
        - control_accuracy: Accuracy on reality questions (no ToM needed)
        - specificity: tom_accuracy - control_accuracy (should be > 0)
        - first_order_accuracy: Accuracy on first-order beliefs
        - second_order_accuracy: Accuracy on second-order beliefs
        """
        results = {
            "tom_correct": 0,
            "tom_total": 0,
            "control_correct": 0,
            "control_total": 0,
            "first_order_correct": 0,
            "first_order_total": 0,
            "second_order_correct": 0,
            "second_order_total": 0,
        }

        model.eval()
        with torch.no_grad():
            for i in range(min(num_samples, len(self.dataset))):
                example = self.dataset[i]
                events, _, _ = self.dataset.get_batch(1)

                # Get model predictions
                output = model(events)

                for question in example.questions:
                    # Get predicted answer
                    predicted_idx = output.argmax(dim=-1).item()
                    correct_idx = self.dataset.encoder.locations_vocab.get(question.correct_answer, 0)

                    is_correct = predicted_idx == correct_idx

                    if question.requires_tom:
                        results["tom_total"] += 1
                        if is_correct:
                            results["tom_correct"] += 1
                    else:
                        results["control_total"] += 1
                        if is_correct:
                            results["control_correct"] += 1

                    if question.question_type == "first_order":
                        results["first_order_total"] += 1
                        if is_correct:
                            results["first_order_correct"] += 1
                    elif question.question_type == "second_order":
                        results["second_order_total"] += 1
                        if is_correct:
                            results["second_order_correct"] += 1

        # Calculate metrics
        tom_accuracy = results["tom_correct"] / max(results["tom_total"], 1)
        control_accuracy = results["control_correct"] / max(results["control_total"], 1)
        first_order_accuracy = results["first_order_correct"] / max(results["first_order_total"], 1)
        second_order_accuracy = results["second_order_correct"] / max(results["second_order_total"], 1)

        return {
            "tom_accuracy": tom_accuracy,
            "control_accuracy": control_accuracy,
            "specificity": tom_accuracy - control_accuracy,
            "first_order_accuracy": first_order_accuracy,
            "second_order_accuracy": second_order_accuracy,
            "total_examples": num_samples,
        }


# Export
__all__ = [
    "ToMiDataset",
    "ToMiParser",
    "ToMiEvaluator",
    "ToMiExample",
    "ToMiQuestion",
    "EventEncoder",
]
