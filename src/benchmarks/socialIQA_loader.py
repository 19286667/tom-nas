"""
SocialIQA Benchmark Loader

SocialIQA contains 38,000+ multiple choice questions about social situations:
- What will X want to do next?
- How would X feel?
- Why did X do that?
- What does X need to do before this?

These test naturalistic social reasoning beyond false belief.

Reference:
Sap et al. (2019) "SocialIQA: Commonsense Reasoning about Social Interactions"
https://leaderboard.allenai.org/socialiqa/
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SocialIQAExample:
    """A single SocialIQA question."""

    context: str
    question: str
    answer_a: str
    answer_b: str
    answer_c: str
    correct_label: int  # 0, 1, or 2 (maps to A, B, C)
    question_type: str  # intent, emotion, motivation, subsequent, prerequisite


@dataclass
class SocialIQAMetrics:
    """Evaluation metrics for SocialIQA."""

    accuracy: float
    intent_accuracy: float
    emotion_accuracy: float
    motivation_accuracy: float
    subsequent_accuracy: float
    prerequisite_accuracy: float
    num_examples: int


class SocialIQAEncoder:
    """Encode SocialIQA examples for neural network input."""

    def __init__(self, vocab_size: int = 10000, max_seq_len: int = 128, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Simple word-to-index vocabulary (in production, use tokenizer)
        self.word_to_idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.next_idx = 4

    def build_vocab(self, examples: List[SocialIQAExample]):
        """Build vocabulary from examples."""
        for example in examples:
            for text in [example.context, example.question, example.answer_a, example.answer_b, example.answer_c]:
                for word in text.lower().split():
                    if word not in self.word_to_idx and self.next_idx < self.vocab_size:
                        self.word_to_idx[word] = self.next_idx
                        self.next_idx += 1

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor of indices."""
        words = text.lower().split()[: self.max_seq_len - 2]
        indices = [self.word_to_idx.get("<CLS>", 2)]
        indices += [self.word_to_idx.get(w, 1) for w in words]
        indices.append(self.word_to_idx.get("<SEP>", 3))

        # Pad to max length
        while len(indices) < self.max_seq_len:
            indices.append(0)

        return torch.tensor(indices, dtype=torch.long)

    def encode_example(self, example: SocialIQAExample) -> Dict[str, torch.Tensor]:
        """Encode a complete example."""
        return {
            "context": self.encode_text(example.context),
            "question": self.encode_text(example.question),
            "answer_a": self.encode_text(example.answer_a),
            "answer_b": self.encode_text(example.answer_b),
            "answer_c": self.encode_text(example.answer_c),
            "label": torch.tensor(example.correct_label, dtype=torch.long),
        }


class SocialIQADataset:
    """
    Loader for SocialIQA benchmark.

    Download from: https://leaderboard.allenai.org/socialiqa/
    """

    # Question type classification keywords
    QUESTION_TYPES = {
        "intent": ["want", "intend", "try", "goal"],
        "emotion": ["feel", "emotion", "happy", "sad", "angry", "afraid", "mood"],
        "motivation": ["why", "reason", "because", "motive"],
        "subsequent": ["next", "after", "then", "will"],
        "prerequisite": ["before", "need", "first", "prerequisite"],
    }

    def __init__(self, data_dir: Optional[str] = None, split: str = "train"):
        """
        Initialize SocialIQA dataset.

        Args:
            data_dir: Path to SocialIQA data files
            split: 'train', 'dev', or 'test'
        """
        self.examples: List[SocialIQAExample] = []
        self.split = split
        self.encoder = SocialIQAEncoder()

        if data_dir:
            self.load_from_file(data_dir, split)
        else:
            # Generate synthetic examples for testing
            self.generate_synthetic(num_examples=200)

        self.encoder.build_vocab(self.examples)

    def load_from_file(self, data_dir: str, split: str = "train"):
        """
        Load from SocialIQA JSONL format.

        Expected files:
        - socialIQa_v1.4_{split}.jsonl (questions)
        - socialIQa_v1.4_{split}-labels.lst (labels)
        """
        path = Path(data_dir)

        # Try different file patterns
        data_files = list(path.glob(f"*{split}*.jsonl")) + list(path.glob(f"*{split}*.json"))

        for data_file in data_files:
            self._load_jsonl(data_file)

    def _load_jsonl(self, filepath: Path):
        """Load from JSONL format."""
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    example = self._parse_item(item)
                    if example:
                        self.examples.append(example)

    def _parse_item(self, item: Dict) -> Optional[SocialIQAExample]:
        """Parse a single JSONL item."""
        try:
            context = item.get("context", item.get("story", ""))
            question = item.get("question", "")
            answer_a = item.get("answerA", item.get("answer_a", ""))
            answer_b = item.get("answerB", item.get("answer_b", ""))
            answer_c = item.get("answerC", item.get("answer_c", ""))

            # Parse label (1, 2, 3 or A, B, C)
            label = item.get("correct", item.get("label", 1))
            if isinstance(label, str):
                label = {"A": 0, "B": 1, "C": 2}.get(label.upper(), 0)
            else:
                label = int(label) - 1  # Convert 1-indexed to 0-indexed

            question_type = self._classify_question_type(question)

            return SocialIQAExample(
                context=context,
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                answer_c=answer_c,
                correct_label=label,
                question_type=question_type,
            )
        except (KeyError, ValueError):
            return None

    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on keywords."""
        question_lower = question.lower()

        for qtype, keywords in self.QUESTION_TYPES.items():
            if any(kw in question_lower for kw in keywords):
                return qtype

        return "other"

    def generate_synthetic(self, num_examples: int = 200):
        """Generate synthetic SocialIQA-style examples for testing."""
        templates = self._get_synthetic_templates()

        for i in range(num_examples):
            template = random.choice(templates)

            # Fill in template with random choices
            example = self._fill_template(template)
            self.examples.append(example)

    def _get_synthetic_templates(self) -> List[Dict]:
        """Get templates for synthetic example generation."""
        return [
            # Intent questions
            {
                "context": "Alex told Jordan a secret about their surprise party.",
                "question": "Why did Alex do this?",
                "answers": ["Alex trusts Jordan", "Alex wants to hurt Jordan", "Alex forgot it was a secret"],
                "correct": 0,
                "type": "motivation",
            },
            {
                "context": "Sam got rejected from their dream job.",
                "question": "How does Sam feel?",
                "answers": ["Disappointed and sad", "Excited and happy", "Completely indifferent"],
                "correct": 0,
                "type": "emotion",
            },
            {
                "context": "Casey studied all night for the exam.",
                "question": "What will Casey want to do next?",
                "answers": ["Get some rest", "Study more", "Go to a party"],
                "correct": 0,
                "type": "subsequent",
            },
            {
                "context": "Riley helped Morgan move to a new apartment.",
                "question": "What does Riley need to do before this?",
                "answers": ["Clear their schedule", "Learn to drive", "Buy new furniture"],
                "correct": 0,
                "type": "prerequisite",
            },
            {
                "context": "Taylor gave an expensive gift to Quinn.",
                "question": "Why did Taylor do this?",
                "answers": ["Taylor cares about Quinn", "Taylor was forced to", "Taylor doesn't like Quinn"],
                "correct": 0,
                "type": "motivation",
            },
            {
                "context": "Jesse's friend canceled their plans at the last minute.",
                "question": "How would Jesse feel?",
                "answers": ["Annoyed and disappointed", "Relieved and happy", "Confused about what happened"],
                "correct": 0,
                "type": "emotion",
            },
            {
                "context": "Morgan just finished a marathon.",
                "question": "What will Morgan want to do next?",
                "answers": ["Rest and recover", "Run another marathon", "Go to work"],
                "correct": 0,
                "type": "subsequent",
            },
            {
                "context": "Alex is preparing to ask for a promotion.",
                "question": "What does Alex need to do before this?",
                "answers": ["Document their achievements", "Quit their job", "Criticize their boss"],
                "correct": 0,
                "type": "prerequisite",
            },
            {
                "context": "Jamie saw their coworker take credit for Jamie's work.",
                "question": "How would Jamie feel about this?",
                "answers": ["Frustrated and betrayed", "Proud of their coworker", "Completely unbothered"],
                "correct": 0,
                "type": "emotion",
            },
            {
                "context": "Chris stayed late to help a colleague finish a project.",
                "question": "Why did Chris do this?",
                "answers": [
                    "Chris is helpful and supportive",
                    "Chris was trying to show off",
                    "Chris had nothing else to do",
                ],
                "correct": 0,
                "type": "motivation",
            },
        ]

    def _fill_template(self, template: Dict) -> SocialIQAExample:
        """Create example from template with some variation."""
        # Add slight variations
        context = template["context"]
        question = template["question"]
        answers = template["answers"].copy()

        # Shuffle wrong answers
        if random.random() > 0.5:
            wrong_answers = [answers[1], answers[2]]
            random.shuffle(wrong_answers)
            answers[1], answers[2] = wrong_answers

        return SocialIQAExample(
            context=context,
            question=question,
            answer_a=answers[0],
            answer_b=answers[1],
            answer_c=answers[2],
            correct_label=template["correct"],
            question_type=template["type"],
        )

    def get_batch(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a batch of encoded examples.

        Returns:
            inputs: Dict of input tensors
            labels: Tensor of correct answer indices
        """
        if len(self.examples) < batch_size:
            indices = list(range(len(self.examples)))
        else:
            indices = random.sample(range(len(self.examples)), batch_size)

        batch_examples = [self.examples[i] for i in indices]

        # Encode all examples
        contexts = []
        questions = []
        answers_a = []
        answers_b = []
        answers_c = []
        labels = []

        for example in batch_examples:
            encoded = self.encoder.encode_example(example)
            contexts.append(encoded["context"])
            questions.append(encoded["question"])
            answers_a.append(encoded["answer_a"])
            answers_b.append(encoded["answer_b"])
            answers_c.append(encoded["answer_c"])
            labels.append(encoded["label"])

        inputs = {
            "context": torch.stack(contexts),
            "question": torch.stack(questions),
            "answer_a": torch.stack(answers_a),
            "answer_b": torch.stack(answers_b),
            "answer_c": torch.stack(answers_c),
        }

        return inputs, torch.stack(labels)

    def to_tomi_format(self) -> List[Dict]:
        """
        Convert to ToMi-compatible format for unified evaluation.

        This allows SocialIQA questions to be evaluated using the same
        metrics as ToMi false-belief scenarios.
        """
        tomi_format = []

        for example in self.examples:
            tomi_item = {
                "story": example.context,
                "events": [{"raw_text": example.context}],
                "questions": [
                    {
                        "question": example.question,
                        "type": example.question_type,
                        "answer": [example.answer_a, example.answer_b, example.answer_c][example.correct_label],
                        "choices": [example.answer_a, example.answer_b, example.answer_c],
                        "requires_tom": example.question_type in ["intent", "emotion", "motivation"],
                    }
                ],
            }
            tomi_format.append(tomi_item)

        return tomi_format

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SocialIQAExample:
        return self.examples[idx]


class SocialIQAEvaluator:
    """Evaluate models on SocialIQA benchmark."""

    def __init__(self, dataset: SocialIQADataset):
        self.dataset = dataset

    def evaluate(self, model: nn.Module, num_samples: int = 100, device: str = "cpu") -> SocialIQAMetrics:
        """
        Evaluate model on SocialIQA examples.

        Args:
            model: Model that takes encoded input and outputs answer logits
            num_samples: Number of examples to evaluate
            device: Device to run evaluation on

        Returns:
            SocialIQAMetrics with per-category accuracies
        """
        model = model.to(device)
        model.eval()

        results = {
            "correct": 0,
            "total": 0,
            "intent_correct": 0,
            "intent_total": 0,
            "emotion_correct": 0,
            "emotion_total": 0,
            "motivation_correct": 0,
            "motivation_total": 0,
            "subsequent_correct": 0,
            "subsequent_total": 0,
            "prerequisite_correct": 0,
            "prerequisite_total": 0,
        }

        with torch.no_grad():
            for i in range(min(num_samples, len(self.dataset))):
                example = self.dataset[i]
                encoded = self.dataset.encoder.encode_example(example)

                # Build input tensor
                # Concatenate context, question, and answers
                context = encoded["context"].unsqueeze(0).to(device)
                question = encoded["question"].unsqueeze(0).to(device)

                # Simple approach: concatenate all inputs
                input_tensor = torch.cat(
                    [
                        context.float(),
                        question.float(),
                    ],
                    dim=1,
                )

                # Get model prediction
                output = model(input_tensor)

                # Take first 3 outputs as answer logits
                answer_logits = output[:, :3]
                predicted_label = answer_logits.argmax(dim=-1).item()

                correct = predicted_label == example.correct_label

                results["total"] += 1
                if correct:
                    results["correct"] += 1

                # Per-type metrics
                qtype = example.question_type
                if qtype in ["intent", "emotion", "motivation", "subsequent", "prerequisite"]:
                    results[f"{qtype}_total"] += 1
                    if correct:
                        results[f"{qtype}_correct"] += 1

        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        return SocialIQAMetrics(
            accuracy=safe_div(results["correct"], results["total"]),
            intent_accuracy=safe_div(results["intent_correct"], results["intent_total"]),
            emotion_accuracy=safe_div(results["emotion_correct"], results["emotion_total"]),
            motivation_accuracy=safe_div(results["motivation_correct"], results["motivation_total"]),
            subsequent_accuracy=safe_div(results["subsequent_correct"], results["subsequent_total"]),
            prerequisite_accuracy=safe_div(results["prerequisite_correct"], results["prerequisite_total"]),
            num_examples=results["total"],
        )


# Export
__all__ = [
    "SocialIQADataset",
    "SocialIQAExample",
    "SocialIQAEncoder",
    "SocialIQAEvaluator",
    "SocialIQAMetrics",
]
