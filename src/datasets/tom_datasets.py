"""
ToM Benchmark Dataset Loaders for NAS Experiments

Unified interface for loading and processing Theory of Mind benchmark datasets:
- ToMi: Sally-Anne style false belief tasks
- BigToM: Procedurally generated ToM scenarios
- Hi-ToM: Higher-order Theory of Mind (up to 4th order)
- OpenToM: Longer narratives with personality traits
- SocialIQA: Social commonsense reasoning

Each loader provides a consistent interface for:
1. Loading raw data from various sources
2. Processing into a unified format
3. Converting to PyTorch tensors for training
4. Providing metadata about ToM order and task type
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import re


@dataclass
class ToMExample:
    """Unified representation of a ToM task example"""
    story: str
    question: str
    answer: str
    tom_order: int = 1
    belief_type: str = "unknown"  # true_belief, false_belief, etc.
    task_type: str = "unknown"    # sally_anne, unexpected_transfer, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'story': self.story,
            'question': self.question,
            'answer': self.answer,
            'tom_order': self.tom_order,
            'belief_type': self.belief_type,
            'task_type': self.task_type,
            'metadata': self.metadata,
        }


@dataclass
class ToMBatch:
    """Batch of ToM examples for training"""
    stories: List[str]
    questions: List[str]
    answers: List[str]
    tom_orders: torch.Tensor
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class TextEncoder:
    """Simple text encoder for converting stories/questions to tensors"""

    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2}
        self.idx2word: Dict[int, str] = {0: '<PAD>', 1: '<UNK>', 2: '<SEP>'}
        self.fitted = False

    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts: Dict[str, int] = {}

        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top vocab_size - 3 words
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:self.vocab_size - 3]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.fitted = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor of indices"""
        words = self._tokenize(text)[:self.max_length]
        indices = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>

        # Pad to max_length
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))

        return torch.tensor(indices, dtype=torch.long)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode batch of texts"""
        return torch.stack([self.encode(t) for t in texts])


class BaseToMDataset(Dataset, ABC):
    """Abstract base class for ToM datasets"""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = 'test',
        max_examples: Optional[int] = None,
        encoder: Optional[TextEncoder] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.split = split
        self.max_examples = max_examples
        self.encoder = encoder or TextEncoder()
        self.examples: List[ToMExample] = []
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """Load data from source - to be implemented by subclasses"""
        pass

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Combine story and question for input
        combined_text = f"{example.story} <SEP> {example.question}"

        item = {
            'story': example.story,
            'question': example.question,
            'answer': example.answer,
            'combined_text': combined_text,
            'tom_order': example.tom_order,
            'belief_type': example.belief_type,
            'task_type': example.task_type,
        }

        # Add encoded tensors if encoder is fitted
        if self.encoder.fitted:
            item['input_ids'] = self.encoder.encode(combined_text)

        return item

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Get PyTorch DataLoader for this dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for DataLoader"""
        stories = [b['story'] for b in batch]
        questions = [b['question'] for b in batch]
        answers = [b['answer'] for b in batch]
        tom_orders = torch.tensor([b['tom_order'] for b in batch])

        result = {
            'stories': stories,
            'questions': questions,
            'answers': answers,
            'tom_orders': tom_orders,
        }

        if 'input_ids' in batch[0]:
            result['input_ids'] = torch.stack([b['input_ids'] for b in batch])

        return result

    def fit_encoder(self):
        """Fit the text encoder on all examples"""
        all_texts = []
        for ex in self.examples:
            all_texts.append(ex.story)
            all_texts.append(ex.question)
            all_texts.append(ex.answer)
        self.encoder.fit(all_texts)


class ToMiDataset(BaseToMDataset):
    """
    ToMi Benchmark Dataset
    Sally-Anne style false belief tasks from Facebook Research

    Source: https://github.com/facebookresearch/ToMi
    Paper: "ToMi: A Large-Scale Benchmark for Theory of Mind Tasks"
    """

    def _load_data(self):
        """Load ToMi data - tries HuggingFace first, then local files"""
        try:
            self._load_from_huggingface()
        except Exception:
            self._load_from_local()

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]

    def _load_from_huggingface(self):
        """Load from HuggingFace datasets"""
        try:
            from datasets import load_dataset
            data = load_dataset("facebook/tomi", split=self.split)

            for item in data:
                belief_type = self._classify_belief_type(item)
                self.examples.append(ToMExample(
                    story=item.get('story', item.get('context', '')),
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    tom_order=1,
                    belief_type=belief_type,
                    task_type='sally_anne',
                    metadata={'source': 'huggingface'}
                ))
        except Exception as e:
            raise RuntimeError(f"Could not load from HuggingFace: {e}")

    def _load_from_local(self):
        """Load from local JSON files"""
        data_path = self.data_dir / "tomi" / f"{self.split}.json"

        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)

            for item in data:
                self.examples.append(ToMExample(
                    story=item.get('story', ''),
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    tom_order=1,
                    belief_type=item.get('belief_type', 'unknown'),
                    task_type='sally_anne',
                    metadata={'source': 'local'}
                ))
        else:
            # Generate synthetic ToMi-style data for testing
            self._generate_synthetic()

    def _generate_synthetic(self):
        """Generate synthetic Sally-Anne style examples"""
        templates = [
            {
                'story': "Sally put the marble in the basket. Sally left the room. Anne moved the marble to the box.",
                'question': "Where will Sally look for the marble?",
                'answer': "basket",
                'belief_type': "false_belief",
            },
            {
                'story': "Sally put the ball in the basket. Sally stayed in the room. Anne moved the ball to the box.",
                'question': "Where will Sally look for the ball?",
                'answer': "box",
                'belief_type': "true_belief",
            },
            {
                'story': "John put the apple in the cupboard. John left. Mary took the apple and put it in the fridge.",
                'question': "Where will John look for the apple?",
                'answer': "cupboard",
                'belief_type': "false_belief",
            },
            {
                'story': "Emma placed the book on the shelf. Emma went outside. Tom moved the book to the drawer.",
                'question': "Where does Emma think the book is?",
                'answer': "shelf",
                'belief_type': "false_belief",
            },
            {
                'story': "Alice put her keys in the bowl. Alice watched as Bob moved the keys to the hook.",
                'question': "Where will Alice look for her keys?",
                'answer': "hook",
                'belief_type': "true_belief",
            },
        ]

        # Repeat templates to create more examples
        num_repeats = max(1, 100 // len(templates))
        for _ in range(num_repeats):
            for template in templates:
                self.examples.append(ToMExample(
                    story=template['story'],
                    question=template['question'],
                    answer=template['answer'],
                    tom_order=1,
                    belief_type=template['belief_type'],
                    task_type='sally_anne',
                    metadata={'source': 'synthetic'}
                ))

    def _classify_belief_type(self, item: Dict) -> str:
        """Classify belief type from item data"""
        if 'belief_type' in item:
            return item['belief_type']

        story = item.get('story', '').lower()
        if 'left' in story or 'went' in story:
            return 'false_belief'
        return 'true_belief'


class BigToMDataset(BaseToMDataset):
    """
    BigToM Benchmark Dataset
    Procedurally generated ToM scenarios with controlled conditions

    Source: https://github.com/cicl-stanford/procedural-evals-tom
    Paper: "Procedural Evaluations for Theory of Mind"
    """

    def _load_data(self):
        """Load BigToM data"""
        try:
            self._load_from_local()
        except Exception:
            self._generate_synthetic()

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]

    def _load_from_local(self):
        """Load from local files"""
        data_path = self.data_dir / "bigtom" / f"bigtom_{self.split}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"BigToM data not found at {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            self.examples.append(ToMExample(
                story=item.get('scenario', item.get('story', '')),
                question=item.get('question', ''),
                answer=item.get('answer', ''),
                tom_order=1,
                belief_type=item.get('condition', 'unknown'),
                task_type='bigtom',
                metadata={
                    'control': item.get('control', None),
                    'source': 'local'
                }
            ))

    def _generate_synthetic(self):
        """Generate synthetic BigToM-style examples"""
        scenarios = [
            {
                'scenario': "In the morning, Alex puts the groceries in the kitchen. Alex goes to work. "
                           "During the day, the roommate moves the groceries to the garage.",
                'question': "When Alex returns, where will Alex first look for the groceries?",
                'answer': "kitchen",
                'condition': "false_belief",
            },
            {
                'scenario': "Maya leaves her phone on the table before her meeting. "
                           "While she is in the meeting, her colleague picks up the phone and puts it in Maya's bag.",
                'question': "After the meeting, where will Maya look for her phone?",
                'answer': "table",
                'condition': "false_belief",
            },
            {
                'scenario': "Chris parks the car in spot A. Chris sees the parking attendant move it to spot B.",
                'question': "Where does Chris think the car is?",
                'answer': "spot B",
                'condition': "true_belief",
            },
            {
                'scenario': "At the start of the day, the delivery driver places packages in the lobby. "
                           "The building manager, without telling anyone, moves them to the mailroom.",
                'question': "Where will residents first look for their packages?",
                'answer': "lobby",
                'condition': "false_belief",
            },
        ]

        num_repeats = max(1, 100 // len(scenarios))
        for _ in range(num_repeats):
            for s in scenarios:
                self.examples.append(ToMExample(
                    story=s['scenario'],
                    question=s['question'],
                    answer=s['answer'],
                    tom_order=1,
                    belief_type=s['condition'],
                    task_type='bigtom',
                    metadata={'source': 'synthetic'}
                ))


class HiToMDataset(BaseToMDataset):
    """
    Hi-ToM Benchmark Dataset
    Higher-order Theory of Mind tasks (up to 4th order)

    Source: https://github.com/ying-hui-he/Hi-ToM_dataset
    Paper: "Hi-ToM: A Benchmark for Evaluating Higher-Order Theory of Mind"
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = 'test',
        tom_order: int = 2,
        max_examples: Optional[int] = None,
        encoder: Optional[TextEncoder] = None,
    ):
        self.tom_order_filter = tom_order
        super().__init__(data_dir, split, max_examples, encoder)

    def _load_data(self):
        """Load Hi-ToM data"""
        try:
            self._load_from_local()
        except Exception:
            self._generate_synthetic()

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]

    def _load_from_local(self):
        """Load from local files"""
        data_path = self.data_dir / "hitom" / f"order_{self.tom_order_filter}" / f"{self.split}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Hi-ToM data not found at {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            self.examples.append(ToMExample(
                story=item.get('story', ''),
                question=item.get('question', ''),
                answer=item.get('answer', ''),
                tom_order=self.tom_order_filter,
                belief_type='higher_order',
                task_type='hitom',
                metadata={
                    'has_communication': item.get('tell', False),
                    'source': 'local'
                }
            ))

    def _generate_synthetic(self):
        """Generate synthetic Hi-ToM style examples for different orders"""
        order_templates = {
            1: [
                {
                    'story': "Alice puts a toy in Box A. Alice leaves. Bob moves the toy to Box B.",
                    'question': "Where does Alice think the toy is?",
                    'answer': "Box A",
                },
            ],
            2: [
                {
                    'story': "Alice puts a toy in Box A. Alice leaves. Bob moves the toy to Box B. "
                             "Carol watched Bob move the toy, but Alice didn't see Carol.",
                    'question': "Where does Alice think Carol thinks the toy is?",
                    'answer': "Box A",
                },
                {
                    'story': "Tom hides a gift in the closet. Mary sees this. Tom doesn't know Mary saw him.",
                    'question': "Where does Tom think Mary thinks the gift is?",
                    'answer': "unknown to Mary",
                },
            ],
            3: [
                {
                    'story': "Alice puts a toy in Box A. Alice leaves. Bob moves the toy to Box B. "
                             "Carol watched Bob but Alice doesn't know. Dave knows Carol saw it but Carol doesn't know Dave knows.",
                    'question': "Where does Dave think Carol thinks Alice thinks the toy is?",
                    'answer': "Box A",
                },
            ],
            4: [
                {
                    'story': "Alice hides treasure in Cave A. Bob sees this. Carol tells Dave that Bob saw Alice. "
                             "Eve overheard Carol but Carol doesn't know. Frank knows Eve heard but Eve doesn't know.",
                    'question': "Where does Frank think Eve thinks Carol thinks Bob thinks the treasure is?",
                    'answer': "Cave A",
                },
            ],
        }

        templates = order_templates.get(self.tom_order_filter, order_templates[1])
        num_repeats = max(1, 100 // len(templates))

        for _ in range(num_repeats):
            for template in templates:
                self.examples.append(ToMExample(
                    story=template['story'],
                    question=template['question'],
                    answer=template['answer'],
                    tom_order=self.tom_order_filter,
                    belief_type='higher_order',
                    task_type='hitom',
                    metadata={'source': 'synthetic'}
                ))


class OpenToMDataset(BaseToMDataset):
    """
    OpenToM Benchmark Dataset
    Longer narratives with explicit personality traits

    Source: https://github.com/seacowx/OpenToM
    Paper: "OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning"
    """

    def _load_data(self):
        """Load OpenToM data"""
        try:
            self._load_from_local()
        except Exception:
            self._generate_synthetic()

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]

    def _load_from_local(self):
        """Load from local files"""
        data_path = self.data_dir / "opentom" / "opentom.json"

        if not data_path.exists():
            raise FileNotFoundError(f"OpenToM data not found at {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            questions = item.get('questions', [item.get('question', '')])
            if isinstance(questions, str):
                questions = [questions]

            for q in questions:
                self.examples.append(ToMExample(
                    story=item.get('story', ''),
                    question=q if isinstance(q, str) else q.get('question', ''),
                    answer=item.get('answer', q.get('answer', '') if isinstance(q, dict) else ''),
                    tom_order=1,
                    belief_type='narrative',
                    task_type='opentom',
                    metadata={
                        'character_traits': item.get('traits', {}),
                        'question_types': item.get('question_types', []),
                        'source': 'local'
                    }
                ))

    def _generate_synthetic(self):
        """Generate synthetic OpenToM-style examples"""
        stories = [
            {
                'story': "Sarah is a cautious person who always double-checks things. "
                         "She placed her important documents in the safe before leaving for vacation. "
                         "Her helpful but forgetful brother Mike came to water the plants and "
                         "moved the documents to the desk 'for safekeeping'. Sarah was not informed.",
                'questions': [
                    {"question": "Where will Sarah look for her documents first?", "answer": "safe"},
                    {"question": "How will Sarah feel about Mike's action?", "answer": "upset or worried"},
                ],
                'traits': {'Sarah': 'cautious', 'Mike': 'forgetful but helpful'},
            },
            {
                'story': "James is known for being very organized. He keeps his tools in labeled boxes in the garage. "
                         "His teenage son borrowed the power drill for a project and returned it to the wrong box. "
                         "James hasn't been to the garage since.",
                'questions': [
                    {"question": "Where will James look for the drill?", "answer": "the labeled drill box"},
                    {"question": "What will James likely do when he can't find it?", "answer": "check other boxes systematically"},
                ],
                'traits': {'James': 'organized', 'son': 'careless'},
            },
        ]

        num_repeats = max(1, 50 // len(stories))
        for _ in range(num_repeats):
            for story_data in stories:
                for q in story_data['questions']:
                    self.examples.append(ToMExample(
                        story=story_data['story'],
                        question=q['question'],
                        answer=q['answer'],
                        tom_order=1,
                        belief_type='narrative',
                        task_type='opentom',
                        metadata={
                            'character_traits': story_data['traits'],
                            'source': 'synthetic'
                        }
                    ))


class SocialIQADataset(BaseToMDataset):
    """
    SocialIQA Dataset
    Social commonsense reasoning about people's actions and reactions

    Source: https://leaderboard.allenai.org/socialiqa/
    Paper: "SocialIQA: Commonsense Reasoning about Social Interactions"
    """

    def _load_data(self):
        """Load SocialIQA data"""
        try:
            self._load_from_huggingface()
        except Exception:
            try:
                self._load_from_local()
            except Exception:
                self._generate_synthetic()

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]

    def _load_from_huggingface(self):
        """Load from HuggingFace datasets"""
        from datasets import load_dataset
        data = load_dataset("social_i_qa", split=self.split)

        for item in data:
            # Convert multiple choice to story format
            context = item['context']
            question = item['question']

            # Get correct answer based on label
            answers = [item['answerA'], item['answerB'], item['answerC']]
            label = int(item['label']) - 1  # Labels are 1-indexed
            correct_answer = answers[label] if 0 <= label < 3 else answers[0]

            self.examples.append(ToMExample(
                story=context,
                question=question,
                answer=correct_answer,
                tom_order=1,
                belief_type='social_reasoning',
                task_type='socialqa',
                metadata={
                    'all_answers': answers,
                    'correct_label': label,
                    'source': 'huggingface'
                }
            ))

    def _load_from_local(self):
        """Load from local files"""
        data_path = self.data_dir / "socialqa" / f"{self.split}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"SocialIQA data not found at {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            self.examples.append(ToMExample(
                story=item.get('context', ''),
                question=item.get('question', ''),
                answer=item.get('answer', ''),
                tom_order=1,
                belief_type='social_reasoning',
                task_type='socialqa',
                metadata={'source': 'local'}
            ))

    def _generate_synthetic(self):
        """Generate synthetic SocialIQA-style examples"""
        scenarios = [
            {
                'context': "Jordan got a promotion at work after years of hard work.",
                'question': "How will Jordan feel?",
                'answer': "proud and happy",
            },
            {
                'context': "Alex forgot their best friend's birthday.",
                'question': "What will Alex want to do next?",
                'answer': "apologize and make it up to their friend",
            },
            {
                'context': "Casey's team lost the championship game in the final seconds.",
                'question': "How will Casey feel?",
                'answer': "disappointed and sad",
            },
            {
                'context': "Morgan helped an elderly person carry their groceries.",
                'question': "What will others think of Morgan?",
                'answer': "that Morgan is kind and helpful",
            },
            {
                'context': "Riley told a lie to their parents about where they were going.",
                'question': "How will Riley feel later?",
                'answer': "guilty",
            },
        ]

        num_repeats = max(1, 100 // len(scenarios))
        for _ in range(num_repeats):
            for s in scenarios:
                self.examples.append(ToMExample(
                    story=s['context'],
                    question=s['question'],
                    answer=s['answer'],
                    tom_order=1,
                    belief_type='social_reasoning',
                    task_type='socialqa',
                    metadata={'source': 'synthetic'}
                ))


class ToMDatasetLoader:
    """Unified interface for loading all ToM benchmark datasets"""

    DATASETS = {
        'tomi': ToMiDataset,
        'bigtom': BigToMDataset,
        'hitom': HiToMDataset,
        'opentom': OpenToMDataset,
        'socialqa': SocialIQADataset,
    }

    @classmethod
    def load(
        cls,
        dataset_name: str,
        split: str = 'test',
        data_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
        **kwargs
    ) -> BaseToMDataset:
        """
        Load a ToM benchmark dataset.

        Args:
            dataset_name: Name of dataset ('tomi', 'bigtom', 'hitom', 'opentom', 'socialqa')
            split: Data split ('train', 'validation', 'test')
            data_dir: Directory containing data files
            max_examples: Maximum number of examples to load
            **kwargs: Additional arguments for specific datasets (e.g., tom_order for hitom)

        Returns:
            Dataset object
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")

        dataset_class = cls.DATASETS[dataset_name]
        return dataset_class(
            data_dir=data_dir,
            split=split,
            max_examples=max_examples,
            **kwargs
        )

    @classmethod
    def load_tomi(cls, split: str = 'test', **kwargs) -> ToMiDataset:
        """Load ToMi dataset"""
        return cls.load('tomi', split=split, **kwargs)

    @classmethod
    def load_bigtom(cls, split: str = 'test', **kwargs) -> BigToMDataset:
        """Load BigToM dataset"""
        return cls.load('bigtom', split=split, **kwargs)

    @classmethod
    def load_hitom(cls, order: int = 2, split: str = 'test', **kwargs) -> HiToMDataset:
        """Load Hi-ToM dataset for specific order"""
        return cls.load('hitom', split=split, tom_order=order, **kwargs)

    @classmethod
    def load_opentom(cls, split: str = 'test', **kwargs) -> OpenToMDataset:
        """Load OpenToM dataset"""
        return cls.load('opentom', split=split, **kwargs)

    @classmethod
    def load_socialqa(cls, split: str = 'validation', **kwargs) -> SocialIQADataset:
        """Load SocialIQA dataset"""
        return cls.load('socialqa', split=split, **kwargs)

    @classmethod
    def load_all(
        cls,
        split: str = 'test',
        data_dir: Optional[str] = None,
        max_examples_per_dataset: Optional[int] = None
    ) -> Dict[str, BaseToMDataset]:
        """Load all ToM datasets"""
        datasets = {}
        for name in cls.DATASETS.keys():
            try:
                if name == 'hitom':
                    # Load multiple orders for Hi-ToM
                    for order in [1, 2, 3, 4]:
                        datasets[f'hitom_order{order}'] = cls.load(
                            name,
                            split=split,
                            data_dir=data_dir,
                            max_examples=max_examples_per_dataset,
                            tom_order=order
                        )
                else:
                    datasets[name] = cls.load(
                        name,
                        split=split,
                        data_dir=data_dir,
                        max_examples=max_examples_per_dataset
                    )
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

        return datasets


def get_task_complexity(dataset_name: str, tom_order: int = 1) -> Dict[str, Any]:
    """Get complexity information for a dataset task"""
    complexity_map = {
        'tomi': {
            'cognitive_demand': 'medium',
            'expected_architecture': 'skip + possible attention',
            'tom_order': 1,
        },
        'bigtom': {
            'cognitive_demand': 'medium-high',
            'expected_architecture': 'skip + attention',
            'tom_order': 1,
        },
        'hitom': {
            'cognitive_demand': {1: 'medium', 2: 'high', 3: 'very_high', 4: 'extreme'},
            'expected_architecture': 'deep recursion + attention',
            'tom_order': tom_order,
        },
        'opentom': {
            'cognitive_demand': 'high',
            'expected_architecture': 'full complexity',
            'tom_order': 1,
        },
        'socialqa': {
            'cognitive_demand': 'high',
            'expected_architecture': 'full complexity',
            'tom_order': 1,
        },
    }

    return complexity_map.get(dataset_name, {'cognitive_demand': 'unknown', 'tom_order': tom_order})
