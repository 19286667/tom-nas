"""
Control Task Dataset Loaders for NAS Experiments

Non-ToM tasks for controlled comparison:
- Simple Sequence Prediction: Pattern matching without ToM
- bAbI Tasks: Various reasoning types (factual, relational)
- Sort-of-CLEVR: Relational reasoning without ToM
- Synthetic Relational: Object relationship tasks

These control tasks test whether discovered architectural features
(skip connections, attention) are specific to ToM or general to complex reasoning.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod


@dataclass
class ControlExample:
    """Unified representation of a control task example"""
    input_sequence: Any  # Can be list, string, or tensor
    target: Any
    task_type: str
    complexity: str = "simple"  # simple, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'input': self.input_sequence if not isinstance(self.input_sequence, np.ndarray)
                     else self.input_sequence.tolist(),
            'target': self.target if not isinstance(self.target, np.ndarray)
                      else self.target.tolist(),
            'task_type': self.task_type,
            'complexity': self.complexity,
            'metadata': self.metadata,
        }


class BaseControlDataset(Dataset, ABC):
    """Abstract base class for control datasets"""

    def __init__(
        self,
        num_examples: int = 10000,
        max_length: int = 50,
        seed: Optional[int] = 42,
    ):
        self.num_examples = num_examples
        self.max_length = max_length
        self.seed = seed
        self.examples: List[ControlExample] = []

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._generate_data()

    @abstractmethod
    def _generate_data(self):
        """Generate dataset - to be implemented by subclasses"""
        pass

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Convert to tensors
        if isinstance(example.input_sequence, (list, np.ndarray)):
            input_tensor = torch.tensor(example.input_sequence, dtype=torch.float32)
        else:
            input_tensor = example.input_sequence

        if isinstance(example.target, (int, float)):
            target_tensor = torch.tensor([example.target], dtype=torch.float32)
        elif isinstance(example.target, (list, np.ndarray)):
            target_tensor = torch.tensor(example.target, dtype=torch.float32)
        else:
            target_tensor = example.target

        return {
            'input': input_tensor,
            'target': target_tensor,
            'task_type': example.task_type,
            'complexity': example.complexity,
        }

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class SimpleSequenceDataset(BaseControlDataset):
    """
    Simple sequence prediction - no ToM required.

    Patterns include:
    - Periodic sequences (a, b, c, a, b, c, ...)
    - Arithmetic sequences (1, 2, 3, 4, ...)
    - Geometric sequences (1, 2, 4, 8, ...)
    - Fibonacci-like sequences
    """

    def __init__(
        self,
        num_examples: int = 10000,
        max_length: int = 50,
        pattern_types: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ):
        self.pattern_types = pattern_types or ['periodic', 'arithmetic', 'geometric', 'fibonacci']
        super().__init__(num_examples, max_length, seed)

    def _generate_data(self):
        """Generate simple sequence examples"""
        examples_per_type = self.num_examples // len(self.pattern_types)

        for pattern_type in self.pattern_types:
            for _ in range(examples_per_type):
                if pattern_type == 'periodic':
                    example = self._generate_periodic()
                elif pattern_type == 'arithmetic':
                    example = self._generate_arithmetic()
                elif pattern_type == 'geometric':
                    example = self._generate_geometric()
                elif pattern_type == 'fibonacci':
                    example = self._generate_fibonacci()
                else:
                    example = self._generate_periodic()

                self.examples.append(example)

    def _generate_periodic(self) -> ControlExample:
        """Generate periodic sequence (e.g., 1,2,3,1,2,3,...)"""
        period = np.random.randint(2, 6)
        base_pattern = list(range(period))
        length = np.random.randint(10, self.max_length)

        sequence = []
        for i in range(length):
            sequence.append(base_pattern[i % period])

        target = base_pattern[length % period]

        return ControlExample(
            input_sequence=sequence,
            target=target,
            task_type='periodic_sequence',
            complexity='simple',
            metadata={'period': period}
        )

    def _generate_arithmetic(self) -> ControlExample:
        """Generate arithmetic sequence (e.g., 2,4,6,8,...)"""
        start = np.random.randint(0, 10)
        diff = np.random.randint(1, 5)
        length = np.random.randint(5, min(20, self.max_length))

        sequence = [start + i * diff for i in range(length)]
        target = start + length * diff

        # Normalize
        max_val = max(sequence + [target])
        sequence = [s / max_val for s in sequence]
        target = target / max_val

        return ControlExample(
            input_sequence=sequence,
            target=target,
            task_type='arithmetic_sequence',
            complexity='simple',
            metadata={'start': start, 'diff': diff}
        )

    def _generate_geometric(self) -> ControlExample:
        """Generate geometric sequence (e.g., 1,2,4,8,...)"""
        start = np.random.randint(1, 4)
        ratio = np.random.choice([2, 3, 0.5])
        length = np.random.randint(4, min(10, self.max_length))

        sequence = [start * (ratio ** i) for i in range(length)]
        target = start * (ratio ** length)

        # Normalize to prevent overflow
        max_val = max(abs(s) for s in sequence + [target])
        if max_val > 0:
            sequence = [s / max_val for s in sequence]
            target = target / max_val

        return ControlExample(
            input_sequence=sequence,
            target=target,
            task_type='geometric_sequence',
            complexity='medium',
            metadata={'start': start, 'ratio': ratio}
        )

    def _generate_fibonacci(self) -> ControlExample:
        """Generate Fibonacci-like sequence"""
        a, b = np.random.randint(1, 5), np.random.randint(1, 5)
        length = np.random.randint(6, min(15, self.max_length))

        sequence = [a, b]
        for _ in range(length - 2):
            sequence.append(sequence[-1] + sequence[-2])

        target = sequence[-1] + sequence[-2]

        # Normalize
        max_val = max(sequence + [target])
        sequence = [s / max_val for s in sequence]
        target = target / max_val

        return ControlExample(
            input_sequence=sequence,
            target=target,
            task_type='fibonacci_sequence',
            complexity='medium',
            metadata={'start': (a, b)}
        )


class BAbIDataset(BaseControlDataset):
    """
    bAbI Tasks - Various reasoning types.

    Simplified implementation focusing on:
    - Task 1: Single supporting fact
    - Task 2: Two supporting facts
    - Task 3: Three supporting facts
    - Task 4: Two argument relations
    - Task 5: Three argument relations

    These test reasoning without ToM requirements.
    """

    def __init__(
        self,
        task_id: int = 1,
        num_examples: int = 10000,
        max_length: int = 50,
        data_dir: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        self.task_id = task_id
        self.data_dir = Path(data_dir) if data_dir else Path("data/babi")
        super().__init__(num_examples, max_length, seed)

    def _generate_data(self):
        """Generate or load bAbI task data"""
        try:
            self._load_from_huggingface()
        except Exception:
            try:
                self._load_from_local()
            except Exception:
                self._generate_synthetic()

    def _load_from_huggingface(self):
        """Load from HuggingFace datasets"""
        from datasets import load_dataset
        data = load_dataset("facebook/babi_qa", f"en-10k-qa{self.task_id}", split='train')

        for item in data:
            story = ' '.join(item['story']['text'])
            question = item['question']
            answer = item['answer']

            self.examples.append(ControlExample(
                input_sequence=f"{story} {question}",
                target=answer,
                task_type=f'babi_task{self.task_id}',
                complexity=self._get_task_complexity(),
                metadata={'source': 'huggingface'}
            ))

            if len(self.examples) >= self.num_examples:
                break

    def _load_from_local(self):
        """Load from local files"""
        data_path = self.data_dir / f"qa{self.task_id}_train.txt"
        if not data_path.exists():
            raise FileNotFoundError(f"bAbI data not found at {data_path}")

        # Parse bAbI format
        current_story = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                num, content = line.split(' ', 1)
                num = int(num)

                if num == 1:
                    current_story = []

                if '\t' in content:
                    # Question line
                    parts = content.split('\t')
                    question = parts[0]
                    answer = parts[1]

                    story_text = ' '.join(current_story)
                    self.examples.append(ControlExample(
                        input_sequence=f"{story_text} {question}",
                        target=answer,
                        task_type=f'babi_task{self.task_id}',
                        complexity=self._get_task_complexity(),
                        metadata={'source': 'local'}
                    ))

                    if len(self.examples) >= self.num_examples:
                        return
                else:
                    current_story.append(content)

    def _generate_synthetic(self):
        """Generate synthetic bAbI-style examples"""
        generators = {
            1: self._generate_single_fact,
            2: self._generate_two_facts,
            3: self._generate_three_facts,
            4: self._generate_two_arg_relations,
            5: self._generate_three_arg_relations,
        }

        generator = generators.get(self.task_id, self._generate_single_fact)

        for _ in range(self.num_examples):
            self.examples.append(generator())

    def _get_task_complexity(self) -> str:
        """Get complexity level for task"""
        if self.task_id <= 2:
            return 'simple'
        elif self.task_id <= 4:
            return 'medium'
        else:
            return 'hard'

    def _generate_single_fact(self) -> ControlExample:
        """Task 1: Single supporting fact"""
        names = ['Mary', 'John', 'Sandra', 'Daniel']
        locations = ['garden', 'kitchen', 'hallway', 'bedroom', 'bathroom']

        name = np.random.choice(names)
        location = np.random.choice(locations)

        story = f"{name} went to the {location}."
        question = f"Where is {name}?"
        answer = location

        return ControlExample(
            input_sequence=f"{story} {question}",
            target=answer,
            task_type='babi_task1',
            complexity='simple',
            metadata={'source': 'synthetic'}
        )

    def _generate_two_facts(self) -> ControlExample:
        """Task 2: Two supporting facts"""
        names = ['Mary', 'John', 'Sandra', 'Daniel']
        objects = ['football', 'apple', 'milk']
        locations = ['garden', 'kitchen', 'hallway', 'bedroom']

        name = np.random.choice(names)
        obj = np.random.choice(objects)
        loc1 = np.random.choice(locations)
        loc2 = np.random.choice([l for l in locations if l != loc1])

        story = f"{name} picked up the {obj}. {name} went to the {loc2}."
        question = f"Where is the {obj}?"
        answer = loc2

        return ControlExample(
            input_sequence=f"{story} {question}",
            target=answer,
            task_type='babi_task2',
            complexity='simple',
            metadata={'source': 'synthetic'}
        )

    def _generate_three_facts(self) -> ControlExample:
        """Task 3: Three supporting facts"""
        names = ['Mary', 'John', 'Sandra']
        objects = ['football', 'apple']
        locations = ['garden', 'kitchen', 'hallway', 'bedroom']

        name = np.random.choice(names)
        obj = np.random.choice(objects)
        locs = np.random.choice(locations, size=3, replace=False)

        story = f"{name} went to the {locs[0]}. {name} picked up the {obj}. {name} went to the {locs[2]}."
        question = f"Where is the {obj}?"
        answer = locs[2]

        return ControlExample(
            input_sequence=f"{story} {question}",
            target=answer,
            task_type='babi_task3',
            complexity='medium',
            metadata={'source': 'synthetic'}
        )

    def _generate_two_arg_relations(self) -> ControlExample:
        """Task 4: Two argument relations"""
        names = ['Mary', 'John', 'Sandra', 'Daniel']
        relations = ['north', 'south', 'east', 'west']

        name1, name2 = np.random.choice(names, size=2, replace=False)
        relation = np.random.choice(relations)

        story = f"The {name1} is {relation} of the {name2}."
        question = f"What is {relation} of the {name2}?"
        answer = name1

        return ControlExample(
            input_sequence=f"{story} {question}",
            target=answer,
            task_type='babi_task4',
            complexity='medium',
            metadata={'source': 'synthetic'}
        )

    def _generate_three_arg_relations(self) -> ControlExample:
        """Task 5: Three argument relations"""
        people = ['Mary', 'John', 'Sandra', 'Daniel', 'Julie']
        objects = ['milk', 'football', 'apple']

        person = np.random.choice(people)
        giver = np.random.choice([p for p in people if p != person])
        obj = np.random.choice(objects)

        story = f"{giver} gave the {obj} to {person}."
        question = f"Who gave the {obj} to {person}?"
        answer = giver

        return ControlExample(
            input_sequence=f"{story} {question}",
            target=answer,
            task_type='babi_task5',
            complexity='medium',
            metadata={'source': 'synthetic'}
        )


class RelationalReasoningDataset(BaseControlDataset):
    """
    Relational reasoning tasks - Object relationships without ToM.

    Inspired by Sort-of-CLEVR but simplified for NAS experiments.
    Tests relational reasoning: same/different, count, compare, etc.
    """

    def __init__(
        self,
        num_examples: int = 10000,
        num_objects: int = 6,
        grid_size: int = 5,
        seed: Optional[int] = 42,
    ):
        self.num_objects = num_objects
        self.grid_size = grid_size
        super().__init__(num_examples, max_length=50, seed=seed)

    def _generate_data(self):
        """Generate relational reasoning examples"""
        question_types = ['closest', 'count_color', 'same_shape', 'furthest']
        examples_per_type = self.num_examples // len(question_types)

        for q_type in question_types:
            for _ in range(examples_per_type):
                example = self._generate_example(q_type)
                self.examples.append(example)

    def _generate_example(self, question_type: str) -> ControlExample:
        """Generate a single relational reasoning example"""
        # Generate random objects
        colors = ['red', 'green', 'blue', 'yellow', 'gray', 'cyan']
        shapes = ['circle', 'square', 'triangle']

        objects = []
        positions = set()

        for _ in range(self.num_objects):
            # Random unique position
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                if (x, y) not in positions:
                    positions.add((x, y))
                    break

            obj = {
                'color': np.random.choice(colors),
                'shape': np.random.choice(shapes),
                'x': x,
                'y': y,
            }
            objects.append(obj)

        # Create scene description
        scene_desc = self._describe_scene(objects)

        # Generate question and answer based on type
        if question_type == 'closest':
            question, answer = self._question_closest(objects)
        elif question_type == 'count_color':
            question, answer = self._question_count_color(objects)
        elif question_type == 'same_shape':
            question, answer = self._question_same_shape(objects)
        elif question_type == 'furthest':
            question, answer = self._question_furthest(objects)
        else:
            question, answer = self._question_closest(objects)

        return ControlExample(
            input_sequence=f"{scene_desc} {question}",
            target=answer,
            task_type=f'relational_{question_type}',
            complexity='medium',
            metadata={'objects': objects, 'question_type': question_type, 'source': 'synthetic'}
        )

    def _describe_scene(self, objects: List[Dict]) -> str:
        """Create text description of scene"""
        descriptions = []
        for i, obj in enumerate(objects):
            descriptions.append(
                f"Object {i+1} is a {obj['color']} {obj['shape']} at position ({obj['x']}, {obj['y']})."
            )
        return ' '.join(descriptions)

    def _question_closest(self, objects: List[Dict]) -> Tuple[str, str]:
        """Question about closest object"""
        ref_idx = np.random.randint(0, len(objects))
        ref_obj = objects[ref_idx]

        min_dist = float('inf')
        closest_idx = -1

        for i, obj in enumerate(objects):
            if i != ref_idx:
                dist = np.sqrt((obj['x'] - ref_obj['x'])**2 + (obj['y'] - ref_obj['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

        question = f"What is closest to the {ref_obj['color']} {ref_obj['shape']}?"
        answer = f"Object {closest_idx + 1}" if closest_idx >= 0 else "none"

        return question, answer

    def _question_furthest(self, objects: List[Dict]) -> Tuple[str, str]:
        """Question about furthest object"""
        ref_idx = np.random.randint(0, len(objects))
        ref_obj = objects[ref_idx]

        max_dist = -1
        furthest_idx = -1

        for i, obj in enumerate(objects):
            if i != ref_idx:
                dist = np.sqrt((obj['x'] - ref_obj['x'])**2 + (obj['y'] - ref_obj['y'])**2)
                if dist > max_dist:
                    max_dist = dist
                    furthest_idx = i

        question = f"What is furthest from the {ref_obj['color']} {ref_obj['shape']}?"
        answer = f"Object {furthest_idx + 1}" if furthest_idx >= 0 else "none"

        return question, answer

    def _question_count_color(self, objects: List[Dict]) -> Tuple[str, str]:
        """Question about counting objects of a color"""
        colors = [obj['color'] for obj in objects]
        target_color = np.random.choice(colors)
        count = sum(1 for c in colors if c == target_color)

        question = f"How many {target_color} objects are there?"
        answer = str(count)

        return question, answer

    def _question_same_shape(self, objects: List[Dict]) -> Tuple[str, str]:
        """Question about objects with same shape"""
        ref_idx = np.random.randint(0, len(objects))
        ref_obj = objects[ref_idx]

        same_shape = [i for i, obj in enumerate(objects)
                     if obj['shape'] == ref_obj['shape'] and i != ref_idx]

        question = f"Which objects have the same shape as the {ref_obj['color']} {ref_obj['shape']}?"
        if same_shape:
            answer = ', '.join([f"Object {i+1}" for i in same_shape])
        else:
            answer = "none"

        return question, answer


class ControlDatasetLoader:
    """Unified interface for loading control (non-ToM) datasets"""

    DATASETS = {
        'simple_sequence': SimpleSequenceDataset,
        'babi': BAbIDataset,
        'relational': RelationalReasoningDataset,
    }

    @classmethod
    def load_simple_sequence(
        cls,
        num_examples: int = 10000,
        pattern_types: Optional[List[str]] = None,
        seed: int = 42,
    ) -> SimpleSequenceDataset:
        """Load simple sequence prediction dataset"""
        return SimpleSequenceDataset(
            num_examples=num_examples,
            pattern_types=pattern_types,
            seed=seed,
        )

    @classmethod
    def load_babi(
        cls,
        task_id: int = 1,
        num_examples: int = 10000,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ) -> BAbIDataset:
        """Load bAbI task dataset"""
        return BAbIDataset(
            task_id=task_id,
            num_examples=num_examples,
            data_dir=data_dir,
            seed=seed,
        )

    @classmethod
    def load_relational(
        cls,
        num_examples: int = 10000,
        num_objects: int = 6,
        seed: int = 42,
    ) -> RelationalReasoningDataset:
        """Load relational reasoning dataset"""
        return RelationalReasoningDataset(
            num_examples=num_examples,
            num_objects=num_objects,
            seed=seed,
        )

    @classmethod
    def load_all(
        cls,
        num_examples_per_dataset: int = 5000,
        seed: int = 42,
    ) -> Dict[str, BaseControlDataset]:
        """Load all control datasets"""
        datasets = {
            'simple_sequence': cls.load_simple_sequence(num_examples_per_dataset, seed=seed),
            'babi_1': cls.load_babi(1, num_examples_per_dataset, seed=seed),
            'babi_2': cls.load_babi(2, num_examples_per_dataset, seed=seed),
            'babi_3': cls.load_babi(3, num_examples_per_dataset, seed=seed),
            'relational': cls.load_relational(num_examples_per_dataset, seed=seed),
        }
        return datasets


def get_control_task_complexity(task_name: str) -> Dict[str, Any]:
    """Get complexity information for a control task"""
    complexity_map = {
        'simple_sequence': {
            'cognitive_demand': 'simple',
            'expected_architecture': 'feedforward, minimal depth',
            'tom_order': 0,
        },
        'babi_1': {
            'cognitive_demand': 'simple',
            'expected_architecture': 'minimal',
            'tom_order': 0,
        },
        'babi_2': {
            'cognitive_demand': 'simple',
            'expected_architecture': 'minimal + memory',
            'tom_order': 0,
        },
        'babi_3': {
            'cognitive_demand': 'medium',
            'expected_architecture': 'skip connections possible',
            'tom_order': 0,
        },
        'babi_4': {
            'cognitive_demand': 'medium',
            'expected_architecture': 'relational',
            'tom_order': 0,
        },
        'babi_5': {
            'cognitive_demand': 'medium',
            'expected_architecture': 'relational + memory',
            'tom_order': 0,
        },
        'relational': {
            'cognitive_demand': 'medium',
            'expected_architecture': 'some skip connections',
            'tom_order': 0,
        },
    }

    return complexity_map.get(task_name, {
        'cognitive_demand': 'unknown',
        'expected_architecture': 'unknown',
        'tom_order': 0,
    })
