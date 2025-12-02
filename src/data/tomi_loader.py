"""
Load and parse ToMi benchmark into our Event format.

This module provides synthetic Sally-Anne scenario generation and
parsing of the ToMi (Theory of Mind) benchmark dataset.
"""

import json
import re
import random
from typing import List, Tuple, Optional
from pathlib import Path
from .events import Event, Scenario
from .beliefs import BeliefComputer


class ToMiLoader:
    """
    Load ToMi benchmark and convert to our format.

    Can either load real ToMi data or generate synthetic Sally-Anne
    style scenarios for testing and development.
    """

    LOCATIONS = ['basket', 'box', 'container', 'drawer', 'cupboard',
                 'kitchen', 'bathroom', 'bedroom', 'garden', 'pantry']
    OBJECTS = ['marble', 'ball', 'apple', 'orange', 'book', 'keys',
               'toy', 'phone', 'hat', 'bag']
    AGENTS = ['Sally', 'Anne', 'Emma', 'Jack', 'Mark', 'Lisa',
              'Tom', 'Kate', 'Ben', 'Lily']

    def __init__(self, data_path: str = None, seed: int = 42):
        """
        Initialize loader.

        Args:
            data_path: Path to ToMi dataset (if None, generates synthetic)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.scenarios = []
        self.belief_computer = BeliefComputer()
        random.seed(seed)

    def load(self, num_samples: int = 1000) -> List[Scenario]:
        """
        Load ToMi data or generate synthetic examples.

        Args:
            num_samples: Number of scenarios to generate

        Returns:
            List of Scenario objects
        """
        # Try to load real data first
        if self.data_path and Path(self.data_path).exists():
            return self._load_real_tomi()
        else:
            # Generate synthetic Sally-Anne scenarios
            return self._generate_synthetic(num_samples)

    def _generate_synthetic(self, n: int) -> List[Scenario]:
        """
        Generate Sally-Anne style scenarios.

        Creates scenarios with information asymmetry where one agent
        leaves before an object is moved, creating a false belief.
        """
        scenarios = []

        for i in range(n):
            # Choose scenario type
            scenario_type = random.choice([
                'basic_sally_anne',
                'multi_move',
                'three_agents',
                'concurrent_presence'
            ])

            if scenario_type == 'basic_sally_anne':
                scenario = self._generate_basic_sally_anne()
            elif scenario_type == 'multi_move':
                scenario = self._generate_multi_move()
            elif scenario_type == 'three_agents':
                scenario = self._generate_three_agents()
            else:
                scenario = self._generate_concurrent_presence()

            scenarios.append(scenario)

        return scenarios

    def _generate_basic_sally_anne(self) -> Scenario:
        """
        Generate classic Sally-Anne scenario.

        Story: Sally puts marble in basket, leaves, Anne moves it to box.
        Question: Where does Sally think the marble is?
        Answer: basket (she didn't see the move)
        """
        # Random elements
        agent1, agent2 = random.sample(self.AGENTS, 2)
        obj = random.choice(self.OBJECTS)
        loc1, loc2 = random.sample(self.LOCATIONS, 2)

        # Create events
        events = [
            Event(0, agent1, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            Event(1, agent2, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            Event(2, agent1, 'put', target_location=loc1, object=obj,
                  observed_by={agent1, agent2}),
            Event(3, agent1, 'leave', source_location='room',
                  observed_by={agent2}),  # agent1 leaves, doesn't see next
            Event(4, agent2, 'move', source_location=loc1,
                  target_location=loc2, object=obj,
                  observed_by={agent2}),  # Only agent2 sees this!
            Event(5, agent1, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
        ]

        # Randomly choose question type (weighted toward ToM)
        q_type = random.choices(
            ['reality', 'first_order_belief'],
            weights=[0.3, 0.7]
        )[0]

        scenario = Scenario(
            events=events,
            question_type=q_type,
            question_target_agent=agent1 if q_type != 'reality' else None,
            question_target_object=obj,
            difficulty='easy'
        )

        # Compute ground truth
        if q_type == 'reality':
            scenario.ground_truth_answer = loc2  # Where it actually is
        else:
            scenario.ground_truth_answer = loc1  # Where agent1 BELIEVES it is

        return scenario

    def _generate_multi_move(self) -> Scenario:
        """
        Generate scenario with multiple object moves.

        Object is moved multiple times, agent sees some but not all.
        """
        agent1, agent2 = random.sample(self.AGENTS, 2)
        obj = random.choice(self.OBJECTS)
        loc1, loc2, loc3 = random.sample(self.LOCATIONS, 3)

        events = [
            # Both enter
            Event(0, agent1, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            Event(1, agent2, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            # Initial placement (both see)
            Event(2, agent1, 'put', target_location=loc1, object=obj,
                  observed_by={agent1, agent2}),
            # Agent1 leaves
            Event(3, agent1, 'leave', source_location='room',
                  observed_by={agent2}),
            # First move (only agent2 sees)
            Event(4, agent2, 'move', source_location=loc1,
                  target_location=loc2, object=obj,
                  observed_by={agent2}),
            # Agent1 returns
            Event(5, agent1, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            # Second move (both see)
            Event(6, agent2, 'move', source_location=loc2,
                  target_location=loc3, object=obj,
                  observed_by={agent1, agent2}),
        ]

        # Question types
        q_type = random.choice(['reality', 'first_order_belief'])

        scenario = Scenario(
            events=events,
            question_type=q_type,
            question_target_agent=agent1 if q_type != 'reality' else None,
            question_target_object=obj,
            difficulty='medium'
        )

        if q_type == 'reality':
            scenario.ground_truth_answer = loc3  # Final location
        else:
            # Agent1 saw: put in loc1, then move to loc3
            # But NOT the intermediate move to loc2
            scenario.ground_truth_answer = loc3  # They saw the final move

        return scenario

    def _generate_three_agents(self) -> Scenario:
        """
        Generate scenario with three agents.

        More complex information asymmetry with multiple agents.
        """
        agents = random.sample(self.AGENTS, 3)
        agent1, agent2, agent3 = agents
        obj = random.choice(self.OBJECTS)
        loc1, loc2, loc3 = random.sample(self.LOCATIONS, 3)

        events = [
            # All enter
            Event(0, agent1, 'enter', target_location='room',
                  observed_by=set(agents)),
            Event(1, agent2, 'enter', target_location='room',
                  observed_by=set(agents)),
            Event(2, agent3, 'enter', target_location='room',
                  observed_by=set(agents)),
            # Initial placement
            Event(3, agent1, 'put', target_location=loc1, object=obj,
                  observed_by=set(agents)),
            # Agent1 leaves
            Event(4, agent1, 'leave', source_location='room',
                  observed_by={agent2, agent3}),
            # Agent2 moves (only agent3 sees)
            Event(5, agent3, 'leave', source_location='room',
                  observed_by={agent2}),
            Event(6, agent2, 'move', source_location=loc1,
                  target_location=loc2, object=obj,
                  observed_by={agent2}),
            # Both return
            Event(7, agent1, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            Event(8, agent3, 'enter', target_location='room',
                  observed_by={agent1, agent2, agent3}),
        ]

        # Choose which agent to ask about
        q_agent = random.choice([agent1, agent3])
        q_type = random.choice(['reality', 'first_order_belief'])

        scenario = Scenario(
            events=events,
            question_type=q_type,
            question_target_agent=q_agent if q_type != 'reality' else None,
            question_target_object=obj,
            difficulty='hard'
        )

        if q_type == 'reality':
            scenario.ground_truth_answer = loc2
        elif q_agent == agent1:
            scenario.ground_truth_answer = loc1  # Never saw any move
        else:  # agent3
            scenario.ground_truth_answer = loc1  # Left before the move

        return scenario

    def _generate_concurrent_presence(self) -> Scenario:
        """
        Generate scenario with overlapping presence.

        Agents enter and leave at different times.
        """
        agent1, agent2 = random.sample(self.AGENTS, 2)
        obj = random.choice(self.OBJECTS)
        loc1, loc2 = random.sample(self.LOCATIONS, 2)

        events = [
            # Agent1 enters alone
            Event(0, agent1, 'enter', target_location='room',
                  observed_by={agent1}),
            # Puts object
            Event(1, agent1, 'put', target_location=loc1, object=obj,
                  observed_by={agent1}),
            # Agent2 enters
            Event(2, agent2, 'enter', target_location='room',
                  observed_by={agent1, agent2}),
            # Agent1 leaves
            Event(3, agent1, 'leave', source_location='room',
                  observed_by={agent2}),
            # Agent2 moves object
            Event(4, agent2, 'move', source_location=loc1,
                  target_location=loc2, object=obj,
                  observed_by={agent2}),
        ]

        q_type = random.choice(['reality', 'first_order_belief'])

        scenario = Scenario(
            events=events,
            question_type=q_type,
            question_target_agent=agent1 if q_type != 'reality' else None,
            question_target_object=obj,
            difficulty='medium'
        )

        if q_type == 'reality':
            scenario.ground_truth_answer = loc2
        else:
            scenario.ground_truth_answer = loc1  # Agent1 never saw the move

        return scenario

    def _load_real_tomi(self) -> List[Scenario]:
        """Load real ToMi dataset from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        scenarios = []
        for item in data:
            if 'story' in item:
                events = self._parse_tomi_story(item['story'])
                scenario = Scenario(
                    events=events,
                    question_type=self._map_question_type(item.get('question_type', '')),
                    question_target_agent=item.get('target_agent'),
                    question_target_object=item.get('target_object'),
                    ground_truth_answer=item.get('answer'),
                )
                scenarios.append(scenario)

        return scenarios

    def _parse_tomi_story(self, story: str) -> List[Event]:
        """
        Parse ToMi natural language story into events.

        Handles patterns like:
        - "Sally entered the kitchen."
        - "Sally put the marble in the basket."
        - "Sally exited the kitchen."
        """
        events = []
        timestamp = 0
        present_agents = set()

        sentences = story.split('.')
        for sent in sentences:
            sent = sent.strip().lower()
            if not sent:
                continue

            # Parse different sentence types
            event = self._parse_sentence(sent, timestamp, present_agents)
            if event:
                events.append(event)
                timestamp += 1

                # Track who's present
                if event.action == 'enter':
                    present_agents.add(event.actor)
                elif event.action == 'leave':
                    present_agents.discard(event.actor)

        return events

    def _parse_sentence(self, sent: str, ts: int,
                        present: set) -> Optional[Event]:
        """Parse single sentence into Event."""
        # Entry pattern
        enter_match = re.match(r'(\w+) entered the (\w+)', sent)
        if enter_match:
            actor = enter_match.group(1).capitalize()
            loc = enter_match.group(2)
            return Event(ts, actor, 'enter', target_location=loc,
                        observed_by=present | {actor})

        # Exit pattern
        exit_match = re.match(r'(\w+) exited the (\w+)', sent)
        if exit_match:
            actor = exit_match.group(1).capitalize()
            loc = exit_match.group(2)
            return Event(ts, actor, 'leave', source_location=loc,
                        observed_by=present)

        # Left pattern (alternative)
        left_match = re.match(r'(\w+) left the (\w+)', sent)
        if left_match:
            actor = left_match.group(1).capitalize()
            loc = left_match.group(2)
            return Event(ts, actor, 'leave', source_location=loc,
                        observed_by=present)

        # Put pattern
        put_match = re.match(r'(\w+) put the (\w+) in the (\w+)', sent)
        if put_match:
            actor = put_match.group(1).capitalize()
            obj = put_match.group(2)
            loc = put_match.group(3)
            return Event(ts, actor, 'put', object=obj, target_location=loc,
                        observed_by=present)

        # Move pattern
        move_match = re.match(
            r'(\w+) moved the (\w+) from the (\w+) to the (\w+)', sent)
        if move_match:
            actor = move_match.group(1).capitalize()
            obj = move_match.group(2)
            src = move_match.group(3)
            dst = move_match.group(4)
            return Event(ts, actor, 'move', object=obj,
                        source_location=src, target_location=dst,
                        observed_by=present)

        # Simple move pattern (without 'from')
        simple_move = re.match(r'(\w+) moved the (\w+) to the (\w+)', sent)
        if simple_move:
            actor = simple_move.group(1).capitalize()
            obj = simple_move.group(2)
            dst = simple_move.group(3)
            return Event(ts, actor, 'move', object=obj, target_location=dst,
                        observed_by=present)

        return None

    def _map_question_type(self, tomi_type: str) -> str:
        """Map ToMi question types to our format."""
        mapping = {
            'reality': 'reality',
            'memory': 'reality',
            'first_order': 'first_order_belief',
            'first-order': 'first_order_belief',
            'second_order': 'second_order_belief',
            'second-order': 'second_order_belief',
        }
        return mapping.get(tomi_type.lower(), 'first_order_belief')

    def get_balanced_split(self, scenarios: List[Scenario],
                           train_ratio: float = 0.8) -> Tuple[List[Scenario], List[Scenario]]:
        """
        Split scenarios into train/test with balanced question types.

        Ensures both sets have similar proportions of reality vs belief questions.
        """
        by_type = {}
        for s in scenarios:
            t = s.question_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)

        train, test = [], []
        for q_type, items in by_type.items():
            random.shuffle(items)
            split_idx = int(len(items) * train_ratio)
            train.extend(items[:split_idx])
            test.extend(items[split_idx:])

        random.shuffle(train)
        random.shuffle(test)

        return train, test
