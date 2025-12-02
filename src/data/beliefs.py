"""
Compute ground truth beliefs from observations.

This module provides the BeliefComputer class which computes what agents
should believe based on what they observed. This is used to generate
ground truth labels for training and evaluation.
"""

from typing import Dict, Optional, List, Set
from .events import Event, Scenario


class BeliefComputer:
    """
    Compute agent beliefs from their observations.

    The key insight: an agent's belief about object locations is based
    ONLY on events they observed. If they didn't see an object being moved,
    they still believe it's in the old location (false belief).
    """

    def compute_object_belief(self, agent: str, obj: str,
                              events: List[Event]) -> Optional[str]:
        """
        What does agent believe about object's location?

        Traces through events the agent observed to find their
        current belief about where the object is.

        Args:
            agent: The agent whose belief we're computing
            obj: The object we're asking about
            events: All events (we filter to observed ones)

        Returns:
            Location the agent believes the object is in, or None if unknown
        """
        # Filter to events this agent observed
        observed = [e for e in events if agent in e.observed_by]

        location = None
        for event in observed:
            if event.object == obj:
                if event.action in ('put', 'move'):
                    location = event.target_location
                elif event.action == 'take':
                    location = None  # Object is being held/moved

        return location

    def compute_agent_presence(self, agent: str,
                               events: List[Event]) -> bool:
        """
        Is the agent currently present (in the room/scene)?

        Tracks enter/leave events to determine presence.
        """
        # Filter to events involving this agent
        present = False
        for event in events:
            if event.actor == agent:
                if event.action == 'enter':
                    present = True
                elif event.action == 'leave':
                    present = False
        return present

    def compute_who_was_present(self, events: List[Event],
                                 up_to_timestamp: int) -> Set[str]:
        """
        Which agents were present at a given timestamp?

        Useful for computing who observed subsequent events.
        """
        present = set()
        for event in events:
            if event.timestamp > up_to_timestamp:
                break
            if event.action == 'enter':
                present.add(event.actor)
            elif event.action == 'leave':
                present.discard(event.actor)
        return present

    def compute_ground_truth(self, scenario: Scenario) -> str:
        """
        Compute correct answer based on question type.

        Args:
            scenario: The complete scenario with question

        Returns:
            The ground truth answer as a string (typically a location)
        """
        if scenario.question_type == 'reality':
            # What is the actual state?
            return self._compute_reality(scenario)

        elif scenario.question_type == 'first_order_belief':
            # What does agent believe?
            agent = scenario.question_target_agent
            obj = scenario.question_target_object
            observed = scenario.get_agent_observations(agent)
            return self.compute_object_belief(agent, obj, observed)

        elif scenario.question_type == 'second_order_belief':
            # What does agent1 think agent2 believes?
            return self._compute_second_order(scenario)

        return None

    def _compute_reality(self, scenario: Scenario) -> str:
        """
        Compute actual object location from all events.

        Reality considers ALL events, not just observed ones.
        """
        obj = scenario.question_target_object
        location = None
        for event in scenario.events:
            if event.object == obj and event.action in ('put', 'move'):
                location = event.target_location
        return location

    def _compute_second_order(self, scenario: Scenario) -> str:
        """
        Agent1's belief about agent2's belief.

        This requires modeling:
        1. What agent1 knows about agent2's observations
        2. Based on that, what agent1 thinks agent2 believes

        Key insight: agent1 can only model agent2's beliefs based on
        what agent1 KNOWS agent2 observed (which depends on agent1's
        own observations of agent2's presence).
        """
        agent1 = scenario.question_target_agent  # The one we're asking
        agent2 = scenario.question_about_agent    # Whose belief agent1 is modeling
        obj = scenario.question_target_object

        # What did agent1 observe?
        agent1_observations = scenario.get_agent_observations(agent1)

        # From agent1's perspective, when was agent2 present?
        # (agent1 can only know about agent2's presence if agent1 observed it)
        agent2_present_according_to_agent1 = set()
        for event in agent1_observations:
            if event.actor == agent2:
                if event.action == 'enter':
                    agent2_present_according_to_agent1.add(event.timestamp)
                elif event.action == 'leave':
                    # Mark end of presence
                    pass

        # Events agent1 thinks agent2 observed
        agent2_believed_observations = []
        agent2_present = False

        for event in agent1_observations:
            if event.actor == agent2:
                if event.action == 'enter':
                    agent2_present = True
                elif event.action == 'leave':
                    agent2_present = False

            if agent2_present or event.actor == agent2:
                # Agent1 believes agent2 saw this event
                agent2_believed_observations.append(event)

        # Now compute what agent1 THINKS agent2 believes about the object
        return self.compute_object_belief(agent2, obj, agent2_believed_observations)

    def validate_scenario(self, scenario: Scenario) -> Dict[str, bool]:
        """
        Validate that a scenario is well-formed for ToM testing.

        Returns dict with validation results:
        - has_false_belief: Does scenario create false beliefs?
        - reality_differs_from_belief: Is reality different from some agent's belief?
        - has_target_agent: Is target agent specified for belief questions?
        - has_target_object: Is target object specified?
        """
        results = {
            'has_false_belief': False,
            'reality_differs_from_belief': False,
            'has_target_agent': scenario.question_target_agent is not None,
            'has_target_object': scenario.question_target_object is not None,
        }

        if scenario.question_target_object:
            obj = scenario.question_target_object
            reality = self._compute_reality(scenario)

            # Check each agent's belief
            for agent in scenario.get_all_agents():
                observed = scenario.get_agent_observations(agent)
                belief = self.compute_object_belief(agent, obj, observed)

                if belief is not None and belief != reality:
                    results['has_false_belief'] = True
                    results['reality_differs_from_belief'] = True
                    break

        return results
