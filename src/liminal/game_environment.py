"""
Liminal Game Environment - Main Environment Class

This is the core environment that ties together all systems for
NAS agent training and evaluation. It provides:

1. World state management (realms, NPCs, player)
2. Step-based simulation for reinforcement learning
3. Observation encoding for neural network input
4. Action interpretation from neural network output
5. Reward calculation for ToM-based objectives

The environment follows a gym-like interface for compatibility with
standard RL training frameworks.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import random
import json

from .soul_map import SoulMap, SoulMapDelta, REALM_DIMENSIONS
from .realms import Realm, RealmType, REALMS, get_realm, RealmTransition, RealmLocation
from .npcs.base_npc import BaseNPC, NPCState, NPCBehavior
from .npcs.heroes import HERO_NPCS, create_hero_npc
from .npcs.archetypes import ARCHETYPES, create_archetype_npc, populate_realm
from .mechanics.soul_scanner import SoulScanner, AnalysisResult, AnalysisDepth
from .mechanics.cognitive_hazards import CognitiveHazard, HAZARD_REGISTRY, apply_hazard, HazardCategory
from .mechanics.ontological_instability import OntologicalInstability, InstabilityLevel, calculate_hazard_instability


class ActionType(Enum):
    """Types of actions agents can take."""

    MOVE = "move"
    ANALYZE = "analyze"
    INTERVENE = "intervene"
    INTERACT = "interact"
    WAIT = "wait"
    PREDICT = "predict"


@dataclass
class GameState:
    """Complete state of the game at a given tick."""

    tick: int
    current_realm: RealmType
    player_position: Tuple[float, float]
    player_tom_depth: int
    player_energy: float
    instability: float
    instability_level: InstabilityLevel
    nearby_npcs: List[str]  # NPC IDs
    targeted_npc: Optional[str]
    active_quests: List[str]
    realm_state: Dict[str, Any]


@dataclass
class Observation:
    """Observation returned to agents."""

    # Player state
    player_soul_map: torch.Tensor  # 65 dims
    player_context: torch.Tensor  # Additional context

    # Environment state
    realm_features: torch.Tensor  # Realm-specific features
    instability_state: torch.Tensor

    # Social state
    nearby_npc_states: List[torch.Tensor]  # List of NPC soul maps
    npc_relationships: torch.Tensor  # Relationship matrix

    # Full tensor for model input
    full_tensor: torch.Tensor


@dataclass
class StepResult:
    """Result of taking an action in the environment."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
    game_state: GameState


class LiminalEnvironment:
    """
    The main game environment for Liminal Architectures.

    This environment simulates the psychological open-world game,
    providing a training ground for NAS agents to develop Theory of Mind.
    """

    def __init__(
        self,
        population_size: int = 200,
        include_heroes: bool = True,
        starting_realm: RealmType = RealmType.PEREGRINE,
        max_episode_length: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the game environment.

        Args:
            population_size: Total NPC population across all realms
            include_heroes: Whether to include hero NPCs
            starting_realm: Starting realm for player
            max_episode_length: Maximum ticks per episode
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.population_size = population_size
        self.include_heroes = include_heroes
        self.starting_realm = starting_realm
        self.max_episode_length = max_episode_length

        # Core systems
        self.instability = OntologicalInstability()
        self.soul_scanner = SoulScanner(player_tom_depth=3)

        # World state
        self.tick = 0
        self.npcs: Dict[str, BaseNPC] = {}
        self.player_position = (0.0, 0.0)
        self.player_soul_map = SoulMap()  # Player has a soul map too
        self.current_realm = starting_realm

        # Episode state
        self.episode_rewards: List[float] = []
        self.predictions_made: List[Dict[str, Any]] = []
        self.interventions_made: List[Dict[str, Any]] = []

        # Observation dimensions
        self.soul_map_dim = 65  # 60 soul map + 5 realm modifiers
        self.context_dim = 20
        self.max_nearby_npcs = 10

        # Total observation dimension
        self.observation_dim = (
            self.soul_map_dim  # Player soul map
            + self.context_dim  # Player context
            + 10  # Realm features
            + 5  # Instability state
            + self.max_nearby_npcs * self.soul_map_dim  # Nearby NPCs
            + self.max_nearby_npcs  # Relationships
        )

        # Initialize world
        self._initialize_world()

    def _initialize_world(self) -> None:
        """Initialize the game world with NPCs."""
        self.npcs = {}

        # Add hero NPCs if enabled
        if self.include_heroes:
            for hero_id in HERO_NPCS:
                npc = create_hero_npc(hero_id)
                self.npcs[npc.npc_id] = npc

        # Populate each realm
        npcs_per_realm = (self.population_size - len(self.npcs)) // len(RealmType)
        for realm_type in RealmType:
            if realm_type == RealmType.THE_NOTHING:
                # Fewer NPCs in The Nothing
                count = npcs_per_realm // 3
            else:
                count = npcs_per_realm

            realm_npcs = populate_realm(realm_type, count)
            for npc in realm_npcs:
                self.npcs[npc.npc_id] = npc

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment for a new episode.

        Returns:
            Initial observation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset tick counter
        self.tick = 0

        # Reset player state
        self.player_position = (0.0, 0.0)
        self.player_soul_map = SoulMap()
        self.current_realm = self.starting_realm

        # Reset instability
        self.instability = OntologicalInstability()

        # Reset episode tracking
        self.episode_rewards = []
        self.predictions_made = []
        self.interventions_made = []

        # Reinitialize NPCs (or just reset their states)
        self._initialize_world()

        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: Dict with 'type' and action-specific parameters

        Returns:
            StepResult with observation, reward, done, info
        """
        self.tick += 1

        # Process action
        action_reward, action_info = self._process_action(action)

        # Update world state
        self._update_world()

        # Process instability
        instability_result = self.instability.tick()

        # Calculate total reward
        reward = self._calculate_reward(action_reward, action_info)
        self.episode_rewards.append(reward)

        # Check if episode is done
        done = self._check_done()

        # Build observation
        observation = self._get_observation()

        # Build game state
        game_state = self._get_game_state()

        # Build info dict
        info = {
            "tick": self.tick,
            "action_info": action_info,
            "instability": instability_result,
            "cumulative_reward": sum(self.episode_rewards),
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            game_state=game_state,
        )

    def _process_action(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process the player's action."""
        action_type = action.get("type", ActionType.WAIT)
        if isinstance(action_type, str):
            action_type = ActionType(action_type)

        reward = 0.0
        info = {"action_type": action_type.value}

        if action_type == ActionType.MOVE:
            reward, info = self._process_move(action)

        elif action_type == ActionType.ANALYZE:
            reward, info = self._process_analyze(action)

        elif action_type == ActionType.INTERVENE:
            reward, info = self._process_intervene(action)

        elif action_type == ActionType.PREDICT:
            reward, info = self._process_predict(action)

        elif action_type == ActionType.INTERACT:
            reward, info = self._process_interact(action)

        return reward, info

    def _process_move(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process movement action."""
        target = action.get("target", (0, 0))
        self.player_position = target

        # Check for realm transition
        new_realm = action.get("new_realm")
        if new_realm and new_realm != self.current_realm:
            can_transition, reason = RealmTransition.can_transition(
                get_realm(self.current_realm), get_realm(new_realm), self.player_soul_map
            )
            if can_transition:
                RealmTransition.apply_transition_effects(
                    get_realm(self.current_realm), get_realm(new_realm), self.player_soul_map
                )
                self.current_realm = new_realm

        return 0.0, {"new_position": target, "realm": self.current_realm.value}

    def _process_analyze(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process analyze action (Soul Scanner)."""
        target_id = action.get("target_id")
        depth = action.get("depth", AnalysisDepth.MODERATE)

        if target_id not in self.npcs:
            return -0.1, {"error": "Invalid target"}

        target_npc = self.npcs[target_id]

        # Perform analysis
        if depth == AnalysisDepth.PREDICTIVE:
            result = self.soul_scanner.predictive_scan(target_npc, self._get_context_for_npc(target_npc))
        elif depth == AnalysisDepth.DEEP:
            result = self.soul_scanner.deep_scan(target_npc)
        elif depth == AnalysisDepth.MODERATE:
            result = self.soul_scanner.moderate_scan(target_npc)
        else:
            result = self.soul_scanner.shallow_scan(target_npc)

        # Reward for successful analysis
        # Map depth to numeric value (enum values are strings)
        depth_values = {
            AnalysisDepth.PASSIVE: 0,
            AnalysisDepth.SHALLOW: 1,
            AnalysisDepth.MODERATE: 2,
            AnalysisDepth.DEEP: 3,
            AnalysisDepth.PREDICTIVE: 4,
        }
        depth_num = depth_values.get(depth, 0) if isinstance(depth, AnalysisDepth) else 0
        reward = 0.1 + depth_num * 0.05

        return reward, {
            "analysis_result": result,
            "target_name": target_npc.name,
        }

    def _process_intervene(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process intervention action (cognitive hazard)."""
        target_id = action.get("target_id")
        hazard_name = action.get("hazard", "doubt")
        intensity = action.get("intensity", 1.0)

        if target_id not in self.npcs:
            return -0.1, {"error": "Invalid target"}

        target_npc = self.npcs[target_id]

        # Apply hazard
        success, result = apply_hazard(
            hazard_name, target_npc, player_tom=self.soul_scanner.player_tom_depth, intensity_modifier=intensity
        )

        if not success:
            return -0.1, {"error": result.get("error", "Unknown error")}

        # Add instability
        hazard = HAZARD_REGISTRY.get(hazard_name)
        if hazard:
            instability_amount = calculate_hazard_instability(hazard.category.value)
            self.instability.add_instability(
                instability_amount, f"hazard:{hazard_name}", f"Applied {hazard_name} to {target_npc.name}", self.tick
            )

        # Record intervention
        self.interventions_made.append(
            {
                "tick": self.tick,
                "target": target_id,
                "hazard": hazard_name,
                "result": result,
            }
        )

        # Reward based on outcome
        reward = 0.0
        if result.get("stability_change", 0) < -0.1:
            reward = 0.2  # Successfully destabilized
        elif result.get("stability_change", 0) > 0.1:
            reward = 0.1  # Successfully stabilized

        return reward, result

    def _process_predict(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Process prediction action.

        This is the core ToM training mechanism - the agent predicts
        what an NPC will do, and we validate against actual behavior.
        """
        target_id = action.get("target_id")
        prediction = action.get("prediction")  # What the agent thinks will happen

        if target_id not in self.npcs:
            return -0.1, {"error": "Invalid target"}

        target_npc = self.npcs[target_id]

        # Get NPC's actual decision
        context = self._get_context_for_npc(target_npc)
        actual_action = target_npc.decide_action(context)

        # Compare prediction to actual
        prediction_accuracy = self._compare_prediction(prediction, actual_action)

        # Record prediction
        self.predictions_made.append(
            {
                "tick": self.tick,
                "target": target_id,
                "prediction": prediction,
                "actual": actual_action,
                "accuracy": prediction_accuracy,
            }
        )

        # Reward based on accuracy
        reward = prediction_accuracy * 0.5  # Up to 0.5 reward for perfect prediction

        return reward, {
            "prediction": prediction,
            "actual": actual_action,
            "accuracy": prediction_accuracy,
        }

    def _process_interact(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Process interaction action (dialogue, quest)."""
        target_id = action.get("target_id")
        interaction_type = action.get("interaction_type", "talk")

        if target_id not in self.npcs:
            return -0.1, {"error": "Invalid target"}

        target_npc = self.npcs[target_id]

        # Update NPC awareness
        target_npc.awareness_of_player = min(1.0, target_npc.awareness_of_player + 0.3)

        # Simple interaction reward
        reward = 0.05

        return reward, {
            "target": target_id,
            "interaction_type": interaction_type,
            "npc_response": target_npc.emotional_state,
        }

    def _update_world(self) -> None:
        """Update all NPCs and world state."""
        current_realm = get_realm(self.current_realm)

        for npc_id, npc in self.npcs.items():
            # Apply realm effects if NPC is in current realm
            if npc.current_realm == self.current_realm:
                npc.update_from_realm(current_realm, duration=1.0)

            # NPC perceives environment
            observation = self._generate_npc_observation(npc)
            npc.perceive(observation, self.tick)

            # NPC decides action (if they have a NAS model attached)
            context = self._get_context_for_npc(npc)
            action = npc.decide_action(context)

            # Update NPC state based on action
            self._apply_npc_action(npc, action)

    def _generate_npc_observation(self, npc: BaseNPC) -> Dict[str, Any]:
        """Generate what an NPC perceives."""
        observation = {
            "tick": self.tick,
            "realm": self.current_realm.value,
        }

        # Add player info if NPC is aware
        if npc.awareness_of_player > 0.1:
            distance = self._calculate_distance(npc.position, self.player_position)
            observation["player"] = {
                "distance": distance,
                "awareness": npc.awareness_of_player,
            }

        return observation

    def _get_context_for_npc(self, npc: BaseNPC) -> Dict[str, Any]:
        """Get context for NPC decision making."""
        return {
            "threat_level": self._calculate_threat_for_npc(npc),
            "social_density": self._calculate_social_density(npc.position),
            "realm_vibe_match": get_realm(self.current_realm).get_vibe_compatibility(npc.soul_map),
            "time_of_day": (self.tick % 1000) / 1000,
            "player_nearby": self._calculate_distance(npc.position, self.player_position) < 30,
        }

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _calculate_threat_for_npc(self, npc: BaseNPC) -> float:
        """Calculate perceived threat level for an NPC."""
        base_threat = self.instability.instability / 100
        player_threat = 0.2 * npc.awareness_of_player
        return min(1.0, base_threat + player_threat)

    def _calculate_social_density(self, position: Tuple[float, float]) -> float:
        """Calculate NPC density around a position."""
        nearby = sum(1 for npc in self.npcs.values() if self._calculate_distance(npc.position, position) < 20)
        return min(1.0, nearby / 20)

    def _apply_npc_action(self, npc: BaseNPC, action: Dict[str, Any]) -> None:
        """Apply an NPC's decided action."""
        action_type = action.get("action_type", "continue")

        if action_type == "flee":
            npc.current_state = NPCState.FLEEING
            npc.emotional_state = "frightened"
        elif action_type == "avoid":
            npc.current_behavior = NPCBehavior.AVOID
        elif action_type == "approach":
            npc.current_behavior = NPCBehavior.APPROACH
        elif action_type == "investigate":
            npc.current_state = NPCState.INVESTIGATING
            npc.emotional_state = "curious"

    def _compare_prediction(self, prediction: Any, actual: Dict[str, Any]) -> float:
        """Compare prediction to actual NPC action."""
        if prediction is None:
            return 0.0

        if isinstance(prediction, dict):
            pred_action = prediction.get("action_type", "")
        else:
            pred_action = str(prediction)

        actual_action = actual.get("action_type", "")

        # Exact match
        if pred_action == actual_action:
            return 1.0

        # Partial match (similar actions)
        similar_groups = [
            {"flee", "avoid"},
            {"approach", "interact", "investigate"},
            {"continue", "observe"},
        ]

        for group in similar_groups:
            if pred_action in group and actual_action in group:
                return 0.5

        return 0.0

    def _calculate_reward(self, action_reward: float, action_info: Dict[str, Any]) -> float:
        """Calculate total reward for this step."""
        reward = action_reward

        # Penalty for high instability
        instability_penalty = -0.01 * (self.instability.instability / 100)
        reward += instability_penalty

        # Bonus for maintaining stability
        if self.instability.current_level == InstabilityLevel.STABLE:
            reward += 0.01

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Max length reached
        if self.tick >= self.max_episode_length:
            return True

        # Reality collapsed
        if self.instability.current_level == InstabilityLevel.COLLAPSE:
            if self.instability.collapse_timer is not None and self.instability.collapse_timer <= 0:
                return True

        return False

    def _get_observation(self) -> Observation:
        """Build observation for the agent."""
        # Player soul map tensor
        player_soul_tensor = self.player_soul_map.to_tensor()

        # Player context
        player_context = torch.tensor(
            [
                self.player_position[0] / 1000,
                self.player_position[1] / 1000,
                float(self.current_realm.value == "peregrine"),
                float(self.current_realm.value == "ministry"),
                float(self.current_realm.value == "spleen_towns"),
                float(self.current_realm.value == "city"),
                float(self.current_realm.value == "hollow"),
                float(self.current_realm.value == "nothing"),
                self.soul_scanner.player_tom_depth / 5.0,
                self.tick / self.max_episode_length,
            ]
            + [0.0] * (self.context_dim - 10),
            dtype=torch.float32,
        )

        # Realm features
        realm = get_realm(self.current_realm)
        realm_features = torch.tensor(
            [
                realm.get_vibe_compatibility(self.player_soul_map),
                len(realm.locations) / 10,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )

        # Instability state
        instability_state = torch.tensor(
            [
                self.instability.instability / 100,
                self.instability.current_level.value / 5,
                float(self.instability.nothing_manifested),
                (self.instability.collapse_timer or 1000) / 1000,
                len(self.instability.event_history) / 100,
            ],
            dtype=torch.float32,
        )

        # Nearby NPCs
        nearby_npc_states = []
        nearby_npcs = self._get_nearby_npcs(self.max_nearby_npcs)
        for npc in nearby_npcs:
            nearby_npc_states.append(npc.soul_map.to_tensor())

        # Pad if needed
        while len(nearby_npc_states) < self.max_nearby_npcs:
            nearby_npc_states.append(torch.zeros(self.soul_map_dim))

        # Relationships (simplified)
        npc_relationships = torch.tensor(
            [npc.reputation_with_player for npc in nearby_npcs] + [0.0] * (self.max_nearby_npcs - len(nearby_npcs)),
            dtype=torch.float32,
        )

        # Build full tensor
        full_tensor = torch.cat(
            [
                player_soul_tensor,
                player_context,
                realm_features,
                instability_state,
                torch.stack(nearby_npc_states).flatten(),
                npc_relationships,
            ]
        )

        return Observation(
            player_soul_map=player_soul_tensor,
            player_context=player_context,
            realm_features=realm_features,
            instability_state=instability_state,
            nearby_npc_states=nearby_npc_states,
            npc_relationships=npc_relationships,
            full_tensor=full_tensor,
        )

    def _get_nearby_npcs(self, max_count: int) -> List[BaseNPC]:
        """Get NPCs near the player position."""
        # Filter to same realm and sort by distance
        same_realm = [npc for npc in self.npcs.values() if npc.current_realm == self.current_realm]

        sorted_npcs = sorted(same_realm, key=lambda n: self._calculate_distance(n.position, self.player_position))

        return sorted_npcs[:max_count]

    def _get_game_state(self) -> GameState:
        """Build the current game state."""
        nearby = self._get_nearby_npcs(20)

        return GameState(
            tick=self.tick,
            current_realm=self.current_realm,
            player_position=self.player_position,
            player_tom_depth=self.soul_scanner.player_tom_depth,
            player_energy=1.0,
            instability=self.instability.instability,
            instability_level=self.instability.current_level,
            nearby_npcs=[npc.npc_id for npc in nearby],
            targeted_npc=None,
            active_quests=[],
            realm_state=get_realm(self.current_realm).realm_variables,
        )

    def get_npc(self, npc_id: str) -> Optional[BaseNPC]:
        """Get an NPC by ID."""
        return self.npcs.get(npc_id)

    def get_all_npcs(self) -> List[BaseNPC]:
        """Get all NPCs."""
        return list(self.npcs.values())

    def get_npcs_in_realm(self, realm_type: RealmType) -> List[BaseNPC]:
        """Get all NPCs in a specific realm."""
        return [npc for npc in self.npcs.values() if npc.current_realm == realm_type]

    def attach_nas_model(self, npc_id: str, model: Any) -> bool:
        """Attach a NAS model to an NPC for genuine ToM reasoning."""
        if npc_id not in self.npcs:
            return False

        self.npcs[npc_id]._nas_model = model
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "total_npcs": len(self.npcs),
            "npcs_per_realm": {realm.value: len(self.get_npcs_in_realm(realm)) for realm in RealmType},
            "hero_count": sum(1 for npc in self.npcs.values() if npc.is_hero),
            "zombie_count": sum(1 for npc in self.npcs.values() if npc.is_zombie),
            "current_tick": self.tick,
            "total_predictions": len(self.predictions_made),
            "prediction_accuracy": (
                sum(p["accuracy"] for p in self.predictions_made) / len(self.predictions_made)
                if self.predictions_made
                else 0.0
            ),
            "total_interventions": len(self.interventions_made),
            "instability_level": self.instability.current_level.value,
        }


# Export
__all__ = [
    "LiminalEnvironment",
    "GameState",
    "Observation",
    "StepResult",
    "ActionType",
]
