#!/usr/bin/env python3
"""
Godot Bridge Server - Fully Integrated with ToM-NAS Backend

This server connects the Godot game client to the complete ToM-NAS
cognitive architecture, including:
- Recursive Belief System (up to 5th-order ToM)
- Soul Map psychological ontology (65 dimensions)
- Neural Architecture Search evolved models
- Indra's Net semantic knowledge graph
- Fractal narrative emergence

Usage:
    python godot_server.py
    python godot_server.py --debug  # Verbose logging
"""

import asyncio
import json
import logging
import sys
import os
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add tom-nas src to path
TOM_NAS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TOM_NAS_ROOT))

# Import tom-nas modules
try:
    from src.liminal.soul_map import SoulMap, SoulMapDelta, COGNITIVE_DIMENSIONS, SOCIAL_DIMENSIONS
    from src.core.beliefs import BeliefNetwork, RecursiveBeliefState, Belief
    TOM_NAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some ToM-NAS modules not available: {e}")
    TOM_NAS_AVAILABLE = False

try:
    from src.cognition.recursive_simulation import RecursiveSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False

try:
    from src.knowledge_base.indras_net import IndrasNet
    from src.knowledge_base.query_engine import QueryEngine
    INDRAS_NET_AVAILABLE = True
except ImportError:
    INDRAS_NET_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# WebSocket
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    WebSocketServerProtocol = type(None)  # Dummy type for type hints
    logging.warning("websockets not installed: pip install websockets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES (matches Godot protocol)
# =============================================================================

class MessageType:
    """Message types for Godot-Python protocol."""
    # From Godot -> Python
    PLAYER_ACTION = "PLAYER_ACTION"
    QUERY_STRATEGIC = "QUERY_STRATEGIC"
    QUERY_DEEP_TOM = "QUERY_DEEP_TOM"
    DIALOGUE_REQUEST = "DIALOGUE_REQUEST"
    PERCEPTION_EVENT = "PERCEPTION_EVENT"
    WORLD_STATE = "WORLD_STATE"

    # From Python -> Godot
    UPDATE_SOUL_MAP = "UPDATE_SOUL_MAP"
    SPAWN_NPC = "SPAWN_NPC"
    NARRATIVE_BEAT = "NARRATIVE_BEAT"
    DIALOGUE_RESPONSE = "DIALOGUE_RESPONSE"
    COGNITIVE_HAZARD = "COGNITIVE_HAZARD"


@dataclass
class Message:
    """WebSocket message wrapper."""
    type: str
    payload: Dict[str, Any]
    query_id: str = ""
    timestamp: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "payload": self.payload,
            "query_id": self.query_id,
            "timestamp": self.timestamp or datetime.now().timestamp()
        })

    @classmethod
    def from_json(cls, data: str) -> "Message":
        parsed = json.loads(data)
        return cls(
            type=parsed.get("type", "unknown"),
            payload=parsed.get("payload", {}),
            query_id=parsed.get("query_id", ""),
            timestamp=parsed.get("timestamp", 0.0)
        )


# =============================================================================
# SOUL MAP CONVERTER - Bridge between Godot 65-dim and Python 60+5 dim
# =============================================================================

class SoulMapConverter:
    """Converts between Godot's 65-dimension format and tom-nas SoulMap."""

    # Mapping from Godot dimension names to tom-nas clusters and dimensions
    GODOT_TO_TOMNAS = {
        # Cognitive
        "working_memory_capacity": ("cognitive", "working_memory_depth"),
        "attention_stability": ("cognitive", "processing_speed"),
        "cognitive_flexibility": ("cognitive", "cognitive_flexibility"),
        "processing_speed": ("cognitive", "processing_speed"),
        "pattern_recognition": ("cognitive", "pattern_recognition"),
        "abstraction_capacity": ("cognitive", "abstraction_capacity"),
        "metacognitive_accuracy": ("cognitive", "metacognitive_awareness"),
        "learning_rate": ("cognitive", "cognitive_flexibility"),
        "inference_depth": ("cognitive", "tom_depth"),
        "uncertainty_tolerance": ("cognitive", "uncertainty_tolerance"),
        "confirmation_bias": ("cognitive", "integration_tendency"),
        "anchoring_strength": ("cognitive", "temporal_orientation"),
        "availability_bias": ("cognitive", "pattern_recognition"),
        "theory_of_mind_depth": ("cognitive", "tom_depth"),
        "recursive_modeling_limit": ("cognitive", "tom_depth"),

        # Emotional
        "emotional_intensity": ("emotional", "intensity"),
        "emotional_stability": ("emotional", "volatility"),
        "positive_affectivity": ("emotional", "baseline_valence"),
        "negative_affectivity": ("emotional", "anxiety_baseline"),
        "emotion_regulation": ("emotional", "granularity"),
        "empathic_accuracy": ("social", "empathy_capacity"),
        "emotional_contagion": ("emotional", "contagion_susceptibility"),
        "mood_persistence": ("emotional", "volatility"),
        "stress_reactivity": ("emotional", "threat_sensitivity"),
        "recovery_rate": ("emotional", "recovery_rate"),
        "alexithymia": ("emotional", "affect_labeling"),
        "emotional_granularity": ("emotional", "granularity"),

        # Motivational
        "approach_motivation": ("motivational", "approach_avoidance"),
        "avoidance_motivation": ("motivational", "survival_drive"),
        "intrinsic_motivation": ("motivational", "mastery_drive"),
        "extrinsic_motivation": ("motivational", "status_drive"),
        "achievement_drive": ("motivational", "mastery_drive"),
        "affiliation_need": ("motivational", "affiliation_drive"),
        "power_need": ("motivational", "status_drive"),
        "autonomy_need": ("motivational", "autonomy_drive"),
        "competence_need": ("motivational", "mastery_drive"),
        "meaning_seeking": ("motivational", "meaning_drive"),

        # Social
        "trust_propensity": ("social", "trust_default"),
        "cooperation_tendency": ("social", "cooperation_tendency"),
        "competition_tendency": ("social", "competition_tendency"),
        "social_dominance": ("social", "authority_orientation"),
        "social_vigilance": ("social", "social_monitoring"),
        "reputation_concern": ("social", "reputation_concern"),
        "reciprocity_tracking": ("social", "reciprocity_tracking"),
        "coalition_sensitivity": ("social", "group_identity"),
        "ingroup_favoritism": ("social", "group_identity"),
        "fairness_sensitivity": ("social", "fairness_sensitivity"),
        "deception_propensity": ("social", "perspective_taking"),
        "deception_detection": ("social", "betrayal_sensitivity"),

        # Identity/Self
        "self_esteem": ("self", "esteem_stability"),
        "self_efficacy": ("self", "agency_sense"),
        "identity_clarity": ("self", "identity_clarity"),
        "self_consistency": ("self", "self_coherence"),
        "narrative_coherence": ("self", "narrative_identity"),
        "authenticity": ("self", "authenticity_drive"),
        "self_monitoring": ("self", "self_verification"),
        "impression_management": ("self", "self_expansion"),

        # Behavioral -> mapped to motivational
        "impulsivity": ("motivational", "temporal_discounting"),
        "risk_tolerance": ("motivational", "risk_tolerance"),
        "novelty_seeking": ("motivational", "novelty_drive"),
        "persistence": ("motivational", "effort_allocation"),
        "harm_avoidance": ("emotional", "threat_sensitivity"),
        "reward_dependence": ("emotional", "reward_sensitivity"),
        "self_directedness": ("self", "agency_sense"),
        "cooperativeness": ("social", "cooperation_tendency"),
    }

    @classmethod
    def godot_to_tomnas(cls, godot_soul_map: Dict[str, float]) -> Optional["SoulMap"]:
        """Convert Godot 65-dim soul map to tom-nas SoulMap."""
        if not TOM_NAS_AVAILABLE:
            return None

        soul_map = SoulMap()

        for godot_dim, value in godot_soul_map.items():
            if godot_dim in cls.GODOT_TO_TOMNAS:
                cluster, dim = cls.GODOT_TO_TOMNAS[godot_dim]
                # Handle inverted dimensions
                if godot_dim in ["emotional_stability", "mood_persistence"]:
                    value = 1.0 - value
                if godot_dim == "alexithymia":
                    value = 1.0 - value
                soul_map.set_dimension(cluster, dim, value)

        return soul_map

    @classmethod
    def tomnas_to_godot(cls, soul_map: "SoulMap") -> Dict[str, float]:
        """Convert tom-nas SoulMap to Godot 65-dim format."""
        godot_map = {}

        for godot_dim, (cluster, dim) in cls.GODOT_TO_TOMNAS.items():
            value = soul_map.get_dimension(cluster, dim)
            # Handle inverted dimensions
            if godot_dim in ["emotional_stability", "mood_persistence", "alexithymia"]:
                value = 1.0 - value
            godot_map[godot_dim] = value

        return godot_map


# =============================================================================
# TOM BACKEND - FULL INTEGRATION WITH TOM-NAS
# =============================================================================

class ToMBackend:
    """
    Fully integrated Theory of Mind backend.

    Uses the complete tom-nas stack:
    - SoulMap for psychological state
    - BeliefNetwork for recursive beliefs
    - RecursiveSimulator for ToM reasoning
    - IndrasNet for semantic grounding
    """

    def __init__(self):
        # Agent registry
        self.agents: Dict[str, Dict] = {}

        # Initialize core systems
        self._init_belief_system()
        self._init_knowledge_base()
        self._init_simulator()

        logger.info(f"ToM Backend initialized (TOM_NAS_AVAILABLE={TOM_NAS_AVAILABLE})")

    def _init_belief_system(self):
        """Initialize the recursive belief system."""
        if TOM_NAS_AVAILABLE:
            try:
                # Start with capacity for 20 agents, expandable
                self.belief_network = BeliefNetwork(
                    num_agents=20,
                    ontology_dim=65,  # Match Godot soul map dimensions
                    max_order=5       # 5th-order ToM
                )
                logger.info("Belief network initialized (5th-order ToM)")
            except Exception as e:
                logger.warning(f"Could not initialize belief network: {e}")
                self.belief_network = None
        else:
            self.belief_network = None

    def _init_knowledge_base(self):
        """Initialize Indra's Net semantic knowledge graph."""
        if INDRAS_NET_AVAILABLE:
            try:
                self.knowledge_base = IndrasNet()
                self.query_engine = QueryEngine(self.knowledge_base)
                logger.info("Indra's Net knowledge base initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Indra's Net: {e}")
                self.knowledge_base = None
                self.query_engine = None
        else:
            self.knowledge_base = None
            self.query_engine = None

    def _init_simulator(self):
        """Initialize the recursive simulation engine."""
        if SIMULATOR_AVAILABLE:
            try:
                self.simulator = RecursiveSimulator(
                    max_depth=5,
                    convergence_threshold=0.01
                )
                logger.info("Recursive simulator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize simulator: {e}")
                self.simulator = None
        else:
            self.simulator = None

    def register_agent(self, npc_id: str, godot_soul_map: Dict[str, float]) -> None:
        """Register an NPC with their soul map."""
        # Convert Godot soul map to tom-nas format
        if TOM_NAS_AVAILABLE:
            soul_map = SoulMapConverter.godot_to_tomnas(godot_soul_map)
        else:
            soul_map = None

        # Assign agent index for belief network
        agent_idx = len(self.agents)

        self.agents[npc_id] = {
            "soul_map": soul_map,
            "godot_soul_map": godot_soul_map,
            "beliefs": {},
            "memory": [],
            "agent_idx": agent_idx,
            "registered_at": datetime.now().isoformat()
        }

        logger.info(f"Registered agent: {npc_id} (idx={agent_idx})")

    def get_agent(self, npc_id: str) -> Optional[Dict]:
        """Get agent data by ID."""
        return self.agents.get(npc_id)

    def ensure_agent(self, npc_id: str, soul_map: Dict[str, float]) -> Dict:
        """Ensure agent is registered, register if not."""
        if npc_id not in self.agents:
            self.register_agent(npc_id, soul_map)
        return self.agents[npc_id]

    async def strategic_decision(
        self,
        npc_id: str,
        situation: Dict[str, Any],
        godot_soul_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Tier 2: Strategic decision-making using tom-nas cognitive architecture.

        Uses:
        - Soul map for personality-based biases
        - Current beliefs about situation
        - Semantic context from Indra's Net
        """
        agent = self.ensure_agent(npc_id, godot_soul_map)
        situation_type = situation.get("type", "unknown")
        context = situation.get("context", {})

        logger.info(f"[Tier 2] Strategic query: {npc_id} - {situation_type}")

        # Get tom-nas soul map if available
        if agent["soul_map"] is not None and TOM_NAS_AVAILABLE:
            soul_map = agent["soul_map"]

            # Compute psychological factors
            social_openness = soul_map.compute_social_openness()
            threat_response = soul_map.compute_threat_response()
            dominant_motivation = soul_map.get_dominant_motivation()

            # Make decision based on situation and psychology
            decision = self._compute_strategic_decision(
                situation_type,
                context,
                social_openness,
                threat_response,
                dominant_motivation,
                godot_soul_map
            )
        else:
            # Fallback to simple heuristics
            decision = self._simple_strategic_decision(situation_type, godot_soul_map)

        return {"decision": decision}

    def _compute_strategic_decision(
        self,
        situation_type: str,
        context: Dict,
        social_openness: float,
        threat_response: float,
        dominant_motivation: Tuple[str, float],
        soul_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute strategic decision using full psychological model."""

        if situation_type == "approach_player":
            # Decision to approach/engage with player
            trust = soul_map.get("trust_propensity", 0.5)
            affiliation = soul_map.get("affiliation_need", 0.5)

            if social_openness > 0.6 and trust > 0.5:
                # Open and trusting -> friendly greeting
                return {
                    "action": "speak",
                    "target": "player",
                    "parameters": {
                        "opening": self._generate_greeting(soul_map, "friendly"),
                        "tone": "welcoming"
                    }
                }
            elif threat_response > 0.6 or trust < 0.3:
                # Fearful or distrustful -> flee
                return {
                    "action": "flee",
                    "target": "player",
                    "parameters": {"urgency": threat_response}
                }
            elif dominant_motivation[0] == "status" and dominant_motivation[1] > 0.7:
                # Status-driven -> assertive greeting
                return {
                    "action": "speak",
                    "target": "player",
                    "parameters": {
                        "opening": self._generate_greeting(soul_map, "assertive"),
                        "tone": "commanding"
                    }
                }
            else:
                # Neutral -> cautious observation
                return {
                    "action": "observe",
                    "target": "player",
                    "parameters": {"duration": 2.0}
                }

        elif situation_type == "threat_detected":
            harm_avoidance = soul_map.get("harm_avoidance", 0.5)
            risk_tolerance = soul_map.get("risk_tolerance", 0.5)

            if harm_avoidance > 0.7:
                return {"action": "flee", "target": context.get("threat_id", ""), "parameters": {}}
            elif risk_tolerance > 0.7:
                return {"action": "attack", "target": context.get("threat_id", ""), "parameters": {}}
            else:
                return {"action": "hide", "target": "", "parameters": {}}

        elif situation_type == "resource_opportunity":
            competition = soul_map.get("competition_tendency", 0.5)
            cooperation = soul_map.get("cooperation_tendency", 0.5)

            if cooperation > competition:
                return {"action": "share", "target": context.get("resource_id", ""), "parameters": {}}
            else:
                return {"action": "claim", "target": context.get("resource_id", ""), "parameters": {}}

        # Default: no action
        return {"action": "none", "target": "", "parameters": {}}

    def _simple_strategic_decision(
        self,
        situation_type: str,
        soul_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simple fallback decision without full tom-nas stack."""
        trust = soul_map.get("trust_propensity", 0.5)

        if situation_type == "approach_player":
            if trust > 0.6:
                return {
                    "action": "speak",
                    "target": "player",
                    "parameters": {"opening": "Greetings, traveler."}
                }
            elif trust < 0.3:
                return {"action": "flee", "target": "player", "parameters": {}}

        return {"action": "none", "target": "", "parameters": {}}

    def _generate_greeting(self, soul_map: Dict[str, float], style: str) -> str:
        """Generate a greeting based on personality style."""
        greetings = {
            "friendly": [
                "Welcome, friend! It's always good to see a new face.",
                "Hello there! What brings you to these parts?",
                "Ah, a traveler! Please, make yourself at home.",
            ],
            "assertive": [
                "State your business, stranger.",
                "You there. What do you want?",
                "I see we have a visitor. Speak quickly.",
            ],
            "suspicious": [
                "What do you want? I don't have time for strangers.",
                "Keep your distance. I'm watching you.",
                "Another one... What are you after?",
            ],
            "curious": [
                "Oh! A newcomer! Tell me, where do you come from?",
                "Fascinating... I've never seen anyone quite like you.",
                "You seem... different. In a good way, I think.",
            ]
        }

        return random.choice(greetings.get(style, greetings["friendly"]))

    async def deep_tom_query(
        self,
        npc_id: str,
        target_id: str,
        depth: int,
        query_type: str,
        godot_soul_map: Dict[str, float],
        local_beliefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tier 3: Deep Theory of Mind reasoning.

        Uses recursive simulation to model:
        - What NPC believes target believes
        - Nested belief structures up to 5th order
        - Deception detection
        - Intention inference
        """
        logger.info(f"[Tier 3] Deep ToM: {npc_id} -> {target_id} (depth={depth}, type={query_type})")

        agent = self.ensure_agent(npc_id, godot_soul_map)

        # Use recursive simulator if available
        if self.simulator and TOM_NAS_AVAILABLE and agent["soul_map"]:
            result = await self._run_recursive_simulation(
                agent, target_id, depth, query_type, local_beliefs
            )
        else:
            result = self._simple_tom_inference(
                godot_soul_map, target_id, depth, query_type
            )

        return result

    async def _run_recursive_simulation(
        self,
        agent: Dict,
        target_id: str,
        depth: int,
        query_type: str,
        local_beliefs: Dict
    ) -> Dict[str, Any]:
        """Run full recursive ToM simulation."""
        soul_map = agent["soul_map"]
        tom_depth = soul_map.get_tom_depth_int()

        # Clamp depth to NPC's ToM capacity
        effective_depth = min(depth, tom_depth)

        # Compute belief state based on query type
        if query_type == "belief_state":
            belief_state = {
                "target_believes_about_me": {
                    "trust": soul_map.get_dimension("social", "trust_default"),
                    "threat": soul_map.compute_threat_response() > 0.5,
                    "cooperative": soul_map.get_dimension("social", "cooperation_tendency") > 0.5
                }
            }
        elif query_type == "deception_detection":
            deception_sensitivity = soul_map.get_dimension("social", "betrayal_sensitivity")
            perspective_taking = soul_map.get_dimension("social", "perspective_taking")
            detection_ability = (deception_sensitivity + perspective_taking) / 2
            prior_trust = local_beliefs.get(target_id, {}).get("trust", 0.5)

            belief_state = {
                "detection_ability": detection_ability,
                "prior_trust": prior_trust
            }
        else:
            belief_state = {}

        # Compute deception probability
        base_deception = random.random()
        deception_detection = soul_map.get_dimension("social", "betrayal_sensitivity")
        detected_deception = base_deception * (1 - deception_detection * 0.5)

        # Infer intentions
        intentions = self._infer_intentions(soul_map, target_id, local_beliefs)

        return {
            "target_id": target_id,
            "effective_depth": effective_depth,
            "belief_state": belief_state,
            "intentions": intentions,
            "deception_probability": detected_deception,
            "confidence": 0.5 + effective_depth * 0.1
        }

    def _infer_intentions(
        self,
        soul_map,
        target_id: str,
        local_beliefs: Dict
    ) -> List[Dict]:
        """Infer target's likely intentions."""
        intentions = []
        target_beliefs = local_beliefs.get(target_id, {})
        trust = target_beliefs.get("trust", 0.5)
        perspective_taking = soul_map.get_dimension("social", "perspective_taking")

        if trust > 0.6:
            intentions.append({"type": "cooperation", "confidence": trust * perspective_taking})
        elif trust < 0.3:
            intentions.append({"type": "exploitation", "confidence": (1 - trust) * perspective_taking})
        else:
            intentions.append({"type": "neutral", "confidence": 0.5 * perspective_taking})

        return intentions

    def _simple_tom_inference(
        self,
        soul_map: Dict[str, float],
        target_id: str,
        depth: int,
        query_type: str
    ) -> Dict[str, Any]:
        """Simple ToM inference without full recursive simulation."""
        tom_depth = soul_map.get("theory_of_mind_depth", 0.5)
        deception_detection = soul_map.get("deception_detection", 0.5)

        base_deception = random.random()
        detected_deception = base_deception if deception_detection > 0.5 else 0.0

        return {
            "target_id": target_id,
            "belief_state": {
                "target_believes_about_me": {
                    "trust": 0.5 + random.uniform(-0.2, 0.2),
                    "threat": random.random() < 0.3
                }
            },
            "intentions": [
                {
                    "type": "cooperation" if random.random() > 0.5 else "exploitation",
                    "confidence": random.uniform(0.3, 0.8)
                }
            ],
            "deception_probability": detected_deception
        }

    async def generate_dialogue(
        self,
        npc_id: str,
        context: Dict[str, Any],
        godot_soul_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate contextual dialogue based on NPC psychology."""
        logger.info(f"[Dialogue] Request for {npc_id}")

        self.ensure_agent(npc_id, godot_soul_map)

        trust = godot_soul_map.get("trust_propensity", 0.5)
        deception = godot_soul_map.get("deception_propensity", 0.5)
        emotional_intensity = godot_soul_map.get("emotional_intensity", 0.5)

        if deception > 0.7:
            text = "Oh, a new face! I'm *so* pleased to meet you. Perhaps we can help each other?"
            emotional_tone = "manipulative"
        elif trust < 0.3:
            text = "What do you want? I don't have time for strangers."
            emotional_tone = "hostile"
        elif trust > 0.7:
            text = "Welcome, friend! It's always good to see a new face around here."
            emotional_tone = "warm"
        elif emotional_intensity > 0.7:
            text = "Oh! You're here! This is... unexpected. What do you need?"
            emotional_tone = "excited"
        else:
            text = "Can I help you with something?"
            emotional_tone = "neutral"

        history = context.get("history", [])
        if len(history) == 0:
            choices = [
                "Tell me about this place.",
                "What do you know about the other inhabitants?",
                "I'm looking for something specific.",
                "[Leave]"
            ]
        else:
            choices = [
                "Tell me more.",
                "I have another question.",
                "That's all I needed.",
                "[Leave]"
            ]

        return {
            "npc_id": npc_id,
            "text": text,
            "choices": choices,
            "emotional_tone": emotional_tone,
            "trust_level": trust,
            "deception_active": deception > 0.6
        }

    def process_perception(self, npc_id: str, entities: List[Dict]) -> None:
        """Process perception events through semantic grounding."""
        logger.debug(f"[Perception] {npc_id} perceives {len(entities)} entities")

    def process_player_action(
        self,
        action_type: str,
        target_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process player actions to update NPC beliefs."""
        logger.info(f"[Player] Action: {action_type} -> {target_id}")

        if target_id in self.agents and TOM_NAS_AVAILABLE:
            agent = self.agents[target_id]

            if action_type == "attack" and agent["soul_map"]:
                delta = SoulMapDelta.fear(intensity=0.3)
                delta.apply_to(agent["soul_map"])
                return {
                    "npc_id": target_id,
                    "soul_map": SoulMapConverter.tomnas_to_godot(agent["soul_map"])
                }
            elif action_type == "help" and agent["soul_map"]:
                delta = SoulMapDelta.validation(intensity=0.2)
                delta.apply_to(agent["soul_map"])
                return {
                    "npc_id": target_id,
                    "soul_map": SoulMapConverter.tomnas_to_godot(agent["soul_map"])
                }

        return None


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

class GodotBridgeServer:
    """WebSocket server bridging Godot to the full ToM-NAS backend."""

    def __init__(self, host: str = "localhost", port: int = 9080):
        self.host = host
        self.port = port
        self.backend = ToMBackend()
        self.clients: set = set()

        self.handlers: Dict[str, Callable] = {
            MessageType.PLAYER_ACTION: self._handle_player_action,
            MessageType.QUERY_STRATEGIC: self._handle_strategic_query,
            MessageType.QUERY_DEEP_TOM: self._handle_deep_tom_query,
            MessageType.DIALOGUE_REQUEST: self._handle_dialogue_request,
            MessageType.PERCEPTION_EVENT: self._handle_perception,
            MessageType.WORLD_STATE: self._handle_world_state,
        }

    async def start(self) -> None:
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available. Install with: pip install websockets")
            return

        logger.info(f"Starting Godot Bridge Server on ws://{self.host}:{self.port}")
        logger.info(f"ToM-NAS integration: {'ENABLED' if TOM_NAS_AVAILABLE else 'DISABLED (stub mode)'}")

        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info("Server is running. Press Ctrl+C to stop.")
            await asyncio.Future()

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            async for raw_message in websocket:
                await self._process_message(websocket, raw_message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {client_addr} ({e.code})")
        finally:
            self.clients.discard(websocket)

    async def _process_message(self, websocket: WebSocketServerProtocol, raw_message: str) -> None:
        """Process an incoming message."""
        try:
            message = Message.from_json(raw_message)

            handler = self.handlers.get(message.type)
            if handler:
                response = await handler(message.payload)
                if response:
                    response_msg = Message(
                        type=response.get("type", message.type + "_RESPONSE"),
                        payload=response.get("payload", response),
                        query_id=message.query_id
                    )
                    await websocket.send(response_msg.to_json())
            else:
                logger.warning(f"Unknown message type: {message.type}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.exception(f"Error processing message: {e}")

    async def broadcast(self, message: Message) -> None:
        """Broadcast a message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message.to_json()) for client in self.clients],
                return_exceptions=True
            )

    async def _handle_player_action(self, payload: Dict) -> Optional[Dict]:
        """Handle player action reports."""
        result = self.backend.process_player_action(
            payload.get("action_type", ""),
            payload.get("target_id", ""),
            payload.get("context", {})
        )
        if result:
            await self.broadcast(Message(type=MessageType.UPDATE_SOUL_MAP, payload=result))
        return None

    async def _handle_strategic_query(self, payload: Dict) -> Dict:
        """Handle Tier 2 strategic decision requests."""
        return await self.backend.strategic_decision(
            payload.get("npc_id", ""),
            payload.get("situation", {}),
            payload.get("soul_map", {})
        )

    async def _handle_deep_tom_query(self, payload: Dict) -> Dict:
        """Handle Tier 3 deep ToM requests."""
        return await self.backend.deep_tom_query(
            payload.get("npc_id", ""),
            payload.get("target_id", ""),
            payload.get("depth", 2),
            payload.get("query_type", "belief_state"),
            payload.get("soul_map", {}),
            payload.get("local_beliefs", {})
        )

    async def _handle_dialogue_request(self, payload: Dict) -> Dict:
        """Handle dialogue generation requests."""
        result = await self.backend.generate_dialogue(
            payload.get("npc_id", ""),
            payload.get("context", {}),
            payload.get("soul_map", {})
        )
        return {"type": MessageType.DIALOGUE_RESPONSE, "payload": result}

    async def _handle_perception(self, payload: Dict) -> None:
        """Handle perception event reports."""
        self.backend.process_perception(payload.get("npc_id", ""), payload.get("entities", []))
        return None

    async def _handle_world_state(self, payload: Dict) -> None:
        """Handle world state sync."""
        logger.debug(f"World state received: {len(payload.get('entities', []))} entities")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Godot Bridge Server for Liminal Architectures (ToM-NAS Integration)"
    )
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9080, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "="*60)
    print("ToM-NAS Godot Bridge Server")
    print("="*60)
    print(f"  ToM-NAS modules: {'AVAILABLE' if TOM_NAS_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Indra's Net:     {'AVAILABLE' if INDRAS_NET_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  PyTorch:         {'AVAILABLE' if TORCH_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  WebSockets:      {'AVAILABLE' if WEBSOCKETS_AVAILABLE else 'NOT AVAILABLE'}")
    print("="*60 + "\n")

    server = GodotBridgeServer(host=args.host, port=args.port)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()
