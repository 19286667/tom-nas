"""
REST API Server for ToM-NAS Game Integration

Exposes ToM-NAS functionality to game engines (Unity, Unreal, Web).
Provides real-time NPC reasoning, Soul Map updates, and dialogue generation.
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import torch
import asyncio
import json
import uuid
from datetime import datetime

# Import ToM-NAS core
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.agents.architectures import TRN, RSAN, TransformerToM
from src.world.social_world import SocialWorld4

# Initialize FastAPI
app = FastAPI(
    title="LIMINAL ARCHITECTURES API",
    description="ToM-NAS Game Integration Server",
    version="0.1.0"
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
ontology = SoulMapOntology()
active_sessions = {}
active_npcs = {}

# ============================================================================
# Data Models
# ============================================================================

class SoulMapState(BaseModel):
    """Complete Soul Map state (181 dimensions)"""
    dimensions: Dict[str, float]
    timestamp: Optional[str] = None

class NPCConfig(BaseModel):
    """Configuration for an NPC"""
    npc_id: str
    name: str
    architecture: str  # "TRN", "RSAN", or "Transformer"
    initial_soul_map: Optional[Dict[str, float]] = None
    personality_traits: Optional[Dict[str, Any]] = None

class DialogueRequest(BaseModel):
    """Request for NPC dialogue"""
    session_id: str
    npc_id: str
    player_soul_map: Dict[str, float]
    context: str
    player_last_utterance: Optional[str] = None

class DialogueResponse(BaseModel):
    """NPC's dialogue response with ToM reasoning"""
    npc_id: str
    text: str
    tom_reasoning: Dict[str, Any]
    npc_soul_map: Dict[str, float]
    predicted_player_soul_map: Dict[str, float]
    emotional_state: Dict[str, float]
    options: List[Dict[str, Any]]

class CombatAction(BaseModel):
    """Combat action with psychological component"""
    session_id: str
    attacker_id: str
    defender_id: str
    action_type: str  # "physical", "psychological", "hybrid"
    action_name: str
    soul_map_target: Optional[str] = None  # Which dimension to target

class ToMAnalysisRequest(BaseModel):
    """Request for Theory of Mind analysis"""
    session_id: str
    observer_id: str  # Who is doing the reasoning
    target_id: str    # Who is being reasoned about
    context: str
    order: int = 1    # 1st order, 2nd order, etc.

# ============================================================================
# Session Management
# ============================================================================

class GameSession:
    """Manages a single game session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.npcs: Dict[str, 'NPCController'] = {}
        self.player_soul_map = ontology.get_default_state()
        self.world = None
        self.created_at = datetime.now()
        self.last_update = datetime.now()

    def add_npc(self, config: NPCConfig) -> 'NPCController':
        """Add an NPC to the session"""
        npc = NPCController(config, ontology)
        self.npcs[config.npc_id] = npc
        return npc

    def update_player_soul_map(self, soul_map: Dict[str, float]):
        """Update player's Soul Map"""
        self.player_soul_map = ontology.encode(soul_map)
        self.last_update = datetime.now()

        # NPCs observe player changes
        for npc in self.npcs.values():
            npc.observe_player(self.player_soul_map)

    def get_npc(self, npc_id: str) -> Optional['NPCController']:
        """Get NPC by ID"""
        return self.npcs.get(npc_id)

# ============================================================================
# NPC Controller
# ============================================================================

class NPCController:
    """
    Controls a single NPC with ToM capabilities.
    Uses TRN, RSAN, or Transformer architecture.
    """

    def __init__(self, config: NPCConfig, ontology: SoulMapOntology):
        self.config = config
        self.ontology = ontology

        # Initialize Soul Map
        if config.initial_soul_map:
            self.soul_map = ontology.encode(config.initial_soul_map)
        else:
            self.soul_map = ontology.get_default_state()

        # Initialize ToM architecture
        self.agent = self._create_agent(config.architecture)

        # Belief network for recursive ToM
        self.beliefs = BeliefNetwork(ontology.total_dims)

        # Memory and state
        self.conversation_history = []
        self.observations = []
        self.predicted_player_soul_map = ontology.get_default_state()

        # Relationships
        self.relationship_with_player = 0.5  # 0 = hostile, 1 = friendly
        self.trust_in_player = 0.5

    def _create_agent(self, architecture: str):
        """Create the appropriate ToM agent"""
        hidden_dim = 256

        if architecture == "TRN":
            return TRN(
                input_dim=self.ontology.total_dims,
                hidden_dim=hidden_dim
            )
        elif architecture == "RSAN":
            return RSAN(
                input_dim=self.ontology.total_dims,
                hidden_dim=hidden_dim,
                num_recursions=3
            )
        elif architecture == "Transformer":
            return TransformerToM(
                input_dim=self.ontology.total_dims,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def observe_player(self, player_soul_map: torch.Tensor):
        """Observe player's Soul Map and update beliefs"""
        self.observations.append({
            'timestamp': datetime.now(),
            'player_soul_map': player_soul_map.clone()
        })

        # Use ToM to predict player's state
        with torch.no_grad():
            # Self state + observed player state
            state = torch.cat([self.soul_map, player_soul_map])
            prediction = self.agent(state.unsqueeze(0))

            # Extract predicted player soul map
            self.predicted_player_soul_map = prediction[0, :self.ontology.total_dims]

        # Update relationship based on observations
        self._update_relationship(player_soul_map)

    def _update_relationship(self, player_soul_map: torch.Tensor):
        """Update relationship metrics based on player state"""
        # Simple heuristic: alignment with player's values
        # In full implementation, this would be much more sophisticated

        # Check emotional alignment
        player_valence = player_soul_map[1]  # affect.valence
        npc_valence = self.soul_map[1]

        alignment = 1.0 - abs(player_valence - npc_valence)

        # Gradual relationship update
        self.relationship_with_player = (
            0.9 * self.relationship_with_player + 0.1 * alignment
        )

    def generate_dialogue(
        self,
        context: str,
        player_utterance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate NPC dialogue with ToM reasoning.

        Returns dialogue text, reasoning process, and response options.
        """

        # ToM reasoning process
        tom_reasoning = {
            'first_order': self._first_order_reasoning(),
            'second_order': self._second_order_reasoning(),
            'observed_player_state': self._describe_soul_map(
                self.predicted_player_soul_map
            ),
            'npc_emotional_state': self._describe_soul_map(self.soul_map),
            'relationship_assessment': {
                'affinity': float(self.relationship_with_player),
                'trust': float(self.trust_in_player)
            }
        }

        # Generate dialogue based on reasoning
        dialogue_text = self._generate_dialogue_text(
            context,
            player_utterance,
            tom_reasoning
        )

        # Generate response options for player
        options = self._generate_dialogue_options(context, tom_reasoning)

        return {
            'text': dialogue_text,
            'tom_reasoning': tom_reasoning,
            'npc_soul_map': self._soul_map_to_dict(self.soul_map),
            'predicted_player_soul_map': self._soul_map_to_dict(
                self.predicted_player_soul_map
            ),
            'options': options
        }

    def _first_order_reasoning(self) -> str:
        """First-order ToM: What does the player want/feel?"""
        # Simplified - in full version, use language model

        valence = float(self.predicted_player_soul_map[1])
        arousal = float(self.predicted_player_soul_map[2])

        if valence < 0.3:
            emotion = "distressed or sad"
        elif valence > 0.7:
            emotion = "content or happy"
        else:
            emotion = "neutral"

        if arousal > 0.7:
            energy = "agitated or excited"
        elif arousal < 0.3:
            energy = "calm or tired"
        else:
            energy = "balanced"

        return f"The player seems {emotion} and {energy}."

    def _second_order_reasoning(self) -> str:
        """Second-order ToM: What does the player think I believe?"""
        # Simplified implementation

        if self.relationship_with_player > 0.7:
            return "The player likely trusts me and expects empathy."
        elif self.relationship_with_player < 0.3:
            return "The player is wary of me and may expect hostility."
        else:
            return "The player is uncertain about my intentions."

    def _describe_soul_map(self, soul_map: torch.Tensor) -> Dict[str, float]:
        """Extract key dimensions from Soul Map"""
        return {
            'valence': float(soul_map[1]),
            'arousal': float(soul_map[2]),
            'dominance': float(soul_map[3]),
            'joy': float(soul_map[4]) if len(soul_map) > 4 else 0.5,
            'sadness': float(soul_map[5]) if len(soul_map) > 5 else 0.5,
        }

    def _soul_map_to_dict(self, soul_map: torch.Tensor) -> Dict[str, float]:
        """Convert Soul Map tensor to dictionary"""
        result = {}
        for i, dim in enumerate(self.ontology.dimensions):
            if i < len(soul_map):
                result[dim.name] = float(soul_map[i])
        return result

    def _generate_dialogue_text(
        self,
        context: str,
        player_utterance: Optional[str],
        tom_reasoning: Dict[str, Any]
    ) -> str:
        """Generate actual dialogue text"""
        # In full implementation, use LLM conditioned on ToM reasoning
        # For now, template-based

        name = self.config.name

        if self.relationship_with_player > 0.7:
            greeting = f"Hello, friend."
        elif self.relationship_with_player < 0.3:
            greeting = f"What do you want?"
        else:
            greeting = f"Greetings, traveler."

        return greeting

    def _generate_dialogue_options(
        self,
        context: str,
        tom_reasoning: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate player dialogue options"""

        options = [
            {
                'text': "Tell me about yourself.",
                'soul_map_effect': {'social.openness': 0.1},
                'tom_interpretation': "Player shows interest in me"
            },
            {
                'text': "I need your help.",
                'soul_map_effect': {'social.trust': 0.2, 'affect.vulnerability': 0.1},
                'tom_interpretation': "Player is vulnerable and seeks assistance"
            },
            {
                'text': "Goodbye.",
                'soul_map_effect': {},
                'tom_interpretation': "Player wishes to disengage"
            }
        ]

        return options

    def execute_combat_action(
        self,
        action_type: str,
        action_name: str,
        target_soul_map_dim: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a combat action with psychological component"""

        result = {
            'action': action_name,
            'physical_damage': 0,
            'psychological_damage': {},
            'tom_analysis': {}
        }

        if action_type == "physical":
            result['physical_damage'] = 10  # Base damage

        elif action_type == "psychological":
            # Analyze target for vulnerabilities
            vulnerabilities = self._find_psychological_vulnerabilities()

            if target_soul_map_dim in vulnerabilities:
                # Critical hit!
                result['psychological_damage'][target_soul_map_dim] = -0.3
                result['tom_analysis']['hit_vulnerability'] = True
            else:
                result['psychological_damage'][target_soul_map_dim] = -0.1

        return result

    def _find_psychological_vulnerabilities(self) -> List[str]:
        """Use ToM to identify psychological weaknesses"""
        # Check for extreme or unbalanced dimensions
        vulnerabilities = []

        soul_map_dict = self._soul_map_to_dict(self.predicted_player_soul_map)

        for dim, value in soul_map_dict.items():
            if value < 0.2 or value > 0.8:
                vulnerabilities.append(dim)

        return vulnerabilities

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "LIMINAL ARCHITECTURES API",
        "status": "operational",
        "version": "0.1.0",
        "active_sessions": len(active_sessions)
    }

@app.post("/session/create")
async def create_session() -> Dict[str, str]:
    """Create a new game session"""
    session_id = str(uuid.uuid4())
    session = GameSession(session_id)
    active_sessions[session_id] = session

    return {
        "session_id": session_id,
        "created_at": session.created_at.isoformat()
    }

@app.post("/session/{session_id}/npc/create")
async def create_npc(session_id: str, config: NPCConfig) -> Dict[str, Any]:
    """Create an NPC in a session"""

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    npc = session.add_npc(config)

    return {
        "npc_id": config.npc_id,
        "name": config.name,
        "architecture": config.architecture,
        "status": "created",
        "initial_soul_map": npc._soul_map_to_dict(npc.soul_map)
    }

@app.post("/session/{session_id}/player/update_soul_map")
async def update_player_soul_map(
    session_id: str,
    soul_map: Dict[str, float]
) -> Dict[str, str]:
    """Update player's Soul Map"""

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    session.update_player_soul_map(soul_map)

    return {"status": "updated"}

@app.post("/dialogue/generate")
async def generate_dialogue(request: DialogueRequest) -> DialogueResponse:
    """Generate NPC dialogue with ToM reasoning"""

    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[request.session_id]
    npc = session.get_npc(request.npc_id)

    if not npc:
        raise HTTPException(status_code=404, detail="NPC not found")

    # Update NPC's observation of player
    player_soul_map_tensor = ontology.encode(request.player_soul_map)
    npc.observe_player(player_soul_map_tensor)

    # Generate dialogue
    response = npc.generate_dialogue(
        context=request.context,
        player_utterance=request.player_last_utterance
    )

    return DialogueResponse(
        npc_id=request.npc_id,
        text=response['text'],
        tom_reasoning=response['tom_reasoning'],
        npc_soul_map=response['npc_soul_map'],
        predicted_player_soul_map=response['predicted_player_soul_map'],
        emotional_state=response['tom_reasoning']['npc_emotional_state'],
        options=response['options']
    )

@app.post("/combat/action")
async def execute_combat_action(action: CombatAction) -> Dict[str, Any]:
    """Execute a combat action"""

    if action.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[action.session_id]

    # Get attacker (NPC or player)
    if action.attacker_id == "player":
        attacker = None  # Player action
    else:
        attacker = session.get_npc(action.attacker_id)

    # Get defender
    if action.defender_id == "player":
        defender = None  # Player is defender
    else:
        defender = session.get_npc(action.defender_id)

    # Execute action
    if attacker:
        result = attacker.execute_combat_action(
            action.action_type,
            action.action_name,
            action.soul_map_target
        )
    else:
        # Player attacking - simplified
        result = {
            'action': action.action_name,
            'physical_damage': 15,
            'psychological_damage': {}
        }

    return result

@app.post("/tom/analyze")
async def analyze_tom(request: ToMAnalysisRequest) -> Dict[str, Any]:
    """Perform Theory of Mind analysis"""

    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[request.session_id]
    observer = session.get_npc(request.observer_id)

    if not observer:
        raise HTTPException(status_code=404, detail="Observer NPC not found")

    # Perform ToM reasoning
    reasoning = observer._first_order_reasoning()
    second_order = observer._second_order_reasoning()

    return {
        'order': request.order,
        'observer_id': request.observer_id,
        'target_id': request.target_id,
        'first_order_belief': reasoning,
        'second_order_belief': second_order if request.order >= 2 else None,
        'confidence': float(observer.trust_in_player)
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates"""
    await websocket.accept()

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Process message
            if message['type'] == 'soul_map_update':
                # Update player soul map
                if session_id in active_sessions:
                    session = active_sessions[session_id]
                    session.update_player_soul_map(message['soul_map'])

                    # Broadcast NPC reactions
                    reactions = {}
                    for npc_id, npc in session.npcs.items():
                        reactions[npc_id] = {
                            'relationship': float(npc.relationship_with_player),
                            'predicted_state': npc._describe_soul_map(
                                npc.predicted_player_soul_map
                            )
                        }

                    await websocket.send_json({
                        'type': 'npc_reactions',
                        'reactions': reactions
                    })

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("LIMINAL ARCHITECTURES - ToM-NAS Game Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
