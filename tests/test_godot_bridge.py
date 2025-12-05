"""
Tests for Godot Bridge Integration

Tests the WebSocket communication, protocol handling, and
symbol grounding between Python and Godot.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.godot_bridge.protocol import (
    MessageType, GodotMessage, EntityUpdate, AgentPerception,
    WorldState, AgentCommand, Vector3, InteractionEvent, UtteranceEvent
)
from src.godot_bridge.bridge import GodotBridge, BridgeConfig, ConnectionState
from src.godot_bridge.symbol_grounding import (
    SymbolGrounder, GroundedSymbol, GroundingContext, VisualFeatures
)
from src.godot_bridge.action import ActionExecutor, GodotAction, ActionType, ActionResult
from src.godot_bridge.perception import PerceptionProcessor, SensoryInput, PerceptualField


class TestProtocol:
    """Test protocol message handling."""

    def test_vector3_creation(self):
        """Test Vector3 creation and operations."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)

        assert v1.to_tuple() == (1.0, 2.0, 3.0)
        assert Vector3.from_tuple((1.0, 2.0, 3.0)).x == 1.0

        # Distance calculation
        distance = v1.distance_to(v2)
        assert distance == pytest.approx(5.196, rel=0.01)

    def test_entity_update_serialization(self):
        """Test EntityUpdate serialization."""
        entity = EntityUpdate(
            godot_id=12345,
            entity_type="agent",
            name="TestAgent",
            position=Vector3(1.0, 0.0, 2.0),
            semantic_tags=["agent", "test"],
            affordances=["can_speak"]
        )

        data = entity.to_dict()
        assert data['godot_id'] == 12345
        assert data['entity_type'] == "agent"
        assert data['position']['x'] == 1.0

        # Roundtrip
        restored = EntityUpdate.from_dict(data)
        assert restored.godot_id == entity.godot_id
        assert restored.name == entity.name

    def test_agent_perception_creation(self):
        """Test AgentPerception creation."""
        perception = AgentPerception(
            agent_godot_id=1001,
            agent_name="Observer",
            visible_entities=[],
            own_position=Vector3(5.0, 0.0, 5.0),
            energy_level=0.8
        )

        data = perception.to_dict()
        assert data['agent_godot_id'] == 1001
        assert data['energy_level'] == 0.8

    def test_world_state_creation(self):
        """Test WorldState creation."""
        state = WorldState(
            simulation_time=100.5,
            timestep=6000,
            time_of_day=14.5,
            weather="cloudy"
        )

        data = state.to_dict()
        assert data['simulation_time'] == 100.5
        assert data['weather'] == "cloudy"

    def test_agent_command_creation(self):
        """Test AgentCommand creation."""
        command = AgentCommand(
            agent_godot_id=1001,
            command_type="move",
            target_position=Vector3(10.0, 0.0, 10.0),
            command_id="cmd_001"
        )

        data = command.to_dict()
        assert data['command_type'] == "move"
        assert data['target_position']['x'] == 10.0

    def test_godot_message_serialization(self):
        """Test GodotMessage JSON serialization."""
        msg = GodotMessage(
            message_type=MessageType.ENTITY_UPDATE,
            payload={'test': 'data'},
            sequence_id=42
        )

        json_str = msg.to_json()
        parsed = json.loads(json_str)
        assert parsed['type'] == 'ENTITY_UPDATE'
        assert parsed['sequence_id'] == 42

        # Roundtrip
        restored = GodotMessage.from_json(json_str)
        assert restored.message_type == msg.message_type
        assert restored.payload == msg.payload

    def test_message_factory_methods(self):
        """Test GodotMessage factory methods."""
        # Heartbeat
        heartbeat = GodotMessage.create_heartbeat()
        assert heartbeat.message_type == MessageType.HEARTBEAT

        # Error
        error = GodotMessage.create_error("Test error", 500)
        assert error.message_type == MessageType.ERROR
        assert error.payload['error'] == "Test error"


class TestSymbolGrounding:
    """Test symbol grounding system."""

    def test_symbol_grounder_initialization(self):
        """Test SymbolGrounder initialization."""
        grounder = SymbolGrounder()
        assert len(grounder.grounded_symbols) == 0
        assert len(grounder.category_prototypes) > 0

    def test_ground_entity(self):
        """Test grounding an entity."""
        grounder = SymbolGrounder()

        entity = EntityUpdate(
            godot_id=1001,
            entity_type="object",
            name="Chair",
            position=Vector3(5.0, 0.0, 5.0),
            scale=Vector3(1.0, 1.0, 1.0),
            semantic_tags=["furniture", "seating"]
        )

        symbol = grounder.ground_entity(entity)

        assert symbol.godot_id == 1001
        assert symbol.name == "Chair"
        assert symbol.entity_type == "object"
        assert symbol.position.x == 5.0

    def test_grounding_context(self):
        """Test grounding with context."""
        grounder = SymbolGrounder()

        entity = EntityUpdate(
            godot_id=1002,
            entity_type="object",
            name="Gavel",
            position=Vector3(0.0, 1.0, 0.0)
        )

        context = GroundingContext(
            current_institution="court",
            simulation_time=500.0
        )

        symbol = grounder.ground_entity(entity, context)
        assert symbol is not None

    def test_visual_feature_extraction(self):
        """Test visual feature extraction."""
        grounder = SymbolGrounder()

        entity = EntityUpdate(
            godot_id=1003,
            entity_type="object",
            name="Table",
            scale=Vector3(2.0, 0.8, 1.5),
            color=(0.5, 0.3, 0.2, 1.0)
        )

        features = grounder._extract_visual_features(entity)

        assert features.width == 2.0
        assert features.height == 0.8
        assert features.volume == pytest.approx(2.4)

    def test_prototype_matching(self):
        """Test category prototype matching."""
        grounder = SymbolGrounder()

        # Features similar to chair prototype
        features = VisualFeatures(
            volume=0.5,
            height=0.8,
            width=0.5,
            depth=0.5,
            has_legs=True,
            is_flat=False
        )

        category, similarity = grounder._match_prototype(features)
        assert category in grounder.category_prototypes
        assert 0 <= similarity <= 1

    def test_affordance_inference(self):
        """Test affordance inference from features."""
        grounder = SymbolGrounder()

        # Small pickupable object
        small_features = VisualFeatures(
            volume=0.05,
            height=0.2
        )
        affordances = grounder._infer_affordances(small_features)
        assert 'can_pick_up' in affordances

        # Large sitting surface
        seat_features = VisualFeatures(
            height=0.5,
            has_legs=True,
            is_elongated=False
        )
        affordances = grounder._infer_affordances(seat_features)
        assert 'can_sit_on' in affordances

    def test_symbol_position_tracking(self):
        """Test tracking symbol position changes."""
        grounder = SymbolGrounder()

        entity1 = EntityUpdate(
            godot_id=1004,
            entity_type="object",
            name="Ball",
            position=Vector3(0.0, 0.0, 0.0)
        )

        symbol = grounder.ground_entity(entity1)
        assert symbol.is_stable

        # Update with moved position
        entity2 = EntityUpdate(
            godot_id=1004,
            entity_type="object",
            name="Ball",
            position=Vector3(5.0, 0.0, 5.0)
        )

        symbol = grounder.ground_entity(entity2)
        assert not symbol.is_stable
        assert len(symbol.change_history) > 0

    def test_get_symbols_by_category(self):
        """Test retrieving symbols by category."""
        grounder = SymbolGrounder()

        # Ground multiple entities
        for i, name in enumerate(['ChairA', 'ChairB', 'Table']):
            entity = EntityUpdate(
                godot_id=2000 + i,
                entity_type="object",
                name=name,
                position=Vector3(i * 2.0, 0.0, 0.0)
            )
            grounder.ground_entity(entity)

        # Query by category
        chairs = grounder.get_symbols_in_category('chair')
        # Note: actual matches depend on prototype matching
        assert isinstance(chairs, list)


class TestActionExecutor:
    """Test action execution system."""

    def test_action_executor_initialization(self):
        """Test ActionExecutor initialization."""
        grounder = SymbolGrounder()
        executor = ActionExecutor(grounder)

        assert len(executor.action_queues) == 0
        assert len(executor.executing) == 0

    def test_plan_action(self):
        """Test action planning."""
        grounder = SymbolGrounder()
        executor = ActionExecutor(grounder)

        action = executor.plan_action(
            agent_id=1001,
            action_type=ActionType.MOVE_TO,
            target_position=Vector3(10.0, 0.0, 10.0),
            reason="Testing movement"
        )

        assert action is not None
        assert action.agent_godot_id == 1001
        assert action.action_type == ActionType.MOVE_TO

    def test_action_queue(self):
        """Test action queuing."""
        grounder = SymbolGrounder()
        executor = ActionExecutor(grounder)

        action1 = GodotAction(
            action_type=ActionType.MOVE_TO,
            agent_godot_id=1001,
            target_position=Vector3(5.0, 0.0, 5.0),
            priority=0
        )

        action2 = GodotAction(
            action_type=ActionType.SPEAK,
            agent_godot_id=1001,
            utterance="Hello",
            priority=1  # Higher priority
        )

        executor.queue_action(action1)
        executor.queue_action(action2)

        queue = executor.action_queues[1001]
        # Higher priority should be first
        assert queue[0].priority > queue[1].priority

    def test_action_to_command_conversion(self):
        """Test converting action to Godot command."""
        action = GodotAction(
            action_type=ActionType.PICK_UP,
            agent_godot_id=1001,
            target_entity_id=2001,
            command_id="cmd_test"
        )

        command = action.to_command()
        assert command.agent_godot_id == 1001
        assert command.command_type == "pick_up"
        assert command.target_entity_id == 2001


class TestPerceptionProcessor:
    """Test perception processing."""

    def test_perception_processor_initialization(self):
        """Test PerceptionProcessor initialization."""
        grounder = SymbolGrounder()
        processor = PerceptionProcessor(grounder)

        assert processor.grounder is grounder
        assert processor.perceptions_processed == 0

    def test_process_perception(self):
        """Test processing a perception."""
        grounder = SymbolGrounder()
        processor = PerceptionProcessor(grounder)

        # Create test perception
        visible_entity = EntityUpdate(
            godot_id=2001,
            entity_type="object",
            name="TestObject",
            position=Vector3(3.0, 0.0, 3.0)
        )

        perception = AgentPerception(
            agent_godot_id=1001,
            agent_name="Observer",
            visible_entities=[visible_entity],
            own_position=Vector3(0.0, 0.0, 0.0)
        )

        field = processor.process_perception(perception)

        assert field.agent_godot_id == 1001
        assert len(field.visual_inputs) > 0
        assert processor.perceptions_processed == 1

    def test_salience_computation(self):
        """Test salience detection."""
        grounder = SymbolGrounder()
        processor = PerceptionProcessor(grounder)

        # Moving entity should be salient
        moving_entity = EntityUpdate(
            godot_id=2002,
            entity_type="agent",
            name="MovingAgent",
            position=Vector3(2.0, 0.0, 2.0),
            velocity=Vector3(2.0, 0.0, 0.0)  # Moving
        )

        context = GroundingContext(
            observer_position=Vector3(0.0, 0.0, 0.0)
        )

        is_salient = processor._compute_salience(moving_entity, context)
        # Movement and being an agent should make it salient
        assert is_salient


class TestBridgeConfig:
    """Test bridge configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BridgeConfig()

        assert config.host == "localhost"
        assert config.port == 9080
        assert config.heartbeat_interval_ms == 1000.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BridgeConfig(
            host="0.0.0.0",
            port=9090,
            enable_logging=False
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert not config.enable_logging


class TestIntegration:
    """Integration tests for the complete bridge system."""

    def test_full_perception_pipeline(self):
        """Test complete perception processing pipeline."""
        # Create components
        grounder = SymbolGrounder()
        processor = PerceptionProcessor(grounder)

        # Simulate multiple entities
        entities = [
            EntityUpdate(
                godot_id=3001 + i,
                entity_type="object" if i < 3 else "agent",
                name=f"Entity{i}",
                position=Vector3(i * 2.0, 0.0, i * 2.0),
                semantic_tags=["test"]
            )
            for i in range(5)
        ]

        # Create perception
        perception = AgentPerception(
            agent_godot_id=1001,
            agent_name="TestAgent",
            visible_entities=entities,
            own_position=Vector3(0.0, 0.0, 0.0)
        )

        # Process
        field = processor.process_perception(perception)

        # Verify
        assert len(field.grounded_symbols) == 5
        assert all(s.godot_id in grounder.grounded_symbols for s in field.grounded_symbols)

    def test_action_execution_flow(self):
        """Test complete action execution flow."""
        grounder = SymbolGrounder()
        executor = ActionExecutor(grounder)

        # Plan action
        action = executor.plan_action(
            agent_id=1001,
            action_type=ActionType.EXAMINE,
            target_entity_id=2001,
            reason="Investigating"
        )

        # Queue it
        executor.queue_action(action)

        # Execute (without actual sending)
        executing = executor.execute_next(1001)
        assert executing is not None
        assert executing.status == "executing"

        # Simulate result
        result = executor.handle_result(
            command_id=action.command_id,
            success=True,
            state_changes={"examined": True}
        )

        assert result is not None
        assert result.success
        assert 1001 not in executor.executing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
