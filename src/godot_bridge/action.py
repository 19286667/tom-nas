"""
Action Executor - From Cognition to Physics

Translates cognitive intents (from Mentalese) into Godot commands
that execute in the physical simulation.

The Action Pipeline:
1. IntentBlock from cognitive system
2. Action planning (affordance checking)
3. Command generation (AgentCommand)
4. Execution in Godot
5. Result verification

This closes the sensorimotor loop - perception leads to
cognition leads to action leads to new perception.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .protocol import AgentCommand, GodotMessage, MessageType, Vector3
from .symbol_grounding import GroundedSymbol, SymbolGrounder


class ActionType(Enum):
    """Types of actions agents can perform."""

    # Movement
    MOVE_TO = auto()
    TURN_TO = auto()
    FOLLOW = auto()
    FLEE = auto()

    # Object interaction
    PICK_UP = auto()
    PUT_DOWN = auto()
    USE = auto()
    EXAMINE = auto()
    GIVE = auto()

    # Social
    SPEAK = auto()
    GESTURE = auto()
    LOOK_AT = auto()

    # Meta
    WAIT = auto()
    CANCEL = auto()


@dataclass
class GodotAction:
    """
    A planned action to be executed in Godot.
    """

    action_type: ActionType
    agent_godot_id: int

    # Target
    target_entity_id: Optional[int] = None
    target_position: Optional[Vector3] = None
    target_agent_id: Optional[int] = None

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    utterance: Optional[str] = None

    # Intent link (traceability)
    source_intent_id: Optional[str] = None
    reason: str = ""

    # Execution control
    priority: int = 0
    interruptible: bool = True
    timeout_seconds: float = 10.0

    # State
    command_id: str = ""
    status: str = "pending"  # pending, executing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_command(self) -> AgentCommand:
        """Convert to AgentCommand for Godot."""
        command_type_map = {
            ActionType.MOVE_TO: "move",
            ActionType.TURN_TO: "turn",
            ActionType.FOLLOW: "follow",
            ActionType.FLEE: "flee",
            ActionType.PICK_UP: "pick_up",
            ActionType.PUT_DOWN: "put_down",
            ActionType.USE: "use",
            ActionType.EXAMINE: "examine",
            ActionType.GIVE: "give",
            ActionType.SPEAK: "speak",
            ActionType.GESTURE: "gesture",
            ActionType.LOOK_AT: "look_at",
            ActionType.WAIT: "wait",
            ActionType.CANCEL: "cancel",
        }

        return AgentCommand(
            agent_godot_id=self.agent_godot_id,
            command_type=command_type_map.get(self.action_type, "wait"),
            target_position=self.target_position,
            target_entity_id=self.target_entity_id or self.target_agent_id,
            utterance_text=self.utterance,
            priority=self.priority,
            interruptible=self.interruptible,
            timeout_seconds=self.timeout_seconds,
            command_id=self.command_id,
            reason=self.reason,
        )


@dataclass
class ActionResult:
    """
    Result of executing an action in Godot.
    """

    action: GodotAction
    success: bool = False
    error_message: Optional[str] = None

    # State changes
    state_changes: Dict[str, Any] = field(default_factory=dict)
    affected_entities: List[int] = field(default_factory=list)

    # Timing
    execution_time_ms: float = 0.0

    # Feedback for learning
    was_expected: bool = True  # Did outcome match prediction?
    prediction_error: float = 0.0


class ActionExecutor:
    """
    Executes cognitive actions in the Godot simulation.

    Handles:
    - Action planning and affordance checking
    - Command generation and sending
    - Execution monitoring and result handling
    - Action queue management
    """

    def __init__(self, symbol_grounder: SymbolGrounder, send_callback: Optional[Callable[[GodotMessage], None]] = None):
        """
        Initialize action executor.

        Args:
            symbol_grounder: Symbol grounding system
            send_callback: Function to send messages to Godot
        """
        self.grounder = symbol_grounder
        self.send_callback = send_callback

        # Action queues per agent
        self.action_queues: Dict[int, List[GodotAction]] = {}

        # Current executing action per agent
        self.executing: Dict[int, GodotAction] = {}

        # Pending results (awaiting confirmation)
        self.pending_results: Dict[str, GodotAction] = {}

        # Command counter for IDs
        self.command_counter = 0

        # Action history
        self.action_history: List[ActionResult] = []

    def plan_action(
        self,
        agent_id: int,
        action_type: ActionType,
        target_entity_id: Optional[int] = None,
        target_position: Optional[Vector3] = None,
        parameters: Optional[Dict[str, Any]] = None,
        intent_id: Optional[str] = None,
        reason: str = "",
    ) -> Optional[GodotAction]:
        """
        Plan an action, checking affordances.

        Args:
            agent_id: Godot ID of acting agent
            action_type: Type of action
            target_entity_id: Target entity Godot ID
            target_position: Target position
            parameters: Additional parameters
            intent_id: Source intent block ID
            reason: Why this action

        Returns:
            GodotAction if valid, None if not possible
        """
        # Check affordances if targeting an entity
        if target_entity_id:
            symbol = self.grounder.get_grounded_symbol(target_entity_id)
            if symbol:
                if not self._check_affordance(action_type, symbol):
                    return None  # Action not afforded by target

        # Create action
        self.command_counter += 1
        action = GodotAction(
            action_type=action_type,
            agent_godot_id=agent_id,
            target_entity_id=target_entity_id,
            target_position=target_position,
            parameters=parameters or {},
            source_intent_id=intent_id,
            reason=reason,
            command_id=f"cmd_{agent_id}_{self.command_counter}",
        )

        return action

    def _check_affordance(self, action_type: ActionType, symbol: GroundedSymbol) -> bool:
        """Check if an action is afforded by a symbol."""
        affordance_map = {
            ActionType.PICK_UP: "can_pick_up",
            ActionType.USE: "can_use",
            ActionType.EXAMINE: None,  # Always allowed
            ActionType.MOVE_TO: None,  # Always allowed for locations
        }

        required = affordance_map.get(action_type)
        if required is None:
            return True

        return required in symbol.physical_affordances

    def queue_action(self, action: GodotAction):
        """Add action to agent's queue."""
        agent_id = action.agent_godot_id

        if agent_id not in self.action_queues:
            self.action_queues[agent_id] = []

        # Insert by priority
        queue = self.action_queues[agent_id]
        inserted = False
        for i, queued in enumerate(queue):
            if action.priority > queued.priority:
                queue.insert(i, action)
                inserted = True
                break

        if not inserted:
            queue.append(action)

    def execute_next(self, agent_id: int) -> Optional[GodotAction]:
        """
        Execute next action in agent's queue.

        Returns:
            The action being executed, or None if queue empty
        """
        # Check if already executing
        if agent_id in self.executing:
            return self.executing[agent_id]

        # Get next action
        queue = self.action_queues.get(agent_id, [])
        if not queue:
            return None

        action = queue.pop(0)
        action.status = "executing"
        action.started_at = datetime.now()

        self.executing[agent_id] = action
        self.pending_results[action.command_id] = action

        # Send to Godot
        if self.send_callback:
            command = action.to_command()
            message = GodotMessage.create_agent_command(command)
            self.send_callback(message)

        return action

    def cancel_action(self, agent_id: int, reason: str = "") -> bool:
        """Cancel currently executing action for an agent."""
        if agent_id not in self.executing:
            return False

        action = self.executing[agent_id]
        if not action.interruptible:
            return False

        action.status = "cancelled"
        action.completed_at = datetime.now()

        del self.executing[agent_id]
        self.pending_results.pop(action.command_id, None)

        # Send cancel to Godot
        if self.send_callback:
            cancel_action = GodotAction(
                action_type=ActionType.CANCEL,
                agent_godot_id=agent_id,
                reason=reason,
            )
            message = GodotMessage.create_agent_command(cancel_action.to_command())
            self.send_callback(message)

        return True

    def handle_result(
        self, command_id: str, success: bool, state_changes: Dict[str, Any] = None, error: str = None
    ) -> Optional[ActionResult]:
        """
        Handle result from Godot after action execution.

        Args:
            command_id: ID of completed command
            success: Whether action succeeded
            state_changes: Changes in world state
            error: Error message if failed

        Returns:
            ActionResult if action was pending
        """
        if command_id not in self.pending_results:
            return None

        action = self.pending_results.pop(command_id)
        action.status = "completed" if success else "failed"
        action.completed_at = datetime.now()

        # Calculate execution time
        if action.started_at:
            execution_time = (action.completed_at - action.started_at).total_seconds() * 1000
        else:
            execution_time = 0.0

        result = ActionResult(
            action=action,
            success=success,
            error_message=error,
            state_changes=state_changes or {},
            execution_time_ms=execution_time,
        )

        # Remove from executing
        agent_id = action.agent_godot_id
        if agent_id in self.executing:
            if self.executing[agent_id].command_id == command_id:
                del self.executing[agent_id]

        self.action_history.append(result)

        return result

    def create_from_intent(self, intent, agent_id: int) -> Optional[GodotAction]:
        """
        Create an action from a Mentalese IntentBlock.

        Args:
            intent: IntentBlock from cognitive system
            agent_id: Godot ID of acting agent

        Returns:
            GodotAction if translatable
        """
        # Map intent action types to Godot actions
        action_map = {
            "move": ActionType.MOVE_TO,
            "pick_up": ActionType.PICK_UP,
            "put_down": ActionType.PUT_DOWN,
            "use": ActionType.USE,
            "examine": ActionType.EXAMINE,
            "speak": ActionType.SPEAK,
            "give": ActionType.GIVE,
            "follow": ActionType.FOLLOW,
            "flee": ActionType.FLEE,
        }

        action_type_str = getattr(intent, "action_type", "wait")
        action_type = action_map.get(action_type_str.lower(), ActionType.WAIT)

        # Get target from intent
        target_entity = getattr(intent, "target_entity", None)
        target_agent = getattr(intent, "target_agent", None)

        target_entity_id = None
        if target_entity:
            # Look up in symbol grounder
            symbol = self.grounder.get_by_semantic_id(target_entity)
            if symbol:
                target_entity_id = symbol.godot_id

        return self.plan_action(
            agent_id=agent_id,
            action_type=action_type,
            target_entity_id=target_entity_id or None,
            intent_id=getattr(intent, "block_id", None),
            reason=getattr(intent, "motivation", ""),
        )

    def get_agent_status(self, agent_id: int) -> Dict[str, Any]:
        """Get action status for an agent."""
        return {
            "queue_length": len(self.action_queues.get(agent_id, [])),
            "is_executing": agent_id in self.executing,
            "current_action": self.executing[agent_id].action_type.name if agent_id in self.executing else None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        total_actions = len(self.action_history)
        successful = sum(1 for r in self.action_history if r.success)

        return {
            "total_actions": total_actions,
            "successful": successful,
            "success_rate": successful / total_actions if total_actions > 0 else 0.0,
            "agents_executing": len(self.executing),
            "pending_commands": len(self.pending_results),
            "total_queued": sum(len(q) for q in self.action_queues.values()),
        }
