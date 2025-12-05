"""
Curriculum Learning System for ToM-NAS
Progressive training from SW1 (simple) to SW4 (complex society)

Curriculum Stages:
SW1: Basic physical world - object permanence, simple actions
SW2: Two-agent interactions - simple ToM, cooperation/defection
SW3: Multi-agent groups - coalitions, communication, norms
SW4: Full society - zombies, deception, mythology, emergent language
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class CurriculumStage(Enum):
    """Training curriculum stages."""
    SW1 = 1  # Basic physical world
    SW2 = 2  # Two-agent interactions
    SW3 = 3  # Multi-agent groups
    SW4 = 4  # Full society


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    stage: CurriculumStage
    name: str
    description: str
    num_agents: int
    tom_order_required: int
    features_enabled: List[str]
    min_fitness_to_advance: float
    max_episodes: int


# Stage configurations
STAGE_CONFIGS = {
    CurriculumStage.SW1: StageConfig(
        stage=CurriculumStage.SW1,
        name="Physical World",
        description="Basic object tracking, physical causality",
        num_agents=1,
        tom_order_required=0,
        features_enabled=['object_tracking', 'causality', 'memory'],
        min_fitness_to_advance=0.6,
        max_episodes=1000
    ),
    CurriculumStage.SW2: StageConfig(
        stage=CurriculumStage.SW2,
        name="Dyadic Interactions",
        description="Two-agent ToM, cooperation/defection games",
        num_agents=2,
        tom_order_required=1,
        features_enabled=['object_tracking', 'causality', 'memory',
                         'basic_tom', 'cooperation', 'simple_communication'],
        min_fitness_to_advance=0.55,
        max_episodes=2000
    ),
    CurriculumStage.SW3: StageConfig(
        stage=CurriculumStage.SW3,
        name="Multi-Agent Groups",
        description="Coalitions, norms, reputation, complex ToM",
        num_agents=5,
        tom_order_required=2,
        features_enabled=['object_tracking', 'causality', 'memory',
                         'basic_tom', 'cooperation', 'simple_communication',
                         'coalitions', 'reputation', 'norms', 'higher_tom'],
        min_fitness_to_advance=0.5,
        max_episodes=3000
    ),
    CurriculumStage.SW4: StageConfig(
        stage=CurriculumStage.SW4,
        name="Full Society",
        description="Zombies, deception, mythology, emergent language",
        num_agents=10,
        tom_order_required=3,
        features_enabled=['object_tracking', 'causality', 'memory',
                         'basic_tom', 'cooperation', 'simple_communication',
                         'coalitions', 'reputation', 'norms', 'higher_tom',
                         'zombies', 'deception', 'mythology', 'emergent_language',
                         'resource_competition', 'cultural_transmission'],
        min_fitness_to_advance=0.45,
        max_episodes=5000
    )
}


class CurriculumManager:
    """
    Manages curriculum progression and environment adaptation.
    """

    def __init__(self, start_stage: CurriculumStage = CurriculumStage.SW1):
        self.current_stage = start_stage
        self.configs = STAGE_CONFIGS
        self.stage_history: List[Dict] = []
        self.episodes_in_stage = 0
        self.best_fitness_in_stage = 0.0

    def get_current_config(self) -> StageConfig:
        """Get current stage configuration."""
        return self.configs[self.current_stage]

    def should_advance(self, fitness: float) -> bool:
        """Check if agent should advance to next stage."""
        config = self.get_current_config()

        # Update tracking
        self.best_fitness_in_stage = max(self.best_fitness_in_stage, fitness)
        self.episodes_in_stage += 1

        # Check advancement criteria
        if self.best_fitness_in_stage >= config.min_fitness_to_advance:
            return True

        # Force advance after max episodes (prevent getting stuck)
        if self.episodes_in_stage >= config.max_episodes:
            return True

        return False

    def advance_stage(self) -> Optional[CurriculumStage]:
        """Advance to next curriculum stage."""
        current_value = self.current_stage.value

        if current_value >= 4:  # Already at SW4
            return None

        # Record stage completion
        self.stage_history.append({
            'stage': self.current_stage.name,
            'episodes': self.episodes_in_stage,
            'best_fitness': self.best_fitness_in_stage
        })

        # Advance
        self.current_stage = CurriculumStage(current_value + 1)
        self.episodes_in_stage = 0
        self.best_fitness_in_stage = 0.0

        return self.current_stage

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage."""
        config = self.get_current_config()

        env_config = {
            'num_agents': config.num_agents,
            'features': config.features_enabled,
            'tom_order': config.tom_order_required,
            'complexity': config.stage.value / 4.0
        }

        # Stage-specific settings
        if config.stage == CurriculumStage.SW1:
            env_config['world_type'] = 'physical'
            env_config['objects'] = ['ball', 'box', 'basket']
            env_config['actions'] = ['move', 'observe', 'remember']

        elif config.stage == CurriculumStage.SW2:
            env_config['world_type'] = 'dyadic'
            env_config['games'] = ['cooperation', 'communication']
            env_config['actions'] = ['cooperate', 'defect', 'communicate']

        elif config.stage == CurriculumStage.SW3:
            env_config['world_type'] = 'group'
            env_config['games'] = ['cooperation', 'communication', 'coalition']
            env_config['actions'] = ['cooperate', 'defect', 'communicate',
                                    'join_coalition', 'leave_coalition', 'share']

        elif config.stage == CurriculumStage.SW4:
            env_config['world_type'] = 'society'
            env_config['games'] = ['cooperation', 'communication', 'coalition',
                                  'zombie_detection', 'deception', 'mythology']
            env_config['actions'] = ['cooperate', 'defect', 'communicate',
                                    'join_coalition', 'leave_coalition', 'share',
                                    'detect_zombie', 'deceive', 'create_myth']
            env_config['zombie_ratio'] = 0.2

        return env_config

    def get_fitness_weights(self) -> Dict[str, float]:
        """Get fitness component weights for current stage."""
        stage_value = self.current_stage.value

        # Base weights (evolve with stage)
        weights = {
            'survival': 0.3 - 0.05 * stage_value,
            'resource_accumulation': 0.2,
            'cooperation': 0.1 + 0.05 * stage_value,
            'tom_accuracy': 0.1 + 0.1 * stage_value,
            'belief_consistency': 0.1,
            'zombie_detection': 0.0 if stage_value < 4 else 0.15,
            'communication': 0.0 if stage_value < 2 else 0.1,
            'coalition_success': 0.0 if stage_value < 3 else 0.1
        }

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get curriculum progress summary."""
        return {
            'current_stage': self.current_stage.name,
            'stage_value': self.current_stage.value,
            'episodes_in_stage': self.episodes_in_stage,
            'best_fitness': self.best_fitness_in_stage,
            'threshold_to_advance': self.get_current_config().min_fitness_to_advance,
            'history': self.stage_history
        }


class AdaptiveCurriculum(CurriculumManager):
    """
    Adaptive curriculum that adjusts difficulty based on agent performance.
    """

    def __init__(self, start_stage: CurriculumStage = CurriculumStage.SW1):
        super().__init__(start_stage)
        self.difficulty_multiplier = 1.0
        self.recent_fitness: List[float] = []
        self.adaptation_window = 50

    def update(self, fitness: float):
        """Update curriculum based on fitness."""
        self.recent_fitness.append(fitness)
        if len(self.recent_fitness) > self.adaptation_window:
            self.recent_fitness.pop(0)

        # Adapt difficulty
        if len(self.recent_fitness) >= self.adaptation_window:
            mean_fitness = np.mean(self.recent_fitness)
            std_fitness = np.std(self.recent_fitness)

            # If doing too well, increase difficulty
            if mean_fitness > 0.7:
                self.difficulty_multiplier = min(2.0, self.difficulty_multiplier * 1.1)

            # If struggling, decrease difficulty
            elif mean_fitness < 0.3:
                self.difficulty_multiplier = max(0.5, self.difficulty_multiplier * 0.9)

        # Check for stage advancement
        if self.should_advance(fitness):
            self.advance_stage()

    def get_adjusted_config(self) -> Dict[str, Any]:
        """Get configuration adjusted for current difficulty."""
        base_config = self.get_environment_config()

        # Adjust based on difficulty multiplier
        if self.difficulty_multiplier > 1.2:
            # Harder: more agents, stricter thresholds
            base_config['num_agents'] = int(base_config['num_agents'] * 1.2)
            base_config['difficulty'] = 'hard'

        elif self.difficulty_multiplier < 0.8:
            # Easier: fewer agents, more forgiving
            base_config['num_agents'] = max(1, int(base_config['num_agents'] * 0.8))
            base_config['difficulty'] = 'easy'

        else:
            base_config['difficulty'] = 'normal'

        return base_config


class TrainingScheduler:
    """
    Manages training schedule across curriculum stages.
    """

    def __init__(self, curriculum: CurriculumManager):
        self.curriculum = curriculum
        self.total_episodes = 0
        self.learning_rate_schedule: Dict[int, float] = {
            1: 0.001,   # SW1
            2: 0.0008,  # SW2
            3: 0.0005,  # SW3
            4: 0.0003   # SW4
        }

    def get_learning_rate(self) -> float:
        """Get learning rate for current stage."""
        stage_value = self.curriculum.current_stage.value
        return self.learning_rate_schedule.get(stage_value, 0.0001)

    def get_batch_size(self) -> int:
        """Get batch size for current stage."""
        stage_value = self.curriculum.current_stage.value
        # Larger batches for later stages (more complex, need stability)
        return 32 * stage_value

    def get_training_params(self) -> Dict[str, Any]:
        """Get all training parameters for current stage."""
        config = self.curriculum.get_current_config()

        return {
            'learning_rate': self.get_learning_rate(),
            'batch_size': self.get_batch_size(),
            'episodes_per_update': 10 * config.stage.value,
            'gradient_clip': 1.0,
            'entropy_coef': 0.01 / config.stage.value,
            'value_coef': 0.5,
            'max_grad_norm': 0.5
        }

    def step(self, fitness: float):
        """Step the scheduler."""
        self.total_episodes += 1

        # Check curriculum advancement
        if self.curriculum.should_advance(fitness):
            old_stage = self.curriculum.current_stage
            new_stage = self.curriculum.advance_stage()

            if new_stage:
                print(f"Curriculum advanced: {old_stage.name} -> {new_stage.name}")


class CurriculumDataGenerator:
    """
    Generates training data appropriate for current curriculum stage.
    """

    def __init__(self, curriculum: CurriculumManager, input_dim: int = 191,
                 device: str = 'cpu'):
        self.curriculum = curriculum
        self.input_dim = input_dim
        self.device = device

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict]:
        """Generate training batch for current stage."""
        stage = self.curriculum.current_stage

        if stage == CurriculumStage.SW1:
            return self._generate_sw1_batch(batch_size)
        elif stage == CurriculumStage.SW2:
            return self._generate_sw2_batch(batch_size)
        elif stage == CurriculumStage.SW3:
            return self._generate_sw3_batch(batch_size)
        else:  # SW4
            return self._generate_sw4_batch(batch_size)

    def _generate_sw1_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict]:
        """SW1: Physical world scenarios."""
        seq_len = 5
        data = torch.zeros(batch_size, seq_len, self.input_dim, device=self.device)

        # Simple object tracking scenarios
        for b in range(batch_size):
            # Object position changes
            obj_pos = np.random.randint(0, 5)
            data[b, :, obj_pos] = 1.0

            # Movement at random timestep
            move_t = np.random.randint(1, seq_len)
            new_pos = (obj_pos + 1) % 5
            data[b, move_t:, obj_pos] = 0.0
            data[b, move_t:, new_pos] = 1.0

        targets = {
            'type': 'object_tracking',
            'task': 'predict_location'
        }

        return data, targets

    def _generate_sw2_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict]:
        """SW2: Two-agent interaction scenarios."""
        seq_len = 6
        data = torch.zeros(batch_size, seq_len, self.input_dim, device=self.device)

        for b in range(batch_size):
            # Agent 1 state
            data[b, :, 0] = 1.0  # Agent 1 present

            # Agent 2 state
            data[b, :, 1] = 1.0  # Agent 2 present

            # Interaction scenario (cooperation game)
            action_1 = np.random.choice([0, 1])  # cooperate/defect
            action_2 = np.random.choice([0, 1])

            data[b, 3:, 2] = action_1  # Agent 1 action
            data[b, 3:, 3] = action_2  # Agent 2 action

            # False belief scenario (Sally-Anne style)
            # Agent 2 leaves, state changes, Agent 2 returns
            data[b, 2, 4] = 1.0  # State change marker
            data[b, 2, 1] = 0.0  # Agent 2 leaves
            data[b, 4, 1] = 1.0  # Agent 2 returns

        targets = {
            'type': 'dyadic_interaction',
            'task': 'predict_belief'
        }

        return data, targets

    def _generate_sw3_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict]:
        """SW3: Multi-agent group scenarios."""
        seq_len = 8
        data = torch.zeros(batch_size, seq_len, self.input_dim, device=self.device)

        for b in range(batch_size):
            num_agents = 5

            # Agent presence
            for a in range(num_agents):
                data[b, :, a] = 1.0

            # Coalition formation
            coalition_members = np.random.choice(num_agents, size=3, replace=False)
            for m in coalition_members:
                data[b, 4:, 10 + m] = 1.0  # Coalition membership

            # Reputation dynamics
            for a in range(num_agents):
                rep = np.random.rand()
                data[b, :, 20 + a] = rep

        targets = {
            'type': 'group_dynamics',
            'task': 'predict_coalition'
        }

        return data, targets

    def _generate_sw4_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict]:
        """SW4: Full society scenarios."""
        seq_len = 10
        data = torch.zeros(batch_size, seq_len, self.input_dim, device=self.device)

        for b in range(batch_size):
            num_agents = 10
            num_zombies = 2

            # Agent presence
            for a in range(num_agents):
                data[b, :, a] = 1.0

            # Mark zombies
            zombies = np.random.choice(num_agents, size=num_zombies, replace=False)
            for z in zombies:
                data[b, :, 30 + z] = 1.0  # Zombie marker (hidden from agents)

            # Complex social dynamics
            # Coalitions
            for c in range(3):
                members = np.random.choice(num_agents, size=3, replace=False)
                for m in members:
                    data[b, 5:, 50 + c * 10 + m] = 1.0

            # Communication events
            comm_times = np.random.choice(seq_len, size=3, replace=False)
            for t in comm_times:
                sender = np.random.randint(0, num_agents)
                data[b, t, 80 + sender] = 1.0

        targets = {
            'type': 'full_society',
            'task': 'zombie_detection',
            'zombies': zombies.tolist() if 'zombies' in dir() else []
        }

        return data, targets


def create_curriculum_trainer(model: nn.Module, start_stage: CurriculumStage = CurriculumStage.SW1,
                             device: str = 'cpu') -> Dict[str, Any]:
    """Create a complete curriculum training setup."""
    curriculum = AdaptiveCurriculum(start_stage)
    scheduler = TrainingScheduler(curriculum)
    data_gen = CurriculumDataGenerator(curriculum, device=device)

    return {
        'curriculum': curriculum,
        'scheduler': scheduler,
        'data_generator': data_gen,
        'model': model
    }
