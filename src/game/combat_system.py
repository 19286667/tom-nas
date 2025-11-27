"""
Psychological Combat System

Combat that targets both body and mind.
Uses Theory of Mind to discover and exploit psychological vulnerabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.ontology import SoulMapOntology

# ============================================================================
# Combat Enums
# ============================================================================

class DamageType(Enum):
    """Type of damage dealt"""
    PHYSICAL = "physical"
    PSYCHOLOGICAL = "psychological"
    HYBRID = "hybrid"

class DefenseType(Enum):
    """Type of defense mechanism"""
    RATIONALIZATION = "rationalization"
    DENIAL = "denial"
    PROJECTION = "projection"
    SUBLIMATION = "sublimation"
    REPRESSION = "repression"

# ============================================================================
# Combat Actions
# ============================================================================

@dataclass
class CombatAction:
    """A combat action (attack or defense)"""
    action_id: str
    name: str
    damage_type: DamageType
    base_damage: float

    # Physical component
    physical_damage: float = 0.0

    # Psychological component
    targeted_dimensions: List[str] = None  # Soul Map dimensions to target
    psychological_damage: float = 0.0

    # ToM requirements
    requires_tom_order: int = 0  # Minimum ToM order needed
    vulnerability_bonus: float = 2.0  # Multiplier if vulnerability detected

    # Costs
    stamina_cost: float = 10.0
    mental_energy_cost: float = 0.0

    # Description
    description: str = ""

@dataclass
class CombatEffect:
    """Effect applied after a combat action"""
    effect_id: str
    name: str
    duration: int  # Turns
    soul_map_delta: Dict[str, float]  # Ongoing changes
    stat_modifiers: Dict[str, float]  # ACC, EVA, etc.

# ============================================================================
# Combatant
# ============================================================================

class Combatant:
    """A participant in combat (player or NPC)"""

    def __init__(
        self,
        combatant_id: str,
        name: str,
        soul_map: torch.Tensor,
        ontology: SoulMapOntology,
        max_hp: float = 100.0,
        max_stamina: float = 100.0
    ):
        self.combatant_id = combatant_id
        self.name = name
        self.soul_map = soul_map.clone()
        self.ontology = ontology

        # Physical stats
        self.max_hp = max_hp
        self.current_hp = max_hp
        self.max_stamina = max_stamina
        self.current_stamina = max_stamina

        # Psychological stats
        self.coherence = 1.0  # 0 = shattered psyche, 1 = stable
        self.mental_energy = 100.0

        # Combat state
        self.active_effects: List[CombatEffect] = []
        self.vulnerabilities: List[str] = []
        self.resistances: List[str] = []

        # ToM capability
        self.tom_order = 1  # Default first-order ToM

        # Analyze initial state
        self._analyze_vulnerabilities()

    def _analyze_vulnerabilities(self):
        """Identify psychological vulnerabilities based on Soul Map"""

        self.vulnerabilities = []

        # Check for extreme values (unstable dimensions)
        for i, dim in enumerate(self.ontology.dimensions):
            if i >= len(self.soul_map):
                break

            value = float(self.soul_map[i])

            # Extreme low or high values are vulnerabilities
            if value < 0.2 or value > 0.8:
                self.vulnerabilities.append(dim.name)

        # Check for contradictions (cognitive dissonance)
        # Example: high moral.justice but low moral.compassion
        # This creates vulnerability to "hypocrisy" attacks

    def take_damage(
        self,
        physical: float,
        psychological: Dict[str, float],
        attacker_tom_order: int = 1
    ) -> Dict[str, Any]:
        """
        Take damage from an attack.

        Returns dict with damage report and effects.
        """

        report = {
            'physical_damage_dealt': 0.0,
            'psychological_damage_dealt': {},
            'vulnerabilities_hit': [],
            'critical_hit': False,
            'total_damage': 0.0
        }

        # Apply physical damage
        if physical > 0:
            actual_damage = physical * self._calculate_physical_resistance()
            self.current_hp -= actual_damage
            report['physical_damage_dealt'] = actual_damage

        # Apply psychological damage
        if psychological:
            for dim_name, damage in psychological.items():
                # Check if this is a vulnerability
                is_vulnerable = dim_name in self.vulnerabilities

                # Calculate actual damage
                multiplier = 2.0 if is_vulnerable else 1.0
                actual_damage = damage * multiplier

                # Apply to soul map
                if dim_name in self.ontology.name_to_idx:
                    idx = self.ontology.name_to_idx[dim_name]
                    self.soul_map[idx] -= actual_damage
                    self.soul_map[idx] = torch.clamp(self.soul_map[idx], 0, 1)

                    report['psychological_damage_dealt'][dim_name] = actual_damage

                    if is_vulnerable:
                        report['vulnerabilities_hit'].append(dim_name)
                        report['critical_hit'] = True

                # Reduce coherence
                self.coherence -= actual_damage * 0.1
                self.coherence = max(0.0, self.coherence)

        # Calculate total damage equivalent
        report['total_damage'] = report['physical_damage_dealt'] + sum(
            report['psychological_damage_dealt'].values()
        ) * 10  # Psych damage weighs more

        # Check for incapacitation
        if self.current_hp <= 0 or self.coherence <= 0:
            report['incapacitated'] = True
        else:
            report['incapacitated'] = False

        return report

    def use_defense(self, defense_type: DefenseType) -> Dict[str, Any]:
        """Use a psychological defense mechanism"""

        defenses = {
            DefenseType.RATIONALIZATION: {
                'description': "Rationalize the attack away",
                'coherence_restore': 0.1,
                'soul_map_delta': {'cognitive.rationality': 0.05},
                'cost': 15.0
            },
            DefenseType.DENIAL: {
                'description': "Deny the psychological impact",
                'coherence_restore': 0.15,
                'soul_map_delta': {'affect.defensiveness': 0.1},
                'cost': 20.0
            },
            DefenseType.PROJECTION: {
                'description': "Project the attack back at opponent",
                'coherence_restore': 0.05,
                'soul_map_delta': {'social.aggression': 0.1},
                'cost': 25.0
            }
        }

        if defense_type not in defenses:
            return {'success': False}

        defense = defenses[defense_type]

        # Check mental energy
        if self.mental_energy < defense['cost']:
            return {'success': False, 'reason': 'insufficient_mental_energy'}

        # Apply defense
        self.coherence += defense['coherence_restore']
        self.coherence = min(1.0, self.coherence)

        self.mental_energy -= defense['cost']

        # Apply soul map changes
        for dim_name, delta in defense['soul_map_delta'].items():
            if dim_name in self.ontology.name_to_idx:
                idx = self.ontology.name_to_idx[dim_name]
                self.soul_map[idx] += delta
                self.soul_map[idx] = torch.clamp(self.soul_map[idx], 0, 1)

        return {
            'success': True,
            'defense_type': defense_type.value,
            'description': defense['description'],
            'coherence_restored': defense['coherence_restore']
        }

    def _calculate_physical_resistance(self) -> float:
        """Calculate physical damage resistance based on psychological state"""

        # Fear reduces resistance
        # Confidence increases resistance

        # Simplified - would use actual soul map values
        base_resistance = 1.0

        # Check arousal/stress
        arousal = float(self.soul_map[2]) if len(self.soul_map) > 2 else 0.5

        if arousal > 0.8:  # High stress reduces resistance
            base_resistance *= 1.2  # Takes more damage
        elif arousal < 0.3:  # Calm increases resistance
            base_resistance *= 0.8  # Takes less damage

        return base_resistance

    def is_alive(self) -> bool:
        """Check if combatant is still able to fight"""
        return self.current_hp > 0 and self.coherence > 0.2

    def get_status(self) -> Dict[str, Any]:
        """Get current combat status"""
        return {
            'name': self.name,
            'hp': f"{self.current_hp:.1f}/{self.max_hp}",
            'hp_percent': self.current_hp / self.max_hp,
            'stamina': f"{self.current_stamina:.1f}/{self.max_stamina}",
            'coherence': f"{self.coherence:.2f}",
            'mental_energy': f"{self.mental_energy:.1f}",
            'vulnerabilities': self.vulnerabilities,
            'active_effects': len(self.active_effects),
            'status': 'alive' if self.is_alive() else 'incapacitated'
        }

# ============================================================================
# Combat System
# ============================================================================

class CombatSystem:
    """
    Manages ToM-driven combat encounters.

    Features:
    - Physical + Psychological damage
    - Vulnerability detection via ToM
    - Defense mechanisms
    - Status effects
    - Turn-based combat flow
    """

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology
        self.combat_actions = self._build_combat_actions()
        self.active_combats: Dict[str, 'Combat'] = {}

    def start_combat(
        self,
        combat_id: str,
        combatants: List[Combatant]
    ) -> 'Combat':
        """Start a new combat encounter"""

        combat = Combat(
            combat_id=combat_id,
            combatants=combatants,
            ontology=self.ontology
        )

        self.active_combats[combat_id] = combat

        return combat

    def _build_combat_actions(self) -> Dict[str, CombatAction]:
        """Build library of combat actions"""

        actions = {}

        # Physical attacks
        actions['strike'] = CombatAction(
            action_id='strike',
            name="Strike",
            damage_type=DamageType.PHYSICAL,
            base_damage=15.0,
            physical_damage=15.0,
            stamina_cost=10.0,
            description="A basic physical attack"
        )

        # Psychological attacks
        actions['intimidate'] = CombatAction(
            action_id='intimidate',
            name="Intimidate",
            damage_type=DamageType.PSYCHOLOGICAL,
            base_damage=0.2,
            targeted_dimensions=['affect.fear', 'affect.dominance'],
            psychological_damage=0.2,
            requires_tom_order=1,
            mental_energy_cost=15.0,
            description="Attempt to instill fear in the opponent"
        )

        actions['reveal_hypocrisy'] = CombatAction(
            action_id='reveal_hypocrisy',
            name="Reveal Hypocrisy",
            damage_type=DamageType.PSYCHOLOGICAL,
            base_damage=0.3,
            targeted_dimensions=['moral.integrity', 'cognitive.coherence'],
            psychological_damage=0.3,
            requires_tom_order=2,
            vulnerability_bonus=3.0,
            mental_energy_cost=25.0,
            description="Point out contradictions in opponent's beliefs"
        )

        actions['shatter_worldview'] = CombatAction(
            action_id='shatter_worldview',
            name="Shatter Worldview",
            damage_type=DamageType.PSYCHOLOGICAL,
            base_damage=0.5,
            targeted_dimensions=['existential.meaning', 'narrative.identity'],
            psychological_damage=0.5,
            requires_tom_order=3,
            vulnerability_bonus=4.0,
            mental_energy_cost=40.0,
            description="Attack the core of opponent's identity and beliefs"
        )

        # Hybrid attacks
        actions['brutal_truth'] = CombatAction(
            action_id='brutal_truth',
            name="Brutal Truth",
            damage_type=DamageType.HYBRID,
            base_damage=20.0,
            physical_damage=10.0,
            targeted_dimensions=['affect.pride', 'social.status'],
            psychological_damage=0.25,
            requires_tom_order=2,
            stamina_cost=15.0,
            mental_energy_cost=20.0,
            description="A harsh physical blow accompanied by cutting words"
        )

        return actions

# ============================================================================
# Combat Encounter
# ============================================================================

class Combat:
    """A single combat encounter"""

    def __init__(
        self,
        combat_id: str,
        combatants: List[Combatant],
        ontology: SoulMapOntology
    ):
        self.combat_id = combat_id
        self.combatants = combatants
        self.ontology = ontology

        # Combat state
        self.turn = 0
        self.turn_order = combatants.copy()
        self.combat_log: List[Dict[str, Any]] = []
        self.is_active = True

    def execute_action(
        self,
        attacker_id: str,
        defender_id: str,
        action: CombatAction
    ) -> Dict[str, Any]:
        """Execute a combat action"""

        # Find combatants
        attacker = self._get_combatant(attacker_id)
        defender = self._get_combatant(defender_id)

        if not attacker or not defender:
            return {'error': 'Combatant not found'}

        # Check costs
        if attacker.current_stamina < action.stamina_cost:
            return {'error': 'Insufficient stamina'}

        if attacker.mental_energy < action.mental_energy_cost:
            return {'error': 'Insufficient mental energy'}

        # Pay costs
        attacker.current_stamina -= action.stamina_cost
        attacker.mental_energy -= action.mental_energy_cost

        # Calculate damage
        physical_damage = action.physical_damage
        psychological_damage = {}

        if action.targeted_dimensions:
            # Use ToM to analyze vulnerabilities
            vulnerabilities_found = self._analyze_vulnerabilities(
                attacker,
                defender,
                action.targeted_dimensions
            )

            for dim in action.targeted_dimensions:
                base_dmg = action.psychological_damage / len(action.targeted_dimensions)

                # Bonus for hitting vulnerability
                if dim in vulnerabilities_found:
                    base_dmg *= action.vulnerability_bonus

                psychological_damage[dim] = base_dmg

        # Apply damage
        damage_report = defender.take_damage(
            physical=physical_damage,
            psychological=psychological_damage,
            attacker_tom_order=attacker.tom_order
        )

        # Create combat log entry
        log_entry = {
            'turn': self.turn,
            'attacker': attacker.name,
            'defender': defender.name,
            'action': action.name,
            'damage_report': damage_report
        }

        self.combat_log.append(log_entry)

        # Check for combat end
        if not defender.is_alive():
            self.is_active = False
            log_entry['combat_end'] = f"{defender.name} has been defeated!"

        return log_entry

    def _analyze_vulnerabilities(
        self,
        attacker: Combatant,
        defender: Combatant,
        targeted_dimensions: List[str]
    ) -> List[str]:
        """Use ToM to find vulnerabilities in targeted dimensions"""

        # Attacker with higher ToM can better identify vulnerabilities
        if attacker.tom_order < 1:
            return []  # Can't detect vulnerabilities

        # Check which targeted dimensions are actually vulnerabilities
        found = []
        for dim in targeted_dimensions:
            if dim in defender.vulnerabilities:
                # Higher ToM = higher chance to detect
                detection_chance = 0.5 + (attacker.tom_order * 0.15)
                if np.random.random() < detection_chance:
                    found.append(dim)

        return found

    def _get_combatant(self, combatant_id: str) -> Optional[Combatant]:
        """Get combatant by ID"""
        for c in self.combatants:
            if c.combatant_id == combatant_id:
                return c
        return None

    def next_turn(self):
        """Advance to next turn"""
        self.turn += 1

        # Regenerate resources
        for combatant in self.combatants:
            if combatant.is_alive():
                # Stamina regen
                combatant.current_stamina = min(
                    combatant.max_stamina,
                    combatant.current_stamina + 10
                )

                # Mental energy regen
                combatant.mental_energy = min(
                    100.0,
                    combatant.mental_energy + 5
                )

    def get_combat_status(self) -> Dict[str, Any]:
        """Get full combat status"""
        return {
            'combat_id': self.combat_id,
            'turn': self.turn,
            'is_active': self.is_active,
            'combatants': [c.get_status() for c in self.combatants],
            'recent_log': self.combat_log[-5:] if self.combat_log else []
        }

# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Psychological Combat System Demo")
    print("=" * 60)

    # Create ontology
    ontology = SoulMapOntology()

    # Create combatants
    # Player: balanced
    player_soul_map = ontology.get_default_state()

    # Enemy: fearful (vulnerability to intimidation)
    enemy_soul_map = ontology.get_default_state()
    enemy_soul_map[5] = 0.8  # High fear

    player = Combatant(
        combatant_id="player",
        name="Player",
        soul_map=player_soul_map,
        ontology=ontology,
        max_hp=100.0
    )
    player.tom_order = 2  # Second-order ToM

    enemy = Combatant(
        combatant_id="enemy",
        name="Shadow",
        soul_map=enemy_soul_map,
        ontology=ontology,
        max_hp=80.0
    )

    # Start combat
    combat_system = CombatSystem(ontology)
    combat = combat_system.start_combat(
        combat_id="demo_combat",
        combatants=[player, enemy]
    )

    print(f"\n⚔️ Combat Start!")
    print(f"\n{player.name}: {player.get_status()['hp']} HP, {player.get_status()['coherence']} Coherence")
    print(f"{enemy.name}: {enemy.get_status()['hp']} HP, {enemy.get_status()['coherence']} Coherence")
    print(f"\nVulnerabilities detected: {enemy.vulnerabilities[:3]}")

    # Execute intimidate action
    print(f"\n--- Turn 1 ---")
    print(f"{player.name} uses Intimidate!")

    result = combat.execute_action(
        attacker_id="player",
        defender_id="enemy",
        action=combat_system.combat_actions['intimidate']
    )

    print(f"\nResult:")
    print(f"  Physical damage: {result['damage_report']['physical_damage_dealt']}")
    print(f"  Psychological damage: {result['damage_report']['psychological_damage_dealt']}")
    print(f"  Critical hit: {result['damage_report']['critical_hit']}")
    print(f"  Vulnerabilities hit: {result['damage_report']['vulnerabilities_hit']}")

    print(f"\n{enemy.name} status:")
    status = enemy.get_status()
    print(f"  HP: {status['hp']}")
    print(f"  Coherence: {status['coherence']}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
