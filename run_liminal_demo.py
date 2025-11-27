#!/usr/bin/env python3
"""
Liminal Architectures Demo Runner

This script demonstrates the Liminal game environment for ToM-NAS.
It showcases:
1. World initialization with 200+ NPCs across 5 realms
2. Hero NPC creation and Soul Map visualization
3. Soul Scanner analysis of NPCs
4. Cognitive hazard interventions
5. Ontological instability system
6. Integration with NAS models (if available)

Usage:
    python run_liminal_demo.py [--interactive] [--episodes N]
"""

import argparse
import sys
import random
import time

# Ensure src is in path
sys.path.insert(0, '.')

from src.liminal import (
    LiminalEnvironment,
    SoulMap,
    Realm,
    RealmType,
    REALMS,
)
from src.liminal.npcs import (
    BaseNPC,
    create_hero_npc,
    create_archetype_npc,
    HERO_NPCS,
    ARCHETYPES,
)
from src.liminal.mechanics import (
    SoulScanner,
    CognitiveHazard,
    HAZARD_REGISTRY,
    apply_hazard,
    OntologicalInstability,
    InstabilityLevel,
)
from src.liminal.game_environment import ActionType


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_soul_map_summary(soul_map: SoulMap) -> None:
    """Print a summary of a Soul Map."""
    print(f"  Stability: {soul_map.compute_stability():.2f}")
    print(f"  Threat Response: {soul_map.compute_threat_response():.2f}")
    print(f"  Social Openness: {soul_map.compute_social_openness():.2f}")
    dominant = soul_map.get_dominant_motivation()
    print(f"  Dominant Drive: {dominant[0]} ({dominant[1]:.2f})")
    print(f"  ToM Depth: {soul_map.get_tom_depth_int()}")


def demo_soul_map() -> None:
    """Demonstrate the Soul Map system."""
    print_header("SOUL MAP SYSTEM")

    print("\n1. Creating a basic Soul Map...")
    soul = SoulMap()
    print(f"   Default Soul Map: {soul}")

    print("\n2. Creating from archetype 'bureaucrat'...")
    bureaucrat_soul = SoulMap.from_archetype("bureaucrat", variance=0.1)
    print_soul_map_summary(bureaucrat_soul)

    print("\n3. Converting to tensor...")
    tensor = bureaucrat_soul.to_tensor()
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   First 10 values: {tensor[:10].tolist()}")

    print("\n4. Creating Soul Map Delta (Doubt)...")
    from src.liminal.soul_map import SoulMapDelta
    doubt = SoulMapDelta.doubt(intensity=0.3)
    print(f"   Doubt changes: {doubt.changes}")

    print("\n5. Applying Doubt to bureaucrat...")
    pre_stability = bureaucrat_soul.compute_stability()
    doubt.apply_to(bureaucrat_soul)
    post_stability = bureaucrat_soul.compute_stability()
    print(f"   Stability change: {pre_stability:.2f} -> {post_stability:.2f}")


def demo_realms() -> None:
    """Demonstrate the Realm system."""
    print_header("THE FIVE REALMS")

    for realm_type, realm in REALMS.items():
        print(f"\n{realm.name.upper()}")
        print(f"  Type: {realm.realm_type.value}")
        print(f"  Aesthetic: {realm.aesthetic[:80]}...")
        print(f"  Key Mechanic: {realm.key_mechanic}")
        print(f"  Population: {realm.population_name}")
        print(f"  Traversal: {', '.join(realm.traversal_methods[:3])}")


def demo_heroes() -> None:
    """Demonstrate Hero NPCs."""
    print_header("HERO NPCs")

    for hero_id in HERO_NPCS[:5]:  # Show first 5
        print(f"\n--- {hero_id.upper()} ---")
        hero = create_hero_npc(hero_id)
        print(f"  Name: {hero.name}")
        print(f"  Archetype: {hero.archetype}")
        print(f"  Realm: {hero.current_realm.value if hero.current_realm else 'None'}")
        print(f"  ToM Depth: {hero.tom_depth}")
        print(f"  Emotional State: {hero.emotional_state}")
        print(f"  Active Goal: {hero.active_goal}")
        print_soul_map_summary(hero.soul_map)

        if hero.dialogue_tree:
            intro = hero.dialogue_tree.get("intro", "")[:100]
            print(f"  Dialogue: \"{intro}...\"")


def demo_archetypes() -> None:
    """Demonstrate NPC archetypes."""
    print_header("NPC ARCHETYPES")

    for archetype_name in list(ARCHETYPES.keys())[:6]:
        arch = ARCHETYPES[archetype_name]
        print(f"\n{archetype_name.upper()}")
        print(f"  Description: {arch.get('description', 'N/A')[:60]}...")
        print(f"  Realm: {arch.get('realm', 'N/A')}")
        print(f"  Spawn Count: {arch.get('spawn_count', 0)}")
        print(f"  Behaviors: {arch.get('behaviors', [])[:3]}")


def demo_soul_scanner() -> None:
    """Demonstrate the Soul Scanner."""
    print_header("SOUL SCANNER ANALYSIS")

    # Create scanner and target
    scanner = SoulScanner(player_tom_depth=4)
    target = create_hero_npc("arthur_peregrine")

    print("\n1. Passive Scan (hover)...")
    passive = scanner.passive_scan(target)
    print(f"   Aura Color: {passive.aura_color}")
    print(f"   Aura Intensity: {passive.aura_intensity:.2f}")
    print(f"   Threat Level: {passive.threat_level:.2f}")

    print("\n2. Moderate Scan (standard analysis)...")
    moderate = scanner.moderate_scan(target)
    print(f"   Dominant Emotion: {moderate.dominant_emotion}")
    print(f"   Is Aware of Player: {moderate.is_aware_of_player}")
    print(f"   Cluster Summaries:")
    for cluster, summary in moderate.cluster_summaries.items():
        key_trait = summary.get('key_trait', 'N/A')
        print(f"     {cluster}: {key_trait}")

    print("\n3. Predictive Scan (behavior forecast)...")
    context = {"threat_present": True, "stranger_present": True}
    predictive = scanner.predictive_scan(target, context)
    print(f"   Predicted Behaviors:")
    for pred in predictive.predicted_behaviors:
        print(f"     - {pred}")
    print(f"   Prediction Confidence: {predictive.behavior_confidence:.2f}")


def demo_cognitive_hazards() -> None:
    """Demonstrate cognitive hazards."""
    print_header("COGNITIVE HAZARDS")

    print("\nAvailable Hazards:")
    for name, hazard in list(HAZARD_REGISTRY.items())[:8]:
        print(f"  {name.upper()}: {hazard.description[:50]}...")
        print(f"    Category: {hazard.category.value}, Cost: {hazard.cost}")

    print("\n\nApplying 'doubt' to a bureaucrat...")
    target = create_archetype_npc("bureaucrat", name="Form Processor #42")
    print(f"  Before: stability={target.soul_map.compute_stability():.2f}")

    success, result = apply_hazard("doubt", target, player_tom=3)
    print(f"  Success: {success}")
    print(f"  Stability Change: {result.get('stability_change', 0):.3f}")
    print(f"  Resistance: {result.get('resistance', 0):.2f}")
    print(f"  After: stability={target.soul_map.compute_stability():.2f}")
    print(f"  New Emotional State: {result.get('new_emotional_state', 'N/A')}")


def demo_ontological_instability() -> None:
    """Demonstrate the ontological instability system."""
    print_header("ONTOLOGICAL INSTABILITY (The 'Wanted' System)")

    instability = OntologicalInstability()

    print("\nSimulating reality disruption...")
    events = [
        ("hazard:doubt", 5.0, "Applied doubt to bureaucrat"),
        ("hazard:fear", 3.0, "Applied fear to mourner"),
        ("realm_violation", 8.0, "Ran in the Ministry"),
        ("hazard:paradox", 6.0, "Applied paradox to enforcer"),
        ("npc_psychological_break", 12.0, "Caused psychological break"),
    ]

    for source, amount, desc in events:
        instability.add_instability(amount, source, desc)
        display = instability.get_display_data()
        print(f"  [{display['level_name']}] {display['instability_percent']:.1f}% - {desc}")
        print(f"    Color: {display['danger_color']}")

        if instability.nothing_manifested:
            print("    WARNING: The Nothing has manifested!")

    print("\n  Natural decay over 50 ticks...")
    for _ in range(50):
        instability.tick()

    display = instability.get_display_data()
    print(f"  Final: [{display['level_name']}] {display['instability_percent']:.1f}%")


def demo_environment() -> None:
    """Demonstrate the full game environment."""
    print_header("LIMINAL GAME ENVIRONMENT")

    print("\nInitializing environment with 200 NPCs...")
    env = LiminalEnvironment(
        population_size=200,
        include_heroes=True,
        starting_realm=RealmType.PEREGRINE,
        max_episode_length=100,
        seed=42,
    )

    stats = env.get_statistics()
    print(f"  Total NPCs: {stats['total_npcs']}")
    print(f"  Hero NPCs: {stats['hero_count']}")
    print(f"  NPCs per Realm:")
    for realm, count in stats['npcs_per_realm'].items():
        print(f"    {realm}: {count}")

    print("\nRunning 20 random steps...")
    observation = env.reset()

    for step in range(20):
        # Random action
        action_type = random.choice([
            ActionType.WAIT, ActionType.ANALYZE,
            ActionType.MOVE, ActionType.INTERVENE
        ])

        action = {"type": action_type}

        # Add target if needed
        game_state = env._get_game_state()
        if action_type in [ActionType.ANALYZE, ActionType.INTERVENE]:
            if game_state.nearby_npcs:
                action["target_id"] = game_state.nearby_npcs[0]
                if action_type == ActionType.INTERVENE:
                    action["hazard"] = random.choice(["doubt", "validation", "curiosity"])

        result = env.step(action)

        if step % 5 == 0:
            print(f"  Step {step}: reward={result.reward:.3f}, "
                  f"instability={result.game_state.instability:.1f}%")

    final_stats = env.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Predictions: {final_stats['total_predictions']}")
    print(f"  Prediction Accuracy: {final_stats['prediction_accuracy']:.2%}")
    print(f"  Total Interventions: {final_stats['total_interventions']}")
    print(f"  Instability Level: {final_stats['instability_level']}")


def run_interactive_session() -> None:
    """Run an interactive demo session."""
    print_header("INTERACTIVE LIMINAL SESSION")
    print("\nCommands:")
    print("  scan <npc_id>  - Analyze an NPC")
    print("  hazard <name> <npc_id>  - Apply cognitive hazard")
    print("  list npcs  - List nearby NPCs")
    print("  list hazards  - List available hazards")
    print("  status  - Show current status")
    print("  quit  - Exit")

    env = LiminalEnvironment(
        population_size=100,
        include_heroes=True,
        seed=42,
    )
    env.reset()
    scanner = SoulScanner(player_tom_depth=4)

    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd:
                continue

            if cmd[0] == "quit":
                break

            elif cmd[0] == "list" and len(cmd) > 1:
                if cmd[1] == "npcs":
                    game_state = env._get_game_state()
                    for npc_id in game_state.nearby_npcs[:10]:
                        npc = env.get_npc(npc_id)
                        print(f"  {npc_id}: {npc.name} ({npc.archetype})")

                elif cmd[1] == "hazards":
                    for name in list(HAZARD_REGISTRY.keys()):
                        print(f"  {name}")

            elif cmd[0] == "scan" and len(cmd) > 1:
                npc = env.get_npc(cmd[1])
                if npc:
                    result = scanner.moderate_scan(npc)
                    print(f"  {result.npc_name}: {result.emotional_state}")
                    print(f"  Aura: {result.aura_color} (intensity={result.aura_intensity:.2f})")
                    for cluster, summary in result.cluster_summaries.items():
                        print(f"  {cluster}: {summary.get('key_trait', 'N/A')}")
                else:
                    print("  NPC not found")

            elif cmd[0] == "hazard" and len(cmd) > 2:
                npc = env.get_npc(cmd[2])
                if npc:
                    success, result = apply_hazard(cmd[1], npc, player_tom=4)
                    if success:
                        print(f"  Applied {cmd[1]} to {npc.name}")
                        print(f"  Stability change: {result.get('stability_change', 0):.3f}")
                        print(f"  New state: {result.get('new_emotional_state', 'N/A')}")
                    else:
                        print(f"  Failed: {result.get('error', 'Unknown')}")
                else:
                    print("  NPC not found")

            elif cmd[0] == "status":
                stats = env.get_statistics()
                game_state = env._get_game_state()
                print(f"  Tick: {game_state.tick}")
                print(f"  Realm: {game_state.current_realm.value}")
                print(f"  Instability: {game_state.instability:.1f}% ({game_state.instability_level.name})")
                print(f"  Nearby NPCs: {len(game_state.nearby_npcs)}")

            else:
                print("  Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Liminal Architectures Demo")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run interactive session")
    parser.add_argument("--demo", "-d", type=str, default="all",
                       choices=["all", "soulmap", "realms", "heroes",
                               "archetypes", "scanner", "hazards",
                               "instability", "environment"],
                       help="Which demo to run")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("     LIMINAL ARCHITECTURES: GRAND THEFT ONTOLOGY")
    print("     ToM-NAS Game Environment Demo")
    print("=" * 60)

    if args.interactive:
        run_interactive_session()
        return

    demos = {
        "soulmap": demo_soul_map,
        "realms": demo_realms,
        "heroes": demo_heroes,
        "archetypes": demo_archetypes,
        "scanner": demo_soul_scanner,
        "hazards": demo_cognitive_hazards,
        "instability": demo_ontological_instability,
        "environment": demo_environment,
    }

    if args.demo == "all":
        for demo_fn in demos.values():
            demo_fn()
            print()
    else:
        demos[args.demo]()

    print("\n" + "=" * 60)
    print("     Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
