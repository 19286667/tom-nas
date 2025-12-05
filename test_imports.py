#!/usr/bin/env python3
"""
Comprehensive import testing script for ToM-NAS codebase.
Tests all modules for import errors, syntax errors, and circular imports.
"""

import sys
import py_compile
import importlib
import traceback
from pathlib import Path

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def check_syntax_errors():
    """Check all Python files for syntax errors."""
    print_header("1. Checking Python files for syntax errors")

    errors = []
    success_count = 0

    src_path = Path("src")
    for py_file in src_path.rglob("*.py"):
        try:
            py_compile.compile(str(py_file), doraise=True)
            success_count += 1
        except py_compile.PyCompileError as e:
            errors.append((str(py_file), str(e)))
            print_error(f"Syntax error in {py_file}")
            print(f"  {e}")

    if errors:
        print_error(f"Found {len(errors)} files with syntax errors")
        return False
    else:
        print_success(f"All {success_count} Python files have valid syntax")
        return True

def test_main_import():
    """Test importing the main src module."""
    print_header("2. Testing main 'src' module import")

    try:
        import src
        print_success("Successfully imported 'src' module")
        print(f"  Available attributes: {dir(src)}")
        return True
    except Exception as e:
        print_error(f"Failed to import 'src' module")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False

def test_submodule_imports():
    """Test importing all submodules."""
    print_header("3. Testing individual submodule imports")

    # List of all submodules to test
    submodules = [
        "src.core",
        "src.core.beliefs",
        "src.core.events",
        "src.core.ontology",

        "src.agents",
        "src.agents.architectures",
        "src.agents.cognitive_extensions",

        "src.cognition",
        "src.cognition.mentalese",
        "src.cognition.recursive_simulation",
        "src.cognition.trm",

        "src.world",
        "src.world.social_world",

        "src.worldsim",

        "src.evolution",
        "src.evolution.linas",
        "src.evolution.mutation_controller",
        "src.evolution.poet_controller",
        "src.evolution.tom_fitness",
        "src.evolution.zero_cost_proxies",
        "src.evolution.supernet",
        "src.evolution.operators",

        "src.evaluation",
        "src.evaluation.metrics",
        "src.evaluation.benchmarks",
        "src.evaluation.tom_benchmarks",
        "src.evaluation.zombie_detection",
        "src.evaluation.scientific_validation",

        "src.liminal",
        "src.liminal.coevolution_demo",
        "src.liminal.game_environment",
        "src.liminal.narrative_emergence",
        "src.liminal.nas_integration",
        "src.liminal.psychosocial_coevolution",
        "src.liminal.realms",
        "src.liminal.soul_map",

        "src.liminal.mechanics",
        "src.liminal.mechanics.ontological_instability",
        "src.liminal.mechanics.cognitive_hazards",
        "src.liminal.mechanics.soul_scanner",

        "src.liminal.npcs",
        "src.liminal.npcs.base_npc",
        "src.liminal.npcs.archetypes",
        "src.liminal.npcs.heroes",

        "src.benchmarks",
        "src.benchmarks.socialIQA_loader",
        "src.benchmarks.social_games",
        "src.benchmarks.tomi_loader",

        "src.visualization",
        "src.visualization.app",
        "src.visualization.belief_inspector",
        "src.visualization.nas_dashboard",
        "src.visualization.world_renderer",

        "src.simulation",
        "src.simulation.fractal_node",

        "src.godot_bridge",
        "src.godot_bridge.bridge",
        "src.godot_bridge.action",
        "src.godot_bridge.protocol",
        "src.godot_bridge.perception",
        "src.godot_bridge.symbol_grounding",

        "src.knowledge_base",
        "src.knowledge_base.indras_net",
        "src.knowledge_base.query_engine",
        "src.knowledge_base.schemas",
        "src.knowledge_base.taxonomy",

        "src.knowledge",

        "src.training",
        "src.training.curriculum",

        "src.transparency",
        "src.transparency.tools",

        "src.game",
        "src.game.api_server",
        "src.game.combat_system",
        "src.game.dialogue_system",
        "src.game.quest_system",
        "src.game.soul_map_visualizer",

        "src.utils",
    ]

    success = []
    failures = []

    for module_name in submodules:
        try:
            module = importlib.import_module(module_name)
            print_success(f"Successfully imported {module_name}")
            success.append(module_name)
        except ModuleNotFoundError as e:
            print_warning(f"Module not found: {module_name}")
            print(f"  Error: {e}")
            failures.append((module_name, "ModuleNotFoundError", str(e)))
        except ImportError as e:
            print_error(f"Import error in {module_name}")
            print(f"  Error: {e}")
            failures.append((module_name, "ImportError", str(e)))
        except Exception as e:
            print_error(f"Unexpected error importing {module_name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            failures.append((module_name, type(e).__name__, str(e)))

    print(f"\n{GREEN}Successfully imported: {len(success)}/{len(submodules)} modules{RESET}")
    if failures:
        print(f"{RED}Failed to import: {len(failures)} modules{RESET}")
        return False
    return True

def check_init_files():
    """Check all __init__.py files and their exports."""
    print_header("4. Checking __init__.py files")

    src_path = Path("src")
    init_files = list(src_path.rglob("__init__.py"))

    print(f"Found {len(init_files)} __init__.py files:\n")

    for init_file in sorted(init_files):
        relative_path = init_file.relative_to(src_path.parent)
        print(f"\n{BLUE}Checking {relative_path}{RESET}")

        try:
            with open(init_file, 'r') as f:
                content = f.read()

            if not content.strip():
                print_warning(f"  Empty __init__.py")
            else:
                lines = content.strip().split('\n')
                print(f"  Lines: {len(lines)}")

                # Check for __all__ definition
                if "__all__" in content:
                    print_success(f"  Defines __all__")

                # Count imports
                import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
                if import_lines:
                    print(f"  Import statements: {len(import_lines)}")

        except Exception as e:
            print_error(f"  Error reading file: {e}")

    return True

def check_circular_imports():
    """Attempt to detect circular import issues."""
    print_header("5. Checking for circular import issues")

    # Try importing modules in different orders
    test_sequences = [
        ["src.core", "src.agents", "src.world"],
        ["src.agents", "src.cognition", "src.evolution"],
        ["src.liminal", "src.liminal.npcs", "src.liminal.mechanics"],
        ["src.evaluation", "src.benchmarks"],
        ["src.visualization", "src.simulation"],
    ]

    issues = []

    for sequence in test_sequences:
        try:
            # Clear imported modules
            for mod_name in sequence:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]

            # Try importing in sequence
            for mod_name in sequence:
                importlib.import_module(mod_name)

            print_success(f"No circular imports detected in: {' -> '.join(sequence)}")

        except Exception as e:
            print_error(f"Possible circular import in: {' -> '.join(sequence)}")
            print(f"  Error: {e}")
            issues.append((sequence, str(e)))

    if issues:
        print(f"\n{RED}Found {len(issues)} potential circular import issues{RESET}")
        return False
    else:
        print(f"\n{GREEN}No circular import issues detected{RESET}")
        return True

def main():
    """Run all import tests."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}ToM-NAS Codebase Import Verification{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")

    results = {
        "Syntax Check": check_syntax_errors(),
        "Main Import": test_main_import(),
        "Submodule Imports": test_submodule_imports(),
        "Init Files": check_init_files(),
        "Circular Imports": check_circular_imports(),
    }

    # Summary
    print_header("SUMMARY")

    all_passed = True
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            all_passed = False

    print(f"\n{BLUE}{'='*80}{RESET}")
    if all_passed:
        print(f"{GREEN}All import checks PASSED ✓{RESET}")
        return 0
    else:
        print(f"{RED}Some import checks FAILED ✗{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
