"""
Loading utilities for compiled DSPy programs.
Day 4: Load GEPA-compiled ScoreImportance module.
"""

import os
import logging
import dspy
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_compiled_scorer(path: str = 'compiled/compiled_scorer.json') -> Optional[dspy.Module]:
    """
    Load compiled scorer from disk.

    Args:
        path: Path to compiled scorer JSON file

    Returns:
        Compiled scorer module, or None if file doesn't exist
    """
    from dspy_modules import ScoreImportance

    # Resolve path relative to project root
    if not os.path.isabs(path):
        project_root = Path(__file__).parent.parent
        path = project_root / path

    if not os.path.exists(path):
        logger.warning(f"Compiled scorer not found at {path}")
        return None

    try:
        # Create uncompiled baseline structure
        compiled = dspy.ChainOfThought(ScoreImportance)

        # Load compiled state
        compiled.load(str(path))

        logger.info(f"✅ Compiled scorer loaded from {path}")
        return compiled

    except Exception as e:
        logger.error(f"Failed to load compiled scorer: {e}")
        return None


def load_all_compiled_modules(compiled_dir: str = 'compiled') -> dict:
    """
    Load all available compiled modules.

    Args:
        compiled_dir: Directory containing compiled modules

    Returns:
        Dictionary mapping module names to loaded modules
    """
    modules = {}

    # Load scorer
    scorer = load_compiled_scorer(f"{compiled_dir}/compiled_scorer.json")
    if scorer:
        modules['scorer'] = scorer

    # Future: Load reflector, planner, etc.
    # reflector = load_compiled_reflector(f"{compiled_dir}/compiled_reflector.json")
    # if reflector:
    #     modules['reflector'] = reflector

    logger.info(f"Loaded {len(modules)} compiled module(s)")
    return modules


def get_compilation_info(compiled_dir: str = 'compiled') -> dict:
    """
    Get information about compiled modules.

    Args:
        compiled_dir: Directory containing compiled modules

    Returns:
        Dictionary with compilation metadata
    """
    import json

    info = {
        'available_modules': [],
        'compilation_results': None
    }

    # Check for compiled scorer
    project_root = Path(__file__).parent.parent
    scorer_path = project_root / compiled_dir / 'compiled_scorer.json'

    if scorer_path.exists():
        info['available_modules'].append({
            'name': 'scorer',
            'path': str(scorer_path),
            'size_kb': scorer_path.stat().st_size / 1024
        })

    # Check for compilation results
    results_path = project_root / compiled_dir / 'compilation_results.json'
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                info['compilation_results'] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load compilation results: {e}")

    return info


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from dspy_modules import configure_dspy

    # Configure DSPy first
    configure_dspy()

    # Load compiled modules
    print("=" * 70)
    print("LOADING COMPILED MODULES")
    print("=" * 70)

    modules = load_all_compiled_modules()

    if 'scorer' in modules:
        print("\n✅ Compiled scorer loaded successfully")

        # Test it
        print("\nTesting compiled scorer...")
        result = modules['scorer'](
            observation="Alice invited me to a party at 7pm tonight",
            agent_goal="Build relationships in the neighborhood",
            agent_personality="social, optimistic"
        )
        print(f"Test prediction: {result.score}")
        print(f"Reasoning: {result.reasoning}")
    else:
        print("\n⚠️  No compiled scorer found")
        print("   Run Day 4 compilation first")

    # Show compilation info
    print("\n" + "=" * 70)
    print("COMPILATION INFO")
    print("=" * 70)

    info = get_compilation_info()
    print(f"\nAvailable modules: {len(info['available_modules'])}")
    for mod in info['available_modules']:
        print(f"  - {mod['name']}: {mod['size_kb']:.1f} KB")

    if info['compilation_results']:
        results = info['compilation_results']
        print(f"\nCompilation time: {results['compilation_time_hours']:.2f} hours")
        print(f"Uncompiled ±2 accuracy: {results['uncompiled']['accuracy_within_2']:.1f}%")
        print(f"Compiled ±2 accuracy:   {results['compiled']['accuracy_within_2']:.1f}%")
        print(f"Improvement: +{results['improvement']['accuracy_delta']:.1f}%")
