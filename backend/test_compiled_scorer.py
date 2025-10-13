"""
Test compiled scorer in isolation to diagnose loading/execution issues.
Day 5: Debug why compiled modules cause agent stalling.
"""

import asyncio
import logging
import sys
from pathlib import Path
import time
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dspy_modules import configure_dspy, use_compiled, load_compiled_modules, get_current_scorer, score_observation
from utils import LatencyTracker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


async def test_scorer_basic():
    """Test basic scorer functionality with sample observations."""

    test_cases = [
        {
            "observation": "Maria invited you to a party at 7pm tonight",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_score_range": (7, 9)
        },
        {
            "observation": "The sky is blue today",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_score_range": (1, 3)
        },
        {
            "observation": "Bob mentioned he needs help moving furniture tomorrow",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_score_range": (6, 8)
        },
    ]

    results = {
        'uncompiled': [],
        'compiled': []
    }

    # Test uncompiled first
    logger.info("=" * 70)
    logger.info("TESTING UNCOMPILED SCORER")
    logger.info("=" * 70)

    configure_dspy()
    use_compiled(False)

    for i, test in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}: '{test['observation'][:50]}...'")

        try:
            start_time = time.time()
            score = await score_observation(
                observation=test['observation'],
                agent_goal=test['goal'],
                agent_personality=test['personality']
            )
            elapsed = time.time() - start_time

            logger.info(f"  ✅ Score: {score} (expected: {test['expected_score_range']}) in {elapsed:.2f}s")

            results['uncompiled'].append({
                'test_id': i+1,
                'score': score,
                'elapsed': elapsed,
                'success': True
            })

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            results['uncompiled'].append({
                'test_id': i+1,
                'error': str(e),
                'success': False
            })

    # Test compiled
    logger.info("\n" + "=" * 70)
    logger.info("TESTING COMPILED SCORER")
    logger.info("=" * 70)

    configure_dspy()
    success = load_compiled_modules(config['compilation']['compiled_dir'])

    if not success:
        logger.error("❌ Failed to load compiled modules")
        return results

    use_compiled(True)
    logger.info("✅ Compiled modules loaded")

    for i, test in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}: '{test['observation'][:50]}...'")

        try:
            start_time = time.time()
            score = await score_observation(
                observation=test['observation'],
                agent_goal=test['goal'],
                agent_personality=test['personality']
            )
            elapsed = time.time() - start_time

            logger.info(f"  ✅ Score: {score} (expected: {test['expected_score_range']}) in {elapsed:.2f}s")

            results['compiled'].append({
                'test_id': i+1,
                'score': score,
                'elapsed': elapsed,
                'success': True
            })

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            results['compiled'].append({
                'test_id': i+1,
                'error': str(e),
                'success': False
            })

    return results


def print_summary(results):
    """Print summary of test results."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for mode in ['uncompiled', 'compiled']:
        print(f"\n{mode.upper()}:")

        mode_results = results[mode]
        if not mode_results:
            print("  No results")
            continue

        successful = [r for r in mode_results if r.get('success')]
        failed = [r for r in mode_results if not r.get('success')]

        print(f"  Successful: {len(successful)}/{len(mode_results)}")
        print(f"  Failed: {len(failed)}/{len(mode_results)}")

        if successful:
            avg_elapsed = sum(r['elapsed'] for r in successful) / len(successful)
            print(f"  Avg latency: {avg_elapsed:.2f}s")

            scores = [r['score'] for r in successful]
            print(f"  Score range: {min(scores)} - {max(scores)}")

        if failed:
            print(f"  Errors:")
            for r in failed:
                print(f"    Test {r['test_id']}: {r.get('error', 'Unknown error')}")

    print("=" * 70)


async def main():
    """Run diagnostic tests."""
    logger.info("Starting compiled scorer diagnostic tests...")

    results = await test_scorer_basic()

    print_summary(results)

    # Save results
    import json
    output_path = Path(__file__).parent.parent / "results" / "compiled_scorer_diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
