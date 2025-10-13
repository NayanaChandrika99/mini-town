"""
Baseline performance test for uncompiled ScoreImportance module.
Tests all seeds to establish baseline accuracy before compilation.
"""

import json
import sys
import asyncio
import logging
from pathlib import Path
from collections import defaultdict, Counter

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dspy_modules import configure_dspy, score_observation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_seeds(filepath="scorer_v1.json"):
    """Load seeds from JSON file."""
    seed_path = Path(__file__).parent / filepath
    with open(seed_path, 'r') as f:
        data = json.load(f)
    return data


async def test_seed(seed, seed_id, total):
    """Test a single seed and return results."""
    logger.info(f"Testing seed {seed_id}/{total}: '{seed['observation'][:50]}...'")

    try:
        predicted_score = await score_observation(
            observation=seed['observation'],
            agent_goal=seed['agent_goal'],
            agent_personality=seed['agent_personality']
        )

        gold_score = seed['gold_score']
        error = abs(predicted_score - gold_score)

        logger.info(f"  Gold: {gold_score}, Predicted: {predicted_score}, Error: {error}")

        return {
            'seed_id': seed['id'],
            'category': seed['category'],
            'gold_score': gold_score,
            'predicted_score': predicted_score,
            'error': error,
            'within_1': error <= 1,
            'within_2': error <= 2,
            'exact': error == 0
        }

    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {
            'seed_id': seed['id'],
            'category': seed['category'],
            'gold_score': seed['gold_score'],
            'predicted_score': None,
            'error': None,
            'within_1': False,
            'within_2': False,
            'exact': False,
            'failed': True
        }


async def run_baseline_test():
    """Run baseline test on all seeds."""
    logger.info("=" * 70)
    logger.info("BASELINE PERFORMANCE TEST")
    logger.info("=" * 70)

    # Configure DSPy
    logger.info("\n1. Configuring DSPy...")
    configure_dspy()
    logger.info("✅ DSPy configured (uncompiled baseline)\n")

    # Load seeds
    logger.info("2. Loading seeds...")
    seeds_data = load_seeds()
    seeds = seeds_data['seeds']
    total = len(seeds)
    logger.info(f"✅ Loaded {total} seeds\n")

    # Test all seeds
    logger.info(f"3. Testing all {total} seeds...")
    logger.info("-" * 70)

    results = []
    for i, seed in enumerate(seeds, 1):
        result = await test_seed(seed, i, total)
        results.append(result)

        # Brief pause to avoid rate limiting
        if i % 5 == 0:
            await asyncio.sleep(1)

    logger.info("-" * 70)
    logger.info("✅ All seeds tested\n")

    # Analyze results
    analyze_results(results, seeds_data)


def analyze_results(results, seeds_data):
    """Analyze and report baseline performance."""
    logger.info("=" * 70)
    logger.info("BASELINE PERFORMANCE ANALYSIS")
    logger.info("=" * 70)

    # Filter out failed tests
    valid_results = [r for r in results if not r.get('failed', False)]
    failed_count = len(results) - len(valid_results)

    if failed_count > 0:
        logger.warning(f"\n⚠️  {failed_count} tests failed (excluded from analysis)")

    # Overall metrics
    total_valid = len(valid_results)
    exact_matches = sum(1 for r in valid_results if r['exact'])
    within_1 = sum(1 for r in valid_results if r['within_1'])
    within_2 = sum(1 for r in valid_results if r['within_2'])

    errors = [r['error'] for r in valid_results]
    mean_error = sum(errors) / len(errors) if errors else 0
    max_error = max(errors) if errors else 0

    logger.info(f"\n📊 Overall Performance:")
    logger.info(f"  Total seeds tested: {total_valid}")
    logger.info(f"  Exact matches:      {exact_matches:2d} ({exact_matches/total_valid*100:.1f}%)")
    logger.info(f"  Within ±1:          {within_1:2d} ({within_1/total_valid*100:.1f}%)")
    logger.info(f"  Within ±2:          {within_2:2d} ({within_2/total_valid*100:.1f}%)")
    logger.info(f"  Mean Absolute Error: {mean_error:.2f}")
    logger.info(f"  Max Error:          {max_error}")

    # Performance by category
    logger.info(f"\n📊 Performance by Category:")

    category_errors = defaultdict(list)
    for r in valid_results:
        category_errors[r['category']].append(r['error'])

    for category in sorted(category_errors.keys()):
        errors = category_errors[category]
        mean = sum(errors) / len(errors)
        within_2_count = sum(1 for e in errors if e <= 2)
        accuracy = within_2_count / len(errors) * 100

        logger.info(f"  {category:20s}: MAE={mean:.2f}, ±2 accuracy={accuracy:.0f}% ({len(errors)} seeds)")

    # Performance by score range
    logger.info(f"\n📊 Performance by Gold Score Range:")

    score_ranges = {
        "Low (1-3)": (1, 3),
        "Medium (4-6)": (4, 6),
        "High (7-10)": (7, 10)
    }

    for range_name, (min_score, max_score) in score_ranges.items():
        range_results = [r for r in valid_results
                         if min_score <= r['gold_score'] <= max_score]

        if range_results:
            range_errors = [r['error'] for r in range_results]
            mean = sum(range_errors) / len(range_errors)
            within_2_count = sum(1 for e in range_errors if e <= 2)
            accuracy = within_2_count / len(range_errors) * 100

            logger.info(f"  {range_name:20s}: MAE={mean:.2f}, ±2 accuracy={accuracy:.0f}% ({len(range_results)} seeds)")

    # Confusion matrix
    logger.info(f"\n📊 Confusion Analysis (Predicted vs Gold):")

    # Group scores into ranges for clarity
    def score_bucket(score):
        if score <= 3:
            return "Low (1-3)"
        elif score <= 6:
            return "Med (4-6)"
        else:
            return "High (7-10)"

    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid_results:
        gold_bucket = score_bucket(r['gold_score'])
        pred_bucket = score_bucket(r['predicted_score'])
        confusion[gold_bucket][pred_bucket] += 1

    buckets = ["Low (1-3)", "Med (4-6)", "High (7-10)"]
    header = "Gold \\ Pred"
    logger.info(f"\n  {header:20s} | {'Low (1-3)':>12s} | {'Med (4-6)':>12s} | {'High (7-10)':>12s}")
    logger.info("  " + "-" * 66)

    for gold in buckets:
        row = f"  {gold:20s} |"
        for pred in buckets:
            count = confusion[gold][pred]
            row += f" {count:12d} |"
        logger.info(row)

    # Top errors
    logger.info(f"\n🔍 Top 5 Largest Errors:")

    sorted_results = sorted(valid_results, key=lambda r: r['error'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        seed = next(s for s in seeds_data['seeds'] if s['id'] == r['seed_id'])
        logger.info(f"\n  {i}. Seed #{r['seed_id']}: Error = {r['error']}")
        logger.info(f"     Observation: \"{seed['observation'][:60]}...\"")
        logger.info(f"     Gold: {r['gold_score']}, Predicted: {r['predicted_score']}")
        logger.info(f"     Category: {r['category']}")

    # Save results to JSON
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_seeds': total_valid,
                'exact_matches': exact_matches,
                'within_1': within_1,
                'within_2': within_2,
                'mean_absolute_error': mean_error,
                'max_error': max_error,
                'exact_accuracy': exact_matches / total_valid * 100,
                'within_1_accuracy': within_1 / total_valid * 100,
                'within_2_accuracy': within_2 / total_valid * 100
            },
            'results': results
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n📈 Key Metrics (Uncompiled):")
    logger.info(f"  ±2 Accuracy: {within_2/total_valid*100:.1f}%")
    logger.info(f"  Mean Error:  {mean_error:.2f}")

    logger.info(f"\n🎯 Compilation Target:")
    logger.info(f"  Goal: ±2 accuracy > 80% (current: {within_2/total_valid*100:.1f}%)")
    logger.info(f"  Goal: MAE < 1.5 (current: {mean_error:.2f})")

    improvement_needed = (80 - within_2/total_valid*100) if within_2/total_valid*100 < 80 else 0
    if improvement_needed > 0:
        logger.info(f"  Need: +{improvement_needed:.1f}% improvement")
    else:
        logger.info(f"  Status: Already exceeding goal! 🎉")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(run_baseline_test())
