"""
Seed validation script for scorer_v1.json
Validates distribution, calculates statistics, and generates plots.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add backend to path for utils if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def load_seeds(filepath="scorer_v1.json"):
    """Load seeds from JSON file."""
    seed_path = Path(__file__).parent / filepath
    with open(seed_path, 'r') as f:
        data = json.load(f)
    return data


def validate_distribution(seeds_data):
    """Validate that all score ranges 1-10 are represented."""
    print("=" * 70)
    print("SCORE DISTRIBUTION VALIDATION")
    print("=" * 70)

    scores = [seed['gold_score'] for seed in seeds_data['seeds']]
    score_counts = Counter(scores)

    print(f"\nTotal seeds: {len(scores)}")
    print(f"\nScore distribution:")
    for score in range(1, 11):
        count = score_counts.get(score, 0)
        bar = "█" * count
        status = "✅" if count >= 2 else "⚠️" if count >= 1 else "❌"
        print(f"  Score {score:2d}: {count:2d} seeds {bar} {status}")

    # Check minimum requirement (2 per score)
    missing_scores = [s for s in range(1, 11) if score_counts.get(s, 0) < 2]
    if missing_scores:
        print(f"\n⚠️  Scores with < 2 examples: {missing_scores}")
        print("   Recommendation: Add more seeds for these scores")
    else:
        print(f"\n✅ All scores 1-10 have at least 2 examples")

    return score_counts


def validate_categories(seeds_data):
    """Validate category distribution."""
    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)

    categories = [seed['category'] for seed in seeds_data['seeds']]
    category_counts = Counter(categories)

    expected = seeds_data.get('categories', {})

    print(f"\nCategory breakdown:")
    for category, count in category_counts.items():
        expected_count = expected.get(category, "N/A")
        status = "✅" if count == expected_count else "⚠️"
        print(f"  {category:20s}: {count:2d} seeds (expected: {expected_count}) {status}")

    total = sum(category_counts.values())
    print(f"\nTotal: {total} seeds")

    return category_counts


def analyze_by_personality(seeds_data):
    """Analyze score distribution by agent personality."""
    print("\n" + "=" * 70)
    print("PERSONALITY ANALYSIS")
    print("=" * 70)

    personality_scores = defaultdict(list)

    for seed in seeds_data['seeds']:
        personality = seed['agent_personality']
        score = seed['gold_score']
        personality_scores[personality].append(score)

    print(f"\nScore statistics by personality:")
    for personality, scores in sorted(personality_scores.items()):
        mean = np.mean(scores)
        std = np.std(scores)
        count = len(scores)
        print(f"  {personality:30s}: {count:2d} seeds, mean={mean:.1f}, std={std:.1f}")

    return personality_scores


def analyze_by_goal(seeds_data):
    """Analyze score distribution by agent goal."""
    print("\n" + "=" * 70)
    print("GOAL ANALYSIS")
    print("=" * 70)

    goal_scores = defaultdict(list)

    for seed in seeds_data['seeds']:
        goal = seed['agent_goal']
        score = seed['gold_score']
        goal_scores[goal].append(score)

    print(f"\nScore statistics by goal:")
    for goal, scores in sorted(goal_scores.items()):
        mean = np.mean(scores)
        std = np.std(scores)
        count = len(scores)
        print(f"  {goal[:40]:40s}: {count:2d} seeds, mean={mean:.1f}, std={std:.1f}")

    return goal_scores


def calculate_cohen_kappa(rater1_scores, rater2_scores):
    """
    Calculate Cohen's kappa for inter-rater agreement.

    Args:
        rater1_scores: List of scores from rater 1
        rater2_scores: List of scores from rater 2

    Returns:
        kappa: Cohen's kappa coefficient
    """
    if len(rater1_scores) != len(rater2_scores):
        raise ValueError("Rater score lists must be same length")

    n = len(rater1_scores)

    # Calculate observed agreement (allowing ±1 tolerance)
    agreements = sum(1 for r1, r2 in zip(rater1_scores, rater2_scores) if abs(r1 - r2) <= 1)
    p_observed = agreements / n

    # Calculate expected agreement by chance
    rater1_dist = Counter(rater1_scores)
    rater2_dist = Counter(rater2_scores)

    p_expected = 0
    for score in range(1, 11):
        p1 = rater1_dist.get(score, 0) / n
        p2 = rater2_dist.get(score, 0) / n
        p_expected += p1 * p2

    # Cohen's kappa
    if p_expected == 1:
        return 1.0

    kappa = (p_observed - p_expected) / (1 - p_expected)
    return kappa


def generate_plots(seeds_data, output_dir=None):
    """Generate distribution plots."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    scores = [seed['gold_score'] for seed in seeds_data['seeds']]
    categories = [seed['category'] for seed in seeds_data['seeds']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Score distribution
    ax1 = axes[0, 0]
    score_counts = Counter(scores)
    ax1.bar(score_counts.keys(), score_counts.values(), color='steelblue', edgecolor='black')
    ax1.set_xlabel('Importance Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Score Distribution (1-10)')
    ax1.set_xticks(range(1, 11))
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Category distribution
    ax2 = axes[0, 1]
    category_counts = Counter(categories)
    ax2.bar(range(len(category_counts)), category_counts.values(),
            tick_label=list(category_counts.keys()), color='coral', edgecolor='black')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Count')
    ax2.set_title('Category Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Score histogram
    ax3 = axes[1, 0]
    ax3.hist(scores, bins=10, range=(0.5, 10.5), color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Importance Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Histogram')
    ax3.set_xticks(range(1, 11))
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Category vs Mean Score
    ax4 = axes[1, 1]
    cat_scores = defaultdict(list)
    for seed in seeds_data['seeds']:
        cat_scores[seed['category']].append(seed['gold_score'])

    categories_list = list(cat_scores.keys())
    means = [np.mean(cat_scores[cat]) for cat in categories_list]
    stds = [np.std(cat_scores[cat]) for cat in categories_list]

    ax4.bar(range(len(categories_list)), means, yerr=stds,
            tick_label=categories_list, color='mediumpurple',
            edgecolor='black', capsize=5)
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Mean Score')
    ax4.set_title('Mean Score by Category (±1 std)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "seed_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Distribution plot saved to: {output_path}")

    return output_path


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("SEED VALIDATION REPORT")
    print("=" * 70)

    # Load seeds
    seeds_data = load_seeds()
    print(f"\nLoaded: {seeds_data['description']}")
    print(f"Version: {seeds_data['version']}")
    print(f"Created: {seeds_data['created']}")

    # Run validations
    score_counts = validate_distribution(seeds_data)
    category_counts = validate_categories(seeds_data)
    personality_scores = analyze_by_personality(seeds_data)
    goal_scores = analyze_by_goal(seeds_data)

    # Generate plots
    try:
        plot_path = generate_plots(seeds_data)
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
        print("   (matplotlib may not be installed)")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_seeds = len(seeds_data['seeds'])
    score_range = (min(s['gold_score'] for s in seeds_data['seeds']),
                   max(s['gold_score'] for s in seeds_data['seeds']))

    print(f"\n✅ Total seeds: {total_seeds}")
    print(f"✅ Score range: {score_range[0]}-{score_range[1]}")
    print(f"✅ Categories: {len(category_counts)}")
    print(f"✅ Unique personalities: {len(personality_scores)}")
    print(f"✅ Unique goals: {len(goal_scores)}")

    # Check requirements
    print("\n📋 Requirements checklist:")
    checks = [
        (total_seeds >= 30, f"30-40 seeds collected ({total_seeds})"),
        (len(category_counts) >= 5, f"All categories covered ({len(category_counts)})"),
        (all(score_counts.get(s, 0) >= 1 for s in range(1, 11)), "All scores 1-10 represented"),
        (all(seed.get('rationale') for seed in seeds_data['seeds']), "All seeds have rationale"),
    ]

    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {description}")

    all_passed = all(check[0] for check in checks)

    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("   Ready for Day 4 compilation.")
    else:
        print("\n⚠️  Some checks failed. Review and add more seeds.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
