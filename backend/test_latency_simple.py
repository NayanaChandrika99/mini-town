"""
Simple latency test - just measure LLM call latencies without database.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from dspy_modules import configure_dspy, score_observation, generate_reflection
from utils import get_latency_tracker, timed_llm_call

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run simple latency test."""

    logger.info("=" * 70)
    logger.info("SIMPLE LATENCY TEST")
    logger.info("=" * 70)

    # Configure DSPy
    logger.info("\n1. Configuring DSPy...")
    configure_dspy()
    logger.info("✅ DSPy configured\n")

    # Test observations
    test_observations = [
        ("Alice invited me to a party at 7pm tonight", "Build relationships", "social, optimistic"),
        ("The grass is green today", "Build relationships", "social, optimistic"),
        ("Fire alarm going off!", "Build relationships", "social, optimistic"),
        ("Bob seems busy with work", "Complete research", "analytical, introverted"),
        ("Found an interesting book", "Complete research", "analytical, introverted"),
    ]

    logger.info(f"2. Testing ScoreImportance with {len(test_observations)} observations...\n")

    for i, (obs, goal, personality) in enumerate(test_observations, 1):
        logger.info(f"Test {i}/{len(test_observations)}: '{obs}'")
        try:
            score = await timed_llm_call(
                score_observation,
                signature_name="ScoreImportance",
                timeout=5.0,
                observation=obs,
                agent_goal=goal,
                agent_personality=personality
            )
            logger.info(f"  → Score: {score}/10\n")
        except Exception as e:
            logger.error(f"  → Failed: {e}\n")

    # Test reflection
    logger.info("\n3. Testing Reflect...\n")

    memories = [
        "Alice invited me to a party",
        "Bob is busy with work",
        "Found an interesting book about local history",
        "The weather is nice today"
    ]

    try:
        insight = await timed_llm_call(
            generate_reflection,
            signature_name="Reflect",
            timeout=5.0,
            recent_memories=memories,
            agent_personality="analytical, introverted",
            agent_goal="Complete research on local history"
        )
        logger.info(f"Insight: {insight}\n")
    except Exception as e:
        logger.error(f"Failed: {e}\n")

    # Get statistics
    logger.info("=" * 70)
    logger.info("LATENCY STATISTICS")
    logger.info("=" * 70 + "\n")

    tracker = get_latency_tracker()
    stats = tracker.get_stats()

    for signature_name, sig_stats in stats.items():
        logger.info(f"{signature_name}:")
        logger.info(f"  Count:        {sig_stats['count']}")
        logger.info(f"  Success Rate: {sig_stats['success_rate']:.1f}%")
        logger.info(f"  p50:          {sig_stats['p50_ms']}ms")
        logger.info(f"  p95:          {sig_stats['p95_ms']}ms")
        logger.info(f"  p99:          {sig_stats['p99_ms']}ms")
        logger.info(f"  Mean:         {sig_stats['mean_ms']}ms\n")

    # Calculate tick decision
    p95_max = max([s.get('p95_ms', 0) for s in stats.values()])

    logger.info("=" * 70)
    logger.info("TICK INTERVAL DECISION")
    logger.info("=" * 70)
    logger.info(f"\nMaximum p95 latency: {p95_max}ms")
    logger.info(f"Current tick interval: 2.0s (2000ms)\n")

    if p95_max > 2500:
        decision = "INVESTIGATE 🚨"
        recommendation = "p95 > 2500ms - Too slow! Check model or API."
    elif p95_max > 1500:
        decision = "ADJUST TO 3s ⚠️"
        recommendation = "p95 1500-2500ms - Increase tick_interval to 3.0s"
    else:
        decision = "KEEP 2s ✅"
        recommendation = "p95 < 1500ms - Current 2s ticks are safe!"

    logger.info(f"Decision: {decision}")
    logger.info(f"Recommendation: {recommendation}\n")

    logger.info("=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
