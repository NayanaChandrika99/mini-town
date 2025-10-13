"""
Unit tests for DSPy modules.
Tests ScoreImportance and Reflect without full simulation.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dspy_modules import configure_dspy, score_observation, generate_reflection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_score_importance():
    """Test ScoreImportance module."""
    logger.info("=" * 60)
    logger.info("Testing ScoreImportance")
    logger.info("=" * 60)

    test_cases = [
        {
            "observation": "Alice invited me to a party at 7pm tonight",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (7, 10)
        },
        {
            "observation": "The grass is green today",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (1, 3)
        },
        {
            "observation": "Fire alarm going off in the building",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (8, 10)
        }
    ]

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test['observation'][:50]}...")

        try:
            score = await score_observation(
                observation=test['observation'],
                agent_goal=test['goal'],
                agent_personality=test['personality']
            )

            logger.info(f"  Score: {score}")

            min_score, max_score = test['expected_range']
            if min_score <= score <= max_score:
                logger.info(f"  ✅ PASS (expected {min_score}-{max_score})")
            else:
                logger.warning(f"  ⚠️  Outside expected range {min_score}-{max_score}")

        except Exception as e:
            logger.error(f"  ❌ FAILED: {e}")


async def test_reflect():
    """Test Reflect module."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Reflect")
    logger.info("=" * 60)

    recent_memories = [
        "Alice invited me to a party at 7pm",
        "Bob said he's too busy to attend",
        "Carol is always punctual for events",
        "I feel excited about social gatherings"
    ]

    logger.info(f"\nMemories ({len(recent_memories)}):")
    for mem in recent_memories:
        logger.info(f"  - {mem}")

    try:
        insight = await generate_reflection(
            recent_memories=recent_memories,
            agent_personality="social, optimistic",
            agent_goal="Build relationships in the neighborhood"
        )

        logger.info(f"\nInsight: {insight}")
        logger.info("✅ PASS")

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting DSPy Module Tests")
    logger.info("Configuring DSPy with Groq...")

    configure_dspy()

    await test_score_importance()
    await test_reflect()

    logger.info("\n" + "=" * 60)
    logger.info("Tests Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
