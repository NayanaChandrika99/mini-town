"""
Diagnostic tests to verify Day 6 fixes.
Tests the wait behavior, time window logic, and arrival detection.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_get_upcoming_step():
    """Test that get_upcoming_step() correctly identifies upcoming plan steps."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: get_upcoming_step() method")
    logger.info("=" * 70)

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal",
        personality="test"
    )

    # Create a mock plan with time steps
    agent.current_plan = "10:00 AM - 11:00 AM: Go to party at location (200, 150)"
    agent.plan_last_updated = datetime.now()

    # Parse plan
    parsed_steps = agent.parse_plan()

    if not parsed_steps:
        logger.error("❌ FAILED: Plan parsing failed")
        return False

    logger.info(f"✅ Parsed {len(parsed_steps)} plan steps")

    # Test 1: Current time is 9:45 AM (15 min before event)
    # Should return upcoming step
    test_time = datetime.strptime("09:45 AM", "%I:%M %p")
    upcoming = agent.get_upcoming_step(parsed_steps, test_time, window_minutes=15)

    if upcoming:
        logger.info(f"✅ PASS: Found upcoming step at 9:45 AM: {upcoming['description'][:50]}...")
        return True
    else:
        logger.error("❌ FAILED: No upcoming step found at 9:45 AM (should be within 15 min window)")
        return False


def test_wait_behavior():
    """Test that agents wait at location when arriving early."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Wait at location behavior")
    logger.info("=" * 70)

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=195,  # Very close to target (200, 150)
        y=148,
        goal="Test goal",
        personality="test"
    )

    # Create a plan for 15 minutes in the future
    future_time = (datetime.now() + timedelta(minutes=15)).strftime("%I:%M %p")
    end_time = (datetime.now() + timedelta(minutes=45)).strftime("%I:%M %p")
    agent.current_plan = f"{future_time} - {end_time}: Attend party at location (200, 150)"
    agent.plan_last_updated = datetime.now()

    # Parse plan
    agent.parsed_plan = agent.parse_plan()
    agent._last_parsed_time = agent.plan_last_updated

    logger.info(f"Agent initial position: ({agent.x:.1f}, {agent.y:.1f})")
    logger.info(f"Plan: {agent.current_plan}")

    # Simulate 5 update ticks
    for i in range(5):
        state = agent.update([])

        logger.info(f"Tick {i+1}: Position ({agent.x:.1f}, {agent.y:.1f}), State: {agent.state}, Velocity: ({agent.vx:.1f}, {agent.vy:.1f})")

    # Check if agent stayed at location
    final_distance = ((agent.x - 200)**2 + (agent.y - 150)**2) ** 0.5

    if agent.state == "waiting" and final_distance < 15:
        logger.info(f"✅ PASS: Agent is waiting at location (distance: {final_distance:.1f}px, state: {agent.state})")
        return True
    else:
        logger.error(f"❌ FAILED: Agent is not waiting (distance: {final_distance:.1f}px, state: {agent.state})")
        return False


def test_time_window_matching():
    """Test that get_current_step() correctly handles time windows."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Time window matching")
    logger.info("=" * 70)

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal",
        personality="test"
    )

    # Create plan with specific time range
    agent.current_plan = "10:00 AM - 11:00 AM: Work at location (200, 150)"
    agent.plan_last_updated = datetime.now()

    parsed_steps = agent.parse_plan()

    if not parsed_steps:
        logger.error("❌ FAILED: Plan parsing failed")
        return False

    # Test 1: Time BEFORE window (9:45 AM) - should return None
    test_time_before = datetime.strptime("09:45 AM", "%I:%M %p")
    current_step_before = agent.get_current_step(parsed_steps, test_time_before)

    # Test 2: Time WITHIN window (10:15 AM) - should return step
    test_time_within = datetime.strptime("10:15 AM", "%I:%M %p")
    current_step_within = agent.get_current_step(parsed_steps, test_time_within)

    # Test 3: Time AFTER window (11:15 AM) - should return None
    test_time_after = datetime.strptime("11:15 AM", "%I:%M %p")
    current_step_after = agent.get_current_step(parsed_steps, test_time_after)

    results = {
        "before": current_step_before is None,
        "within": current_step_within is not None,
        "after": current_step_after is None
    }

    if all(results.values()):
        logger.info("✅ PASS: Time window matching works correctly")
        logger.info(f"  - Before window (9:45 AM): {current_step_before is None}")
        logger.info(f"  - Within window (10:15 AM): {current_step_within is not None}")
        logger.info(f"  - After window (11:15 AM): {current_step_after is None}")
        return True
    else:
        logger.error(f"❌ FAILED: Time window matching incorrect: {results}")
        return False


def main():
    """Run all diagnostic tests."""
    logger.info("\n" + "=" * 70)
    logger.info("DAY 6 FIXES DIAGNOSTIC TEST SUITE")
    logger.info("=" * 70)

    tests = [
        ("get_upcoming_step() method", test_get_upcoming_step),
        ("Wait at location behavior", test_wait_behavior),
        ("Time window matching", test_time_window_matching)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            logger.error(f"❌ EXCEPTION in {test_name}: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.error("❌ SOME TESTS FAILED")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
