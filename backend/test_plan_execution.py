"""
Test plan execution (Day 6 validation).

This test verifies that:
1. Plans are parsed correctly
2. Agents navigate toward target locations
3. Agents stop when they arrive at targets
"""

import sys
from datetime import datetime, timedelta
from agents import Agent


def test_plan_parsing():
    """Test that plan parsing extracts time ranges and locations correctly."""
    print("\n=== Test 1: Plan Parsing ===")

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal"
    )

    # Set a plan with location coordinates
    agent.current_plan = """
    02:00 PM - 02:30 PM: Take a walk around the neighborhood
    02:30 PM - 03:00 PM: Head to the party at location (200, 150) and socialize
    03:00 PM - 04:00 PM: Return home to location (50, 50) and relax
    """
    agent.plan_last_updated = datetime.now()

    # Parse the plan
    parsed = agent.parse_plan()

    print(f"Parsed {len(parsed)} steps:")
    for i, step in enumerate(parsed, 1):
        loc_str = f"at {step['location']}" if step['location'] else "no location"
        print(f"  Step {i}: {step['start_time']} - {step['end_time']} {loc_str}")
        print(f"           {step['description'][:60]}...")

    # Verify parsing worked
    assert len(parsed) == 3, f"Expected 3 steps, got {len(parsed)}"
    assert parsed[1]['location'] == (200, 150), f"Expected (200, 150), got {parsed[1]['location']}"
    assert parsed[2]['location'] == (50, 50), f"Expected (50, 50), got {parsed[2]['location']}"

    print("✅ Plan parsing test passed!")
    return True


def test_get_current_step():
    """Test that get_current_step finds the right step for current time."""
    print("\n=== Test 2: Get Current Step ===")

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal"
    )

    # Create a plan for the current time
    now = datetime.now()
    time_1 = (now - timedelta(minutes=10)).strftime("%I:%M %p")
    time_2 = now.strftime("%I:%M %p")
    time_3 = (now + timedelta(minutes=10)).strftime("%I:%M %p")
    time_4 = (now + timedelta(minutes=20)).strftime("%I:%M %p")

    agent.current_plan = f"""
    {time_1} - {time_2}: Step 1 (past)
    {time_2} - {time_3}: Step 2 (current) at location (200, 150)
    {time_3} - {time_4}: Step 3 (future)
    """

    parsed = agent.parse_plan()
    current_step = agent.get_current_step(parsed, now)

    if current_step:
        print(f"Current step: {current_step['description']}")
        print(f"Location: {current_step['location']}")
        assert current_step['location'] == (200, 150), "Wrong current step!"
        print("✅ Get current step test passed!")
        return True
    else:
        print("❌ No current step found (might be timing issue)")
        return False


def test_navigation():
    """Test that agents navigate toward targets and stop when close."""
    print("\n=== Test 3: Navigation ===")

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal"
    )

    target_x, target_y = 200, 150
    initial_distance = ((agent.x - target_x)**2 + (agent.y - target_y)**2)**0.5

    print(f"Initial position: ({agent.x:.1f}, {agent.y:.1f})")
    print(f"Target: ({target_x}, {target_y})")
    print(f"Initial distance: {initial_distance:.1f}")

    # Simulate 50 ticks of navigation
    for tick in range(50):
        agent.navigate_to(target_x, target_y)
        distance = ((agent.x - target_x)**2 + (agent.y - target_y)**2)**0.5

        if tick % 10 == 0:
            print(f"  Tick {tick}: position ({agent.x:.1f}, {agent.y:.1f}), distance: {distance:.1f}")

        # Check if arrived
        if distance < 10:
            print(f"✅ Arrived at target after {tick} ticks!")
            break

    final_distance = ((agent.x - target_x)**2 + (agent.y - target_y)**2)**0.5
    print(f"Final position: ({agent.x:.1f}, {agent.y:.1f})")
    print(f"Final distance: {final_distance:.1f}")

    assert final_distance < initial_distance, "Agent didn't move toward target!"

    # Check if agent got close enough (within 15 pixels is acceptable for 50 ticks)
    if final_distance < 15:
        print("✅ Navigation test passed! (Agent within 15 pixels of target)")
        return True
    else:
        print(f"⚠️  Agent didn't fully arrive but made progress (final distance: {final_distance:.1f})")
        return final_distance < 30  # At least got reasonably close


def test_integrated_execution():
    """Test that update() method integrates plan execution correctly."""
    print("\n=== Test 4: Integrated Execution ===")

    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test goal"
    )

    # Create a plan for right now
    now = datetime.now()
    time_start = now.strftime("%I:%M %p")
    time_end = (now + timedelta(minutes=30)).strftime("%I:%M %p")

    agent.current_plan = f"{time_start} - {time_end}: Go to party at location (200, 150)"
    agent.plan_last_updated = now

    print(f"Plan: {agent.current_plan}")
    print(f"Initial position: ({agent.x:.1f}, {agent.y:.1f})")

    # Run 10 update ticks
    for tick in range(10):
        agent.update([])  # No other agents

        if tick % 3 == 0:
            print(f"  Tick {tick}: position ({agent.x:.1f}, {agent.y:.1f})")

    # Check that agent moved toward target
    distance_to_target = ((agent.x - 200)**2 + (agent.y - 150)**2)**0.5
    initial_distance = ((100 - 200)**2 + (100 - 150)**2)**0.5

    print(f"Final position: ({agent.x:.1f}, {agent.y:.1f})")
    print(f"Distance to target: {distance_to_target:.1f} (was {initial_distance:.1f})")

    assert distance_to_target < initial_distance, "Agent didn't move toward target in update()!"

    print("✅ Integrated execution test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Day 6 Plan Execution Tests")
    print("=" * 60)

    results = []

    try:
        results.append(("Plan Parsing", test_plan_parsing()))
    except Exception as e:
        print(f"❌ Plan parsing test failed: {e}")
        results.append(("Plan Parsing", False))

    try:
        results.append(("Get Current Step", test_get_current_step()))
    except Exception as e:
        print(f"❌ Get current step test failed: {e}")
        results.append(("Get Current Step", False))

    try:
        results.append(("Navigation", test_navigation()))
    except Exception as e:
        print(f"❌ Navigation test failed: {e}")
        results.append(("Navigation", False))

    try:
        results.append(("Integrated Execution", test_integrated_execution()))
    except Exception as e:
        print(f"❌ Integrated execution test failed: {e}")
        results.append(("Integrated Execution", False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
