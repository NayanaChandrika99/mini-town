"""
Quick event scenario test: 3-minute party test.
Focuses on verifying the fixes work without waiting 30 minutes.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding
from dspy_modules import configure_dspy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


async def quick_party_test():
    """
    Quick 3-minute test:
    - T+0: Agents spawn, invitations sent immediately
    - T+1 min: Plans generated
    - T+2 min: Party time (check attendance)
    """
    logger.info("=" * 70)
    logger.info("QUICK PARTY TEST (3 minutes)")
    logger.info("=" * 70)

    # Configure DSPy
    configure_dspy()

    # Initialize database
    test_db_path = Path(__file__).parent.parent / "data" / "test_quick_party.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    if test_db_path.exists():
        test_db_path.unlink()

    memory_store = MemoryStore(str(test_db_path))

    # Initialize agents
    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']

    # Party location: center of map
    party_x, party_y = 200.0, 150.0

    # Spawn agents at different distances
    agents = [
        Agent(
            agent_id=1,
            name="Alice",
            x=180.0,  # 22px from party (will arrive early)
            y=160.0,
            goal="Meet new people",
            personality="social, curious",
            map_width=map_width,
            map_height=map_height
        ),
        Agent(
            agent_id=2,
            name="Bob",
            x=150.0,  # 56px from party
            y=120.0,
            goal="Complete research project",
            personality="analytical, introverted",
            map_width=map_width,
            map_height=map_height
        )
    ]

    # Create agents in DB
    for agent in agents:
        memory_store.create_agent(
            agent_id=agent.id,
            name=agent.name,
            x=agent.x,
            y=agent.y,
            goal=agent.goal,
            personality=agent.personality
        )

    logger.info(f"\n📍 Agent spawn positions:")
    for agent in agents:
        distance = math.sqrt((agent.x - party_x)**2 + (agent.y - party_y)**2)
        logger.info(f"  {agent.name}: ({agent.x:.0f}, {agent.y:.0f}) - {distance:.1f}px from party")

    # Timeline
    start_time = datetime.now()
    party_time = start_time + timedelta(minutes=2)  # Party in 2 minutes

    logger.info(f"\n⏰ Timeline:")
    logger.info(f"  T+0 (now): Send invitations")
    logger.info(f"  T+30s: Generate plans")
    logger.info(f"  T+2min: Party time ({party_time.strftime('%I:%M:%S %p')})")
    logger.info(f"  T+3min: End test")

    # T+0: Send invitations immediately
    logger.info(f"\n📧 T+0: Sending invitations...")
    invitation_text = (
        f"Maria invited you to a party at {party_time.strftime('%I:%M %p')} "
        f"at location ({party_x:.0f}, {party_y:.0f}). "
        f"This is a social gathering - great opportunity to build relationships!"
    )

    for agent in agents:
        embedding = generate_embedding(invitation_text)
        memory_store.store_memory(
            agent_id=agent.id,
            content=invitation_text,
            importance=0.85,
            embedding=embedding,
            timestamp=start_time
        )
        logger.info(f"  ✅ Invited: {agent.name}")

    # Wait 30 seconds, then generate plans
    logger.info(f"\n⏳ Waiting 30 seconds before generating plans...")
    await asyncio.sleep(30)

    # T+30s: Generate plans
    current_time = datetime.now()
    logger.info(f"\n📋 T+30s: Generating plans...")

    for agent in agents:
        try:
            plan = await agent.update_plan(memory_store, current_time)
            if plan:
                logger.info(f"  ✅ {agent.name} plan:")
                logger.info(f"     {plan[:120]}...")

                # Check if plan mentions correct time
                party_time_str = party_time.strftime('%I:%M %p')
                if party_time_str in plan:
                    logger.info(f"     ✅ Plan mentions correct party time ({party_time_str})")
                else:
                    logger.warning(f"     ⚠️  Plan does NOT mention party time ({party_time_str})")

                # Check if plan mentions location
                if f"({party_x:.0f}, {party_y:.0f})" in plan:
                    logger.info(f"     ✅ Plan mentions correct location ({party_x:.0f}, {party_y:.0f})")
                else:
                    logger.warning(f"     ⚠️  Plan does NOT mention location")
        except Exception as e:
            logger.error(f"  ❌ {agent.name} planning failed: {e}")

    # Simulate agent movement until party time
    logger.info(f"\n🎬 Simulating agent movement until party time...")
    tick_count = 0
    tick_interval = 2.0  # 2 seconds per tick

    while datetime.now() < party_time + timedelta(seconds=30):
        tick_count += 1
        current_time = datetime.now()

        # Update agents
        for agent in agents:
            agent.update(agents)
            memory_store.update_agent_position(agent.id, agent.x, agent.y)

        # Log positions every 15 seconds
        if tick_count % 8 == 0:
            elapsed = (current_time - start_time).total_seconds()
            logger.info(f"\n⏱️  T+{elapsed:.0f}s:")
            for agent in agents:
                distance = math.sqrt((agent.x - party_x)**2 + (agent.y - party_y)**2)
                logger.info(f"  {agent.name}: ({agent.x:.1f}, {agent.y:.1f}) - {distance:.1f}px from party, state: {agent.state}")

        await asyncio.sleep(tick_interval)

    # Final check: Are agents at party?
    logger.info(f"\n🎉 PARTY TIME CHECK (T+{(party_time - start_time).total_seconds():.0f}s):")

    attendance = []
    for agent in agents:
        distance = math.sqrt((agent.x - party_x)**2 + (agent.y - party_y)**2)
        at_party = distance <= 100  # Within 100px radius

        logger.info(f"  {agent.name}:")
        logger.info(f"    Position: ({agent.x:.1f}, {agent.y:.1f})")
        logger.info(f"    Distance from party: {distance:.1f}px")
        logger.info(f"    State: {agent.state}")
        logger.info(f"    At party: {'✅ YES' if at_party else '❌ NO'}")

        if at_party:
            attendance.append(agent.name)

    # Calculate event coherence
    event_coherence = len(attendance) / len(agents)

    logger.info(f"\n📊 RESULTS:")
    logger.info(f"  Event Coherence: {event_coherence:.0%} ({len(attendance)}/{len(agents)} agents attended)")
    logger.info(f"  Attendees: {', '.join(attendance) if attendance else 'None'}")

    # Check if fixes worked
    logger.info(f"\n🔍 FIX VERIFICATION:")

    # Check Fix #1: Did agents wait at location?
    waiting_agents = [a for a in agents if a.state == "waiting"]
    if waiting_agents:
        logger.info(f"  ✅ Fix #1 (Wait behavior): {len(waiting_agents)} agent(s) in 'waiting' state")
    else:
        logger.warning(f"  ⚠️  Fix #1: No agents in 'waiting' state")

    # Check Fix #2: Did plans have correct time?
    # (Already checked during plan generation)

    # Overall assessment
    if event_coherence >= 0.5:
        logger.info(f"\n✅ SUCCESS: Event coherence {event_coherence:.0%} >= 50%")
        logger.info(f"   Fixes appear to be working!")
    elif event_coherence > 0:
        logger.info(f"\n⚠️  PARTIAL SUCCESS: Event coherence {event_coherence:.0%}")
        logger.info(f"   Some improvement, but not optimal")
    else:
        logger.error(f"\n❌ FAILED: Event coherence still 0%")
        logger.error(f"   Fixes may not be working as expected")

    # Cleanup
    memory_store.close()

    return {
        'event_coherence': event_coherence,
        'attendance': attendance,
        'total_agents': len(agents)
    }


async def main():
    try:
        results = await quick_party_test()

        # Save results
        output_path = Path(__file__).parent.parent / "results" / "quick_party_test.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n💾 Results saved to {output_path}")

        return results['event_coherence'] >= 0.5

    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
