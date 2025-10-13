"""
Investigate why Bob's plan doesn't preserve party time.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding
from dspy_modules import configure_dspy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


async def investigate_bob():
    """
    Investigate Bob's planning behavior step by step.
    """
    logger.info("=" * 70)
    logger.info("INVESTIGATING BOB'S PLANNING")
    logger.info("=" * 70)

    # Configure DSPy
    configure_dspy()

    # Initialize database
    test_db_path = Path(__file__).parent.parent / "data" / "test_bob_investigation.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    if test_db_path.exists():
        test_db_path.unlink()

    memory_store = MemoryStore(str(test_db_path))

    # Create Bob
    bob = Agent(
        agent_id=1,
        name="Bob",
        x=150.0,
        y=120.0,
        goal="Complete research project",
        personality="analytical, introverted",
        map_width=config['simulation']['map_width'],
        map_height=config['simulation']['map_height']
    )

    memory_store.create_agent(
        agent_id=bob.id,
        name=bob.name,
        x=bob.x,
        y=bob.y,
        goal=bob.goal,
        personality=bob.personality
    )

    logger.info(f"\n👤 Agent: {bob.name}")
    logger.info(f"   Goal: {bob.goal}")
    logger.info(f"   Personality: {bob.personality}")

    # Send invitation
    current_time = datetime.now()
    party_time = current_time + timedelta(minutes=2)

    invitation_text = (
        f"Maria invited you to a party at {party_time.strftime('%I:%M %p')} "
        f"at location (200, 150). "
        f"This is a social gathering - great opportunity to build relationships!"
    )

    logger.info(f"\n📧 Storing invitation:")
    logger.info(f"   {invitation_text}")

    embedding = generate_embedding(invitation_text)
    memory_store.store_memory(
        agent_id=bob.id,
        content=invitation_text,
        importance=0.85,
        embedding=embedding,
        timestamp=current_time
    )

    # Check what query Bob will use
    logger.info(f"\n🔍 Query generation logic:")
    personality_lower = bob.personality.lower()
    if "introverted" in personality_lower or "analytical" in personality_lower or "reclusive" in personality_lower:
        query_text = f"Explicit invitations addressed to me. Events I was specifically invited to. Goal: {bob.goal}"
        logger.info(f"   ✅ Using INTROVERTED query (Fix #4)")
    else:
        query_text = f"Recent party invitations, social events, and plans. Goal: {bob.goal}"
        logger.info(f"   Using SOCIAL query")

    logger.info(f"\n   Query text: \"{query_text}\"")

    # Test retrieval
    logger.info(f"\n🧪 Testing memory retrieval...")
    query_embedding = generate_embedding(query_text)

    memories = memory_store.retrieve_memories_by_vector(
        agent_id=bob.id,
        query_embedding=query_embedding,
        top_k=8,
        alpha=0.4,  # relevance
        beta=0.4,   # recency
        gamma=0.2   # importance
    )

    logger.info(f"   Retrieved {len(memories)} memories:")
    for i, mem in enumerate(memories, 1):
        logger.info(f"   {i}. (score: {mem.get('score', 0):.3f}) {mem['content'][:80]}...")

    # Check if invitation is in retrieved memories
    invitation_found = any('invited' in m['content'].lower() or 'party' in m['content'].lower()
                          for m in memories)

    if invitation_found:
        logger.info(f"\n   ✅ Invitation FOUND in top-8 retrieved memories")
    else:
        logger.error(f"\n   ❌ Invitation NOT FOUND in top-8 retrieved memories")

    # Generate plan
    logger.info(f"\n📋 Generating Bob's plan...")
    logger.info(f"   Current time: {current_time.strftime('%I:%M %p')}")
    logger.info(f"   Party time: {party_time.strftime('%I:%M %p')}")

    # Get the recent events and memories that will be passed to the LLM
    recent_cutoff = current_time - timedelta(minutes=10)
    recent_events = []
    relevant_memories = []

    for mem in memories:
        mem_time = mem.get('ts', mem.get('timestamp'))
        if mem_time and mem_time >= recent_cutoff:
            content = mem['content']
            if any(keyword in content.lower() for keyword in ['invited', 'party', 'event', 'meeting', 'gathering']):
                recent_events.append(content)
            else:
                relevant_memories.append(content)
        else:
            relevant_memories.append(mem['content'])

    recent_events = recent_events[:5]
    relevant_memories = relevant_memories[:5]

    logger.info(f"\n   Recent events passed to LLM ({len(recent_events)}):")
    for i, event in enumerate(recent_events, 1):
        logger.info(f"   {i}. {event[:100]}...")

    logger.info(f"\n   Relevant memories passed to LLM ({len(relevant_memories)}):")
    for i, mem in enumerate(relevant_memories, 1):
        logger.info(f"   {i}. {mem[:100]}...")

    # Generate plan
    plan = await bob.update_plan(memory_store, current_time)

    logger.info(f"\n📝 Generated plan:")
    logger.info(f"   {plan}")

    # Check if plan mentions party time
    party_time_str = party_time.strftime('%I:%M %p')
    if party_time_str in plan:
        logger.info(f"\n   ✅ Plan mentions correct party time ({party_time_str})")
    else:
        logger.error(f"\n   ❌ Plan does NOT mention party time ({party_time_str})")
        logger.error(f"\n   🔍 Investigating why...")

        # Check if LLM received the time
        if recent_events and party_time_str in recent_events[0]:
            logger.error(f"      ✅ Party time WAS in recent_events passed to LLM")
            logger.error(f"      ❌ LLM ignored or rescheduled the party time")
            logger.error(f"      💡 Possible cause: LLM instruction not strong enough")
        else:
            logger.error(f"      ❌ Party time NOT in recent_events")
            logger.error(f"      💡 Possible cause: Memory retrieval or filtering issue")

    # Check if plan mentions location
    if "(200, 150)" in plan:
        logger.info(f"   ✅ Plan mentions correct location (200, 150)")
    else:
        logger.warning(f"   ⚠️  Plan does NOT mention location (200, 150)")

    # Cleanup
    memory_store.close()

    return plan


async def main():
    try:
        await investigate_bob()
        return True
    except Exception as e:
        logger.error(f"Investigation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
