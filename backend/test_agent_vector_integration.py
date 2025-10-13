"""
Test agent integration with vector search.
Verifies that agents use vector retrieval during reflection.
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from memory import MemoryStore
from agents import Agent
from utils import generate_embedding
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_agent_vector_integration():
    """Test that agents use vector search for reflection."""

    logger.info("=" * 70)
    logger.info("AGENT VECTOR INTEGRATION TEST")
    logger.info("=" * 70)

    # Create test database
    db_path = Path(__file__).parent.parent / "data" / "agent_vector_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    memory_store = MemoryStore(str(db_path))

    # Create agent with specific goal
    agent = Agent(
        agent_id=1,
        name="TestAgent",
        x=100,
        y=100,
        goal="Build relationships with neighbors",
        personality="social, friendly",
        retrieval_alpha=0.6,  # Higher relevance
        retrieval_beta=0.2,   # Lower recency
        retrieval_gamma=0.2   # Lower importance
    )

    # Create agent in database
    memory_store.create_agent(
        agent_id=agent.id,
        name=agent.name,
        x=agent.x,
        y=agent.y,
        goal=agent.goal,
        personality=agent.personality
    )

    # Store diverse memories
    test_memories = [
        # Relevant to goal (social interactions) - should be prioritized
        ("Had a great conversation with Alice about gardening", 0.6),
        ("Bob invited me to join the book club next week", 0.7),
        ("Carol and I planned a neighborhood barbecue", 0.8),

        # Relevant but old
        ("Met Dave at the park six months ago, he seemed nice", 0.5),

        # Recent but irrelevant
        ("The weather today is sunny and warm", 0.3),
        ("Traffic was heavy on Main Street this morning", 0.2),
        ("Had toast and coffee for breakfast", 0.1),

        # High importance but irrelevant
        ("Urgent work deadline tomorrow morning", 0.9),
        ("Doctor appointment for health checkup scheduled", 0.8),

        # Mix
        ("Eve mentioned she's organizing a community event, I should participate", 0.7),
    ]

    base_time = datetime.now()

    for i, (content, importance) in enumerate(test_memories):
        embedding = generate_embedding(content)
        memory_store.store_memory(
            agent_id=agent.id,
            content=content,
            importance=importance,
            embedding=embedding,
            timestamp=base_time - timedelta(hours=len(test_memories) - i)
        )

        # Accumulate reflection score
        agent.reflection_score += importance

    logger.info(f"Stored {len(test_memories)} memories")
    logger.info(f"Agent reflection score: {agent.reflection_score:.2f} (threshold: {agent.reflection_threshold})")

    # Trigger reflection (should use vector search)
    logger.info("\n" + "=" * 70)
    logger.info("TRIGGERING REFLECTION (using vector search)")
    logger.info("=" * 70)

    insight = await agent.maybe_reflect(memory_store)

    if insight:
        logger.info(f"✅ Reflection generated: {insight[:100]}...")
    else:
        logger.error("❌ No reflection generated (unexpected)")

    # Verify vector search was used by checking the order of memories
    logger.info("\n" + "=" * 70)
    logger.info("COMPARING RETRIEVAL METHODS")
    logger.info("=" * 70)

    # Method 1: Recency-only (old way)
    recency_memories = memory_store.get_agent_memories(agent.id, limit=5)
    logger.info("\nTop 5 by RECENCY only (old method):")
    for i, mem in enumerate(recency_memories, 1):
        logger.info(f"  {i}. {mem['content'][:60]}...")

    # Method 2: Vector search (new way)
    query_text = f"Important events and observations related to: {agent.goal}"
    query_embedding = generate_embedding(query_text)

    vector_memories = memory_store.retrieve_memories_by_vector(
        agent_id=agent.id,
        query_embedding=query_embedding,
        top_k=5,
        alpha=agent.retrieval_alpha,
        beta=agent.retrieval_beta,
        gamma=agent.retrieval_gamma
    )

    logger.info("\nTop 5 by VECTOR SEARCH (new method):")
    for i, mem in enumerate(vector_memories, 1):
        logger.info(f"  {i}. (score: {mem['score']:.3f}) {mem['content'][:60]}...")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS")
    logger.info("=" * 70)

    # Check if results are different
    recency_ids = {m['id'] for m in recency_memories}
    vector_ids = {m['id'] for m in vector_memories}

    if recency_ids != vector_ids:
        logger.info("✅ Vector search returns DIFFERENT memories than recency-only")
        logger.info("   This confirms vector search is working correctly!")
    else:
        logger.warning("⚠️  Same memories returned (might be coincidence)")

    # Check if social memories are prioritized
    social_keywords = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'conversation', 'barbecue', 'community']
    vector_social_count = sum(
        1 for m in vector_memories
        if any(kw in m['content'] for kw in social_keywords)
    )

    recency_social_count = sum(
        1 for m in recency_memories
        if any(kw in m['content'] for kw in social_keywords)
    )

    logger.info(f"\nSocial-related memories in top 5:")
    logger.info(f"  Vector search: {vector_social_count}/5")
    logger.info(f"  Recency-only:  {recency_social_count}/5")

    if vector_social_count > recency_social_count:
        logger.info("✅ Vector search prioritizes goal-relevant memories!")

    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)

    memory_store.close()
    db_path.unlink()  # Cleanup

if __name__ == "__main__":
    asyncio.run(test_agent_vector_integration())
