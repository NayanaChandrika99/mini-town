"""
Test script for Day 1: Vector search and triad scoring.
Generates 100 test phrases, stores them with embeddings, and tests retrieval.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory import MemoryStore
from utils import generate_embeddings_batch, generate_embedding, format_memory_for_display

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test data: 100 diverse phrases across different categories
TEST_PHRASES = [
    # Social interactions (20)
    "Alice invited me to a party at 7pm tonight",
    "Bob said hello when we met at the park",
    "Carol asked if I wanted to join the book club",
    "Dave and I had a great conversation about music",
    "Eve mentioned she's organizing a community event",
    "I saw Alice at the coffee shop this morning",
    "Bob looked upset when I asked about his day",
    "Carol complimented my new hairstyle",
    "Dave invited everyone to his birthday celebration",
    "Eve shared some interesting gossip about the neighborhood",
    "Alice seems very social and friendly",
    "Bob prefers to keep to himself",
    "Carol is always punctual for meetings",
    "Dave has an impulsive personality",
    "Eve is curious about everyone's stories",
    "I feel happy when talking to Alice",
    "Bob makes me think deeply about things",
    "Carol helps me stay organized",
    "Dave brings excitement to the group",
    "Eve makes me want to explore new ideas",

    # Environmental observations (15)
    "The weather is sunny and warm today",
    "It's raining heavily outside",
    "The temperature dropped significantly overnight",
    "Beautiful sunset with orange and pink colors",
    "Strong winds are blowing through the trees",
    "The park looks especially green after the rain",
    "Snow is falling gently from the sky",
    "The moon is full and bright tonight",
    "Clouds are forming in the distance",
    "Perfect weather for a walk outside",
    "The flowers in the garden are blooming",
    "Autumn leaves are falling everywhere",
    "The air feels crisp and fresh",
    "Thunder and lightning in the distance",
    "Morning dew covers the grass",

    # Goal-related (20)
    "I need to finish my research project by Friday",
    "Making progress on understanding local history",
    "The community garden needs watering today",
    "I want to document more neighborhood stories",
    "Building relationships is important to me",
    "I should organize my notes better",
    "Time to plan my daily schedule",
    "I need to follow up on that invitation",
    "Making every day an adventure is my goal",
    "I want to learn more about my neighbors",
    "The research is going well so far",
    "I've been neglecting the garden lately",
    "Many stories still need to be collected",
    "I met three new people this week",
    "My notes are getting disorganized",
    "I accomplished my main task today",
    "Need to find more time for socializing",
    "The project deadline is approaching",
    "I'm making good progress on my goals",
    "Everything is going according to plan",

    # Emotional states (15)
    "I feel really happy and energized today",
    "Feeling a bit anxious about the upcoming event",
    "So excited about the party tonight",
    "I'm feeling lonely and isolated",
    "Grateful for having such good neighbors",
    "Frustrated that my plans got canceled",
    "Feeling calm and peaceful right now",
    "Worried about finishing everything on time",
    "Proud of what I accomplished today",
    "Disappointed by the weather forecast",
    "Feeling curious and adventurous",
    "A bit overwhelmed with all the tasks",
    "Content and satisfied with my progress",
    "Nervous about meeting new people",
    "Joy from helping others today",

    # Events and activities (20)
    "There's a party at Maria's house tonight",
    "The book club meets every Thursday",
    "Community garden work happens on Saturdays",
    "Fire safety drill scheduled for tomorrow",
    "Annual neighborhood picnic next month",
    "Weekly farmers market opens at 8am",
    "Carol is hosting a potluck dinner",
    "Bob is giving a talk about history",
    "Dave organized a music jam session",
    "Eve is starting a photography club",
    "The library closes early on Sundays",
    "Town hall meeting about new playground",
    "Alice is teaching a cooking class",
    "Yoga in the park every Wednesday morning",
    "Movie night at the community center",
    "Volunteer day for beach cleanup",
    "Carol's birthday party next week",
    "Holiday decorations going up soon",
    "Spring festival planning committee",
    "Summer concert series announced",

    # Mundane/background (10)
    "I had toast and coffee for breakfast",
    "Need to buy groceries later",
    "The mail arrived at noon today",
    "My phone battery is running low",
    "I should do laundry this weekend",
    "Traffic was heavy on Main Street",
    "Heard a dog barking in the distance",
    "The streetlight outside is flickering",
    "Someone left a package at my door",
    "The TV weather forecast was wrong again",
]


def test_vector_search():
    """Main test function for vector search capabilities."""

    logger.info("=" * 60)
    logger.info("Day 1 Vector Search Test")
    logger.info("=" * 60)

    # Initialize database
    db_path = Path(__file__).parent.parent / "data" / "town_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing test database
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing test database")

    memory_store = MemoryStore(str(db_path))

    # Create a test agent
    memory_store.create_agent(
        agent_id=999,
        name="TestAgent",
        x=100,
        y=100,
        goal="Test vector search",
        personality="test"
    )

    logger.info(f"\nGenerating embeddings for {len(TEST_PHRASES)} test phrases...")

    # Generate embeddings for all phrases
    embeddings = generate_embeddings_batch(TEST_PHRASES)
    logger.info(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

    # Store memories with embeddings at different times
    logger.info("\nStoring memories with varying timestamps and importance...")
    base_time = datetime.now() - timedelta(days=7)  # Start 7 days ago

    for i, (phrase, embedding) in enumerate(zip(TEST_PHRASES, embeddings)):
        # Vary timestamp (spread over past week)
        timestamp = base_time + timedelta(hours=i * 1.5)

        # Vary importance (random but weighted by category)
        if "party" in phrase.lower() or "event" in phrase.lower():
            importance = random.uniform(0.7, 1.0)  # High importance
        elif "weather" in phrase.lower() or "breakfast" in phrase.lower():
            importance = random.uniform(0.1, 0.4)  # Low importance
        else:
            importance = random.uniform(0.4, 0.7)  # Medium importance

        memory_store.store_memory(
            agent_id=999,
            content=phrase,
            importance=importance,
            embedding=embedding,
            timestamp=timestamp
        )

    logger.info(f"Stored {len(TEST_PHRASES)} memories successfully")

    # Test queries
    test_queries = [
        "party and social gatherings",
        "weather conditions",
        "emotional feelings",
        "daily tasks and goals",
        "Alice and friends"
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Testing Vector Search with Triad Scoring")
    logger.info("=" * 60)

    for query in test_queries:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"{'=' * 60}")

        # Generate query embedding
        query_embedding = generate_embedding(query)

        # Test with default weights
        results = memory_store.retrieve_memories_by_vector(
            agent_id=999,
            query_embedding=query_embedding,
            top_k=5,
            alpha=0.5,  # relevance
            beta=0.3,   # recency
            gamma=0.2   # importance
        )

        logger.info(f"\nTop 5 results (α=0.5, β=0.3, γ=0.2):")
        for i, mem in enumerate(results, 1):
            logger.info(f"\n{i}. {format_memory_for_display(mem)}")
            logger.info(f"   Relevance: {mem['relevance']:.3f} | "
                       f"Recency: {mem['recency']:.3f} | "
                       f"Importance: {mem['importance']:.3f}")

    # Test different weight configurations
    logger.info("\n" + "=" * 60)
    logger.info("Testing Different Weight Configurations")
    logger.info("=" * 60)

    query = "party tonight"
    query_embedding = generate_embedding(query)

    weight_configs = [
        (0.7, 0.2, 0.1, "High relevance"),
        (0.3, 0.6, 0.1, "High recency"),
        (0.3, 0.1, 0.6, "High importance"),
    ]

    for alpha, beta, gamma, description in weight_configs:
        logger.info(f"\n{description} (α={alpha}, β={beta}, γ={gamma}):")
        results = memory_store.retrieve_memories_by_vector(
            agent_id=999,
            query_embedding=query_embedding,
            top_k=3,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        for i, mem in enumerate(results, 1):
            logger.info(f"{i}. {format_memory_for_display(mem)}")

    logger.info("\n" + "=" * 60)
    logger.info("Vector Search Test Complete!")
    logger.info("=" * 60)

    memory_store.close()


if __name__ == "__main__":
    test_vector_search()
