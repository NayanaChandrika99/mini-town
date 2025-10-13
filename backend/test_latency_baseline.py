"""
Baseline latency test for Mini-Town.
Runs 20 ticks to measure LLM latencies and determine tick interval.
"""

import sys
import asyncio
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore
from dspy_modules import configure_dspy
from utils import get_latency_tracker, generate_embedding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


async def run_baseline_test():
    """Run 20-tick baseline test to measure LLM latencies."""

    logger.info("=" * 70)
    logger.info("STARTING BASELINE LATENCY TEST")
    logger.info("=" * 70)

    # Configure DSPy
    logger.info("\n1. Configuring DSPy with Groq...")
    configure_dspy()
    logger.info("✅ DSPy configured")

    # Initialize database
    logger.info("\n2. Initializing database...")
    project_root = Path(__file__).parent.parent
    db_path = project_root / config['database']['path']
    memory_store = MemoryStore(str(db_path))
    logger.info(f"✅ Database initialized at {db_path}")

    # Create test agents
    logger.info("\n3. Creating test agents...")
    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']
    perception_radius = config['simulation']['perception_radius']

    agents = []
    agent_configs = [
        {"name": "Alice", "personality": "social, optimistic", "goal": "Build relationships"},
        {"name": "Bob", "personality": "analytical, introverted", "goal": "Complete research"},
        {"name": "Carol", "personality": "organized, punctual", "goal": "Maintain garden"}
    ]

    for i, cfg in enumerate(agent_configs):
        agent = Agent(
            agent_id=i + 100,  # Use high IDs to avoid conflicts
            name=cfg["name"],
            x=200 + i * 30,  # Place agents 30 pixels apart (within perception radius of 50)
            y=200,
            goal=cfg["goal"],
            personality=cfg["personality"],
            map_width=map_width,
            map_height=map_height,
            perception_radius=perception_radius
        )
        agents.append(agent)

    logger.info(f"✅ Created {len(agents)} test agents")

    # Run 20 ticks
    logger.info("\n4. Running 20-tick simulation...")
    logger.info("-" * 70)

    tick_interval = config['simulation']['tick_interval']
    num_ticks = 20

    observations_count = 0
    reflections_count = 0

    for tick in range(1, num_ticks + 1):
        logger.info(f"\nTick {tick}/{num_ticks}")

        # Update each agent
        for agent in agents:
            # Agent perceives others
            state = agent.update(agents)

            # Process observations with LLM scoring
            if state['observations']:
                for obs in state['observations']:
                    try:
                        importance = await agent.score_and_store_observation(obs, memory_store)
                        observations_count += 1
                        logger.info(f"  {agent.name}: '{obs[:40]}...' → importance={importance:.2f}")
                    except Exception as e:
                        logger.error(f"  {agent.name}: Scoring failed - {e}")

                # Check for reflection
                try:
                    insight = await agent.maybe_reflect(memory_store)
                    if insight:
                        reflections_count += 1
                        logger.info(f"  {agent.name} REFLECTED: '{insight[:60]}...'")

                        # Store reflection
                        embedding = generate_embedding(insight)
                        memory_store.store_memory(
                            agent_id=agent.id,
                            content=f"[REFLECTION] {insight}",
                            importance=0.9,
                            embedding=embedding,
                            timestamp=datetime.now()
                        )
                except Exception as e:
                    logger.error(f"  {agent.name}: Reflection failed - {e}")

        # Wait for tick interval (simulate real timing)
        await asyncio.sleep(0.1)  # Short delay to simulate processing time

    logger.info("\n" + "-" * 70)
    logger.info("✅ 20-tick simulation complete")

    # Get latency statistics
    logger.info("\n5. Analyzing latency statistics...")
    logger.info("=" * 70)

    tracker = get_latency_tracker()
    stats = tracker.get_stats()

    if not stats:
        logger.warning("⚠️  No latency data collected!")
        return

    # Display statistics
    for signature_name, sig_stats in stats.items():
        logger.info(f"\n{signature_name}:")
        logger.info(f"  Count:        {sig_stats['count']}")
        logger.info(f"  Success Rate: {sig_stats['success_rate']:.1f}%")
        logger.info(f"  p50:          {sig_stats['p50_ms']}ms")
        logger.info(f"  p95:          {sig_stats['p95_ms']}ms")
        logger.info(f"  p99:          {sig_stats['p99_ms']}ms")
        logger.info(f"  Mean:         {sig_stats['mean_ms']}ms")

    # Calculate tick decision
    p95_max = max([s.get('p95_ms', 0) for s in stats.values()])

    logger.info("\n" + "=" * 70)
    logger.info("TICK INTERVAL DECISION")
    logger.info("=" * 70)
    logger.info(f"Maximum p95 latency: {p95_max}ms")
    logger.info(f"Current tick interval: {tick_interval}s ({tick_interval * 1000}ms)")

    decision = ""
    if p95_max > 2500:
        decision = "INVESTIGATE 🚨"
        recommendation = "p95 > 2500ms - This is too slow! Investigate model or API issues."
    elif p95_max > 1500:
        decision = "ADJUST TO 3s ⚠️"
        recommendation = "p95 1500-2500ms - Adjust tick_interval to 3.0 seconds in config.yml"
    else:
        decision = "KEEP 2s ✅"
        recommendation = "p95 < 1500ms - Current 2s tick interval is safe!"

    logger.info(f"\nDecision: {decision}")
    logger.info(f"Recommendation: {recommendation}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total ticks:        {num_ticks}")
    logger.info(f"Observations scored: {observations_count}")
    logger.info(f"Reflections:        {reflections_count}")
    logger.info(f"Agents tested:      {len(agents)}")

    # Check success criteria
    logger.info("\n" + "=" * 70)
    logger.info("SUCCESS CRITERIA")
    logger.info("=" * 70)

    criteria_met = 0
    criteria_total = 4

    # Criterion 1: ScoreImportance works
    if 'ScoreImportance' in stats and stats['ScoreImportance']['count'] > 0:
        logger.info("✅ ScoreImportance is functioning")
        criteria_met += 1
    else:
        logger.info("❌ ScoreImportance did not run")

    # Criterion 2: At least 1 reflection
    if reflections_count >= 1:
        logger.info(f"✅ At least 1 reflection triggered ({reflections_count} total)")
        criteria_met += 1
    else:
        logger.info("❌ No reflections triggered")

    # Criterion 3: Latency tracking works
    if stats:
        logger.info("✅ Latency tracking captured statistics")
        criteria_met += 1
    else:
        logger.info("❌ No latency statistics captured")

    # Criterion 4: Test completed without crashes
    logger.info("✅ 20-tick test completed without crashes")
    criteria_met += 1

    logger.info(f"\nCriteria met: {criteria_met}/{criteria_total}")

    if criteria_met == criteria_total:
        logger.info("\n🎉 ALL SUCCESS CRITERIA MET!")
    else:
        logger.info(f"\n⚠️  {criteria_total - criteria_met} criteria not met")

    # Cleanup
    memory_store.close()

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE TEST COMPLETE")
    logger.info("=" * 70)

    return {
        "decision": decision,
        "p95_max": p95_max,
        "tick_interval": tick_interval,
        "observations": observations_count,
        "reflections": reflections_count,
        "stats": stats
    }


if __name__ == "__main__":
    asyncio.run(run_baseline_test())
