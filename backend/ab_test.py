"""
A/B Testing framework for compiled vs uncompiled agents.
Day 5: Compare performance of GEPA-compiled vs baseline modules.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding, get_latency_tracker
from dspy_modules import configure_dspy, use_compiled, load_compiled_modules
from metrics import calculate_scenario_metrics, format_metrics_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


# ============ Scenario Runner ============

async def run_scenario(
    use_compiled_modules: bool,
    duration_minutes: int = 20,
    scenario_name: str = "baseline"
) -> dict:
    """
    Run a simulation scenario with compiled or uncompiled modules.

    Args:
        use_compiled_modules: True to use compiled scorer, False for baseline
        duration_minutes: How long to run simulation
        scenario_name: Name for logging/results

    Returns:
        Dictionary with scenario results and metrics
    """
    logger.info("=" * 70)
    logger.info(f"RUNNING SCENARIO: {scenario_name}")
    logger.info(f"Mode: {'COMPILED' if use_compiled_modules else 'UNCOMPILED'}")
    logger.info(f"Duration: {duration_minutes} minutes")
    logger.info("=" * 70)

    # Configure DSPy
    configure_dspy()

    # Load compiled modules if requested
    if use_compiled_modules:
        success = load_compiled_modules(config['compilation']['compiled_dir'])
        if not success:
            logger.error("Failed to load compiled modules, aborting")
            return None
        use_compiled(True)
        logger.info("✅ Using compiled modules")
    else:
        use_compiled(False)
        logger.info("Using uncompiled baseline modules")

    # Initialize database (use temp DB for testing)
    test_db_path = Path(__file__).parent.parent / "data" / f"test_{scenario_name}.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old test DB if exists
    if test_db_path.exists():
        test_db_path.unlink()

    memory_store = MemoryStore(str(test_db_path))
    logger.info(f"Initialized test database: {test_db_path}")

    # Initialize agents
    agents = initialize_test_agents(memory_store)
    logger.info(f"Initialized {len(agents)} test agents")

    # Run simulation
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    tick_count = 0
    tick_interval = config['simulation']['tick_interval']

    observation_count = 0
    reflection_count = 0

    logger.info(f"Starting simulation loop (target: {duration_minutes} min)...")

    while datetime.now() < end_time:
        tick_count += 1

        # Update each agent
        for agent in agents:
            # Pass other agents for perception
            state = agent.update(agents)

            # Update position in database
            memory_store.update_agent_position(agent.id, agent.x, agent.y)

            # Store observations as memories with LLM-based importance scoring
            if state['observations']:
                for obs in state['observations']:
                    try:
                        importance = await agent.score_and_store_observation(obs, memory_store)
                        observation_count += 1
                        logger.debug(f"Agent {agent.id} scored '{obs[:30]}...' = {importance:.2f}")
                    except Exception as e:
                        logger.error(f"Error scoring observation: {e}")
                        continue

                # Check for reflection
                try:
                    insight = await agent.maybe_reflect(memory_store)
                    if insight:
                        reflection_count += 1
                        # Store insight as special memory
                        embedding = generate_embedding(insight)
                        memory_store.store_memory(
                            agent_id=agent.id,
                            content=f"[REFLECTION] {insight}",
                            importance=0.9,
                            embedding=embedding,
                            timestamp=datetime.now()
                        )
                        logger.info(f"Agent {agent.id} reflected: {insight[:60]}...")
                except Exception as e:
                    logger.error(f"Error generating reflection: {e}")

        # Wait for next tick
        await asyncio.sleep(tick_interval)

        # Log progress every 30 ticks
        if tick_count % 30 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            logger.info(f"Tick {tick_count} ({elapsed:.1f} min elapsed)")

    # Simulation complete
    actual_duration = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"✅ Simulation complete: {tick_count} ticks in {actual_duration:.1f} minutes")

    # Gather results
    results = {
        'scenario_name': scenario_name,
        'use_compiled': use_compiled_modules,
        'start_time': start_time.isoformat(),
        'duration_minutes': actual_duration,
        'tick_count': tick_count,
        'observation_count': observation_count,
        'reflection_count': reflection_count,
        'agents': []
    }

    # Get agent stats
    for agent in agents:
        memories = memory_store.get_agent_memories(agent.id, limit=100)
        results['agents'].append({
            'id': agent.id,
            'name': agent.name,
            'final_position': {'x': agent.x, 'y': agent.y},
            'memory_count': len(memories),
            'reflection_sum': agent.reflection_score
        })

    # Get latency stats
    latency_stats = get_latency_tracker().get_stats()
    results['latency'] = latency_stats

    # Cleanup
    memory_store.close()

    logger.info(f"Total observations: {observation_count}")
    logger.info(f"Total reflections: {reflection_count}")

    return results


def initialize_test_agents(memory_store: MemoryStore) -> List[Agent]:
    """
    Initialize test agents for scenario.

    Returns:
        List of Agent instances
    """
    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']
    perception_radius = config['simulation']['perception_radius']

    # Agent configs (use same as main.py for consistency)
    agent_configs = [
        {"name": "Alice", "personality": "social, optimistic", "goal": "Build relationships in the neighborhood"},
        {"name": "Bob", "personality": "analytical, introverted", "goal": "Complete research project on local history"},
        {"name": "Carol", "personality": "organized, punctual", "goal": "Maintain community garden"}
    ]

    agents = []
    import random
    random.seed(42)  # Reproducible positions

    # Spawn agents in town center cluster (within 100x100 area)
    center_x = map_width / 2
    center_y = map_height / 2
    cluster_size = 100

    for i, config_data in enumerate(agent_configs):
        # Random position within cluster, but not on top of each other
        x = center_x + random.uniform(-cluster_size/2, cluster_size/2)
        y = center_y + random.uniform(-cluster_size/2, cluster_size/2)

        agent = Agent(
            agent_id=i + 1,
            name=config_data["name"],
            x=x,
            y=y,
            goal=config_data["goal"],
            personality=config_data["personality"],
            map_width=map_width,
            map_height=map_height,
            perception_radius=perception_radius
        )
        agents.append(agent)

        # Store agent in database
        memory_store.create_agent(
            agent_id=agent.id,
            name=agent.name,
            x=agent.x,
            y=agent.y,
            goal=agent.goal,
            personality=agent.personality
        )

    return agents


def cos(angle):
    """Cosine helper (avoids math import)."""
    import math
    return math.cos(angle)


def sin(angle):
    """Sine helper (avoids math import)."""
    import math
    return math.sin(angle)


# ============ Comparison & Reporting ============

def compare_scenarios(uncompiled_results: dict, compiled_results: dict) -> dict:
    """
    Compare uncompiled vs compiled scenario results.

    Args:
        uncompiled_results: Results from uncompiled run
        compiled_results: Results from compiled run

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'uncompiled': extract_summary(uncompiled_results),
        'compiled': extract_summary(compiled_results),
        'improvements': {}
    }

    # Calculate improvements
    u = comparison['uncompiled']
    c = comparison['compiled']

    # Observations per minute
    comparison['improvements']['obs_rate_delta'] = (
        c['obs_per_min'] - u['obs_per_min']
    )

    # Reflections per minute
    comparison['improvements']['reflect_rate_delta'] = (
        c['reflect_per_min'] - u['reflect_per_min']
    )

    # Latency comparison
    if u.get('avg_latency_ms') and c.get('avg_latency_ms'):
        comparison['improvements']['latency_delta_ms'] = (
            c['avg_latency_ms'] - u['avg_latency_ms']
        )

    return comparison


def extract_summary(results: dict) -> dict:
    """Extract summary stats from scenario results."""
    duration = results['duration_minutes']

    summary = {
        'duration_minutes': duration,
        'tick_count': results['tick_count'],
        'observation_count': results['observation_count'],
        'reflection_count': results['reflection_count'],
        'obs_per_min': results['observation_count'] / duration if duration > 0 else 0,
        'reflect_per_min': results['reflection_count'] / duration if duration > 0 else 0,
    }

    # Extract average latency
    latency = results.get('latency', {})
    if latency:
        # Average across all signatures
        latencies = [s.get('mean_ms', 0) for s in latency.values()]
        summary['avg_latency_ms'] = sum(latencies) / len(latencies) if latencies else 0

    return summary


def format_comparison_report(comparison: dict) -> str:
    """
    Format comparison as human-readable report.

    Args:
        comparison: Output from compare_scenarios()

    Returns:
        Formatted string report
    """
    report = []
    report.append("\n" + "=" * 70)
    report.append("A/B TEST COMPARISON REPORT")
    report.append("=" * 70)

    u = comparison['uncompiled']
    c = comparison['compiled']
    i = comparison['improvements']

    report.append("\n📊 UNCOMPILED BASELINE:")
    report.append(f"  Observations/min:    {u['obs_per_min']:.2f}")
    report.append(f"  Reflections/min:     {u['reflect_per_min']:.2f}")
    report.append(f"  Avg latency:         {u.get('avg_latency_ms', 0):.0f} ms")

    report.append("\n📊 COMPILED (GEPA):")
    report.append(f"  Observations/min:    {c['obs_per_min']:.2f}")
    report.append(f"  Reflections/min:     {c['reflect_per_min']:.2f}")
    report.append(f"  Avg latency:         {c.get('avg_latency_ms', 0):.0f} ms")

    report.append("\n🎯 IMPROVEMENTS:")
    report.append(f"  Obs rate delta:      {i.get('obs_rate_delta', 0):+.2f}/min")
    report.append(f"  Reflect rate delta:  {i.get('reflect_rate_delta', 0):+.2f}/min")
    if i.get('latency_delta_ms'):
        report.append(f"  Latency delta:       {i['latency_delta_ms']:+.0f} ms")

    report.append("=" * 70)

    return "\n".join(report)


def save_results(results: dict, output_dir: str = "results"):
    """
    Save scenario results to JSON file.

    Args:
        results: Scenario results dictionary
        output_dir: Directory to save results
    """
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{results['scenario_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"💾 Results saved to {filepath}")


# ============ CLI Interface ============

async def main():
    """Run A/B test comparing compiled vs uncompiled."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B test compiled vs uncompiled agents")
    parser.add_argument(
        '--duration',
        type=int,
        default=20,
        help='Scenario duration in minutes (default: 20)'
    )
    parser.add_argument(
        '--skip-uncompiled',
        action='store_true',
        help='Skip uncompiled baseline run'
    )
    parser.add_argument(
        '--skip-compiled',
        action='store_true',
        help='Skip compiled run'
    )

    args = parser.parse_args()

    # Run uncompiled baseline
    uncompiled_results = None
    if not args.skip_uncompiled:
        uncompiled_results = await run_scenario(
            use_compiled_modules=False,
            duration_minutes=args.duration,
            scenario_name="uncompiled_baseline"
        )
        if uncompiled_results:
            save_results(uncompiled_results)

    # Run compiled version
    compiled_results = None
    if not args.skip_compiled:
        compiled_results = await run_scenario(
            use_compiled_modules=True,
            duration_minutes=args.duration,
            scenario_name="compiled_gepa"
        )
        if compiled_results:
            save_results(compiled_results)

    # Compare results
    if uncompiled_results and compiled_results:
        comparison = compare_scenarios(uncompiled_results, compiled_results)
        print(format_comparison_report(comparison))

        # Save comparison
        comparison_path = Path(__file__).parent.parent / "results" / "comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"💾 Comparison saved to {comparison_path}")
    else:
        logger.warning("⚠️  Could not compare (missing one or both results)")


if __name__ == "__main__":
    asyncio.run(main())
