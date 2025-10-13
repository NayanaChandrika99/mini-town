"""
Event scenario testing: Party attendance.
Day 5: Test whether agents attend events they're invited to.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import yaml
import json
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding
from dspy_modules import configure_dspy, use_compiled, load_compiled_modules
from metrics import event_coherence_metric, format_metrics_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config.yml"
with open(config_path) as f:
    config = yaml.safe_load(f)


# ============ Event Scenario ============

class PartyScenario:
    """Scenario: Maria invites agents to a party at specific time/location."""

    def __init__(
        self,
        party_time: datetime,
        party_location: tuple,
        host_agent_id: int,
        invitee_ids: List[int],
        use_compiled: bool = False
    ):
        self.party_time = party_time
        self.party_location = party_location  # (x, y)
        self.host_agent_id = host_agent_id
        self.invitee_ids = invitee_ids
        self.use_compiled = use_compiled

        self.attendees = []  # List of {agent_id, arrival_time, distance_from_party}

        logger.info(f"Party scenario initialized:")
        logger.info(f"  Time: {party_time.strftime('%I:%M %p')}")
        logger.info(f"  Location: {party_location}")
        logger.info(f"  Host: Agent {host_agent_id}")
        logger.info(f"  Invitees: {invitee_ids}")
        logger.info(f"  Using compiled: {use_compiled}")

    def send_invitations(self, agents: List[Agent], memory_store: MemoryStore):
        """
        Send party invitations to invitees.

        This creates high-importance memories for invited agents.
        """
        logger.info(f"\n📧 Sending party invitations...")

        invitation_time = self.party_time - timedelta(minutes=10)  # 10 min before party

        for agent in agents:
            if agent.id in self.invitee_ids:
                # Create invitation memory
                invitation_text = (
                    f"Maria invited you to a party at {self.party_time.strftime('%I:%M %p')} "
                    f"at location ({self.party_location[0]:.0f}, {self.party_location[1]:.0f}). "
                    f"This is a social gathering - great opportunity to build relationships!"
                )

                embedding = generate_embedding(invitation_text)

                memory_store.store_memory(
                    agent_id=agent.id,
                    content=invitation_text,
                    importance=0.85,  # High importance
                    embedding=embedding,
                    timestamp=invitation_time
                )

                logger.info(f"  ✅ Invited: {agent.name} (Agent {agent.id})")

    def check_attendance(self, agents: List[Agent], current_time: datetime):
        """
        Check which agents are near the party location.

        Day 6 Fix: Only count as 'arrived' if currently at location during party time window,
        not just at spawn time.

        Args:
            agents: List of all agents
            current_time: Current simulation time
        """
        attendance_radius = 100  # pixels

        # Only check attendance during a window AROUND event time
        # From 5 minutes before to 10 minutes after party time
        time_diff_seconds = (current_time - self.party_time).total_seconds()
        time_diff_minutes = time_diff_seconds / 60

        # Only record attendance during the party window
        if -5 <= time_diff_minutes <= 10:
            for agent in agents:
                if agent.id in self.invitee_ids:
                    # Calculate distance to party location
                    dx = agent.x - self.party_location[0]
                    dy = agent.y - self.party_location[1]
                    distance = math.sqrt(dx**2 + dy**2)

                    # Check if agent is at party
                    if distance <= attendance_radius:
                        # Check if not already recorded during THIS time window
                        existing = [a for a in self.attendees if a['agent_id'] == agent.id]

                        if not existing:
                            # First time recording this agent
                            self.attendees.append({
                                'agent_id': agent.id,
                                'agent_name': agent.name,
                                'arrival_time': current_time,
                                'distance_from_party': distance
                            })
                            logger.info(
                                f"  🎉 {agent.name} arrived at party! "
                                f"(distance: {distance:.1f}px, time: {current_time.strftime('%I:%M:%S %p')})"
                            )
                        else:
                            # Agent was recorded before - only update if they left and came back
                            last_recorded = existing[-1]['arrival_time']
                            time_since_last = (current_time - last_recorded).total_seconds()

                            if time_since_last > 60:  # More than 1 minute gap
                                self.attendees.append({
                                    'agent_id': agent.id,
                                    'agent_name': agent.name,
                                    'arrival_time': current_time,
                                    'distance_from_party': distance
                                })
                                logger.info(
                                    f"  🎉 {agent.name} returned to party! "
                                    f"(distance: {distance:.1f}px, time: {current_time.strftime('%I:%M:%S %p')})"
                                )

    def get_results(self) -> dict:
        """
        Get scenario results for metric calculation.

        Returns:
            Dictionary with event data and attendees
        """
        return {
            'event': {
                'time': self.party_time,
                'location': self.party_location,
                'invitees': self.invitee_ids
            },
            'attendees': self.attendees,
            'use_compiled': self.use_compiled
        }


# ============ Scenario Runner ============

async def run_party_scenario(
    use_compiled_modules: bool,
    duration_minutes: int = 30,
    scenario_name: str = "party_test"
) -> dict:
    """
    Run party attendance scenario.

    Args:
        use_compiled_modules: True to use compiled scorer
        duration_minutes: How long to run (should include before + after party time)
        scenario_name: Name for logging

    Returns:
        Dictionary with scenario results
    """
    logger.info("=" * 70)
    logger.info(f"PARTY SCENARIO: {scenario_name}")
    logger.info(f"Mode: {'COMPILED' if use_compiled_modules else 'UNCOMPILED'}")
    logger.info("=" * 70)

    # Configure DSPy
    configure_dspy()

    if use_compiled_modules:
        success = load_compiled_modules(config['compilation']['compiled_dir'])
        if not success:
            logger.error("Failed to load compiled modules")
            return None
        use_compiled(True)
    else:
        use_compiled(False)

    # Initialize database
    test_db_path = Path(__file__).parent.parent / "data" / f"test_{scenario_name}.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    if test_db_path.exists():
        test_db_path.unlink()

    memory_store = MemoryStore(str(test_db_path))

    # Initialize agents
    agents = initialize_party_agents(memory_store)

    # Create party scenario
    # Party time: 15 minutes into simulation
    start_time = datetime.now()
    party_time = start_time + timedelta(minutes=15)

    # Party location: center of map
    party_location = (
        config['simulation']['map_width'] / 2,
        config['simulation']['map_height'] / 2
    )

    # Invitees: all agents except host (Agent 1 is Maria, the host)
    host_id = 1
    invitee_ids = [agent.id for agent in agents if agent.id != host_id]

    scenario = PartyScenario(
        party_time=party_time,
        party_location=party_location,
        host_agent_id=host_id,
        invitee_ids=invitee_ids,
        use_compiled=use_compiled_modules
    )

    # Send invitations at T-10 minutes (5 minutes into sim)
    invitation_time = start_time + timedelta(minutes=5)

    # Run simulation
    end_time = start_time + timedelta(minutes=duration_minutes)
    tick_count = 0
    tick_interval = config['simulation']['tick_interval']

    invitations_sent = False
    plans_generated = False  # Track if agents have generated plans after invitation

    logger.info(f"\n⏰ Simulation timeline:")
    logger.info(f"  Start: {start_time.strftime('%I:%M:%S %p')}")
    logger.info(f"  Invitations: {invitation_time.strftime('%I:%M:%S %p')} (T+5 min)")
    logger.info(f"  Party: {party_time.strftime('%I:%M:%S %p')} (T+15 min)")
    logger.info(f"  End: {end_time.strftime('%I:%M:%S %p')} (T+{duration_minutes} min)")
    logger.info("")

    while datetime.now() < end_time:
        tick_count += 1
        current_time = datetime.now()

        # Send invitations at the right time
        if not invitations_sent and current_time >= invitation_time:
            scenario.send_invitations(agents, memory_store)
            invitations_sent = True

        # Generate plans 1 minute after invitations (give agents time to receive)
        plan_generation_time = invitation_time + timedelta(minutes=1)
        if invitations_sent and not plans_generated and current_time >= plan_generation_time:
            logger.info(f"\n📋 Generating agent plans...")
            for agent in agents:
                try:
                    plan = await agent.update_plan(memory_store, current_time)
                    if plan:
                        # Store plan as a high-importance memory
                        embedding = generate_embedding(f"[PLAN] {plan}")
                        memory_store.store_memory(
                            agent_id=agent.id,
                            content=f"[PLAN] {plan}",
                            importance=0.7,
                            embedding=embedding,
                            timestamp=current_time
                        )
                        logger.info(f"  ✅ {agent.name} planned: {plan[:80]}...")
                except Exception as e:
                    logger.error(f"  ❌ {agent.name} planning failed: {e}")
            plans_generated = True

        # Update each agent
        for agent in agents:
            # Update agent
            state = agent.update(agents)
            memory_store.update_agent_position(agent.id, agent.x, agent.y)

            # Store observations
            if state['observations']:
                for obs in state['observations']:
                    try:
                        await agent.score_and_store_observation(obs, memory_store)
                    except Exception as e:
                        logger.error(f"Error scoring observation: {e}")

                # Maybe reflect
                try:
                    insight = await agent.maybe_reflect(memory_store)
                    if insight:
                        embedding = generate_embedding(insight)
                        memory_store.store_memory(
                            agent_id=agent.id,
                            content=f"[REFLECTION] {insight}",
                            importance=0.9,
                            embedding=embedding,
                            timestamp=current_time
                        )
                except Exception as e:
                    logger.error(f"Error reflecting: {e}")

        # Check party attendance
        scenario.check_attendance(agents, current_time)

        # Wait for next tick
        await asyncio.sleep(tick_interval)

        # Log progress
        if tick_count % 30 == 0:
            elapsed = (current_time - start_time).total_seconds() / 60
            logger.info(f"⏱️  Tick {tick_count} ({elapsed:.1f} min elapsed)")

    # Simulation complete
    actual_duration = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"\n✅ Scenario complete: {tick_count} ticks in {actual_duration:.1f} minutes")

    # Get results
    results = scenario.get_results()
    results['scenario_name'] = scenario_name
    results['start_time'] = start_time.isoformat()
    results['duration_minutes'] = actual_duration
    results['tick_count'] = tick_count

    # Add agent plans to results
    results['agent_plans'] = {
        agent.id: {
            'name': agent.name,
            'plan': agent.current_plan,
            'plan_last_updated': agent.plan_last_updated.isoformat() if agent.plan_last_updated else None
        }
        for agent in agents
    }

    # Calculate event coherence
    coherence = event_coherence_metric(results)
    results['event_coherence'] = coherence

    logger.info(f"\n📊 Event Coherence: {coherence:.2%}")
    logger.info(f"   {len(results['attendees'])}/{len(invitee_ids)} invitees attended")
    logger.info(f"\n📋 Agent Plans:")
    for agent in agents:
        logger.info(f"   {agent.name}: {agent.current_plan[:60]}...")

    # Cleanup
    memory_store.close()

    return results


def initialize_party_agents(memory_store: MemoryStore) -> List[Agent]:
    """
    Initialize agents for party scenario.

    Agent 1 (Maria) is the host.
    """
    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']
    perception_radius = config['simulation']['perception_radius']

    agent_configs = [
        {"name": "Maria", "personality": "social, optimistic, outgoing", "goal": "Build relationships in the neighborhood"},
        {"name": "Alice", "personality": "social, curious", "goal": "Meet new people"},
        {"name": "Bob", "personality": "analytical, introverted", "goal": "Complete research project"}
    ]

    agents = []
    import random
    random.seed(42)  # Reproducible positions

    # Spawn agents in town center cluster (within 100x100 area)
    center_x = map_width / 2
    center_y = map_height / 2
    cluster_size = 100

    for i, config_data in enumerate(agent_configs):
        # Random position within cluster
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

        memory_store.create_agent(
            agent_id=agent.id,
            name=agent.name,
            x=agent.x,
            y=agent.y,
            goal=agent.goal,
            personality=agent.personality
        )

        logger.info(f"  Initialized: {agent.name} at ({agent.x:.0f}, {agent.y:.0f})")

    return agents


# ============ Comparison ============

def compare_party_scenarios(uncompiled_results: dict, compiled_results: dict) -> dict:
    """
    Compare uncompiled vs compiled party scenarios.

    Args:
        uncompiled_results: Results from uncompiled run
        compiled_results: Results from compiled run

    Returns:
        Comparison dictionary
    """
    comparison = {
        'uncompiled': {
            'coherence': uncompiled_results['event_coherence'],
            'attendees': len(uncompiled_results['attendees']),
            'invitees': len(uncompiled_results['event']['invitees'])
        },
        'compiled': {
            'coherence': compiled_results['event_coherence'],
            'attendees': len(compiled_results['attendees']),
            'invitees': len(compiled_results['event']['invitees'])
        }
    }

    comparison['improvement'] = {
        'coherence_delta': compiled_results['event_coherence'] - uncompiled_results['event_coherence'],
        'attendee_delta': len(compiled_results['attendees']) - len(uncompiled_results['attendees'])
    }

    return comparison


def format_party_comparison(comparison: dict) -> str:
    """Format party scenario comparison."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("PARTY SCENARIO COMPARISON")
    report.append("=" * 70)

    u = comparison['uncompiled']
    c = comparison['compiled']
    i = comparison['improvement']

    report.append(f"\n📊 UNCOMPILED:")
    report.append(f"  Event coherence: {u['coherence']:.2%}")
    report.append(f"  Attendance: {u['attendees']}/{u['invitees']} agents")

    report.append(f"\n📊 COMPILED (GEPA):")
    report.append(f"  Event coherence: {c['coherence']:.2%}")
    report.append(f"  Attendance: {c['attendees']}/{c['invitees']} agents")

    report.append(f"\n🎯 IMPROVEMENT:")
    report.append(f"  Coherence delta: {i['coherence_delta']:+.2%}")
    report.append(f"  Attendee delta: {i['attendee_delta']:+d} agents")

    target_coherence = 0.60  # 60% target from CLAUDE.md
    if c['coherence'] >= target_coherence:
        report.append(f"\n✅ SUCCESS: Compiled agents meet target coherence (≥{target_coherence:.0%})")
    else:
        report.append(f"\n⚠️  BELOW TARGET: Compiled coherence {c['coherence']:.2%} < target {target_coherence:.0%}")

    report.append("=" * 70)

    return "\n".join(report)


# ============ CLI Interface ============

async def main():
    """Run party scenario testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Test party attendance scenario")
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Scenario duration in minutes (default: 30)'
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

    # Run uncompiled
    uncompiled_results = None
    if not args.skip_uncompiled:
        uncompiled_results = await run_party_scenario(
            use_compiled_modules=False,
            duration_minutes=args.duration,
            scenario_name="party_uncompiled"
        )

        if uncompiled_results:
            # Save results
            output_path = Path(__file__).parent.parent / "results" / "party_uncompiled.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(uncompiled_results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to {output_path}")

    # Run compiled
    compiled_results = None
    if not args.skip_compiled:
        compiled_results = await run_party_scenario(
            use_compiled_modules=True,
            duration_minutes=args.duration,
            scenario_name="party_compiled"
        )

        if compiled_results:
            # Save results
            output_path = Path(__file__).parent.parent / "results" / "party_compiled.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(compiled_results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to {output_path}")

    # Compare
    if uncompiled_results and compiled_results:
        comparison = compare_party_scenarios(uncompiled_results, compiled_results)
        print(format_party_comparison(comparison))

        # Save comparison
        output_path = Path(__file__).parent.parent / "results" / "party_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"💾 Comparison saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
