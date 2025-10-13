"""
Evaluation metrics for Mini-Town.
Day 5: A/B testing framework for compiled vs uncompiled agents.
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import Levenshtein

logger = logging.getLogger(__name__)


# ============ Core Metrics ============

def event_coherence_metric(scenario_result: dict) -> float:
    """
    Calculate event coherence: did invited agents attend the event?

    Args:
        scenario_result: Dictionary containing:
            - event: {time: datetime, invitees: List[int]}
            - attendees: List[{agent_id: int, arrival_time: datetime}]

    Returns:
        Float between 0 and 1 (percentage of invitees who attended on time)

    Example:
        >>> result = {
        ...     'event': {'time': datetime(2025, 10, 11, 19, 0), 'invitees': [1, 2, 3]},
        ...     'attendees': [
        ...         {'agent_id': 1, 'arrival_time': datetime(2025, 10, 11, 19, 5)},
        ...         {'agent_id': 2, 'arrival_time': datetime(2025, 10, 11, 19, 2)}
        ...     ]
        ... }
        >>> event_coherence_metric(result)
        0.6666666666666666  # 2 out of 3 attended
    """
    event = scenario_result.get('event', {})
    event_time = event.get('time')
    invitees = event.get('invitees', [])
    attendees = scenario_result.get('attendees', [])

    if not invitees:
        logger.warning("No invitees specified for event coherence")
        return 0.0

    # Define attendance window (±10 minutes)
    window = timedelta(minutes=10)

    # Count how many invitees arrived on time
    on_time = []
    for attendee in attendees:
        agent_id = attendee.get('agent_id')
        arrival_time = attendee.get('arrival_time')

        if agent_id in invitees and arrival_time:
            time_diff = abs(arrival_time - event_time)
            if time_diff <= window:
                on_time.append(agent_id)

    coherence = len(on_time) / len(invitees)
    logger.info(f"Event coherence: {len(on_time)}/{len(invitees)} attended = {coherence:.2%}")

    return coherence


def plan_fidelity_metric(
    planned_timeline: List[str],
    executed_timeline: List[str]
) -> float:
    """
    Calculate plan fidelity: how well did agent follow their stated plan?

    Uses normalized Levenshtein (edit) distance between planned and executed actions.

    Args:
        planned_timeline: List of planned actions (e.g., ["wake up", "eat breakfast", "go to work"])
        executed_timeline: List of executed actions

    Returns:
        Float between 0 and 1 (1 = perfect adherence, 0 = completely different)

    Example:
        >>> planned = ["wake up", "eat breakfast", "go to work"]
        >>> executed = ["wake up", "check phone", "eat breakfast"]
        >>> plan_fidelity_metric(planned, executed)
        0.666...  # 2/3 similar
    """
    if not planned_timeline and not executed_timeline:
        return 1.0  # Both empty = perfect match

    if not planned_timeline or not executed_timeline:
        return 0.0  # One empty = no match

    # Calculate edit distance
    # Convert lists to strings for Levenshtein
    planned_str = " | ".join(planned_timeline)
    executed_str = " | ".join(executed_timeline)

    edit_distance = Levenshtein.distance(planned_str, executed_str)
    max_length = max(len(planned_str), len(executed_str))

    # Normalize: 1 - (distance / max_length)
    normalized_distance = edit_distance / max_length if max_length > 0 else 0
    fidelity = 1 - normalized_distance

    logger.debug(f"Plan fidelity: {fidelity:.2%} (edit distance: {edit_distance})")

    return max(0.0, min(1.0, fidelity))


def memory_hit_rate(
    test_queries: List[Tuple[str, List[int]]],
    retrieval_results: Dict[str, List[int]]
) -> float:
    """
    Calculate memory hit rate: can agent retrieve correct memories?

    Args:
        test_queries: List of (query, expected_memory_ids) tuples
        retrieval_results: Dictionary mapping query -> retrieved_memory_ids

    Returns:
        Float between 0 and 1 (percentage of queries where expected memory was found)

    Example:
        >>> queries = [
        ...     ("party invitation", [10, 15]),
        ...     ("Alice's phone number", [23])
        ... ]
        >>> results = {
        ...     "party invitation": [10, 12, 15, 18, 20],  # Top-5 retrieval
        ...     "Alice's phone number": [25, 23, 30, 19, 7]
        ... }
        >>> memory_hit_rate(queries, results)
        1.0  # Both queries found expected memory in top-5
    """
    if not test_queries:
        logger.warning("No test queries provided for memory hit rate")
        return 0.0

    hits = 0
    for query, expected_ids in test_queries:
        retrieved_ids = retrieval_results.get(query, [])

        # Check if ANY expected memory is in retrieved set
        if any(mem_id in retrieved_ids for mem_id in expected_ids):
            hits += 1

    hit_rate = hits / len(test_queries)
    logger.info(f"Memory hit rate: {hits}/{len(test_queries)} = {hit_rate:.2%}")

    return hit_rate


def town_score(
    scenario_results: dict,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    Calculate combined town score (weighted average of all metrics).

    This is the primary metric for DSPy compilation optimization.

    Args:
        scenario_results: Dictionary containing all metric inputs:
            - event_coherence: float (0-1)
            - plan_fidelity: float (0-1)
            - memory_hit_rate: float (0-1)
        weights: (w_event, w_plan, w_memory) - must sum to 1.0

    Returns:
        Float between 0 and 1 (combined score)

    Example:
        >>> results = {
        ...     'event_coherence': 0.75,
        ...     'plan_fidelity': 0.60,
        ...     'memory_hit_rate': 0.80
        ... }
        >>> town_score(results, weights=(0.4, 0.3, 0.3))
        0.72  # (0.4*0.75 + 0.3*0.60 + 0.3*0.80)
    """
    w_event, w_plan, w_memory = weights

    # Validate weights
    if abs(sum(weights) - 1.0) > 0.01:
        logger.warning(f"Weights don't sum to 1.0: {weights}")

    event_coherence = scenario_results.get('event_coherence', 0.0)
    plan_fidelity = scenario_results.get('plan_fidelity', 0.0)
    memory_hit_rate_val = scenario_results.get('memory_hit_rate', 0.0)

    score = (
        w_event * event_coherence +
        w_plan * plan_fidelity +
        w_memory * memory_hit_rate_val
    )

    logger.info(
        f"Town score: {score:.3f} "
        f"(event={event_coherence:.2f}, plan={plan_fidelity:.2f}, memory={memory_hit_rate_val:.2f})"
    )

    return score


# ============ Retrieval Quality Metrics ============

def retrieval_precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 5
) -> float:
    """
    Calculate precision@k for retrieval.

    Args:
        retrieved_ids: List of retrieved memory IDs (in order)
        relevant_ids: List of actually relevant memory IDs
        k: Consider only top-k retrieved items

    Returns:
        Float between 0 and 1 (percentage of top-k that are relevant)
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = len([mid for mid in top_k if mid in relevant_ids])

    precision = relevant_in_top_k / k
    return precision


def retrieval_recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 5
) -> float:
    """
    Calculate recall@k for retrieval.

    Args:
        retrieved_ids: List of retrieved memory IDs (in order)
        relevant_ids: List of actually relevant memory IDs
        k: Consider only top-k retrieved items

    Returns:
        Float between 0 and 1 (percentage of relevant items found in top-k)
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = len([mid for mid in top_k if mid in relevant_ids])

    recall = relevant_in_top_k / len(relevant_ids)
    return recall


# ============ Scenario Utilities ============

def calculate_scenario_metrics(
    event_data: dict,
    plan_data: Dict[int, dict],
    memory_data: Dict[int, dict]
) -> dict:
    """
    Calculate all metrics for a completed scenario.

    Args:
        event_data: Event coherence data
        plan_data: Plan fidelity data per agent
        memory_data: Memory hit rate data per agent

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # Event coherence
    if event_data:
        metrics['event_coherence'] = event_coherence_metric(event_data)
    else:
        metrics['event_coherence'] = None

    # Plan fidelity (average across agents)
    if plan_data:
        fidelities = []
        for agent_id, data in plan_data.items():
            fidelity = plan_fidelity_metric(
                data.get('planned', []),
                data.get('executed', [])
            )
            fidelities.append(fidelity)
        metrics['plan_fidelity'] = sum(fidelities) / len(fidelities) if fidelities else 0.0
    else:
        metrics['plan_fidelity'] = None

    # Memory hit rate (average across agents)
    if memory_data:
        hit_rates = []
        for agent_id, data in memory_data.items():
            hit_rate_val = memory_hit_rate(
                data.get('queries', []),
                data.get('results', {})
            )
            hit_rates.append(hit_rate_val)
        metrics['memory_hit_rate'] = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    else:
        metrics['memory_hit_rate'] = None

    # Town score (only if all components available)
    if all(metrics[k] is not None for k in ['event_coherence', 'plan_fidelity', 'memory_hit_rate']):
        metrics['town_score'] = town_score(metrics)
    else:
        metrics['town_score'] = None

    return metrics


def format_metrics_report(metrics: dict) -> str:
    """
    Format metrics as a human-readable report.

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 70)
    report.append("SCENARIO METRICS REPORT")
    report.append("=" * 70)

    if metrics.get('event_coherence') is not None:
        report.append(f"Event Coherence:     {metrics['event_coherence']:.2%}")

    if metrics.get('plan_fidelity') is not None:
        report.append(f"Plan Fidelity:       {metrics['plan_fidelity']:.2%}")

    if metrics.get('memory_hit_rate') is not None:
        report.append(f"Memory Hit Rate:     {metrics['memory_hit_rate']:.2%}")

    if metrics.get('town_score') is not None:
        report.append("-" * 70)
        report.append(f"TOWN SCORE:          {metrics['town_score']:.3f}")

    report.append("=" * 70)

    return "\n".join(report)
