"""
Retrieval weight tuning via grid search.
Day 5: Find optimal α (relevance), β (recency), γ (importance) weights.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory import MemoryStore
from utils import generate_embedding
from metrics import retrieval_precision_at_k, retrieval_recall_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ Benchmark Scenarios ============

BENCHMARK_SCENARIOS = {
    "emergency": {
        "description": "Agent needs immediate action info during crisis",
        "query": "What should I do right now about the fire?",
        "test_memories": [
            # Relevant but low importance (not yet scored as critical)
            {"content": "Fire safety procedures: exit building immediately, use stairs not elevator, go to assembly point", "importance": 0.45, "relevant": True},
            {"content": "Emergency contacts: fire department 911, building security 555-0100", "importance": 0.50, "relevant": True},
            {"content": "Fire extinguisher locations: near kitchen, main hallway, by back exit", "importance": 0.40, "relevant": True},
            # Irrelevant but HIGH importance (important for other reasons)
            {"content": "Major work deadline tomorrow, presentation to CEO is crucial for my career", "importance": 0.95, "relevant": False},
            {"content": "Doctor appointment for serious health concern scheduled for this afternoon", "importance": 0.90, "relevant": False},
            {"content": "Need to pick up kids from school at 3pm, can't be late again", "importance": 0.85, "relevant": False},
            # Irrelevant and low importance
            {"content": "Weather forecast shows sunny skies for the weekend", "importance": 0.15, "relevant": False},
            {"content": "Grocery shopping list: milk, bread, eggs, vegetables", "importance": 0.20, "relevant": False},
            {"content": "Birthday party invitation for next month", "importance": 0.35, "relevant": False},
            {"content": "Car needs oil change soon, check engine light is on", "importance": 0.50, "relevant": False},
        ],
        "optimal_weights": {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},
        "description_note": "High relevance (α) is critical, importance matters less"
    },
    "social_planning": {
        "description": "Agent planning to attend a social event",
        "query": "Who will be at the party tonight?",
        "test_memories": [
            # OLD irrelevant but high importance
            {"content": "Critical project deadline on Friday, must finish presentation slides", "importance": 0.95, "relevant": False},
            {"content": "Doctor called with test results last week, need to follow up", "importance": 0.90, "relevant": False},
            {"content": "Anniversary dinner with spouse planned long ago, very important", "importance": 0.85, "relevant": False},
            # OLD irrelevant and low importance
            {"content": "Weather from last week was pleasant", "importance": 0.20, "relevant": False},
            {"content": "Old grocery list from a week ago", "importance": 0.25, "relevant": False},
            {"content": "Remembered I need to call my parents sometime", "importance": 0.40, "relevant": False},
            {"content": "Car was making strange noise last week, got it fixed", "importance": 0.50, "relevant": False},
            # RECENT AND relevant (but LOW importance - just casual party info, not life-changing)
            {"content": "Carol texted me an hour ago saying she'll be there with her boyfriend", "importance": 0.30, "relevant": True},
            {"content": "Party invitation received this morning from Maria, starts at 7pm at her place", "importance": 0.35, "relevant": True},
            {"content": "Maria just mentioned Alice and Bob are definitely coming to the party tonight", "importance": 0.25, "relevant": True},
        ],
        "optimal_weights": {"alpha": 0.3, "beta": 0.6, "gamma": 0.1},
        "description_note": "Recent info (β) is most valuable, relevance helps filter"
    },
    "long_term_relationship": {
        "description": "Agent recalling history with another person",
        "query": "What's my relationship with Bob like?",
        "test_memories": [
            # Relevant AND important (defining relationship moments)
            {"content": "Bob helped me move apartments last year when I had no one else, stayed all day - very grateful for his friendship", "importance": 0.90, "relevant": True},
            {"content": "Bob and I go way back to college days, we were roommates freshman year and bonded over late night study sessions", "importance": 0.85, "relevant": True},
            {"content": "Bob stood up for me when I was wrongly accused at work two years ago, really showed his character", "importance": 0.92, "relevant": True},
            # Irrelevant but ALSO important (confounders)
            {"content": "My mother's battle with cancer last year was the hardest thing I've faced, still affects me daily", "importance": 0.95, "relevant": False},
            {"content": "Getting promoted to senior manager changed my career trajectory fundamentally", "importance": 0.88, "relevant": False},
            # Recent Bob memory but trivial (not important, just recent)
            {"content": "Saw Bob at grocery store yesterday afternoon, waved hello but he seemed busy", "importance": 0.20, "relevant": False},
            # Irrelevant and low importance
            {"content": "Random encounter with a stranger on the bus this morning", "importance": 0.15, "relevant": False},
            {"content": "Need to remember to water the plants today", "importance": 0.18, "relevant": False},
            {"content": "Traffic was terrible this morning, took an extra 20 minutes", "importance": 0.16, "relevant": False},
            {"content": "Dentist appointment scheduled for next Tuesday at 3pm", "importance": 0.45, "relevant": False},
        ],
        "optimal_weights": {"alpha": 0.3, "beta": 0.1, "gamma": 0.6},
        "description_note": "Important memories (γ) define relationship, but need relevance to filter out OTHER important memories"
    },
    "goal_pursuit": {
        "description": "Agent working towards their stated goal",
        "query": "How can I make progress on my research project?",
        "test_memories": [
            # Relevant (research-related) with varying importance
            {"content": "Research notes: need to interview 3 more people for qualitative data, focus on demographics 25-35", "importance": 0.70, "relevant": True},
            {"content": "Library closes at 8pm on weekdays and 5pm on weekends, plan research time accordingly", "importance": 0.40, "relevant": True},
            {"content": "Had very productive research session yesterday, analyzed 50 survey responses and found interesting patterns", "importance": 0.65, "relevant": True},
            {"content": "Found excellent academic paper on similar methodology last week, saved PDF for reference", "importance": 0.60, "relevant": True},
            # Irrelevant but HIGH importance (competing priorities)
            {"content": "Family emergency last night, mother in hospital, need to visit today", "importance": 0.95, "relevant": False},
            {"content": "Job interview scheduled for tomorrow for dream position, must prepare tonight", "importance": 0.90, "relevant": False},
            {"content": "Landlord threatening eviction if rent isn't paid by Friday, very urgent", "importance": 0.88, "relevant": False},
            # Irrelevant and low importance
            {"content": "Weather is nice today, might go for a walk later", "importance": 0.12, "relevant": False},
            {"content": "Favorite coffee shop has new seasonal drinks available", "importance": 0.15, "relevant": False},
            {"content": "Need to renew gym membership this month", "importance": 0.25, "relevant": False},
        ],
        "optimal_weights": {"alpha": 0.5, "beta": 0.2, "gamma": 0.3},
        "description_note": "Balanced: relevance + importance, some recency"
    }
}


# ============ Test Memory Store Setup ============

def create_test_memory_store(scenario_name: str, scenario_data: dict) -> Tuple[MemoryStore, int, str, List[int]]:
    """
    Create a temporary memory store with test data for a scenario.

    Args:
        scenario_name: Name of the benchmark scenario
        scenario_data: Scenario configuration

    Returns:
        Tuple of (memory_store, agent_id, query, relevant_memory_ids)
    """
    # Create temp DB
    test_db_path = Path(__file__).parent.parent / "data" / f"tune_{scenario_name}.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old DB if exists
    if test_db_path.exists():
        test_db_path.unlink()

    memory_store = MemoryStore(str(test_db_path))

    # Create test agent
    agent_id = 1
    memory_store.create_agent(
        agent_id=agent_id,
        name="TestAgent",
        x=0, y=0,
        goal="Test goal",
        personality="test personality"
    )

    # Insert test memories
    base_time = datetime.now()
    relevant_ids = []

    for i, mem in enumerate(scenario_data['test_memories']):
        # Memories spread over 10 days (oldest first, most recent last)
        # This creates meaningful recency differences for testing
        days_ago = (len(scenario_data['test_memories']) - i - 1)  # 9, 8, 7, ..., 1, 0
        timestamp = base_time - timedelta(days=days_ago)

        embedding = generate_embedding(mem['content'])

        mem_id = memory_store.store_memory(
            agent_id=agent_id,
            content=mem['content'],
            importance=mem['importance'],
            embedding=embedding,
            timestamp=timestamp
        )

        if mem.get('relevant', False):
            relevant_ids.append(mem_id)

    query = scenario_data['query']

    logger.info(f"Created test memory store for '{scenario_name}' with {len(scenario_data['test_memories'])} memories")

    return memory_store, agent_id, query, relevant_ids


# ============ Diagnostic Functions ============

def debug_retrieval(memory_store: MemoryStore, agent_id: int, query: str, weights: Dict[str, float], top_k: int = 5):
    """
    Debug function to show detailed retrieval scores.

    Args:
        memory_store: Memory store to query
        agent_id: Agent ID
        query: Query string
        weights: Dict with 'alpha', 'beta', 'gamma'
        top_k: Number of results to show
    """
    from utils import generate_embedding

    query_embedding = generate_embedding(query)

    retrieved = memory_store.retrieve_memories_by_vector(
        agent_id=agent_id,
        query_embedding=query_embedding,
        top_k=top_k,
        alpha=weights['alpha'],
        beta=weights['beta'],
        gamma=weights['gamma']
    )

    print(f"\nDEBUG: Query: '{query}'")
    print(f"Weights: α={weights['alpha']:.2f}, β={weights['beta']:.2f}, γ={weights['gamma']:.2f}\n")
    print(f"{'ID':<5} {'Score':<7} {'α*rel':<8} {'β*rec':<8} {'γ*imp':<8} {'Content':<60}")
    print("-" * 110)

    for mem in retrieved:
        alpha_contrib = weights['alpha'] * mem['relevance']
        beta_contrib = weights['beta'] * mem['recency']
        gamma_contrib = weights['gamma'] * mem['importance']

        content_preview = mem['content'][:60] + "..." if len(mem['content']) > 60 else mem['content']
        print(f"{mem['id']:<5} {mem['score']:<7.3f} {alpha_contrib:<8.3f} {beta_contrib:<8.3f} {gamma_contrib:<8.3f} {content_preview}")


# ============ Grid Search ============

def grid_search_weights(
    memory_store: MemoryStore,
    agent_id: int,
    query: str,
    relevant_ids: List[int],
    alpha_range: np.ndarray,
    beta_range: np.ndarray,
    top_k: int = 5
) -> List[Dict]:
    """
    Perform grid search over α, β, γ weight combinations.

    Args:
        memory_store: Memory store with test data
        agent_id: Agent ID to query
        query: Query string
        relevant_ids: List of relevant memory IDs
        alpha_range: Array of alpha values to try
        beta_range: Array of beta values to try
        top_k: Number of results to retrieve

    Returns:
        List of dictionaries with results for each combination
    """
    results = []

    query_embedding = generate_embedding(query)

    # Grid search over α and β (γ = 1 - α - β)
    for alpha in alpha_range:
        for beta in beta_range:
            gamma = 1.0 - alpha - beta

            # Skip invalid combinations
            if gamma < 0 or gamma > 1:
                continue

            # Retrieve with these weights
            try:
                retrieved = memory_store.retrieve_memories_by_vector(
                    agent_id=agent_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )

                retrieved_ids = [mem['id'] for mem in retrieved]

                # Calculate precision and recall
                precision = retrieval_precision_at_k(retrieved_ids, relevant_ids, k=top_k)
                recall = retrieval_recall_at_k(retrieved_ids, relevant_ids, k=top_k)

                # F1 score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

            except Exception as e:
                logger.error(f"Error with weights α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}: {e}")
                continue

    return results


def find_best_weights(results: List[Dict], metric: str = 'f1') -> Dict:
    """
    Find the best weight combination from grid search results.

    Args:
        results: List of grid search results
        metric: Metric to optimize ('f1', 'precision', or 'recall')

    Returns:
        Best weight configuration
    """
    if not results:
        return None

    best = max(results, key=lambda x: x[metric])
    return best


# ============ Benchmark All Scenarios ============

def benchmark_all_scenarios(
    alpha_range: np.ndarray = None,
    beta_range: np.ndarray = None,
    top_k: int = 5
) -> Dict[str, Dict]:
    """
    Run grid search on all benchmark scenarios.

    Args:
        alpha_range: Array of alpha values to try (default: 0.0 to 1.0 in steps of 0.1)
        beta_range: Array of beta values to try (default: 0.0 to 1.0 in steps of 0.1)
        top_k: Number of results to retrieve

    Returns:
        Dictionary mapping scenario_name -> best_weights
    """
    if alpha_range is None:
        alpha_range = np.arange(0.0, 1.1, 0.1)

    if beta_range is None:
        beta_range = np.arange(0.0, 1.1, 0.1)

    logger.info("=" * 70)
    logger.info("RETRIEVAL WEIGHT TUNING - GRID SEARCH")
    logger.info("=" * 70)
    logger.info(f"α range: {alpha_range[0]:.1f} to {alpha_range[-1]:.1f} (steps: {len(alpha_range)})")
    logger.info(f"β range: {beta_range[0]:.1f} to {beta_range[-1]:.1f} (steps: {len(beta_range)})")
    logger.info(f"top_k: {top_k}")
    logger.info("")

    scenario_results = {}

    for scenario_name, scenario_data in BENCHMARK_SCENARIOS.items():
        logger.info(f"\n📊 Testing scenario: {scenario_name}")
        logger.info(f"   {scenario_data['description']}")
        logger.info(f"   Query: '{scenario_data['query']}'")

        # Create test memory store
        memory_store, agent_id, query, relevant_ids = create_test_memory_store(
            scenario_name, scenario_data
        )

        # Run grid search
        logger.info(f"   Running grid search ({len(alpha_range) * len(beta_range)} combinations)...")
        results = grid_search_weights(
            memory_store, agent_id, query, relevant_ids,
            alpha_range, beta_range, top_k
        )

        # Find best weights
        best = find_best_weights(results, metric='f1')

        if best:
            logger.info(f"   ✅ Best weights: α={best['alpha']:.2f}, β={best['beta']:.2f}, γ={best['gamma']:.2f}")
            logger.info(f"      F1={best['f1']:.3f}, Precision={best['precision']:.3f}, Recall={best['recall']:.3f}")
        else:
            logger.warning(f"   ⚠️  No valid results for {scenario_name}")

        scenario_results[scenario_name] = {
            'best_weights': best,
            'all_results': results,
            'expected_weights': scenario_data['optimal_weights']
        }

        # Cleanup
        memory_store.close()

    return scenario_results


def format_tuning_report(scenario_results: Dict[str, Dict]) -> str:
    """
    Format tuning results as human-readable report.

    Args:
        scenario_results: Output from benchmark_all_scenarios()

    Returns:
        Formatted string report
    """
    report = []
    report.append("\n" + "=" * 70)
    report.append("RETRIEVAL WEIGHT TUNING RESULTS")
    report.append("=" * 70)

    for scenario_name, data in scenario_results.items():
        scenario_info = BENCHMARK_SCENARIOS[scenario_name]
        best = data['best_weights']
        expected = data['expected_weights']

        report.append(f"\n📌 {scenario_name.upper()}")
        report.append(f"   {scenario_info['description']}")
        report.append(f"   Query: '{scenario_info['query']}'")

        if best:
            report.append(f"\n   Best weights found:")
            report.append(f"     α (relevance):  {best['alpha']:.2f}")
            report.append(f"     β (recency):    {best['beta']:.2f}")
            report.append(f"     γ (importance): {best['gamma']:.2f}")
            report.append(f"   Performance:")
            report.append(f"     F1:        {best['f1']:.3f}")
            report.append(f"     Precision: {best['precision']:.3f}")
            report.append(f"     Recall:    {best['recall']:.3f}")

        report.append(f"\n   Expected weights (from CLAUDE.md):")
        report.append(f"     α={expected['alpha']:.2f}, β={expected['beta']:.2f}, γ={1 - expected['alpha'] - expected['beta']:.2f}")

    report.append("\n" + "=" * 70)
    report.append("RECOMMENDATIONS")
    report.append("=" * 70)

    # Calculate average best weights
    all_best = [data['best_weights'] for data in scenario_results.values() if data['best_weights']]

    if all_best:
        avg_alpha = np.mean([w['alpha'] for w in all_best])
        avg_beta = np.mean([w['beta'] for w in all_best])
        avg_gamma = np.mean([w['gamma'] for w in all_best])

        report.append(f"\n📍 Average best weights across all scenarios:")
        report.append(f"   α (relevance):  {avg_alpha:.2f}")
        report.append(f"   β (recency):    {avg_beta:.2f}")
        report.append(f"   γ (importance): {avg_gamma:.2f}")

        report.append(f"\n💡 Consider using scenario-specific weights for best performance:")
        report.append(f"   - Emergency: high α (relevance matters most)")
        report.append(f"   - Social planning: high β (recency matters most)")
        report.append(f"   - Long-term relationships: high γ (importance matters most)")
        report.append(f"   - Goal pursuit: balanced α, γ (relevance + importance)")

    report.append("=" * 70)

    return "\n".join(report)


def save_tuning_results(scenario_results: Dict[str, Dict], output_dir: str = "results"):
    """
    Save tuning results to JSON file.

    Args:
        scenario_results: Output from benchmark_all_scenarios()
        output_dir: Directory to save results
    """
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"retrieval_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename

    # Convert to JSON-serializable format
    json_results = {}
    for scenario_name, data in scenario_results.items():
        json_results[scenario_name] = {
            'best_weights': data['best_weights'],
            'expected_weights': data['expected_weights'],
            # Don't save all_results (too large)
        }

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"💾 Tuning results saved to {filepath}")


# ============ CLI Interface ============

def main():
    """Run retrieval weight tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Tune retrieval weights via grid search")
    parser.add_argument(
        '--granularity',
        type=float,
        default=0.1,
        help='Grid search step size (default: 0.1)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of memories to retrieve (default: 5)'
    )

    args = parser.parse_args()

    # Define search ranges
    alpha_range = np.arange(0.0, 1.0 + args.granularity, args.granularity)
    beta_range = np.arange(0.0, 1.0 + args.granularity, args.granularity)

    # Run benchmark
    results = benchmark_all_scenarios(alpha_range, beta_range, args.top_k)

    # Print report
    print(format_tuning_report(results))

    # Save results
    save_tuning_results(results)


if __name__ == "__main__":
    main()
