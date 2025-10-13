"""Quick test of single scenario"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tune_retrieval import BENCHMARK_SCENARIOS, create_test_memory_store, grid_search_weights
import numpy as np

# Test just social_planning
scenario_name = "social_planning"
scenario_data = BENCHMARK_SCENARIOS[scenario_name]

print(f"Testing {scenario_name}...")
print(f"Query: '{scenario_data['query']}'")
print()

# Create test store
memory_store, agent_id, query, relevant_ids = create_test_memory_store(
    scenario_name, scenario_data
)

print(f"Relevant memory IDs: {relevant_ids}")
print()

# Run grid search
alpha_range = np.arange(0.0, 1.1, 0.1)
beta_range = np.arange(0.0, 1.1, 0.1)

results = grid_search_weights(
    memory_store, agent_id, query, relevant_ids,
    alpha_range, beta_range, top_k=5
)

# Find best
best = max(results, key=lambda x: x['f1'])

print(f"Best weights: α={best['alpha']:.2f}, β={best['beta']:.2f}, γ={best['gamma']:.2f}")
print(f"F1={best['f1']:.3f}, Precision={best['precision']:.3f}, Recall={best['recall']:.3f}")
print()
print(f"Expected: α=0.30, β=0.60, γ=0.10")

memory_store.close()
