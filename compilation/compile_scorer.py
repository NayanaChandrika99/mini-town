# This file was auto-generated from compilation/compile_scorer.ipynb
# Please ensure you have installed the necessary dependencies from requirements.txt

# Set API key
import os
from getpass import getpass
if not os.getenv("GROQ_API_KEY"):
    os.environ['GROQ_API_KEY'] = getpass('Enter your GROQ_API_KEY: ')

# Configure Together.ai
if not os.getenv("TOGETHER_API_KEY"):
    together_key = getpass("Enter your Together.ai API key: ")
    os.environ["TOGETHER_API_KEY"] = together_key
    print("✅ Together.ai API key set in environment")

# Configure Together.ai LM
import dspy

lm = dspy.LM(
    model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.3,
    max_tokens=512
)

reflection_lm = dspy.LM(
    model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.5,
    max_tokens=512
)

# Re-configure DSPy with Together.ai
dspy.settings.configure(lm=lm)

print("✅ DSPy configured with Together.ai")
print("   Model: Meta-Llama-3.1-8B-Instruct-Turbo")

class ScoreImportance(dspy.Signature):
    """Rate how important this observation is for the agent's goals.

    Score 1-10 where:
    - 1-2: Trivial, background noise (e.g., "grass is green")
    - 3-4: Mildly interesting but not actionable
    - 5-6: Relevant to goals, worth remembering
    - 7-8: Directly impacts current plans or goals
    - 9-10: Life-changing, urgent, critical to goals
    """

    observation: str = dspy.InputField(desc="What the agent observed")
    agent_goal: str = dspy.InputField(desc="Agent's current high-level goal")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits")

    reasoning: str = dspy.OutputField(desc="Brief explanation of score")
    score: int = dspy.OutputField(desc="Importance score (1-10)")

print("✅ ScoreImportance signature defined")

import json

# Load seeds
with open('seeds/scorer_v1.json', 'r') as f:
    seeds_data = json.load(f)

print(f"Loaded {len(seeds_data['seeds'])} seeds")
print(f"Categories: {seeds_data['categories']}")

# Convert to DSPy examples
trainset = []
for seed in seeds_data['seeds']:
    example = dspy.Example(
        observation=seed['observation'],
        agent_goal=seed['agent_goal'],
        agent_personality=seed['agent_personality'],
        score=seed['gold_score'],
        category=seed['category'],  # For analysis
        seed_id=seed['id']  # For tracking
    ).with_inputs("observation", "agent_goal", "agent_personality")
    trainset.append(example)

print(f"✅ Created {len(trainset)} training examples")

# Show sample
print("\nSample example:")
print(f"Observation: {trainset[0].observation}")
print(f"Goal: {trainset[0].agent_goal}")
print(f"Personality: {trainset[0].agent_personality}")
print(f"Gold score: {trainset[0].score}")

def importance_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Metric for ScoreImportance compilation (GEPA-compatible).
    """
    try:
        if hasattr(gold, 'score'):
            gold_score = int(gold.score)
        else:
            gold_score = int(gold)

        if hasattr(pred, 'score'):
            pred_score = int(pred.score)
        else:
            pred_score = int(pred)
    except (ValueError, AttributeError, TypeError):
        return 0.0

    pred_score = max(1, min(10, pred_score))
    error = abs(pred_score - gold_score)

    if error == 0:
        return 1.0
    elif error <= 1:
        return 0.8
    elif error <= 2:
        return 0.5
    elif error <= 3:
        return 0.2
    else:
        return 0.0

print("✅ GEPA-compatible importance metric defined")

# Uncompiled baseline (ChainOfThought)
uncompiled_scorer = dspy.ChainOfThought(ScoreImportance)

print("✅ Uncompiled baseline created")
print("Module type:", type(uncompiled_scorer).__name__)

def evaluate_module(module, testset, verbose=False):
    """Evaluate module on test set."""
    results = {
        'exact': 0,
        'within_1': 0,
        'within_2': 0,
        'errors': [],
        'predictions': []
    }

    for i, example in enumerate(testset):
        try:
            pred = module(
                observation=example.observation,
                agent_goal=example.agent_goal,
                agent_personality=example.agent_personality
            )
            pred_score = int(pred.score)
            pred_score = max(1, min(10, pred_score))  # Clamp
        except Exception as e:
            if verbose:
                print(f"Error on example {i}: {e}")
            pred_score = 5  # Default

        gold_score = int(example.score)
        error = abs(pred_score - gold_score)

        results['errors'].append(error)
        results['predictions'].append(pred_score)

        if error == 0:
            results['exact'] += 1
        if error <= 1:
            results['within_1'] += 1
        if error <= 2:
            results['within_2'] += 1

    n = len(testset)
    results['accuracy_exact'] = results['exact'] / n * 100
    results['accuracy_within_1'] = results['within_1'] / n * 100
    results['accuracy_within_2'] = results['within_2'] / n * 100
    results['mean_error'] = sum(results['errors']) / len(results['errors'])
    results['max_error'] = max(results['errors'])

    return results

print("Evaluating uncompiled baseline (this may take 2-3 minutes)...\n")
uncompiled_results = evaluate_module(uncompiled_scorer, trainset, verbose=True)

print("=" * 70)
print("UNCOMPILED BASELINE PERFORMANCE")
print("=" * 70)
print(f"Exact matches:      {uncompiled_results['exact']:2d}/40 ({uncompiled_results['accuracy_exact']:.1f}%)")
print(f"Within ±1:          {uncompiled_results['within_1']:2d}/40 ({uncompiled_results['accuracy_within_1']:.1f}%)")
print(f"Within ±2:          {uncompiled_results['within_2']:2d}/40 ({uncompiled_results['accuracy_within_2']:.1f}%)")
print(f"Mean Absolute Error: {uncompiled_results['mean_error']:.2f}")
print(f"Max Error:          {uncompiled_results['max_error']}")
print("=" * 70)

from dspy.teleprompt import GEPA
import time

# GEPA optimizer with correct configuration
optimizer = GEPA(
    metric=importance_metric,
    auto="medium",
    reflection_minibatch_size=5,
    track_stats=True,
    reflection_lm=reflection_lm
)

print("✅ GEPA optimizer initialized")
print("Budget level: medium")

start_time = time.time()
os.makedirs("compilation/checkpoints", exist_ok=True)
CHECKPOINT_DIR = "compilation/checkpoints"

print("=" * 70)
print("STARTING GEPA COMPILATION")
print("=" * 70)
print(f"Training set size: {len(trainset)}")
print(f"Budget: {optimizer.budget}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

print(" Compilation running... (this will take a while)")

compiled_scorer = optimizer.compile(
    student=uncompiled_scorer,
    trainset=trainset,
    valset=trainset, # Evaluate on training data
    patience=3, # Stop if no improvement after 3 iterations
    checkpoint_dir=CHECKPOINT_DIR,
    max_bootstrapped_demos=3,
    max_labeled_demos=3
)

end_time = time.time()
duration = end_time - start_time

print("\n", "=" * 70)
print("COMPILATION COMPLETE")
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
print("=" * 70)

print("Evaluating compiled scorer...")
compiled_results = evaluate_module(compiled_scorer, trainset)

print("\n", "=" * 70)
print("COMPILED SCORER PERFORMANCE")
print("=" * 70)
print(f"Exact matches:      {compiled_results['exact']:2d}/40 ({compiled_results['accuracy_exact']:.1f}%)")
print(f"Within ±1:          {compiled_results['within_1']:2d}/40 ({compiled_results['accuracy_within_1']:.1f}%)")
print(f"Within ±2:          {compiled_results['within_2']:2d}/40 ({compiled_results['accuracy_within_2']:.1f}%)")
print(f"Mean Absolute Error: {compiled_results['mean_error']:.2f}")
print(f"Max Error:          {compiled_results['max_error']}")
print("=" * 70)

import json

os.makedirs("compiled", exist_ok=True)
SAVE_PATH = "compiled/compiled_scorer.json"
compiled_scorer.save(SAVE_PATH)

print(f"✅ Compiled scorer saved to {SAVE_PATH}")

os.makedirs("results", exist_ok=True)
DIAGNOSTICS_PATH = "results/compiled_scorer_diagnostics.json"
diagnostics = {
    "uncompiled_results": uncompiled_results,
    "compiled_results": compiled_results,
    "compilation_duration_minutes": duration / 60,
    "compilation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
}

with open(DIAGNOSTICS_PATH, 'w') as f:
    json.dump(diagnostics, f, indent=2)

print(f"✅ Diagnostics saved to {DIAGNOSTICS_PATH}")
