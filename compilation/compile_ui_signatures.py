# SPDX-License-Identifier: MIT
"""
UI Signature compilation workflow.

Compiles PlanStepExplainer and ObservationSummarizer modules using BootstrapFewShot.
These signatures enhance the frontend UI with contextual explanations and summaries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import dspy
from dspy.teleprompt import BootstrapFewShot

# Ensure backend package is importable when running from /compilation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from backend.dspy_modules import (  # noqa: E402
    PlanStepExplainer,
    ObservationSummarizer,
    configure_dspy,
)


def load_step_explainer_seeds(path: Path) -> List[dspy.Example]:
    """Load seed data for PlanStepExplainer signature."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[dspy.Example] = []
    for seed in data["seeds"]:
        example = dspy.Example(
            agent_name=seed["agent_name"],
            agent_goal=seed["agent_goal"],
            agent_personality=seed["agent_personality"],
            step_summary=seed["step_summary"],
            location=seed["location"],
            recent_memories=seed["recent_memories"],
            gold_explanation=seed["gold_explanation"],
        ).with_inputs(
            "agent_name",
            "agent_goal",
            "agent_personality",
            "step_summary",
            "location",
            "recent_memories",
        )
        examples.append(example)

    return examples


def load_observation_summarizer_seeds(path: Path) -> List[dspy.Example]:
    """Load seed data for ObservationSummarizer signature."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[dspy.Example] = []
    for seed in data["seeds"]:
        example = dspy.Example(
            agent_name=seed["agent_name"],
            agent_goal=seed["agent_goal"],
            observations=seed["observations"],
            gold_summary=seed["gold_summary"],
        ).with_inputs(
            "agent_name",
            "agent_goal",
            "observations",
        )
        examples.append(example)

    return examples


def step_explainer_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Simple metric for step explanations.
    
    Rewards explanations that:
    - Mention the goal or personality
    - Are concise (1-2 sentences)
    - Provide meaningful context
    """
    explanation: str = prediction.explanation if hasattr(prediction, "explanation") else ""
    if not explanation:
        return 0.0

    score = 0.0
    
    # Check if explanation mentions goal keywords
    goal_words = example.agent_goal.lower().split()
    if any(word in explanation.lower() for word in goal_words if len(word) > 4):
        score += 0.4
    
    # Reward appropriate length (1-3 sentences, ~20-150 chars)
    sentence_count = explanation.count('.') + explanation.count('!') + explanation.count('?')
    if 1 <= sentence_count <= 3 and 20 <= len(explanation) <= 150:
        score += 0.4
    
    # Reward if it references the step
    step_words = example.step_summary.lower().split()
    if any(word in explanation.lower() for word in step_words if len(word) > 4):
        score += 0.2
    
    return min(1.0, score)


def observation_summarizer_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Simple metric for observation summaries.
    
    Rewards summaries that:
    - Use bullet points
    - Are concise
    - Capture key themes
    """
    summary: str = ""
    if hasattr(prediction, "summary_points"):
        summary = prediction.summary_points
    elif hasattr(prediction, "summary"):
        summary = prediction.summary
    else:
        summary = str(prediction)
    
    if not summary:
        return 0.0

    score = 0.0
    
    # Reward bullet point format
    if '•' in summary or '-' in summary or '*' in summary:
        score += 0.4
    
    # Reward appropriate length (2-4 bullet points, ~50-250 chars)
    bullet_count = summary.count('•') + summary.count('\n-') + summary.count('\n*')
    if 2 <= bullet_count <= 4 and 50 <= len(summary) <= 250:
        score += 0.4
    
    # Reward if summary references goal
    goal_words = example.agent_goal.lower().split()
    if any(word in summary.lower() for word in goal_words if len(word) > 4):
        score += 0.2
    
    return min(1.0, score)


def compile_step_explainer(args: argparse.Namespace) -> None:
    """Compile PlanStepExplainer signature."""
    seed_path = Path(args.step_explainer_seeds)
    compiled_dir = PROJECT_ROOT / "compiled"
    compiled_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("🎯 Compiling PlanStepExplainer")
    print("="*60)
    
    print(f"📚 Loading seeds from {seed_path}")
    trainset = load_step_explainer_seeds(seed_path)
    print(f"✅ Loaded {len(trainset)} step explainer seeds")

    print("📐 Creating uncompiled baseline...")
    uncompiled_explainer = dspy.Predict(PlanStepExplainer)

    optimizer = BootstrapFewShot(
        metric=step_explainer_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )

    print(f"🚀 Running BootstrapFewShot optimization (max {args.max_demos} demos)...")
    compiled_explainer = optimizer.compile(uncompiled_explainer, trainset=trainset)

    save_path = compiled_dir / "compiled_step_explainer.json"
    print(f"💾 Saving compiled explainer to {save_path}")
    compiled_explainer.save(str(save_path))

    # Dump prompt for auditing
    prompt_path = compiled_dir / "prompt_step_explainer.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(str(compiled_explainer.dump_state()))

    print("✅ PlanStepExplainer compilation complete!")
    print(f"   Compiled module: {save_path}")
    print(f"   Prompt snapshot: {prompt_path}")


def compile_observation_summarizer(args: argparse.Namespace) -> None:
    """Compile ObservationSummarizer signature."""
    seed_path = Path(args.observation_summarizer_seeds)
    compiled_dir = PROJECT_ROOT / "compiled"
    compiled_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("📊 Compiling ObservationSummarizer")
    print("="*60)
    
    print(f"📚 Loading seeds from {seed_path}")
    trainset = load_observation_summarizer_seeds(seed_path)
    print(f"✅ Loaded {len(trainset)} observation summarizer seeds")

    print("📐 Creating uncompiled baseline...")
    uncompiled_summarizer = dspy.ChainOfThought(ObservationSummarizer)

    optimizer = BootstrapFewShot(
        metric=observation_summarizer_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )

    print(f"🚀 Running BootstrapFewShot optimization (max {args.max_demos} demos)...")
    compiled_summarizer = optimizer.compile(uncompiled_summarizer, trainset=trainset)

    save_path = compiled_dir / "compiled_observation_summarizer.json"
    print(f"💾 Saving compiled summarizer to {save_path}")
    compiled_summarizer.save(str(save_path))

    # Dump prompt for auditing
    prompt_path = compiled_dir / "prompt_observation_summarizer.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(str(compiled_summarizer.dump_state()))

    print("✅ ObservationSummarizer compilation complete!")
    print(f"   Compiled module: {save_path}")
    print(f"   Prompt snapshot: {prompt_path}")


def main(args: argparse.Namespace) -> None:
    print("⚙️ Configuring DSPy...")
    # Use specified provider/model if provided, otherwise use config.yml
    if args.provider:
        configure_dspy(provider=args.provider, model=args.model)
    else:
        configure_dspy()  # Uses config.yml / env vars

    if args.signature == "all" or args.signature == "step_explainer":
        compile_step_explainer(args)
    
    if args.signature == "all" or args.signature == "observation_summarizer":
        compile_observation_summarizer(args)
    
    print("\n" + "="*60)
    print("🎉 All compilations complete!")
    print("="*60)
    print("\nTo use these compiled modules:")
    print("1. Set 'use_compiled: true' in config.yml")
    print("2. Restart the backend server")
    print("3. Check the frontend UI for enhanced explanations and summaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile UI enhancement signatures (PlanStepExplainer, ObservationSummarizer) with DSPy BootstrapFewShot."
    )
    parser.add_argument(
        "--signature",
        choices=["all", "step_explainer", "observation_summarizer"],
        default="all",
        help="Which signature(s) to compile (default: all).",
    )
    parser.add_argument(
        "--step-explainer-seeds",
        default=str(PROJECT_ROOT / "seeds" / "step_explainer_v1.json"),
        help="Path to step explainer seeds JSON file.",
    )
    parser.add_argument(
        "--observation-summarizer-seeds",
        default=str(PROJECT_ROOT / "seeds" / "observation_summarizer_v1.json"),
        help="Path to observation summarizer seeds JSON file.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=6,
        help="Maximum number of demonstrations for BootstrapFewShot (default: 6).",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "together", "openai"],
        help="LLM provider (overrides config.yml)",
    )
    parser.add_argument(
        "--model",
        help="Model name (overrides config.yml)",
    )
    args = parser.parse_args()
    main(args)

