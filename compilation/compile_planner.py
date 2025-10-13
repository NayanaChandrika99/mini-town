# SPDX-License-Identifier: MIT
"""
PlanDay compilation workflow.

This script mirrors the `compile_scorer.ipynb` notebook but targets the
PlanDay module. Run it in an environment with valid LLM credentials
(e.g., Colab with GROQ/Together API keys configured).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

import dspy
from dspy.optimizers import GEPA

# Ensure backend package is importable when running from /compilation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from backend.dspy_modules import (  # noqa: E402  (import after path tweak)
    PlanDay,
    configure_dspy,
)


def load_seeds(path: Path) -> List[dspy.Example]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[dspy.Example] = []
    for seed in data["seeds"]:
        example = dspy.Example(
            agent_goal=seed["agent_goal"],
            agent_personality=seed["agent_personality"],
            current_time=seed["current_time"],
            current_location=seed["current_location"],
            recent_events=seed["recent_events"],
            relevant_memories=seed["relevant_memories"],
            gold_plan=seed["gold_plan"],
            rationale=seed.get("rationale", ""),
        ).with_inputs(
            "agent_goal",
            "agent_personality",
            "current_time",
            "current_location",
            "recent_events",
            "relevant_memories",
        )
        examples.append(example)

    return examples


TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}\s*[AP]M)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")


def extract_times(text: str) -> List[str]:
    return [match.strip() for match in TIME_PATTERN.findall(text)]


def extract_locations(text: str) -> List[str]:
    return [f"({m[0]}, {m[1]})" for m in LOCATION_PATTERN.findall(text)]


def planning_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Reward plans that preserve invited event times and locations.

    - Each invitation contributes two checks: time + location.
    - Memory-generated tasks (without explicit coordinates/times) are optional.
    """
    plan_text: str = prediction.plan if hasattr(prediction, "plan") else ""
    if not plan_text:
        return 0.0

    total_checks = 0
    satisfied = 0

    for event in example.recent_events:
        event_times = extract_times(event)
        event_locations = extract_locations(event)

        if event_times:
            total_checks += 1
            if any(evt_time in plan_text for evt_time in event_times):
                satisfied += 1

        if event_locations:
            total_checks += 1
            if any(loc in plan_text for loc in event_locations):
                satisfied += 1

    if total_checks == 0:
        # No explicit invitations. Reward plans that at least reference the goal.
        return 1.0 if example.agent_goal.split()[0].lower() in plan_text.lower() else 0.5

    return satisfied / total_checks


def main(args: argparse.Namespace) -> None:
    seed_path = Path(args.seeds)
    compiled_dir = PROJECT_ROOT / "compiled"
    compiled_dir.mkdir(exist_ok=True)

    print(f"📚 Loading seeds from {seed_path}")
    trainset = load_seeds(seed_path)
    print(f"✅ Loaded {len(trainset)} planning seeds")

    print("⚙️ Configuring DSPy...")
    # Use Together API if specified, otherwise use config.yml
    if args.provider:
        configure_dspy(provider=args.provider, model=args.model)
    else:
        configure_dspy()  # Uses config.yml / env vars

    print("📐 Creating uncompiled PlanDay baseline...")
    uncompiled_planner = dspy.Predict(PlanDay)

    optimizer = GEPA(
        metric=planning_metric,
        budget=args.budget,
        accelerator=args.accelerator,
        progress_bar=True,
    )

    print(f"🚀 Running GEPA optimization for {args.budget} iterations...")
    compiled_planner = optimizer.compile(uncompiled_planner, trainset=trainset)

    save_path = compiled_dir / "compiled_planner.json"
    print(f"💾 Saving compiled planner to {save_path}")
    compiled_planner.save(str(save_path))

    # Dump prompt for auditing
    prompt_path = compiled_dir / "prompt_planner.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(str(compiled_planner.dump_state()))

    print("✅ PlanDay compilation complete!")
    print(f"   Compiled module: {save_path}")
    print(f"   Prompt snapshot: {prompt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile PlanDay module with DSPy GEPA.")
    parser.add_argument(
        "--seeds",
        default=str(PROJECT_ROOT / "seeds" / "planner" / "planner_seeds_v1.json"),
        help="Path to planner seeds JSON file.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=40,
        help="Number of GEPA iterations (default: 40).",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="GEPA accelerator argument (default: auto).",
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
