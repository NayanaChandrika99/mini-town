#!/usr/bin/env python3
"""GEPA compilation driver for the TownAgent DSPy program."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import dspy

from backend.dspy_modules import configure_dspy
from evaluation.town_agent_dataset import load_town_agent_examples
from metrics import town_agent_metric
from programs import TownAgentProgram

DEFAULT_DATASET = Path("datasets/town_agent_dev.jsonl")
COMPILED_DIR = Path("compiled")


def split_examples(examples: List[dspy.Example], ratio: float) -> tuple[List[dspy.Example], List[dspy.Example]]:
    if not examples:
        raise ValueError("Dataset is empty; cannot compile TownAgent program.")

    cutoff = max(1, int(len(examples) * ratio))
    cutoff = min(cutoff, len(examples) - 1) if len(examples) > 1 else 1
    trainset = examples[:cutoff]
    devset = examples[cutoff:] or examples[:]
    return trainset, devset


def run_evaluation(module: dspy.Module, dataset: List[dspy.Example]) -> float:
    evaluator = dspy.Evaluate(metric=town_agent_metric, verbose=False)
    scores = evaluator(module, dataset)
    if isinstance(scores, list):
        return sum(scores) / len(scores) if scores else 0.0
    return float(scores)


def save_artifacts(compiled_module: dspy.Module, baseline_score: float, compiled_score: float) -> None:
    COMPILED_DIR.mkdir(exist_ok=True)
    model_path = COMPILED_DIR / "compiled_town_agent.json"
    prompt_path = COMPILED_DIR / "prompt_town_agent.txt"
    results_path = COMPILED_DIR / "town_agent_results.json"

    compiled_module.save(str(model_path))
    prompt_path.write_text(str(compiled_module.dump_state()), encoding="utf-8")

    results_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_score": baseline_score,
        "compiled_score": compiled_score,
        "improvement": compiled_score - baseline_score,
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    print(f"✅ Saved compiled TownAgent to {model_path}")
    print(f"✅ Prompt snapshot written to {prompt_path}")
    print(f"📊 Results stored in {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile TownAgent with GEPA.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to TownAgent JSONL dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (0-1).")
    parser.add_argument("--budget", type=int, default=30, help="GEPA iteration budget.")
    parser.add_argument("--provider", choices=["groq", "together", "openai"], help="Override provider from config.yml")
    parser.add_argument("--model", help="Override model name from config.yml")
    args = parser.parse_args()

    configure_dspy(provider=args.provider, model=args.model)

    examples = load_town_agent_examples(args.dataset)
    trainset, devset = split_examples(examples, args.train_ratio)
    print(f"📚 Loaded {len(examples)} examples ({len(trainset)} train / {len(devset)} dev)")

    baseline = TownAgentProgram()
    baseline_score = run_evaluation(baseline, devset)
    print(f"🧪 Baseline mean score: {baseline_score:.3f}")

    optimizer = dspy.teleprompt.GEPA(
        metric=town_agent_metric,
        budget=args.budget,
        auto="medium",
        track_stats=True,
    )
    compiled_module = optimizer.compile(TownAgentProgram(), trainset=trainset)
    compiled_score = run_evaluation(compiled_module, devset)
    print(f"🚀 Compiled mean score: {compiled_score:.3f}")

    save_artifacts(compiled_module, baseline_score, compiled_score)


if __name__ == "__main__":
    main()
