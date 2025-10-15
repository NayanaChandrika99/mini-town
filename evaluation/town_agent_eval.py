from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List

from backend.dspy_modules import configure_dspy, get_town_agent_program
from evaluation.town_agent_dataset import load_town_agent_dataset
from metrics.town_metric import town_metric


def evaluate_dataset(dataset_path: Path, limit: int | None = None) -> dict:
    configure_dspy()
    program = get_town_agent_program()

    dataset = load_town_agent_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]

    scores: List[float] = []
    results = []

    for example in dataset:
        response = program(
            agent_name=example.agent_name,
            agent_goal=example.agent_goal,
            agent_personality=example.agent_personality,
            current_time=example.current_time,
            current_location=example.current_location,
            recent_observations=example.recent_observations,
            recent_events=example.recent_events,
            relevant_memories=example.relevant_memories,
            candidate_actions=example.candidate_actions,
        )

        prediction_payload = {
            "steps": response.plan_structured.get("steps", []),
            "summary": response.plan_structured.get("summary"),
        }
        example_payload = {
            "agent_goal": example.agent_goal,
            "agent_personality": example.agent_personality,
            "recent_events": example.recent_events,
            "relevant_memories": example.relevant_memories,
        }

        metric = town_metric(example_payload, prediction_payload)
        scores.append(metric.score)
        results.append(
            {
                "agent": example.agent_name,
                "score": metric.score,
                "feedback": metric.feedback,
                "preserved_event_times": response.plan_validation.preserved_event_times,
                "missing_event_times": response.plan_validation.missing_event_times,
                "overlaps_detected": response.plan_validation.overlaps_detected,
                "invalid_locations": response.plan_validation.invalid_locations,
            }
        )

    aggregates = {
        "dataset": str(dataset_path),
        "examples": len(dataset),
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "details": results,
    }
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TownAgentProgram on a JSONL dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/town_agent_dev.jsonl"),
        help="Path to town agent JSONL dataset.",
    )
    parser.add_argument("--limit", type=int, help="Optional cap on examples to evaluate.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results.")

    args = parser.parse_args()
    results = evaluate_dataset(args.dataset, args.limit)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"✅ Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
