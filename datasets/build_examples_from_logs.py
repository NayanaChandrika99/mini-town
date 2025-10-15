#!/usr/bin/env python3
"""
Utility script for extracting DSPy training/eval examples from Mini-Town logs.

Phase 1 focuses on plan-generation data, but the script is written so it can
emit examples for scoring and reflection modules as well.

Example usage:
    python datasets/build_examples_from_logs.py \
        --log-file logs/agent_events.jsonl \
        --output datasets/dev_plan_examples.jsonl \
        --kind plan
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional

LOG_KINDS = Literal["plan", "score", "reflect"]
DEFAULT_LOG_PATH = Path("logs/agent_events.jsonl")
DEFAULT_OUTPUT = Path("datasets/dev_plan_examples.jsonl")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b", re.IGNORECASE)


@dataclass
class PlanExample:
    agent_name: str
    agent_goal: str
    agent_personality: str
    current_time: str
    current_location: str
    recent_events: List[str]
    relevant_memories: List[str]
    gold_plan: str
    rationale: Optional[str] = None


@dataclass
class ScoreExample:
    observation: str
    agent_goal: str
    agent_personality: str
    gold_score: int
    rationale: Optional[str] = None


@dataclass
class ReflectExample:
    recent_memories: List[str]
    agent_goal: str
    agent_personality: str
    gold_insight: str


def load_records(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a .jsonl log file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_plan_examples(records: Iterable[dict]) -> Iterator[PlanExample]:
    """
    Identify plan-related events and convert them into PlanDay training examples.

    Expected schema (best-effort):
        {
            "type": "plan_generated",
            "agent": {...},
            "context": {...},
            "plan_text": "...",
            "reasoning": "...",
        }
    """
    for record in records:
        if record.get("type") != "plan_generated":
            continue

        agent = record.get("agent", {})
        context = record.get("context", {})

        # Skip if we lack essential context
        required_keys = ["goal", "personality", "current_time", "current_location"]
        if not all(key in context for key in required_keys):
            continue

        plan_text = record.get("plan_text") or record.get("plan")
        if not plan_text:
            continue

        # Ensure plan text contains at least one timeblock to qualify as ground truth
        if not TIME_RE.findall(plan_text):
            continue

        yield PlanExample(
            agent_name=agent.get("name", "Unknown Agent"),
            agent_goal=context["goal"],
            agent_personality=context["personality"],
            current_time=context["current_time"],
            current_location=context["current_location"],
            recent_events=context.get("recent_events", []),
            relevant_memories=context.get("relevant_memories", []),
            gold_plan=plan_text,
            rationale=record.get("reasoning"),
        )


def extract_score_examples(records: Iterable[dict]) -> Iterator[ScoreExample]:
    for record in records:
        if record.get("type") != "observation_scored":
            continue

        prediction = record.get("prediction")
        if prediction is None or "score" not in prediction:
            continue

        yield ScoreExample(
            observation=record.get("observation", ""),
            agent_goal=record.get("agent_goal", ""),
            agent_personality=record.get("agent_personality", ""),
            gold_score=int(prediction["score"]),
            rationale=prediction.get("reasoning"),
        )


def extract_reflection_examples(records: Iterable[dict]) -> Iterator[ReflectExample]:
    for record in records:
        if record.get("type") != "reflection_generated":
            continue
        insight = record.get("insight")
        if not insight:
            continue
        yield ReflectExample(
            recent_memories=record.get("recent_memories", []),
            agent_goal=record.get("agent_goal", ""),
            agent_personality=record.get("agent_personality", ""),
            gold_insight=insight,
        )


EXTRACTORS = {
    "plan": extract_plan_examples,
    "score": extract_score_examples,
    "reflect": extract_reflection_examples,
}


def write_jsonl(path: Path, examples: Iterable[dataclass]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} examples to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DSPy examples from logs.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kind",
        choices=list(EXTRACTORS.keys()),
        default="plan",
        help="Which example type to extract.",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        raise SystemExit(f"Log file not found: {args.log_file}")

    records = list(load_records(args.log_file))
    extractor = EXTRACTORS[args.kind]
    examples = list(extractor(records))
    if not examples:
        print("No matching records found; nothing written.")
        return
    write_jsonl(args.output, examples)


if __name__ == "__main__":
    main()
