from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import dspy


@dataclass
class TownAgentExample:
    agent_name: str
    agent_goal: str
    agent_personality: str
    current_time: str
    current_location: str
    recent_observations: List[str]
    recent_events: List[str]
    relevant_memories: List[str]
    candidate_actions: List[str]
    gold_plan_text: str
    notes: str | None = None

    def to_inputs(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "agent_goal": self.agent_goal,
            "agent_personality": self.agent_personality,
            "current_time": self.current_time,
            "current_location": self.current_location,
            "recent_observations": self.recent_observations,
            "recent_events": self.recent_events,
            "relevant_memories": self.relevant_memories,
            "candidate_actions": self.candidate_actions,
        }


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_town_agent_dataset(path: str | Path) -> List[TownAgentExample]:
    dataset: List[TownAgentExample] = []
    for record in _load_jsonl(Path(path)):
        dataset.append(
            TownAgentExample(
                agent_name=record["agent_name"],
                agent_goal=record["agent_goal"],
                agent_personality=record["agent_personality"],
                current_time=record["current_time"],
                current_location=record["current_location"],
                recent_observations=record.get("recent_observations", []),
                recent_events=record.get("recent_events", []),
                relevant_memories=record.get("relevant_memories", []),
                candidate_actions=record.get("candidate_actions", []),
                gold_plan_text=record.get("gold_plan_text", ""),
                notes=record.get("notes"),
            )
        )
    return dataset


def load_town_agent_examples(path: str | Path) -> List[dspy.Example]:
    dataset = load_town_agent_dataset(path)
    examples: List[dspy.Example] = []
    for example in dataset:
        dsp_example = dspy.Example(
            **example.to_inputs(),
            gold_plan_text=example.gold_plan_text,
            notes=example.notes,
        ).with_inputs(
            "agent_name",
            "agent_goal",
            "agent_personality",
            "current_time",
            "current_location",
            "recent_observations",
            "recent_events",
            "relevant_memories",
            "candidate_actions",
        )
        examples.append(dsp_example)
    return examples
