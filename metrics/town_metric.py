from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from backend.dspy_modules import fallback_plan_from_text

TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b", re.IGNORECASE)
COORD_PATTERN = re.compile(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)")


def _to_minutes(timestamp: str) -> Optional[int]:
    try:
        parsed = datetime.strptime(timestamp.strip(), "%I:%M %p")
        return parsed.hour * 60 + parsed.minute
    except ValueError:
        return None


def _extract_times(texts: Iterable[str]) -> List[str]:
    times: List[str] = []
    for text in texts:
        times.extend(match.group(0).strip() for match in TIME_PATTERN.finditer(text))
    return times


@dataclass(frozen=True)
class PlanMetricBreakdown:
    plan_fidelity: float
    memory_grounding: float
    constraint_satisfaction: float
    persona_alignment: float

    def weighted_score(self) -> float:
        return (
            self.plan_fidelity * 0.4
            + self.memory_grounding * 0.2
            + self.constraint_satisfaction * 0.3
            + self.persona_alignment * 0.1
        )


@dataclass(frozen=True)
class TownMetricResult:
    score: float
    breakdown: PlanMetricBreakdown
    feedback: str


def evaluate_plan_candidate(
    example: Mapping[str, object],
    prediction: Mapping[str, object],
) -> PlanMetricBreakdown:
    """Heuristic evaluation used both in unit tests and GEPA metric feedback."""
    expected_events = example.get("recent_events", []) or []
    if isinstance(expected_events, str):
        expected_events = [expected_events]

    plan_steps: List[Mapping[str, object]] = prediction.get("steps", []) or []
    memories = example.get("relevant_memories", []) or []
    if isinstance(memories, str):
        memories = [memories]

    persona = (example.get("agent_personality") or "").lower()
    goal = (example.get("agent_goal") or "").lower()

    # Plan fidelity: preserve invited times and chronological order
    invited_times = _extract_times(expected_events)
    step_times = [(_to_minutes(step.get("start")), _to_minutes(step.get("end"))) for step in plan_steps]
    fidelity_hits = sum(
        any(invited_time in (step.get("start", ""), step.get("end", "")) for step in plan_steps)
        for invited_time in invited_times
    )
    fidelity_score = 0.0
    if invited_times:
        fidelity_score = fidelity_hits / len(invited_times)
    if step_times and all(start is not None and end is not None and start < end for start, end in step_times):
        # small boost for chronological ordering
        chronological = all(step_times[i][1] <= step_times[i + 1][0] for i in range(len(step_times) - 1))
        if chronological:
            fidelity_score = min(1.0, fidelity_score + 0.2)

    # Memory grounding: count references to supplied memories
    memories_lower = [mem.lower() for mem in memories]
    grounding_hits = 0
    for step in plan_steps:
        desc = str(step.get("description", "")).lower()
        grounding_hits += sum(1 for mem in memories_lower if mem and mem.split()[0] in desc)
    grounding_score = min(1.0, grounding_hits / max(1, len(plan_steps)))

    # Constraint satisfaction: simple structural checks
    coordinate_hits = sum(1 for step in plan_steps if COORD_PATTERN.search(str(step.get("location", ""))))
    no_overlap = 1.0 if _no_overlap(step_times) else 0.0
    constraint_score = 0.5 * (coordinate_hits / max(1, len(plan_steps))) + 0.5 * no_overlap

    # Persona / goal fit: naive keyword match
    persona_keywords = [word for word in re.split(r"\W+", persona) if len(word) > 4]
    goal_keywords = [word for word in re.split(r"\W+", goal) if len(word) > 4]
    persona_score = 0.0
    if persona_keywords or goal_keywords:
        combined = persona_keywords + goal_keywords
        mentions = 0
        for step in plan_steps:
            desc = str(step.get("description", "")).lower()
            mentions += sum(1 for kw in combined if kw in desc)
        persona_score = min(1.0, mentions / max(1, len(plan_steps)))

    return PlanMetricBreakdown(
        plan_fidelity=round(fidelity_score, 3),
        memory_grounding=round(grounding_score, 3),
        constraint_satisfaction=round(constraint_score, 3),
        persona_alignment=round(persona_score, 3),
    )


def town_metric(example: Mapping[str, object], prediction: Mapping[str, object]) -> TownMetricResult:
    breakdown = evaluate_plan_candidate(example, prediction)
    score = round(breakdown.weighted_score(), 3)
    feedback = _build_feedback(example, prediction, breakdown)
    return TownMetricResult(score=score, breakdown=breakdown, feedback=feedback)


def town_agent_metric(example: Mapping[str, object], prediction: object, **_: object) -> float:
    """Metric wrapper suitable for dspy.Evaluate / GEPA with TownAgent responses."""

    if hasattr(prediction, "plan_structured"):
        structured = getattr(prediction, "plan_structured")
        summary = getattr(prediction, "plan_structured", {}).get("summary") if isinstance(structured, dict) else None
    elif isinstance(prediction, dict):
        structured = prediction.get("plan_structured") or prediction.get("plan") or prediction.get("steps")
        if isinstance(structured, dict):
            summary = structured.get("summary")
        else:
            summary = None
    else:
        structured = getattr(prediction, "plan", None)
        summary = getattr(prediction, "summary", None)

    if structured is None:
        plan_text = getattr(prediction, "plan_text", "") if hasattr(prediction, "plan_text") else prediction
        structured = fallback_plan_from_text(str(plan_text))
    elif isinstance(structured, list):
        structured = {"steps": structured, "summary": summary}

    prediction_payload = {
        "steps": structured.get("steps", []) if isinstance(structured, dict) else [],
        "summary": structured.get("summary") if isinstance(structured, dict) else summary,
    }

    example_payload = {
        "agent_goal": example.get("agent_goal"),
        "agent_personality": example.get("agent_personality"),
        "recent_events": example.get("recent_events", []),
        "relevant_memories": example.get("relevant_memories", []),
    }

    return town_metric(example_payload, prediction_payload).score


def _build_feedback(
    example: Mapping[str, object],
    prediction: Mapping[str, object],
    breakdown: PlanMetricBreakdown,
) -> str:
    # Provide short actionable feedback for GEPA
    messages: List[str] = []
    if breakdown.plan_fidelity < 0.8:
        missing_times = _missing_invited_times(example, prediction)
        if missing_times:
            messages.append(f"Missing invitation times: {', '.join(sorted(missing_times))}.")
    if breakdown.constraint_satisfaction < 0.8:
        messages.append("Plan has overlaps or missing coordinates.")
    if breakdown.memory_grounding < 0.6 and (example.get("relevant_memories") or []):
        messages.append("Tie steps more explicitly to retrieved memories.")
    if not messages:
        messages.append("Plan respects invitations and constraints. Good job!")
    return " ".join(messages)


def _missing_invited_times(
    example: Mapping[str, object],
    prediction: Mapping[str, object],
) -> List[str]:
    invited = set(_extract_times(example.get("recent_events", []) or []))
    present = set()
    for step in prediction.get("steps", []) or []:
        for field in ("start", "end"):
            value = step.get(field)
            if isinstance(value, str) and TIME_PATTERN.search(value):
                present.add(TIME_PATTERN.search(value).group(0))
    return sorted(invited - present)


def _no_overlap(step_times: List[Tuple[Optional[int], Optional[int]]]) -> bool:
    sanitized: List[Tuple[int, int]] = []
    for start, end in step_times:
        if start is None or end is None:
            return False
        if start >= end:
            return False
        sanitized.append((start, end))
    for idx in range(len(sanitized) - 1):
        _, current_end = sanitized[idx]
        next_start, _ = sanitized[idx + 1]
        if current_end > next_start:
            return False
    return True
