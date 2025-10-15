"""
End-to-end TownAgent DSPy program that composes scoring, reflection, planning,
and action selection into a single module suitable for GEPA compilation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import dspy

from backend.dspy_modules import (
    ChooseNextAction,
    PlanDay,
    PlanOutputDict,
    PlanValidation,
    Reflect,
    ScoreImportance,
    coerce_plan_output,
    fallback_plan_from_text,
    format_plan_text,
    validate_plan_output,
)


@dataclass
class ScoredObservation:
    text: str
    score: int
    reasoning: Optional[str]


@dataclass
class TownAgentResponse:
    plan_structured: PlanOutputDict
    plan_text: str
    plan_validation: PlanValidation
    reflection: Optional[str]
    scored_observations: List[ScoredObservation]
    next_action: Optional[str]
    next_action_reasoning: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_structured": self.plan_structured,
            "plan_text": self.plan_text,
            "plan_validation": asdict(self.plan_validation),
            "reflection": self.reflection,
            "scored_observations": [asdict(obs) for obs in self.scored_observations],
            "next_action": self.next_action,
            "next_action_reasoning": self.next_action_reasoning,
        }

    # Mapping protocol for compatibility with dspy.Evaluate / GEPA adapter.
    def _mapping(self) -> Dict[str, Any]:
        return self.to_dict()

    def items(self):
        return self._mapping().items()

    def __contains__(self, key: str) -> bool:
        return key in self._mapping()

    def __iter__(self):
        return iter(self._mapping())

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping().get(key, default)


class TownAgentProgram(dspy.Module):
    """
    Compose Mini-Town cognitive primitives into a single DSPy module.

    This module is intentionally synchronous (no async helpers) so it can be
    compiled directly with GEPA.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scorer = dspy.ChainOfThought(ScoreImportance)
        self.reflector = dspy.ChainOfThought(Reflect)
        self.planner = dspy.Predict(PlanDay)
        self.action_selector = dspy.Predict(ChooseNextAction)

    def forward(
        self,
        agent_name: str,
        agent_goal: str,
        agent_personality: str,
        current_time: str,
        current_location: str,
        recent_observations: Optional[List[str]] = None,
        recent_events: Optional[List[str]] = None,
        relevant_memories: Optional[List[str]] = None,
        candidate_actions: Optional[List[str]] = None,
    ) -> TownAgentResponse:
        recent_observations = recent_observations or []
        recent_events = recent_events or []
        relevant_memories = relevant_memories or []
        candidate_actions = candidate_actions or []

        def _record_trace(module_obj: Any, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            trace_buffer = getattr(dspy.settings, "trace", None)
            if isinstance(trace_buffer, list):
                trace_buffer.append((module_obj, inputs, outputs))

        scored_observations: List[ScoredObservation] = []
        for obs in recent_observations:
            if not obs:
                continue
            result = self.scorer(
                observation=obs,
                agent_goal=agent_goal,
                agent_personality=agent_personality,
            )
            try:
                score_value = int(getattr(result, "score", 5))
            except (TypeError, ValueError):
                score_value = 5
            reasoning_text = getattr(result, "reasoning", None)
            reasoning = reasoning_text.strip() if isinstance(reasoning_text, str) else None
            scored_observations.append(ScoredObservation(text=obs, score=score_value, reasoning=reasoning))
            _record_trace(
                self.scorer,
                {
                    "observation": obs,
                    "goal": agent_goal,
                    "personality": agent_personality,
                },
                {
                    "score": score_value,
                    "reasoning": reasoning,
                },
            )

        reflection = None
        if relevant_memories:
            memories_str = "\n".join(f"- {mem}" for mem in relevant_memories)
            reflection_result = self.reflector(
                recent_memories=memories_str,
                agent_personality=agent_personality,
                agent_goal=agent_goal,
            )
            reflection_value = getattr(reflection_result, "insight", None)
            if isinstance(reflection_value, str):
                reflection = reflection_value.strip()
            _record_trace(
                self.reflector,
                {
                    "recent_memories": relevant_memories[:5],
                    "goal": agent_goal,
                },
                {
                    "insight": reflection,
                },
            )

        recent_events_str = "\n".join(f"- {event}" for event in recent_events) if recent_events else "No recent events"
        relevant_memories_str = (
            "\n".join(f"- {mem}" for mem in relevant_memories) if relevant_memories else "No relevant memories"
        )
        plan_prediction = self.planner(
            agent_goal=agent_goal,
            agent_personality=agent_personality,
            current_time=current_time,
            current_location=current_location,
            recent_events=recent_events_str,
            relevant_memories=relevant_memories_str,
        )

        raw_plan = getattr(plan_prediction, "plan", {})
        try:
            structured_plan = coerce_plan_output(raw_plan)
        except ValueError:
            structured_plan = fallback_plan_from_text(str(raw_plan))

        plan_validation = validate_plan_output(structured_plan, recent_events)
        plan_text = format_plan_text(structured_plan)

        first_step = None
        steps = structured_plan.get("steps", []) if isinstance(structured_plan, dict) else []
        if steps:
            first_step = steps[0]

        _record_trace(
            self.planner,
            {
                "current_time": current_time,
                "location": current_location,
                "events": recent_events[:3],
                "memories": relevant_memories[:3],
            },
            {
                "summary": structured_plan.get("summary") if isinstance(structured_plan, dict) else None,
                "first_step": first_step,
                "plan_text": plan_text,
                "validation": {
                    "missing": plan_validation.missing_event_times,
                    "invalid": plan_validation.invalid_locations,
                    "overlaps": plan_validation.overlaps_detected,
                },
            },
        )

        next_action = None
        next_action_reasoning = None
        if candidate_actions:
            selector_result = self.action_selector(
                agent_name=agent_name,
                agent_goal=agent_goal,
                agent_personality=agent_personality,
                step_summary=structured_plan.get("summary", "") or (plan_text.splitlines()[0] if plan_text else ""),
                recent_events="\n".join(recent_events) if recent_events else "No recent events",
                location=current_location,
                possible_actions=", ".join(candidate_actions),
            )
            next_action = getattr(selector_result, "chosen_action", None) or getattr(selector_result, "action", None)
            next_reason = getattr(selector_result, "reasoning", None)
            if isinstance(next_reason, str):
                next_action_reasoning = next_reason.strip()
            _record_trace(
                self.action_selector,
                {
                    "candidate_actions": candidate_actions,
                    "current_step_summary": structured_plan.get("summary") if isinstance(structured_plan, dict) else None,
                },
                {
                    "chosen_action": next_action,
                    "reasoning": next_action_reasoning,
                },
            )

        return TownAgentResponse(
            plan_structured=structured_plan,
            plan_text=plan_text,
            plan_validation=plan_validation,
            reflection=reflection,
            scored_observations=scored_observations,
            next_action=next_action,
            next_action_reasoning=next_action_reasoning,
        )
