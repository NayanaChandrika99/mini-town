"""
Utilities for executing structured plans produced by the TownAgent program.

The helpers here are intentionally framework-agnostic so they can be reused
both in the legacy Agent loop and in future executors.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from backend.dspy_modules import PlanOutputDict, fallback_plan_from_text

TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M):\s*(.*?)(?=\n|$)", re.IGNORECASE)
COORD_PATTERN = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")


def steps_from_structured_plan(plan: PlanOutputDict) -> List[Dict[str, Any]]:
    """Convert PlanOutputDict into the legacy parsed-plan format used by Agent."""
    steps: List[Dict[str, Any]] = []
    for step in plan.get("steps", []) or []:
        start_str = str(step.get("start", "")).strip()
        end_str = str(step.get("end", "")).strip()
        description = str(step.get("description", "")).strip()
        rationale = str(step.get("rationale", "")).strip() or None
        location_str = str(step.get("location", "")).strip()

        try:
            start_time = datetime.strptime(start_str, "%I:%M %p").time()
            end_time = datetime.strptime(end_str, "%I:%M %p").time()
        except ValueError:
            continue

        location = None
        if location_str:
            coords = re.findall(r"\d+(?:\.\d+)?", location_str)
            if len(coords) == 2:
                try:
                    location = (float(coords[0]), float(coords[1]))
                except ValueError:
                    location = None

        steps.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "start_time_str": start_str,
                "end_time_str": end_str,
                "start_minutes": start_time.hour * 60 + start_time.minute,
                "end_minutes": end_time.hour * 60 + end_time.minute,
                "location": location,
                "description": description,
                "rationale": rationale,
            }
        )
    return steps


def steps_from_plan_text(plan_text: str) -> List[Dict[str, Any]]:
    """Fallback parser for legacy text-only plans."""
    structured = fallback_plan_from_text(plan_text)
    return steps_from_structured_plan(structured)


def select_active_step(
    parsed_steps: List[Dict[str, Any]],
    simulation_minutes: Optional[int],
    offset_minutes: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    if not parsed_steps or simulation_minutes is None:
        return None, None

    sim_mod = (simulation_minutes + offset_minutes) % (24 * 60)
    for idx, step in enumerate(parsed_steps):
        start = step.get("start_minutes")
        end = step.get("end_minutes")
        if start is None or end is None:
            continue
        if start <= sim_mod <= end:
            return step, idx
    return None, None


def select_upcoming_step(
    parsed_steps: List[Dict[str, Any]],
    simulation_minutes: Optional[int],
    offset_minutes: int = 0,
    window_minutes: int = 15,
) -> Optional[Dict[str, Any]]:
    if not parsed_steps or simulation_minutes is None:
        return None

    sim_mod = (simulation_minutes + offset_minutes) % (24 * 60)
    future_mod = (sim_mod + window_minutes) % (24 * 60)
    wrap = sim_mod >= future_mod

    for step in parsed_steps:
        start = step.get("start_minutes")
        if start is None:
            continue
        if wrap:
            if start > sim_mod or start <= future_mod:
                return step
        else:
            if sim_mod < start <= future_mod:
                return step
    return None


def navigate_for_plan(
    agent: Any,
    current_step: Optional[Dict[str, Any]],
    upcoming_step: Optional[Dict[str, Any]],
    other_agents: List[Any],
    current_time: datetime,
    action_enum: Type[Any],
) -> List[str]:
    """
    Update the agent's movement/state to follow the plan.

    Returns extra observations (e.g., social interactions) triggered by loitering.
    """
    extra_observations: List[str] = []

    if current_step and current_step.get("location"):
        target_x, target_y = current_step["location"]
        distance = math.hypot(agent.x - target_x, agent.y - target_y)
        if distance > 12.0:
            agent._set_action(action_enum.NAVIGATE)
            agent.navigate_to(target_x, target_y)
            agent._loiter_target = None
        else:
            agent._set_action(action_enum.LOITER)
            extra_observations.extend(
                agent._loiter_and_socialize(target_x, target_y, other_agents, current_time)
            )
    elif upcoming_step and upcoming_step.get("location"):
        target_x, target_y = upcoming_step["location"]
        distance = math.hypot(agent.x - target_x, agent.y - target_y)
        if distance < 10.0:
            agent.vx = 0
            agent.vy = 0
            agent._loiter_target = None
            agent._set_action(action_enum.WAIT)
        else:
            agent._set_action(action_enum.NAVIGATE)
            agent.navigate_to(target_x, target_y)
    else:
        agent._loiter_target = None
        agent._set_action(action_enum.EXPLORE)
        agent._random_walk()

    return extra_observations


def dispatch_next_action(
    agent: Any,
    action: Optional[str],
    current_step: Optional[Dict[str, Any]],
    other_agents: List[Any],
    current_time: datetime,
    action_enum: Type[Any],
) -> List[str]:
    """Interpret the TownAgent next_action signal into concrete behavior."""

    if not action:
        return []

    normalized = action.strip().lower()
    extras: List[str] = []

    if normalized == "move":
        target = current_step.get("location") if current_step else None
        if target:
            agent._set_action(action_enum.NAVIGATE)
            agent.navigate_to(*target)
            agent._loiter_target = None
        else:
            agent._set_action(action_enum.EXPLORE)
            agent._random_walk()

    elif normalized == "talk":
        agent._set_action(action_enum.CONVERSE)
        summary = current_step.get("description") if current_step else "sharing updates"
        observation = f"[CONVERSATION] Discussed {summary}"
        agent.queue_observation(observation)
        extras.append(observation)
        agent._next_conversation_time = current_time  # reset cool-down

    elif normalized == "wait":
        agent._set_action(action_enum.WAIT)
        agent.vx = 0
        agent.vy = 0

    elif normalized == "observe":
        agent._set_action(action_enum.OBSERVE)
        agent.vx = 0
        agent.vy = 0
        extras.append("[OBSERVE] Taking in the surroundings")
        agent.queue_observation(extras[-1])

    else:
        agent._set_action(action_enum.EXPLORE)
        agent._random_walk()

    return extras
