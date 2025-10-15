"""
Unit tests for runtime.executor helpers.
"""

from runtime.executor import (
    dispatch_next_action,
    select_active_step,
    select_upcoming_step,
    steps_from_structured_plan,
)
from backend.agents import Agent, ActionType
from datetime import datetime


def test_steps_from_structured_plan_parses_coordinates():
    structured_plan = {
        "steps": [
            {
                "start": "08:00 AM",
                "end": "08:45 AM",
                "location": "(120, 140)",
                "description": "Meet Sam for breakfast",
                "rationale": "Nurture friendships early in the day",
            },
            {
                "start": "09:00 AM",
                "end": "09:30 AM",
                "location": "(130, 160)",
                "description": "Check library drop box",
                "rationale": "Return borrowed books",
            },
        ],
        "summary": "Morning social time followed by library errand.",
    }

    parsed = steps_from_structured_plan(structured_plan)

    assert len(parsed) == 2
    assert parsed[0]["location"] == (120.0, 140.0)
    assert parsed[1]["start_time_str"] == "09:00 AM"
    assert parsed[1]["end_minutes"] == 9 * 60 + 30


def test_select_active_and_upcoming_steps():
    plan = {
        "steps": [
            {
                "start": "01:00 PM",
                "end": "01:30 PM",
                "location": "(200, 220)",
                "description": "Lunch at cafe",
                "rationale": "",
            },
            {
                "start": "02:00 PM",
                "end": "02:30 PM",
                "location": "(210, 230)",
                "description": "Meet Alex at park",
                "rationale": "",
            },
        ]
    }
    parsed = steps_from_structured_plan(plan)

    active, idx = select_active_step(parsed, 13 * 60 + 15)
    assert active is not None
    assert idx == 0

    upcoming = select_upcoming_step(parsed, 13 * 60 + 15, window_minutes=60)
    assert upcoming is not None
    assert upcoming["description"].startswith("Meet Alex")

    # Wrap-around midnight
    active_midnight, _ = select_active_step(parsed, (24 * 60) - 10)
    assert active_midnight is None

    upcoming_wrap = select_upcoming_step(parsed, (24 * 60) - 10, window_minutes=30)
    assert upcoming_wrap is not None


def test_dispatch_next_action_variants():
    agent = Agent(
        agent_id=42,
        name="TestAgent",
        x=50,
        y=50,
        goal="Test",
        personality="curious",
    )

    current_step = {
        "location": (100.0, 100.0),
        "description": "Meet with Sam",
    }

    # Move should set navigation intent
    dispatch_next_action(agent, "move", current_step, [], datetime.now(), ActionType)
    assert agent.action_type is ActionType.NAVIGATE

    # Talk should queue conversation observation
    observations = dispatch_next_action(agent, "talk", current_step, [], datetime.now(), ActionType)
    assert agent.action_type is ActionType.CONVERSE
    assert observations and "[CONVERSATION]" in observations[0]

    # Wait should zero velocity
    agent.vx = 1.0
    agent.vy = 1.0
    dispatch_next_action(agent, "wait", current_step, [], datetime.now(), ActionType)
    assert agent.action_type is ActionType.WAIT
    assert agent.vx == 0 and agent.vy == 0
