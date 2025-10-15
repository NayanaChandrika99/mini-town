"""
Unit tests for the new structured plan helpers (Phase 1).
"""

from dspy_modules import PlanOutputDict, format_plan_text, validate_plan_output


def test_validate_plan_output_success():
    structured: PlanOutputDict = {
        "steps": [
            {
                "start": "08:00 AM",
                "end": "09:00 AM",
                "location": "(100, 120)",
                "description": "Breakfast with Alex",
                "rationale": "Keep social commitments on time",
            },
            {
                "start": "10:00 AM",
                "end": "10:30 AM",
                "location": "(110, 125)",
                "description": "Meet Sam at the library",
                "rationale": "Follow up on shared project",
            },
        ],
        "summary": "Morning focused on social catch-ups.",
    }
    events = ["Alex confirmed breakfast at 08:00 AM (100, 120)", "Sam invited me to the library at 10:00 AM"]

    validation = validate_plan_output(structured, events)

    assert validation.is_valid
    assert sorted(validation.preserved_event_times) == ["08:00 AM", "10:00 AM"]
    assert not validation.missing_event_times
    assert not validation.invalid_locations


def test_validate_plan_output_detects_overlap():
    structured: PlanOutputDict = {
        "steps": [
            {
                "start": "01:00 PM",
                "end": "02:00 PM",
                "location": "(90, 75)",
                "description": "Visit the market",
                "rationale": "Stock up on ingredients",
            },
            {
                "start": "01:30 PM",
                "end": "02:30 PM",
                "location": "(95, 80)",
                "description": "Meet Jamie at the park",
                "rationale": "Catch up on latest news",
            },
        ],
        "summary": None,
    }
    events = ["Jamie asked me to meet at 01:30 PM"]

    validation = validate_plan_output(structured, events)

    assert validation.overlaps_detected
    assert not validation.is_valid
    assert validation.missing_event_times == []


def test_format_plan_text_outputs_human_readable():
    structured: PlanOutputDict = {
        "steps": [
            {
                "start": "05:00 PM",
                "end": "05:30 PM",
                "location": "(150, 200)",
                "description": "Walk to the cafe",
                "rationale": "Arrive early for the meetup",
            }
        ],
        "summary": "Evening meetup at the cafe.",
    }

    text = format_plan_text(structured)

    assert "05:00 PM - 05:30 PM" in text
    assert "@ (150, 200)" in text
    assert "Summary:" in text
