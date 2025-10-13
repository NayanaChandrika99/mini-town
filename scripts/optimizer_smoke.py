#!/usr/bin/env python3
"""
Quick smoke test for the DSPy optimizer service.

Usage:
    DSPY_OPT_TEST_URL=http://127.0.0.1:8001 \
    DSPY_OPT_TEST_TOKEN=dev-secret \
    python scripts/optimizer_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILED_DIR = PROJECT_ROOT / "compiled"


def abort(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    sys.exit(1)


def ensure_artifact(filename: str) -> None:
    if not (COMPILED_DIR / filename).exists():
        abort(f"Missing compiled artifact: {filename}. Run GEPA compilation first.")


def request_headers() -> Dict[str, str]:
    token = os.getenv("DSPY_OPT_TEST_TOKEN")
    return {"X-Optimizer-Token": token} if token else {}


def main() -> None:
    ensure_artifact("compiled_planner.json")
    ensure_artifact("compiled_scorer.json")

    base_url = os.getenv("DSPY_OPT_TEST_URL", "http://127.0.0.1:8001")
    headers = request_headers()

    # Health check
    resp = requests.get(f"{base_url}/healthz", headers=headers, timeout=10)
    if resp.status_code != 200:
        abort(f"Health check failed: {resp.status_code} {resp.text}")

    data = resp.json()
    print(f"[OK] healthz -> {json.dumps(data)}")

    # Planner
    planner_payload = {
        "agent_name": "Alice",
        "agent_goal": "Prepare the community garden for visitors",
        "agent_personality": "Helpful, collaborative",
        "current_time": "8:45 AM",
        "current_location": "(200, 150)",
        "recent_events": ["09:15 AM - Garden meetup at (260, 180)"],
        "relevant_memories": ["Carol requested fresh flowers for the plaza"],
    }
    resp = requests.post(
        f"{base_url}/planner/plan_day",
        headers={"Content-Type": "application/json", **headers},
        json=planner_payload,
        timeout=30,
    )
    if resp.status_code != 200:
        abort(f"Planner endpoint failed: {resp.status_code} {resp.text}")
    plan = resp.json()
    print(f"[OK] planner -> {json.dumps(plan, indent=2)}")

    # Scorer
    scorer_payload = {
        "agent_name": "Alice",
        "observation": "Bob invited me to review the plaza design at 10:30 AM",
        "agent_goal": "Plan the garden layout",
        "agent_personality": "Focused, kind",
    }
    resp = requests.post(
        f"{base_url}/scorer/score_importance",
        headers={"Content-Type": "application/json", **headers},
        json=scorer_payload,
        timeout=30,
    )
    if resp.status_code != 200:
        abort(f"Scorer endpoint failed: {resp.status_code} {resp.text}")
    score = resp.json()
    print(f"[OK] scorer -> {json.dumps(score, indent=2)}")

    print("[PASS] Optimizer smoke test succeeded.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        abort(f"HTTP request error: {exc}")
