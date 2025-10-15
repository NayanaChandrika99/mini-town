"""
Utilities for logging TownAgent training episodes to JSONL for dataset growth.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

DATASET_LOG_PATH = Path("datasets/town_agent_corpus.jsonl")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_list(value: Optional[Iterable[str]]) -> list[str]:
    if not value:
        return []
    return [str(item) for item in value if item]


def log_town_agent_episode(
    *,
    agent_id: int,
    agent_name: str,
    agent_goal: str,
    agent_personality: str,
    current_time: str,
    current_location: str,
    recent_observations: Iterable[str],
    recent_events: Iterable[str],
    relevant_memories: Iterable[str],
    candidate_actions: Iterable[str],
    plan_text: str,
    plan_structured: Dict[str, Any] | None,
    plan_source: str,
    use_town_agent: bool,
    notes: Optional[str] = None,
) -> None:
    """
    Append a TownAgent training example to the JSONL dataset log.
    """
    try:
        _ensure_parent(DATASET_LOG_PATH)
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_goal": agent_goal,
            "agent_personality": agent_personality,
            "current_time": current_time,
            "current_location": current_location,
            "recent_observations": _to_list(recent_observations),
            "recent_events": _to_list(recent_events),
            "relevant_memories": _to_list(relevant_memories),
            "candidate_actions": _to_list(candidate_actions),
            "plan_text": plan_text,
            "plan_structured": plan_structured,
            "plan_source": plan_source,
            "use_town_agent": use_town_agent,
            "notes": notes,
        }

        with DATASET_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log TownAgent episode: %s", exc)
