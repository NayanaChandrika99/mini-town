"""
Lightweight telemetry helpers for recording DSPy plan metrics during Phase 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanTelemetryPayload:
    agent_id: int
    plan_source: str
    preserved_event_times: list[str]
    missing_event_times: list[str]
    overlaps_detected: bool
    invalid_locations: list[str]
    reasoning: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def record_plan_validation(payload: PlanTelemetryPayload) -> None:
    """
    Emit a structured log line describing the outcome of plan validation.

    This makes it easy to feed dashboards without requiring a new telemetry system.
    """
    logger.info("plan_validation %s", json.dumps(payload.to_dict(), ensure_ascii=False))
