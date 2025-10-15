"""Plan presets curated for the Mini-Town demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PlanPreset:
    id: str
    label: str
    summary: str
    plan: str
    landmark_id: Optional[str] = None


class PlanPresetCatalog:
    def __init__(self) -> None:
        self._metadata: Dict[str, str] = {}
        self._by_agent: Dict[int, List[PlanPreset]] = {}

    @property
    def metadata(self) -> Dict[str, str]:
        return self._metadata

    def load_from_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Plan preset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self._metadata = payload.get("metadata", {})
        self._by_agent = {}
        raw_agents = payload.get("agents", {})
        for agent_key, presets in raw_agents.items():
            try:
                agent_id = int(agent_key)
            except (TypeError, ValueError):
                continue

            parsed: List[PlanPreset] = []
            for preset in presets:
                plan = preset.get("plan")
                preset_id = preset.get("id")
                label = preset.get("label")
                summary = preset.get("summary", "")
                if not plan or not preset_id or not label:
                    continue
                parsed.append(
                    PlanPreset(
                        id=preset_id,
                        label=label,
                        summary=summary,
                        plan=plan,
                        landmark_id=preset.get("landmark_id"),
                    )
                )
            if parsed:
                self._by_agent[agent_id] = parsed

    def presets_for_agent(self, agent_id: int) -> List[PlanPreset]:
        return self._by_agent.get(agent_id, [])

    def preset_for_agent(self, agent_id: int, preset_id: str) -> Optional[PlanPreset]:
        for preset in self._by_agent.get(agent_id, []):
            if preset.id == preset_id:
                return preset
        return None

    def serialize(self) -> Dict[str, List[Dict[str, Optional[str]]]]:
        return {
            str(agent_id): [
                {
                    "id": preset.id,
                    "label": preset.label,
                    "summary": preset.summary,
                    "landmark_id": preset.landmark_id,
                }
                for preset in presets
            ]
            for agent_id, presets in self._by_agent.items()
        }


plan_preset_catalog = PlanPresetCatalog()


def load_default_plan_presets() -> None:
    """Load plan presets from the compiled presets directory."""
    base_path = Path(__file__).parent.parent
    preset_path = base_path / "compiled" / "presets" / "plans.json"
    plan_preset_catalog.load_from_file(preset_path)

