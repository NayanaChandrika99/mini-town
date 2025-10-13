from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict

import dspy

from backend.dspy_modules import PlanDay, ScoreImportance

logger = logging.getLogger(__name__)


class CompiledModuleCache:
    """
    Lazy loader for compiled DSPy modules stored as JSON artifacts.

    Each logical module maps to a factory that constructs the base DSPy
    program (e.g., ``dspy.Predict(PlanDay)``). The cached instance will
    have ``.load(...)`` invoked with the compiled JSON path.
    """

    FILE_MAP: Dict[str, str] = {
        "plan_day": "compiled_planner.json",
        "score_importance": "compiled_scorer.json",
    }

    FACTORIES: Dict[str, Callable[[], dspy.Module]] = {
        "plan_day": lambda: dspy.Predict(PlanDay),
        "score_importance": lambda: dspy.ChainOfThought(ScoreImportance),
    }

    def __init__(self, compiled_dir: Path) -> None:
        self._compiled_dir = compiled_dir
        self._cache: Dict[str, dspy.Module] = {}

    def available_modules(self) -> Dict[str, Path]:
        """Return files that exist on disk."""
        available: Dict[str, Path] = {}
        for name, filename in self.FILE_MAP.items():
            file_path = self._compiled_dir / filename
            if file_path.exists():
                available[name] = file_path
        return available

    def get(self, name: str) -> dspy.Module:
        """Fetch (and cache) the compiled module by logical name."""
        if name in self._cache:
            return self._cache[name]

        module = self._build_module(name)
        path = self._resolve_file(name)

        logger.info("Loading compiled module '%s' from %s", name, path)
        try:
            module.load(str(path))
        except Exception as exc:
            logger.exception("Failed to load compiled module '%s' from %s", name, path)
            raise RuntimeError(f"Failed to load compiled module '{name}': {exc}") from exc
        self._cache[name] = module
        return module

    def _build_module(self, name: str) -> dspy.Module:
        try:
            factory = self.FACTORIES[name]
        except KeyError as exc:
            raise KeyError(f"Unknown compiled module '{name}'") from exc
        return factory()

    def _resolve_file(self, name: str) -> Path:
        try:
            filename = self.FILE_MAP[name]
        except KeyError as exc:
            raise KeyError(f"Unknown compiled module '{name}'") from exc

        file_path = self._compiled_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Compiled artifact not found: {file_path}")

        return file_path
