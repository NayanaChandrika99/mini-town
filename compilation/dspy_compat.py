"""DSPy/GEPA compatibility helpers confined to the repository."""

from __future__ import annotations

import sys
import types
from typing import Any, Iterable, List, Tuple

import dspy

# ---------------------------------------------------------------------------
# Legacy adapter expectations
# ---------------------------------------------------------------------------

TraceEntry = Tuple[Any, dict[str, Any], Any]


class CompatTraceData(dict):
    __slots__ = ()

    def __init__(
        self,
        *,
        example: dspy.Example,
        prediction: Any,
        score: Any,
        trace: List[TraceEntry],
        error: str | None = None,
    ) -> None:
        super().__init__(
            example=example,
            prediction=prediction,
            score=score,
            trace=trace,
            error=error,
        )

    def __getattr__(self, item: str) -> Any:  # pragma: no cover
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        self[key] = value


class CompatFailedPrediction(Exception):
    def __init__(self, *, completion_text: str = "", error: Exception | None = None):
        super().__init__(completion_text)
        self.completion_text = completion_text
        self.error = error


def _freeze_trace(trace: List[Any]) -> List[TraceEntry]:
    frozen: List[TraceEntry] = []
    for entry in trace:
        if not isinstance(entry, tuple) or len(entry) != 3:
            continue
        predictor, inputs, outputs = entry
        if isinstance(inputs, dict):
            inputs_copy = inputs.copy()
        else:
            inputs_copy = inputs
        frozen.append((predictor, inputs_copy, outputs))
    return frozen


def compat_bootstrap_trace_data(
    *,
    program: dspy.Module,
    dataset: Iterable[dspy.Example],
    metric,
    num_threads: int | None = None,
    raise_on_error: bool = False,
    capture_failed_parses: bool = True,
    failure_score: Any | None = None,
    format_failure_score: Any | None = None,
) -> List[CompatTraceData]:
    del num_threads, capture_failed_parses, format_failure_score

    trajectories: List[CompatTraceData] = []
    for example in dataset:
        with dspy.settings.context(trace=[]):
            try:
                prediction = program(**example.inputs())
            except Exception as exc:  # pragma: no cover
                if raise_on_error:
                    raise
                trace = list(dspy.settings.trace or [])
                trajectories.append(
                    CompatTraceData(
                        example=example,
                        prediction=CompatFailedPrediction(completion_text=str(exc), error=exc),
                        score=failure_score,
                        trace=_freeze_trace(trace),
                        error=str(exc),
                    )
                )
                continue
            trace = list(dspy.settings.trace or [])

        try:
            score = metric(example, prediction)
            error_message = None
        except Exception as exc:  # pragma: no cover
            if raise_on_error:
                raise
            score = failure_score
            error_message = str(exc)

        trajectories.append(
            CompatTraceData(
                example=example,
                prediction=prediction,
                score=score,
                trace=_freeze_trace(trace),
                error=error_message,
            )
        )

    return trajectories


# ---------------------------------------------------------------------------
# Shim installers
# ---------------------------------------------------------------------------


def _install_history_shim() -> None:
    try:
        from dspy.adapters.types import History  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    module = types.ModuleType("dspy.adapters.types")

    class History(list):
        def __init__(self, messages=None):
            super().__init__(messages or [])
            self.messages = list(messages or [])

    module.History = History  # type: ignore[attr-defined]
    sys.modules["dspy.adapters.types"] = module


def _install_bootstrap_shim() -> None:
    try:
        import dspy.teleprompt.bootstrap_trace  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    module = types.ModuleType("dspy.teleprompt.bootstrap_trace")
    module.TraceData = CompatTraceData  # type: ignore[attr-defined]
    module.FailedPrediction = CompatFailedPrediction  # type: ignore[attr-defined]
    module.bootstrap_trace_data = compat_bootstrap_trace_data  # type: ignore[attr-defined]
    sys.modules["dspy.teleprompt.bootstrap_trace"] = module


def _patch_adapter_evaluate(adapter_cls, evaluation_batch_cls) -> None:
    if getattr(adapter_cls.evaluate, "_mini_town_patched", False):  # pragma: no cover
        return

    original = adapter_cls.evaluate

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        if capture_traces:
            return original(self, batch, candidate, capture_traces=True)

        program = self.build_program(candidate)
        evaluator = dspy.Evaluate(
            devset=batch,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(batch) * 100,
        )

        result = evaluator(program)
        if hasattr(result, "results"):
            records = result.results  # type: ignore[attr-defined]
            outputs = [row[1] for row in records]
            scores_raw = [row[2] for row in records]
        elif isinstance(result, tuple):
            if len(result) == 3:
                _, records, scores_raw = result
            elif len(result) == 2:
                _, records = result
                scores_raw = [row[2] for row in records]
            else:  # pragma: no cover
                raise TypeError(f"Unexpected Evaluate output: {type(result)}")
            outputs = [row[1] for row in records]
        else:  # pragma: no cover
            raise TypeError(f"Unsupported Evaluate output type: {type(result)}")

        scores = [raw.get("score") if isinstance(raw, dict) else raw for raw in scores_raw]
        return evaluation_batch_cls(outputs=outputs, scores=scores, trajectories=None)

    evaluate._mini_town_patched = True  # type: ignore[attr-defined]
    adapter_cls.evaluate = evaluate  # type: ignore[assignment]


def install():
    _install_history_shim()
    _install_bootstrap_shim()

    from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter, ScoreWithFeedback
    from gepa.core.adapter import EvaluationBatch

    _patch_adapter_evaluate(DspyAdapter, EvaluationBatch)
    history_cls = sys.modules["dspy.adapters.types"].History  # type: ignore[attr-defined]

    return DspyAdapter, ScoreWithFeedback, EvaluationBatch, history_cls


__all__ = ["install"]
