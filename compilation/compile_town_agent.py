#!/usr/bin/env python3
"""GEPA compilation driver for the TownAgent DSPy program."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import dspy

from backend.dspy_modules import configure_dspy
from evaluation.town_agent_dataset import load_town_agent_examples
from metrics import town_agent_metric
from programs import TownAgentProgram

from compilation.dspy_compat import install as install_dspy_compat
from gepa.api import optimize as gepa_optimize
from gepa.logging.logger import StdOutLogger

DspyAdapter, ScoreWithFeedback, _, _ = install_dspy_compat()

DEFAULT_DATASET = Path("datasets/town_agent_dev.jsonl")
COMPILED_DIR = Path("compiled")


def split_examples(examples: List[dspy.Example], ratio: float) -> tuple[List[dspy.Example], List[dspy.Example]]:
    if not examples:
        raise ValueError("Dataset is empty; cannot compile TownAgent program.")

    cutoff = max(1, int(len(examples) * ratio))
    cutoff = min(cutoff, len(examples) - 1) if len(examples) > 1 else 1
    trainset = examples[:cutoff]
    devset = examples[cutoff:] or examples[:]
    return trainset, devset


def _example_to_inputs(example: dspy.Example) -> dict:
    if hasattr(example, "toDict"):
        return example.toDict()
    if hasattr(example, "_store"):
        return dict(example._store)
    return dict(example)


def _metric_wrapper(example: dspy.Example, prediction: object, **_: object) -> float:
    return float(town_agent_metric(_example_to_inputs(example), prediction))


def run_evaluation(module: dspy.Module, dataset: List[dspy.Example]) -> float:
    if not dataset:
        return 0.0

    scores: List[float] = []
    for example in dataset:
        inputs = _example_to_inputs(example.inputs())
        prediction = module(**inputs)
        score = _metric_wrapper(example.inputs(), prediction)
        scores.append(score)

    return sum(scores) / len(scores)


def save_artifacts(compiled_module: dspy.Module, baseline_score: float, compiled_score: float) -> None:
    COMPILED_DIR.mkdir(exist_ok=True)
    model_path = COMPILED_DIR / "compiled_town_agent.json"
    prompt_path = COMPILED_DIR / "prompt_town_agent.txt"
    results_path = COMPILED_DIR / "town_agent_results.json"

    compiled_module.save(str(model_path))
    prompt_path.write_text(str(compiled_module.dump_state()), encoding="utf-8")

    results_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_score": baseline_score,
        "compiled_score": compiled_score,
        "improvement": compiled_score - baseline_score,
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    print(f"✅ Saved compiled TownAgent to {model_path}")
    print(f"✅ Prompt snapshot written to {prompt_path}")
    print(f"📊 Results stored in {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile TownAgent with GEPA.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to TownAgent JSONL dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (0-1).")
    parser.add_argument("--budget", type=int, default=80, help="GEPA iteration budget.")
    parser.add_argument("--provider", choices=["groq", "together", "openai"], help="Override provider from config.yml")
    parser.add_argument("--model", help="Override model name from config.yml")
    args = parser.parse_args()

    configure_dspy(provider=args.provider, model=args.model)

    examples = load_town_agent_examples(args.dataset)
    trainset, devset = split_examples(examples, args.train_ratio)
    print(f"📚 Loaded {len(examples)} examples ({len(trainset)} train / {len(devset)} dev)")

    baseline = TownAgentProgram()
    baseline_score = run_evaluation(baseline, devset)
    print(f"🧪 Baseline mean score: {baseline_score:.3f}")

    student = TownAgentProgram()

    seed_candidate: dict[str, str] = {}
    for name, predictor in student.named_predictors():
        instructions = getattr(getattr(predictor, "signature", None), "instructions", None)
        if isinstance(instructions, str):
            seed_candidate[name] = instructions
    if not seed_candidate:
        raise RuntimeError("Failed to extract initial instructions for GEPA candidate.")

    def make_feedback_fn(component_name: str):
        def feedback_fn(
            predictor_output,
            predictor_inputs,
            module_inputs,
            module_outputs,
            captured_trace,
        ) -> ScoreWithFeedback:
            score = _metric_wrapper(module_inputs, module_outputs)

            inputs_dict = _example_to_inputs(module_inputs)
            outputs_payload: Dict[str, Any] = {}
            if hasattr(module_outputs, "to_dict"):
                try:
                    outputs_payload = module_outputs.to_dict()  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover
                    outputs_payload = {}

            plan_structured = outputs_payload.get("plan_structured") or {}
            plan_summary = plan_structured.get("summary") if isinstance(plan_structured, dict) else None
            plan_steps = plan_structured.get("steps") if isinstance(plan_structured, dict) else None
            first_step_desc = None
            if isinstance(plan_steps, list) and plan_steps:
                first = plan_steps[0]
                if isinstance(first, dict):
                    first_step_desc = f"{first.get('description')} ({first.get('start')}–{first.get('end')})"

            feedback_bits: List[str] = [
                f"[{component_name}] metric {score:.3f}",
                f"Goal: {inputs_dict.get('agent_goal')}",
                f"Personality: {inputs_dict.get('agent_personality')}",
            ]

            recent_events = inputs_dict.get("recent_events")
            if isinstance(recent_events, (list, tuple)) and recent_events:
                feedback_bits.append("Recent events: " + "; ".join(map(str, recent_events[:3])))
            elif recent_events:
                feedback_bits.append(f"Recent events: {recent_events}")

            relevant_memories = inputs_dict.get("relevant_memories")
            if isinstance(relevant_memories, (list, tuple)) and relevant_memories:
                feedback_bits.append("Relevant memories: " + "; ".join(map(str, relevant_memories[:3])))
            elif relevant_memories:
                feedback_bits.append(f"Relevant memories: {relevant_memories}")

            if plan_summary:
                feedback_bits.append(f"Plan summary: {plan_summary}")
            if first_step_desc:
                feedback_bits.append(f"First step: {first_step_desc}")

            validation = getattr(module_outputs, "plan_validation", None)
            if validation:
                missing = getattr(validation, "missing_event_times", []) or []
                invalid = getattr(validation, "invalid_locations", []) or []
                overlaps = bool(getattr(validation, "overlaps_detected", False))
                if missing:
                    feedback_bits.append("Missing invitation times: " + ", ".join(map(str, missing)))
                if invalid:
                    feedback_bits.append("Invalid locations: " + ", ".join(map(str, invalid)))
                feedback_bits.append(f"Overlaps detected: {overlaps}")

            if component_name == "scorer":
                obs_preview = predictor_inputs.get("observation")
                if obs_preview is None:
                    recent_obs = inputs_dict.get("recent_observations")
                    if isinstance(recent_obs, (list, tuple)) and recent_obs:
                        obs_preview = recent_obs[0]
                    elif recent_obs:
                        obs_preview = recent_obs
                feedback_bits.append(f"Scored observation: {obs_preview}")
                reasoning = getattr(predictor_output, "reasoning", None)
                if reasoning:
                    feedback_bits.append(f"Scorer reasoning: {reasoning}")

            if component_name == "reflector" and getattr(module_outputs, "reflection", None):
                feedback_bits.append(f"Reflection insight: {module_outputs.reflection}")

            plan_text_full = outputs_payload.get("plan_text")
            if component_name == "planner" and plan_text_full:
                plan_preview = " | ".join(str(plan_text_full).splitlines()[:3])
                feedback_bits.append("Current plan text preview: " + plan_preview)

            if component_name == "action_selector":
                possible_actions = predictor_inputs.get("possible_actions") or predictor_inputs.get("candidate_actions")
                if possible_actions:
                    feedback_bits.append(f"Proposed actions: {possible_actions}")
                chosen_action = outputs_payload.get("next_action")
                action_reason = outputs_payload.get("next_action_reasoning")
                feedback_bits.append(f"Chosen action: {chosen_action} (reason: {action_reason})")

            if captured_trace:
                trace_summary = []
                for entry in captured_trace[-3:]:
                    name = entry[0] if entry else "trace"
                    trace_summary.append(str(name))
                if trace_summary:
                    feedback_bits.append("Recent trace components: " + ", ".join(trace_summary))

            feedback = "\n".join(str(bit) for bit in feedback_bits if bit)
            return ScoreWithFeedback(score=score, feedback=feedback)

        return feedback_fn

    feedback_map = {name: make_feedback_fn(name) for name in seed_candidate}

    adapter = DspyAdapter(
        student_module=student,
        metric_fn=_metric_wrapper,
        feedback_map=feedback_map,
        failure_score=0.0,
        num_threads=None,
        add_format_failure_as_feedback=True,
    )

    lm = getattr(dspy.settings, "lm", None)
    if lm is None:
        raise RuntimeError("DSPy LLM not configured; call configure_dspy() first.")

    def reflection_lm(prompt: str) -> str:
        response = lm(prompt)
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return str(response.text)
        if hasattr(response, "completion"):
            return str(response.completion)
        if hasattr(response, "completions") and response.completions:
            return str(response.completions[0])
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if isinstance(choice, dict):
                return str(choice.get("text") or choice.get("message", {}).get("content", ""))
            text = getattr(choice, "text", None)
            if text is not None:
                return str(text)
            message = getattr(choice, "message", None)
            if message and isinstance(message, dict):
                return str(message.get("content", ""))
        return str(response)

    max_metric_calls = max(args.budget * max(1, len(trainset)), len(trainset))

    print("🚀 Running GEPA optimization via gepa.optimize ...")
    gepa_result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=devset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        logger=StdOutLogger(),
        track_best_outputs=False,
        display_progress_bar=True,
        reflection_minibatch_size=6,
        skip_perfect_score=True,
    )

    best_candidate = gepa_result.best_candidate
    total_calls = gepa_result.total_metric_calls if gepa_result.total_metric_calls is not None else max_metric_calls
    best_val = max(gepa_result.val_aggregate_scores) if gepa_result.val_aggregate_scores else 0.0
    print(f"✅ GEPA finished after {total_calls} metric calls. Best val score: {best_val:.3f}")

    compiled_module = student.deepcopy()
    for name, predictor in compiled_module.named_predictors():
        if name in best_candidate:
            predictor.signature = predictor.signature.with_instructions(best_candidate[name])

    compiled_score = run_evaluation(compiled_module, devset)
    print(f"🚀 Compiled mean score: {compiled_score:.3f}")

    save_artifacts(compiled_module, baseline_score, compiled_score)


if __name__ == "__main__":
    main()
