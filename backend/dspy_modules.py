"""
DSPy modules for Mini-Town.
Day 2: Uncompiled ScoreImportance and Reflect modules.
"""

import json
import os
import logging
import re
from datetime import datetime
import dspy
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Global DSPy configuration
_configured = False


def configure_dspy(api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None):
    """
    Configure DSPy with LLM from config.yml.

    Args:
        api_key: API key (defaults to env var based on provider)
        provider: LLM provider (groq | together | openai), defaults to config.yml
        model: Model name, defaults to config.yml
    """
    global _configured

    if _configured:
        return

    # Load config if provider/model not specified
    if provider is None or model is None:
        config_path = Path(__file__).parent.parent / "config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        provider = provider or config['llm']['provider']
        model = model or config['llm']['model']
        temperature = config['llm']['temperature']
        max_tokens = config['llm']['max_tokens']
    else:
        temperature = 0.3
        max_tokens = 512

    # Get API key based on provider
    if api_key is None:
        if provider == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
        elif provider == 'together':
            api_key = os.getenv('TOGETHER_API_KEY')
        elif provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        else:
            raise ValueError(f"Unknown provider: {provider}")

    if not api_key:
        raise ValueError(f"API key not found for provider: {provider}")

    # Configure LLM based on provider
    if provider == 'groq':
        # Groq exposes an OpenAI-compatible endpoint.
        lm = dspy.OpenAI(
            model=model,
            api_key=api_key,
            api_provider='openai',
            api_base="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == 'together':
        lm = dspy.Together(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == 'openai':
        lm = dspy.OpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    dspy.settings.configure(lm=lm)
    _configured = True
    logger.info(f"DSPy configured with {provider} LLM ({model})")


# ============ Signatures ============


class PlanStepDict(TypedDict):
    start: str
    end: str
    location: str
    description: str
    rationale: Optional[str]


class PlanOutputDict(TypedDict):
    steps: List[PlanStepDict]
    summary: Optional[str]

class ScoreImportance(dspy.Signature):
    """Rate how important this observation is for the agent's goals.

    Score 1-10 where:
    - 1-2: Trivial, background noise (e.g., "grass is green")
    - 3-4: Mildly interesting but not actionable
    - 5-6: Relevant to goals, worth remembering
    - 7-8: Directly impacts current plans or goals
    - 9-10: Life-changing, urgent, critical to goals
    """

    observation: str = dspy.InputField(desc="What the agent observed")
    agent_goal: str = dspy.InputField(desc="Agent's current high-level goal")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits")

    reasoning: str = dspy.OutputField(desc="Brief explanation of score")
    score: int = dspy.OutputField(desc="Importance score (1-10)")


class Reflect(dspy.Signature):
    """Synthesize a high-level insight from recent memories.

    Generate an abstract realization or pattern that helps the agent
    understand their experiences and make better decisions.
    """

    recent_memories: str = dspy.InputField(desc="Recent important memories (newline-separated)")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits")
    agent_goal: str = dspy.InputField(desc="Agent's current goal")

    reasoning: str = dspy.OutputField(desc="Thought process")
    insight: str = dspy.OutputField(desc="High-level insight or realization")


class PlanDay(dspy.Signature):
    """Create a simple daily plan given goal and context.

    Generate a time-blocked plan that helps the agent achieve their goals
    while responding to recent events and invitations.

    CRITICAL RULES - MUST FOLLOW EXACTLY:
    1. When invited to an event, you MUST use the EXACT TIME from the invitation
       - If invitation says "party at 8:15 PM", plan MUST say "8:15 PM" (NOT 10:15 PM, NOT 9:15 PM)
       - NEVER reschedule invited events to later times
       - NEVER add buffer time before invited events
       - If current time is past the event time, plan to go immediately
    2. Include exact coordinates in format (x, y) for all locations
    3. For introverted agents: you can plan prep time, but PRESERVE the event time
       Example: "8:00 PM - 8:15 PM: Prep for party, 8:15 PM - 9:00 PM: Attend party at (x,y)"

    VIOLATION EXAMPLES (DO NOT DO THIS):
    ❌ Invitation: "party at 8:15 PM" → Plan: "10:15 PM: Attend party" (WRONG - rescheduled!)
    ❌ Invitation: "party at 8:15 PM" → Plan: "8:00 PM: Prep, 10:15 PM: Attend" (WRONG - changed time!)

    CORRECT EXAMPLES:
    ✅ Invitation: "party at 8:15 PM" → Plan: "8:15 PM - 9:00 PM: Attend party at (200, 150)"
    ✅ Invitation: "party at 8:15 PM" → Plan: "8:00 PM - 8:15 PM: Prep, 8:15 PM - 9:00 PM: Attend party at (200, 150)"
    """

    agent_goal: str = dspy.InputField(desc="Agent's high-level goal")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits (introverted agents still attend on time)")
    current_time: str = dspy.InputField(desc="Current time (e.g., '2:30 PM')")
    current_location: str = dspy.InputField(desc="Agent's current location coordinates")
    recent_events: str = dspy.InputField(desc="Recent invitations WITH EXACT TIMES that must be preserved (newline-separated)")
    relevant_memories: str = dspy.InputField(desc="Relevant memories from recent past")

    reasoning: str = dspy.OutputField(desc="Explain how you preserved exact event times from invitations")
    plan: str = dspy.OutputField(
        desc=(
            "Respond with a JSON object containing a `steps` array (start/end/location/description/rationale) that"
            " preserves every invitation time exactly. Include optional summary field."
        )
    )


# Additional demo-focused signatures


class ChooseNextAction(dspy.Signature):
    """Select the next immediate action from a shortlist based on plan context."""

    agent_name: str = dspy.InputField(desc="Agent name")
    agent_goal: str = dspy.InputField(desc="Agent's current goal")
    agent_personality: str = dspy.InputField(desc="Agent personality traits")
    step_summary: str = dspy.InputField(desc="Current plan step summary")
    recent_events: str = dspy.InputField(desc="Recent observations or events")
    location: str = dspy.InputField(desc="Current landmark or coordinates")
    possible_actions: str = dspy.InputField(desc="Comma-separated list of allowed actions")

    chosen_action: str = dspy.OutputField(desc="Chosen action (must be one of possible_actions)")
    reasoning: str = dspy.OutputField(desc="Short rationale linked to goal/personality")


class PlanConversation(dspy.Signature):
    """Generate a short conversation between two agents collaborating on a plan step."""

    speaker_name: str = dspy.InputField(desc="Primary agent name")
    speaker_goal: str = dspy.InputField(desc="Primary agent goal")
    speaker_personality: str = dspy.InputField(desc="Primary agent personality")
    partner_name: str = dspy.InputField(desc="Partner agent name")
    partner_goal: str = dspy.InputField(desc="Partner goal")
    partner_personality: str = dspy.InputField(desc="Partner personality traits")
    location: str = dspy.InputField(desc="Meeting location")
    step_summary: str = dspy.InputField(desc="Plan step objective")
    recent_memories: str = dspy.InputField(desc="Relevant recent memories/observations")

    dialogue: str = dspy.OutputField(desc="Multi-turn conversation transcript")


class PlanStepExplainer(dspy.Signature):
    """Explain why the current plan step matters."""

    agent_name: str = dspy.InputField(desc="Agent name")
    agent_goal: str = dspy.InputField(desc="Agent's overarching goal")
    agent_personality: str = dspy.InputField(desc="Personality cues")
    step_summary: str = dspy.InputField(desc="Current plan step summary")
    location: str = dspy.InputField(desc="Where the step occurs")
    recent_memories: str = dspy.InputField(desc="Recent context/observations")

    explanation: str = dspy.OutputField(desc="1-2 sentence motivational tooltip")


class ObservationSummarizer(dspy.Signature):
    """Summarize key themes from high-importance observations."""

    agent_name: str = dspy.InputField(desc="Agent name")
    agent_goal: str = dspy.InputField(desc="Agent goal")
    observations: str = dspy.InputField(desc="Bulleted/JSON list of observations with importance")

    summary_points: str = dspy.OutputField(desc="Bullet list (<=3 items) with key takeaways")


class ReflectionRefiner(dspy.Signature):
    """Refine raw reflection text into structured insights."""

    raw_reflection: str = dspy.InputField(desc="Raw reflection text")
    agent_personality: str = dspy.InputField(desc="Agent personality")
    agent_goal: str = dspy.InputField(desc="Agent goal")

    structured_reflection: str = dspy.OutputField(desc="JSON-like summary with objective + emotional tone")


# ============ Uncompiled Modules ============

# Simple predictor (uncompiled baseline)
scorer = dspy.ChainOfThought(ScoreImportance)

# Chain-of-thought for reflection (uncompiled baseline)
reflector = dspy.ChainOfThought(Reflect)

# Simple predictor for planning (uncompiled, Day 6)
planner = dspy.Predict(PlanDay)

# Additional demo-focused predictors (uncompiled baselines)
action_selector = dspy.Predict(ChooseNextAction)
conversation_generator = dspy.ChainOfThought(PlanConversation)
step_explainer_program = dspy.Predict(PlanStepExplainer)
observation_summarizer_program = dspy.ChainOfThought(ObservationSummarizer)
reflection_refiner_program = dspy.ChainOfThought(ReflectionRefiner)


# ============ Helper Functions ============

@dataclass
class ObservationScore:
    score: int
    reasoning: Optional[str] = None

    def clamp(self) -> "ObservationScore":
        self.score = max(1, min(10, self.score))
        if self.reasoning:
            self.reasoning = self.reasoning.strip()
        return self


@dataclass
class ActionDecision:
    action: str
    reasoning: Optional[str] = None


@dataclass
class ConversationResult:
    dialogue: str


@dataclass
class PlanStepExplanation:
    text: str


@dataclass
class ObservationSummary:
    summary: str


@dataclass
class StructuredReflection:
    text: str


@dataclass
class PlanValidation:
    preserved_event_times: List[str]
    missing_event_times: List[str]
    overlaps_detected: bool
    invalid_locations: List[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_event_times and not self.overlaps_detected and not self.invalid_locations


@dataclass
class PlanGeneration:
    text: str
    structured: PlanOutputDict
    reasoning: Optional[str]
    validation: PlanValidation


TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b", re.IGNORECASE)
COORD_RE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)")


def _parse_time_to_minutes(value: str) -> Optional[int]:
    try:
        parsed = datetime.strptime(value.strip(), "%I:%M %p")
        return parsed.hour * 60 + parsed.minute
    except (ValueError, TypeError):
        return None


def _extract_invited_times(events: List[str]) -> List[str]:
    invited: List[str] = []
    for event in events:
        invited.extend(match.group(0).strip() for match in TIME_RE.finditer(event))
    return invited


def _coerce_plan_output(raw_plan: Any) -> PlanOutputDict:
    if isinstance(raw_plan, dict):
        return raw_plan  # JSONAdapter already handled structure
    if isinstance(raw_plan, str):
        try:
            parsed = json.loads(raw_plan)
            if isinstance(parsed, dict):
                return parsed  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
    raise ValueError("Planner did not return valid JSON plan output")


def _fallback_plan_from_text(plan_text: str) -> PlanOutputDict:
    steps: List[PlanStepDict] = []
    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M):\s*(.*?)(?=\n|$)",
        re.IGNORECASE,
    )
    for match in time_pattern.finditer(plan_text):
        start, end, description = match.groups()
        location_match = COORD_RE.search(description)
        location = location_match.group(0) if location_match else ""
        steps.append(
            {
                "start": start.strip(),
                "end": end.strip(),
                "location": location.strip(),
                "description": description.strip(),
                "rationale": None,
            }
        )
    return {"steps": steps, "summary": None}


def format_plan_text(structured_plan: PlanOutputDict) -> str:
    steps = structured_plan.get("steps", []) or []
    summary = structured_plan.get("summary")
    lines: List[str] = []
    for step in steps:
        start = str(step.get("start", "")).strip()
        end = str(step.get("end", "")).strip()
        location = str(step.get("location", "")).strip()
        description = str(step.get("description", "")).strip()
        rationale = str(step.get("rationale", "")).strip()

        detail = f"{start} - {end}: {description}"
        if location:
            detail += f" @ {location}"
        if rationale:
            detail += f" ({rationale})"
        lines.append(detail)
    if summary:
        lines.append("")
        lines.append(f"Summary: {summary}")
    return "\n".join(lines).strip()


def validate_plan_output(plan: PlanOutputDict, recent_events: List[str]) -> PlanValidation:
    steps = plan.get("steps", []) or []
    invited_times = _extract_invited_times(recent_events)
    missing_times: List[str] = []
    preserved: List[str] = []
    seen_times = {time: False for time in invited_times}

    invalid_locations: List[str] = []
    timeline: List[Tuple[Optional[int], Optional[int]]] = []

    for step in steps:
        start = str(step.get("start", "")).strip()
        end = str(step.get("end", "")).strip()
        location = str(step.get("location", "")).strip()
        if location and not COORD_RE.fullmatch(location):
            invalid_locations.append(location)

        for invited in invited_times:
            if invited.lower() in (start.lower(), end.lower()):
                seen_times[invited] = True
                preserved.append(invited)

        timeline.append((_parse_time_to_minutes(start), _parse_time_to_minutes(end)))

    for time_str, matched in seen_times.items():
        if not matched:
            missing_times.append(time_str)

    overlaps = not _timeline_without_overlap(timeline)

    return PlanValidation(
        preserved_event_times=sorted(set(preserved)),
        missing_event_times=sorted(set(missing_times)),
        overlaps_detected=overlaps,
        invalid_locations=invalid_locations,
    )


def _timeline_without_overlap(timeline: List[Tuple[Optional[int], Optional[int]]]) -> bool:
    sanitized: List[Tuple[int, int]] = []
    for start, end in timeline:
        if start is None or end is None or start >= end:
            return False
        sanitized.append((start, end))
    for idx in range(len(sanitized) - 1):
        _, current_end = sanitized[idx]
        next_start, _ = sanitized[idx + 1]
        if current_end > next_start:
            return False
    return True


def coerce_plan_output(raw_plan: Any) -> PlanOutputDict:
    """Public helper to ensure plan outputs conform to PlanOutputDict."""
    return _coerce_plan_output(raw_plan)


def fallback_plan_from_text(plan_text: str) -> PlanOutputDict:
    """Best-effort conversion of legacy text plans to structured output."""
    return _fallback_plan_from_text(plan_text)


async def score_observation(
    observation: str,
    agent_goal: str,
    agent_personality: str
) -> ObservationScore:
    """
    Score importance of an observation using DSPy.

    Uses compiled scorer if available, otherwise uncompiled baseline.

    Args:
        observation: What the agent observed
        agent_goal: Agent's goal
        agent_personality: Agent's personality

    Returns:
        Importance score (1-10)

    Raises:
        Exception: If LLM call fails
    """
    # Get current scorer (compiled or uncompiled)
    current_scorer = get_current_scorer()

    result = current_scorer(
        observation=observation,
        agent_goal=agent_goal,
        agent_personality=agent_personality
    )

    # Parse score (handle various output formats)
    try:
        raw_score = int(result.score)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid score from LLM: {getattr(result, 'score', None)}, using default")
        raw_score = 5  # Default middle score

    reasoning = None
    for attr in ("reasoning", "analysis", "explanation", "thought_process", "rationale"):
        value = getattr(result, attr, None)
        if value:
            reasoning = str(value)
            break

    return ObservationScore(score=raw_score, reasoning=reasoning).clamp()


async def generate_reflection(
    recent_memories: list[str],
    agent_personality: str,
    agent_goal: str
) -> str:
    """
    Generate a reflection insight from recent memories.

    Args:
        recent_memories: List of recent memory strings
        agent_personality: Agent's personality
        agent_goal: Agent's goal

    Returns:
        Insight string

    Raises:
        Exception: If LLM call fails
    """
    # Format memories as newline-separated string
    memories_str = "\n".join([f"- {mem}" for mem in recent_memories])

    result = reflector(
        recent_memories=memories_str,
        agent_personality=agent_personality,
        agent_goal=agent_goal
    )

    return result.insight


async def generate_plan(
    agent_goal: str,
    agent_personality: str,
    current_time: str,
    current_location: str,
    recent_events: list[str],
    relevant_memories: list[str]
) -> PlanGeneration:
    """
    Generate a daily plan for the agent and return both structured JSON and formatted text.

    Args:
        agent_goal: Agent's high-level goal
        agent_personality: Agent's personality traits
        current_time: Current time string (e.g., "2:30 PM")
        current_location: Current location (e.g., "(100, 150)")
        recent_events: List of recent event strings (especially invitations)
        relevant_memories: List of relevant memory strings

    Returns:
        PlanGeneration dataclass containing formatted text, structured JSON, reasoning, and validation info.
    """
    # Format events and memories as newline-separated strings
    events_str = "\n".join([f"- {event}" for event in recent_events]) if recent_events else "No recent events"
    memories_str = "\n".join([f"- {mem}" for mem in relevant_memories]) if relevant_memories else "No relevant memories"

    current_planner = get_current_planner()

    result = current_planner(
        agent_goal=agent_goal,
        agent_personality=agent_personality,
        current_time=current_time,
        current_location=current_location,
        recent_events=events_str,
        relevant_memories=memories_str
    )

    try:
        structured_plan = _coerce_plan_output(result.plan)
    except ValueError as exc:
        if isinstance(result.plan, str):
            logger.warning("Planner returned text plan; falling back to regex parser: %s", exc)
            structured_plan = _fallback_plan_from_text(result.plan)
        else:
            logger.error("Planner returned invalid structure: %s", exc)
            raise

    validation = validate_plan_output(structured_plan, recent_events)
    if not validation.is_valid:
        logger.warning(
            "Plan validation issues detected (missing=%s, overlaps=%s, invalid_locations=%s)",
            validation.missing_event_times,
            validation.overlaps_detected,
            validation.invalid_locations,
        )

    plan_text = format_plan_text(structured_plan)
    reasoning = getattr(result, "reasoning", None)

    return PlanGeneration(
        text=plan_text,
        structured=structured_plan,
        reasoning=reasoning,
        validation=validation,
    )


def get_planner_source() -> str:
    """
    Return a human-readable label describing which planner is active.

    Returns:
        "compiled" if the compiled planner is enabled and available, otherwise "baseline".
    """
    if _use_compiled and _compiled_planner is not None:
        return "compiled"
    return "baseline"


def get_current_action_selector():
    if _use_compiled and _compiled_action_selector is not None:
        return _compiled_action_selector
    return action_selector


def get_current_conversation_generator():
    if _use_compiled and _compiled_conversation is not None:
        return _compiled_conversation
    return conversation_generator


def get_current_step_explainer():
    if _use_compiled and _compiled_step_explainer is not None:
        return _compiled_step_explainer
    return step_explainer_program


def get_current_observation_summarizer():
    if _use_compiled and _compiled_observation_summarizer is not None:
        return _compiled_observation_summarizer
    return observation_summarizer_program


def get_current_reflection_refiner():
    if _use_compiled and _compiled_reflection_refiner is not None:
        return _compiled_reflection_refiner
    return reflection_refiner_program


async def choose_next_action(
    agent_name: str,
    agent_goal: str,
    agent_personality: str,
    step_summary: str,
    recent_events: str,
    location: str,
    possible_actions: str,
) -> ActionDecision:
    selector = get_current_action_selector()
    result = selector(
        agent_name=agent_name,
        agent_goal=agent_goal,
        agent_personality=agent_personality,
        step_summary=step_summary,
        recent_events=recent_events,
        location=location,
        possible_actions=possible_actions,
    )
    action = getattr(result, "chosen_action", None) or getattr(result, "action", None) or "rest"
    reasoning = getattr(result, "reasoning", None)
    return ActionDecision(action=action, reasoning=reasoning)


async def generate_plan_conversation(
    speaker_name: str,
    speaker_goal: str,
    speaker_personality: str,
    partner_name: str,
    partner_goal: str,
    partner_personality: str,
    location: str,
    step_summary: str,
    recent_memories: str,
) -> ConversationResult:
    generator = get_current_conversation_generator()
    result = generator(
        speaker_name=speaker_name,
        speaker_goal=speaker_goal,
        speaker_personality=speaker_personality,
        partner_name=partner_name,
        partner_goal=partner_goal,
        partner_personality=partner_personality,
        location=location,
        step_summary=step_summary,
        recent_memories=recent_memories,
    )
    dialogue = getattr(result, "dialogue", None) or str(result)
    return ConversationResult(dialogue=dialogue)


async def explain_plan_step(
    agent_name: str,
    agent_goal: str,
    agent_personality: str,
    step_summary: str,
    location: str,
    recent_memories: str,
) -> PlanStepExplanation:
    explainer = get_current_step_explainer()
    result = explainer(
        agent_name=agent_name,
        agent_goal=agent_goal,
        agent_personality=agent_personality,
        step_summary=step_summary,
        location=location,
        recent_memories=recent_memories,
    )
    text = getattr(result, "explanation", None) or str(result)
    return PlanStepExplanation(text=text)


async def summarize_observations(agent_name: str, agent_goal: str, observations: str) -> ObservationSummary:
    summarizer = get_current_observation_summarizer()
    result = summarizer(
        agent_name=agent_name,
        agent_goal=agent_goal,
        observations=observations,
    )
    summary = getattr(result, "summary_points", None) or str(result)
    return ObservationSummary(summary=summary)


async def refine_reflection(
    raw_reflection: str,
    agent_personality: str,
    agent_goal: str,
) -> StructuredReflection:
    program = get_current_reflection_refiner()
    result = program(
        raw_reflection=raw_reflection,
        agent_personality=agent_personality,
        agent_goal=agent_goal,
    )
    text = getattr(result, "structured_reflection", None) or str(result)
    return StructuredReflection(text=text)


# ============ Compiled Module Management ============

# Compiled modules (loaded from disk)
_compiled_scorer = None
_compiled_planner = None
_compiled_action_selector = None
_compiled_conversation = None
_compiled_step_explainer = None
_compiled_observation_summarizer = None
_compiled_reflection_refiner = None
_use_compiled = False
_town_agent_program = None


def load_compiled_modules(compiled_dir: str = "compiled"):
    """
    Load compiled programs from compiled/ directory.

    Args:
        compiled_dir: Directory containing compiled modules
    """
    global _compiled_scorer, _compiled_planner, _compiled_action_selector, _compiled_conversation
    global _compiled_step_explainer, _compiled_observation_summarizer, _compiled_reflection_refiner

    from pathlib import Path
    import os

    project_root = Path(__file__).parent.parent
    compiled_path = project_root / compiled_dir / "compiled_scorer.json"
    planner_path = project_root / compiled_dir / "compiled_planner.json"
    action_path = project_root / compiled_dir / "compiled_action_selector.json"
    conversation_path = project_root / compiled_dir / "compiled_conversation.json"
    step_explainer_path = project_root / compiled_dir / "compiled_step_explainer.json"
    observation_summarizer_path = project_root / compiled_dir / "compiled_observation_summarizer.json"
    reflection_refiner_path = project_root / compiled_dir / "compiled_reflection_refiner.json"

    loaded = True

    if os.path.exists(compiled_path):
        try:
            _compiled_scorer = dspy.ChainOfThought(ScoreImportance)
            _compiled_scorer.load(str(compiled_path))
            logger.info(f"✅ Compiled scorer loaded from {compiled_path}")
        except Exception as e:
            logger.error(f"Failed to load compiled scorer: {e}")
            _compiled_scorer = None
            loaded = False
    else:
        logger.warning(f"Compiled scorer not found at {compiled_path}")
        loaded = False

    if os.path.exists(planner_path):
        try:
            _compiled_planner = dspy.Predict(PlanDay)
            _compiled_planner.load(str(planner_path))
            logger.info(f"✅ Compiled planner loaded from {planner_path}")
        except Exception as e:
            logger.error(f"Failed to load compiled planner: {e}")
            _compiled_planner = None
            loaded = False
    else:
        logger.info(f"Compiled planner not found at {planner_path}")

    def _load_optional(path, program_ctor, label, attr_name: str):
        nonlocal loaded
        if os.path.exists(path):
            try:
                program = program_ctor
                program.load(str(path))
                globals()[attr_name] = program
                logger.info(f"✅ {label} loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load {label}: {e}")
                globals()[attr_name] = None
                loaded = False
        else:
            logger.debug(f"{label} not found at {path}")
            globals()[attr_name] = None

    _load_optional(action_path, dspy.Predict(ChooseNextAction), "Compiled action selector", "_compiled_action_selector")
    _load_optional(conversation_path, dspy.ChainOfThought(PlanConversation), "Compiled plan conversation", "_compiled_conversation")
    _load_optional(step_explainer_path, dspy.Predict(PlanStepExplainer), "Compiled plan step explainer", "_compiled_step_explainer")
    _load_optional(observation_summarizer_path, dspy.ChainOfThought(ObservationSummarizer), "Compiled observation summarizer", "_compiled_observation_summarizer")
    _load_optional(reflection_refiner_path, dspy.ChainOfThought(ReflectionRefiner), "Compiled reflection refiner", "_compiled_reflection_refiner")

    return loaded


def use_compiled(enabled: bool = True):
    """
    Switch between compiled and uncompiled modules.

    Args:
        enabled: True to use compiled modules, False for uncompiled baseline
    """
    global _use_compiled

    if enabled and (_compiled_scorer is None and _compiled_planner is None):
        logger.warning("Compiled modules not loaded, attempting to load...")
        if not load_compiled_modules():
            logger.error("Cannot enable compiled modules, using uncompiled")
            _use_compiled = False
            return

    _use_compiled = enabled
    status = "compiled" if enabled else "uncompiled"
    logger.info(f"Using {status} modules")


def get_current_scorer():
    """
    Get the currently active scorer (compiled or uncompiled).

    Returns:
        Active scorer module
    """
    if _use_compiled and _compiled_scorer is not None:
        return _compiled_scorer
    return scorer


def get_current_planner():
    """
    Get the currently active planner (compiled or uncompiled).
    """
    if _use_compiled and _compiled_planner is not None:
        return _compiled_planner
    return planner


# ============ Module Info ============

def get_module_info():
    """Return info about configured modules."""
    return {
        "configured": _configured,
        "use_compiled": _use_compiled,
        "modules": {
            "scorer": {
                "type": "ChainOfThought",
                "signature": "ScoreImportance",
                "compiled": _use_compiled and _compiled_scorer is not None,
                "available_compiled": _compiled_scorer is not None
            },
            "reflector": {
                "type": "ChainOfThought",
                "signature": "Reflect",
                "compiled": False
            },
            "planner": {
                "type": "Predict",
                "signature": "PlanDay",
                "compiled": _use_compiled and _compiled_planner is not None,
                "available_compiled": _compiled_planner is not None
            }
        }
    }


def get_town_agent_program():
    """Return (and lazily construct) the composite TownAgent program."""
    global _town_agent_program
    if _town_agent_program is None:
        from programs import TownAgentProgram  # Local import to avoid circular dependency

        _town_agent_program = TownAgentProgram()
        compiled_path = Path(__file__).parent.parent / "compiled" / "compiled_town_agent.json"
        if compiled_path.exists():
            try:
                _town_agent_program.load(str(compiled_path))
                logger.info("✅ Compiled TownAgent loaded from %s", compiled_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load compiled TownAgent module: %s", exc)
    return _town_agent_program
