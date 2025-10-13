"""
DSPy modules for Mini-Town.
Day 2: Uncompiled ScoreImportance and Reflect modules.
"""

import os
import logging
import dspy
import yaml
from typing import Optional
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
        lm = dspy.LM(
            model=f"groq/{model}",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == 'together':
        lm = dspy.LM(
            model=f"together_ai/{model}",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == 'openai':
        lm = dspy.LM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    dspy.settings.configure(lm=lm)
    _configured = True
    logger.info(f"DSPy configured with {provider} LLM ({model})")


# ============ Signatures ============

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
    plan: str = dspy.OutputField(desc="Time-blocked plan. CRITICAL: Use EXACT event times from recent_events. Include coordinates (x, y) for locations.")


# ============ Uncompiled Modules ============

# Simple predictor (uncompiled baseline)
scorer = dspy.ChainOfThought(ScoreImportance)

# Chain-of-thought for reflection (uncompiled baseline)
reflector = dspy.ChainOfThought(Reflect)

# Simple predictor for planning (uncompiled, Day 6)
planner = dspy.Predict(PlanDay)


# ============ Helper Functions ============

async def score_observation(
    observation: str,
    agent_goal: str,
    agent_personality: str
) -> int:
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
        score = int(result.score)
        # Clamp to 1-10
        return max(1, min(10, score))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid score from LLM: {result.score}, using default")
        return 5  # Default middle score


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
) -> str:
    """
    Generate a daily plan for the agent.

    Args:
        agent_goal: Agent's high-level goal
        agent_personality: Agent's personality traits
        current_time: Current time string (e.g., "2:30 PM")
        current_location: Current location (e.g., "(100, 150)")
        recent_events: List of recent event strings (especially invitations)
        relevant_memories: List of relevant memory strings

    Returns:
        Plan string (time-blocked text)

    Raises:
        Exception: If LLM call fails
    """
    # Format events and memories as newline-separated strings
    events_str = "\n".join([f"- {event}" for event in recent_events]) if recent_events else "No recent events"
    memories_str = "\n".join([f"- {mem}" for mem in relevant_memories]) if relevant_memories else "No relevant memories"

    result = planner(
        agent_goal=agent_goal,
        agent_personality=agent_personality,
        current_time=current_time,
        current_location=current_location,
        recent_events=events_str,
        relevant_memories=memories_str
    )

    return result.plan


# ============ Compiled Module Management ============

# Compiled modules (loaded from disk)
_compiled_scorer = None
_use_compiled = False


def load_compiled_modules(compiled_dir: str = "compiled"):
    """
    Load compiled programs from compiled/ directory.

    Args:
        compiled_dir: Directory containing compiled modules
    """
    global _compiled_scorer

    from pathlib import Path
    import os

    project_root = Path(__file__).parent.parent
    compiled_path = project_root / compiled_dir / "compiled_scorer.json"

    if os.path.exists(compiled_path):
        try:
            # Create baseline structure
            _compiled_scorer = dspy.ChainOfThought(ScoreImportance)
            # Load compiled state
            _compiled_scorer.load(str(compiled_path))
            logger.info(f"✅ Compiled scorer loaded from {compiled_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load compiled scorer: {e}")
            return False
    else:
        logger.warning(f"Compiled scorer not found at {compiled_path}")
        return False


def use_compiled(enabled: bool = True):
    """
    Switch between compiled and uncompiled modules.

    Args:
        enabled: True to use compiled modules, False for uncompiled baseline
    """
    global _use_compiled

    if enabled and _compiled_scorer is None:
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
                "compiled": False
            }
        }
    }
