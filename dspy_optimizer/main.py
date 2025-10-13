from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status

from backend.dspy_modules import configure_dspy

from .config import AppState, load_settings
from .loader import CompiledModuleCache
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PlannerRequest,
    PlannerResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DSPy Optimizer Service",
    description="Expose compiled DSPy prompt programs over HTTP.",
    version="0.1.0",
)


def get_app_state() -> AppState:
    if not hasattr(app.state, "optimizer_state"):
        settings = load_settings()
        compiled_cache = CompiledModuleCache(settings.compiled_dir)
        app.state.optimizer_state = AppState(settings=settings)
        app.state.compiled_cache = compiled_cache
    return app.state.optimizer_state


def get_cache() -> CompiledModuleCache:
    # Ensure state initialized
    get_app_state()
    return app.state.compiled_cache  # type: ignore[attr-defined]


def authorize_request(
    state: AppState = Depends(get_app_state),
    token: Optional[str] = Header(default=None, alias="X-Optimizer-Token"),
) -> None:
    """Simple shared-secret auth guard."""
    expected = state.settings.auth_token
    if expected and token != expected:
        logger.warning("Rejected request with invalid auth token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")


def ensure_dspy(state: AppState) -> None:
    """Ensure DSPy global settings are configured once."""
    if state.dspy_configured:
        return

    settings = state.settings
    api_key = (
        settings.groq_api_key
        if settings.provider == "groq"
        else settings.together_api_key
    )
    configure_dspy(api_key=api_key, provider=settings.provider, model=settings.model)
    state.dspy_configured = True
    logger.info("DSPy configured for optimizer service (provider=%s)", settings.provider)


@app.get("/healthz", response_model=HealthResponse)
def healthcheck(state: AppState = Depends(get_app_state)) -> HealthResponse:
    cache: CompiledModuleCache = get_cache()
    modules = sorted(cache.available_modules().keys())
    return HealthResponse(status="ok", compiled_modules=modules)


@app.post(
    "/planner/plan_day",
    response_model=PlannerResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def plan_day(
    request: PlannerRequest,
    state: AppState = Depends(get_app_state),
    cache: CompiledModuleCache = Depends(get_cache),
    _: None = Depends(authorize_request),
) -> PlannerResponse:
    """
    Execute the compiled PlanDay module if available.
    """
    ensure_dspy(state)

    logger.info("Received plan_day request for agent '%s'", request.agent_name)

    planner = None
    compiled = False

    if "plan_day" in cache.available_modules():
        try:
            planner = cache.get("plan_day")
            compiled = True
        except RuntimeError as exc:
            logger.warning("Falling back to uncompiled planner: %s", exc)

    if planner is None:
        import dspy
        from backend.dspy_modules import PlanDay  # avoid circular import at module scope

        planner = dspy.Predict(PlanDay)

    recent_events = "\n".join(request.recent_events) if request.recent_events else ""
    relevant_memories = "\n".join(request.relevant_memories) if request.relevant_memories else ""

    try:
        result = planner(
            agent_goal=request.agent_goal,
            agent_personality=request.agent_personality,
            current_time=request.current_time,
            current_location=request.current_location,
            recent_events=recent_events,
            relevant_memories=relevant_memories,
        )
    except Exception as exc:
        logger.exception("PlanDay inference failed for agent '%s'", request.agent_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    reasoning = getattr(result, "reasoning", "")
    plan = getattr(result, "plan", "")

    return PlannerResponse(reasoning=reasoning, plan=plan, compiled=compiled)


@app.post(
    "/scorer/score_importance",
    response_model=ScoreResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def score_importance(
    request: ScoreRequest,
    state: AppState = Depends(get_app_state),
    cache: CompiledModuleCache = Depends(get_cache),
    _: None = Depends(authorize_request),
) -> ScoreResponse:
    """Evaluate observation importance using compiled scorer."""
    ensure_dspy(state)

    scorer = None
    compiled = False

    if "score_importance" in cache.available_modules():
        try:
            scorer = cache.get("score_importance")
            compiled = True
        except RuntimeError as exc:
            logger.warning("Falling back to uncompiled scorer: %s", exc)

    if scorer is None:
        import dspy
        from backend.dspy_modules import ScoreImportance

        scorer = dspy.ChainOfThought(ScoreImportance)

    try:
        result = scorer(
            observation=request.observation,
            agent_goal=request.agent_goal,
            agent_personality=request.agent_personality,
        )
    except Exception as exc:
        logger.exception("ScoreImportance inference failed for agent '%s'", request.agent_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    reasoning = getattr(result, "reasoning", "")
    score = int(getattr(result, "score", 0))

    return ScoreResponse(reasoning=reasoning, score=score, compiled=compiled)


@app.on_event("startup")
def on_startup() -> None:
    settings = load_settings()
    logger.info(
        "Starting DSPy optimizer service (compiled_dir=%s, allow_compile=%s)",
        settings.compiled_dir,
        settings.allow_compile,
    )
