from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PlannerRequest(BaseModel):
    """Payload for requesting a day plan."""

    agent_name: str = Field(..., description="Human-readable agent identifier.")
    agent_goal: str = Field(..., description="High-level objective for the agent.")
    agent_personality: str = Field(..., description="Personality traits or modifiers.")
    current_time: str = Field(..., description="Current simulation time (e.g., '8:45 AM').")
    current_location: str = Field(..., description="Current location coordinates '(x, y)'.")
    recent_events: List[str] = Field(
        default_factory=list,
        description="Recent invitations with exact times (newline-safe list).",
    )
    relevant_memories: List[str] = Field(
        default_factory=list,
        description="Recent memories to feed into the planner.",
    )
    force_refresh: bool = Field(
        default=False,
        description="Bypass cache and recompute even if cached output exists.",
    )


class PlannerResponse(BaseModel):
    """Response object containing reasoning and plan text."""

    reasoning: str = Field(..., description="Model reasoning for audit/debug.")
    plan: str = Field(..., description="Time-blocked plan text.")
    compiled: bool = Field(..., description="Whether the compiled module handled the request.")


class ScoreRequest(BaseModel):
    """Payload for scoring an observation's importance."""

    agent_name: str = Field(..., description="Agent identifier for logging.")
    observation: str = Field(..., description="What the agent observed.")
    agent_goal: str = Field(..., description="Agent's current high-level goal.")
    agent_personality: str = Field(..., description="Description of personality traits.")
    force_refresh: bool = Field(
        default=False,
        description="Reserved for future caching control.",
    )


class ScoreResponse(BaseModel):
    reasoning: str = Field(..., description="Explanation for the assigned score.")
    score: int = Field(..., description="Importance score (1-10).")
    compiled: bool = Field(..., description="Whether the compiled module served the response.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Static status string, usually 'ok'.")
    compiled_modules: List[str] = Field(default_factory=list, description="Names of compiled modules in cache.")


class ErrorResponse(BaseModel):
    detail: str
