"""
Agent class for Mini-Town.
Day 0.5: Hardcoded random walk behavior + perception.
Day 2: LLM-based importance scoring and reflection.
Day 6: Plan execution with navigation.
Day 7+: TownAgent composite program integration.
"""

import asyncio
import re
import random
import logging
import math
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Deque
from dspy_modules import (
    score_observation,
    generate_reflection,
    generate_plan,
    get_planner_source,
    ObservationScore,
    PlanGeneration,
    PlanValidation,
    get_town_agent_program,
)
from utils import timed_llm_call, generate_embedding
from runtime.executor import (
    navigate_for_plan,
    select_active_step,
    select_upcoming_step,
    dispatch_next_action,
    steps_from_plan_text,
    steps_from_structured_plan,
)
from runtime.dataset_logger import log_town_agent_episode
from telemetry import PlanTelemetryPayload, record_plan_validation

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Represents high-level agent actions for UI/state tracking."""

    IDLE = "idle"
    EXPLORE = "active"
    NAVIGATE = "navigating"
    WAIT = "waiting"
    LOITER = "loitering"
    CONVERSE = "conversing"
    OBSERVE = "observing"


class Agent:
    """An autonomous agent with hardcoded behaviors for Day 0.5."""

    def __init__(
        self,
        agent_id: int,
        name: str,
        x: float,
        y: float,
        goal: str = "",
        personality: str = "",
        map_width: int = 800,
        map_height: int = 600,
        perception_radius: float = 50.0,
        retrieval_alpha: float = 0.5,
        retrieval_beta: float = 0.3,
        retrieval_gamma: float = 0.2,
        speed: float = 2.0,
        use_town_agent_program: bool = False,
    ):
        """Initialize agent with position and metadata."""
        self.id = agent_id
        self.name = name
        self.x = x
        self.y = y
        self.goal = goal
        self.personality = personality
        self.state = "active"
        self.current_plan = "Wandering around"
        self.current_plan_structured: Optional[Dict[str, Any]] = None
        self.plan_last_updated = None  # Timestamp of last plan update
        self.plan_source: str = "unknown"  # Tracks how the plan was generated
        self.last_plan_validation: Optional[PlanValidation] = None

        # Plan execution (Day 6+)
        self.parsed_plan = []  # List of parsed plan steps
        self._last_parsed_time = None  # Track when plan was last parsed
        self.plan_preset_id: Optional[str] = None
        self.current_step_index: int = 0
        self.last_plan_step: Optional[Dict[str, Any]] = None
        self.last_plan_step_switch_tick: Optional[int] = None
        self.last_simulation_minutes: Optional[int] = None
        self.plan_time_offset: int = 0
        self.completed_conversation_steps: set[str] = set()
        self.use_town_agent_program = use_town_agent_program
        self.pending_next_action: Optional[str] = None
        self.pending_next_action_reasoning: Optional[str] = None
        self.last_executed_action: Optional[str] = None
        self.last_action_reasoning: Optional[str] = None
        self.available_actions: List[str] = ["move", "talk", "wait", "observe"]

        # Map boundaries
        self.map_width = map_width
        self.map_height = map_height

        # Perception
        self.perception_radius = perception_radius

        # Movement parameters for random walk
        self.speed = speed  # pixels per tick
        self.direction_change_probability = 0.25  # Chance to change direction

        # Current velocity
        self.vx = 0.0
        self.vy = 0.0

        # Reflection tracking (Day 2)
        self.reflection_score = 0.0  # Accumulates importance scores
        self.reflection_threshold = 3.5  # From config or personality

        # Retrieval weights for memory search (personality-specific)
        self.retrieval_alpha = retrieval_alpha  # relevance
        self.retrieval_beta = retrieval_beta    # recency
        self.retrieval_gamma = retrieval_gamma  # importance

        # Action/context
        self.action_type = ActionType.EXPLORE
        self._queued_observations: List[str] = []
        self.recent_observations: Deque[Dict[str, Any]] = deque(maxlen=5)
        self._loiter_target: Optional[Tuple[float, float]] = None
        self._loiter_retarget_at: Optional[datetime] = None
        self._next_conversation_time: datetime = datetime.now()
        self._next_plan_update_tick: int = 0

        # Enhanced UI fields (populated by DSPy signatures)
        self.step_explanation: Optional[str] = None
        self.observation_summary: Optional[str] = None
        self.recent_conversations: List[Dict[str, Any]] = []

        logger.info(f"Agent {self.id} ({self.name}) initialized at ({x}, {y})")

    def _set_action(self, action: ActionType):
        """Update current action type and exposed state string."""
        self.action_type = action
        self.state = action.value

    def queue_observation(self, text: str):
        """Queue an observation to be emitted on the next update call."""
        self._queued_observations.append(text)

    def _move_towards(self, target_x: float, target_y: float, speed: float):
        """Move toward a target position with the given speed."""
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            self.vx = 0.0
            self.vy = 0.0
            return

        step = min(speed, distance)
        self.vx = (dx / distance) * step
        self.vy = (dy / distance) * step

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        self.x = max(0.0, min(self.map_width, new_x))
        self.y = max(0.0, min(self.map_height, new_y))

    def _loiter_and_socialize(
        self,
        anchor_x: float,
        anchor_y: float,
        other_agents: List['Agent'],
        current_time: datetime
    ) -> List[str]:
        """Wander around an event location and occasionally converse."""
        self._set_action(ActionType.LOITER)

        if (
            self._loiter_target is None
            or self._loiter_retarget_at is None
            or current_time >= self._loiter_retarget_at
        ):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(6.0, 20.0)
            target_x = anchor_x + math.cos(angle) * radius
            target_y = anchor_y + math.sin(angle) * radius
            target_x = max(0.0, min(self.map_width, target_x))
            target_y = max(0.0, min(self.map_height, target_y))
            self._loiter_target = (target_x, target_y)
            self._loiter_retarget_at = current_time + timedelta(seconds=random.randint(5, 12))

        if self._loiter_target:
            target_x, target_y = self._loiter_target
            self._move_towards(target_x, target_y, self.speed * 0.6)
            if math.hypot(self.x - target_x, self.y - target_y) < 2.0:
                self._loiter_target = None

        conversations = self._maybe_converse(other_agents, current_time)
        return conversations

    def _maybe_converse(self, other_agents: List['Agent'], current_time: datetime) -> List[str]:
        """Attempt to start a simple conversation with nearby agents."""
        if current_time < self._next_conversation_time:
            return []

        candidates = []
        for other in other_agents:
            if other.id == self.id:
                continue
            if getattr(other, "action_type", ActionType.EXPLORE) not in {
                ActionType.LOITER,
                ActionType.CONVERSE,
                ActionType.WAIT,
            }:
                continue
            distance = math.hypot(self.x - other.x, self.y - other.y)
            if distance <= 25.0:
                candidates.append((distance, other))

        if not candidates:
            return []

        _, partner = min(candidates, key=lambda item: item[0])
        if self.id > partner.id:
            # Let the lower-ID agent initiate to avoid duplicate conversations.
            return []

        cooldown = random.randint(40, 80)
        self._next_conversation_time = current_time + timedelta(seconds=cooldown)
        partner._next_conversation_time = max(
            partner._next_conversation_time,
            current_time + timedelta(seconds=20)
        )

        self_line = f"Had a conversation with {partner.name} at the event."
        partner_line = f"Had a conversation with {self.name} at the event."
        partner.queue_observation(partner_line)

        # Track conversation for both agents
        conversation_summary = f"{self.name} and {partner.name} had a conversation"
        self.track_conversation(conversation_summary, [self.id, partner.id])
        partner.track_conversation(conversation_summary, [self.id, partner.id])

        partner._set_action(ActionType.CONVERSE)
        self._set_action(ActionType.CONVERSE)

        return [self_line]

    def update(
        self,
        other_agents: List['Agent'],
        simulation_minutes: Optional[int] = None,
        sim_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update agent state for one tick, including navigation, waiting, and loitering behavior.

        Returns:
            Dict with agent state and observations
        """
        current_time = datetime.now()
        sim_minutes = simulation_minutes if simulation_minutes is not None else current_time.hour * 60 + current_time.minute
        self.last_simulation_minutes = sim_minutes
        extra_observations: List[str] = []

        current_step = None
        upcoming_step = None

        if self.current_plan and self.plan_last_updated:
            if not self.parsed_plan or self._last_parsed_time != self.plan_last_updated:
                self.parsed_plan = self.parse_plan()
                self._last_parsed_time = self.plan_last_updated
                if self.parsed_plan:
                    logger.info(f"Agent {self.id} parsed plan into {len(self.parsed_plan)} steps")

            if self.parsed_plan:
                current_step, current_index = select_active_step(
                    self.parsed_plan,
                    sim_minutes,
                    getattr(self, "plan_time_offset", 0),
                )
                upcoming_step = select_upcoming_step(
                    self.parsed_plan,
                    sim_minutes,
                    getattr(self, "plan_time_offset", 0),
                    window_minutes=15,
                )

                extra_observations.extend(
                    navigate_for_plan(
                        agent=self,
                        current_step=current_step,
                        upcoming_step=upcoming_step,
                        other_agents=other_agents,
                        current_time=current_time,
                        action_enum=ActionType,
                    )
                )

                if current_step and self.last_plan_step is not current_step:
                    self.last_plan_step = current_step
                    if current_index is not None:
                        self.current_step_index = current_index
                    self.last_plan_step_switch_tick = sim_tick
            else:
                self._loiter_target = None
                self._set_action(ActionType.EXPLORE)
                self._random_walk()
        else:
            self._loiter_target = None
            self._set_action(ActionType.EXPLORE)
            self._random_walk()

        observations = self._perceive(other_agents)
        if extra_observations:
            observations.extend(extra_observations)
        if self._queued_observations:
            observations.extend(self._queued_observations)
            self._queued_observations.clear()

        if self.pending_next_action:
            dispatch_observations = dispatch_next_action(
                agent=self,
                action=self.pending_next_action,
                current_step=current_step or upcoming_step,
                other_agents=other_agents,
                current_time=current_time,
                action_enum=ActionType,
            )
            if dispatch_observations:
                observations.extend(dispatch_observations)
            self.last_executed_action = self.pending_next_action
            self.last_action_reasoning = self.pending_next_action_reasoning
            self.pending_next_action = None
            self.pending_next_action_reasoning = None

        active_step_payload: Optional[Dict[str, Any]] = None
        if self.last_plan_step:
            loc = self.last_plan_step.get('location')
            active_step_payload = {
                "start": self.last_plan_step.get('start_time_str'),
                "end": self.last_plan_step.get('end_time_str'),
                "description": self.last_plan_step.get('description'),
                "location": list(loc) if loc else None,
            }

        if observations:
            logger.info(
                f"Agent {self.id} ({self.name}) at ({self.x:.1f}, {self.y:.1f}) "
                f"perceives: {observations}"
            )

        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "state": self.state,
            "observations": observations,
            "current_plan": self.current_plan,
            "plan_source": self.plan_source,
            "plan_last_updated": self.plan_last_updated.isoformat() if self.plan_last_updated else None,
            "plan_preset_id": self.plan_preset_id,
            "active_plan_step": active_step_payload,
            "simulation_minutes": sim_minutes,
            "current_step_index": self.current_step_index,
        }

    def _random_walk(self):
        """Move agent in a random walk pattern."""
        self._set_action(ActionType.EXPLORE)

        # Occasionally change direction
        if random.random() < self.direction_change_probability or (self.vx == 0 and self.vy == 0):
            # Pick a random direction
            angle = random.uniform(0, 2 * math.pi)
            self.vx = math.cos(angle) * self.speed
            self.vy = math.sin(angle) * self.speed

        # Update position
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Bounce off walls
        if new_x < 0 or new_x > self.map_width:
            self.vx = -self.vx
            new_x = max(0, min(self.map_width, new_x))

        if new_y < 0 or new_y > self.map_height:
            self.vy = -self.vy
            new_y = max(0, min(self.map_height, new_y))

        self.x = new_x
        self.y = new_y

    def navigate_to(self, target_x: float, target_y: float):
        """
        Navigate agent toward a target location.

        Day 6 Fix 2A: Simple direct navigation without pathfinding.
        Agent moves in a straight line toward target and stops when close.

        Args:
            target_x: Target x coordinate
            target_y: Target y coordinate
        """
        distance = math.hypot(target_x - self.x, target_y - self.y)

        if distance < 10.0:
            self.vx = 0
            self.vy = 0
            self.x = target_x
            self.y = target_y
            logger.debug(f"Agent {self.id} arrived at target ({target_x:.0f}, {target_y:.0f})")
            return

        self._set_action(ActionType.NAVIGATE)
        self._move_towards(target_x, target_y, self.speed)
        logger.debug(f"Agent {self.id} navigating to ({target_x:.0f}, {target_y:.0f}), distance: {distance:.1f}")

    def _perceive(self, other_agents: List['Agent']) -> List[str]:
        """
        Perceive nearby agents within perception radius.

        Args:
            other_agents: List of all other agents in the simulation

        Returns:
            List of observation strings
        """
        observations = []

        for other in other_agents:
            if other.id == self.id:
                continue

            # Calculate distance
            distance = self._distance_to(other)

            # Check if within perception radius
            if distance <= self.perception_radius:
                observation = f"{other.name} is nearby (distance: {distance:.1f})"
                observations.append(observation)

        return observations

    def _distance_to(self, other: 'Agent') -> float:
        """Calculate Euclidean distance to another agent."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        active_step_payload = None
        if self.last_plan_step:
            loc = self.last_plan_step.get('location')
            active_step_payload = {
                "start": self.last_plan_step.get('start_time_str'),
                "end": self.last_plan_step.get('end_time_str'),
                "description": self.last_plan_step.get('description'),
                "location": list(loc) if loc else None,
                "explanation": self.last_plan_step.get('explanation'),
            }
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "goal": self.goal,
            "personality": self.personality,
            "state": self.state,
            "current_plan": self.current_plan,
            "structured_plan": self.current_plan_structured,
            "plan_validation": (
                {
                    "preserved_event_times": self.last_plan_validation.preserved_event_times,
                    "missing_event_times": self.last_plan_validation.missing_event_times,
                    "overlaps_detected": self.last_plan_validation.overlaps_detected,
                    "invalid_locations": self.last_plan_validation.invalid_locations,
                }
                if self.last_plan_validation
                else None
            ),
            "plan_source": self.plan_source,
            "plan_last_updated": self.plan_last_updated.isoformat() if self.plan_last_updated else None,
            "plan_preset_id": self.plan_preset_id,
            "current_step_index": self.current_step_index,
            "active_plan_step": active_step_payload,
            "observations": list(self.recent_observations),
            "step_explanation": self.step_explanation,
            "observation_summary": self.observation_summary,
            "recent_conversations": self.recent_conversations,
            "use_town_agent": self.use_town_agent_program,
            "last_action": self.last_executed_action,
            "last_action_reasoning": self.last_action_reasoning,
        }

    async def score_and_store_observation(self, obs: str, memory_store) -> float:
        """
        Score observation importance via LLM and store in memory.

        Args:
            obs: Observation string
            memory_store: MemoryStore instance

        Returns:
            Importance score (0-1 normalized)
        """
        score_value = 5
        reasoning_text: Optional[str] = None
        timestamp = datetime.now()

        try:
            score_result = await timed_llm_call(
                score_observation,
                signature_name="ScoreImportance",
                timeout=5.0,
                observation=obs,
                agent_goal=self.goal,
                agent_personality=self.personality
            )

            if isinstance(score_result, ObservationScore):
                score_value = score_result.score
                reasoning_text = self._summarize_reasoning(score_result.reasoning)
            else:
                score_value = int(score_result)

            importance = max(0.0, min(1.0, score_value / 10.0))

        except Exception as e:
            logger.warning(f"Agent {self.id} LLM scoring failed: {e}, using default")
            importance = 0.3  # Fallback
            score_value = int(importance * 10)
            reasoning_text = "Scoring failed; defaulted to importance 0.3"
            reasoning_text = self._summarize_reasoning(reasoning_text)
            self.state = "confused"

        # Generate embedding
        embedding = generate_embedding(obs)

        # Store memory
        memory_store.store_memory(
            agent_id=self.id,
            content=obs,
            importance=importance,
            embedding=embedding,
            timestamp=timestamp
        )

        self.recent_observations.appendleft({
            "text": obs,
            "score": score_value,
            "importance": round(importance, 3),
            "reasoning": reasoning_text,
            "timestamp": timestamp.isoformat()
        })

        # Accumulate for reflection
        self.reflection_score += importance

        return importance

    async def maybe_reflect(self, memory_store):
        """
        Check if agent should reflect and generate insight if threshold exceeded.

        Args:
            memory_store: MemoryStore instance

        Returns:
            Insight string if reflected, None otherwise
        """
        if self.reflection_score < self.reflection_threshold:
            return None

        logger.info(f"Agent {self.id} ({self.name}) triggering reflection (score: {self.reflection_score:.2f})")

        # Get relevant memories using vector search
        # Generate query embedding based on agent's goal
        from utils import generate_embedding

        query_text = f"Important events and observations related to: {self.goal}"
        query_embedding = generate_embedding(query_text)

        # Retrieve using triad scoring (relevance + recency + importance)
        # Use personality-specific weights
        memories = memory_store.retrieve_memories_by_vector(
            agent_id=self.id,
            query_embedding=query_embedding,
            top_k=10,
            alpha=self.retrieval_alpha,
            beta=self.retrieval_beta,
            gamma=self.retrieval_gamma
        )

        if not memories:
            self.reflection_score = 0.0
            return None

        # Format as strings
        memory_strings = [mem['content'] for mem in memories]

        try:
            # Generate reflection via LLM
            insight = await timed_llm_call(
                generate_reflection,
                signature_name="Reflect",
                timeout=5.0,
                recent_memories=memory_strings,
                agent_personality=self.personality,
                agent_goal=self.goal
            )

            logger.info(f"Agent {self.id} insight: {insight[:100]}...")

            # Reset accumulator
            self.reflection_score = 0.0

            return insight

        except Exception as e:
            logger.error(f"Agent {self.id} reflection failed: {e}")
            self.state = "confused"
            return None

    async def _update_plan_with_town_agent(
        self,
        memory_store,
        current_time: datetime,
        simulation_minutes: Optional[int],
        recent_events: List[str],
        relevant_memories: List[str],
    ) -> Optional[str]:
        program = get_town_agent_program()

        recent_obs_text = [obs.get("text", "") for obs in list(self.recent_observations)]

        try:
            response = await asyncio.to_thread(
                program,
                agent_name=self.name,
                agent_goal=self.goal,
                agent_personality=self.personality,
                current_time=current_time.strftime('%I:%M %p'),
                current_location=f"({self.x:.0f}, {self.y:.0f})",
                recent_observations=recent_obs_text,
                recent_events=recent_events,
                relevant_memories=relevant_memories,
                candidate_actions=self.available_actions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Agent {self.id} TownAgent program failed: {exc}")
            self.state = "confused"
            return None

        plan_text = response.plan_text
        self.current_plan = plan_text
        self.current_plan_structured = response.plan_structured
        self.last_plan_validation = response.plan_validation
        self.plan_last_updated = current_time

        compiled_path = Path(__file__).parent.parent / "compiled" / "compiled_town_agent.json"
        self.plan_source = "town_agent_compiled" if compiled_path.exists() else "town_agent"

        self.plan_preset_id = None
        self.parsed_plan = self.parse_plan()
        self.current_step_index = 0
        self.last_plan_step = None
        self.last_plan_step_switch_tick = None
        self._last_parsed_time = self.plan_last_updated
        self.align_plan_execution(simulation_minutes)
        self.completed_conversation_steps.clear()

        self.pending_next_action = response.next_action
        self.pending_next_action_reasoning = response.next_action_reasoning
        self.last_executed_action = None
        self.last_action_reasoning = None

        if response.reflection:
            reflection_text = response.reflection.strip()
            if reflection_text:
                memory_store.store_memory(
                    agent_id=self.id,
                    content=f"[REFLECTION] {reflection_text}",
                    importance=0.9,
                    embedding=generate_embedding(reflection_text),
                    timestamp=current_time,
                )
                self.reflection_score = 0.0

        record_plan_validation(
            PlanTelemetryPayload(
                agent_id=self.id,
                plan_source=self.plan_source,
                preserved_event_times=self.last_plan_validation.preserved_event_times,
                missing_event_times=self.last_plan_validation.missing_event_times,
                overlaps_detected=self.last_plan_validation.overlaps_detected,
                invalid_locations=self.last_plan_validation.invalid_locations,
                reasoning=response.next_action_reasoning,
                summary=response.plan_structured.get("summary") if isinstance(response.plan_structured, dict) else None,
            )
        )

        memory_store.update_agent_plan(
            agent_id=self.id,
            plan=self.current_plan,
            plan_source=self.plan_source,
            plan_updated_at=current_time,
        )

        recent_obs_text = [obs.get("text", "") for obs in list(self.recent_observations)]
        log_town_agent_episode(
            agent_id=self.id,
            agent_name=self.name,
            agent_goal=self.goal,
            agent_personality=self.personality,
            current_time=current_time.strftime('%I:%M %p'),
            current_location=f"({self.x:.0f}, {self.y:.0f})",
            recent_observations=recent_obs_text,
            recent_events=recent_events,
            relevant_memories=relevant_memories,
            candidate_actions=self.available_actions,
            plan_text=plan_text,
            plan_structured=response.plan_structured,
            plan_source=self.plan_source,
            use_town_agent=True,
            notes=response.plan_structured.get("summary") if isinstance(response.plan_structured, dict) else None,
        )

        return plan_text

    async def update_plan(
        self,
        memory_store,
        current_time: Optional[datetime] = None,
        simulation_minutes: Optional[int] = None,
    ) -> Optional[str]:
        """
        Generate or update the agent's plan based on recent events and memories.

        Args:
            memory_store: MemoryStore instance
            current_time: Current simulation time (defaults to now)

        Returns:
            Plan string if generated, None otherwise
        """
        if current_time is None:
            current_time = datetime.now()

        logger.info(f"Agent {self.id} ({self.name}) updating plan at {current_time.strftime('%I:%M %p')}")

        # Get recent important memories (especially event invitations)
        from utils import generate_embedding

        # Query for event-related and goal-related memories
        # Day 6 Fix 1B: Improved query for better semantic matching with party invitations
        # Day 6 Fix 4: Customize query based on personality for better retrieval
        if "introverted" in self.personality.lower() or "analytical" in self.personality.lower() or "reclusive" in self.personality.lower():
            # For introverted agents: emphasize explicit invitations over general social events
            query_text = f"Explicit invitations addressed to me. Events I was specifically invited to. Goal: {self.goal}"
        else:
            # For social agents: broader social events query
            query_text = f"Recent party invitations, social events, and plans. Goal: {self.goal}"

        query_embedding = generate_embedding(query_text)

        # Retrieve relevant memories with emphasis on recent events
        # Use higher beta (recency) for planning than reflection
        memories = memory_store.retrieve_memories_by_vector(
            agent_id=self.id,
            query_embedding=query_embedding,
            top_k=8,
            alpha=0.4,  # relevance
            beta=0.4,   # recency (higher for planning)
            gamma=0.2   # importance
        )

        if not memories:
            logger.warning(f"Agent {self.id} has no memories for planning")
            return None

        # Extract recent events (last 10 minutes)
        recent_cutoff = current_time - timedelta(minutes=10)
        recent_events = []
        relevant_memories = []

        for mem in memories:
            mem_time = mem.get('ts', mem.get('timestamp'))
            if mem_time and mem_time >= recent_cutoff:
                # Check if it's an invitation or event
                content = mem['content']
                if any(keyword in content.lower() for keyword in ['invited', 'party', 'event', 'meeting', 'gathering']):
                    recent_events.append(content)
                else:
                    relevant_memories.append(content)
            else:
                relevant_memories.append(mem['content'])

        # Limit to most relevant
        recent_events = recent_events[:5]
        relevant_memories = relevant_memories[:5]

        if self.use_town_agent_program:
            plan_text = await self._update_plan_with_town_agent(
                memory_store,
                current_time,
                simulation_minutes,
                recent_events,
                relevant_memories,
            )
            if plan_text:
                logger.info(f"Agent {self.id} plan (TownAgent): {plan_text[:100]}...")
                return plan_text
            # Fall back to baseline if TownAgent fails
            logger.warning(f"Agent {self.id} falling back to baseline planner after TownAgent failure")

        try:
            # Format current time and location
            time_str = current_time.strftime('%I:%M %p')
            location_str = f"({self.x:.0f}, {self.y:.0f})"

            # Generate plan via LLM
            plan_result: PlanGeneration = await timed_llm_call(
                generate_plan,
                signature_name="PlanDay",
                timeout=5.0,
                agent_goal=self.goal,
                agent_personality=self.personality,
                current_time=time_str,
                current_location=location_str,
                recent_events=recent_events,
                relevant_memories=relevant_memories
            )

            plan_text = plan_result.text
            logger.info(f"Agent {self.id} plan: {plan_text[:100]}...")

            # Update agent's current plan
            self.current_plan = plan_text
            self.current_plan_structured = plan_result.structured
            self.last_plan_validation = plan_result.validation
            self.plan_last_updated = current_time
            self.plan_source = get_planner_source()
            self.plan_preset_id = None
            self.parsed_plan = self.parse_plan()
            self.current_step_index = 0
            self.last_plan_step = None
            self.last_plan_step_switch_tick = None
            self._last_parsed_time = self.plan_last_updated
            self.align_plan_execution(simulation_minutes)
            self.completed_conversation_steps.clear()
            memory_store.update_agent_plan(
                agent_id=self.id,
                plan=self.current_plan,
                plan_source=self.plan_source,
                plan_updated_at=current_time,
            )

            summary = None
            if isinstance(self.current_plan_structured, dict):
                summary = self.current_plan_structured.get("summary")
            record_plan_validation(
                PlanTelemetryPayload(
                    agent_id=self.id,
                    plan_source=self.plan_source,
                    preserved_event_times=self.last_plan_validation.preserved_event_times if self.last_plan_validation else [],
                    missing_event_times=self.last_plan_validation.missing_event_times if self.last_plan_validation else [],
                    overlaps_detected=bool(self.last_plan_validation.overlaps_detected) if self.last_plan_validation else False,
                    invalid_locations=self.last_plan_validation.invalid_locations if self.last_plan_validation else [],
                    reasoning=plan_result.reasoning,
                    summary=summary,
                )
            )

            log_town_agent_episode(
                agent_id=self.id,
                agent_name=self.name,
                agent_goal=self.goal,
                agent_personality=self.personality,
                current_time=time_str,
                current_location=location_str,
                recent_observations=[obs.get("text", "") for obs in list(self.recent_observations)],
                recent_events=recent_events,
                relevant_memories=relevant_memories,
                candidate_actions=self.available_actions,
                plan_text=plan_text,
                plan_structured=plan_result.structured,
                plan_source=self.plan_source,
                use_town_agent=False,
                notes=summary,
            )

            return plan_text

        except Exception as e:
            logger.error(f"Agent {self.id} planning failed: {e}")
            self.state = "confused"
            self.plan_source = "error"
            return None

    async def generate_step_explanation(self, memory_store) -> Optional[str]:
        """
        Generate an explanation for why the current plan step matters.

        Args:
            memory_store: MemoryStore instance

        Returns:
            Explanation string if generated, None otherwise
        """
        if not self.last_plan_step or not self.last_plan_step.get('description'):
            return None
            
        try:
            from dspy_modules import explain_plan_step
            from utils import generate_embedding
            
            # Get recent memories for context
            query_text = f"{self.last_plan_step.get('description')} and {self.goal}"
            query_embedding = generate_embedding(query_text)
            memories = memory_store.retrieve_memories_by_vector(
                agent_id=self.id,
                query_embedding=query_embedding,
                top_k=5,
                alpha=0.5,
                beta=0.3,
                gamma=0.2
            )
            
            memories_str = "\n".join([f"- {mem['content']}" for mem in memories]) if memories else "No recent context"
            location_str = f"({self.x:.0f}, {self.y:.0f})"
            
            result = await explain_plan_step(
                agent_name=self.name,
                agent_goal=self.goal,
                agent_personality=self.personality,
                step_summary=self.last_plan_step.get('description', 'Current task'),
                location=location_str,
                recent_memories=memories_str
            )
            
            explanation = result.text if hasattr(result, 'text') else str(result)
            self.step_explanation = explanation
            
            # Store explanation in the plan step dict as well
            if self.last_plan_step:
                self.last_plan_step['explanation'] = explanation
            
            logger.debug(f"Agent {self.id} step explanation: {explanation[:80]}...")
            return explanation
            
        except Exception as e:
            logger.warning(f"Agent {self.id} step explanation failed: {e}")
            return None

    async def generate_observation_summary(self) -> Optional[str]:
        """
        Generate a summary of high-importance recent observations.
        
        Returns:
            Summary string if generated, None otherwise
        """
        if not self.recent_observations or len(self.recent_observations) == 0:
            return None
            
        try:
            from dspy_modules import summarize_observations
            
            # Filter for higher-importance observations (>= 0.5)
            important_obs = [
                obs for obs in self.recent_observations 
                if obs.get('importance', 0) >= 0.5
            ]
            
            if not important_obs:
                return None
            
            # Format observations as bullet list
            obs_lines = []
            for obs in important_obs[:10]:  # Limit to 10 most recent
                importance = obs.get('importance', 0)
                text = obs.get('text', '')
                obs_lines.append(f"• [{importance:.2f}] {text}")
            
            observations_str = "\n".join(obs_lines)
            
            result = await summarize_observations(
                agent_name=self.name,
                agent_goal=self.goal,
                observations=observations_str
            )
            
            summary = result.summary if hasattr(result, 'summary') else str(result)
            self.observation_summary = summary
            
            logger.debug(f"Agent {self.id} observation summary: {summary[:80]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Agent {self.id} observation summary failed: {e}")
            return None

    def track_conversation(self, conversation_text: str, participants: Optional[List[int]] = None):
        """
        Track a conversation event for display in the UI.
        
        Args:
            conversation_text: The conversation content or summary
            participants: List of agent IDs involved (optional)
        """
        # Keep only last 5 conversations
        max_conversations = 5
        
        conversation_digest = {
            "id": f"conv-{self.id}-{datetime.now().isoformat()}",
            "timestamp": datetime.now().isoformat(),
            "summary": conversation_text,
            "agents": participants if participants else [self.id],
            "step_description": self.last_plan_step.get('description') if self.last_plan_step else None
        }
        
        self.recent_conversations.insert(0, conversation_digest)
        
        # Keep only most recent conversations
        if len(self.recent_conversations) > max_conversations:
            self.recent_conversations = self.recent_conversations[:max_conversations]

    @staticmethod
    def _time_to_minutes(time_obj: datetime) -> int:
        return time_obj.hour * 60 + time_obj.minute

    def parse_plan(self) -> List[Dict[str, Any]]:
        """
        Parse the agent's current plan to extract actionable steps.

        Returns:
            List of plan step dicts with keys: start_time, end_time, location, description
        """
        if self.current_plan_structured:
            steps_structured = steps_from_structured_plan(self.current_plan_structured)
            if steps_structured:
                logger.debug("Agent %s parsed %d structured plan steps", self.id, len(steps_structured))
                return steps_structured

        if not self.current_plan:
            return []

        try:
            steps_text = steps_from_plan_text(self.current_plan)
            logger.debug("Agent %s parsed %d text plan steps", self.id, len(steps_text))
            return steps_text
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent %s plan parsing failed: %s", self.id, exc)
            return []

    def align_plan_execution(self, simulation_minutes: Optional[int]) -> None:
        """Align internal pointers with the simulation clock so steps execute immediately."""
        if not self.parsed_plan:
            self.plan_time_offset = 0
            self.current_step_index = 0
            return

        if simulation_minutes is None:
            self.plan_time_offset = 0
            self.current_step_index = 0
            return

        first_start = self.parsed_plan[0].get('start_minutes')
        if first_start is not None:
            self.plan_time_offset = first_start - simulation_minutes
        else:
            self.plan_time_offset = 0

        current_step, current_index = self.get_current_step(self.parsed_plan, simulation_minutes)
        if current_step is not None:
            self.last_plan_step = current_step
            if current_index is not None:
                self.current_step_index = current_index
        else:
            # Reset to first step if none active yet (plan hasn't started)
            self.current_step_index = 0
            if self.parsed_plan:
                self.last_plan_step = None

    def get_current_step(self, parsed_steps: List[Dict[str, Any]], simulation_minutes: Optional[int]) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Return the active plan step (and index) for the given simulation minute."""
        return select_active_step(parsed_steps, simulation_minutes, getattr(self, "plan_time_offset", 0))

    def get_upcoming_step(self, parsed_steps: List[Dict[str, Any]], simulation_minutes: Optional[int], window_minutes: int = 15) -> Optional[Dict[str, Any]]:
        return select_upcoming_step(
            parsed_steps,
            simulation_minutes,
            getattr(self, "plan_time_offset", 0),
            window_minutes=window_minutes,
        )

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"Agent({self.id}, {self.name}, x={self.x:.1f}, y={self.y:.1f})"
    def _summarize_reasoning(self, reasoning: Optional[str]) -> Optional[str]:
        if not reasoning:
            return None

        summary = reasoning.strip()
        summary = summary.replace("Reasoning:", "").strip()
        if "Score:" in summary:
            summary = summary.split("Score:")[0].strip()

        # Split into sentences and keep the last meaningful one
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
        if sentences:
            return sentences[-1]
        return summary[:200]
