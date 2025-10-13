"""
Agent class for Mini-Town.
Day 0.5: Hardcoded random walk behavior + perception.
Day 2: LLM-based importance scoring and reflection.
Day 6: Plan execution with navigation.
"""

import re
import random
import logging
import math
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dspy_modules import score_observation, generate_reflection, generate_plan, get_planner_source
from utils import timed_llm_call

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
        retrieval_gamma: float = 0.2
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
        self.plan_last_updated = None  # Timestamp of last plan update
        self.plan_source: str = "unknown"  # Tracks how the plan was generated

        # Plan execution (Day 6)
        self.parsed_plan = []  # List of parsed plan steps
        self._last_parsed_time = None  # Track when plan was last parsed

        # Map boundaries
        self.map_width = map_width
        self.map_height = map_height

        # Perception
        self.perception_radius = perception_radius

        # Movement parameters for random walk
        self.speed = 2.0  # pixels per tick
        self.direction_change_probability = 0.2  # Chance to change direction

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
        self._loiter_target: Optional[Tuple[float, float]] = None
        self._loiter_retarget_at: Optional[datetime] = None
        self._next_conversation_time: datetime = datetime.now()

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

        partner._set_action(ActionType.CONVERSE)
        self._set_action(ActionType.CONVERSE)

        return [self_line]

    def update(self, other_agents: List['Agent']) -> Dict[str, Any]:
        """
        Update agent state for one tick, including navigation, waiting, and loitering behavior.

        Returns:
            Dict with agent state and observations
        """
        current_time = datetime.now()
        extra_observations: List[str] = []

        if self.current_plan and self.plan_last_updated:
            if not self.parsed_plan or self._last_parsed_time != self.plan_last_updated:
                self.parsed_plan = self.parse_plan()
                self._last_parsed_time = self.plan_last_updated
                if self.parsed_plan:
                    logger.info(f"Agent {self.id} parsed plan into {len(self.parsed_plan)} steps")

            if self.parsed_plan:
                current_step = self.get_current_step(self.parsed_plan, current_time)
                upcoming_step = self.get_upcoming_step(self.parsed_plan, current_time, window_minutes=15)

                if current_step and current_step.get('location'):
                    target_x, target_y = current_step['location']
                    distance = math.hypot(self.x - target_x, self.y - target_y)

                    if distance > 12.0:
                        logger.debug(f"Agent {self.id} navigating to active step: {current_step['description'][:50]}...")
                        self._set_action(ActionType.NAVIGATE)
                        self.navigate_to(target_x, target_y)
                        self._loiter_target = None
                    else:
                        extra_observations.extend(
                            self._loiter_and_socialize(target_x, target_y, other_agents, current_time)
                        )
                elif upcoming_step and upcoming_step.get('location'):
                    target_x, target_y = upcoming_step['location']
                    distance = math.hypot(self.x - target_x, self.y - target_y)

                    if distance < 10.0:
                        self.vx = 0
                        self.vy = 0
                        self._loiter_target = None
                        self._set_action(ActionType.WAIT)
                        logger.debug(f"Agent {self.id} waiting for upcoming event: {upcoming_step['description'][:50]}...")
                    else:
                        logger.debug(f"Agent {self.id} heading toward upcoming event: {upcoming_step['description'][:50]}...")
                        self._set_action(ActionType.NAVIGATE)
                        self.navigate_to(target_x, target_y)
                else:
                    self._loiter_target = None
                    self._set_action(ActionType.EXPLORE)
                    self._random_walk()
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
            "plan_last_updated": self.plan_last_updated.isoformat() if self.plan_last_updated else None
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
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "goal": self.goal,
            "personality": self.personality,
            "state": self.state,
            "current_plan": self.current_plan,
            "plan_source": self.plan_source,
            "plan_last_updated": self.plan_last_updated.isoformat() if self.plan_last_updated else None
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
        from utils import generate_embedding

        try:
            # Score via LLM (1-10)
            score_raw = await timed_llm_call(
                score_observation,
                signature_name="ScoreImportance",
                timeout=5.0,
                observation=obs,
                agent_goal=self.goal,
                agent_personality=self.personality
            )

            # Normalize to 0-1
            importance = score_raw / 10.0

        except Exception as e:
            logger.warning(f"Agent {self.id} LLM scoring failed: {e}, using default")
            importance = 0.3  # Fallback
            self.state = "confused"

        # Generate embedding
        embedding = generate_embedding(obs)

        # Store memory
        memory_store.store_memory(
            agent_id=self.id,
            content=obs,
            importance=importance,
            embedding=embedding,
            timestamp=datetime.now()
        )

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

    async def update_plan(self, memory_store, current_time: Optional[datetime] = None) -> Optional[str]:
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

        try:
            # Format current time and location
            time_str = current_time.strftime('%I:%M %p')
            location_str = f"({self.x:.0f}, {self.y:.0f})"

            # Generate plan via LLM
            plan = await timed_llm_call(
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

            logger.info(f"Agent {self.id} plan: {plan[:100]}...")

            # Update agent's current plan
            self.current_plan = plan
            self.plan_last_updated = current_time
            self.plan_source = get_planner_source()
            memory_store.update_agent_plan(
                agent_id=self.id,
                plan=self.current_plan,
                plan_source=self.plan_source,
                plan_updated_at=current_time,
            )

            return plan

        except Exception as e:
            logger.error(f"Agent {self.id} planning failed: {e}")
            self.state = "confused"
            self.plan_source = "error"
            return None

    def parse_plan(self) -> List[Dict[str, Any]]:
        """
        Parse the agent's current plan to extract actionable steps.

        Returns:
            List of plan step dicts with keys: start_time, end_time, location, description
        """
        if not self.current_plan:
            return []

        steps = []

        # Regex to match time ranges (e.g., "03:00 PM - 03:30 PM: Description")
        time_pattern = r'(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M):\s*(.*?)(?=\d{1,2}:\d{2}\s*[AP]M|$)'

        # Regex to match location coordinates (e.g., "(200, 150)")
        location_pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'

        try:
            for match in re.finditer(time_pattern, self.current_plan, re.DOTALL):
                start_time_str = match.group(1).strip()
                end_time_str = match.group(2).strip()
                description = match.group(3).strip()

                # Parse times (assume same day, just comparing times)
                try:
                    start_time = datetime.strptime(start_time_str, "%I:%M %p").time()
                    end_time = datetime.strptime(end_time_str, "%I:%M %p").time()
                except ValueError:
                    logger.warning(f"Agent {self.id} could not parse time: {start_time_str} - {end_time_str}")
                    continue

                # Extract location if present
                loc_match = re.search(location_pattern, description)
                location = None
                if loc_match:
                    try:
                        x = float(loc_match.group(1))
                        y = float(loc_match.group(2))
                        location = (x, y)
                    except ValueError:
                        logger.warning(f"Agent {self.id} could not parse location from: {loc_match.group(0)}")

                steps.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'location': location,
                    'description': description
                })

            logger.debug(f"Agent {self.id} parsed {len(steps)} plan steps")
            return steps

        except Exception as e:
            logger.error(f"Agent {self.id} plan parsing failed: {e}")
            return []

    def get_current_step(self, parsed_steps: List[Dict], current_time: datetime) -> Optional[Dict]:
        """
        Find the plan step that should be executed at the current time.

        Args:
            parsed_steps: List of parsed plan steps
            current_time: Current simulation time

        Returns:
            Current step dict or None if no step is active
        """
        if not parsed_steps:
            return None

        current_time_only = current_time.time()

        for step in parsed_steps:
            start = step['start_time']
            end = step['end_time']

            # Handle time comparison (simple: assume same day)
            if start <= current_time_only <= end:
                return step

        return None

    def get_upcoming_step(self, parsed_steps: List[Dict], current_time: datetime, window_minutes: int = 15) -> Optional[Dict]:
        """
        Find the next plan step that starts within the specified time window.

        This is used to allow agents to arrive early at event locations and wait.

        Args:
            parsed_steps: List of parsed plan steps
            current_time: Current simulation time
            window_minutes: Look ahead this many minutes for upcoming steps

        Returns:
            Upcoming step dict or None if no upcoming step within window
        """
        if not parsed_steps:
            return None

        current_time_only = current_time.time()
        future_time = (current_time + timedelta(minutes=window_minutes)).time()

        for step in parsed_steps:
            start = step['start_time']

            # Check if step starts within the window
            # Handle simple case (same day comparison)
            if current_time_only < start <= future_time:
                return step

        return None

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"Agent({self.id}, {self.name}, x={self.x:.1f}, y={self.y:.1f})"
