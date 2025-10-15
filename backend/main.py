"""
FastAPI server for Mini-Town.
Day 0.5: Simulation loop with hardcoded agents + WebSocket broadcasting.
Day 2: LLM-based observation scoring and reflection.
"""

import copy
import json
import os
import asyncio
import logging
import random
import math
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding, get_latency_tracker
from dspy_modules import (
    configure_dspy,
    generate_plan_conversation,
)
from landmarks import list_landmarks, get_landmark
from presets import load_default_plan_presets, plan_preset_catalog

# Reduce tokenizer fork warnings when using huggingface tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "config.yml"
try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}")
    logger.error("Please ensure config.yml exists in the project root")
    raise

USE_TOWN_AGENT_PROGRAM = config.get('compilation', {}).get('use_town_agent_gepa', False)
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yml: {e}")
    raise

# Initialize FastAPI
app = FastAPI(title="Mini-Town API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
agents: List[Agent] = []
memory_store: MemoryStore = None
connected_clients: Set[WebSocket] = set()
simulation_running = False
simulation_paused = False
simulation_step_requested = False
latest_town_score: Optional[float] = None
event_queue: asyncio.Queue = asyncio.Queue()
recent_events: Deque[Dict[str, Any]] = deque(maxlen=10)
event_impacts: Dict[int, int] = {}
EVENT_ALERT_TICKS = 5
current_tick: int = 0
last_tick_timestamp: Optional[datetime] = None

AI_TOWN_WORLD_ID = "mini-town"

PLAN_REFRESH_INTERVAL_TICKS = 120  # roughly every 4 minutes at 2s tick
PLAN_STALE_MINUTES = 15
CONVERSATION_DISTANCE = 28.0

demo_selected_agent_id: Optional[int] = None
SIM_MINUTES_PER_TICK = config['simulation'].get('minutes_per_tick', 5)
simulation_minutes: int = 0

world_map_path = Path(__file__).parent / "data" / "gentle_map.json"
WORLD_MAP_DATA: Optional[Dict[str, Any]] = None
try:
    with open(world_map_path, "r") as f:
        WORLD_MAP_DATA = json.load(f)
except FileNotFoundError:
    logger.warning("AI Town map JSON not found at %s; using minimal fallback.", world_map_path)
    WORLD_MAP_DATA = None


class InjectEventRequest(BaseModel):
    """Payload for /god/inject_event."""

    type: str = Field(..., description="Identifier for the injected event")
    severity: Optional[float] = Field(
        default=None,
        description="Optional severity score between 0 and 1"
    )
    location: Optional[List[float]] = Field(
        default=None,
        description="Optional [x, y] location for the event"
    )


class PauseRequest(BaseModel):
    """Payload for /god/pause."""

    paused: Optional[bool] = Field(
        default=None,
        description="Explicit pause state; toggles if omitted"
    )


class SelectAgentRequest(BaseModel):
    agent_id: int


class ApplyPresetRequest(BaseModel):
    agent_id: int
    preset_id: str


class TeleportRequest(BaseModel):
    agent_id: int
    landmark_id: str


def find_agent(agent_id: int) -> Optional[Agent]:
    """Helper to locate an agent by id."""
    return next((agent for agent in agents if agent.id == agent_id), None)


def get_system_state() -> Dict[str, Any]:
    """Assemble system metrics for frontend consumption."""
    tracker = get_latency_tracker()
    stats = tracker.get_stats()

    latency_values = [
        data.get("mean_ms", 0)
        for data in stats.values()
        if data.get("count", 0) > 0
    ]
    avg_latency = float(sum(latency_values) / len(latency_values)) if latency_values else 0.0

    compilation_cfg = config.get('compilation', {})
    optimizer = "baseline"
    if compilation_cfg.get('use_compiled'):
        optimizer = compilation_cfg.get('optimizer', 'compiled')

    town_score_value = latest_town_score if latest_town_score is not None else 0.0

    return {
        "llm_provider": config['llm']['provider'],
        "llm_model": config['llm']['model'],
        "optimizer": optimizer,
        "town_score": town_score_value,
        "avg_latency": avg_latency,
        "tick_interval": config['simulation']['tick_interval'],
        "paused": simulation_paused,
        "recent_events": list(recent_events),
        "tick": current_tick,
        "last_tick_at": last_tick_timestamp.isoformat() if last_tick_timestamp else None,
        "simulation_minutes": simulation_minutes,
    }


async def broadcast_json(message: Dict[str, Any]) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    if not connected_clients:
        return

    disconnected = set()
    for client in list(connected_clients):
        try:
            await client.send_json(message)
        except Exception as error:
            logger.error(f"Error sending to client: {error}")
            disconnected.add(client)

    for client in disconnected:
        connected_clients.discard(client)


async def broadcast_system_update() -> None:
    """Push current system metrics to subscribers."""
    await broadcast_json({
        "type": "system_update",
        "state": get_system_state()
    })


async def broadcast_agent_snapshot(agent: Agent) -> None:
    """Broadcast a single-agent state update to subscribers."""
    await broadcast_json({
        "type": "agents_update",
        "tick": current_tick,
        "timestamp": datetime.now().isoformat(),
        "agents": [serialize_agent_for_ai(agent)],
        "world": build_ai_town_world_snapshot(),
    })


async def handle_injected_event(event: Dict[str, Any]) -> None:
    """Handle a God Mode injected event."""
    logger.info("Processing injected event: %s", event)

    # Mark agents as alert for a few ticks and store a memory.
    for agent in agents:
        event_impacts[agent.id] = max(event_impacts.get(agent.id, 0), EVENT_ALERT_TICKS)

        description = f"Responded to {event['type']} event"
        embedding = generate_embedding(description)
        memory_store.store_memory(
            agent_id=agent.id,
            content=description,
            importance=0.8,
            embedding=embedding,
            timestamp=datetime.now()
        )

    await broadcast_system_update()


@app.post("/god/refresh_plan")
async def refresh_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger DSPy plan regeneration for one or all agents."""
    agent_id = payload.get("agent_id")

    if agent_id is not None:
        target_agents = [agent for agent in agents if agent.id == agent_id]
        if not target_agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    else:
        target_agents = list(agents)

    refreshed: List[Dict[str, Any]] = []
    for agent in target_agents:
        plan = await agent.update_plan(
            memory_store,
            simulation_minutes=simulation_minutes,
        )
        if plan:
            refreshed.append({
                "agent_id": agent.id,
                "plan": plan,
                "plan_source": agent.plan_source,
                "plan_last_updated": agent.plan_last_updated.isoformat() if agent.plan_last_updated else None,
            })

    # Broadcast updated agent states so UI refreshes immediately.
    await broadcast_json({
        "type": "agents_update",
        "tick": current_tick,
        "agents": [serialize_agent_for_ai(agent) for agent in agents],
        "world": build_ai_town_world_snapshot(),
    })

    return {
        "refreshed": len(refreshed),
        "agents": refreshed,
    }


def serialize_agent(agent: Agent) -> Dict[str, Any]:
    """Return the public fields for an agent."""
    return agent.to_dict()


def serialize_agent_detail(agent: Agent) -> Dict[str, Any]:
    """Return detailed agent info including memories and reflections."""
    agent_data = serialize_agent(agent)

    memories_raw = memory_store.get_agent_memories(agent.id, limit=5)
    memories = [
        {
            "id": mem["id"],
            "ts": mem["ts"].isoformat(),
            "content": mem["content"],
            "importance": mem["importance"]
        }
        for mem in memories_raw
    ]

    latest_reflection_record = memory_store.get_latest_reflection(agent.id)
    latest_reflection = None
    if latest_reflection_record:
        latest_reflection = latest_reflection_record["content"]

    agent_data.update({
        "memories": memories,
        "latest_reflection": latest_reflection
    })
    return agent_data


def serialize_agent_for_ai(agent: Agent) -> Dict[str, Any]:
    """Return agent data tailored for the AI Town UI."""
    return {
        "id": agent.id,
        "agentId": f"a:{agent.id}",
        "name": agent.name,
        "x": agent.x,
        "y": agent.y,
        "goal": agent.goal,
        "personality": agent.personality,
        "state": agent.state,
        "current_plan": agent.current_plan,
        "plan_source": agent.plan_source or "unknown",
        "plan_last_updated": agent.plan_last_updated.isoformat() if agent.plan_last_updated else None,
        "structured_plan": agent.current_plan_structured,
        "plan_validation": (
            {
                "preserved_event_times": agent.last_plan_validation.preserved_event_times,
                "missing_event_times": agent.last_plan_validation.missing_event_times,
                "overlaps_detected": agent.last_plan_validation.overlaps_detected,
                "invalid_locations": agent.last_plan_validation.invalid_locations,
            }
            if agent.last_plan_validation
            else None
        ),
        "observations": list(agent.recent_observations),
        "plan_preset_id": getattr(agent, "plan_preset_id", None),
        "current_step_index": getattr(agent, "current_step_index", 0),
        "active_plan_step": _serialize_plan_step(agent.last_plan_step),
        "simulation_minutes": simulation_minutes,
        "step_explanation": agent.step_explanation,
        "observation_summary": agent.observation_summary,
        "recent_conversations": agent.recent_conversations,
        "use_town_agent": getattr(agent, "use_town_agent_program", False),
        "last_action": getattr(agent, "last_executed_action", None),
        "last_action_reasoning": getattr(agent, "last_action_reasoning", None),
    }


def _serialize_plan_step(step: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not step:
        return None
    location = step.get("location")
    return {
        "start": step.get("start_time_str"),
        "end": step.get("end_time_str"),
        "description": step.get("description"),
        "location": list(location) if location else None,
        "explanation": step.get("explanation"),
    }


def serialize_landmark_payload() -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for landmark in list_landmarks():
        payload.append({
            "id": landmark.id,
            "name": landmark.name,
            "description": landmark.description,
            "x": landmark.x,
            "y": landmark.y,
        })
    return payload


def _agents_are_close(agent_a: Agent, agent_b: Agent, threshold: float = CONVERSATION_DISTANCE) -> bool:
    return math.hypot(agent_a.x - agent_b.x, agent_a.y - agent_b.y) <= threshold


def _format_recent_observations_for_conversation(agent_a: Agent, agent_b: Agent, limit: int = 3) -> str:
    snippets: List[str] = []
    for agent in (agent_a, agent_b):
        for obs in list(agent.recent_observations)[:limit]:
            if isinstance(obs, dict):
                text = obs.get("text")
            else:
                text = None
            if text:
                formatted = f"{agent.name}: {text}"
                if formatted not in snippets:
                    snippets.append(formatted)
    if not snippets:
        return "No recent observations"
    return "\n".join(f"- {item}" for item in snippets[:limit])


async def maybe_trigger_plan_conversations(sim_minutes: int) -> None:
    if not agents:
        return

    step_groups: Dict[Tuple[Any, Any, str], List[Agent]] = defaultdict(list)

    for agent in agents:
        step = getattr(agent, "last_plan_step", None)
        if not step:
            continue
        description = step.get("description")
        if not description:
            continue
        start_minutes = step.get("start_minutes")
        end_minutes = step.get("end_minutes")
        step_groups[(start_minutes, end_minutes, description)].append(agent)

    for key, participants in step_groups.items():
        if len(participants) < 2:
            continue

        for idx in range(len(participants)):
            for jdx in range(idx + 1, len(participants)):
                agent_a = participants[idx]
                agent_b = participants[jdx]

                if not _agents_are_close(agent_a, agent_b):
                    continue

                step_a = agent_a.last_plan_step
                step_b = agent_b.last_plan_step
                if not step_a or not step_b:
                    continue

                signature_a = f"{key}-{agent_b.id}"
                signature_b = f"{key}-{agent_a.id}"
                if signature_a in agent_a.completed_conversation_steps or signature_b in agent_b.completed_conversation_steps:
                    continue

                location_tuple = step_a.get("location") or step_b.get("location")
                location_label = (
                    f"({location_tuple[0]:.0f}, {location_tuple[1]:.0f})"
                    if location_tuple
                    else "the town"
                )

                recent_memories = _format_recent_observations_for_conversation(agent_a, agent_b)

                conversation = await generate_plan_conversation(
                    speaker_name=agent_a.name,
                    speaker_goal=agent_a.goal or "",
                    speaker_personality=agent_a.personality or "",
                    partner_name=agent_b.name,
                    partner_goal=agent_b.goal or "",
                    partner_personality=agent_b.personality or "",
                    location=location_label,
                    step_summary=step_a.get("description", ""),
                    recent_memories=recent_memories,
                )

                dialogue = conversation.dialogue.strip()
                if not dialogue:
                    continue

                lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
                if not lines:
                    continue

                agent_a.completed_conversation_steps.add(signature_a)
                agent_b.completed_conversation_steps.add(signature_b)

                for line in lines:
                    agent_a.queue_observation(f"[CONVERSATION] {line}")
                    agent_b.queue_observation(f"[CONVERSATION] {line}")

                recent_events.appendleft({
                    "id": str(uuid4()),
                    "type": "conversation",
                    "timestamp": datetime.now().isoformat(),
                    "agents": [agent_a.id, agent_b.id],
                    "summary": lines[0][:160],
                })


def build_ai_town_agent_descriptions() -> List[Dict[str, Any]]:
    """Create agent description payload mirroring AI Town semantics."""
    descriptions: List[Dict[str, Any]] = []
    for agent in agents:
        updated_at_iso = agent.plan_last_updated.isoformat() if agent.plan_last_updated else None
        updated_at_ms = int(agent.plan_last_updated.timestamp() * 1000) if agent.plan_last_updated else None
        descriptions.append({
            "agentId": f"a:{agent.id}",
            "id": agent.id,
            "name": agent.name,
            "identity": f"{agent.name} — {agent.personality or 'Mini-Town resident'}",
            "goal": agent.goal,
            "personality": agent.personality,
            "plan": agent.current_plan or "",
            "planSource": agent.plan_source or "unknown",
            "planUpdatedAt": updated_at_ms,
            "planUpdatedAtIso": updated_at_iso,
        })
    return descriptions


def build_ai_town_world_snapshot() -> Dict[str, Any]:
    """Build a lightweight world snapshot for the AI Town UI."""
    players = []
    for agent in agents:
        players.append({
            "playerId": f"p:{agent.id}",
            "agentId": agent.id,
            "name": agent.name,
            "state": agent.state,
            "position": {"x": agent.x, "y": agent.y},
            "goal": agent.goal,
            "personality": agent.personality,
        })

    return {
        "worldId": AI_TOWN_WORLD_ID,
        "tick": current_tick,
        "lastTickAt": last_tick_timestamp.isoformat() if last_tick_timestamp else None,
        "players": players,
        "events": list(recent_events),
    }


def get_world_map_payload() -> Optional[Dict[str, Any]]:
    """Return a copy of the world map payload if available."""
    if WORLD_MAP_DATA is None:
        return None
    payload = copy.deepcopy(WORLD_MAP_DATA)
    tile_url = payload.get("tileSetUrl")
    if isinstance(tile_url, str):
        payload["tileSetUrl"] = tile_url.replace("/ai-town/", "/")
    sprites = payload.get("animatedSprites") or []
    for sprite in sprites:
        sheet = sprite.get("sheet")
        if isinstance(sheet, str):
            sprite["sheet"] = sheet.replace("/ai-town/", "/")
    return payload


# ============ Simulation Logic ============

def initialize_agents():
    """Initialize 3 agents with random positions, or load existing ones."""
    global agents

    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']
    perception_radius = config['simulation']['perception_radius']
    agent_speed = config['simulation'].get('agent_speed', 2.0)

    # Check if agents already exist in database
    existing_agents = memory_store.get_all_agents()

    if existing_agents:
        # Load existing agents from database
        logger.info(f"Loading {len(existing_agents)} existing agents from database")
        agents = []
        for agent_data in existing_agents:
            agent = Agent(
                agent_id=agent_data['id'],
                name=agent_data['name'],
                x=agent_data['x'],
                y=agent_data['y'],
                goal=agent_data['goal'],
                personality=agent_data['personality'],
                map_width=map_width,
                map_height=map_height,
                perception_radius=perception_radius,
                speed=agent_speed,
                use_town_agent_program=USE_TOWN_AGENT_PROGRAM,
            )
            agent.state = agent_data['state']
            agent.current_plan = agent_data['current_plan']
            agent.plan_source = agent_data.get('plan_source') or "unknown"
            agent.speed = agent_speed
            agent.plan_preset_id = None
            plan_updated_at = agent_data.get('plan_updated_at')
            if plan_updated_at:
                if isinstance(plan_updated_at, datetime):
                    agent.plan_last_updated = plan_updated_at
                else:
                    try:
                        agent.plan_last_updated = datetime.fromisoformat(str(plan_updated_at))
                    except Exception:
                        agent.plan_last_updated = datetime.now()
            elif agent.current_plan:
                agent.plan_last_updated = datetime.now()
            agents.append(agent)

        logger.info(f"Loaded {len(agents)} agents from database")
        return

    # Create new agents if none exist
    logger.info("Creating new agents...")

    # Agent names and personalities
    agent_configs = [
        {"name": "Alice", "personality": "social, optimistic"},
        {"name": "Bob", "personality": "analytical, introverted"},
        {"name": "Carol", "personality": "organized, punctual"}
    ]

    agents = []
    for i, agent_config in enumerate(agent_configs):
        x = random.uniform(100, map_width - 100)
        y = random.uniform(100, map_height - 100)

        agent = Agent(
            agent_id=i + 1,
            name=agent_config["name"],
            x=x,
            y=y,
            goal="Explore the town",
            personality=agent_config["personality"],
            map_width=map_width,
            map_height=map_height,
            perception_radius=perception_radius,
            speed=agent_speed,
            use_town_agent_program=USE_TOWN_AGENT_PROGRAM,
        )
        agents.append(agent)

        # Store agent in database
        memory_store.create_agent(
            agent_id=agent.id,
            name=agent.name,
            x=agent.x,
            y=agent.y,
            goal=agent.goal,
            personality=agent.personality
        )

    logger.info(f"Initialized {len(agents)} new agents")


async def simulation_loop():
    """Main simulation loop that updates agents and broadcasts state."""
    global simulation_running, simulation_paused, simulation_step_requested, current_tick, last_tick_timestamp, simulation_minutes

    tick_interval = config['simulation']['tick_interval']
    tick_count = 0
    simulation_minutes = 0

    logger.info("Starting simulation loop...")
    simulation_running = True

    while simulation_running:
        if simulation_paused and not simulation_step_requested:
            await asyncio.sleep(0.1)
            continue

        # Handle any pending God Mode events.
        while True:
            try:
                injected_event = event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await handle_injected_event(injected_event)

        tick_count += 1
        logger.debug(f"Tick {tick_count}")
        tick_timestamp = datetime.now()
        current_tick = tick_count
        last_tick_timestamp = tick_timestamp
        simulation_minutes += SIM_MINUTES_PER_TICK

        # Update each agent
        agent_states = []
        for agent in agents:
            # Track if plan step changed
            old_step = agent.last_plan_step
            
            # Pass other agents for perception
            state = agent.update(agents, simulation_minutes=simulation_minutes, sim_tick=tick_count)
            public_state = serialize_agent_for_ai(agent)
            public_state["observations"] = state.get("observations", [])
            agent_states.append(public_state)

            # Update position in database
            memory_store.update_agent_position(agent.id, agent.x, agent.y)
            
            # Generate step explanation if plan step changed
            if agent.last_plan_step and agent.last_plan_step is not old_step:
                try:
                    await agent.generate_step_explanation(memory_store)
                except Exception as e:
                    logger.debug(f"Agent {agent.id} step explanation failed: {e}")

            # Override agent state if reacting to recent event
            if event_impacts.get(agent.id, 0) > 0:
                agent.state = "alert"
                public_state['state'] = "alert"
                event_impacts[agent.id] -= 1
                if event_impacts[agent.id] <= 0:
                    event_impacts.pop(agent.id, None)

            # Store observations as memories with LLM-based importance scoring
            if state['observations']:
                for obs in state['observations']:
                    # Score and store via LLM
                    importance = await agent.score_and_store_observation(obs, memory_store)
                    logger.debug(f"Agent {agent.id} scored '{obs[:30]}...' = {importance:.2f}")

                # Check for reflection
                insight = await agent.maybe_reflect(memory_store)
                if insight:
                    # Store insight as special memory
                    embedding = generate_embedding(insight)
                    memory_store.store_memory(
                        agent_id=agent.id,
                        content=f"[REFLECTION] {insight}",
                        importance=0.9,  # High importance
                        embedding=embedding,
                        timestamp=tick_timestamp
                    )
                
                # Generate observation summary periodically (every 10 ticks)
                if tick_count % 10 == agent.id % 10:  # Stagger across agents
                    try:
                        await agent.generate_observation_summary()
                    except Exception as e:
                        logger.debug(f"Agent {agent.id} observation summary failed: {e}")

            plan_stale = (
                agent.plan_last_updated is None
                or (tick_timestamp - agent.plan_last_updated).total_seconds() > PLAN_STALE_MINUTES * 60
            )
            ready_for_refresh = tick_count >= getattr(agent, "_next_plan_update_tick", 0)
            stagger_match = PLAN_REFRESH_INTERVAL_TICKS > 0 and (
                tick_count % PLAN_REFRESH_INTERVAL_TICKS
            ) == (agent.id % PLAN_REFRESH_INTERVAL_TICKS)

            if PLAN_REFRESH_INTERVAL_TICKS > 0 and ready_for_refresh and (plan_stale or stagger_match):
                try:
                    new_plan = await agent.update_plan(
                        memory_store,
                        current_time=tick_timestamp,
                        simulation_minutes=simulation_minutes,
                    )
                    if new_plan:
                        public_state["current_plan"] = agent.current_plan
                        public_state["plan_source"] = agent.plan_source
                        public_state["plan_last_updated"] = (
                            agent.plan_last_updated.isoformat() if agent.plan_last_updated else None
                        )
                except Exception as e:
                    logger.warning(f"Agent {agent.id} plan update failed: {e}")
                finally:
                    agent._next_plan_update_tick = tick_count + PLAN_REFRESH_INTERVAL_TICKS

        await maybe_trigger_plan_conversations(simulation_minutes)

        # Broadcast agent and system updates
        await broadcast_json({
            "type": "agents_update",
            "tick": tick_count,
            "timestamp": tick_timestamp.isoformat(),
            "agents": agent_states,
            "world": build_ai_town_world_snapshot()
        })
        await broadcast_system_update()

        if simulation_step_requested:
            simulation_step_requested = False
            simulation_paused = True

        # Wait for next tick
        await asyncio.sleep(tick_interval)


# ============ API Endpoints ============

@app.on_event("startup")
async def startup_event():
    """Initialize database and agents on startup."""
    global memory_store

    logger.info("Starting Mini-Town API...")

    # Configure DSPy with Groq
    logger.info("Configuring DSPy...")
    configure_dspy()

    # Load plan presets for demo controls
    try:
        load_default_plan_presets()
        logger.info("Plan presets loaded: %s", list(plan_preset_catalog.metadata.values()) or "default")
    except Exception as exc:
        logger.warning("Failed to load plan presets: %s", exc)

    # Initialize database (resolve path relative to project root)
    project_root = Path(__file__).parent.parent
    db_path = project_root / config['database']['path']
    memory_store = MemoryStore(str(db_path))

    # Initialize agents
    initialize_agents()

    # Start simulation loop in background
    asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global simulation_running

    logger.info("Shutting down Mini-Town API...")
    simulation_running = False

    if memory_store:
        memory_store.close()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mini-Town API (Day 0.5)",
        "agents": len(agents),
        "simulation_running": simulation_running
    }


@app.get("/agents")
async def get_agents():
    """Get all agents and their current state."""
    return {
        "agents": [serialize_agent(agent) for agent in agents]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: int):
    """Get specific agent details."""
    agent = find_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    memories = memory_store.get_agent_memories(agent_id, limit=10)
    reflection_record = memory_store.get_latest_reflection(agent_id)

    return {
        "agent": serialize_agent(agent),
        "memories": memories,
        "latest_reflection": reflection_record["content"] if reflection_record else None
    }


@app.get("/latency")
async def get_latency_stats():
    """Get LLM latency statistics."""
    tracker = get_latency_tracker()
    stats = tracker.get_stats()

    # Calculate tick viability
    p95_max = max([s.get('p95_ms', 0) for s in stats.values()] + [0])

    decision = "keep_2s"
    if p95_max > 2500:
        decision = "investigate"
    elif p95_max > 1500:
        decision = "adjust_to_3s"

    return {
        "signatures": stats,
        "p95_max_ms": p95_max,
        "tick_decision": decision
    }


@app.get("/api/agents")
async def api_get_agents():
    """Public API: list agents."""
    return {
        "agents": [serialize_agent(agent) for agent in agents]
    }


@app.get("/api/agents/{agent_id}")
async def api_get_agent(agent_id: int):
    """Public API: detailed agent view."""
    agent = find_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return serialize_agent_detail(agent)


@app.get("/api/system")
async def api_get_system_state():
    """Expose current system metrics."""
    return get_system_state()


@app.get("/ai-town/state")
async def ai_town_state():
    """Combined state payload for the AI Town UI."""
    return {
        "world": build_ai_town_world_snapshot(),
        "agents": [serialize_agent_for_ai(agent) for agent in agents],
        "agentDescriptions": build_ai_town_agent_descriptions(),
        "system": get_system_state(),
        "worldMap": get_world_map_payload(),
    }


@app.get("/ai-town/agents")
async def ai_town_agents():
    """Return agents only."""
    return {
        "agents": [serialize_agent_for_ai(agent) for agent in agents],
        "agentDescriptions": build_ai_town_agent_descriptions(),
    }


@app.get("/ai-town/system")
async def ai_town_system():
    """Return system status for AI Town UI consumption."""
    return get_system_state()


@app.get("/ai-town/map")
async def ai_town_map():
    """Expose the static map payload used by the AI Town UI."""
    payload = get_world_map_payload()
    if payload is None:
        raise HTTPException(status_code=503, detail="World map not available")
    return payload


@app.get("/ai-town/world")
async def ai_town_world():
    """Return world snapshot including players and recent events."""
    return build_ai_town_world_snapshot()


@app.get("/ai-town/control/landmarks")
async def ai_town_control_landmarks():
    """Expose landmark metadata for the interactive demo."""
    return {"landmarks": serialize_landmark_payload()}


@app.get("/ai-town/control/presets")
async def ai_town_control_presets():
    """Expose available plan presets by agent."""
    return {
        "metadata": plan_preset_catalog.metadata,
        "presets": plan_preset_catalog.serialize(),
    }


@app.post("/ai-town/control/select_agent")
async def ai_town_control_select_agent(request: SelectAgentRequest):
    """Select a demo agent to focus on in the UI."""
    global demo_selected_agent_id

    agent = find_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    demo_selected_agent_id = agent.id
    return {
        "selected_agent_id": demo_selected_agent_id,
        "agent": serialize_agent_for_ai(agent),
    }


@app.post("/ai-town/control/apply_plan")
async def ai_town_control_apply_plan(request: ApplyPresetRequest):
    """Apply a curated preset plan to an agent."""
    agent = find_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    preset = plan_preset_catalog.preset_for_agent(agent.id, request.preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found for agent")

    now = datetime.now()
    agent.current_plan = preset.plan
    agent.plan_last_updated = now
    agent.plan_source = "preset"
    agent.plan_preset_id = preset.id
    agent.parsed_plan = []
    agent._last_parsed_time = None
    agent._next_plan_update_tick = current_tick + PLAN_REFRESH_INTERVAL_TICKS
    agent.parsed_plan = agent.parse_plan()
    agent._last_parsed_time = agent.plan_last_updated
    agent.current_step_index = 0
    agent.last_plan_step = None
    agent.last_plan_step_switch_tick = None
    agent.align_plan_execution(simulation_minutes)
    agent.completed_conversation_steps.clear()

    memory_store.update_agent_plan(
        agent_id=agent.id,
        plan=agent.current_plan,
        plan_source=agent.plan_source,
        plan_updated_at=now,
    )

    await broadcast_agent_snapshot(agent)
    await broadcast_system_update()

    return {
        "status": "applied",
        "agent": serialize_agent_for_ai(agent),
        "preset": {
            "id": preset.id,
            "label": preset.label,
            "summary": preset.summary,
            "landmark_id": preset.landmark_id,
        },
    }


@app.post("/ai-town/control/teleport")
async def ai_town_control_teleport(request: TeleportRequest):
    """Teleport an agent to a named landmark."""
    agent = find_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    landmark = get_landmark(request.landmark_id)
    if not landmark:
        raise HTTPException(status_code=404, detail="Landmark not found")

    agent.x = landmark.x
    agent.y = landmark.y
    agent.vx = 0.0
    agent.vy = 0.0
    agent._loiter_target = None
    memory_store.update_agent_position(agent.id, agent.x, agent.y)

    await broadcast_agent_snapshot(agent)

    return {
        "status": "teleported",
        "agent": serialize_agent_for_ai(agent),
        "landmark": {
            "id": landmark.id,
            "name": landmark.name,
        },
    }


@app.post("/god/pause")
async def god_pause(payload: Optional[PauseRequest] = None):
    """Toggle or explicitly set the simulation pause state."""
    global simulation_paused

    desired_state = payload.paused if payload and payload.paused is not None else not simulation_paused
    simulation_paused = desired_state

    status = "paused" if simulation_paused else "resumed"
    await broadcast_system_update()
    return {"status": status}


@app.post("/god/step")
async def god_step():
    """Advance the simulation by a single tick."""
    global simulation_step_requested

    simulation_step_requested = True
    await broadcast_system_update()
    return {"status": "stepped"}


@app.post("/god/inject_event")
async def god_inject_event(payload: InjectEventRequest):
    """Inject an event into the simulation (stub implementation)."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Simulation not ready")

    logger.info(
        "Injecting event via God Mode: type=%s severity=%s location=%s",
        payload.type,
        payload.severity,
        payload.location
    )

    event = {
        "id": str(uuid4()),
        "type": payload.type,
        "severity": payload.severity,
        "location": payload.location,
        "timestamp": datetime.now().isoformat()
    }

    recent_events.appendleft(event)
    await event_queue.put(event)
    await broadcast_json({
        "type": "event",
        "event": event,
        "world": build_ai_town_world_snapshot(),
    })

    return {"status": "injected", "event": event}


@app.post("/ai-town/god/pause")
async def ai_town_god_pause(payload: Optional[PauseRequest] = None):
    """AI Town alias for pause endpoint."""
    return await god_pause(payload)


@app.post("/ai-town/god/step")
async def ai_town_god_step():
    """AI Town alias for step endpoint."""
    return await god_step()


@app.post("/ai-town/god/inject_event")
async def ai_town_god_inject_event(payload: InjectEventRequest):
    """AI Town alias for inject event endpoint."""
    return await god_inject_event(payload)


@app.post("/ai-town/god/refresh_plan")
async def ai_town_god_refresh_plan(payload: Dict[str, Any]):
    """AI Town alias for refresh plan endpoint."""
    return await refresh_plan(payload)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates."""
    await websocket.accept()
    connected_clients.add(websocket)

    logger.info(f"Client connected. Total clients: {len(connected_clients)}")

    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "agents": [serialize_agent_for_ai(agent) for agent in agents],
            "system": get_system_state(),
            "world": build_ai_town_world_snapshot(),
            "worldMap": get_world_map_payload(),
            "config": {
                "map_width": config['simulation']['map_width'],
                "map_height": config['simulation']['map_height']
            }
        })

        # Keep connection alive
        while True:
            # Wait for any messages from client (ping/pong)
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)


# ============ Entry Point ============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
