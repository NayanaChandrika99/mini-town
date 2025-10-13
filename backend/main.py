"""
FastAPI server for Mini-Town.
Day 0.5: Simulation loop with hardcoded agents + WebSocket broadcasting.
Day 2: LLM-based observation scoring and reflection.
"""

import os
import asyncio
import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding, get_latency_tracker
from dspy_modules import configure_dspy

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
        "recent_events": list(recent_events)
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
        plan = await agent.update_plan(memory_store)
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
        "agents": [serialize_agent(agent) for agent in agents],
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


# ============ Simulation Logic ============

def initialize_agents():
    """Initialize 3 agents with random positions, or load existing ones."""
    global agents

    map_width = config['simulation']['map_width']
    map_height = config['simulation']['map_height']
    perception_radius = config['simulation']['perception_radius']

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
                perception_radius=perception_radius
            )
            agent.state = agent_data['state']
            agent.current_plan = agent_data['current_plan']
            agent.plan_source = agent_data.get('plan_source') or "unknown"
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
            perception_radius=perception_radius
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
    global simulation_running, simulation_paused, simulation_step_requested

    tick_interval = config['simulation']['tick_interval']
    tick_count = 0

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

        # Update each agent
        agent_states = []
        for agent in agents:
            # Pass other agents for perception
            state = agent.update(agents)
            agent_states.append(state)

            # Update position in database
            memory_store.update_agent_position(agent.id, agent.x, agent.y)

            # Override agent state if reacting to recent event
            if event_impacts.get(agent.id, 0) > 0:
                agent.state = "alert"
                state['state'] = "alert"
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

        # Broadcast agent and system updates
        await broadcast_json({
            "type": "agents_update",
            "tick": tick_count,
            "timestamp": tick_timestamp.isoformat(),
            "agents": agent_states
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
    await broadcast_json({"type": "event", "event": event})

    return {"status": "injected", "event": event}


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
            "agents": [serialize_agent(agent) for agent in agents],
            "system": get_system_state(),
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
