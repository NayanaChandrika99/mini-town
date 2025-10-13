"""
FastAPI server for Mini-Town.
Day 0.5: Simulation loop with hardcoded agents + WebSocket broadcasting.
Day 2: LLM-based observation scoring and reflection.
"""

import asyncio
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import random

from agents import Agent
from memory import MemoryStore
from utils import generate_embedding, get_latency_tracker
from dspy_modules import configure_dspy

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
    global simulation_running

    tick_interval = config['simulation']['tick_interval']
    tick_count = 0

    logger.info("Starting simulation loop...")
    simulation_running = True

    while simulation_running:
        tick_count += 1
        logger.debug(f"Tick {tick_count}")

        # Update each agent
        agent_states = []
        for agent in agents:
            # Pass other agents for perception
            state = agent.update(agents)
            agent_states.append(state)

            # Update position in database
            memory_store.update_agent_position(agent.id, agent.x, agent.y)

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
                        timestamp=datetime.now()
                    )

        # Broadcast state to all connected clients
        await broadcast_state({
            "tick": tick_count,
            "timestamp": datetime.now().isoformat(),
            "agents": agent_states
        })

        # Wait for next tick
        await asyncio.sleep(tick_interval)


async def broadcast_state(state: dict):
    """Broadcast simulation state to all connected WebSocket clients."""
    if not connected_clients:
        return

    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(state)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(client)

    # Remove disconnected clients
    for client in disconnected:
        connected_clients.discard(client)


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
        "agents": [agent.to_dict() for agent in agents]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: int):
    """Get specific agent details."""
    agent = next((a for a in agents if a.id == agent_id), None)
    if not agent:
        return {"error": "Agent not found"}, 404

    # Get recent memories
    memories = memory_store.get_agent_memories(agent_id, limit=10)

    return {
        "agent": agent.to_dict(),
        "memories": memories
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
            "agents": [agent.to_dict() for agent in agents],
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
