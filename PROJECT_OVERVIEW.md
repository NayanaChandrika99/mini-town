# Mini-Town Project Overview

## 1. High-Level Summary

Mini-Town is an agent-based simulation where autonomous characters, powered by Large Language Models (LLMs), live, perceive, and interact within a 2D world. The project's core is a sophisticated cognitive architecture built with the **DSPy** framework, which allows agents to reason about their observations, form memories, reflect on their experiences, and generate plans to achieve their goals.

The primary goal of this project is to serve as a testbed and demonstration platform for DSPy's optimization capabilities, showing how compiled and optimized LLM programs can lead to more consistent, efficient, and intelligent agent behavior compared to uncompiled baseline models.

## 2. Core Technologies

- **Backend:** **FastAPI** (Python) for the main server, simulation loop, and API.
- **Frontend:** **Next.js** / **React** (TypeScript) with **TailwindCSS** for the user interface.
- **Real-time Communication:** **WebSockets** for broadcasting simulation state from the backend to the frontend.
- **AI / Cognition:** **DSPy** for structuring LLM interactions and enabling optimization.
- **Database:** **DuckDB** with the `vss` extension for efficient storage and vector-based semantic search of agent memories.
- **LLM Providers:** Flexible integration with providers like Groq, Together, and OpenAI, configured via `config.yml`.

## 3. Architecture

The application follows a standard client-server model:

- **Backend Server (`backend/`):** A single FastAPI application that is the heart of the project. It is responsible for:
    1.  Running the main `simulation_loop`.
    2.  Managing agent state and orchestrating their cognitive functions.
    3.  Interacting with the DuckDB database (`MemoryStore`).
    4.  Serving an HTTP API for querying simulation state.
    5.  Broadcasting real-time updates to all connected clients via WebSockets.

- **Frontend Client (`frontend/`):** A Next.js web application that:
    1.  Connects to the backend's WebSocket endpoint (`/ws`).
    2.  Receives real-time state updates (agent positions, plans, etc.).
    3.  Renders the 2D town, animations, and agents.
    4.  Provides a user interface for observing the simulation and (soon) interacting with it in a controlled "Demo Mode".

---

## 4. Key Backend Components (`backend/`)

This directory contains the core logic of the simulation.

### `main.py`: The Conductor

This is the main entry point for the backend. Its key responsibilities are:
- **FastAPI App:** Sets up the web server and all API endpoints.
- **Initialization:** On startup, it configures DSPy, initializes the `MemoryStore`, loads or creates agents, and starts the main simulation loop as a background task.
- **`simulation_loop()`:** The "heartbeat" of the town. On each "tick," this loop iterates through every agent and orchestrates their cognitive cycle. **Crucially, this loop is responsible for calling the agent's planning and reflection methods periodically.**
- **API Endpoints:** Defines all HTTP and WebSocket endpoints used by the frontend (e.g., `/ws`, `/api/agents`, `/god/refresh_plan`).

### `agents.py`: The Agent's "Body" and State

This file defines the `Agent` class, which represents a single character in the simulation.
- **State Management:** An `Agent` instance holds its current state, including its ID, name, personality, goal, position (`x`, `y`), current plan, and recent observations.
- **Action/Movement (`update()`):** This primary method is called by the `simulation_loop` on every tick. It handles the agent's physical actions based on its current plan, such as navigating to a target, loitering, or performing a default random walk if it has no plan.
- **Perception (`_perceive()`):** Allows the agent to see other agents or objects in its immediate vicinity.
- **Cognitive Triggers:** Contains the methods that initiate the agent's "thinking" processes, which in turn call the DSPy modules. These are:
    - `score_and_store_observation()`: Scores an observation's importance.
    - `maybe_reflect()`: Triggers reflection if an importance threshold is met.
    - `update_plan()`: Generates a new daily plan.

### `dspy_modules.py`: The Agent's "Brain"

This is the core of the agent's intelligence, defining its cognitive architecture using DSPy.
- **`Signatures`:** These classes (`ScoreImportance`, `Reflect`, `PlanDay`) are essentially strongly-typed, structured prompts. They define the inputs (e.g., `observation`, `agent_goal`) and outputs (e.g., `score`, `insight`, `plan`) for each cognitive task.
- **DSPy Modules (`scorer`, `reflector`, `planner`):** These are the actual DSPy programs. They are initialized as simple `dspy.ChainOfThought` or `dspy.Predict` modules, which wrap the signatures.
- **Compiled Module Management:** This file includes logic (`load_compiled_modules`, `use_compiled`) to load optimized DSPy programs from the `/compiled` directory. This allows the application to switch between the baseline (uncompiled) and optimized "brains" to compare performance and quality.
- **Helper Functions (`score_observation`, `generate_reflection`, `generate_plan`):** These are the `async` functions called by `agents.py` to execute the DSPy modules and get results from the LLM.

### `memory.py`: The Agent's Long-Term Memory

This file defines the `MemoryStore` class, which manages all database interactions.
- **Database Connection:** It handles the connection to the DuckDB database file (`data/town.db`).
- **Schema Definition:** It creates the `agents` and `memories` tables. The `memories` table is critically important and includes columns for `content`, `importance`, and `embedding`.
- **Vector Search:** It uses the `vss` extension to create an `HNSW` index on the `embedding` column. This enables **semantic search**.
- **`retrieve_memories_by_vector()`:** This is the most important method. Instead of just fetching recent memories, it retrieves memories based on a combination of **relevance** (cosine similarity to a query vector), **recency**, and **importance**. This allows an agent to recall memories that are semantically related to its current situation, which is fundamental for reflection and planning.

---

## 5. The Cognitive Loop: How It All Works Together

The interaction between these components forms the agent's "life" cycle:

1.  **Perceive & Act (`main.py` -> `agents.py`):** The `simulation_loop` calls `agent.update()`. The agent moves according to its current plan and perceives its surroundings, generating observations (e.g., "Alice is nearby").
2.  **Score & Store (`main.py` -> `agents.py` -> `dspy_modules.py` -> `memory.py`):** The `simulation_loop` takes the new observations and calls `agent.score_and_store_observation()`. This uses the `ScoreImportance` signature to ask the LLM for an importance score. The observation, score, and a vector embedding are then saved to the database via `memory.py`.
3.  **Reflect (`main.py` -> `agents.py` -> `memory.py` -> `dspy_modules.py`):** When an agent's accumulated importance score crosses a threshold, the `simulation_loop` calls `agent.maybe_reflect()`. The agent retrieves relevant memories from `memory.py` (using vector search) and feeds them into the `Reflect` signature in `dspy_modules.py` to generate a high-level insight. This insight is then stored as a new, important memory.
4.  **Plan (`main.py` -> `agents.py` -> `memory.py` -> `dspy_modules.py`):** Periodically, the `simulation_loop` calls `agent.update_plan()`. The agent retrieves relevant memories (especially recent events and invitations) and uses the `PlanDay` signature to generate a new, time-blocked plan for the day.
5.  **Execute (`agents.py`):** With its `current_plan` updated, the agent's `update()` method now has a schedule to follow. It parses the plan and navigates to the specified locations at the specified times, breaking the cycle of random wandering.

This entire process enables the agents to exhibit intelligent, goal-driven behavior that evolves based on their experiences.
