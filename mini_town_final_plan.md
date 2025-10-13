# Compiled Generative-Agents Mini-Town (DSPy Edition)
## Budget: $5 | Timeline: 7-10 Days | Solo Developer

---

## One-Liner

A small, visually rich "mini-town" of **5 autonomous NPCs** that perceive → reflect → plan → act—with every sub-skill defined as a typed DSPy program and compiled (starting with **MIPROv2 or GEPA**) against measurable goals like event attendance, plan fidelity, and memory quality. Hybrid Colab Pro + strategic cloud GPU architecture keeps costs near-zero.

---

## Why This Matters (2025 Context)

- **Builds on "Smallville" generative agents** (Park et al., 2023) but treats the agent brain as a **compiled program** with modern optimizers
- **Uses Colab Pro's free GPU** for compilation + development, strategic cloud APIs (Groq free tier, Together.ai) for live demos
- **DuckDB + vss extension** replaces heavy database setup with a single-file vector store
- **Start with 5 agents, one optimizer (MIPROv2 or GEPA)**, prove the concept, then scale

---

## System Architecture (Hybrid Cloud)

```
┌────────────────────────────────────────────────────┐
│  Frontend (Next.js on Vercel free tier)            │
│  - Top-down map (5 agents)                         │
│  - Click agent → see memories, current plan        │
│  - Event timeline + system status panel            │
│  - God mode: inject events, pause/step sim         │
└────────────────────────────────────────────────────┘
                        ↓ WebSocket
┌────────────────────────────────────────────────────┐
│  Backend (FastAPI, local + ngrok for demos)        │
│                                                    │
│  Simulation Loop (every 2 seconds):                │
│    1. Perceive (events + nearby agents)            │
│    2. ScoreImportance (DSPy Predict)               │
│    3. Retrieve top-10 memories (DuckDB vector)     │
│    4. Reflect IF threshold (ChainOfThought)        │
│    5. Plan (Predict or lightweight ReAct)          │
│    6. Act (move, speak, invite)                    │
│    7. Error handling (timeout → "confused" state)  │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  Storage (DuckDB: single file town.db)             │
│  - agents: id, name, x, y, goal, personality       │
│  - memories: id, agent_id, text, importance, embed │
│  - notes: id, agent_id, insight, embed, links      │
│  - vss indexes for fast ANN retrieval              │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  LLM (Hybrid)                                      │
│  - Development: Groq Llama-3.2-3B (free tier)      │
│  - Compilation: Colab Pro T4 GPU (free)            │
│  - Live Demo: Together.ai Qwen2.5-7B (~$0.50)     │
└────────────────────────────────────────────────────┘
```

---

## Tech Stack (Final)

| Component | Choice | Why | Cost |
|-----------|--------|-----|------|
| **LLM (dev)** | Groq Llama-3.2-3B (30 req/min free) | Fast, zero cost, good for iteration | $0 |
| **LLM (demo)** | Together.ai Qwen2.5-7B | $0.20/1M tokens, reliable | ~$0.50 |
| **Compilation** | Colab Pro (T4 GPU) | Free compute for overnight runs | $0 |
| **Embeddings** | Colab: sentence-transformers/all-MiniLM-L6-v2 (384-dim) | Fast, local, good quality | $0 |
| **Vector DB** | DuckDB + vss extension | Single file, no server, HNSW support | $0 |
| **Backend** | FastAPI (Python) | Lightweight, async, easy WebSockets | $0 |
| **Frontend** | Next.js (Vercel free tier) | React, SSR, free hosting | $0 |
| **Observability** | Python logging + JSON files (start simple) | grep/jq, no external deps | $0 |
| **DSPy** | Latest from GitHub | Typed signatures, MIPROv2/GEPA | $0 |

**Total Estimated Spend**: $0.10 - $2.00 (leaves $3-4.90 for contingency)

---

## Starting Scope (Critical Constraints)

### Agent Count: 5 (not 10-20)
- **Why**: Easier to debug, faster sim loop, less token usage
- **Scale later**: Once architecture works, adding agents is trivial

### Optimizer: Start with ONE (GEPA primary, MIPROv2 fallback)
- **GEPA (2025)** ⭐ **PRIMARY CHOICE**: Reflective prompt evolution, reportedly faster convergence with fewer rollouts than MIPROv2 and GRPO. Better for budget constraints since it needs fewer training examples (~40 rollouts vs 50-100+).
- **MIPROv2** (fallback): Joint optimization of instructions + few-shot examples. Proven, well-documented. Use if GEPA API issues or convergence problems.

**Why GEPA first?**
✅ **More efficient**: Needs ~40 rollouts vs MIPROv2's 50-100+
✅ **Budget-friendly**: Less Colab time per compilation cycle
✅ **Cutting-edge**: 2025 technique with reflective prompt evolution
⚠️ **Trade-off**: Less community examples than MIPROv2 (but APIs are similar)

**Future optimizers** (noted for later expansion):
- **SIMBA**: Harden reflection on difficult mini-batches
- **GRPO**: RL-based optimization for end-to-end reward
- Add a `--optimizer` CLI flag once MIPROv2/GEPA works

### Compilation Budget: 2-3 Cycles
- First compile will likely need adjustment (metric definition, seed quality)
- Budget 3-4 hours of Colab time per cycle
- Total: ~10-12 hours across the project (well within Colab Pro limits)

### ReAct Complexity: Start Simple
- **Day 1-5**: Use `dspy.Predict(PlanDay)` for simple text plans
- **Day 6+**: Optionally add lightweight ReAct with **max_iters=3** (not 8)
- **Fast path**: If plan is "continue current activity", skip LLM call entirely

---

## DSPy Integration (Typed Signatures)

### Core Signatures (Start with 3)

```python
import dspy

class ScoreImportance(dspy.Signature):
    """Rate how important this observation is for the agent's goals."""
    observation: str = dspy.InputField()
    agent_goal: str = dspy.InputField(desc="Agent's current high-level goal")
    score: int = dspy.OutputField(desc="1=trivial, 10=life-changing")

class Reflect(dspy.Signature):
    """Synthesize a high-level insight from recent memories."""
    memories: list[str] = dspy.InputField(desc="Recent important memories")
    agent_personality: str = dspy.InputField(desc="Agent's traits")
    insight: str = dspy.OutputField(desc="Abstract pattern or realization")

class PlanDay(dspy.Signature):
    """Create a simple daily plan given goal and context."""
    goal: str = dspy.InputField()
    current_time: str = dspy.InputField()
    recent_events: list[str] = dspy.InputField()
    plan: str = dspy.OutputField(desc="Time-blocked plan as simple text")
```

### Module Implementations

```python
# Simple predictor for scoring
scorer = dspy.Predict(ScoreImportance)

# Chain-of-thought for reflection
reflector = dspy.ChainOfThought(Reflect)

# Start simple, add ReAct later
planner = dspy.Predict(PlanDay)

# Future: Lightweight ReAct (max_iters=3)
# planner = dspy.ReAct(
#     PlanDay,
#     tools=[now, distance, calendar_add],
#     max_iters=3
# )
```

---

## Data Model (DuckDB + vss)

### Setup Script

```sql
-- Install vss extension (do once)
INSTALL vss;
LOAD vss;

-- Agents table
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    x REAL DEFAULT 0,
    y REAL DEFAULT 0,
    goal TEXT,
    personality TEXT,  -- e.g., "social, risk-averse" or "reclusive, curious"
    current_plan TEXT,
    state TEXT DEFAULT 'active'  -- active | confused | idle
);

-- Memories with embeddings (384-dim for all-MiniLM-L6-v2)
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    ts TIMESTAMP NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
    keywords TEXT[],
    embedding FLOAT[384]
);

-- Notes (high-level insights)
CREATE TABLE notes (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    summary TEXT NOT NULL,
    tags TEXT[],
    links JSON,  -- [{to: note_id, rel: "caused_by"}]
    embedding FLOAT[384]
);

-- Vector indexes (HNSW for fast ANN)
CREATE INDEX mem_vec_idx ON memories USING HNSW (embedding);
CREATE INDEX note_vec_idx ON notes USING HNSW (embedding);

-- Regular indexes for filtering
CREATE INDEX mem_agent_ts ON memories(agent_id, ts DESC);
CREATE INDEX mem_importance ON memories(importance DESC);
```

### Retrieval Score (Tunable Triad)

```python
def compute_retrieval_score(
    memory,
    query_embedding,
    current_time,
    alpha=0.5,   # relevance weight (tune per scenario)
    beta=0.3,    # recency weight
    gamma=0.2,   # importance weight
    agent_id=None  # for agent-specific weights
):
    """
    Smallville's retrieval triad with scenario/agent-specific tuning.
    
    Scenario examples:
    - Social gathering: α=0.3, β=0.5, γ=0.2 (prioritize recent)
    - Long-term planning: α=0.4, β=0.1, γ=0.5 (prioritize important)
    - Crisis response: α=0.6, β=0.3, γ=0.1 (prioritize relevant)
    """
    relevance = cosine_similarity(query_embedding, memory.embedding)
    hours_ago = (current_time - memory.ts).total_seconds() / 3600
    recency = np.exp(-0.1 * hours_ago)  # decay factor
    importance = memory.importance
    
    return alpha * relevance + beta * recency + gamma * importance
```

**TODO**: Add agent-specific weights table for personality-based retrieval differences.

---

## Simulation Loop (Backend)

### Main Tick Function

```python
import asyncio
from fastapi import FastAPI, WebSocket
import dspy

app = FastAPI()

# Global state
agents = {}
tool_cache = {}  # Cache tool results per tick

async def simulation_tick():
    """Run every 2 seconds."""
    current_time = datetime.now()
    tool_cache.clear()  # Fresh cache each tick
    
    for agent_id, agent in agents.items():
        try:
            # 1. Perceive
            observations = perceive_environment(agent_id)
            
            # 2. Score importance (with timeout)
            scored_obs = []
            for obs in observations:
                try:
                    result = await asyncio.wait_for(
                        scorer(observation=obs, agent_goal=agent.goal),
                        timeout=5.0
                    )
                    scored_obs.append((obs, result.score / 10.0))
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {agent_id} scoring timeout")
                    scored_obs.append((obs, 0.5))  # default mid-importance
            
            # 3. Store memories
            for obs, importance in scored_obs:
                embedding = embed_text(obs)  # cached embeddings
                store_memory(agent_id, obs, importance, embedding, current_time)
            
            # 4. Reflect (if threshold met)
            recent_importance_sum = get_recent_importance_sum(agent_id, window_hours=4)
            if recent_importance_sum >= agent.reflect_threshold:
                memories = retrieve_memories(agent_id, query="recent events", top_k=10)
                insight = reflector(
                    memories=[m.content for m in memories],
                    agent_personality=agent.personality
                ).insight
                store_note(agent_id, insight, embedding=embed_text(insight))
                logger.info(f"Agent {agent_id} reflected: {insight[:50]}...")
            
            # 5. Plan (with fast path)
            if should_replan(agent):
                recent_events = [m.content for m in retrieve_memories(agent_id, top_k=5)]
                plan = planner(
                    goal=agent.goal,
                    current_time=str(current_time),
                    recent_events=recent_events
                ).plan
                agent.current_plan = plan
            
            # 6. Act (execute plan)
            action = parse_plan_to_action(agent.current_plan, current_time)
            execute_action(agent_id, action)
            
            agent.state = "active"
            
        except Exception as e:
            logger.error(f"Agent {agent_id} error: {e}")
            agent.state = "confused"  # Visible in UI
            agent.current_plan = "Trying to recover..."

async def simulation_loop():
    while True:
        await simulation_tick()
        await asyncio.sleep(2.0)

@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())
```

### Tool Validation & Caching

```python
from pydantic import BaseModel, validator
from functools import lru_cache

class CalendarAddTool(BaseModel):
    """Typed tool call for adding calendar events."""
    event_name: str
    time: str  # ISO format
    attendees: list[str]
    
    @validator('time')
    def validate_time(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Time must be ISO format")

@lru_cache(maxsize=100)
def cached_now():
    """Tool calls cached per tick via tool_cache."""
    return datetime.now().isoformat()

def safe_tool_call(tool_name, **kwargs):
    """Graceful failure wrapper."""
    try:
        if tool_name == "now":
            return tool_cache.get("now", cached_now())
        elif tool_name == "calendar.add":
            event = CalendarAddTool(**kwargs)
            return add_to_calendar(event)
        # ... other tools
    except Exception as e:
        logger.warning(f"Tool call failed: {tool_name}, {e}")
        return {"error": str(e)}
```

---

## Compilation Workflow (Colab Pro)

### Colab Notebook Structure

```python
# Cell 1: Setup
!pip install dspy-ai sentence-transformers

from google.colab import drive
drive.mount('/content/drive')

import dspy
from dspy.optimizers import GEPA  # PRIMARY: Use GEPA first
# from dspy.teleprompt import MIPROv2  # FALLBACK: Uncomment if GEPA issues

# Cell 2: Load seeds from Drive
import json
with open('/content/drive/MyDrive/mini-town/seeds_scorer.json') as f:
    seeds = json.load(f)

trainset = [
    dspy.Example(
        observation=s['observation'],
        agent_goal=s['agent_goal'],
        score=s['score']
    ).with_inputs('observation', 'agent_goal')
    for s in seeds
]

# Cell 3: Define metric (tune over 2-3 cycles)
def importance_metric(pred, example, trace=None):
    """Allow ±1 point error on 1-10 scale."""
    error = abs(pred.score - example.score)
    return error <= 1

# Cell 4: Compile (GEPA primary, MIPROv2 fallback)

# PRIMARY: GEPA (2025 - more efficient, fewer rollouts)
compiled_scorer = GEPA(
    metric=importance_metric,
    budget=40,  # fewer rollouts than MIPROv2 (40 vs 50-100+)
    verbose=True
).compile(
    dspy.Predict(ScoreImportance),
    trainset=trainset
)

# FALLBACK: MIPROv2 (use if GEPA has issues)
# compiled_scorer = MIPROv2(
#     metric=importance_metric,
#     auto="medium",  # light | medium | heavy
#     max_bootstrapped_demos=4,
#     max_labeled_demos=5,
#     num_trials=10,
#     verbose=True
# ).compile(
#     dspy.Predict(ScoreImportance),
#     trainset=trainset
# )

# Cell 5: Save compiled program
compiled_scorer.save('/content/drive/MyDrive/mini-town/compiled_scorer.json')

# Also export just the optimized prompt for inspection
with open('/content/drive/MyDrive/mini-town/prompt_scorer.txt', 'w') as f:
    f.write(str(compiled_scorer))
```

### Iteration Budget (Realistic - Using GEPA)

| Cycle | Goal | Colab Time (GEPA) | Estimated Cost |
|-------|------|-------------------|----------------|
| **1** | Baseline compile with 30-40 seeds | 2-3 hours | $0 (Colab) |
| **2** | Refine metric, add 10 more seeds | 2-3 hours | $0 (Colab) |
| **3** | Final compile with best config | 2-3 hours | $0 (Colab) |

**Total**: ~6-9 hours of Colab Pro usage with GEPA (vs 8-10 with MIPROv2)

**GEPA Efficiency Gain**: ~30-40% faster compilation due to fewer rollouts (40 vs 50-100+)

---

## Evaluation Metrics (Measured)

### 1. Event Coherence
```python
def event_coherence_metric(scenario_result):
    """Did invited agents attend the event?"""
    event_time = scenario_result['event']['time']
    invitees = scenario_result['event']['invitees']
    attendees = scenario_result['attendees']
    
    window = timedelta(minutes=10)
    on_time = [
        a for a in attendees
        if abs(a['arrival_time'] - event_time) <= window
    ]
    
    return len(on_time) / len(invitees)
```

### 2. Plan Fidelity
```python
def plan_fidelity_metric(agent_id, duration_minutes=30):
    """How well did agent follow their plan?"""
    planned_timeline = get_planned_timeline(agent_id)
    executed_timeline = get_executed_timeline(agent_id, duration_minutes)
    
    # Normalized edit distance
    from Levenshtein import distance
    norm_dist = distance(planned_timeline, executed_timeline) / max(
        len(planned_timeline), len(executed_timeline)
    )
    
    return 1 - norm_dist
```

### 3. Memory Hit-Rate
```python
def memory_hit_rate(agent_id, test_queries):
    """Can agent retrieve correct info from note graph?"""
    correct = 0
    for query, expected_note_id in test_queries:
        results = retrieve_notes(agent_id, query, top_k=5)
        if expected_note_id in [r.id for r in results]:
            correct += 1
    
    return correct / len(test_queries)
```

### 4. Combined Town Score
```python
def town_score(scenario_results, weights=(0.4, 0.3, 0.3)):
    """Unified metric for compilation."""
    w_event, w_plan, w_memory = weights
    
    return (
        w_event * event_coherence_metric(scenario_results) +
        w_plan * np.mean([plan_fidelity_metric(a) for a in agents]) +
        w_memory * np.mean([memory_hit_rate(a, queries) for a in agents])
    )
```

---

## Missing Pieces (Implementation Notes)

### 1. Failure Handling
```python
class AgentState(Enum):
    ACTIVE = "active"
    CONFUSED = "confused"  # LLM timeout/parse error
    IDLE = "idle"
    RECOVERING = "recovering"

# In simulation loop
try:
    result = await asyncio.wait_for(llm_call(), timeout=5.0)
except asyncio.TimeoutError:
    agent.state = AgentState.CONFUSED
    agent.current_plan = "Lost in thought..."
    # Try again next tick with cached response or default
```

### 2. Simulation Control (God Mode)
```python
@app.post("/god/inject_event")
async def inject_event(event: dict):
    """Manually trigger an event for testing."""
    # {"type": "fire_alarm", "location": [5, 5], "severity": 8}
    broadcast_to_nearby_agents(event)
    return {"status": "injected"}

@app.post("/god/pause")
async def pause_sim():
    global sim_paused
    sim_paused = True

@app.post("/god/step")
async def step_sim():
    """Single tick for debugging."""
    await simulation_tick()

@app.get("/god/agent/{agent_id}/memories")
async def get_agent_memories(agent_id: int, limit: int = 50):
    """Inspect agent's memory for debugging."""
    return fetch_memories(agent_id, limit)
```

### 3. Agent Diversity (Personality System)
```python
PERSONALITIES = {
    "alice": {"traits": "social, optimistic", "alpha": 0.4, "beta": 0.4, "gamma": 0.2},
    "bob": {"traits": "reclusive, analytical", "alpha": 0.6, "beta": 0.1, "gamma": 0.3},
    "carol": {"traits": "risk-averse, organized", "alpha": 0.3, "beta": 0.2, "gamma": 0.5},
    "dave": {"traits": "impulsive, friendly", "alpha": 0.3, "beta": 0.5, "gamma": 0.2},
    "eve": {"traits": "curious, introverted", "alpha": 0.5, "beta": 0.2, "gamma": 0.3},
}

# Use personality in prompts
scorer(
    observation=obs,
    agent_goal=agent.goal,
    personality=agent.personality  # influences reasoning
)

# Use personality for retrieval weights
weights = PERSONALITIES[agent.name]
retrieve_memories(agent_id, alpha=weights['alpha'], beta=weights['beta'], gamma=weights['gamma'])
```

### 4. Memory Pruning
```python
async def prune_low_importance_memories():
    """Run daily: archive memories with importance < 0.2."""
    cutoff_date = datetime.now() - timedelta(days=7)
    
    archived = con.execute("""
        DELETE FROM memories
        WHERE importance < 0.2 
        AND ts < ?
        RETURNING id
    """, [cutoff_date]).fetchall()
    
    logger.info(f"Archived {len(archived)} low-importance memories")
```

### 5. UI for Debugging Agent Reasoning
```tsx
// AgentInspector component
export function AgentInspector({ agentId }: { agentId: number }) {
  const { data } = useSWR(`/api/agents/${agentId}/trace`, fetcher);
  
  return (
    <div className="fixed right-0 top-0 w-96 h-screen bg-white border-l p-4 overflow-y-auto">
      <h2 className="font-bold">{data?.name}</h2>
      <div className="mt-2 text-sm">
        <div>State: <span className={data?.state === 'confused' ? 'text-red-600' : ''}>{data?.state}</span></div>
        <div>Goal: {data?.goal}</div>
        <div>Personality: {data?.personality}</div>
      </div>
      
      <h3 className="font-bold mt-4">Current Plan</h3>
      <pre className="text-xs bg-gray-100 p-2 rounded">{data?.current_plan}</pre>
      
      <h3 className="font-bold mt-4">Recent Memories (Top 5)</h3>
      {data?.memories.map(m => (
        <div key={m.id} className="text-xs border-b py-2">
          <div className="font-medium">Importance: {m.importance.toFixed(2)}</div>
          <div>{m.content}</div>
        </div>
      ))}
      
      <h3 className="font-bold mt-4">Latest Reflection</h3>
      <div className="text-xs italic">{data?.latest_reflection}</div>
      
      {/* Show last ChainOfThought trace if available */}
      {data?.last_trace && (
        <>
          <h3 className="font-bold mt-4">Reasoning Trace</h3>
          <pre className="text-xs bg-yellow-50 p-2 rounded overflow-x-auto">
            {data.last_trace}
          </pre>
        </>
      )}
    </div>
  );
}
```

---

## Revised Milestones (7-10 Days)

### Day 0.5: Hardcoded Validation ⚠️ CRITICAL
**Goal**: Prove simulation loop + UI + DB work WITHOUT LLM complexity

- [ ] 3 agents with **hardcoded behaviors** (random walk, triggered speech)
- [ ] DuckDB setup: agents, memories tables (no embeddings yet)
- [ ] Basic FastAPI with WebSocket broadcasting positions
- [ ] Minimal Next.js map (can be simple canvas or Phaser.js)
- [ ] Agents "perceive" when adjacent, log to console

**Output**: Video of 3 agents moving, "seeing" each other, storing fake memories

**Time**: 4-6 hours (don't skip this!)

---

### Day 2: DuckDB + Vector Setup + Latency Baseline
- [ ] Add `vss` extension to DuckDB
- [ ] Generate embeddings for 100 test phrases in Colab
- [ ] Store embeddings in `memories` table
- [ ] Test vector search: query "party" returns relevant memories
- [ ] Implement triad scoring function (with tunable α,β,γ)
- [ ] **NEW: Add latency tracking for LLM calls** (timed_llm_call wrapper)
- [ ] **NEW: Run 20-tick baseline with Groq Llama-3.2-3B**
- [ ] **NEW: Measure p50/p95 latencies for each signature**
- [ ] **NEW: Decision point - keep 2s ticks or adjust to 3s?**
- [ ] Create 3 benchmark scenarios for retrieval testing (emergency, social, strategic)

**Output**: Query interface works, retrieval quality reasonable, **latency baseline established**

**Decision Point**: 
- If p95 latency < 1.5s → proceed with 2-second ticks
- If p95 latency 1.5-2.5s → adjust config to 3-second ticks  
- If p95 latency > 2.5s → investigate (switch model? check API?)

---

### Day 2: Uncompiled DSPy Agent (Groq Free Tier)
- [ ] Integrate Groq API (Llama-3.2-3B)
- [ ] Wire `ScoreImportance` into sim loop (uncompiled)
- [ ] Wire `Reflect` when threshold hit (uncompiled)
- [ ] Agents store real LLM-generated insights
- [ ] Add basic error handling (timeouts → confused state)

**Output**: 5 agents running for 10 minutes, reflecting 2-3 times each

---

### Day 3: Seed Collection + Metric Definition ⚠️ CRITICAL (Budget 6-8 hours)
**This day determines compilation success. Do not rush.**

- [ ] Create seed collection checklist (social, environmental, goal-relevant, emotional, mundane)
- [ ] Manually curate 30-40 observation→score pairs with **rationale** field
- [ ] Validate: All score ranges 1-10 represented (min 2 examples each)
- [ ] Get 2-3 people to independently rate 10 observations
- [ ] Calculate inter-rater agreement (target: Cohen's kappa > 0.6)
- [ ] If kappa < 0.6, clarify scoring rubric and re-rate
- [ ] Add 5-10 edge cases (adversarial examples)
- [ ] Define `importance_metric` function (allow ±1 error initially)
- [ ] Curate 15-20 memory bundles → insights (save to `seeds_reflector.json`)
- [ ] Generate distribution plots (check for gaps in 1-10 range)
- [ ] Upload seeds + analysis to Google Drive
- [ ] Run uncompiled agents, measure baseline town_score
- [ ] Document scoring rubric in `rationale_guide.md` for future iterations

**Output**: High-quality seeds with validated inter-rater agreement, baseline metrics recorded

**Time Breakdown**:
- 2 hours: Diverse observation collection
- 2 hours: Ground truth validation (multi-rater)
- 2 hours: Context enrichment + rationale writing
- 1-2 hours: Edge cases + distribution analysis

---

### Day 4: Compilation Run (Colab Overnight)
- [ ] Set up Colab notebook with model loading
- [ ] Load seeds from Drive
- [ ] Use **GEPA** optimizer (primary choice, ~40 rollouts)
- [ ] Start compilation (leave overnight, expect 4-6 hours with GEPA vs 6-8 with MIPROv2)
- [ ] Download `compiled_scorer.json` and `compiled_reflector.json`
- [ ] If GEPA has issues: fallback to MIPROv2 and re-run

**Output**: Compiled programs ready to test

**Expected Runtime**: GEPA should be ~30-40% faster than MIPROv2 due to fewer rollouts

---

### Day 5: A/B Testing (Compiled vs Uncompiled) + Retrieval Tuning
- [ ] Load compiled programs into FastAPI backend
- [ ] Run 20-minute scenario with compiled agents
- [ ] Compare town_score: compiled vs uncompiled
- [ ] **If improvement < 10%**: Analyze failure modes, iterate on seeds/metric (mini Day 3 redux)
- [ ] **If improvement > 10%**: Proceed confidently
- [ ] **NEW: Run retrieval weight grid search on benchmark scenarios**
- [ ] **NEW: Test scenario-specific weight overrides (emergency vs social)**
- [ ] **NEW: Measure retrieval quality improvement with tuned weights**
- [ ] Log detailed comparison: which agents improved most? Which signatures?

**Output**: Data showing compiled is X% better (target: 15-25% improvement), **retrieval weights validated**

**Iteration Trigger**: If improvement < 10%, budget 4 more hours for seed refinement before Day 6

---

### Day 6: Event Scenario + Planning
- [ ] Hard-code event: "Party at Maria's house at 7pm"
- [ ] Maria invites 2-3 agents at 6:50pm
- [ ] Add simple `PlanDay` signature (text plan, not ReAct yet)
- [ ] Compile `PlanDay` if time permits
- [ ] Measure event coherence: % of invitees attending

**Output**: Event coherence metric (target: 60%+ attendance)

---

### Day 7: God Mode + Debugging UI
- [ ] Add `/god/inject_event`, `/god/pause`, `/god/step` endpoints
- [ ] Build AgentInspector panel (click agent → see memories, trace)
- [ ] Add personality traits to agents (5 different personalities)
- [ ] Test: inject "fire alarm" event, see if agents react differently

**Output**: Debugging tools work, personality differences visible

---

### Day 8-9: Polish + Documentation
- [ ] Add system status panel to UI (show LLM, optimizer, score)
- [ ] Record 3-minute demo video: before/after compilation
- [ ] Write README with results table
- [ ] Deploy frontend to Vercel (free tier)
- [ ] Share Colab notebook publicly

**Output**: Polished demo, README, shareable links

---

### Day 10: (Optional) Future Enhancements
- [ ] Add SIMBA compilation for Reflect
- [ ] Add lightweight ReAct (max_iters=3) for PlanDay
- [ ] Implement memory pruning job
- [ ] Add agent-specific retrieval weights
- [ ] Scale to 8-10 agents

---

## Observability (Start Simple)

### Choice: Python Logging + JSON Files
```python
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('mini_town.log'),
        logging.StreamHandler()
    ]
)

# Log structured events for analysis
def log_agent_event(agent_id, event_type, data):
    """Log to JSON for easy analysis."""
    with open('agent_events.jsonl', 'a') as f:
        json.dump({
            'ts': datetime.now().isoformat(),
            'agent_id': agent_id,
            'event': event_type,
            'data': data
        }, f)
        f.write('\n')

# Usage
log_agent_event(agent_id, 'reflection', {'insight': insight, 'trigger_sum': importance_sum})
```

**Analysis**: Use `jq` for quick queries
```bash
# How many reflections per agent?
cat agent_events.jsonl | jq -r 'select(.event=="reflection") | .agent_id' | sort | uniq -c

# Average importance scores
cat agent_events.jsonl | jq -r 'select(.event=="score") | .data.score' | awk '{sum+=$1; n++} END {print sum/n}'
```

**Future upgrade path**: Add LangSmith or Helicone when you need visual dashboards (post-Day 10)

---

## Config Management (config.yml)

```yaml
# Central configuration (single source of truth)
simulation:
  num_agents: 5
  tick_interval: 2.0  # seconds
  reflect_threshold: 3.5  # sum of importance to trigger reflection

llm:
  provider: groq  # groq | together | openai
  model: llama-3.2-3b-preview
  api_key: ${GROQ_API_KEY}
  temperature: 0.3
  max_tokens: 512
  timeout: 5.0  # seconds before agent enters "confused" state

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384

retrieval:
  default_alpha: 0.5  # relevance
  default_beta: 0.3   # recency
  default_gamma: 0.2  # importance
  top_k: 10
  
  # Scenario-specific overrides
  scenarios:
    social_gathering:
      alpha: 0.3
      beta: 0.5
      gamma: 0.2
    crisis_response:
      alpha: 0.6
      beta: 0.3
      gamma: 0.1

compilation:
  optimizer: gepa  # gepa (primary) | miprov2 (fallback) | simba | grpo
  seeds_path: /content/drive/MyDrive/mini-town/seeds/

  gepa:
    budget: 40  # fewer rollouts than MIPROv2 (40 vs 50-100+)

  miprov2:
    auto: medium  # light | medium | heavy
    max_bootstrapped_demos: 4
    max_labeled_demos: 5
    num_trials: 10

database:
  path: town.db
  memory_prune_days: 7
  importance_threshold: 0.2

observability:
  log_level: INFO
  log_file: mini_town.log
  structured_events: agent_events.jsonl

god_mode:
  enabled: true
  allow_pause: true
  allow_inject: true
```

### Load in Backend
```python
import yaml
import os

with open("config.yml") as f:
    cfg = yaml.safe_load(f)

# Substitute env vars
cfg['llm']['api_key'] = os.getenv(
    cfg['llm']['api_key'].replace('${', '').replace('}', '')
)

# Use throughout app
TICK_INTERVAL = cfg['simulation']['tick_interval']
REFLECT_THRESHOLD = cfg['simulation']['reflect_threshold']
```

---

## Cost Breakdown (Realistic)

| Phase | Provider | Usage | Cost |
|-------|----------|-------|------|
| **Dev (Day 0-3)** | Groq free tier | 3k requests | $0 |
| **Compilation** | Colab Pro T4 | 8 hours | $0 (included) |
| **Testing (Day 4-5)** | Groq → Together.ai | 100k tokens | $0.02 |
| **Demo (Day 6-7)** | Together.ai Qwen2.5-7B | 200k tokens | $0.04 |
| **Polish (Day 8-9)** | Together.ai | 100k tokens | $0.02 |
| **Contingency** | Cloud GPU backup | 1 hour | $0.70 (if needed) |
| **Total** | | | **$0.08 - $0.80** |

**Budget remaining**: $4.20 - $4.92 🎉

---

## Critical Tuning Parameters (Expect to Iterate)

### 1. Tick Interval vs LLM Latency

**The Problem**: 2-second ticks with 5 agents = 2.5 LLM calls/second. If each call takes 1-2 seconds, you'll lag behind real-time.

**Measurement Strategy** (Day 2):
```python
import time
from collections import defaultdict

latency_tracker = defaultdict(list)
failure_tracker = defaultdict(lambda: {'success': 0, 'failed': 0})

async def timed_llm_call(signature, **kwargs):
    """Track both latency and failure rates for all LLM calls."""
    start = time.time()
    try:
        result = await llm_call(signature, **kwargs)
        elapsed = time.time() - start
        latency_tracker[signature.__name__].append(elapsed)
        failure_tracker[signature.__name__]['success'] += 1
        return result
    except Exception as e:
        elapsed = time.time() - start
        failure_tracker[signature.__name__]['failed'] += 1
        logger.error(f"{signature.__name__} failed after {elapsed:.2f}s: {e}")
        raise

# Log percentiles and failure rates every 50 ticks
if tick_count % 50 == 0:
    for sig_name, latencies in latency_tracker.items():
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)

        stats = failure_tracker[sig_name]
        total = stats['success'] + stats['failed']
        success_rate = stats['success'] / total if total > 0 else 0

        logger.info(f"{sig_name}: p50={p50:.2f}s, p95={p95:.2f}s, success_rate={success_rate:.1%}")
```

**Adaptive Tick Interval** (implement by Day 5):
```python
class AdaptiveSimulation:
    def __init__(self):
        self.base_tick = 2.0
        self.current_tick = 2.0
        self.latency_window = []
    
    def adjust_tick(self, avg_latency):
        """Scale tick interval to keep sim responsive."""
        self.latency_window.append(avg_latency)
        if len(self.latency_window) > 10:
            self.latency_window.pop(0)
        
        avg = np.mean(self.latency_window)
        
        # If LLM calls take >80% of tick time, slow down
        if avg > 0.8 * self.current_tick:
            self.current_tick = min(5.0, self.current_tick * 1.2)
            logger.warning(f"Tick interval increased to {self.current_tick:.1f}s")
        
        # If we're comfortably fast, speed up
        elif avg < 0.5 * self.current_tick and self.current_tick > self.base_tick:
            self.current_tick = max(self.base_tick, self.current_tick * 0.9)
            logger.info(f"Tick interval decreased to {self.current_tick:.1f}s")
        
        return self.current_tick
```

**Fallback Options**:
- **Option A**: Stagger agent updates (3 agents on even ticks, 2 on odd ticks)
- **Option B**: Skip Reflect checks every other tick (it's not time-critical)
- **Option C**: Use Groq's blazing fast inference (often <500ms for 3B models)

**Target Latencies** (measured on Day 2):
| Signature | Groq Llama-3.2-3B | Together Qwen2.5-7B | Acceptable? |
|-----------|-------------------|---------------------|-------------|
| ScoreImportance | 200-400ms | 600-900ms | ✅ Yes |
| Reflect (CoT) | 400-800ms | 1.2-1.8s | ⚠️ Monitor |
| PlanDay | 300-600ms | 800-1.5s | ✅ Yes |

**Decision Point** (end of Day 2):
- If p95 latency < 1.5s → keep 2-second ticks
- If p95 latency 1.5-2.5s → move to 3-second ticks
- If p95 latency > 2.5s → investigate (model too large? API issues?)

---

### 2. Seed Data Quality (Day 3 Expanded - Budget 6-8 hours)

**This is the make-or-break day for compilation results.** Poor seeds = wasted Colab time.

#### Seed Collection Workflow

**Phase 1: Observation Diversity (2 hours)**
```python
# Target: 30-40 diverse observations covering:
# - Trivial (score 1-3): "The clock ticks", "A bird flies by"
# - Moderate (score 4-6): "Friend waves hello", "Rain starts falling"
# - Important (score 7-9): "Boss announces promotion", "Fire alarm goes off"
# - Critical (score 10): "Loved one proposes marriage", "Building collapses"

SEED_CHECKLIST = {
    "social": 8,      # greetings, invitations, arguments
    "environmental": 6,  # weather, time passage, location changes
    "goal-relevant": 8,  # directly affects agent's stated goal
    "emotional": 6,     # joy, anger, fear triggers
    "mundane": 6,       # background noise, irrelevant details
}

# Validate coverage
def check_seed_distribution(seeds):
    scores = [s['score'] for s in seeds]
    plt.hist(scores, bins=10)
    plt.title("Seed Score Distribution")
    plt.savefig("seed_distribution.png")
    
    # Check for gaps
    for i in range(1, 11):
        count = sum(1 for s in scores if s == i)
        if count == 0:
            logger.warning(f"No seeds with score={i}")
```

**Phase 2: Ground Truth Validation (2 hours)**
```python
# Have 2-3 people independently rate 10 observations
# Calculate inter-rater reliability (Cohen's kappa)
from sklearn.metrics import cohen_kappa_score

rater_1 = [7, 3, 9, 2, 5, 8, 4, 6, 10, 3]
rater_2 = [8, 3, 9, 1, 6, 7, 4, 6, 10, 2]

kappa = cohen_kappa_score(rater_1, rater_2)
# Target: kappa > 0.6 (moderate agreement)
# If kappa < 0.6, your definitions are ambiguous - clarify and re-rate
```

**Phase 3: Context Richness (2 hours)**
```json
// BAD seed (ambiguous)
{
  "observation": "Maria smiles",
  "agent_goal": "Make friends",
  "score": 5
}

// GOOD seed (contextual)
{
  "observation": "Maria smiles warmly and invites you to her birthday party next week",
  "agent_goal": "Build relationships in the neighborhood",
  "score": 8,
  "rationale": "Direct invitation advances social goal; future commitment"
}
```

**Phase 4: Edge Case Coverage (1-2 hours)**
```python
# Include adversarial examples
EDGE_CASES = [
    {
        "observation": "You hear a loud noise from three blocks away",
        "agent_goal": "Prepare for the town meeting",
        "score": 2,  # Distant, likely irrelevant
        "note": "Tests spatial reasoning"
    },
    {
        "observation": "Your childhood friend mentions the meeting was moved to 3pm",
        "agent_goal": "Attend the 2pm town meeting",
        "score": 10,  # Critical schedule conflict
        "note": "Tests goal-critical information"
    },
]
```

**Seed Validation Script** (run before uploading to Drive):
```python
def validate_seed_quality(seeds_path):
    """Run before uploading to Drive - catches quality issues early."""
    import json
    from collections import Counter

    with open(seeds_path) as f:
        seeds = json.load(f)

    issues = []

    # Check 1: Distribution coverage
    scores = [s['score'] for s in seeds]
    score_counts = Counter(scores)

    print(f"\n📊 Score Distribution:")
    for i in range(1, 11):
        count = score_counts.get(i, 0)
        bar = "█" * count
        status = "✅" if count >= 2 else "⚠️ "
        print(f"  {status} Score {i:2d}: {bar} ({count})")
        if count < 2:
            issues.append(f"Score {i} has only {count} example(s), need ≥2")

    # Check 2: Required fields and quality
    for idx, s in enumerate(seeds):
        if 'rationale' not in s:
            issues.append(f"Seed {idx}: Missing rationale field")
        elif len(s.get('rationale', '')) < 20:
            issues.append(f"Seed {idx}: Rationale too short (<20 chars)")

        if 'observation' not in s or len(s['observation']) < 10:
            issues.append(f"Seed {idx}: Observation too short")

        if 'agent_goal' not in s:
            issues.append(f"Seed {idx}: Missing agent_goal")

    # Check 3: Agent goal diversity
    unique_goals = set(s.get('agent_goal', '') for s in seeds)
    print(f"\n🎯 Goal Diversity: {len(unique_goals)} unique agent goals")
    if len(unique_goals) < 4:
        issues.append(f"Only {len(unique_goals)} unique agent goals, recommend ≥4")

    # Check 4: Context richness (look for overly short observations)
    short_obs = [s for s in seeds if len(s.get('observation', '')) < 30]
    if len(short_obs) > len(seeds) * 0.3:  # >30% are too short
        issues.append(f"{len(short_obs)} observations are <30 chars (may lack context)")

    # Final report
    print(f"\n{'='*60}")
    if not issues:
        print(f"✅ PASSED: Validated {len(seeds)} seeds - ready for compilation!")
    else:
        print(f"⚠️  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\nFix these issues before uploading to Drive.")
    print(f"{'='*60}\n")

    return len(issues) == 0

# Usage
if __name__ == "__main__":
    validate_seed_quality("seeds/scorer_v1.json")
    validate_seed_quality("seeds/reflector_v1.json")
```

**Quality Metrics for Seeds**:
- [ ] 30-40 total seeds (not just 20)
- [ ] All score ranges 1-10 represented (at least 2 examples each)
- [ ] Inter-rater agreement > 0.6 kappa
- [ ] At least 5 seeds explicitly test edge cases
- [ ] Each seed has `rationale` field explaining the score
- [ ] Seeds map to at least 4 different agent personalities

**Day 3 Deliverables**:
```
mini-town/
  seeds/
    scorer_v1.json          (30-40 observations)
    reflector_v1.json       (15-20 memory→insight pairs)
    seed_analysis.ipynb     (distribution plots, kappa scores)
    edge_cases.json         (5-10 adversarial examples)
    rationale_guide.md      (scoring rubric for future seeds)
```

---

### 3. Retrieval Triad Weights (α, β, γ) - Scenario Testing Framework

**The Problem**: Default (0.5, 0.3, 0.2) won't work for all scenarios. You need a systematic way to tune.

#### Benchmark Scenarios (Create on Day 1-2)

```python
# Define test scenarios with expected memory priorities
BENCHMARK_SCENARIOS = {
    "emergency": {
        "description": "Fire alarm just went off",
        "query": "What should I do right now?",
        "expected_top_memory": "Fire safety procedures learned last week",
        "optimal_weights": {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},  # Relevance matters most
    },
    "social_planning": {
        "description": "Deciding whether to attend party tonight",
        "query": "Who will be at the party?",
        "expected_top_memory": "Maria mentioned party attendees 5 minutes ago",
        "optimal_weights": {"alpha": 0.3, "beta": 0.6, "gamma": 0.1},  # Recent matters most
    },
    "long_term_relationship": {
        "description": "Considering asking Bob for a favor",
        "query": "What's my relationship with Bob?",
        "expected_top_memory": "Bob helped me move last year (high importance)",
        "optimal_weights": {"alpha": 0.3, "beta": 0.1, "gamma": 0.6},  # Importance matters most
    },
}
```

#### Automated Tuning Script (Run on Day 5)

```python
def evaluate_retrieval_config(alpha, beta, gamma, scenarios):
    """Test retrieval quality across scenarios."""
    score = 0
    for scenario in scenarios.values():
        results = retrieve_memories(
            agent_id=test_agent,
            query=scenario['query'],
            alpha=alpha, beta=beta, gamma=gamma,
            top_k=5
        )
        
        # Check if expected memory is in top-3
        expected_id = scenario['expected_top_memory_id']
        if expected_id in [r.id for r in results[:3]]:
            score += 1
    
    return score / len(scenarios)

# Grid search (coarse)
best_config = None
best_score = 0

for alpha in [0.3, 0.4, 0.5, 0.6]:
    for beta in [0.2, 0.3, 0.4]:
        gamma = 1.0 - alpha - beta
        if gamma < 0.1 or gamma > 0.5:
            continue
        
        score = evaluate_retrieval_config(alpha, beta, gamma, BENCHMARK_SCENARIOS)
        if score > best_score:
            best_score = score
            best_config = (alpha, beta, gamma)

logger.info(f"Best config: α={best_config[0]}, β={best_config[1]}, γ={best_config[2]}, score={best_score}")
```

#### Per-Scenario Overrides (config.yml)

```yaml
retrieval:
  default:
    alpha: 0.5
    beta: 0.3
    gamma: 0.2
  
  scenario_overrides:
    emergency_response:
      alpha: 0.7  # Prioritize relevance (e.g., "fire procedures")
      beta: 0.2
      gamma: 0.1
    
    social_gathering:
      alpha: 0.3
      beta: 0.6  # Prioritize recency (e.g., "who said they're coming")
      gamma: 0.1
    
    strategic_planning:
      alpha: 0.3
      beta: 0.1
      gamma: 0.6  # Prioritize importance (e.g., "boss's expectations")
  
  personality_modifiers:
    # Agents can have personality-based adjustments
    analytical:
      alpha: +0.1  # More weight on relevance
      beta: -0.05
      gamma: -0.05
    
    impulsive:
      alpha: -0.1
      beta: +0.1  # More weight on recent
      gamma: 0.0
```

#### Dynamic Weight Selection (implement by Day 6)

```python
def get_retrieval_weights(agent, context):
    """Choose weights based on agent personality and current context."""
    base = cfg['retrieval']['default']
    
    # Apply scenario override if detected
    if detect_emergency(context):
        base = cfg['retrieval']['scenario_overrides']['emergency_response']
    elif detect_social_context(context):
        base = cfg['retrieval']['scenario_overrides']['social_gathering']
    
    # Apply personality modifier
    if agent.personality in cfg['retrieval']['personality_modifiers']:
        mod = cfg['retrieval']['personality_modifiers'][agent.personality]
        base['alpha'] += mod.get('alpha', 0)
        base['beta'] += mod.get('beta', 0)
        base['gamma'] += mod.get('gamma', 0)
    
    # Normalize to sum to 1.0
    total = base['alpha'] + base['beta'] + base['gamma']
    return {k: v/total for k, v in base.items()}
```

**Testing Protocol** (Day 5-6):
1. Run each benchmark scenario with default weights → record top-5 memories
2. Run with scenario-specific weights → record top-5 memories
3. Manually inspect: did scenario weights improve relevance?
4. If improvement < 20% → revisit scenario definitions or add more memory diversity

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Compilation doesn't improve scores** | Iterate on metric definition (2-3 cycles budgeted), add more diverse seeds, try MIPROv2 if GEPA stalls |
| **LLM calls too slow for real-time sim** | Use Groq (fastest inference), adaptive tick interval, stagger agent updates, skip non-critical Reflect checks |
| **Poor seed quality ruins compilation** | **Budget 6-8 hours for Day 3**, validate inter-rater agreement (kappa > 0.6), include edge cases, document rationale |
| **Retrieval weights don't generalize** | Create 3+ benchmark scenarios, automated grid search, per-scenario overrides in config, test on Day 5 |
| **Tool calls break the sim** | Pydantic validation on all tools, graceful fallback to "confused" state, never crash |
| **DuckDB vector search too slow** | Pre-filter by agent_id + time before vector search, use HNSW index, keep embeddings 384-dim |
| **Agents behave too similarly** | Add personality system (Day 7), vary retrieval weights per agent, inject diversity in prompts |
| **Memory table grows too large** | Implement pruning (delete importance<0.2 older than 7 days), archive to JSON |
| **Colab disconnects mid-compile** | Use Colab Pro's longer runtime, checkpoint compiled programs every 30 min, have cloud GPU backup ($0.70) |

---

## Success Criteria (Realistic)

### Minimum Viable Demo
- [x] 5 agents running for 30+ minutes without crashes
- [x] At least one compiled DSPy program (ScoreImportance or Reflect)
- [x] Measurable improvement in town_score (even 10-15% is publishable)
- [x] One successful event scenario (party with 60%+ attendance)
- [x] UI with map, agent inspector, god mode controls
- [x] README with results table and Colab notebook link

### Stretch Goals (if time permits)
- [ ] All three modules compiled (ScoreImportance, Reflect, PlanDay)
- [ ] Comparison of GEPA vs MIPROv2 (run both, compare convergence speed and quality)
- [ ] Agent personality system with visible behavior differences
- [ ] Memory graph visualization (note links)
- [ ] Lightweight ReAct with tool use (max_iters=3)

---

## Future Expansion (Post-MVP)

### Phase 2 (Week 2-3)
- Add SIMBA compilation for hard cases
- Implement GRPO for end-to-end RL optimization
- Scale to 10-15 agents
- Add more complex tools (ReAct with max_iters=5)
- Implement memory graph queries (Neo4j or JSON-based)

### Phase 3 (Month 2)
- Multi-scenario support (town square, cafe, park)
- Agent learning from interactions (update DSPy programs online)
- Richer UI (3D map, agent avatars, emotion indicators)
- Public demo with live audience interaction
- Blog post / paper writeup

---

## Key Takeaways

1. **Start small, prove the concept** - 5 agents, 1 optimizer, simple metrics
2. **Day 0.5 is non-negotiable** - Validate your architecture before adding LLM complexity
3. **Compilation takes time** - Budget 3-4 hours per cycle, iterate 2-3 times
4. **Groq free tier is your friend** - Saves $2-3 during development
5. **Colab Pro is clutch** - Free GPU for compilation = stay in budget
6. **Simple observability first** - grep + jq beat fancy dashboards for solo projects
7. **God mode saves you** - Debugging agents is hard; pause/step/inject are essential
8. **Personality matters** - Identical agents are boring; 5 distinct personalities make it interesting

---

## References

- **Generative Agents (Smallville)**: Park et al., 2023 - [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- **DSPy Framework**: [dspy.ai](https://dspy.ai) - Typed signatures, compilation
- **MIPROv2**: [DSPy Docs](https://dspy.ai/api/optimizers/MIPROv2/)
- **GEPA (2025)**: Reflective prompt evolution - [arXiv reference needed]
- **DuckDB vss extension**: [GitHub](https://github.com/duckdb/duckdb)
- **ReAct (Reasoning + Acting)**: Yao et al., 2023 - [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Groq API**: [console.groq.com](https://console.groq.com)
- **Together.ai**: [together.ai](https://together.ai)

---

**Last Updated**: Based on 2025 tooling landscape  
**Estimated Total Cost**: $0.10 - $2.00  
**Timeline**: 7-10 days (solo developer)  
**Risk Level**: Medium (mitigated by Day 0.5 and iteration budget)

---

*This plan is intentionally conservative. If things go well, you'll finish early with budget to spare. If you hit snags, you have buffer time and money. The architecture is solid enough to scale post-MVP.*