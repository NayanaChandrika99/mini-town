# CLAUDE.md - Mini-Town Project Context

## Project Overview

**Name**: Compiled Generative-Agents Mini-Town (DSPy Edition)

**Goal**: Build a small simulation of 5 autonomous NPCs that perceive, reflect, plan, and act—where each cognitive skill is a typed DSPy program compiled against measurable metrics. Demonstrate that prompt optimization (MIPROv2/GEPA) improves agent behavior compared to uncompiled baselines.

**Timeline**: 7-10 days (solo developer)  
**Budget**: $5 maximum  
**Status**: Pre-development (planning phase complete)

---

## Core Philosophy

1. **Start small, prove the concept**: 5 agents, not 10-20. One optimizer, not four.
2. **Validate early**: Day 0.5 uses hardcoded behaviors to prove sim loop + UI + DB work before adding LLM complexity.
3. **Compilation is the star**: The entire project exists to demonstrate DSPy compilation improves agents. Everything else is scaffolding.
4. **Budget constraints drive architecture**: Free resources (Colab Pro, Groq, DuckDB) over paid services. Strategic cloud GPU use only for demos.
5. **Iteration is expected**: 2-3 compilation cycles budgeted. First metric/seed set won't be perfect.
6. **Error Logging**: Add any error occurred to the error_log.md file.
7. **FAilure**: If the code fails while exceutuing the plan, do not switch or implement a different plan before giving me teh options of informing me with teh error
---

## Technical Architecture

### Stack Summary

| Component | Technology | Why | Cost |
|-----------|-----------|-----|------|
| **LLM (dev)** | Groq Llama-3.2-3B | 30 req/min free, fast inference | $0 |
| **LLM (demo)** | Together.ai Qwen2.5-7B | $0.20/1M tokens, reliable | ~$0.50 |
| **Compilation** | Colab Pro T4 GPU | Free compute for overnight runs | $0 |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim, fast, local | $0 |
| **Vector DB** | DuckDB + vss extension | Single file, HNSW indexes | $0 |
| **Backend** | FastAPI (Python 3.10+) | Async, WebSockets, lightweight | $0 |
| **Frontend** | Next.js 14 (TypeScript) | React, SSR, Vercel free hosting | $0 |
| **Observability** | Python logging + JSON files | grep/jq for analysis | $0 |

### Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│  Frontend (Next.js on Vercel)                      │
│  - Top-down map (5 agents as circles/sprites)     │
│  - AgentInspector panel (memories, plans, traces) │
│  - SystemPanel (LLM, optimizer, current score)    │
│  - God mode controls (pause, step, inject event)  │
└────────────────────────────────────────────────────┘
                        ↓ WebSocket
┌────────────────────────────────────────────────────┐
│  Backend (FastAPI, runs locally)                   │
│                                                    │
│  Simulation Loop (every 2-3 seconds):              │
│    1. Perceive: gather nearby events/agents        │
│    2. ScoreImportance: rate observations 1-10      │
│    3. Retrieve: top-10 memories (vector search)    │
│    4. Reflect: synthesize insights (if threshold)  │
│    5. Plan: create daily plan (simple text)        │
│    6. Act: move, speak, invite                     │
│    7. Error handling: timeout → "confused" state   │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  Storage (DuckDB: town.db)                         │
│  - agents: id, name, x, y, goal, personality       │
│  - memories: id, agent_id, text, importance, embed │
│  - notes: id, agent_id, insight, embed, links      │
│  - HNSW indexes for fast ANN retrieval             │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  LLM (Hybrid Strategy)                             │
│  - Dev/Test: Groq free tier (Llama-3.2-3B)        │
│  - Compilation: Colab Pro T4 GPU                   │
│  - Live Demo: Together.ai (Qwen2.5-7B)            │
└────────────────────────────────────────────────────┘
```

---

## DSPy Integration (Critical Context)

### What is DSPy?
DSPy treats prompts as typed programs that can be **compiled** (optimized) against metrics. Instead of manually tuning prompts, you define:
1. **Signatures**: Typed input/output specs (like function signatures)
2. **Modules**: Predict, ChainOfThought, ReAct implementations
3. **Metrics**: Functions that score program outputs
4. **Optimizers**: MIPROv2, GEPA, SIMBA, GRPO that auto-tune prompts

### Core Signatures (Start With 3)

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

### Module Usage

```python
# Simple predictor (uncompiled)
scorer = dspy.Predict(ScoreImportance)

# Chain-of-thought (uncompiled)
reflector = dspy.ChainOfThought(Reflect)

# Start simple for PlanDay (add ReAct later)
planner = dspy.Predict(PlanDay)

# Compilation happens in Colab (Day 4)
from dspy.optimizers import GEPA  # PRIMARY
compiled_scorer = GEPA(
    metric=importance_metric,
    budget=40  # fewer rollouts than MIPROv2
).compile(scorer, trainset=seeds)
```

### Optimizer Choice: GEPA (Primary) vs MIPROv2 (Fallback)

**GEPA** ⭐ **PRIMARY CHOICE** (2025):
- Reflective prompt evolution approach
- Faster convergence with fewer examples (~40 rollouts vs 50-100+)
- More efficient for budget constraints
- Less community documentation, but similar API to MIPROv2

**MIPROv2** (fallback if GEPA has issues):
- Jointly optimizes instructions + few-shot examples
- Well-documented, proven in DSPy
- Needs 20-40 seeds for good results
- More Colab time per compilation cycle

**Why GEPA first?**
- ~30-40% faster compilation (saves 2-3 hours per cycle)
- Fewer rollouts = less Colab usage = more iteration budget
- APIs are similar, easy to swap if needed

---

## Critical Tuning Parameters

### 1. Tick Interval (Expect to Adjust)

**Default**: 2 seconds  
**Constraint**: Must be > p95 LLM latency to avoid backlog

**Measurement** (implement on Day 2):
```python
latency_tracker = defaultdict(list)

async def timed_llm_call(signature, **kwargs):
    start = time.time()
    result = await llm_call(signature, **kwargs)
    elapsed = time.time() - start
    latency_tracker[signature.__name__].append(elapsed)
    return result
```

**Decision rules** (end of Day 2):
- p95 < 1.5s → keep 2s ticks ✅
- p95 1.5-2.5s → adjust to 3s ticks ⚠️
- p95 > 2.5s → investigate (wrong model? API issues?) 🚨

**Adaptive tick** (implement by Day 5):
Auto-adjust based on running latency average to keep sim responsive.

---

### 2. Seed Quality (Budget 6-8 Hours on Day 3)

**This is make-or-break for compilation.** Poor seeds = wasted Colab time.

**Requirements**:
- 30-40 diverse observations (not just 20)
- All score ranges 1-10 represented (min 2 each)
- Inter-rater agreement (Cohen's kappa) > 0.6
- 5-10 edge cases (adversarial examples)
- Every seed has `rationale` field

**Validation workflow**:
1. Collect diverse observations (2 hours)
2. Get 2-3 people to rate 10 examples independently
3. Calculate kappa score
4. If kappa < 0.6 → scoring rubric is ambiguous, clarify and retry
5. Add context-rich descriptions + edge cases
6. Document rubric in `rationale_guide.md`

**Seed categories to cover**:
- Social: 8 seeds (greetings, invitations, arguments)
- Environmental: 6 seeds (weather, time passage, location changes)
- Goal-relevant: 8 seeds (directly affects agent's stated goal)
- Emotional: 6 seeds (joy, anger, fear triggers)
- Mundane: 6 seeds (background noise, irrelevant details)

---

### 3. Retrieval Triad Weights (α, β, γ)

**Formula**: `score = α*relevance + β*recency + γ*importance`

**Default**: α=0.5, β=0.3, γ=0.2  
**Problem**: Won't work for all scenarios

**Benchmark scenarios** (create on Day 2):
```python
BENCHMARK_SCENARIOS = {
    "emergency": {
        "query": "What should I do right now?",
        "expected_top_memory": "Fire safety procedures",
        "optimal_weights": {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},
    },
    "social_planning": {
        "query": "Who will be at the party?",
        "expected_top_memory": "Maria mentioned attendees 5 min ago",
        "optimal_weights": {"alpha": 0.3, "beta": 0.6, "gamma": 0.1},
    },
    "long_term_relationship": {
        "query": "What's my relationship with Bob?",
        "expected_top_memory": "Bob helped me move last year",
        "optimal_weights": {"alpha": 0.3, "beta": 0.1, "gamma": 0.6},
    },
}
```

**Grid search** (run on Day 5):
Test combinations of α, β, γ on benchmark scenarios. Choose config that maximizes retrieval quality across all scenarios (or use scenario-specific overrides).

---

## Data Model (DuckDB Schema)

```sql
-- Agents table
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    x REAL DEFAULT 0,
    y REAL DEFAULT 0,
    goal TEXT,
    personality TEXT,  -- e.g., "social, risk-averse"
    current_plan TEXT,
    state TEXT DEFAULT 'active'  -- active | confused | idle
);

-- Memories with 384-dim embeddings
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

---

## Evaluation Metrics

### 1. Event Coherence
**Question**: Did invited agents attend the event?

```python
def event_coherence_metric(scenario_result):
    event_time = scenario_result['event']['time']
    invitees = scenario_result['event']['invitees']
    attendees = scenario_result['attendees']
    
    window = timedelta(minutes=10)
    on_time = [a for a in attendees 
               if abs(a['arrival_time'] - event_time) <= window]
    
    return len(on_time) / len(invitees)
```

**Target**: 60%+ attendance with compiled agents (vs <30% uncompiled)

---

### 2. Plan Fidelity
**Question**: How well did agent follow their stated plan?

```python
def plan_fidelity_metric(agent_id, duration_minutes=30):
    planned = get_planned_timeline(agent_id)
    executed = get_executed_timeline(agent_id, duration_minutes)
    
    # Normalized edit distance
    norm_dist = levenshtein_distance(planned, executed) / max(len(planned), len(executed))
    return 1 - norm_dist
```

**Target**: Fidelity > 0.6 with compiled agents

---

### 3. Memory Hit-Rate
**Question**: Can agent retrieve correct info from memory?

```python
def memory_hit_rate(agent_id, test_queries):
    correct = 0
    for query, expected_note_id in test_queries:
        results = retrieve_notes(agent_id, query, top_k=5)
        if expected_note_id in [r.id for r in results]:
            correct += 1
    return correct / len(test_queries)
```

**Target**: Hit rate > 0.7 with tuned retrieval weights

---

### 4. Combined Town Score
```python
def town_score(scenario_results, weights=(0.4, 0.3, 0.3)):
    w_event, w_plan, w_memory = weights
    return (
        w_event * event_coherence_metric(scenario_results) +
        w_plan * np.mean([plan_fidelity_metric(a) for a in agents]) +
        w_memory * np.mean([memory_hit_rate(a, queries) for a in agents])
    )
```

**This is the metric used for DSPy compilation.**

---

## Agent Personalities (5 Distinct Types)

```python
PERSONALITIES = {
    "alice": {
        "traits": "social, optimistic, talkative",
        "goal": "Build relationships in the neighborhood",
        "retrieval_weights": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
        "reflect_threshold": 3.0  # Reflects often
    },
    "bob": {
        "traits": "reclusive, analytical, introverted",
        "goal": "Complete research project on local history",
        "retrieval_weights": {"alpha": 0.6, "beta": 0.1, "gamma": 0.3},
        "reflect_threshold": 5.0  # Reflects rarely
    },
    "carol": {
        "traits": "risk-averse, organized, punctual",
        "goal": "Maintain community garden",
        "retrieval_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.5},
        "reflect_threshold": 3.5
    },
    "dave": {
        "traits": "impulsive, friendly, spontaneous",
        "goal": "Make every day an adventure",
        "retrieval_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
        "reflect_threshold": 2.5  # Reflects very often
    },
    "eve": {
        "traits": "curious, introverted, detail-oriented",
        "goal": "Document neighborhood stories",
        "retrieval_weights": {"alpha": 0.5, "beta": 0.2, "gamma": 0.3},
        "reflect_threshold": 4.0
    },
}
```

**Why personality matters**:
- Different retrieval weights → different memory priorities
- Different reflection thresholds → some agents think more/less
- Traits influence prompt context → diverse behaviors
- Makes the simulation more interesting to watch

---

## Milestones & Critical Path

### Day 0.5: Hardcoded Validation (4-6 hours) ⚠️ DO NOT SKIP
**Goal**: Prove sim loop + UI + DB work WITHOUT LLM complexity

- [ ] 3 agents with hardcoded behaviors (random walk)
- [ ] DuckDB setup: agents, memories tables (no embeddings yet)
- [ ] Basic FastAPI with WebSocket broadcasting positions
- [ ] Minimal Next.js map (canvas or Phaser.js)
- [ ] Agents "perceive" when adjacent, log to console

**Output**: Video of 3 agents moving and "seeing" each other

---

### Day 1: DuckDB + Vector Setup (4-6 hours)
- [ ] Add `vss` extension to DuckDB
- [ ] Generate embeddings for 100 test phrases in Colab
- [ ] Store embeddings, test vector search
- [ ] Implement triad scoring function

**Output**: Query "party" returns relevant memories

---

### Day 2: Latency Baseline + Uncompiled DSPy (6-8 hours)
- [ ] Integrate Groq API (Llama-3.2-3B)
- [ ] Wire `ScoreImportance` and `Reflect` (uncompiled)
- [ ] **Add latency tracking wrapper**
- [ ] **Run 20-tick baseline, measure p50/p95**
- [ ] **Decision: keep 2s ticks or adjust to 3s?**
- [ ] Create benchmark scenarios for retrieval testing

**Output**: 5 agents running 10 min, **latency baseline established**

---

### Day 3: Seed Collection (6-8 hours) ⚠️ CRITICAL
**This determines compilation success. Budget full time.**

- [ ] Collect 30-40 diverse observations (use checklist)
- [ ] Get 2-3 people to independently rate 10 examples
- [ ] Calculate inter-rater agreement (Cohen's kappa)
- [ ] If kappa < 0.6, clarify rubric and retry
- [ ] Add edge cases + rationale for each seed
- [ ] Generate distribution plots
- [ ] Document scoring rubric
- [ ] Measure baseline town_score

**Output**: High-quality seeds with validated agreement

---

### Day 4: Compilation (4-6 hours in Colab with GEPA, mostly unattended)
- [ ] Set up Colab notebook with model loading
- [ ] Load seeds from Google Drive
- [ ] Use **GEPA** optimizer (primary choice, ~40 rollouts)
- [ ] Start compilation (leave overnight, expect 4-6 hours vs 6-8 with MIPROv2)
- [ ] Download compiled programs
- [ ] If GEPA has issues: fallback to MIPROv2 and re-run

**Output**: `compiled_scorer.json`, `compiled_reflector.json`

**Expected Runtime**: GEPA should be ~30-40% faster than MIPROv2 due to fewer rollouts

---

### Day 5: A/B Testing + Retrieval Tuning (6-8 hours)
- [ ] Load compiled programs into backend
- [ ] Run 20-min scenario with compiled agents
- [ ] Compare town_score: compiled vs uncompiled
- [ ] **If improvement < 10%**: Iterate on seeds/metric
- [ ] **If improvement > 10%**: Proceed
- [ ] Run retrieval weight grid search
- [ ] Test scenario-specific overrides

**Output**: Compiled agents show 15-25% improvement

---

### Day 6: Event Scenario + Planning (6-8 hours)
- [ ] Hard-code "Party at Maria's at 7pm" event
- [ ] Maria invites 2-3 agents at 6:50pm
- [ ] Add `PlanDay` signature
- [ ] Measure event coherence (% attendance)

**Output**: Event coherence > 60%

---

### Day 7: God Mode + Debugging UI (6-8 hours)
- [ ] Add `/god/inject_event`, `/god/pause`, `/god/step`
- [ ] Build AgentInspector panel
- [ ] Add personality traits to agents
- [ ] Test: inject "fire alarm", observe reactions

**Output**: Debugging tools work, personalities visible

---

### Day 8-9: Polish + Documentation (8-12 hours)
- [ ] Add SystemPanel to UI
- [ ] Record 3-minute demo video
- [ ] Write comprehensive README
- [ ] Deploy frontend to Vercel
- [ ] Share Colab notebook publicly

**Output**: Polished demo, README, shareable links

---

### Day 10: (Optional) Future Enhancements
- [ ] Add SIMBA compilation
- [ ] Add lightweight ReAct (max_iters=3)
- [ ] Implement memory pruning
- [ ] Scale to 8-10 agents

---

## Key Design Decisions & Rationale

### Why DuckDB instead of PostgreSQL + pgvector?
- **Simpler**: Single file, no server, no setup
- **Fast enough**: HNSW indexes work well for 5 agents × 1000 memories
- **Free**: No cloud database needed
- **Portable**: Can copy `town.db` file for backups/sharing

### Why Groq for development?
- **Free tier**: 30 req/min is enough for 5 agents
- **Fast**: <500ms latency for 3B models
- **OpenAI-compatible**: Easy to swap for Together.ai later

### Why start with Predict, not ReAct?
- **Complexity**: ReAct adds tool use, error handling, iteration loops
- **Latency**: ReAct with 8 iterations could take 5-10 seconds
- **Compilation**: Harder to compile ReAct programs (more moving parts)
- **Proof of concept**: Simple Predict is enough to show compilation works

### Why 5 agents, not 10-20?
- **Token cost**: 10 agents = 2× the LLM calls
- **Debug complexity**: Harder to track which agent is misbehaving
- **Compilation time**: More agents = more scenarios to validate
- **Sufficient**: 5 distinct personalities show diversity

### Why budget 6-8 hours for seed collection?
- **Critical path**: Bad seeds = wasted 6-8 hours of Colab compilation
- **Inter-rater validation**: Catches ambiguous scoring rubrics early
- **Edge cases**: Test reasoning, not just happy path
- **Documentation**: Rubric enables future iteration

---

## Common Pitfalls & How to Avoid

### Pitfall 1: Skipping Day 0.5
**Symptom**: Day 2-3 you discover your DB schema doesn't support vector queries, or WebSocket disconnects constantly.

**Solution**: Spend 4-6 hours on Day 0.5 with hardcoded agents. Validate architecture before LLM complexity.

---

### Pitfall 2: Rushing seed collection
**Symptom**: Day 4 compilation finishes, but compiled program is worse than baseline.

**Solution**: Budget full 6-8 hours for Day 3. Check inter-rater kappa > 0.6. Document rationale. Add edge cases.

---

### Pitfall 3: Not measuring latency early
**Symptom**: Day 5 you realize agents can't keep up with 2s ticks, simulation lags.

**Solution**: Add latency tracking on Day 2. Make decision about tick interval before proceeding.

---

### Pitfall 4: Using default retrieval weights for all scenarios
**Symptom**: Agents retrieve irrelevant memories in critical moments (e.g., fire alarm → remembers yesterday's lunch).

**Solution**: Create benchmark scenarios on Day 2. Run grid search on Day 5. Use scenario-specific overrides.

---

### Pitfall 5: No error handling for LLM timeouts
**Symptom**: One slow LLM call crashes entire simulation.

**Solution**: Wrap all LLM calls in `asyncio.wait_for(timeout=5.0)`. Set agent state to "confused" on timeout. Never crash.

---

### Pitfall 6: Identical agent behaviors
**Symptom**: All 5 agents behave the same way. Boring to watch.

**Solution**: Add personality system from Day 1. Different goals, traits, retrieval weights, reflection thresholds.

---

## File Structure

```
mini-town/
├── README.md                    # Main documentation
├── CLAUDE.md                    # This file (project context)
├── config.yml                   # Central configuration
├── requirements.txt             # Python dependencies
│
├── backend/
│   ├── main.py                  # FastAPI app, simulation loop
│   ├── agents.py                # Agent class, perception, action
│   ├── memory.py                # DuckDB integration, retrieval
│   ├── dspy_modules.py          # Signatures, modules, compilation loading
│   ├── metrics.py               # Evaluation functions (town_score, etc.)
│   ├── god_mode.py              # Debug endpoints (pause, inject, step)
│   └── utils.py                 # Logging, embeddings, helpers
│
├── frontend/
│   ├── package.json
│   ├── pages/
│   │   └── index.tsx            # Main map view
│   ├── components/
│   │   ├── Map.tsx              # Agent visualization
│   │   ├── AgentInspector.tsx   # Memory/trace viewer
│   │   ├── SystemPanel.tsx      # LLM/optimizer/score display
│   │   └── GodMode.tsx          # Debug controls
│   └── lib/
│       └── websocket.ts         # Backend connection
│
├── compilation/                 # Colab notebooks (stored in Drive)
│   ├── compile_scorer.ipynb
│   ├── compile_reflector.ipynb
│   └── compile_planner.ipynb
│
├── seeds/                       # Training data for compilation
│   ├── scorer_v1.json           # 30-40 observation→score pairs
│   ├── reflector_v1.json        # 15-20 memory→insight pairs
│   ├── planner_v1.json          # (optional) goal→plan pairs
│   ├── seed_analysis.ipynb      # Distribution plots, kappa scores
│   └── rationale_guide.md       # Scoring rubric documentation
│
├── compiled/                    # Output from Colab compilation
│   ├── compiled_scorer.json
│   ├── compiled_reflector.json
│   └── prompt_*.txt             # Human-readable prompt inspection
│
├── data/
│   └── town.db                  # DuckDB database (gitignored)
│
└── logs/
    ├── mini_town.log            # Standard Python logs
    └── agent_events.jsonl       # Structured event logs (JSON lines)
```

---

## Environment Setup

### Backend (Python 3.10+)
```bash
pip install fastapi uvicorn websockets
pip install dspy-ai sentence-transformers
pip install duckdb duckdb-vss-extension
pip install numpy pandas pyyaml python-Levenshtein
```

### Frontend (Node 18+)
```bash
cd frontend
npm install next react react-dom
npm install @types/node @types/react
npm install swr  # For data fetching
```

### Colab Notebook
```python
!pip install dspy-ai transformers accelerate bitsandbytes sentence-transformers
```

---

## Configuration (config.yml)

```yaml
simulation:
  num_agents: 5
  tick_interval: 2.0  # Will be adjusted based on latency
  reflect_threshold: 3.5

llm:
  provider: groq  # groq | together | openai
  model: llama-3.2-3b-preview
  api_key: ${GROQ_API_KEY}
  temperature: 0.3
  max_tokens: 512
  timeout: 5.0

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384

retrieval:
  default_alpha: 0.5
  default_beta: 0.3
  default_gamma: 0.2
  top_k: 10

compilation:
  optimizer: gepa  # gepa (primary) | miprov2 (fallback)
  seeds_path: ./seeds/
  gepa:
    budget: 40  # fewer rollouts than MIPROv2 (40 vs 50-100+)
  miprov2:
    auto: medium
    max_bootstrapped_demos: 4
    max_labeled_demos: 5
    num_trials: 10

database:
  path: data/town.db
  memory_prune_days: 7
  importance_threshold: 0.2

observability:
  log_level: INFO
  log_file: logs/mini_town.log
  structured_events: logs/agent_events.jsonl

god_mode:
  enabled: true
```

---

## Testing Strategy

### Unit Tests (Optional, but helpful)
```python
# test_retrieval.py
def test_retrieval_triad():
    # Given memories with known embeddings, importance, timestamps
    # When retrieving with different α, β, γ
    # Then correct memories should rank highest
    pass

# test_dspy_modules.py
def test_score_importance_uncompiled():
    # Given observation + agent_goal
    # When calling uncompiled scorer
    # Then score should be 1-10 integer
    pass
```

### Integration Tests
```python
# test_simulation_loop.py
def test_full_tick_no_errors():
    # Given 3 agents in DB
    # When running one simulation tick
    # Then no exceptions, all agents updated
    pass
```

### Manual Testing Checklist (Day 0.5)
- [ ] Start backend, check logs for errors
- [ ] Open frontend, see 3 agents on map
- [ ] Agents move (hardcoded random walk)
- [ ] Click agent, see basic info
- [ ] No WebSocket disconnects after 5 minutes

---

## Success Criteria (What "Done" Looks Like)

### Minimum Viable Demo (MVP)
- [x] 5 agents running for 30+ minutes without crashes
- [x] At least one compiled DSPy program (ScoreImportance or Reflect)
- [x] Measurable improvement in town_score (10-25%)
- [x] One successful event scenario (party with 60%+ attendance)
- [x] UI with map, agent inspector, god mode controls
- [x] README with results table and Colab notebook link
- [x] 3-minute demo video showing before/after compilation

### Stretch Goals (if time permits)
- [ ] All three modules compiled (ScoreImportance, Reflect, PlanDay)
- [ ] Comparison of GEPA vs MIPROv2 optimizers (run both, compare convergence speed and quality)
- [ ] Agent personality system with visible behavior differences
- [ ] Memory graph visualization (note links)
- [ ] Lightweight ReAct with tool use (max_iters=3)

---

## Budget Breakdown (Realistic)

| Phase | Provider | Usage | Cost |
|-------|----------|-------|------|
| **Dev (Day 0-3)** | Groq free tier | 3k requests | $0 |
| **Compilation (Day 4)** | Colab Pro T4 (GEPA) | 4-6 hours | $0 (included) |
| **Testing (Day 5)** | Groq → Together.ai | 100k tokens | $0.02 |
| **Demo (Day 6-7)** | Together.ai Qwen2.5-7B | 200k tokens | $0.04 |
| **Polish (Day 8-9)** | Together.ai | 100k tokens | $0.02 |
| **Contingency** | Cloud GPU backup | 1 hour (if needed) | $0.70 |
| **Total** | | | **$0.08 - $0.80** |

**Budget remaining**: $4.20 - $4.92 for celebration coffee ☕

**Note**: GEPA's efficiency (~30-40% faster than MIPROv2) saves 2-3 hours per compilation cycle, leaving more iteration budget.

---

## How to Collaborate With Claude (or any AI assistant)

### Effective Prompts for Implementation Help

#### ✅ GOOD: Specific, contextual requests
```
"Using the DuckDB schema from CLAUDE.md, write the `store_memory()` function 
that takes agent_id, content, importance, embedding, and timestamp. Include 
error handling for duplicate IDs."
```

#### ❌ BAD: Vague requests without context
```
"Write a function to store memories"
```

---

### ✅ GOOD: Reference specific sections
```
"Based on the 'Retrieval Triad Weights' section, implement the 
compute_retrieval_score() function with configurable α, β, γ parameters. 
Use the formula from CLAUDE.md."
```

#### ❌ BAD: Assume the AI remembers everything
```
"Implement the retrieval scoring function"
```

---

### ✅ GOOD: Ask for architecture alignment
```
"The plan says to use Groq for development but Together.ai for demos. 
How should I structure the LLM client code to make swapping providers easy?"
```

#### ❌ BAD: Ask for generic solutions
```
"How do I connect to an LLM API?"
```

---

### When to Reference This Document

**Always reference CLAUDE.md when**:
- Starting a new implementation session
- Asking for code that touches multiple components
- Debugging issues (context helps identify root cause)
- Making architecture decisions (check if it aligns with plan)
- Estimating time/cost (budget constraints are documented here)

**Example opening prompt**:
```
"I'm working on the Mini-Town project (see CLAUDE.md). I'm on Day 2 and need 
to implement the latency tracking wrapper described in the 'Critical Tuning 
Parameters' section. Can you write the timed_llm_call() function that measures 
p50/p95 latencies for DSPy signatures?"
```

---

## Key Contacts & Resources

### Primary References
- **DSPy Documentation**: https://dspy.ai
- **MIPROv2 Guide**: https://dspy.ai/api/optimizers/MIPROv2/
- **DuckDB vss Extension**: https://github.com/duckdb/duckdb
- **Groq API Docs**: https://console.groq.com/docs
- **Together.ai Docs**: https://docs.together.ai

### Related Papers
- **Generative Agents (Smallville)**: Park et al., 2023 - arXiv:2304.03442
- **ReAct (Reasoning + Acting)**: Yao et al., 2023 - arXiv:2210.03629
- **GEPA (2025)**: Reflective prompt evolution - [arXiv reference TBD]

### Community Support
- **DSPy Discord**: Best place for optimizer questions
- **r/LocalLLaMA**: For local model deployment help
- **Groq Community**: API-specific questions

---

## Troubleshooting Guide

### Issue: Compilation doesn't improve scores
**Symptoms**: Compiled program performs same or worse than baseline

**Diagnosis**:
1. Check seed quality: Is kappa > 0.6? Are scores distributed across 1-10?
2. Check metric definition: Is it too lenient/strict?
3. Check training set size: 30-40 seeds minimum for GEPA

**Solutions**:
- Re-run seed validation (Day 3 checklist)
- Try different optimizer (MIPROv2 if using GEPA, or vice versa)
- Add more diverse seeds (especially edge cases)
- Adjust metric threshold (e.g., allow ±2 error instead of ±1)

---

### Issue: Agents lag behind real-time
**Symptoms**: Tick interval warnings, simulation slows down over time

**Diagnosis**:
1. Check latency logs: What's p95 for each signature?
2. Check memory table size: >10k memories? (needs pruning)
3. Check embedding dimension: Using 768 instead of 384?

**Solutions**:
- Enable adaptive tick interval
- Switch to faster model (Groq Llama-3.2-3B)
- Implement memory pruning (delete importance < 0.2, older than 7 days)
- Reduce embedding dimension to 384

---

### Issue: Retrieval returns irrelevant memories
**Symptoms**: Agent recalls lunch when fire alarm goes off

**Diagnosis**:
1. Check retrieval weights: Using default α,β,γ for all scenarios?
2. Check embedding quality: Are similar concepts nearby in vector space?
3. Check pre-filtering: Are you filtering by agent_id before vector search?

**Solutions**:
- Run grid search on benchmark scenarios (Day 5)
- Use scenario-specific weight overrides
- Add keyword filtering before vector search
- Test embeddings: query "danger" should return "fire alarm", not "breakfast"

---

### Issue: DuckDB vector search is slow
**Symptoms**: Retrieval takes >500ms per query

**Diagnosis**:
1. Check indexes: Is HNSW index created on embedding column?
2. Check table size: How many memories total?
3. Check query pattern: Searching across all agents or filtering first?

**Solutions**:
```sql
-- Ensure HNSW index exists
CREATE INDEX IF NOT EXISTS mem_vec_idx ON memories USING HNSW (embedding);

-- Pre-filter before vector search
SELECT * FROM memories 
WHERE agent_id = ? 
  AND ts > NOW() - INTERVAL '7 days'
ORDER BY embedding <=> ?::FLOAT[384]  -- Cosine distance
LIMIT 10;
```

---

### Issue: Colab disconnects during compilation
**Symptoms**: Notebook stops mid-compile, lose progress

**Solutions**:
- Use Colab Pro (longer runtime limits)
- Checkpoint every 30 minutes:
  ```python
  if iteration % 10 == 0:
      compiled.save(f'/content/drive/MyDrive/mini-town/checkpoint_{iteration}.json')
  ```
- Run compilation in smaller batches (10 seeds at a time)
- Have cloud GPU backup ready (Modal.com, $0.70/hour)

---

### Issue: LLM timeout causes agent crash
**Symptoms**: Agent stops updating, "confused" state never recovers

**Diagnosis**:
1. Check error logs: Is exception being caught?
2. Check recovery logic: Does agent retry on next tick?

**Solutions**:
```python
try:
    result = await asyncio.wait_for(llm_call(), timeout=5.0)
    agent.state = "active"
except asyncio.TimeoutError:
    logger.warning(f"Agent {agent.id} LLM timeout")
    agent.state = "confused"
    agent.current_plan = "Taking a moment to think..."
    # Use cached response or default behavior
    result = get_fallback_response(agent)
```

---

### Issue: All agents behave identically
**Symptoms**: No visible personality differences, boring to watch

**Diagnosis**:
1. Check personality injection: Are traits being passed to LLM prompts?
2. Check retrieval weights: All agents using default α,β,γ?
3. Check reflection thresholds: All agents reflecting at same rate?

**Solutions**:
- Verify personality context in prompts:
  ```python
  scorer(
      observation=obs,
      agent_goal=agent.goal,
      personality=agent.personality  # ← Must include this
  )
  ```
- Use agent-specific retrieval weights from PERSONALITIES dict
- Vary reflection_threshold (2.5 for impulsive, 5.0 for analytical)

---

## Git Workflow

### .gitignore (Important)
```
# Database (can be large, regenerate from seeds)
data/town.db
*.db

# Logs
logs/*.log
logs/*.jsonl

# Environment
.env
__pycache__/
*.pyc
.venv/
venv/

# Compilation outputs (large files, store in Drive)
compiled/*.json
compiled/*.pkl

# Node
node_modules/
.next/
out/

# OS
.DS_Store
Thumbs.db
```

### Commit Strategy
- **Day 0.5**: Commit hardcoded sim scaffold
- **Day 1**: Commit DuckDB schema + vector setup
- **Day 2**: Commit uncompiled DSPy integration
- **Day 3**: Commit seeds (important for reproducibility!)
- **Day 4**: Commit compiled programs (small JSON files OK)
- **Day 5-7**: Commit as features complete
- **Day 8**: Tag `v1.0-mvp` release

### What to Commit
- ✅ Source code (backend, frontend)
- ✅ Configuration (config.yml, requirements.txt)
- ✅ Seeds (critical for reproducibility)
- ✅ Documentation (README)
- ✅ Colab notebooks (as .ipynb files)
- ❌ Database files (regenerate from seeds)
- ❌ Large compiled models (store in Drive, link in README)
- ❌ Logs (ephemeral, regenerate)
- Any Claude related files

---

## Demo Video Script (Day 8-9)

### Scene 1: Introduction (30 seconds)
```
[Screen recording of map with 5 agents]

Voiceover: "This is Mini-Town: 5 autonomous agents that perceive, remember,
reflect, and plan using large language models. But here's the twist—their
prompts aren't handwritten. They're compiled programs, optimized against
measurable goals."

[Show SystemPanel: "Optimizer: GEPA, Score: 0.78"]
```

---

### Scene 2: Uncompiled Baseline (45 seconds)
```
[Reset simulation, show "Uncompiled Agents" overlay]

Voiceover: "First, let's see the baseline. Maria invites Alice and Bob to 
a party at 7pm. Watch what happens..."

[Fast-forward to 7pm, agents scattered around map]

Voiceover: "Only 1 out of 3 invited agents attended. Alice went shopping 
instead. Bob forgot about it. The uncompiled prompts don't prioritize 
social commitments."

[Show event coherence metric: "0.33 (33% attendance)"]
```

---

### Scene 3: Compiled Version (45 seconds)
```
[Reset simulation, show "Compiled Agents (GEPA)" overlay]

Voiceover: "Now the same scenario, but with compiled prompts. I trained
the GEPA optimizer on 40 examples of importance scoring and reflection quality."

[Fast-forward to 7pm, all 3 agents at party location]

Voiceover: "This time, all 3 agents attended. They correctly scored Maria's 
invitation as important, remembered it during planning, and showed up on time."

[Show event coherence metric: "1.00 (100% attendance)"]
[Show town_score comparison: "Uncompiled: 0.52 → Compiled: 0.78 (+50%)"]
```

---

### Scene 4: Agent Inspector (30 seconds)
```
[Click on Alice, show AgentInspector panel]

Voiceover: "Here's Alice's internal state. You can see her recent memories, 
the reflection she had about social connections, and her plan for the day. 
The compiled program generates more coherent, goal-aligned reasoning."

[Scroll through memories, highlight reflection: "Building relationships 
is important for feeling connected to the community"]
```

---

### Scene 5: Closing (30 seconds)
```
[Show GitHub repo, README]

Voiceover: "The entire project—including the compilation notebook, seeds, 
and evaluation code—is open source. The total cost? Less than 50 cents. 
All thanks to Colab Pro and strategic use of free tiers."

[Show cost breakdown table]

Voiceover: "This is just the beginning. Imagine compiled agents for 
customer service, game NPCs, or personal assistants. The future of 
LLM prompting isn't manual—it's compiled."

[Fade to repo link and project title]
```

**Total Runtime**: ~3 minutes

---

## Future Work (Post-MVP)

### Phase 2: Advanced Compilation (Week 2-3)
- [ ] Add SIMBA compilation for Reflect (harden on difficult cases)
- [ ] Add GRPO (RL-based) for end-to-end town_score optimization
- [ ] Compare GEPA vs MIPROv2 vs SIMBA vs GRPO (convergence speed, quality, cost)
- [ ] Publish comparison blog post with charts

### Phase 3: Richer Interactions (Month 2)
- [ ] Implement lightweight ReAct (max_iters=3-5) for PlanDay
- [ ] Add more tools: search_memory, send_message, schedule_meeting
- [ ] Multi-location support (town square, cafe, park)
- [ ] Agent-to-agent dialogue (not just broadcasts)

### Phase 4: Scale & Complexity (Month 3)
- [ ] Scale to 10-15 agents
- [ ] Add emergent scenarios (no hardcoded events)
- [ ] Memory graph visualization (Neo4j or D3.js)
- [ ] Agent learning from interactions (online compilation)

### Phase 5: Research Contributions (Month 4+)
- [ ] Write paper: "Compiled Generative Agents: Prompt Optimization for Autonomous NPCs"
- [ ] Open-source benchmark suite (seed collection, eval metrics)
- [ ] Public demo with live audience interaction
- [ ] Submit to NeurIPS Workshop on Language Gamification

---

## Lessons Learned (To Be Filled Post-Project)

### What Went Well
_(Fill this in after Day 10)_

- 
- 
- 

### What Was Harder Than Expected
_(Fill this in after Day 10)_

- 
- 
- 

### What I'd Do Differently Next Time
_(Fill this in after Day 10)_

- 
- 
- 

### Key Metrics Achieved
_(Fill this in after Day 10)_

- Town score improvement: __%
- Event coherence: __%
- Plan fidelity: __
- Total cost: $__
- Total time: __ hours

---

## Appendix: Quick Reference Commands

### Start Backend
```bash
cd backend
export GROQ_API_KEY="gsk_..."
uvicorn main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev  # Runs on localhost:3000
```

### DuckDB CLI
```bash
duckdb data/town.db
> SELECT COUNT(*) FROM memories;
> SELECT name, x, y, state FROM agents;
> .exit
```

### Analyze Logs
```bash
# Count reflections per agent
cat logs/agent_events.jsonl | jq -r 'select(.event=="reflection") | .agent_id' | sort | uniq -c

# Average importance scores
cat logs/agent_events.jsonl | jq -r 'select(.event=="score") | .data.score' | awk '{sum+=$1; n++} END {print sum/n}'

# Top 10 slowest LLM calls
cat logs/mini_town.log | grep "elapsed" | sort -t'=' -k2 -n | tail -10
```

### Run Compilation (Colab)
```python
# In Colab cell
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/mini-town/compilation
%run compile_scorer.ipynb
```

### Deploy Frontend (Vercel)
```bash
cd frontend
vercel login
vercel  # Follow prompts, deploys to *.vercel.app
```

---

## Glossary

**Agent**: An autonomous NPC with perception, memory, reflection, planning, and action capabilities.

**DSPy**: A framework for treating prompts as typed programs that can be compiled/optimized.

**Signature**: A typed input/output specification for an LLM call (like a function signature).

**Module**: An implementation of a Signature (Predict, ChainOfThought, ReAct).

**Compilation**: The process of optimizing prompts against a metric using training data (seeds).

**MIPROv2**: A DSPy optimizer that jointly tunes instructions and few-shot examples.

**GEPA**: A 2025 DSPy optimizer using reflective prompt evolution (fewer rollouts than MIPROv2).

**Seed**: A training example for compilation (e.g., observation → gold score).

**Triad Scoring**: Retrieval scoring that combines relevance, recency, and importance (α, β, γ weights).

**Event Coherence**: Metric measuring whether invited agents attend events (% attendance).

**Plan Fidelity**: Metric measuring how well agents follow their stated plans (edit distance).

**Town Score**: Combined metric for compilation (weighted sum of coherence, fidelity, hit-rate).

**God Mode**: Debugging interface for pausing sim, injecting events, and inspecting agents.

**Tick**: One cycle of the simulation loop (default: 2 seconds).

**Confused State**: Agent state when LLM call times out or returns invalid response.

**Reflection**: Process where agent synthesizes high-level insights from recent memories.

**Note**: A stored insight/reflection in the knowledge graph.

---

## Version History

**v0.1 (Planning Phase)** - January 2025
- Initial project plan completed
- CLAUDE.md drafted
- Tech stack finalized

**v1.0 (Target: End of Day 10)** - TBD
- MVP complete with 5 agents
- At least one compiled DSPy program
- Measurable improvement in town_score
- Demo video published

**v2.0 (Future)** - TBD
- Multiple optimizer comparison
- ReAct-based planning
- 10-15 agents
- Public demo

---

## Contact & Attribution

**Project Creator**: [Your Name/Handle]  
**GitHub**: [Repo URL]  
**License**: MIT  
**Created**: January 2025  

**Acknowledgments**:
- DSPy team (Stanford NLP) for the compilation framework
- Park et al. for Generative Agents architecture (Smallville paper)
- Groq for generous free tier
- Colab Pro for GPU access
- Claude (Anthropic) for project planning assistance

---

## Final Notes for AI Assistants

When helping with this project:

1. **Always read this document first** - It contains critical context about budget constraints, architecture decisions, and known pitfalls.

2. **Reference specific sections** - Don't reinvent solutions. If the plan says "use Groq for dev, Together.ai for demo", respect that architecture decision.

3. **Respect the critical path** - Day 0.5 and Day 3 are make-or-break. Don't suggest skipping validation steps.

4. **Consider budget constraints** - Every suggestion should keep the project under $5. Don't recommend paid services without cheaper alternatives.

5. **Align with milestones** - If the user is on Day 2, don't give Day 6 code. Stay in sync with timeline.

6. **Explain tradeoffs** - When suggesting changes, explain impact on time/cost/complexity.

7. **Test suggestions** - If possible, mentally trace through the code to catch obvious bugs.

8. **Encourage iteration** - Remind the user that first attempts (seeds, metrics, weights) won't be perfect. That's expected.

9. **Celebrate progress** - Building a compiled agent system in 10 days on a $5 budget is genuinely impressive. Acknowledge milestones.

10. **Stay practical** - Favor working code over perfect code. MVP first, optimization later.

---

**This document should be updated** as the project progresses. Capture lessons learned, actual metrics, and deviations from the plan. It will be valuable for future projects and for sharing with the community.

**Good luck, and may your agents be ever coherent! 🤖✨**