# DSPy × AI Town Integration – Execution Plan

> **2025-10-13 Update:** Convex integration is now deprecated in favour of a direct FastAPI ⇄ Pixi bridge. The Mini‑Town backend exposes `/ai-town/*` REST endpoints + `/ws`; the `ai-town/` Vite frontend consumes them via `useMiniTownWebSocket` and fetch helpers. The sections below remain for historical context, but the active plan focuses on:
>
> - Maintaining FastAPI as the single source of truth (DuckDB + DSPy agents).
> - Delivering AI Town’s Pixi UI from the `ai-town/` directory (no Convex runtime).
> - Gradually removing unused Convex/Clerk code as we stabilise the new bridge.
>
> **Bridge status:**  
> ✅ `/ai-town/state`, `/ai-town/agents`, `/ai-town/map`, `/ai-town/god/*` implemented.  
> ✅ WebSocket now broadcasts `{ agents, world, worldMap, system }`.  
> ✅ Frontend hooks/components ported to REST/WS (`useMiniTownWebSocket`, Pixi map, inspector, God Mode).  
> ⚠️ Remaining Convex code lives in `convex/` for reference but is excluded from builds (`tsconfig.app.json`).  
> 📌 Next docs to update: `plan.md`, `DAY7_ENHANCEMENTS_PLAN.md`, error log (capture new testing steps).


**Owner:** Mini-Town engineering  
**Last Updated:** 2025-10-13  
**Goal:** Keep the Mini-Town DSPy backend while borrowing the AI Town template (Convex + React) and bolting on a reusable GEPA-compiled “brain” service.

---

## 0. Snapshot of the Current State
- ✅ Mini-Town backend (FastAPI + DuckDB + DSPy) runs locally; compiled modules live under `compiled/`.
- ✅ AI Town repo cloned for reference; front-end components already inspired by `TownView`, `MapCanvas`, `AgentInspector`.
- ⚠️ DSPy compiled planner exists, but there is no network service exposing it to external clients.
- ⚠️ No bridge from AI Town’s Convex actions to the DSPy modules.
- ⚠️ No shared plan on how to mix-and-match these systems in production.

---

## 1. High-Level Phases
1. **GEPA Optimizer Service (Python/FastAPI)**  
   - Standalone service that loads compiled DSPy modules and exposes endpoints for compile-time (optional) and runtime inference.
2. **AI Town Backend Integration (Convex)**  
   - Convex actions call the optimizer service for plan/scoring requests; outputs stored back in Convex tables.
3. **Front-End Alignment (React/Pixi)**  
   - Use AI Town UI while pointing data hooks at Convex tables that now contain DSPy-driven decisions.
4. **Validation & Ops**  
   - End-to-end tests, observability, deployment hardening.

Each phase is modular; you can pause after any milestone and still have a working system.

---

## 2. Detailed Work Breakdown

### Phase 1 – Build the DSPy GEPA Optimizer Service
| Status | Task | Notes |
|--------|------|-------|
| ✅ | 1.1 Bootstrap repo `dspy_optimizer/` | `python -m venv venv`, freeze requirements (`dspy-ai`, `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`). |
| ✅ | 1.2 Implement loader for compiled modules | Load JSON artifacts from `/Users/nainy/Documents/Personal/mini-town/compiled/`, cache per module. |
| ✅ | 1.3 Add runtime inference endpoint | `POST /planner/plan_day`, `POST /scorer/score_importance`, etc. Accept structured payloads, return decisions. |
| ☐ | 1.4 (Optional) Add `/compile` endpoint for GEPA runs | Uses Colab-generated datasets; gated behind auth. |
| ✅ | 1.5 Wire logging + health check | `/healthz`, request/response tracing, env-var validation on startup. |
| ✅ | 1.6 Local test | Manual smoke via `scripts/optimizer_smoke.py`; formal pytest suite TBD. |

### Phase 2 – Convex Integration
| Status | Task | Notes |
|--------|------|-------|
| ✅ | 2.1 Create Convex helper for optimizer calls | Implemented in `convex/util/optimizer.ts` with shared fetch + retries. |
| ✅ | 2.2 Add storage table `compiledBrains` | Stores cached optimizer responses with metadata + indexes. |
| ✅ | 2.3 Wire planner usage | `agentRefreshPlan` reads cache, bypasses optimizer when possible, upserts on refresh. |
| ✅ | 2.4 Wire memory scorer | `calculateImportance` now uses optimizer (with cache + fallback) for memory scoring. |
| ✅ | 2.5 Add manual trigger mutation | `testing:refreshAgentPlan` action triggers optimizer refresh on demand. |
| ☐ | 2.6 Update Convex deployment config | Store `DSPY_OPTIMIZER_URL`, auth tokens in `convex/.env` and production dashboard. |

### Phase 3 – Front-End Alignment
| Status | Task | Notes |
|--------|------|-------|
| ✅ | 3.1 Update UI to surface DSPy plans | SystemPanel + Inspector consume streamed plan metadata and show summaries. |
| ✅ | 3.2 Add admin controls | God Mode button hits `/god/refresh_plan` to regenerate plans (single or bulk). |
| ✅ | 3.3 Visual instrumentation | Plan source/last-updated badges and memory importance indicators in Inspector/SystemPanel. |
| ☐ | 3.4 Regression pass on layout | Verify sprites, selection, schedule display still work. |

### Phase 4 – Validation & Ops
| Status | Task | Notes |
|--------|------|-------|
| ☐ | 4.1 End-to-end script | Vercel/Convex + optimizer service running; ensure agents follow compiled plan. |
| ☐ | 4.2 Load/latency tests | Measure optimizer response time; add caching if needed. |
| ☐ | 4.3 Documentation | Update `README.md`, `plan.md`, `error_log.md` with new architecture and operational runbooks. |
| ☐ | 4.4 Deployment | Dockerize optimizer, deploy to Fly.io/GCloud Run; update Convex env vars. |

---

## 3. Immediate Next Moves (Week 1)
1. Confirm inventory of compiled artifacts (`compiled/*.json`), decide which modules need endpoints.  
2. Stand up the FastAPI skeleton with health check and static response (Phase 1 tasks 1.1–1.3).  
3. Draft Convex helper to call the optimizer (Phase 2 task 2.1) and test against the local service.  
4. Log outputs and compare to current AI Town behaviour to validate DSPy responses.  
5. Iterate on Phase 2 integration before touching the React layer.

---

## 4. Dependencies & Risks
- **DuckDB lock contention:** kill stray Python processes (`/opt/anaconda3/...`) before running backend tests.
- **LLM API budget:** GEPA compilation is expensive; reuse compiled JSON artifacts whenever possible.
- **Schema divergence:** AI Town’s agent schema must match DSPy inputs; add adapters to bridge naming/field differences.
- **Security:** Expose optimizer service only behind auth; never store API keys in Convex code.

---

## 5. Definition of Done
- Optimizer service responds to planner/scorer requests using compiled DSPy modules.
- Convex agents consume those outputs during simulation, with logging proving DSPy decisions were applied.
- Front-end UI surfaces DSPy-driven data (plans, conversations) without regressions.
- Documentation covers setup, deployment, and fallback procedures.
- Error logs and monitoring capture optimizer failures with actionable messages.

---

## 6. Tracking & Updates
- Maintain progress in `error_log.md` (failures) and `DAY7_ENHANCEMENTS_PLAN.md` (feature status).
- Update this document at the start/end of each work session; bump `Last Updated` date.
- Use checkboxes above to record completion; keep commits scoped per phase.

---

## 7. Progress Log (2025-10-13)
- **Optimizer service skeleton** (`dspy_optimizer/`):
  - FastAPI app with planner/scorer endpoints backed by compiled DSPy modules and graceful fallback (`dspy_optimizer/main.py`).
  - Pydantic settings and cached loader for compiled artifacts (`dspy_optimizer/config.py`, `dspy_optimizer/loader.py`).
  - Request/response schemas, README, and a smoke-test script under `scripts/optimizer_smoke.py`.
  - Dependency pins updated in `requirements.txt`; DSPy provider wiring adjusted in `backend/dspy_modules.py`.
- **Convex bridge** (`ai-town/convex/`):
  - Optimizer client helpers (`util/optimizer.ts`) to POST plans/scores with env-based configuration.
  - New queries/mutation to fetch and update agent descriptions (`aiTown/agent.ts`).
  - `agentRefreshPlan` internal action to call the optimizer and persist returned plans (`aiTown/agentOperations.ts`).
  - Cached plan + importance results via `compiledBrains` table and supporting helpers (`schema.ts`, `dspy/cache.ts`), including planner reuse and memory scoring through the optimizer with fallback heuristics (`agentOperations.ts`, `agent/memory.ts`).
- **Frontend & Controls** (`frontend/components/`):
  - SystemPanel now lists active plans with source/recency; AgentInspector shows plan metadata + formatted steps.
  - GodMode gains a “Refresh Plans” action wired to `/god/refresh_plan`; notifications surface success/failure.
  - WebSocket payloads carry plan/source/updated timestamps so the UI reacts in real time.
- **Interactive demo upgrade** (2025-10-14):
  - Faster loop (tick 1s, speed 4px) for responsive navigation.
  - Named landmarks catalog + presets loader (`backend/landmarks.py`, `backend/presets.py`) fed from GEPA compiled output (`compiled/presets/plans.json`).
  - New control endpoints (`/ai-town/control/*`) for agent selection, plan application, and landmark teleports.
  - Frontend demo panel (`MiniTownDemoControls.tsx`) consumes the API to drive curated DSPy scenarios.
- **Simulation clock & DSPy conversations** (2025-10-14):
  - Introduced configurable simulation clock (`minutes_per_tick`) so plans execute immediately after assignment.
  - Agents keep active step metadata and UI now surfaces the current block.
  - Added DSPy-powered `PlanConversation` signature hook: agents meeting on the same step trigger generated dialogue that feeds memories, events, and UI bubbles.

---

## 8. Operations Runbook (Local Dev)

1. **Start DSPy optimizer**
   ```bash
   cd /Users/nainy/Documents/Personal/mini-town
   source mini-town/bin/activate
   uvicorn dspy_optimizer.main:app --reload --port 8001
   ```
   - Leave `DSPY_OPT_AUTH_TOKEN` unset during local work to skip auth.
   - Stop with `Ctrl+C`.

2. **Launch Convex backend (with optimizer)**
   ```bash
   cd /Users/nainy/Documents/Personal/mini-town/ai-town
   npm run dev:stack
   ```
   - `dev:stack` starts the optimizer and Convex backend together via `concurrently`.
   - Note: The optimizer is a separate FastAPI process; it cannot be launched from the browser UI. Running this script is the supported way to ensure both services are alive.
   - To run just the backend (no optimizer), use:
   ```bash
   cd /Users/nainy/Documents/Personal/mini-town/ai-town
   npm run dev:backend -- --typecheck=disable
   ```
   - Keeps typecheck off to avoid noise from the upstream template.

3. **Inspect world + agent data**
   ```bash
   WORLD_ID=$(npx convex run world:defaultWorldStatus | jq -r '.worldId')
   npx convex run world:gameDescriptions "{\"worldId\":\"$WORLD_ID\"}" | jq '.agentDescriptions'
   ```

4. **Force a plan refresh (optional)**
   ```bash
   AGENT_ID=<copy from agentDescriptions>
   npx convex run testing:refreshAgentPlan "{\"agentId\":\"$AGENT_ID\"}"
   ```
   - Alternatively, from the Next.js UI use the God Mode “Refresh Plans” control (POST `/god/refresh_plan`).

5. **Verify plan output**
   ```bash
   npx convex run world:gameDescriptions "{\"worldId\":\"$WORLD_ID\"}" | jq '.agentDescriptions'
   ```

6. **Shutdown order**
   - `Ctrl+C` in the Convex terminal.
   - `Ctrl+C` in the optimizer terminal.
