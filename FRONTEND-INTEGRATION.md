<!-- Updated 2025-10-13 after FastAPI ⇄ AI Town bridge refactor -->

# FRONTEND-INTEGRATION.md

## Purpose

Document the new Mini‑Town UI stack and its contract with the FastAPI/DSPy backend. The AI Town Pixi UI now talks directly to FastAPI over REST + WebSocket (`/ai-town/*`, `/ws`); Convex is no longer in the loop.

---

## Stack Overview

- **Frontend:** `ai-town/` Vite + React 18 + Pixi (`@pixi/react`, `pixi-viewport`), TypeScript.
- **Backend:** `backend/` FastAPI + DuckDB + DSPy, exposing REST under `/ai-town/*` and WebSocket `/ws`.
- **Assets:** AI Town sprites/tilesheets served from `ai-town/public/assets`.
- **Build/test:** `npm run build` (tsc -p tsconfig.app.json + vite build), `npm run dev` (Vite dev server).

---

## Directory Layout (frontend)

```
ai-town/
  src/
    App.tsx                     # Shell with header + content layout
    main.tsx                    # React root (imports CSS only)
    components/
      Game.tsx                  # Top-level control-room layout
      PixiGame.tsx              # Map viewport wrapper
      PixiStaticMap.tsx         # Tile renderer (sprites + background)
      Character.tsx             # Animated sprite wrapper
      MiniTownSystemPanel.tsx   # Status + events + plan summaries
      MiniTownAgentInspector.tsx# Polls agent details via REST
      MiniTownGodModeControls.tsx# Pause/step/inject/refresh controls
      MiniTownDemoControls.tsx  # Interactive demo panel for presets/landmarks
    hooks/
      useMiniTownWebSocket.ts   # WS state hook (agents/system/world/worldMap)
    lib/
      minitownConfig.ts         # API/WebSocket base URL helpers
      minitownTypes.ts          # Shared TS types
      worldMap.ts               # WorldMap class + serialization types
    styles/
      ai-town-theme.css         # Palette + layout utilities
  public/assets/                # Sprites, tilesets, audio
  tsconfig.json                 # Base compiler options
  tsconfig.app.json             # Build-specific include/exclude
```

---

## Data Contracts

### WebSocket (`ws://<api>/ws`)

Messages emitted ~every tick (2 s default):

```ts
type MiniTownEvent = {
  id: string;
  type: string;
  timestamp: string;
  severity?: number;
  location?: [number, number];
};

type MiniTownObservation = {
  text: string;
  importance: number;
  score?: number;
  reasoning?: string | null;
  timestamp?: string;
};

type MiniTownAgent = {
  id: number;
  agentId: string;          // e.g., "a:1"
  name: string;
  x: number;
  y: number;
  state: string;
  goal?: string;
  personality?: string;
  current_plan?: string | null;
  plan_source: string;
  plan_last_updated?: string | null;
  plan_preset_id?: string | null;
  observations?: MiniTownObservation[];
};

type MiniTownWorld = {
  worldId: string;
  tick: number;
  lastTickAt?: string | null;
  players: Array<{
    playerId: string;
    agentId: number;
    name: string;
    state: string;
    position: { x: number; y: number };
    goal?: string;
    personality?: string;
  }>;
  events: MiniTownEvent[];
};

type MiniTownWorldMap = {
  width: number;
  height: number;
  tileSetUrl: string;
  tileSetDimX: number;
  tileSetDimY: number;
  tileDim: number;
  bgTiles: number[][][];
  objectTiles: number[][][];   // same shape as bgTiles
  animatedSprites: Array<{ x: number; y: number; w: number; h: number; layer: number; sheet: string; animation: string; }>;
};

type InitMessage = {
  type: 'init';
  agents: MiniTownAgent[];
  system: MiniTownSystemState;
  world?: MiniTownWorld;
  worldMap?: MiniTownWorldMap;
  config: { map_width: number; map_height: number };
};

type AgentsUpdateMessage = {
  type: 'agents_update';
  agents: MiniTownAgent[];
  tick?: number;
  timestamp?: string;
  world?: MiniTownWorld;
};

type SystemUpdateMessage = { type: 'system_update'; state: MiniTownSystemState };
type EventMessage = { type: 'event'; event: MiniTownEvent; world?: MiniTownWorld };
```

### REST (`https://<api>/ai-town/*`)

| Endpoint | Method | Payload / Response |
|----------|--------|--------------------|
| `/ai-town/state` | GET | `{ world, agents, agentDescriptions, system, worldMap }` |
| `/ai-town/agents` | GET | `{ agents, agentDescriptions }` |
| `/ai-town/system` | GET | `MiniTownSystemState` (includes `tick`, `last_tick_at`, `recent_events`) |
| `/ai-town/world` | GET | `MiniTownWorld` |
| `/ai-town/map` | GET | `MiniTownWorldMap` (same schema as WebSocket) |
| `/ai-town/control/landmarks` | GET | `{ landmarks: [{ id, name, description, x, y }] }` |
| `/ai-town/control/presets` | GET | `{ metadata, presets: { agent_id: [{ id, label, summary, landmark_id }] } }` |
| `/ai-town/control/select_agent` | POST `{ agent_id }` → `{ selected_agent_id, agent }` |
| `/ai-town/control/apply_plan` | POST `{ agent_id, preset_id }` → `{ status, agent, preset }` |
| `/ai-town/control/teleport` | POST `{ agent_id, landmark_id }` → `{ status, agent, landmark }` |
| `/api/agents/{id}` | GET | Detailed agent view + memories + reflection (`serialize_agent_detail`) |
| `/ai-town/god/pause` | POST `{ paused?: boolean }` → `{ status: 'paused' | 'resumed' }` |
| `/ai-town/god/step` | POST → `{ status: 'stepped' }` |
| `/ai-town/god/inject_event` | POST `{ type: string; severity?: number; location?: [number, number] }` → `{ status, event }` |
| `/ai-town/god/refresh_plan` | POST `{ agent_id?: number }` → `{ refreshed: number, agents: [{ agent_id, plan, plan_source, plan_last_updated }] }` |

Legacy endpoints `/god/*` and `/api/system` remain for backward compatibility but all new UI code targets `/ai-town/*`.

`MiniTownSystemState` fields (subset of `/ai-town/system` response):

```ts
type MiniTownSystemState = {
  llm_provider: string;
  llm_model: string;
  optimizer: string;
  town_score: number;
  avg_latency: number;
  tick_interval: number;
  paused: boolean;
  recent_events: MiniTownEvent[];
  tick: number;
  last_tick_at?: string | null;
};
```

---

## Key Components & Hooks

- `useMiniTownWebSocket`  
  Handles WS lifecycle + reconnection, returns `{ agents, system, events, world, worldMap, config, connected, tick, lastMessageAt, error }`.  
  Stores latest `worldMap` from `init` and each `agents_update`.

- `Game.tsx`  
  Glues everything together. Tracks map DOM size, selected agent, connection badges. Converts `worldMap` JSON into a `WorldMap` instance and passes to Pixi renderer.

- `PixiGame.tsx` / `PixiStaticMap.tsx`  
  Expect a constructed `WorldMap` plus simulation bounds (`config.map_width/height`). Translate simulation coordinates → pixel coordinates using map tile metadata.

- `MiniTownAgentInspector.tsx`  
  Polls `/api/agents/{id}` every 2 seconds. Displays plan source, last updated (relative), memories with importance badges, reflection snippet.
  Now also surfaces the active plan step (description, time window, coordinates) and recent observation summaries.

- `MiniTownGodModeControls.tsx`  
  Uses `fetch` to POST `/ai-town/god/*`. Shows toasts on success/failure. Refresh button optionally targets a specific agent ID.

- `MiniTownDemoControls.tsx`  
  New demo panel that surfaces `/ai-town/control/*` endpoints. Lets a user pick an agent, snap to a landmark, and apply GEPA-compiled plan presets.

- `MiniTownSystemPanel.tsx`  
  Renders connection state, latency, tick interval, recent events (limit 5) and a plan summary list sorted by last updated.

---

## Running Locally

```bash
# Backend
cd /Users/nainy/Documents/Personal/mini-town/backend
../mini-town/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd /Users/nainy/Documents/Personal/mini-town/ai-town
npm install          # first run only (updates package-lock.json)
npm run dev          # http://localhost:5173 by default
```

Environment overrides:

```
# ai-town/.env.local (optional)
VITE_MINITOWN_API=http://127.0.0.1:8000
```

WebSocket URL derived automatically (`ws://127.0.0.1:8000/ws`).

---

## Testing / Validation Checklist

1. `npm run build` (tsc + vite) → success.  
2. With backend running, open dev server:
   - Agents render on Pixi map, click selects and highlights sprite.
   - System panel shows tick, latency, plan summaries.
   - Inspector updates every 2 s; plan metadata reflects `/ai-town/god/refresh_plan`.
   - God Mode controls call FastAPI endpoints (`/ai-town/god/*`) and broadcast updates.
   - WebSocket reconnects automatically if backend restarts.
3. Optional: hit `/ai-town/state` in browser/`curl` to inspect combined payload.

---

## UI Enhancement Signatures

As of October 2025, the backend includes three DSPy signatures that enhance the frontend with contextual insights:

- **`PlanStepExplainer`**: Generates 1-2 sentence explanations for why an agent's current plan step matters. Triggered when plan steps change.
- **`ObservationSummarizer`**: Synthesizes high-importance observations into 2-3 thematic bullet points. Generated periodically every 10 ticks.
- **Conversation Tracking**: Tracks conversation events with timestamps, participants, and step context.

These signatures populate new fields in agent/system payloads:
- `agent.step_explanation` (string)
- `agent.observation_summary` (string)
- `agent.recent_conversations` (array of conversation digests)
- `agent.active_plan_step.explanation` (string)
- `systemState.observation_rollups` (array of observation summaries)

The frontend components (`MiniTownAgentInspector`, `MiniTownSystemPanel`) automatically display these fields when available, falling back gracefully when data is missing.

**For detailed documentation**, see `UI_SIGNATURE_INTEGRATION.md` for implementation details, seed data format, compilation instructions, and performance considerations.

---

## Notes & Future Work

- Convex code remains in the repo for reference but is excluded from TypeScript builds (`tsconfig.app.json`). Removing it entirely is safe once archives are stored.
- Map data (`backend/data/gentle_map.json`) is generated from upstream AI Town and served verbatim; replace if you adopt different tilesets.
- Consider splitting the large Vite JS bundle (>500 kB) using dynamic imports when adding more UI features.
- UI enhancement signatures can be compiled using `compilation/compile_ui_signatures.py` for optimized performance.
