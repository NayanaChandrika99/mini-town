<!-- 3c0f7735-2d51-4765-b954-dc5ef98e5c6c 3f170824-81fb-4b90-8f7e-58c044e9774e -->
# FRONTEND-INTEGRATION.md

## Purpose

This document specifies the Mini‑Town frontend: UI architecture, components, styling, and integration contracts with the existing FastAPI/DuckDB/DSPy backend. It borrows UI patterns from AI Town while keeping your current backend intact.

Reference: [a16z ai-town](https://github.com/a16z-infra/ai-town)

## UI Architecture

- Map-centric layout with right-docked inspector, top status bar, and floating god‑mode controls.
- Live updates via WebSocket; inspector/debug via REST.
- Tech: Next.js 14 (TypeScript), React, SWR, CSS variables (no framework required).

## Directory Layout

```
frontend/
  pages/
    index.tsx              # Main layout (Map + SystemPanel + Inspector + GodMode)
  components/
    Map.tsx
    AgentInspector.tsx
    SystemPanel.tsx
    GodMode.tsx
  lib/
    websocket.ts           # WS hook and message handling
  styles/
    ai-town-theme.css      # Tokens + shared styles
```

## Styling (AI Town-inspired)

Create styles/ai-town-theme.css and import in pages/_app.tsx.

```css
:root {
  --bg-primary:#0a0e12; --bg-secondary:#0f1419; --bg-panel:#161b22; --bg-hover:rgba(255,255,255,0.05);
  --text-primary:#e6edf3; --text-secondary:#8b949e; --text-muted:#6e7681;
  --accent-blue:#58a6ff; --accent-green:#3fb950; --accent-yellow:#d29922; --accent-red:#f85149;
  --border-default:rgba(240,246,252,0.1); --border-muted:rgba(240,246,252,0.05);
  --radius-sm:6px; --radius-md:10px; --radius-lg:16px;
}
body { background:var(--bg-primary); color:var(--text-primary); margin:0; font:14px/1.5 Inter,sans-serif; }
.panel { background:var(--bg-panel); border:1px solid var(--border-default); border-radius:var(--radius-md); }
.badge { padding:2px 8px; border-radius:12px; font:600 11px/1 Inter; text-transform:uppercase; }
.badge-success{background:rgba(63,185,80,0.15);color:var(--accent-green)}
.badge-warning{background:rgba(210,153,34,0.15);color:var(--accent-yellow)}
.badge-error{background:rgba(248,81,73,0.15);color:var(--accent-red)}
.badge-info{background:rgba(88,166,255,0.15);color:var(--accent-blue)}
```

## Data Contracts

### WebSocket (ws://<backend>/ws)

Messages sent by backend at ~2 Hz:

```ts
// agents_update
interface AgentMsg {
  id:number;
  name:string;
  x:number;
  y:number;
  state:'active'|'confused'|'idle'|'waiting';
  observations?:string[];
  current_plan?:string | null;
  plan_source?:'baseline'|'compiled'|'fallback'|string | null;
  plan_last_updated?:string | null; // ISO timestamp
}
interface AgentsUpdate {
  type:'agents_update';
  agents: AgentMsg[];
  tick:number;
  timestamp:string;
}

// system_update
interface SystemState {
  llm_provider:string;
  llm_model:string;
  optimizer:string;
  town_score:number;
  avg_latency:number;
  tick_interval:number;
  paused:boolean;
  recent_events:{ id:string; type:string; timestamp:string; severity?:number; location?:[number,number] }[];
}
interface SystemUpdate { type:'system_update'; state: SystemState }

// init (first payload on connect)
interface InitMessage {
  type:'init';
  agents: AgentMsg[];
  system: SystemState;
  config: { map_width:number; map_height:number };
}

// event (broadcast when God Mode injects)
interface EventBroadcast {
  type:'event';
  event:{
    id:string;
    type:string;
    timestamp:string;
    severity?:number;
    location?:[number,number];
  };
}
```

### REST Endpoints

- GET /api/agents/{agent_id}
```json
{
  "id": 1,
  "name": "Alice",
  "x": 210.4,
  "y": 142.7,
  "state": "active",
  "personality": "social, optimistic",
  "goal": "Build relationships in the neighborhood",
  "current_plan": "10:30 AM - 11:00 AM: Wave to Bob at (210, 145)\n11:15 AM - 11:45 AM: Visit Maria at (180, 120)",
  "plan_source": "compiled",
  "plan_last_updated": "2025-10-13T17:20:11.432Z",
  "memories": [
    { "id": 101, "ts": "2025-01-02T10:29:00Z", "content": "Bob waved hello", "importance": 0.7 }
  ],
  "latest_reflection": "I'm building connections with neighbors"
}
```

- GET /api/system → latest `SystemState`
- POST /god/pause → { status:"paused" }
- POST /god/step → { status:"stepped" }
- POST /god/inject_event { type:string, severity?:number, location?:[number,number] } → { status:"injected" }
- POST /god/refresh_plan { agent_id?:number } → { refreshed:number, agents:[{agent_id,plan,plan_source,plan_last_updated}] }

## Components

### TownView.tsx

- High-level layout shell (header + map + sidebar) inspired by AI Town.
- Composes `MapCanvas`, `SystemPanel`, `AgentInspector`, and `GodModeControls`.
- Handles agent selection state, connection indicators, and layout spacing.

### MapCanvas.tsx

- Canvas-based map renderer for agents.
- Props: `agents: AgentMsg[]`, `onAgentClick: (id:number)=>void`.
- Attempts to draw `/assets/32x32folk.png` sprites; falls back to colored circles.
- Highlights `state:'alert'` agents in warning color when responding to events.

### SystemPanel.tsx
- Shows optimizer/model/town_score/latency/tick, recent events, and a condensed list of active plans (source + freshness).
- Props: `systemState: SystemState`, `events: TownEvent[]`, `agents: AgentMsg[]`, connection metadata.

### AgentInspector.tsx
- Sidebar card showing selected agent details.
- Polls GET /api/agents/{id} every 2 seconds.
- Sections: status, goal, personality tags, current plan (with source/last updated chips and line-by-line breakdown), memory list, latest reflection.

### GodModeControls.tsx
- Pause / Step / Inject Event / Refresh Plans controls with toast feedback.
- Calls POST `/god/*` endpoints; plan refresh hits `/god/refresh_plan` (all agents by default or targeted by ID).
- Event injections broadcast immediately via WebSocket.

## WebSocket Hook (lib/websocket.ts)

```ts
export function useWebSocket(url:string){ /* open WS; route agents_update/system_update into state; return {agents,systemState} */ }
```

## Integration Steps (Day 0.5 scaffold → polished UI)

1. Ensure `frontend/` Next.js app exists; install dependencies with `npm install`.
2. Add `styles/ai-town-theme.css`; import it in `_app.tsx`.
3. Implement `lib/websocket.ts` and point to `ws://localhost:8000/ws` (env-driven in prod).
4. Build `components/` listed above; wire them in `pages/index.tsx`.
5. Backend: expose `/ws`, `/api/agents/{id}`, `/god/pause`, `/god/step`, `/god/inject_event` using shapes above.
6. Verify CORS/WS: allow frontend origin; test with `npm run dev` and backend `uvicorn` running.

## Dev & Run

- Backend: `uvicorn main:app --reload --port 8000` (ensure WS + endpoints working).
- Frontend: `cd frontend && npm run dev` → open http://localhost:3000.

## Testing Checklist

- Map shows 5 agents moving; clicking opens Inspector.
- SystemPanel updates town_score and latency every second.
- GodMode pause/step/inject work; Inspector reflects changes.
- No failed WS reconnect loops; REST returns 200s.

## Notes

- Keep AI Town aesthetics (dark palette, pills, rounded panels) while retaining FastAPI/DuckDB/DSPy.
- Do not import Convex/Clerk; UI style only.
- Reference: [a16z ai-town](https://github.com/a16z-infra/ai-town)

### To-dos

- [x] Add ai-town-theme.css and import globally
- [x] Implement useWebSocket and wire system/agents updates
- [x] Create SystemPanel showing optimizer/model/score/latency/tick
- [x] Implement AgentInspector fetching GET /api/agents/{id}
- [x] Add GodMode controls calling /god endpoints
- [x] Expose WS + REST endpoints matching the documented shapes
