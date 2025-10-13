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
interface AgentMsg { id:number; name:string; x:number; y:number; state:'active'|'confused'|'idle'; }
interface AgentsUpdate { type:'agents_update'; agents: AgentMsg[] }

// system_update
interface SystemState { llm_provider:string; llm_model:string; optimizer:string; town_score:number; avg_latency:number; tick_interval:number }
interface SystemUpdate { type:'system_update'; state: SystemState }
```

### REST Endpoints

- GET /api/agents/{agent_id}
```json
{
  "id": 1,
  "name": "Alice",
  "state": "active",
  "personality": "social, optimistic",
  "goal": "Build relationships in the neighborhood",
  "current_plan": "10:30 Wave to Bob\n11:00 Visit Maria",
  "memories": [
    { "id": 101, "ts": "2025-01-02T10:29:00Z", "content": "Bob waved hello", "importance": 0.7 }
  ],
  "latest_reflection": "I'm building connections with neighbors"
}
```

- POST /god/pause → { status:"paused" }
- POST /god/step → { status:"stepped" }
- POST /god/inject_event { type:string, severity?:number, location?:[number,number] } → { status:"injected" }

## Components

### SystemPanel.tsx

- Shows optimizer, model, town_score, latency, tick.
- Props: `systemState: SystemState` (from WS `system_update`).

### Map.tsx

- Renders agents on a canvas or absolutely-positioned layer.
- Props: `agents: AgentMsg[]`, `onAgentClick: (id:number)=>void`.

### AgentInspector.tsx

- Right-docked panel for selected agent.
- Fetches GET /api/agents/{id} (SWR 2s refresh).
- Sections: State/Goal/Personality, Current Plan (mono block), Recent Memories (top 5 with importance badges), Latest Reflection.

### GodMode.tsx

- Floating control: Pause / Step / Inject Event.
- Calls POST /god/* endpoints; shows simple toasts.

## WebSocket Hook (lib/websocket.ts)

```ts
export function useWebSocket(url:string){ /* open WS; route agents_update/system_update into state; return {agents,systemState} */ }
```

## Integration Steps (Day 0.5 scaffold → polished UI)

1. Ensure `frontend/` Next.js app exists; `npm install next react react-dom swr`.
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

- [ ] Add ai-town-theme.css and import globally
- [ ] Implement useWebSocket and wire system/agents updates
- [ ] Create SystemPanel showing optimizer/model/score/latency/tick
- [ ] Implement AgentInspector fetching GET /api/agents/{id}
- [ ] Add GodMode controls calling /god endpoints
- [ ] Expose WS + REST endpoints matching the documented shapes


