    # AI Town Integration Guide for Mini-Town Project

**Version**: 1.0  
**Date**: October 13, 2025  
**Purpose**: Strategic extraction of UI/UX, agent logic, and assets from a16z AI Town while preserving Mini-Town's DSPy-focused architecture

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Integration Philosophy](#integration-philosophy)
3. [Asset Extraction Guide](#asset-extraction-guide)
4. [UI/UX Pattern Adaptation](#uiux-pattern-adaptation)
5. [Agent Logic Translation](#agent-logic-translation)
6. [Timeline Integration](#timeline-integration)
7. [Code Translation Reference](#code-translation-reference)
8. [Testing & Validation](#testing--validation)
9. [Common Pitfalls](#common-pitfalls)
10. [Success Criteria](#success-criteria)

---

## Executive Summary

### What We're Taking
- ✅ **Character sprites and pixel art** (direct copy, MIT licensed)
- ✅ **UI layout patterns** (recreate structure, not code)
- ✅ **Agent movement logic** (translate TypeScript → Python)
- ✅ **Perception systems** (translate and adapt)
- ✅ **Design patterns** (color schemes, component hierarchy)

### What We're NOT Taking
- ❌ Convex backend integration
- ❌ PixiJS rendering engine (too complex for 5 agents)
- ❌ Clerk authentication system
- ❌ Their memory/planning systems (conflicts with DSPy)
- ❌ WebSocket patterns (ours is simpler and better)

### Time Investment
- **Total**: 5-7 hours over Days 0-1
- **ROI**: Professional UI + battle-tested agent logic without architectural compromise

### Budget Impact
- **Cost**: $0 (MIT licensed open source)
- **Savings**: ~10-15 hours of UI/UX design work

---

## Integration Philosophy

### Core Principle
**"Steal like an artist, don't copy like a pirate"**

We're extracting **patterns, concepts, and assets** - not dependencies or architectural decisions. Every piece we take must:

1. ✅ Work with FastAPI + DuckDB backend
2. ✅ Not interfere with DSPy compilation workflow
3. ✅ Fit within $5 budget constraint
4. ✅ Complete in 5-7 hours
5. ✅ Enhance, not replace, our plan

### Decision Framework

```
┌─────────────────────────────────────────────────┐
│ Should I take this from AI Town?                │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. Does it require Convex?                     │
│    YES → Skip it                                │
│    NO  → Continue                               │
│                                                 │
│ 2. Does it conflict with DSPy architecture?    │
│    YES → Skip it                                │
│    NO  → Continue                               │
│                                                 │
│ 3. Can I extract it in <2 hours?               │
│    NO  → Skip it                                │
│    YES → Continue                               │
│                                                 │
│ 4. Does it provide immediate value?            │
│    NO  → Skip it                                │
│    YES → TAKE IT                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Asset Extraction Guide

### 1. Setup Reference Repository

```bash
# Clone AI Town for reference (don't modify your main project)
cd ~/projects
git clone https://github.com/a16z-infra/ai-town.git ai-town-reference
cd ai-town-reference

# Quick exploration
ls -la public/assets/        # Sprites and tilesets
ls -la src/components/       # UI components
ls -la convex/engine/        # Game engine logic
ls -la data/                 # Character definitions
```

**Time**: 10 minutes

---

### 2. Extract Character Sprites

**Location**: `ai-town-reference/public/assets/`

**What to copy**:
```bash
cd ~/projects/mini-town

# Copy all sprite assets
mkdir -p frontend/public/assets/spritesheets
cp -r ~/projects/ai-town-reference/public/assets/* frontend/public/assets/

# Verify what you got
ls -lh frontend/public/assets/
# Expected:
# - 32x32folk.png (character sprites)
# - gentle.png (map tileset)
# - Various sprite sheets
```

**File Inventory**:

| File | Purpose | Size | Priority |
|------|---------|------|----------|
| `32x32folk.png` | Main character sprites (walking, idle) | ~50KB | ✅ Critical |
| `gentle.png` | Town background tileset | ~200KB | ✅ High |
| `f1.png`, `f2.png`, etc. | Individual character sheets | ~10KB each | ⚠️ Optional |

**Attribution**:
```markdown
# Add to your README.md

## Assets & Credits

This project uses character sprites and tilesets from [a16z AI Town](https://github.com/a16z-infra/ai-town), which are licensed under the MIT License.

Character sprites:
- 32x32 folk sprites: AI Town project
- Original assets by [ansimuz](https://opengameart.org/content/tiny-rpg-forest)
- Tilesheet: [George Bailey](https://opengameart.org/content/16x16-game-assets)

We've modified and adapted these assets for our simulation environment.
```

**Time**: 15 minutes

---

### 3. Sprite Sheet Structure Analysis

AI Town uses pre-defined sprite sheets with animation frames. Study their structure:

```typescript
// ai-town-reference/data/spritesheets/f1.ts (example)
export const f1SpritesheetData = {
  frames: {
    down: { frame: { x: 0, y: 0, w: 32, h: 32 } },
    left: { frame: { x: 32, y: 0, w: 32, h: 32 } },
    right: { frame: { x: 64, y: 0, w: 32, h: 32 } },
    up: { frame: { x: 96, y: 0, w: 32, h: 32 } },
  },
  meta: {
    size: { w: 128, h: 32 },
  },
};
```

**Your adaptation** (create `frontend/lib/sprites.ts`):

```typescript
// frontend/lib/sprites.ts
export interface SpriteFrame {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SpriteSheet {
  url: string;
  frames: Record<string, SpriteFrame>;
}

export const CHARACTER_SPRITES: Record<string, SpriteSheet> = {
  alice: {
    url: '/assets/32x32folk.png',
    frames: {
      down: { x: 0, y: 0, w: 32, h: 32 },
      left: { x: 32, y: 0, w: 32, h: 32 },
      right: { x: 64, y: 0, w: 32, h: 32 },
      up: { x: 96, y: 0, w: 32, h: 32 },
    },
  },
  bob: {
    url: '/assets/32x32folk.png',
    frames: {
      down: { x: 0, y: 32, w: 32, h: 32 },
      left: { x: 32, y: 32, w: 32, h: 32 },
      right: { x: 64, y: 32, w: 32, h: 32 },
      up: { x: 96, y: 32, w: 32, h: 32 },
    },
  },
  // Add carol, dave, eve...
};
```

**Time**: 30 minutes

---

## UI/UX Pattern Adaptation

### 1. Study AI Town's Component Structure

**Key files to study** (read, don't copy):

```bash
# Layout and structure
ai-town-reference/src/components/Game.tsx           # Main game view
ai-town-reference/src/components/Player.tsx         # Agent rendering
ai-town-reference/src/components/DebugPanel.tsx     # Inspector panel

# Styling
ai-town-reference/src/globals.css                   # Color scheme
```

**AI Town's Layout Pattern**:
```
┌─────────────────────────────────────────────┐
│  Header (Title, Controls)                   │
├──────────────────────┬──────────────────────┤
│                      │                      │
│   Main Canvas        │   Sidebar Panel      │
│   (PixiJS Game)      │   - Agent Info       │
│                      │   - Memories         │
│                      │   - Conversation     │
│   [Agents moving]    │   - Debug Tools      │
│                      │                      │
│                      │                      │
├──────────────────────┴──────────────────────┤
│  Footer (Status, Metrics)                   │
└─────────────────────────────────────────────┘
```

---

### 2. Extract Color Scheme

```css
/* frontend/styles/globals.css */

/* Extracted from AI Town */
:root {
  /* Dark theme colors */
  --bg-primary: #0f1419;
  --bg-secondary: #1c2128;
  --bg-tertiary: #2d333b;
  
  /* Text colors */
  --text-primary: #e6edf3;
  --text-secondary: #7d8590;
  --text-muted: #484f58;
  
  /* Accent colors */
  --accent-blue: #539bf5;
  --accent-green: #57ab5a;
  --accent-yellow: #d4a72c;
  --accent-red: #f47067;
  
  /* Border colors */
  --border-default: #444c56;
  --border-muted: #373e47;
  
  /* UI element colors */
  --btn-bg: #373e47;
  --btn-hover: #444c56;
  --panel-bg: rgba(22, 27, 34, 0.95);
}

/* Dark mode (default) */
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
```

**Time**: 20 minutes

---

### 3. Create Your Layout Components

- Implemented `components/TownView.tsx` to assemble map + sidebar using the WebSocket hook.
- Replaced the old `Map` component with `MapCanvas.tsx` (sprite-aware, pixelated fallback when assets missing).
- Renamed `GodMode` → `GodModeControls` for clarity and consolidated panel styling.

#### 3.1 Main View Component

```typescript
// frontend/components/TownView.tsx
'use client';

import { useEffect, useState } from 'react';
import MapCanvas from './MapCanvas';
import AgentInspector from './AgentInspector';
import SystemPanel from './SystemPanel';
import GodModeControls from './GodModeControls';

interface Agent {
  id: number;
  name: string;
  x: number;
  y: number;
  state: 'active' | 'confused' | 'idle';
  personality: string;
  goal: string;
}

interface SimulationState {
  agents: Agent[];
  tick: number;
  time: string;
}

export default function TownView() {
  const [state, setState] = useState<SimulationState | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('Connected to simulation');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setState(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('Disconnected from simulation');
    };
    
    return () => ws.close();
  }, []);

  const handlePause = async () => {
    const endpoint = isPaused ? '/god/resume' : '/god/pause';
    await fetch(`http://localhost:8000${endpoint}`, { method: 'POST' });
    setIsPaused(!isPaused);
  };

  if (!state) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-400">Connecting to simulation...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Mini-Town</h1>
            <p className="text-sm text-gray-400">
              Compiled Generative Agents with DSPy
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-400">
              Tick: <span className="text-white font-mono">{state.tick}</span>
            </div>
            <div className="text-sm text-gray-400">
              Time: <span className="text-white font-mono">{state.time}</span>
            </div>
            <button
              onClick={handlePause}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              {isPaused ? 'Resume' : 'Pause'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Map */}
        <div className="flex-1 relative">
          <MapCanvas 
            agents={state.agents}
            selectedAgent={selectedAgent}
            onAgentClick={setSelectedAgent}
          />
        </div>

        {/* Right: Sidebar (inspired by AI Town's layout) */}
        <div className="w-96 bg-gray-800 border-l border-gray-700 overflow-y-auto">
          {selectedAgent ? (
            <AgentInspector agent={selectedAgent} />
          ) : (
            <div className="p-6 text-center text-gray-500">
              <p>Click an agent to inspect</p>
            </div>
          )}
          
          <div className="border-t border-gray-700">
            <SystemPanel />
          </div>
          
          <div className="border-t border-gray-700">
            <GodModeControls />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div>
            {state.agents.length} agents active
          </div>
          <div>
            Status: <span className="text-green-400">Running</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
```

**Time**: 1 hour

---

#### 3.2 Map Canvas Component (Simplified)

**Note**: AI Town uses PixiJS. For 5 agents, we use **HTML5 Canvas** instead (much simpler).

```typescript
// frontend/components/MapCanvas.tsx
'use client';

import { useEffect, useRef, useState } from 'react';

interface Agent {
  id: number;
  name: string;
  x: number;
  y: number;
  state: 'active' | 'confused' | 'idle';
}

interface MapCanvasProps {
  agents: Agent[];
  selectedAgent: Agent | null;
  onAgentClick: (agent: Agent) => void;
}

const AGENT_COLORS: Record<string, string> = {
  alice: '#FF6B6B',
  bob: '#4ECDC4',
  carol: '#45B7D1',
  dave: '#FFA07A',
  eve: '#98D8C8',
};

export default function MapCanvas({ agents, selectedAgent, onAgentClick }: MapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sprites, setSprites] = useState<Record<string, HTMLImageElement>>({});

  // Load sprite sheet
  useEffect(() => {
    const img = new Image();
    img.src = '/assets/32x32folk.png';
    img.onload = () => {
      setSprites({ main: img });
    };
  }, []);

  // Draw agents
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid (optional, for debugging)
    ctx.strokeStyle = '#2a2a3e';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw agents
    agents.forEach(agent => {
      const color = AGENT_COLORS[agent.name.toLowerCase()] || '#FFFFFF';
      const isSelected = selectedAgent?.id === agent.id;

      // Draw agent circle (or sprite if loaded)
      if (sprites.main) {
        // Draw sprite from sprite sheet
        // For now, simple circle
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 20, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        // Fallback: colored circle
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 20, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 25, 0, 2 * Math.PI);
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Agent name
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(agent.name, agent.x, agent.y - 30);

      // State indicator
      if (agent.state === 'confused') {
        ctx.fillStyle = '#FF6B6B';
        ctx.font = '16px Arial';
        ctx.fillText('?', agent.x, agent.y + 5);
      }
    });
  }, [agents, selectedAgent, sprites]);

  // Handle click
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find clicked agent
    for (const agent of agents) {
      const distance = Math.sqrt((x - agent.x) ** 2 + (y - agent.y) ** 2);
      if (distance <= 20) {
        onAgentClick(agent);
        break;
      }
    }
  };

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      onClick={handleClick}
      className="w-full h-full cursor-pointer"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
```

**Time**: 1 hour

---

#### 3.3 Agent Inspector Component

```typescript
// frontend/components/AgentInspector.tsx
'use client';

import { useEffect, useState } from 'react';

interface Agent {
  id: number;
  name: string;
  x: number;
  y: number;
  state: string;
  personality: string;
  goal: string;
}

interface Memory {
  id: number;
  content: string;
  importance: number;
  timestamp: string;
}

interface AgentInspectorProps {
  agent: Agent;
}

export default function AgentInspector({ agent }: AgentInspectorProps) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [plan, setPlan] = useState<string>('');

  useEffect(() => {
    // Fetch agent details
    fetch(`http://localhost:8000/api/agents/${agent.id}`)
      .then(res => res.json())
      .then(data => {
        setMemories(data.memories || []);
        setPlan(data.current_plan || '');
      })
      .catch(console.error);
  }, [agent.id]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">{agent.name}</h2>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            agent.state === 'active' ? 'bg-green-900 text-green-200' :
            agent.state === 'confused' ? 'bg-red-900 text-red-200' :
            'bg-gray-700 text-gray-300'
          }`}>
            {agent.state}
          </span>
          <span className="text-sm text-gray-400">
            ({agent.x.toFixed(0)}, {agent.y.toFixed(0)})
          </span>
        </div>
      </div>

      {/* Goal */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Goal</h3>
        <p className="text-white">{agent.goal}</p>
      </div>

      {/* Personality */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Personality</h3>
        <div className="flex flex-wrap gap-2">
          {agent.personality.split(',').map((trait, i) => (
            <span key={i} className="px-2 py-1 bg-blue-900 text-blue-200 rounded text-sm">
              {trait.trim()}
            </span>
          ))}
        </div>
      </div>

      {/* Current Plan */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Current Plan</h3>
        <div className="bg-gray-900 rounded-lg p-3 text-sm text-gray-300">
          {plan || 'No active plan'}
        </div>
      </div>

      {/* Recent Memories */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Recent Memories</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {memories.length > 0 ? (
            memories.map((memory) => (
              <div key={memory.id} className="bg-gray-900 rounded-lg p-3">
                <div className="flex items-start justify-between mb-1">
                  <span className="text-xs text-gray-500">
                    {new Date(memory.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="text-xs font-medium text-yellow-400">
                    {(memory.importance * 10).toFixed(1)}/10
                  </span>
                </div>
                <p className="text-sm text-gray-300">{memory.content}</p>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 italic">No memories yet</p>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Time**: 45 minutes

---

## Agent Logic Translation

> 🔔 **Before testing the loitering/conversation behavior**, make sure each agent’s `current_plan` includes at least one overlapping timeslot and location. After you translate the movement/perception code, seed the DuckDB `agents` table (via a short script or SQL update) so residents converge on the same coordinates—otherwise everyone will stay in random-walk mode and you won’t see the social interactions kick in.

### Planner Seeds & Compilation Prep

- Added `seeds/planner/planner_seeds_v1.json` with 12 curated examples (invitation preservation, location/time compliance).
- Use `python backend/seed_plans.py` to populate shared schedules for local runs.
- Compile PlanDay via `python compilation/compile_planner.py --budget 40` (recommended in Colab with proper API keys).


### 1. Movement System

**AI Town's TypeScript version** (study this):

```typescript
// ai-town-reference/convex/engine/movement.ts (simplified)
export function moveTowards(
  current: { x: number; y: number },
  target: { x: number; y: number },
  speed: number
): { x: number; y: number } {
  const dx = target.x - current.x;
  const dy = target.y - current.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  if (distance <= speed) {
    return target;
  }
  
  const ratio = speed / distance;
  return {
    x: current.x + dx * ratio,
    y: current.y + dy * ratio,
  };
}

export function findPath(
  start: { x: number; y: number },
  end: { x: number; y: number },
  obstacles: Array<{ x: number; y: number; radius: number }>
): Array<{ x: number; y: number }> {
  // Simple A* pathfinding
  // For 5 agents, we can use simple direct movement
  return [start, end];
}
```

**Your Python translation**:

```python
# backend/movement.py
import math
from typing import Dict, List, Tuple, Optional

class Vector2D:
    """2D vector for position and movement"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Vector2D') -> float:
        """Calculate Euclidean distance"""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)
    
    def direction_to(self, other: 'Vector2D') -> 'Vector2D':
        """Get normalized direction vector"""
        distance = self.distance_to(other)
        if distance == 0:
            return Vector2D(0, 0)
        dx = other.x - self.x
        dy = other.y - self.y
        return Vector2D(dx / distance, dy / distance)
    
    def to_dict(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y}


def move_towards(
    current: Dict[str, float],
    target: Dict[str, float],
    speed: float
) -> Dict[str, float]:
    """
    Move current position towards target at given speed.
    Translated from AI Town's TypeScript version.
    
    Args:
        current: Current position {x, y}
        target: Target position {x, y}
        speed: Movement speed (pixels per tick)
    
    Returns:
        New position {x, y}
    """
    dx = target['x'] - current['x']
    dy = target['y'] - current['y']
    distance = math.sqrt(dx * dx + dy * dy)
    
    # If close enough, snap to target
    if distance <= speed:
        return target.copy()
    
    # Move towards target
    ratio = speed / distance
    return {
        'x': current['x'] + dx * ratio,
        'y': current['y'] + dy * ratio
    }


def check_collision(
    pos: Dict[str, float],
    other_pos: Dict[str, float],
    radius: float = 20.0
) -> bool:
    """
    Check if two agents collide.
    
    Args:
        pos: Position of first agent
        other_pos: Position of second agent
        radius: Collision radius (default 20px)
    
    Returns:
        True if colliding
    """
    dx = other_pos['x'] - pos['x']
    dy = other_pos['y'] - pos['y']
    distance = math.sqrt(dx * dx + dy * dy)
    return distance < (radius * 2)


def resolve_collision(
    pos: Dict[str, float],
    other_pos: Dict[str, float],
    radius: float = 20.0
) -> Dict[str, float]:
    """
    Push agent away from collision.
    
    Args:
        pos: Position of agent to move
        other_pos: Position of agent to move away from
        radius: Collision radius
    
    Returns:
        Adjusted position
    """
    dx = pos['x'] - other_pos['x']
    dy = pos['y'] - other_pos['y']
    distance = math.sqrt(dx * dx + dy * dy)
    
    if distance == 0:
        # If exactly on top, move in random direction
        import random
        angle = random.random() * 2 * math.pi
        dx = math.cos(angle)
        dy = math.sin(angle)
        distance = 0.1
    
    # Push apart
    overlap = (radius * 2) - distance
    if overlap > 0:
        push_distance = overlap / 2 + 1
        ratio = push_distance / distance
        return {
            'x': pos['x'] + dx * ratio,
            'y': pos['y'] + dy * ratio
        }
    
    return pos


def find_random_nearby_position(
    current: Dict[str, float],
    min_distance: float = 50.0,
    max_distance: float = 150.0
) -> Dict[str, float]:
    """
    Find random position near current location.
    Used for idle wandering behavior.
    
    Args:
        current: Current position
        min_distance: Minimum distance from current
        max_distance: Maximum distance from current
    
    Returns:
        Random nearby position
    """
    import random
    angle = random.random() * 2 * math.pi
    distance = random.uniform(min_distance, max_distance)
    
    return {
        'x': current['x'] + math.cos(angle) * distance,
        'y': current['y'] + math.sin(angle) * distance
    }


def clamp_to_bounds(
    pos: Dict[str, float],
    width: float = 800.0,
    height: float = 600.0,
    margin: float = 20.0
) -> Dict[str, float]:
    """
    Clamp position to map bounds.
    
    Args:
        pos: Position to clamp
        width: Map width
        height: Map height
        margin: Margin from edges
    
    Returns:
        Clamped position
    """
    return {
        'x': max(margin, min(width - margin, pos['x'])),
        'y': max(margin, min(height - margin, pos['y']))
    }
```

**Time**: 1.5 hours

---

### 2. Perception System

**AI Town's approach** (study this):

```typescript
// ai-town-reference/convex/aiTown/agent.ts (simplified)
function findNearbyAgents(
  myPosition: { x: number; y: number },
  allAgents: Array<{ id: string; x: number; y: number; name: string }>,
  radius: number
): Array<{ id: string; name: string; distance: number }> {
  const nearby = [];
  
  for (const agent of allAgents) {
    if (agent.id === myId) continue;
    
    const dx = agent.x - myPosition.x;
    const dy = agent.y - myPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance <= radius) {
      nearby.push({
        id: agent.id,
        name: agent.name,
        distance,
      });
    }
  }
  
  // Sort by distance
  return nearby.sort((a, b) => a.distance - b.distance);
}
```

**Your Python translation**:

```python
# backend/perception.py
import math
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerceivedEntity:
    """Something an agent perceives"""
    id: int
    type: str  # 'agent' | 'event' | 'location'
    name: str
    distance: float
    data: Dict

PERCEPTION_RADIUS = 100.0  # pixels


def get_nearby_agents(
    observer_id: int,
    observer_pos: Dict[str, float],
    all_agents: List[Dict],
    radius: float = PERCEPTION_RADIUS
) -> List[PerceivedEntity]:
    """
    Find all agents within perception radius.
    Translated from AI Town's findNearbyAgents.
    
    Args:
        observer_id: ID of observing agent
        observer_pos: Position of observer {x, y}
        all_agents: List of all agents in simulation
        radius: Perception radius
    
    Returns:
        List of perceived agents, sorted by distance
    """
    perceived = []
    
    for agent in all_agents:
        # Don't perceive self
        if agent['id'] == observer_id:
            continue
        
        # Calculate distance
        dx = agent['x'] - observer_pos['x']
        dy = agent['y'] - observer_pos['y']
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if within radius
        if distance <= radius:
            perceived.append(PerceivedEntity(
                id=agent['id'],
                type='agent',
                name=agent['name'],
                distance=distance,
                data={
                    'position': {'x': agent['x'], 'y': agent['y']},
                    'state': agent.get('state', 'idle'),
                    'goal': agent.get('goal', ''),
                }
            ))
    
    # Sort by distance (closest first)
    perceived.sort(key=lambda e: e.distance)
    return perceived


def create_observation_text(
    perceived: List[PerceivedEntity],
    current_time: str
) -> str:
    """
    Convert perceived entities into natural language observation.
    This feeds into DSPy ScoreImportance module.
    
    Args:
        perceived: List of perceived entities
        current_time: Current simulation time
    
    Returns:
        Natural language observation
    """
    if not perceived:
        return f"[{current_time}] Nothing nearby. The area is quiet."
    
    observations = [f"[{current_time}] I see:"]
    
    for entity in perceived:
        if entity.type == 'agent':
            distance_desc = (
                "very close" if entity.distance < 30 else
                "nearby" if entity.distance < 70 else
                "in the distance"
            )
            observations.append(
                f"- {entity.name} is {distance_desc} "
                f"(state: {entity.data['state']})"
            )
    
    return "\n".join(observations)


def filter_important_observations(
    observations: List[str],
    importance_scores: List[float],
    threshold: float = 5.0
) -> List[str]:
    """
    Filter observations by importance score.
    Only store observations above threshold.
    
    Args:
        observations: List of observation texts
        importance_scores: Corresponding importance scores (1-10)
        threshold: Minimum score to keep
    
    Returns:
        Filtered observations
    """
    return [
        obs for obs, score in zip(observations, importance_scores)
        if score >= threshold
    ]
```

**Time**: 1 hour

---

### 3. State Machine

**AI Town's pattern** (conceptual):

```typescript
// Simplified state machine concept
enum AgentState {
  IDLE = 'idle',
  MOVING = 'moving',
  TALKING = 'talking',
  CONFUSED = 'confused',
}

function updateAgentState(agent: Agent): AgentState {
  if (agent.hasError) return AgentState.CONFUSED;
  if (agent.isInConversation) return AgentState.TALKING;
  if (agent.isMoving) return AgentState.MOVING;
  return AgentState.IDLE;
}
```

**Your Python version**:

```python
# backend/agents.py
from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime

class AgentState(Enum):
    """Agent states - extracted from AI Town pattern"""
    IDLE = "idle"
    MOVING = "moving"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    CONFUSED = "confused"  # Error state


class Agent:
    """Agent class incorporating AI Town patterns"""
    
    def __init__(
        self,
        id: int,
        name: str,
        x: float,
        y: float,
        personality: str,
        goal: str
    ):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.personality = personality
        self.goal = goal
        
        # State
        self.state = AgentState.IDLE
        self.target_position: Optional[Dict[str, float]] = None
        self.movement_speed = 2.0  # pixels per tick
        
        # Cognitive state
        self.current_plan: Optional[str] = None
        self.last_reflection_time: Optional[datetime] = None
        self.importance_accumulator = 0.0
        self.reflect_threshold = 5.0
        
        # Error handling
        self.error_count = 0
        self.max_errors = 3
    
    def update_state(self):
        """Update agent state based on current conditions"""
        # Handle error state
        if self.error_count >= self.max_errors:
            self.state = AgentState.CONFUSED
            return
        
        # Reset to idle if nothing to do
        if self.state == AgentState.CONFUSED:
            if self.error_count == 0:
                self.state = AgentState.IDLE
        
        # Handle movement
        if self.target_position:
            distance = math.sqrt(
                (self.target_position['x'] - self.x) ** 2 +
                (self.target_position['y'] - self.y) ** 2
            )
            if distance < 5:  # Close enough
                self.target_position = None
                self.state = AgentState.IDLE
            else:
                self.state = AgentState.MOVING
    
    def should_reflect(self) -> bool:
        """Check if agent should reflect (based on importance accumulation)"""
        return self.importance_accumulator >= self.reflect_threshold
    
    def add_importance(self, score: float):
        """Accumulate importance for reflection threshold"""
        self.importance_accumulator += score
    
    def reset_importance(self):
        """Reset after reflection"""
        self.importance_accumulator = 0.0
        self.last_reflection_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """Serialize for WebSocket/API"""
        return {
            'id': self.id,
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'state': self.state.value,
            'personality': self.personality,
            'goal': self.goal,
            'current_plan': self.current_plan,
            'target': self.target_position,
        }
```

**Time**: 45 minutes

---

## Timeline Integration

### Updated Day-by-Day Plan

**Day 0 (Setup - 2 hours)**
- [ ] Clone AI Town reference repo (10 min)
- [ ] Extract sprites and assets (15 min)
- [ ] Set up frontend with Next.js (30 min)
- [ ] Copy color scheme to globals.css (20 min)
- [ ] Create basic component structure (45 min)

**Day 0.5 (Hardcoded Validation - 4-6 hours)** ⚠️ CRITICAL
- [ ] Implement movement.py with AI Town logic (2-3 hours with testing buffer)
- [ ] Implement perception.py (1-1.5 hours with testing)
- [ ] Create basic FastAPI backend (1 hour)
- [ ] Wire up WebSocket (30 min)
- [ ] Implement MapCanvas component (1 hour)
- [ ] Test: 3 agents moving with hardcoded behaviors (1 hour)
- [ ] **Validation**: Run `npm run build` and `uvicorn` smoke test (30 min)

**Day 1 (DuckDB + Vector Setup - 4-6 hours)**
- Continue as planned in CLAUDE.md
- Use Agent class from translated AI Town patterns
- **Checkpoint**: Review plan.md - confirm movement translation is still valuable

**Day 2 (Latency + Uncompiled DSPy - 6-8 hours)**
- Continue as planned with latency tracking
- **Checkpoint**: Review extraction ROI - is UI/movement logic helping or hindering?

**Days 3-10**
- Continue following original CLAUDE.md timeline
- Use UI components you've built
- Leverage movement/perception logic translated from AI Town

---

### Plan Seeding for Loitering/Conversation

Since Day 7 enhancements added loitering and conversation mechanics, we need seed data for realistic schedules:

```python
# backend/seed_plans.py
"""Seed daily plans for agents to support loitering/conversation"""

DAILY_PLAN_SEEDS = [
    {
        "agent": "Alice",
        "time": "07:00",
        "activity": "Morning coffee at town square",
        "location": {"x": 300, "y": 250},
        "duration_minutes": 30,
    },
    {
        "agent": "Alice",
        "time": "09:00",
        "activity": "Chat with neighbors at the park",
        "location": {"x": 450, "y": 180},
        "duration_minutes": 45,
    },
    {
        "agent": "Bob",
        "time": "08:00",
        "activity": "Work on research at home",
        "location": {"x": 150, "y": 350},
        "duration_minutes": 120,
    },
    {
        "agent": "Bob",
        "time": "10:30",
        "activity": "Quick walk to library",
        "location": {"x": 500, "y": 400},
        "duration_minutes": 20,
    },
    # Add more for Carol, Dave, Eve...
]

def seed_daily_plans(db_conn):
    """Insert seed plans into agents table"""
    for plan_seed in DAILY_PLAN_SEEDS:
        db_conn.execute("""
            UPDATE agents
            SET current_plan = ?
            WHERE name = ?
        """, (f"{plan_seed['time']}: {plan_seed['activity']}", plan_seed['agent']))
```

**Usage**: Call `seed_daily_plans()` after agent initialization on Day 6-7 to populate realistic schedules.

---

## Code Translation Reference

### Quick Translation Patterns

| TypeScript (AI Town) | Python (Mini-Town) | Notes |
|---------------------|-------------------|-------|
| `Math.sqrt(x)` | `math.sqrt(x)` | Same function |
| `{ x: 0, y: 0 }` | `{'x': 0, 'y': 0}` | Dictionary |
| `interface Pos` | `@dataclass class Pos` | Type definition |
| `arr.map(x => ...)` | `[... for x in arr]` | List comprehension |
| `arr.filter(x => ...)` | `[x for x in arr if ...]` | Filtering |
| `arr.sort((a,b) => ...)` | `sorted(arr, key=lambda x: ...)` | Sorting |
| `async function` | `async def` | Async syntax |
| `const ws = new WebSocket()` | `ws = websockets.connect()` | WebSocket client |

### Common Gotchas

1. **TypeScript nullability**:
   ```typescript
   // TypeScript
   position?: { x: number; y: number }
   ```
   ```python
   # Python
   position: Optional[Dict[str, float]] = None
   ```

2. **Array operations**:
   ```typescript
   // TypeScript
   agents.forEach(a => process(a));
   ```
   ```python
   # Python
   for agent in agents:
       process(agent)
   ```

3. **Object spread**:
   ```typescript
   // TypeScript
   const newObj = { ...oldObj, x: 5 };
   ```
   ```python
   # Python
   new_obj = {**old_obj, 'x': 5}
   ```

---

## Testing & Validation

### Phase 1: Asset Verification (Day 0)

```bash
# Check sprites loaded
ls -lh frontend/public/assets/32x32folk.png
# Should be ~50KB

# Test in browser
# Add to a test page:
# <img src="/assets/32x32folk.png" />
```

**Success criteria**:
- ✅ Sprites visible in browser
- ✅ No 404 errors in console
- ✅ Attribution in README

---

### Phase 2: UI Component Verification (Day 0.5)

```bash
# Start frontend
cd frontend
npm run dev
```

**Manual test checklist**:
- [ ] TownView renders without errors
- [ ] Map canvas displays (even if empty)
- [ ] Sidebar panels visible
- [ ] Color scheme matches AI Town aesthetic
- [ ] Responsive layout (resize browser)

**Success criteria**:
- ✅ No console errors
- ✅ Layout looks professional
- ✅ All panels render

---

### Phase 3: Movement Logic Verification (Day 0.5)

```python
# backend/test_movement.py
from movement import move_towards, check_collision

def test_move_towards():
    current = {'x': 0, 'y': 0}
    target = {'x': 100, 'y': 0}
    speed = 10
    
    new_pos = move_towards(current, target, speed)
    assert new_pos['x'] == 10
    assert new_pos['y'] == 0
    print("✅ move_towards works")

def test_collision():
    pos1 = {'x': 50, 'y': 50}
    pos2 = {'x': 60, 'y': 50}
    
    assert check_collision(pos1, pos2, radius=20) == True
    print("✅ Collision detection works")

if __name__ == '__main__':
    test_move_towards()
    test_collision()
    print("\n✅ All movement tests passed!")
```

Run tests:
```bash
python backend/test_movement.py
```

**Success criteria**:
- ✅ All tests pass
- ✅ Movement is smooth (visual test)
- ✅ Agents don't overlap (collision resolution)

---

### Phase 4: Integration Test (Day 0.5)

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload

# Terminal 2: Start frontend
cd frontend
npm run dev

# Terminal 3: Watch logs
tail -f logs/mini_town.log
```

**Integration test checklist**:
- [ ] Backend starts without errors
- [ ] Frontend connects via WebSocket
- [ ] 3 agents visible on map
- [ ] Agents move smoothly
- [ ] Click agent → sidebar updates
- [ ] No memory leaks after 5 minutes

**Success criteria**:
- ✅ Full stack running
- ✅ Real-time updates working
- ✅ UI responsive to clicks
- ✅ No performance issues

---

### Phase 5: Build Validation (End of Day 0.5) ⚠️ REQUIRED

**Purpose**: Catch issues early before they compound. These checks have caught lint errors, type mismatches, and import issues in past iterations.

```bash
# Frontend build validation
cd frontend
npm run build
# Expected: ✓ Compiled successfully
# If errors: Fix before proceeding to Day 1

# Backend syntax validation
cd backend
python -m compileall -q *.py
# Expected: No output (silent success)
# If errors: Fix Python syntax issues

# Backend import validation (smoke test)
python -c "import main; import agents; import memory; print('✅ All imports work')"
# Expected: ✅ All imports work
```

**Common issues caught**:
- TypeScript unused variables → build fails
- Missing imports in Python
- Circular dependencies
- Type annotation errors
- CSS syntax errors

**Time budget**: 15-30 minutes (fix any issues immediately)

**Decision rule**:
- ✅ All checks pass → Proceed to Day 1
- ❌ Checks fail → Fix before continuing (do NOT defer)

---

## Common Pitfalls

### Pitfall 1: Over-Engineering the Rendering
**Symptom**: Spending 3+ hours trying to implement PixiJS

**Solution**: Use simple Canvas API or even HTML/CSS for Day 0.5. You have 5 agents, not 100.

```typescript
// ❌ DON'T DO THIS (too complex)
import * as PIXI from 'pixi.js';
const app = new PIXI.Application({...});
// ... 200 lines of PixiJS setup

// ✅ DO THIS (simple & works)
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.arc(agent.x, agent.y, 20, 0, 2*Math.PI);
ctx.fill();
```

---

### Pitfall 2: Copying Code Verbatim
**Symptom**: Getting TypeScript errors in Python files

**Solution**: Always **translate**, never copy. Use the patterns above.

---

### Pitfall 3: Trying to Extract Their Memory System
**Symptom**: Spending hours understanding Convex vector queries

**Solution**: **Skip it entirely.** Their memory system uses Convex. Yours uses DuckDB + DSPy. Completely different architectures.

---

### Pitfall 4: Ignoring Your Timeline
**Symptom**: Still setting up AI Town integration on Day 3

**Solution**: Set strict time limits:
- Day 0: 2 hours max
- Day 0.5: 6 hours max
- If not done by end of Day 0.5 → revert to original plan

---

## Success Criteria

### Phase 1 Success (End of Day 0)
- ✅ Sprites extracted and loading
- ✅ Color scheme applied
- ✅ Component structure created
- ✅ Frontend builds without errors

### Phase 2 Success (End of Day 0.5)
- ✅ 3 agents moving on map (hardcoded)
- ✅ Movement logic working (translated from AI Town)
- ✅ Perception system working
- ✅ Click agent → inspector updates
- ✅ Professional-looking UI

### Final Success (End of Day 10)
- ✅ All features from original CLAUDE.md plan
- ✅ **PLUS** polished UI from AI Town patterns
- ✅ **PLUS** battle-tested movement logic
- ✅ Under $5 budget ✅
- ✅ DSPy compilation working ✅

---

## File Structure After Integration

```
mini-town/
├── README.md                    # With AI Town attribution
├── CLAUDE.md                    # Original plan
├── AI_TOWN_INTEGRATION.md       # This document
├── config.yml
├── requirements.txt
│
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── agents.py                # Agent class (AI Town patterns)
│   ├── movement.py              # ⭐ Translated from AI Town
│   ├── perception.py            # ⭐ Translated from AI Town
│   ├── memory.py                # DuckDB integration (your original)
│   ├── dspy_modules.py          # DSPy modules (your original)
│   └── utils.py
│
├── frontend/
│   ├── package.json
│   ├── styles/
│   │   └── globals.css          # ⭐ AI Town color scheme
│   ├── components/
│   │   ├── TownView.tsx         # ⭐ Inspired by AI Town layout
│   │   ├── MapCanvas.tsx        # ⭐ Simplified version
│   │   ├── AgentInspector.tsx   # ⭐ Based on AI Town design
│   │   ├── SystemPanel.tsx
│   │   └── GodMode.tsx
│   ├── lib/
│   │   ├── sprites.ts           # ⭐ Sprite sheet definitions
│   │   └── websocket.ts
│   └── public/
│       └── assets/              # ⭐ AI Town sprites
│           ├── 32x32folk.png
│           └── gentle.png
│
├── reference/                   # ⚠️ Not committed, local only
│   └── ai-town-reference/       # Cloned AI Town for study
│
├── data/
│   └── town.db
│
└── logs/
```

---

## Appendix A: AI Town Study Guide

### Most Useful Files to Study

**For UI/UX**:
1. `src/components/Game.tsx` - Overall layout
2. `src/components/Player.tsx` - Agent rendering
3. `src/globals.css` - Styling

**For Agent Logic**:
1. `convex/engine/movement.ts` - Movement system
2. `convex/aiTown/agent.ts` - Agent state machine
3. `convex/agent/memory.ts` - Memory management (⚠️ Convex-specific)

**For Sprites**:
1. `public/assets/` - All assets
2. `data/spritesheets/` - Sprite definitions
3. `data/characters.ts` - Character configs

---

## Appendix B: Quick Start Commands

### Setup (One-time)
```bash
# 1. Clone AI Town reference
git clone https://github.com/a16z-infra/ai-town.git ~/projects/ai-town-reference

# 2. Extract assets
cd ~/projects/mini-town
mkdir -p frontend/public/assets
cp -r ~/projects/ai-town-reference/public/assets/* frontend/public/assets/

# 3. Add attribution
cat >> README.md << 'EOF'

## Credits
- Character sprites from [a16z AI Town](https://github.com/a16z-infra/ai-town) (MIT License)
EOF
```

### Development
```bash
# Terminal 1: Frontend
cd frontend
npm run dev

# Terminal 2: Backend
cd backend
uvicorn main:app --reload

# Terminal 3: Logs
tail -f logs/mini_town.log
```

---

## Appendix C: Emergency Rollback Plan

If integration takes too long (>7 hours) or causes issues:

```bash
# 1. Keep the sprites (they work)
# Don't delete frontend/public/assets/

# 2. Revert to simple UI
git checkout main -- frontend/components/

# 3. Use original movement logic
git checkout main -- backend/movement.py
git checkout main -- backend/perception.py

# 4. Continue with Day 1 of original plan
# You'll still have nice sprites!
```

---

## Final Checklist

### Before Starting (Day 0)
- [ ] Read this document completely
- [ ] Read relevant sections of CLAUDE.md
- [ ] Clone AI Town reference repo
- [ ] Set up project structure

### After Day 0
- [ ] Sprites extracted and loading ✅
- [ ] Color scheme applied ✅
- [ ] Component structure created ✅
- [ ] Time spent: ≤2 hours ⚠️

### After Day 0.5
- [ ] Movement logic working ✅
- [ ] Perception system working ✅
- [ ] UI rendering agents ✅
- [ ] Ready for Day 1 (DuckDB setup) ✅
- [ ] Time spent: ≤6 hours ⚠️

### Decision Point (End of Day 0.5)
- [ ] If everything works → Continue ✅
- [ ] If struggles → Rollback to original plan ⚠️
- [ ] Either way, have working prototype ✅

---

## Document Maintenance

**Update this document when**:
- You discover useful AI Town patterns not covered here
- You find better translation approaches
- You encounter new pitfalls
- You complete major milestones

**Version history**:
- v1.0 (Oct 13, 2025): Initial draft

---

**Good luck! Remember: AI Town integration is an enhancement, not a requirement. Your original plan is solid. Use AI Town to polish, not to pivot.** 🚀
