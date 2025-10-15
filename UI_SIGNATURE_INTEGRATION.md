# UI Signature Integration Guide

## Overview

This document describes the integration of new DSPy signatures that enhance the Mini-Town frontend UI with contextual explanations and intelligent summaries. These signatures provide users with deeper insights into agent behavior and observations.

**Date:** October 14, 2025  
**Status:** Complete and integrated

---

## Architecture

### New DSPy Signatures

Three new signatures have been added to enhance UI display:

#### 1. `PlanStepExplainer`
**Purpose:** Generate concise explanations for why an agent's current plan step matters.

**Inputs:**
- `agent_name`: Agent's name
- `agent_goal`: Agent's overarching goal
- `agent_personality`: Personality traits
- `step_summary`: Current plan step description
- `location`: Where the step occurs
- `recent_memories`: Relevant context from memory

**Output:**
- `explanation`: 1-2 sentence motivational tooltip (e.g., "Attending Bob's party helps build neighborhood connections and strengthens existing friendships.")

**Trigger:** Automatically generated when an agent switches to a new plan step during the simulation loop.

#### 2. `ObservationSummarizer`
**Purpose:** Synthesize high-importance observations into key thematic bullet points.

**Inputs:**
- `agent_name`: Agent's name
- `agent_goal`: Agent's goal
- `observations`: Bulleted list of observations with importance scores

**Output:**
- `summary_points`: Bullet list (≤3 items) with key takeaways (e.g., "• Received party invitation from Bob for tonight • Community interest in social gatherings")

**Trigger:** Generated periodically (every 10 ticks, staggered across agents) in the simulation loop.

#### 3. `ReflectionRefiner` (Existing, Enhanced)
**Purpose:** Refine raw reflection text into structured insights.

**Note:** This signature was already present but is now better integrated with the UI through the frontend updates.

---

## Backend Implementation

### Agent Class Extensions

**File:** `backend/agents.py`

Three new fields added to the `Agent` class:

```python
# Enhanced UI fields (populated by DSPy signatures)
self.step_explanation: Optional[str] = None
self.observation_summary: Optional[str] = None
self.recent_conversations: List[Dict[str, Any]] = []
```

**New Methods:**

1. **`async def generate_step_explanation(memory_store) -> Optional[str]`**
   - Retrieves relevant memories for context
   - Calls `explain_plan_step()` from `dspy_modules`
   - Stores result in `self.step_explanation` and `self.last_plan_step['explanation']`

2. **`async def generate_observation_summary() -> Optional[str]`**
   - Filters observations with importance ≥ 0.5
   - Formats as bulleted list with importance scores
   - Calls `summarize_observations()` from `dspy_modules`
   - Stores result in `self.observation_summary`

3. **`def track_conversation(conversation_text, participants=None)`**
   - Creates conversation digest with timestamp, summary, participants
   - Maintains rolling list of last 5 conversations
   - Stores in `self.recent_conversations`

### Simulation Loop Integration

**File:** `backend/main.py`

**Step Explanation (lines 654-671):**
```python
# Track if plan step changed
old_step = agent.last_plan_step

# ... agent.update() ...

# Generate step explanation if plan step changed
if agent.last_plan_step and agent.last_plan_step is not old_step:
    try:
        await agent.generate_step_explanation(memory_store)
    except Exception as e:
        logger.debug(f"Agent {agent.id} step explanation failed: {e}")
```

**Observation Summary (lines 701-706):**
```python
# Generate observation summary periodically (every 10 ticks)
if tick_count % 10 == agent.id % 10:  # Stagger across agents
    try:
        await agent.generate_observation_summary()
    except Exception as e:
        logger.debug(f"Agent {agent.id} observation summary failed: {e}")
```

**Conversation Tracking (lines 217-220 in agents.py):**
```python
# Track conversation for both agents
conversation_summary = f"{self.name} and {partner.name} had a conversation"
self.track_conversation(conversation_summary, [self.id, partner.id])
partner.track_conversation(conversation_summary, [self.id, partner.id])
```

### Serialization Updates

Both `Agent.to_dict()` and `serialize_agent_for_ai()` now include:

```python
"step_explanation": agent.step_explanation,
"observation_summary": agent.observation_summary,
"recent_conversations": agent.recent_conversations,
```

Additionally, `active_plan_step` payloads now include:

```python
"explanation": step.get('explanation')
```

---

## Frontend Implementation

### TypeScript Type Extensions

**File:** `ai-town/src/lib/minitownTypes.ts`

**Updated Interfaces:**

```typescript
export interface MiniTownPlanStep {
  start?: string | null;
  end?: string | null;
  description?: string | null;
  location?: [number, number] | null;
  explanation?: string | null;  // NEW
}

export interface MiniTownAgent {
  // ... existing fields ...
  step_explanation?: string | null;  // NEW
  observation_summary?: string | null;  // NEW
  recent_conversations?: MiniTownConversationDigest[];  // NEW
}

export interface MiniTownEvent {
  // ... existing fields ...
  summary?: string;  // NEW
}

export interface MiniTownSystemState {
  // ... existing fields ...
  observation_rollups?: MiniTownObservationRollup[];  // NEW
}
```

**New Interfaces:**

```typescript
export interface MiniTownConversationDigest {
  id?: string;
  timestamp?: string;
  summary: string;
  agents?: number[];
  step_description?: string | null;
}

export interface MiniTownObservationRollup {
  agent_id: number;
  agent_name: string;
  summary: string;
  generated_at?: string | null;
}
```

### UI Component Updates

#### MiniTownAgentInspector.tsx

**New Sections:**

1. **Step Rationale**
   - Displays `agent.step_explanation`
   - Shows contextual motivation for current step
   - Appears below "Active Step" section

2. **Observation Summary**
   - Parses `agent.observation_summary` into bullet points
   - Strips bullet markers and displays as list
   - Replaces wall of text with digestible themes

3. **Conversation Highlights**
   - Displays `agent.recent_conversations` or extracts from observations
   - Shows timestamp, summary, and related step
   - Provides social interaction history at a glance

4. **Active Step Enhancement**
   - Now displays `active_plan_step.explanation` inline
   - Provides immediate context for what the agent is doing

**Implementation Details:**
- Uses `useMemo` hooks to avoid recomputation on every render
- Falls back gracefully when data is unavailable
- Extracts `[CONVERSATION]` observations as fallback for `recent_conversations`

#### MiniTownSystemPanel.tsx

**New Sections:**

1. **Observation Highlights**
   - System-level view of agent observation summaries
   - Displays `systemState.observation_rollups` or individual `agent.observation_summary`
   - Limited to top 4 agents
   - Shows agent name, summary, and last update time

2. **Active Steps**
   - Panel showing all agents' current plan steps
   - Displays step description, time window, and explanation
   - Sorted alphabetically by agent name
   - Provides bird's-eye view of what everyone is doing

3. **Conversation Feed**
   - Filters events for type "conversation"
   - Shows timestamp, participants, and summary
   - Limited to 5 most recent conversations
   - Creates social narrative of town interactions

4. **Enhanced Recent Events**
   - Now displays `event.summary` for all event types
   - Provides richer context beyond just event type

**Implementation Details:**
- Uses `useMemo` with `nameById` lookup for efficient participant name resolution
- Formats relative timestamps for human readability
- Gracefully handles missing or partial data

---

## Seed Data

### PlanStepExplainer Seeds

**File:** `seeds/step_explainer_v1.json`

- **Total Seeds:** 24
- **Categories:** social (6), goal_oriented (6), routine (6), reactive (6)
- **Coverage:** Diverse agent goals (relationships, gardening, adventure, coffee, fitness, reading, painting)

**Example Seed:**
```json
{
  "id": 1,
  "agent_name": "Alice",
  "agent_goal": "Build relationships in the neighborhood",
  "agent_personality": "social, optimistic",
  "step_summary": "Attend party at Bob's house",
  "location": "(200, 150)",
  "recent_memories": "- Bob invited me to a party tonight\n- Carol mentioned she'll also be attending",
  "gold_explanation": "Attending Bob's party helps build neighborhood connections and strengthens existing friendships with Carol.",
  "category": "social"
}
```

### ObservationSummarizer Seeds

**File:** `seeds/observation_summarizer_v1.json`

- **Total Seeds:** 20
- **Categories:** social_interactions (5), goal_progress (5), environmental (5), mixed_themes (5)
- **Format:** Input observations with importance scores, output as thematic bullet points

**Example Seed:**
```json
{
  "id": 1,
  "agent_name": "Alice",
  "agent_goal": "Build relationships in the neighborhood",
  "observations": "• [0.85] Bob invited me to a party at 7pm tonight\n• [0.65] Carol mentioned she'll also be attending\n• [0.70] Emma asked about organizing a potluck",
  "gold_summary": "• Received party invitation from Bob for tonight with Carol attending\n• Community interest in social gatherings (potluck idea from Emma)\n• Made initial contact with new neighbor Frank",
  "category": "social_interactions"
}
```

---

## Compilation

### Compilation Script

**File:** `compilation/compile_ui_signatures.py`

**Features:**
- Compiles both `PlanStepExplainer` and `ObservationSummarizer` signatures
- Uses **BootstrapFewShot** optimizer (lighter weight than GEPA)
- Custom metrics reward conciseness, relevance, and format adherence
- Saves compiled modules to `compiled/` directory
- Dumps prompt snapshots for auditing

**Usage:**

```bash
# Compile both signatures (default)
python compilation/compile_ui_signatures.py

# Compile only step explainer
python compilation/compile_ui_signatures.py --signature step_explainer

# Compile only observation summarizer
python compilation/compile_ui_signatures.py --signature observation_summarizer

# Custom parameters
python compilation/compile_ui_signatures.py \
  --max-demos 8 \
  --provider groq \
  --model llama3-70b-8192
```

**Arguments:**
- `--signature`: Which signature(s) to compile (all, step_explainer, observation_summarizer)
- `--max-demos`: Maximum demonstrations for BootstrapFewShot (default: 6)
- `--provider`: LLM provider (groq, together, openai)
- `--model`: Model name

**Output:**
- `compiled/compiled_step_explainer.json`
- `compiled/compiled_observation_summarizer.json`
- `compiled/prompt_step_explainer.txt`
- `compiled/prompt_observation_summarizer.txt`

### Metrics

**PlanStepExplainer Metric:**
- +0.4: Mentions goal keywords
- +0.4: Appropriate length (1-3 sentences, 20-150 chars)
- +0.2: References the step
- Max score: 1.0

**ObservationSummarizer Metric:**
- +0.4: Uses bullet point format
- +0.4: Appropriate length (2-4 bullets, 50-250 chars)
- +0.2: References agent goal
- Max score: 1.0

---

## Usage

### Enabling Compiled Modules

Edit `config.yml`:

```yaml
compilation:
  use_compiled: true
  optimizer: "BootstrapFewShot"
```

The backend will automatically load compiled modules on startup if they exist in `compiled/` directory.

### Testing

1. **Start Backend:**
   ```bash
   cd backend
   ../mini-town/bin/python -m uvicorn main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd ai-town
   npm run dev
   ```

3. **Verify:**
   - Open browser to `http://localhost:5173`
   - Click on an agent to open inspector
   - Check for "Step Rationale", "Observation Summary", and "Conversation Highlights" sections
   - Open system panel and verify "Observation Highlights", "Active Steps", and "Conversation Feed" sections

### Debugging

**Backend Logs:**
```
DEBUG:backend.agents:Agent 1 step explanation: Attending Bob's party helps build...
DEBUG:backend.agents:Agent 2 observation summary: • Garden health excellent...
```

**Frontend Console:**
Check browser console for:
- WebSocket messages containing new fields
- React component render cycles with new data

**Missing Data:**
- If sections show "No summary available yet" or "No conversations yet", check:
  1. Backend is calling DSPy functions (check logs)
  2. LLM API is responding (check latency metrics)
  3. Agents have sufficient observations/plan steps to trigger generation

---

## Performance Considerations

### Backend Impact

**Step Explanation:**
- **Frequency:** Only when plan step changes (~every 5-10 minutes per agent)
- **LLM Calls:** 1 per step change
- **Latency:** ~500-1000ms per call (non-blocking)

**Observation Summary:**
- **Frequency:** Every 10 ticks (~20 seconds with default 2s tick interval)
- **Staggered:** Distributed across agents (agent.id % 10) to avoid spikes
- **LLM Calls:** ~1 per minute per agent (if observations exist)
- **Latency:** ~500-1500ms per call (non-blocking)

**Conversation Tracking:**
- **Frequency:** Real-time when conversations occur
- **No LLM Calls:** Simple string formatting
- **Latency:** Negligible (<1ms)

### Frontend Impact

**Data Volume:**
- Each agent payload increases by ~200-500 bytes
- WebSocket message size impact: ~5-15% increase
- No performance degradation observed with 10 agents

**Render Performance:**
- `useMemo` hooks prevent unnecessary recomputation
- Conversation/observation parsing only runs when data changes
- No noticeable lag in UI responsiveness

### Optimization Strategies

If performance issues arise:

1. **Reduce Step Explanation Frequency:**
   - Add cooldown timer (don't explain every step within X minutes)

2. **Reduce Summary Frequency:**
   - Change `tick_count % 10` to `tick_count % 20` (every 40 seconds)

3. **Limit Conversation History:**
   - Reduce `max_conversations` from 5 to 3 in `track_conversation()`

4. **Disable for Low-Priority Agents:**
   - Add agent priority system, only generate for "active" agents

---

## Future Enhancements

### Potential Additions

1. **Conversation Detail Signature:**
   - Generate multi-turn dialogue transcripts
   - Use `PlanConversation` signature (already defined but not integrated)

2. **Goal Progress Tracker:**
   - Analyze memories to assess goal completion percentage
   - Display progress bar in agent inspector

3. **Personality-Aware Summaries:**
   - Customize summary style based on agent personality
   - E.g., introverted agents get more introspective summaries

4. **System-Level Insights:**
   - "Town Digest" summarizing community-wide patterns
   - "Trending Topics" based on shared observations

5. **Historical Explanations:**
   - Store past explanations and summaries in database
   - Allow users to browse agent history timeline

### Compilation Improvements

1. **Larger Training Sets:**
   - Current: 20-24 seeds per signature
   - Target: 50-100 seeds for better generalization

2. **A/B Testing Framework:**
   - Compare compiled vs uncompiled UI signatures
   - Metric: user engagement time with agent inspector

3. **Iterative Refinement:**
   - Collect user feedback on explanation quality
   - Use feedback to refine seed data and recompile

---

## Troubleshooting

### Common Issues

**Issue:** "No summary available yet" persists after 5+ minutes
- **Cause:** Agent has no high-importance observations (all < 0.5)
- **Solution:** Lower threshold in `generate_observation_summary()` from 0.5 to 0.4

**Issue:** Step explanations are generic/unhelpful
- **Cause:** Limited memory context or poor retrieval
- **Solution:** Increase `top_k` from 5 to 10 in `generate_step_explanation()`

**Issue:** Conversations not appearing in feed
- **Cause:** Conversations not happening or tracking not triggered
- **Solution:** Check `_maybe_converse()` is being called and cooldowns aren't too long

**Issue:** Compilation fails with "API rate limit exceeded"
- **Cause:** Too many LLM calls during BootstrapFewShot
- **Solution:** Reduce `--max-demos` or use a provider with higher rate limits

**Issue:** Frontend shows old data after backend restart
- **Cause:** WebSocket reconnection delay
- **Solution:** Hard refresh browser (Cmd+Shift+R) or clear cache

---

## References

- **DSPy Documentation:** https://dspy-docs.vercel.app/
- **BootstrapFewShot:** https://dspy-docs.vercel.app/docs/building-blocks/optimizers#bootstrapfewshot
- **Mini-Town Architecture:** See `PROJECT_OVERVIEW.md`
- **Frontend Integration:** See `FRONTEND-INTEGRATION.md`

---

## Changelog

**2025-10-14:** Initial implementation
- Added PlanStepExplainer and ObservationSummarizer signatures
- Integrated with backend simulation loop
- Updated frontend components (AgentInspector, SystemPanel)
- Created seed data and compilation script
- Documentation complete

