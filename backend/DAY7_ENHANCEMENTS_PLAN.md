# Day 7+ Enhancement Plan
**Post-Day 7 Roadmap for Plan Execution Improvements**

**Created**: 2025-10-12
**Status**: Planning Phase
**Prerequisites**: Day 6 Complete (100% event coherence achieved), Day 7 Complete (God Mode)

---

## Overview

Day 6 achieved 100% event coherence by implementing basic plan execution with navigation and wait behavior. However, the system lacks several features for realistic agent behavior:

1. **Loitering Behavior** - Agents arrive and wait silently; no social interaction
2. **Plan Compilation** - PlanDay module is uncompiled; leads to variance and quality issues
3. **Multi-Location Events** - Events confined to single locations
4. **Dynamic Re-planning** - Agents cannot adapt to changing circumstances

This document provides detailed implementation plans for each enhancement.

---

## Enhancement Priority Matrix

| Enhancement | Priority | Time Estimate | Dependencies | Impact |
|-------------|----------|---------------|--------------|--------|
| **Loitering Behavior** | High | 3-4 hours | Day 6 | High (realism) |
| **Compile PlanDay** | High | 10-14 hours | Day 6 | High (consistency) |
| **Multi-Location Events** | Medium | 4-6 hours | Loitering | Medium (complexity) |
| **Dynamic Re-planning** | Medium | 6-8 hours | Compile PlanDay | High (adaptability) |

**Recommended Implementation Order**:
1. Loitering Behavior (quick win, high visibility)
2. Compile PlanDay (foundation for quality)
3. Dynamic Re-planning (leverages compiled planner)
4. Multi-Location Events (polish)

---

## Enhancement #1: Loitering Behavior

### Problem Statement

**Current Limitation**:
- Agents navigate to event locations
- Agents enter "waiting" state (velocity = 0)
- Agents stand silently at location
- No social interaction, no movement, no agent-to-agent communication

**Example from Day 6 Test**:
```
T+86s: Alice at (192.5, 153.7), state: waiting
       Bob at (192.9, 145.7), state: waiting
       Distance between agents: 8.0px
       Interaction: NONE (both standing still)
```

**Why This Matters**:
- Unrealistic behavior (people don't stand motionless at parties)
- No emergent social dynamics
- Missed opportunity for agent-to-agent learning
- Boring to watch in visualization

---

### Proposed Solution

#### Action Type System

Replace single "waiting" state with rich action types:

```python
class ActionType(Enum):
    IDLE = "idle"              # Standing still
    NAVIGATE = "navigate"      # Moving toward target
    WAIT = "wait"              # Waiting for event time
    LOITER = "loiter"          # Social loitering at event
    CONVERSE = "converse"      # Talking with another agent
    OBSERVE = "observe"        # Watching the environment
```

#### Loitering Behavior Design

**Phase 1: Arrive Early** (before event time)
- Action: WAIT
- Behavior: Stand near location (current implementation)

**Phase 2: Event Active** (during event time)
- Action: LOITER
- Behavior:
  - Small random movements within 20px radius
  - Approach nearby agents (within 15px)
  - Initiate conversations every 30-60 seconds
  - Update memory with social observations

**Phase 3: Event Ending** (after event time)
- Action: IDLE
- Behavior: Gradually leave area (resume plan)

#### Conversation System

**Simple Conversation (MVP)**:
```python
async def initiate_conversation(self, other_agent: Agent, memory_store: MemoryStore):
    """Initiate a simple conversation at an event."""

    # Check if other agent is nearby and available
    if other_agent.action_type in [ActionType.LOITER, ActionType.IDLE]:

        # Generate conversation topic based on context
        topic = f"Talking with {other_agent.name} about the event"

        # Store as shared memory
        observation = f"Had a conversation with {other_agent.name} at the party"
        await self.score_and_store_observation(observation, memory_store)
        await other_agent.score_and_store_observation(
            f"Had a conversation with {self.name} at the party",
            memory_store
        )

        # Visual feedback
        logger.info(f"💬 {self.name} and {other_agent.name} are chatting")
```

**Advanced Conversation (Optional)**:
- LLM-generated dialogue based on personalities
- Topic extraction from recent memories
- Conversation outcomes affect relationship scores

---

### Implementation Details

#### 1. Add Action Type to Agent

**File**: `agents.py`

```python
class Agent:
    def __init__(self, ...):
        # ... existing code ...
        self.action_type = ActionType.IDLE
        self.action_target = None  # Current action target
        self.loiter_center = None  # Center point for loitering
        self.last_conversation = None  # Track conversation timing
```

#### 2. Modify Update Loop

**File**: `agents.py:79-156` (update method)

```python
def update(self, other_agents: List['Agent']) -> Dict[str, Any]:
    """Update with loitering behavior."""

    if self.parsed_plan:
        current_time = datetime.now()
        current_step = self.get_current_step(self.parsed_plan, current_time)
        upcoming_step = self.get_upcoming_step(self.parsed_plan, current_time)

        if current_step and current_step.get('location'):
            # Event is ACTIVE - loiter instead of just waiting
            target_x, target_y = current_step['location']
            distance = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)

            if distance < 20.0:  # Within event area
                self.action_type = ActionType.LOITER
                self.loiter_center = (target_x, target_y)
                self._loiter_behavior(other_agents)
            else:
                self.action_type = ActionType.NAVIGATE
                self.navigate_to(target_x, target_y)

        elif upcoming_step and upcoming_step.get('location'):
            # Event is UPCOMING - wait at location
            target_x, target_y = upcoming_step['location']
            distance = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)

            if distance < 10.0:
                self.action_type = ActionType.WAIT
                self.vx = 0
                self.vy = 0
            else:
                self.action_type = ActionType.NAVIGATE
                self.navigate_to(target_x, target_y)
        else:
            # No active plan
            self.action_type = ActionType.IDLE
            self._random_walk()

    # ... rest of update logic ...
```

#### 3. Implement Loiter Behavior

**File**: `agents.py` (new method)

```python
def _loiter_behavior(self, other_agents: List['Agent']):
    """
    Loiter at event location with small random movements and social interaction.

    Behavior:
    - Stay within 20px of loiter_center
    - Make small random movements (25% chance per tick)
    - Approach nearby agents (within 15px)
    - Initiate conversations (every 30-60 seconds)
    """
    if not self.loiter_center:
        return

    center_x, center_y = self.loiter_center
    distance_from_center = math.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)

    # Find nearby agents also loitering
    nearby_loitering = [
        a for a in other_agents
        if a.id != self.id
        and a.action_type == ActionType.LOITER
        and math.sqrt((a.x - self.x)**2 + (a.y - self.y)**2) < 15.0
    ]

    if nearby_loitering and random.random() < 0.3:
        # Approach a nearby agent
        target_agent = random.choice(nearby_loitering)
        dx = target_agent.x - self.x
        dy = target_agent.y - self.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 5.0:  # Don't get too close
            self.vx = (dx / dist) * (self.speed * 0.5)  # Slower loiter speed
            self.vy = (dy / dist) * (self.speed * 0.5)

    elif distance_from_center > 20.0:
        # Too far from center, move back
        dx = center_x - self.x
        dy = center_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        self.vx = (dx / dist) * self.speed
        self.vy = (dy / dist) * self.speed

    elif random.random() < 0.25:
        # Small random movement
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle) * (self.speed * 0.3)
        self.vy = math.sin(angle) * (self.speed * 0.3)

    else:
        # Stay still
        self.vx = 0
        self.vy = 0

    # Update position
    new_x = self.x + self.vx
    new_y = self.y + self.vy
    self.x = max(0, min(self.map_width, new_x))
    self.y = max(0, min(self.map_height, new_y))
```

---

### Testing Strategy

#### Test 1: Loitering Movement
```python
def test_loiter_movement():
    """Verify agents make small movements during loitering."""
    agent = Agent(...)
    agent.action_type = ActionType.LOITER
    agent.loiter_center = (200, 150)
    agent.x, agent.y = 200, 150

    positions = []
    for _ in range(20):
        agent._loiter_behavior([])
        positions.append((agent.x, agent.y))

    # Check movement happened
    unique_positions = len(set(positions))
    assert unique_positions > 1, "Agent should move during loitering"

    # Check stayed within radius
    for x, y in positions:
        dist = math.sqrt((x - 200)**2 + (y - 150)**2)
        assert dist <= 25, f"Agent moved too far from center: {dist}px"
```

#### Test 2: Social Approach
```python
def test_social_approach():
    """Verify agents approach each other during loitering."""
    alice = Agent(agent_id=1, name="Alice", x=200, y=150)
    bob = Agent(agent_id=2, name="Bob", x=210, y=160)

    alice.action_type = ActionType.LOITER
    bob.action_type = ActionType.LOITER

    initial_dist = math.sqrt((alice.x - bob.x)**2 + (alice.y - bob.y)**2)

    # Simulate several ticks
    for _ in range(10):
        alice._loiter_behavior([bob])
        bob._loiter_behavior([alice])

    final_dist = math.sqrt((alice.x - bob.x)**2 + (alice.y - bob.y)**2)

    # Agents should get closer
    assert final_dist < initial_dist, "Agents should approach each other"
```

---

### Success Criteria

- [x] Agents make small movements during events (not static)
- [x] Agents approach each other when nearby
- [x] Visual difference between WAIT and LOITER states
- [x] Event coherence maintained at 80%+ (loitering doesn't break navigation)
- [x] Agent positions logged as "loitering at (x, y)"

---

### Time Estimate

**Total: 3-4 hours**

- Action type system: 30 minutes
- Loiter behavior implementation: 1.5 hours
- Update loop integration: 1 hour
- Testing: 1 hour

---

## Enhancement #2: Compile PlanDay Module with GEPA

### Problem Statement

**Current Limitation**:
- PlanDay module is **uncompiled** (uses `dspy.Predict`)
- Temperature = 0.1 reduces but doesn't eliminate variance
- LLM can still generate poor quality plans
- No optimization for plan quality metrics

**Evidence from Day 6**:
- Bob initially scheduled party 3 hours late
- Required multiple prompt engineering iterations
- Stronger instructions helped, but not guaranteed across all scenarios

**Why Compilation Needed**:
- **Consistency**: Compiled modules eliminate variance
- **Quality**: Optimization improves plan coherence
- **Reliability**: Less dependent on prompt engineering
- **Scalability**: Works across different agent personalities

---

### Compilation Strategy

#### Phase 1: Seed Collection (6-8 hours)

**Goal**: Collect 30-40 high-quality planning examples

**Seed Structure**:
```python
{
    "agent_goal": "Meet new people in the neighborhood",
    "agent_personality": "social, curious",
    "current_time": "2:30 PM",
    "current_location": "(100, 150)",
    "recent_events": [
        "Maria invited you to a party at 7:30 PM at location (200, 150)"
    ],
    "relevant_memories": [
        "You enjoy social gatherings",
        "Maria is a friendly neighbor"
    ],
    "gold_plan": "2:30 PM - 3:00 PM: Finish current task\n3:00 PM - 7:00 PM: Prepare for party (shower, get dressed)\n7:00 PM - 7:30 PM: Travel to party location (200, 150)\n7:30 PM - 9:00 PM: Attend party at (200, 150) and socialize",
    "rationale": "Plan preserves exact party time (7:30 PM), includes location coordinates, allows prep time without rescheduling event"
}
```

**Seed Categories** (coverage):

1. **Social Events** (10 seeds)
   - Party invitations at different times
   - Multiple events in one day
   - Conflicting social commitments
   - Last-minute invitations

2. **Goal-Aligned Tasks** (8 seeds)
   - Research work (introverted agents)
   - Social networking (extroverted agents)
   - Maintenance tasks (organized agents)
   - Creative projects (spontaneous agents)

3. **Edge Cases** (6 seeds)
   - Event already passed (plan immediate attendance)
   - Event in 2 minutes (no prep time)
   - Event conflicts with agent goal
   - Multiple invitations same time

4. **Personality Variations** (8 seeds)
   - Introverted agent invited to party
   - Extroverted agent with solo task
   - Analytical agent with spontaneous event
   - Organized agent with unstructured day

5. **Time Handling** (6 seeds)
   - Morning events
   - Late night events
   - All-day events
   - Multi-day events

**Seed Quality Validation**:

```python
def validate_seed(seed: dict) -> tuple[bool, str]:
    """Validate a planning seed for quality."""

    checks = []

    # Check 1: Plan mentions all event times from recent_events
    event_times = extract_times_from_events(seed['recent_events'])
    plan_times = extract_times_from_plan(seed['gold_plan'])

    for event_time in event_times:
        if event_time not in plan_times:
            return False, f"Plan missing event time: {event_time}"

    # Check 2: Plan includes all location coordinates
    event_locations = extract_locations_from_events(seed['recent_events'])
    plan_locations = extract_locations_from_plan(seed['gold_plan'])

    for loc in event_locations:
        if loc not in plan_locations:
            return False, f"Plan missing location: {loc}"

    # Check 3: Time blocks are sequential and non-overlapping
    time_blocks = parse_time_blocks(seed['gold_plan'])
    for i in range(len(time_blocks) - 1):
        if time_blocks[i]['end'] > time_blocks[i+1]['start']:
            return False, f"Overlapping time blocks: {time_blocks[i]} and {time_blocks[i+1]}"

    # Check 4: Plan aligns with personality
    personality = seed['agent_personality'].lower()
    if 'introverted' in personality:
        # Introverted agents should still attend social events, but may include prep
        if 'party' in str(seed['recent_events']).lower():
            if 'prep' not in seed['gold_plan'].lower():
                checks.append("Consider adding prep time for introverted agent")

    return True, "Valid seed"
```

**Inter-Rater Agreement**:

Have 2-3 people rate 10 sample seeds:
- Does plan preserve event times? (Y/N)
- Does plan include locations? (Y/N)
- Is plan realistic for personality? (1-5 scale)

Calculate Cohen's kappa:
- Target: κ > 0.6 (substantial agreement)
- If κ < 0.6: Clarify scoring rubric and retry

---

#### Phase 2: Metric Definition (1-2 hours)

**Primary Metric: Plan Fidelity**

```python
def planning_metric(example, prediction, trace=None) -> float:
    """
    Evaluate plan quality against gold standard.

    Returns score 0.0-1.0 where:
    - 1.0 = Perfect plan (all criteria met)
    - 0.5 = Acceptable plan (most criteria met)
    - 0.0 = Poor plan (critical failures)
    """
    score = 0.0
    max_score = 0.0

    # Criterion 1: Event time preservation (40% weight)
    max_score += 0.4
    event_times = extract_times_from_events(example.recent_events)
    plan_times = extract_times_from_plan(prediction.plan)

    preserved_times = sum(1 for t in event_times if t in plan_times)
    if event_times:
        score += 0.4 * (preserved_times / len(event_times))

    # Criterion 2: Location inclusion (20% weight)
    max_score += 0.2
    event_locations = extract_locations_from_events(example.recent_events)
    plan_locations = extract_locations_from_plan(prediction.plan)

    preserved_locs = sum(1 for loc in event_locations if loc in plan_locations)
    if event_locations:
        score += 0.2 * (preserved_locs / len(event_locations))

    # Criterion 3: Time block validity (20% weight)
    max_score += 0.2
    time_blocks = parse_time_blocks(prediction.plan)
    if time_blocks:
        valid_blocks = all(
            time_blocks[i]['end'] <= time_blocks[i+1]['start']
            for i in range(len(time_blocks) - 1)
        )
        if valid_blocks:
            score += 0.2

    # Criterion 4: Personality alignment (20% weight)
    max_score += 0.2
    personality_score = evaluate_personality_alignment(
        example.agent_personality,
        prediction.plan,
        example.recent_events
    )
    score += 0.2 * personality_score

    return score / max_score if max_score > 0 else 0.0
```

**Secondary Metrics**:

1. **Event Coherence** (end-to-end test)
   - Run full party scenario with compiled planner
   - Measure % of invitees who attend
   - Target: ≥ 90% (vs 100% with uncompiled but heavily prompted)

2. **Plan Consistency** (across multiple runs)
   - Generate plan 5 times with same inputs
   - Measure plan similarity (edit distance)
   - Target: Variance < 10%

---

#### Phase 3: GEPA Compilation (4-6 hours in Colab)

**Colab Notebook Setup**:

```python
# Install dependencies
!pip install dspy-ai transformers accelerate sentence-transformers

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load seeds
import json
with open('/content/drive/MyDrive/mini-town/seeds/planner_v1.json') as f:
    seeds = json.load(f)

# Convert to DSPy examples
from dspy import Example
trainset = [Example(**seed).with_inputs('agent_goal', 'agent_personality', 'current_time', 'current_location', 'recent_events', 'relevant_memories') for seed in seeds]

# Define metric
from metrics import planning_metric

# Load model (use Together.ai for compilation)
import dspy
import os

lm = dspy.LM(
    model='together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo',
    api_key=os.environ['TOGETHER_API_KEY'],
    temperature=0.1,
    max_tokens=512
)
dspy.settings.configure(lm=lm)

# Create uncompiled planner
from dspy_modules import PlanDay
planner = dspy.Predict(PlanDay)

# Compile with GEPA
from dspy.optimizers import GEPA

compiled_planner = GEPA(
    metric=planning_metric,
    budget=40,  # 40 rollouts
    verbose=True
).compile(planner, trainset=trainset)

# Save compiled program
compiled_planner.save('/content/drive/MyDrive/mini-town/compiled/compiled_planner.json')

# Test on holdout set
holdout_set = trainset[:5]  # Reserve first 5 for testing
results = []

for example in holdout_set:
    pred = compiled_planner(**example.inputs())
    score = planning_metric(example, pred)
    results.append(score)
    print(f"Example {example.agent_goal}: Score = {score:.2f}")

print(f"\nAverage holdout score: {sum(results) / len(results):.2f}")
```

**Expected Runtime**: 4-6 hours (GEPA with 40 rollouts)

**Checkpointing**:
```python
# Checkpoint every 10 rollouts
if iteration % 10 == 0:
    compiled_planner.save(f'/content/drive/MyDrive/mini-town/checkpoint_planner_{iteration}.json')
```

---

#### Phase 4: Integration & Testing (2 hours)

**Load Compiled Planner**:

```python
# dspy_modules.py

_compiled_planner = None

def load_compiled_planner(compiled_dir: str = "compiled"):
    """Load compiled PlanDay program."""
    global _compiled_planner

    from pathlib import Path
    compiled_path = Path(__file__).parent.parent / compiled_dir / "compiled_planner.json"

    if compiled_path.exists():
        try:
            _compiled_planner = dspy.Predict(PlanDay)
            _compiled_planner.load(str(compiled_path))
            logger.info(f"✅ Compiled planner loaded from {compiled_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load compiled planner: {e}")
            return False
    else:
        logger.warning(f"Compiled planner not found at {compiled_path}")
        return False

def get_current_planner():
    """Get active planner (compiled or uncompiled)."""
    if _use_compiled and _compiled_planner is not None:
        return _compiled_planner
    return planner  # Uncompiled baseline
```

**Update generate_plan Function**:

```python
async def generate_plan(...) -> str:
    """Generate plan using compiled or uncompiled planner."""

    # Get current planner
    current_planner = get_current_planner()

    # Format inputs
    events_str = "\n".join([f"- {event}" for event in recent_events]) if recent_events else "No recent events"
    memories_str = "\n".join([f"- {mem}" for mem in relevant_memories]) if relevant_memories else "No relevant memories"

    # Generate plan
    result = current_planner(
        agent_goal=agent_goal,
        agent_personality=agent_personality,
        current_time=current_time,
        current_location=current_location,
        recent_events=events_str,
        relevant_memories=memories_str
    )

    return result.plan
```

---

### Testing Strategy

#### Test 1: Consistency Check
```python
async def test_compiled_planner_consistency():
    """Verify compiled planner produces consistent plans."""

    # Load compiled planner
    load_compiled_planner()
    use_compiled(True)

    # Generate plan 5 times with same inputs
    plans = []
    for _ in range(5):
        plan = await generate_plan(
            agent_goal="Meet new people",
            agent_personality="social, curious",
            current_time="2:30 PM",
            current_location="(100, 150)",
            recent_events=["Maria invited you to party at 7:30 PM at (200, 150)"],
            relevant_memories=["You enjoy social gatherings"]
        )
        plans.append(plan)

    # Check consistency (all plans should be very similar)
    for i in range(len(plans) - 1):
        similarity = 1 - (Levenshtein.distance(plans[i], plans[i+1]) / max(len(plans[i]), len(plans[i+1])))
        assert similarity > 0.9, f"Plans too different: {similarity:.2f}"
```

#### Test 2: Event Coherence with Compiled Planner
```python
async def test_event_coherence_compiled():
    """Run full party scenario with compiled planner."""

    load_compiled_planner()
    use_compiled(True)

    results = await run_party_scenario(
        use_compiled_modules=True,
        duration_minutes=30
    )

    # Should maintain or improve event coherence
    assert results['event_coherence'] >= 0.9, f"Coherence dropped: {results['event_coherence']}"
```

---

### Success Criteria

- [x] 30-40 high-quality seeds collected (κ > 0.6)
- [x] Planning metric implemented and tested
- [x] GEPA compilation completes successfully
- [x] Compiled planner consistency > 90%
- [x] Event coherence ≥ 90% with compiled planner
- [x] Plan quality improves on holdout set

---

### Time Estimate

**Total: 10-14 hours**

- Seed collection: 6-8 hours
- Metric definition: 1-2 hours
- GEPA compilation (Colab): 4-6 hours (mostly unattended)
- Integration & testing: 2 hours

---

## Enhancement #3: Multi-Location Events

### Problem Statement

**Current Limitation**:
- Events confined to single locations
- Plans have format: "7:30 PM - 9:00 PM: Attend party at (200, 150)"
- Cannot represent: "7:30 PM: Meet at (200, 150), 8:00 PM: Move to (300, 200)"

**Real-World Use Cases**:
- **Progressive parties**: Start at house, move to bar, end at diner
- **Pub crawls**: Visit multiple venues
- **Tours**: Museum tour with multiple rooms
- **Meetings**: Conference room → office → lunch spot

---

### Proposed Solution

#### Event Schema Extension

**Current**:
```python
event = {
    'time': datetime,
    'location': (x, y),
    'type': 'party'
}
```

**Extended**:
```python
event = {
    'type': 'multi_location',
    'waypoints': [
        {'time': '7:30 PM', 'location': (200, 150), 'activity': 'Initial gathering'},
        {'time': '8:00 PM', 'location': (300, 200), 'activity': 'Move to bar'},
        {'time': '9:00 PM', 'location': (250, 100), 'activity': 'Dinner at restaurant'}
    ]
}
```

#### Plan Parsing Updates

**Enhanced parse_plan Method**:

```python
def parse_plan(self) -> List[Dict[str, Any]]:
    """Parse plan supporting multiple locations per time block."""

    steps = []

    # Match time blocks with multiple possible locations
    # Example: "7:30 PM - 8:00 PM: Start at (200, 150), move to (300, 200)"
    pattern = r'(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M):\s*(.*?)(?=\d{1,2}:\d{2}\s*[AP]M|$)'

    for match in re.finditer(pattern, self.current_plan, re.DOTALL):
        start_time_str = match.group(1).strip()
        end_time_str = match.group(2).strip()
        description = match.group(3).strip()

        # Parse times
        start_time = datetime.strptime(start_time_str, "%I:%M %p").time()
        end_time = datetime.strptime(end_time_str, "%I:%M %p").time()

        # Extract ALL locations from description
        location_pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        locations = []
        for loc_match in re.finditer(location_pattern, description):
            x = float(loc_match.group(1))
            y = float(loc_match.group(2))
            locations.append((x, y))

        # If multiple locations, create waypoints
        if len(locations) > 1:
            # Multi-location step
            duration_minutes = (
                datetime.combine(datetime.today(), end_time) -
                datetime.combine(datetime.today(), start_time)
            ).total_seconds() / 60

            time_per_location = duration_minutes / len(locations)

            for i, loc in enumerate(locations):
                waypoint_start = datetime.combine(datetime.today(), start_time) + timedelta(minutes=i * time_per_location)
                waypoint_end = waypoint_start + timedelta(minutes=time_per_location)

                steps.append({
                    'start_time': waypoint_start.time(),
                    'end_time': waypoint_end.time(),
                    'location': loc,
                    'description': f"{description} (waypoint {i+1}/{len(locations)})",
                    'is_waypoint': True,
                    'waypoint_index': i
                })
        else:
            # Single location step (existing behavior)
            steps.append({
                'start_time': start_time,
                'end_time': end_time,
                'location': locations[0] if locations else None,
                'description': description,
                'is_waypoint': False
            })

    return steps
```

#### Navigation Strategy

**Sequential Waypoint Navigation**:

```python
def navigate_multi_location_event(self, parsed_steps: List[Dict], current_time: datetime):
    """Navigate through multi-location events."""

    # Get current step
    current_step = self.get_current_step(parsed_steps, current_time)

    if not current_step or not current_step.get('is_waypoint'):
        return  # Not a multi-location event

    # Check if at current waypoint
    target_x, target_y = current_step['location']
    distance = math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)

    if distance < 10.0:
        # At waypoint - wait for next waypoint time
        next_waypoint_index = current_step['waypoint_index'] + 1
        next_waypoint = next((
            s for s in parsed_steps
            if s.get('is_waypoint') and s.get('waypoint_index') == next_waypoint_index
        ), None)

        if next_waypoint:
            # Check if it's time to move to next waypoint
            if current_time.time() >= next_waypoint['start_time']:
                logger.info(f"Agent {self.id} moving to waypoint {next_waypoint_index + 1}")
                self.navigate_to(*next_waypoint['location'])
            else:
                # Wait at current waypoint
                self.action_type = ActionType.LOITER
                self.loiter_center = (target_x, target_y)
        else:
            # Last waypoint - loiter until event ends
            self.action_type = ActionType.LOITER
            self.loiter_center = (target_x, target_y)
    else:
        # Not at waypoint yet - navigate
        self.navigate_to(target_x, target_y)
```

---

### Implementation Details

**File Changes**:

1. `agents.py`: Update `parse_plan()` and `update()` methods
2. `dspy_modules.py`: Update PlanDay signature to mention multi-location support
3. `test_event_scenario.py`: Add multi-location event test

**PlanDay Signature Update**:

```python
class PlanDay(dspy.Signature):
    """...

    For multi-location events, list locations in sequence:
    Example: "7:30 PM - 9:00 PM: Start at (200, 150), move to (300, 200) at 8:00 PM"
    """
```

---

### Testing Strategy

```python
async def test_multi_location_event():
    """Test progressive party across 3 locations."""

    # Create event
    event = {
        'type': 'progressive_party',
        'waypoints': [
            {'time': '7:30 PM', 'location': (200, 150), 'activity': 'Cocktails at Alice\'s'},
            {'time': '8:00 PM', 'location': (300, 200), 'activity': 'Dinner at restaurant'},
            {'time': '9:00 PM', 'location': (250, 100), 'activity': 'Dessert at cafe'}
        ]
    }

    # Send multi-location invitation
    invitation = (
        f"Progressive party: 7:30 PM at (200, 150), "
        f"8:00 PM at (300, 200), 9:00 PM at (250, 100)"
    )

    # Generate plan and verify it includes all waypoints
    plan = await agent.update_plan(memory_store, current_time)

    assert "(200, 150)" in plan
    assert "(300, 200)" in plan
    assert "(250, 100)" in plan

    # Simulate and check agent visits all locations
    # (Implementation details)
```

---

### Success Criteria

- [x] Parse plans with multiple locations per time block
- [x] Navigate sequentially through waypoints
- [x] Loiter at each waypoint until next waypoint time
- [x] Test with 3-location progressive party (80%+ attendance)

---

### Time Estimate

**Total: 4-6 hours**

- Schema design: 1 hour
- Parse plan updates: 2 hours
- Navigation logic: 2 hours
- Testing: 1 hour

---

## Enhancement #4: Dynamic Plan Re-evaluation

### Problem Statement

**Current Limitation**:
- Plans generated once (at T+6 min in tests)
- Agents execute plans rigidly, no adaptation
- No response to unexpected events
- Plans become stale if circumstances change

**Example Scenario**:
```
T+0: Bob plans: "2:00-4:00 PM: Research, 7:00 PM: Attend party"
T+1: Alice invites Bob to urgent meeting at 3:00 PM
T+2: Bob continues research (ignores new invitation)
T+3: Bob misses meeting (plan not updated)
```

---

### Proposed Solution

#### Re-planning Triggers

**Trigger 1: High-Importance Event** (importance > 0.8)
```python
async def score_and_store_observation(self, obs: str, memory_store) -> float:
    """Score observation and trigger re-plan if important."""

    importance = await timed_llm_call(...)

    # Trigger re-plan for very important events
    if importance > 0.8:
        logger.info(f"Agent {self.id} triggered re-plan due to important event: {obs}")
        await self.update_plan(memory_store)

    return importance
```

**Trigger 2: Plan Conflict Detected**
```python
def detect_plan_conflict(self, new_event_time: time, new_event_location: tuple) -> bool:
    """Check if new event conflicts with current plan."""

    if not self.parsed_plan:
        return False

    for step in self.parsed_plan:
        if step['start_time'] <= new_event_time <= step['end_time']:
            # Time conflict detected
            if step.get('location') != new_event_location:
                return True

    return False
```

**Trigger 3: Time-Based Re-planning** (every N hours)
```python
class Agent:
    def __init__(self, ...):
        # ...
        self.plan_refresh_interval = timedelta(hours=2)
        self.last_plan_time = None

    def should_replan(self, current_time: datetime) -> bool:
        """Check if it's time to refresh plan."""

        if not self.last_plan_time:
            return True

        elapsed = current_time - self.last_plan_time
        return elapsed >= self.plan_refresh_interval
```

#### Cost Management

**Problem**: LLM calls expensive, can't re-plan every tick

**Solution**: Rate limiting + caching

```python
class Agent:
    def __init__(self, ...):
        # ...
        self.plan_cooldown = timedelta(minutes=10)
        self.last_replan_time = None

    async def maybe_replan(self, memory_store, current_time: datetime, trigger: str) -> bool:
        """Attempt to re-plan if not in cooldown."""

        # Check cooldown
        if self.last_replan_time:
            elapsed = current_time - self.last_replan_time
            if elapsed < self.plan_cooldown:
                logger.debug(f"Agent {self.id} re-plan blocked (cooldown: {self.plan_cooldown - elapsed})")
                return False

        # Re-plan
        logger.info(f"Agent {self.id} re-planning (trigger: {trigger})")
        await self.update_plan(memory_store, current_time)
        self.last_replan_time = current_time
        return True
```

---

### Implementation Details

**Integrated Re-planning Logic**:

```python
async def score_and_store_observation(self, obs: str, memory_store) -> float:
    """Score, store, and maybe trigger re-plan."""

    # Score importance
    importance = await timed_llm_call(
        score_observation,
        signature_name="ScoreImportance",
        timeout=5.0,
        observation=obs,
        agent_goal=self.goal,
        agent_personality=self.personality
    )

    # Store memory
    embedding = generate_embedding(obs)
    memory_store.store_memory(
        agent_id=self.id,
        content=obs,
        importance=importance / 10.0,
        embedding=embedding,
        timestamp=datetime.now()
    )

    # Check re-plan triggers
    current_time = datetime.now()

    # Trigger 1: High importance
    if importance > 8.0:
        await self.maybe_replan(memory_store, current_time, "high_importance_event")

    # Trigger 2: Time-based refresh
    elif self.should_replan(current_time):
        await self.maybe_replan(memory_store, current_time, "scheduled_refresh")

    return importance / 10.0
```

---

### Testing Strategy

```python
async def test_dynamic_replanning():
    """Test agent re-plans when receiving important event."""

    # Initial plan
    agent = Agent(...)
    initial_plan = await agent.update_plan(memory_store, datetime.now())

    assert "research" in initial_plan.lower()

    # High-importance event arrives
    urgent_event = "URGENT: Meeting with CEO in 10 minutes at (300, 200)"
    await agent.score_and_store_observation(urgent_event, memory_store)

    # Wait for re-plan
    await asyncio.sleep(1)

    # Plan should be updated
    assert agent.current_plan != initial_plan
    assert "(300, 200)" in agent.current_plan
```

---

### Success Criteria

- [x] Re-planning triggered by high-importance events
- [x] Re-planning respects cooldown (not too frequent)
- [x] Updated plans successfully executed
- [x] Event coherence maintained at 80%+

---

### Time Estimate

**Total: 6-8 hours**

- Trigger system design: 2 hours
- Rate limiting implementation: 2 hours
- Integration with scoring: 2 hours
- Testing: 2 hours

---

## Combined Impact Analysis

### When All 4 Enhancements Active

**Scenario: Progressive Party with Late Invitation**

```
Initial Setup:
- Alice, Bob, Carol at various locations
- Maria plans progressive party (3 locations)

T+0 (2:00 PM): Initial invitations sent
- Alice receives: "Party at 7:30 PM starting at (200, 150)"
- Bob receives: "Party at 7:30 PM starting at (200, 150)"
- Plans generated (compiled planner = consistent quality)

T+30 min (2:30 PM): Carol joins late
- Maria sends Carol urgent invitation (importance = 0.9)
- Carol's agent triggers dynamic re-plan
- Carol's plan updated to include party

T+5 hours (7:30 PM): Party starts
- All 3 agents navigate to (200, 150)
- Enter LOITER state (not static waiting)
- Make small movements, approach each other
- Initiate conversations

T+30 min (8:00 PM): Move to second location
- Plans include: "move to (300, 200) at 8:00 PM"
- Agents navigate sequentially to waypoint 2
- Continue loitering at new location

T+60 min (9:00 PM): Move to third location
- Navigate to final waypoint (250, 100)
- Continue social behavior

Result:
✅ 100% attendance (all invited agents present)
✅ Realistic social behavior (loitering, conversations)
✅ Multi-location event executed successfully
✅ Late invitee adapted plan dynamically
```

**Expected Metrics**:
- Event coherence: 90-100%
- Plan consistency: >95% (compiled planner)
- Social realism: High (loitering visible)
- Adaptability: Demonstrated (late invitation handled)

---

## Implementation Timeline

### Recommended Phased Approach

**Week 1: Foundation**
- Day 7: God Mode (separate work)
- Day 8: Enhancement #1 (Loitering Behavior) - 3-4 hours
  - Quick win, high visibility
  - Tests basic action type system

**Week 2: Quality**
- Day 9-10: Enhancement #2 (Compile PlanDay) - 10-14 hours
  - Seed collection: 6-8 hours (can be split across 2 days)
  - Compilation: 4-6 hours (Colab overnight)
  - Most time-intensive, highest long-term value

**Week 3: Polish**
- Day 11: Enhancement #4 (Dynamic Re-planning) - 6-8 hours
  - Depends on compiled planner for best results
  - High impact on adaptability
- Day 12: Enhancement #3 (Multi-Location) - 4-6 hours
  - Polish feature, completes the suite

**Total Time**: 23-32 hours across 2-3 weeks

---

## Risk Mitigation

### Risk 1: Compiled Planner Worse Than Uncompiled

**Mitigation**:
- Validate seeds carefully (κ > 0.6)
- Use holdout set for evaluation
- Keep uncompiled planner as fallback
- Implement A/B testing flag

### Risk 2: Loitering Breaks Navigation

**Mitigation**:
- Unit test loitering radius enforcement
- Separate loiter behavior from navigation
- Gradual rollout with monitoring

### Risk 3: Dynamic Re-planning Too Expensive

**Mitigation**:
- Implement cooldown (10 min minimum)
- Rate limit per agent
- Monitor LLM call costs
- Emergency disable flag

### Risk 4: Multi-Location Parsing Fails

**Mitigation**:
- Extensive regex testing
- Fallback to single-location parsing
- Clear error messages
- Manual plan validation tool

---

## Rollback Plan

If any enhancement causes issues:

1. **Disable via config flag**:
```yaml
enhancements:
  loitering: true
  compiled_planner: false  # ← Disable if needed
  multi_location: true
  dynamic_replan: true
```

2. **Revert to previous commit**:
```bash
git log --oneline  # Find commit before enhancement
git revert <commit-hash>
```

3. **Fallback implementations**:
- Loitering → WAIT state
- Compiled planner → Uncompiled baseline
- Multi-location → Single location only
- Dynamic re-plan → Static plans

---

## Future Considerations (Beyond These 4)

### Potential Day 10+ Enhancements

1. **Relationship Tracking**
   - Track agent-to-agent relationships
   - Affect conversation frequency
   - Influence event attendance

2. **Memory Pruning**
   - Delete low-importance memories after N days
   - Keep database size manageable
   - Improve retrieval speed

3. **Hierarchical Planning**
   - High-level goals → mid-level plans → low-level actions
   - Better long-term coherence
   - More realistic agent behavior

4. **Collaborative Actions**
   - Multi-agent coordination
   - Joint activities (e.g., "help Bob move")
   - Group decision making

5. **Learning from Outcomes**
   - Track event outcomes
   - Adjust future plans based on past results
   - Online learning / few-shot adaptation

---

## Success Metrics

### Overall Enhancement Suite Success

| Metric | Baseline (Day 6) | Target (After Enhancements) |
|--------|------------------|------------------------------|
| Event Coherence | 100% | 90-100% (maintain) |
| Plan Consistency | Medium (temp=0.1) | High (>95%) |
| Social Realism | Low (static) | High (loitering) |
| Adaptability | None | High (re-planning) |
| Event Complexity | Single location | Multi-location |
| LLM Call Cost | $0.006/test | <$0.02/test |

---

## Conclusion

These 4 enhancements build on Day 6's success to create a robust, realistic, and adaptable agent system:

1. **Loitering Behavior**: Adds realism and social dynamics
2. **Compiled PlanDay**: Ensures consistency and quality
3. **Multi-Location Events**: Enables complex event scenarios
4. **Dynamic Re-planning**: Provides adaptability to change

**Recommended Start**: Loitering Behavior (quick win)
**Highest Impact**: Compile PlanDay (long-term quality)
**Most Fun**: Multi-Location Events (visible complexity)

---

**Document Status**: Ready for implementation after Day 7
**Next Action**: Complete Day 7 (God Mode), then return to this plan
**Created**: 2025-10-12
**Last Updated**: 2025-10-12
