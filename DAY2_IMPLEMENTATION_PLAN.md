# Day 2 Implementation Plan - Detailed Steps

**Date**: 2025-10-11
**Status**: In Progress (Phase 1 complete: DSPy installed)
**Estimated Time Remaining**: 6-7 hours

---

## ✅ Completed

1. **DSPy Installation**
   - Installed dspy-ai==3.0.3
   - Installed openai==2.3.0 (for Groq compatibility)
   - All dependencies resolved successfully

---

## 🔄 Next Steps

### Phase 1: Create DSPy Modules (1.5 hours)

#### 1.1 Create `backend/dspy_modules.py`

**Purpose**: Define DSPy signatures and configure Groq LLM

**Complete Code**:
```python
"""
DSPy modules for Mini-Town.
Day 2: Uncompiled ScoreImportance and Reflect modules.
"""

import os
import logging
import dspy
from typing import Optional

logger = logging.getLogger(__name__)

# Global DSPy configuration
_configured = False


def configure_dspy(api_key: Optional[str] = None):
    """
    Configure DSPy with Groq LLM.

    Args:
        api_key: Groq API key (defaults to GROQ_API_KEY env var)
    """
    global _configured

    if _configured:
        return

    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")

    # Configure Groq LLM via OpenAI-compatible API
    lm = dspy.LM(
        model="groq/llama-3.2-3b-preview",
        api_key=api_key,
        temperature=0.3,
        max_tokens=512
    )

    dspy.settings.configure(lm=lm)
    _configured = True
    logger.info("DSPy configured with Groq LLM (llama-3.2-3b-preview)")


# ============ Signatures ============

class ScoreImportance(dspy.Signature):
    """Rate how important this observation is for the agent's goals.

    Score 1-10 where:
    - 1-2: Trivial, background noise (e.g., "grass is green")
    - 3-4: Mildly interesting but not actionable
    - 5-6: Relevant to goals, worth remembering
    - 7-8: Directly impacts current plans or goals
    - 9-10: Life-changing, urgent, critical to goals
    """

    observation: str = dspy.InputField(desc="What the agent observed")
    agent_goal: str = dspy.InputField(desc="Agent's current high-level goal")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits")

    reasoning: str = dspy.OutputField(desc="Brief explanation of score")
    score: int = dspy.OutputField(desc="Importance score (1-10)")


class Reflect(dspy.Signature):
    """Synthesize a high-level insight from recent memories.

    Generate an abstract realization or pattern that helps the agent
    understand their experiences and make better decisions.
    """

    recent_memories: str = dspy.InputField(desc="Recent important memories (newline-separated)")
    agent_personality: str = dspy.InputField(desc="Agent's personality traits")
    agent_goal: str = dspy.InputField(desc="Agent's current goal")

    reasoning: str = dspy.OutputField(desc="Thought process")
    insight: str = dspy.OutputField(desc="High-level insight or realization")


# ============ Uncompiled Modules ============

# Simple predictor (uncompiled baseline)
scorer = dspy.ChainOfThought(ScoreImportance)

# Chain-of-thought for reflection (uncompiled baseline)
reflector = dspy.ChainOfThought(Reflect)


# ============ Helper Functions ============

async def score_observation(
    observation: str,
    agent_goal: str,
    agent_personality: str
) -> int:
    """
    Score importance of an observation using DSPy.

    Args:
        observation: What the agent observed
        agent_goal: Agent's goal
        agent_personality: Agent's personality

    Returns:
        Importance score (1-10)

    Raises:
        Exception: If LLM call fails
    """
    result = scorer(
        observation=observation,
        agent_goal=agent_goal,
        agent_personality=agent_personality
    )

    # Parse score (handle various output formats)
    try:
        score = int(result.score)
        # Clamp to 1-10
        return max(1, min(10, score))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid score from LLM: {result.score}, using default")
        return 5  # Default middle score


async def generate_reflection(
    recent_memories: list[str],
    agent_personality: str,
    agent_goal: str
) -> str:
    """
    Generate a reflection insight from recent memories.

    Args:
        recent_memories: List of recent memory strings
        agent_personality: Agent's personality
        agent_goal: Agent's goal

    Returns:
        Insight string

    Raises:
        Exception: If LLM call fails
    """
    # Format memories as newline-separated string
    memories_str = "\n".join([f"- {mem}" for mem in recent_memories])

    result = reflector(
        recent_memories=memories_str,
        agent_personality=agent_personality,
        agent_goal=agent_goal
    )

    return result.insight


# ============ Module Info ============

def get_module_info():
    """Return info about configured modules."""
    return {
        "configured": _configured,
        "modules": {
            "scorer": {
                "type": "ChainOfThought",
                "signature": "ScoreImportance",
                "compiled": False
            },
            "reflector": {
                "type": "ChainOfThought",
                "signature": "Reflect",
                "compiled": False
            }
        }
    }
```

**Testing**:
```bash
cd backend
source ../mini-town/bin/activate
python -c "from dspy_modules import configure_dspy, get_module_info; configure_dspy(); print(get_module_info())"
```

---

#### 1.2 Create `backend/test_dspy_modules.py`

**Purpose**: Unit test DSPy modules before integration

**Complete Code**:
```python
"""
Unit tests for DSPy modules.
Tests ScoreImportance and Reflect without full simulation.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dspy_modules import configure_dspy, score_observation, generate_reflection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_score_importance():
    """Test ScoreImportance module."""
    logger.info("=" * 60)
    logger.info("Testing ScoreImportance")
    logger.info("=" * 60)

    test_cases = [
        {
            "observation": "Alice invited me to a party at 7pm tonight",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (7, 10)
        },
        {
            "observation": "The grass is green today",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (1, 3)
        },
        {
            "observation": "Fire alarm going off in the building",
            "goal": "Build relationships in the neighborhood",
            "personality": "social, optimistic",
            "expected_range": (8, 10)
        }
    ]

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test['observation'][:50]}...")

        try:
            score = await score_observation(
                observation=test['observation'],
                agent_goal=test['goal'],
                agent_personality=test['personality']
            )

            logger.info(f"  Score: {score}")

            min_score, max_score = test['expected_range']
            if min_score <= score <= max_score:
                logger.info(f"  ✅ PASS (expected {min_score}-{max_score})")
            else:
                logger.warning(f"  ⚠️  Outside expected range {min_score}-{max_score}")

        except Exception as e:
            logger.error(f"  ❌ FAILED: {e}")


async def test_reflect():
    """Test Reflect module."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Reflect")
    logger.info("=" * 60)

    recent_memories = [
        "Alice invited me to a party at 7pm",
        "Bob said he's too busy to attend",
        "Carol is always punctual for events",
        "I feel excited about social gatherings"
    ]

    logger.info(f"\nMemories ({len(recent_memories)}):")
    for mem in recent_memories:
        logger.info(f"  - {mem}")

    try:
        insight = await generate_reflection(
            recent_memories=recent_memories,
            agent_personality="social, optimistic",
            agent_goal="Build relationships in the neighborhood"
        )

        logger.info(f"\nInsight: {insight}")
        logger.info("✅ PASS")

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting DSPy Module Tests")
    logger.info("Configuring DSPy with Groq...")

    configure_dspy()

    await test_score_importance()
    await test_reflect()

    logger.info("\n" + "=" * 60)
    logger.info("Tests Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

**Run Test**:
```bash
cd backend
source ../mini-town/bin/activate
python test_dspy_modules.py
```

**Expected Output**:
```
Testing ScoreImportance
Test 1: Alice invited me to a party at 7pm tonight...
  Score: 8
  ✅ PASS (expected 7-10)

Test 2: The grass is green today...
  Score: 2
  ✅ PASS (expected 1-3)
...
```

---

### Phase 2: Add Latency Tracking (1 hour)

#### 2.1 Update `backend/utils.py`

**Add to end of file**:
```python
# ============ Latency Tracking ============

from collections import defaultdict
from datetime import datetime
import asyncio
import time
import numpy as np


class LatencyTracker:
    """Tracks LLM call latencies and computes statistics."""

    def __init__(self):
        """Initialize tracker."""
        self.calls = defaultdict(list)  # signature_name -> list of (timestamp, latency, success)
        self.lock = asyncio.Lock()

    async def record(self, signature_name: str, latency_seconds: float, success: bool = True):
        """Record a call."""
        async with self.lock:
            self.calls[signature_name].append((datetime.now(), latency_seconds, success))

    def get_stats(self, signature_name: str = None) -> dict:
        """
        Get statistics for a signature (or all if None).

        Returns:
            Dict with p50, p95, p99, count, success_rate
        """
        if signature_name:
            signatures = [signature_name]
        else:
            signatures = list(self.calls.keys())

        stats = {}
        for sig in signatures:
            calls = self.calls[sig]
            if not calls:
                stats[sig] = {
                    "count": 0,
                    "success_rate": 0.0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0
                }
                continue

            latencies = [lat for _, lat, _ in calls]
            successes = [suc for _, _, suc in calls]

            stats[sig] = {
                "count": len(calls),
                "success_rate": sum(successes) / len(successes) * 100,
                "p50_ms": int(np.percentile(latencies, 50) * 1000),
                "p95_ms": int(np.percentile(latencies, 95) * 1000),
                "p99_ms": int(np.percentile(latencies, 99) * 1000),
                "mean_ms": int(np.mean(latencies) * 1000)
            }

        return stats

    def reset(self):
        """Clear all tracked calls."""
        self.calls.clear()


# Global latency tracker instance
_latency_tracker = LatencyTracker()


def get_latency_tracker() -> LatencyTracker:
    """Get global latency tracker instance."""
    return _latency_tracker


async def timed_llm_call(func, signature_name: str, timeout: float = 5.0, **kwargs):
    """
    Wrapper for LLM calls with latency tracking and timeout.

    Args:
        func: Async function to call (e.g., score_observation)
        signature_name: Name of signature for tracking
        timeout: Timeout in seconds
        **kwargs: Arguments to pass to func

    Returns:
        Result from func

    Raises:
        asyncio.TimeoutError: If call exceeds timeout
    """
    tracker = get_latency_tracker()
    start = time.time()

    try:
        # Wrap in timeout
        result = await asyncio.wait_for(func(**kwargs), timeout=timeout)

        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=True)

        logger.debug(f"{signature_name} completed in {elapsed*1000:.0f}ms")

        return result

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=False)

        logger.warning(f"{signature_name} TIMEOUT after {elapsed*1000:.0f}ms")
        raise

    except Exception as e:
        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=False)

        logger.error(f"{signature_name} FAILED after {elapsed*1000:.0f}ms: {e}")
        raise
```

---

### Phase 3: Update Agents (1.5 hours)

#### 3.1 Update `backend/agents.py`

**Add imports at top**:
```python
from dspy_modules import score_observation, generate_reflection
from utils import timed_llm_call
```

**Add to Agent.__init__**:
```python
# Reflection tracking
self.reflection_score = 0.0  # Accumulates importance scores
self.reflection_threshold = 3.5  # From config or personality
```

**Add new methods to Agent class**:
```python
async def score_and_store_observation(self, obs: str, memory_store) -> float:
    """
    Score observation importance via LLM and store in memory.

    Args:
        obs: Observation string
        memory_store: MemoryStore instance

    Returns:
        Importance score (0-1 normalized)
    """
    from datetime import datetime
    from utils import generate_embedding

    try:
        # Score via LLM (1-10)
        score_raw = await timed_llm_call(
            score_observation,
            signature_name="ScoreImportance",
            timeout=5.0,
            observation=obs,
            agent_goal=self.goal,
            agent_personality=self.personality
        )

        # Normalize to 0-1
        importance = score_raw / 10.0

    except Exception as e:
        logger.warning(f"Agent {self.id} LLM scoring failed: {e}, using default")
        importance = 0.3  # Fallback
        self.state = "confused"

    # Generate embedding
    embedding = generate_embedding(obs)

    # Store memory
    memory_store.store_memory(
        agent_id=self.id,
        content=obs,
        importance=importance,
        embedding=embedding,
        timestamp=datetime.now()
    )

    # Accumulate for reflection
    self.reflection_score += importance

    return importance


async def maybe_reflect(self, memory_store):
    """
    Check if agent should reflect and generate insight if threshold exceeded.

    Args:
        memory_store: MemoryStore instance

    Returns:
        Insight string if reflected, None otherwise
    """
    if self.reflection_score < self.reflection_threshold:
        return None

    logger.info(f"Agent {self.id} ({self.name}) triggering reflection (score: {self.reflection_score:.2f})")

    # Get recent important memories
    memories = memory_store.get_agent_memories(self.id, limit=10)

    if not memories:
        self.reflection_score = 0.0
        return None

    # Format as strings
    memory_strings = [mem['content'] for mem in memories]

    try:
        # Generate reflection via LLM
        insight = await timed_llm_call(
            generate_reflection,
            signature_name="Reflect",
            timeout=5.0,
            recent_memories=memory_strings,
            agent_personality=self.personality,
            agent_goal=self.goal
        )

        logger.info(f"Agent {self.id} insight: {insight[:100]}...")

        # Reset accumulator
        self.reflection_score = 0.0

        return insight

    except Exception as e:
        logger.error(f"Agent {self.id} reflection failed: {e}")
        self.state = "confused"
        return None
```

---

### Phase 4: Update Main Simulation (1 hour)

#### 4.1 Update `backend/main.py`

**Add imports at top**:
```python
from dspy_modules import configure_dspy
from utils import get_latency_tracker
```

**Update startup_event**:
```python
@app.on_event("startup")
async def startup_event():
    """Initialize database and agents on startup."""
    global memory_store

    logger.info("Starting Mini-Town API...")

    # Configure DSPy with Groq
    logger.info("Configuring DSPy...")
    configure_dspy()

    # Initialize database
    project_root = Path(__file__).parent.parent
    db_path = project_root / config['database']['path']
    memory_store = MemoryStore(str(db_path))

    # Initialize agents
    initialize_agents()

    # Start simulation loop in background
    asyncio.create_task(simulation_loop())
```

**Update simulation_loop** (replace observation storage section):
```python
# OLD CODE (remove):
# if state['observations']:
#     for obs in state['observations']:
#         embedding = generate_embedding(obs)
#         memory_store.store_memory(...)

# NEW CODE:
if state['observations']:
    for obs in state['observations']:
        # Score and store via LLM
        importance = await agent.score_and_store_observation(obs, memory_store)
        logger.debug(f"Agent {agent.id} scored '{obs[:30]}...' = {importance:.2f}")

    # Check for reflection
    insight = await agent.maybe_reflect(memory_store)
    if insight:
        # Store insight as special memory
        embedding = generate_embedding(insight)
        memory_store.store_memory(
            agent_id=agent.id,
            content=f"[REFLECTION] {insight}",
            importance=0.9,  # High importance
            embedding=embedding,
            timestamp=datetime.now()
        )
```

**Add new endpoint for latency stats**:
```python
@app.get("/latency")
async def get_latency_stats():
    """Get LLM latency statistics."""
    tracker = get_latency_tracker()
    stats = tracker.get_stats()

    # Calculate tick viability
    p95_max = max([s.get('p95_ms', 0) for s in stats.values()] + [0])

    decision = "keep_2s"
    if p95_max > 2500:
        decision = "investigate"
    elif p95_max > 1500:
        decision = "adjust_to_3s"

    return {
        "signatures": stats,
        "p95_max_ms": p95_max,
        "tick_decision": decision
    }
```

---

### Phase 5: Baseline Testing (1-2 hours)

#### 5.1 Create `backend/test_latency_baseline.py`

**Complete Code** - see separate file creation step below

---

## 🎯 Critical Decision Points

### Decision 1: Groq Rate Limits
**Issue**: Free tier = 30 req/min
**Test**: Run 1 tick with 3 agents, count requests
**Actions**:
- If < 15 req/tick → safe for 3 agents at 2s ticks
- If > 15 req/tick → reduce to 3s ticks OR 3 agents only

### Decision 2: Tick Interval
**Trigger**: After 20-tick baseline test
**Rules**:
- p95 < 1.5s → Keep 2s ticks ✅
- p95 1.5-2.5s → Adjust to 3s ticks ⚠️
- p95 > 2.5s → Investigate 🚨

**Update config.yml**:
```yaml
simulation:
  tick_interval: 3.0  # If needed
```

---

## 🚨 Error Handling

**All errors must be logged to** `@error_log.md`

### Common Errors

**1. Groq API Key Invalid**
```
Error: 401 Unauthorized from Groq API
Solution: Check .env file has valid GROQ_API_KEY
```

**2. Rate Limit Exceeded**
```
Error: 429 Too Many Requests
Solution: Increase tick_interval to 3.0 or reduce agents to 3
```

**3. Timeout Errors**
```
Error: asyncio.TimeoutError after 5.0s
Solution: Agent enters "confused" state, uses fallback importance=0.3
```

**4. DSPy Module Not Found**
```
Error: ModuleNotFoundError: No module named 'dspy'
Solution: Ensure virtual env activated, run: pip install dspy-ai
```

---

## 📊 Success Criteria Checklist

- [ ] ScoreImportance returns 1-10 scores
- [ ] Reflect generates insight strings
- [ ] At least 1 reflection triggered in 20 ticks
- [ ] Latency tracking captures p50/p95
- [ ] 20-tick test completes without crashes
- [ ] Tick interval decision documented
- [ ] All errors logged to @error_log.md

---

## 📝 Next Session Commands

**Start where you left off**:
```bash
cd /Users/nainy/Documents/Personal/mini-town
source mini-town/bin/activate
cd backend

# Test DSPy modules
python test_dspy_modules.py

# Run baseline test
python test_latency_baseline.py

# Start simulation
python main.py
```

**Check latency stats**:
```bash
curl http://localhost:8000/latency
```

---

## 📦 Files to Create

1. ✅ `backend/dspy_modules.py` (see above)
2. ✅ `backend/test_dspy_modules.py` (see above)
3. ⏳ `backend/test_latency_baseline.py` (create next)
4. ⏳ `backend/benchmark_scenarios.py` (create last)
5. ⏳ `DAY2_REVIEW.md` (document results)

---

**Created**: 2025-10-11
**Last Updated**: 2025-10-11
**Context Used**: 65% (save for later continuation)
