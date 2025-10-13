# Day 5: A/B Testing + Retrieval Tuning - Implementation Complete

**Date**: 2025-10-11
**Status**: ✅ **READY TO RUN**
**Phase**: Evaluation & Optimization

---

## Overview

Day 5 implementation creates a comprehensive evaluation framework for testing compiled vs uncompiled agents and optimizing retrieval parameters. All scripts are ready to run.

### What Was Built

1. **`backend/metrics.py`** - Evaluation metrics framework
2. **`backend/ab_test.py`** - A/B testing infrastructure
3. **`backend/tune_retrieval.py`** - Retrieval weight grid search
4. **`backend/test_event_scenario.py`** - Party attendance scenario

---

## Files Created

### 1. metrics.py - Evaluation Metrics

**Purpose**: Core evaluation functions for measuring agent performance.

**Key Functions**:
- `event_coherence_metric(scenario_result)` - Did invited agents attend events?
- `plan_fidelity_metric(planned, executed)` - How well did agents follow plans?
- `memory_hit_rate(queries, results)` - Can agents retrieve correct memories?
- `town_score(scenario_results, weights)` - Combined weighted metric
- `retrieval_precision_at_k()` / `retrieval_recall_at_k()` - Retrieval quality

**Usage**:
```python
from metrics import event_coherence_metric, town_score

# Calculate event coherence
results = {
    'event': {'time': party_time, 'invitees': [1, 2, 3]},
    'attendees': [{'agent_id': 1, 'arrival_time': datetime.now()}]
}
coherence = event_coherence_metric(results)
print(f"Event coherence: {coherence:.2%}")

# Calculate combined town score
metrics = {
    'event_coherence': 0.75,
    'plan_fidelity': 0.60,
    'memory_hit_rate': 0.80
}
score = town_score(metrics, weights=(0.4, 0.3, 0.3))
print(f"Town score: {score:.3f}")
```

---

### 2. ab_test.py - A/B Testing Framework

**Purpose**: Compare compiled vs uncompiled agents in live simulation scenarios.

**Key Functions**:
- `run_scenario(use_compiled, duration_minutes, scenario_name)` - Run simulation scenario
- `compare_scenarios(uncompiled, compiled)` - Generate comparison metrics
- `format_comparison_report(comparison)` - Human-readable report

**Running A/B Tests**:

```bash
cd backend

# Run full A/B test (uncompiled + compiled, 20 min each)
python ab_test.py --duration 20

# Run only compiled (skip baseline)
python ab_test.py --duration 20 --skip-uncompiled

# Quick test (5 minutes)
python ab_test.py --duration 5
```

**Output**:
- `results/uncompiled_baseline_*.json` - Uncompiled scenario results
- `results/compiled_gepa_*.json` - Compiled scenario results
- `results/comparison.json` - Side-by-side comparison
- Console: Formatted comparison report

**Expected Metrics**:
- Observation rate (obs/min)
- Reflection rate (reflections/min)
- Average LLM latency (ms)
- Improvement deltas

---

### 3. tune_retrieval.py - Retrieval Weight Tuning

**Purpose**: Find optimal α (relevance), β (recency), γ (importance) weights via grid search.

**Benchmark Scenarios**:
1. **Emergency** - "What should I do about the fire?" → High α (relevance)
2. **Social Planning** - "Who's at the party?" → High β (recency)
3. **Long-term Relationship** - "What's my history with Bob?" → High γ (importance)
4. **Goal Pursuit** - "How to make research progress?" → Balanced α, γ

**Running Grid Search**:

```bash
cd backend

# Full grid search (0.0 to 1.0 in steps of 0.1)
python tune_retrieval.py

# Fine-grained search (steps of 0.05)
python tune_retrieval.py --granularity 0.05

# Test with top-10 retrieval
python tune_retrieval.py --top-k 10
```

**Output**:
- Console: Best weights for each scenario + recommendations
- `results/retrieval_tuning_*.json` - Tuning results

**Expected Output**:
```
📌 EMERGENCY
   Best weights found:
     α (relevance):  0.70
     β (recency):    0.20
     γ (importance): 0.10
   Performance:
     F1:        0.850
     Precision: 0.900
     Recall:    0.800
```

---

### 4. test_event_scenario.py - Party Attendance Test

**Purpose**: Test whether agents attend events they're invited to (event coherence metric).

**Scenario**:
- **T+0**: Simulation starts, agents spawn randomly
- **T+5 min**: Maria sends party invitations (high-importance memories)
- **T+15 min**: Party starts at center of map
- **T+30 min**: Simulation ends

**Running Party Scenario**:

```bash
cd backend

# Full party test (uncompiled + compiled, 30 min each)
python test_event_scenario.py --duration 30

# Quick test (20 minutes)
python test_event_scenario.py --duration 20

# Run only compiled scenario
python test_event_scenario.py --duration 30 --skip-uncompiled
```

**Output**:
- `results/party_uncompiled.json` - Uncompiled party results
- `results/party_compiled.json` - Compiled party results
- `results/party_comparison.json` - Comparison
- Console: Event coherence comparison

**Success Criteria** (from CLAUDE.md):
- **Target**: ≥60% event coherence (compiled agents)
- **Baseline**: <30% event coherence (uncompiled agents)

**Expected Output**:
```
📊 UNCOMPILED:
  Event coherence: 33%
  Attendance: 1/3 agents

📊 COMPILED (GEPA):
  Event coherence: 67%
  Attendance: 2/3 agents

🎯 IMPROVEMENT:
  Coherence delta: +34%
  Attendee delta: +1 agent

✅ SUCCESS: Compiled agents meet target coherence (≥60%)
```

---

## Running Day 5 Evaluation (Recommended Order)

### Step 1: Retrieval Weight Tuning (15 minutes)

```bash
cd backend
python tune_retrieval.py
```

**Why First**: Optimal retrieval weights improve memory quality for all subsequent tests.

**Action**: Review results and update `config.yml` if needed:
```yaml
retrieval:
  default_alpha: 0.5  # Update based on results
  default_beta: 0.3
  default_gamma: 0.2
```

---

### Step 2: Quick A/B Test (10 minutes total)

```bash
# Quick 5-minute test to verify everything works
python ab_test.py --duration 5
```

**Expected Time**: ~10 minutes (5 min uncompiled + 5 min compiled)

**What to Check**:
- Both scenarios complete without errors
- LLM latency is reasonable (<2s p95)
- Observations and reflections are generated

---

### Step 3: Full A/B Test (40 minutes total)

```bash
# Full 20-minute scenarios
python ab_test.py --duration 20
```

**Expected Time**: ~40 minutes (20 min uncompiled + 20 min compiled)

**What to Measure**:
- Observation rate improvement (compiled vs uncompiled)
- Reflection rate improvement
- Latency differences
- Overall behavioral differences

---

### Step 4: Party Scenario Test (60 minutes total)

```bash
# Full 30-minute party scenarios
python test_event_scenario.py --duration 30
```

**Expected Time**: ~60 minutes (30 min uncompiled + 30 min compiled)

**What to Measure**:
- Event coherence (% of invited agents who attend)
- Arrival times (are agents on time?)
- Behavioral differences (compiled vs uncompiled)

**Critical Metric**: Event coherence ≥60% for compiled agents

---

## Interpreting Results

### Success Indicators ✅

1. **Event Coherence**: Compiled agents ≥60%, uncompiled <30%
2. **Town Score**: Compiled agents show 15-25% improvement
3. **Memory Hit Rate**: ≥70% retrieval accuracy with tuned weights
4. **No Regressions**: Compiled agents don't perform worse on any metric

### Needs Iteration ⚠️

1. **Event Coherence**: Compiled <60% or improvement <10%
   - **Action**: Re-run Day 4 compilation with more seeds
   - **Or**: Adjust importance scoring rubric

2. **Retrieval Poor**: Memory hit rate <50%
   - **Action**: Use scenario-specific retrieval weights
   - **Or**: Increase embedding dimension (384 → 768)

3. **High Latency**: p95 >2.5s
   - **Action**: Adjust tick interval in config.yml
   - **Or**: Switch to faster model

---

## Next Steps (After Day 5)

### If Results Are Good (≥60% coherence, 15%+ improvement)

✅ **Proceed to Day 6**:
- Add PlanDay signature
- Implement daily planning module
- Test plan fidelity metric

### If Results Are Mixed (40-60% coherence, 5-15% improvement)

⚠️ **Optional Iteration**:
1. Analyze which scenarios failed (check logs)
2. Add 10-15 more seeds targeting weak areas
3. Re-run Day 4 compilation with `auto="heavy"`
4. Re-test on Day 5

### If Results Are Poor (<40% coherence, <5% improvement)

🚨 **Investigate**:
1. Check if compiled modules loaded correctly (`load_compiled.py`)
2. Verify LLM is responding (check latency stats)
3. Inspect agent memories (are they storing observations?)
4. Review seed quality (Day 3 validation)

---

## Troubleshooting

### Issue: "Failed to load compiled modules"

**Solution**:
```bash
# Verify compiled scorer exists
ls -lh compiled/compiled_scorer.json

# Test loading manually
cd backend
python load_compiled.py
```

### Issue: "No observations generated"

**Cause**: Agents not perceiving each other (too far apart)

**Solution**: Check `perception_radius` in `config.yml` (should be ~50-100px)

### Issue: "Agents don't attend party"

**Possible Causes**:
1. Invitations not stored as high-importance
2. Agents didn't score invitation highly
3. Movement speed too slow to reach party in time

**Debug**:
```bash
# Check memories for invited agents
cd backend
python -c "
from memory import MemoryStore
store = MemoryStore('data/test_party_compiled.db')
memories = store.get_agent_memories(agent_id=2, limit=20)
for m in memories:
    print(f'{m['importance']:.2f}: {m['content'][:60]}')
"
```

### Issue: "Grid search returns all zeros"

**Cause**: No relevant memories in test scenarios

**Solution**: Check `tune_retrieval.py` - ensure test memories are being inserted correctly

---

## Cost Tracking

**Day 5 Estimated Costs**:
- Retrieval tuning: $0 (local embeddings, no LLM calls)
- A/B test (20 min × 2): ~$0.01 (Groq free tier)
- Party scenario (30 min × 2): ~$0.01 (Groq free tier)

**Total Day 5**: ~$0.02

**Cumulative Budget**: $0.05 / $5.00 ✅ **99% remaining!**

---

## Time Tracking

| Task | Estimated | Implementation | Testing |
|------|-----------|---------------|---------|
| Create metrics.py | 2 hours | ✅ Complete | Pending |
| Create ab_test.py | 1.5 hours | ✅ Complete | Pending |
| Create tune_retrieval.py | 1.5 hours | ✅ Complete | Pending |
| Create test_event_scenario.py | 2 hours | ✅ Complete | Pending |
| Run tests | 2 hours | - | Pending |
| Documentation | 1 hour | ✅ Complete | - |
| **Total** | **10 hours** | **~5 hours** | **~2 hours** |

**Status**: Implementation complete in ~5 hours. Ready for testing phase (~2 hours).

---

## Files Summary

### Created Files ✅
- `backend/metrics.py` (340 lines) - Evaluation metrics
- `backend/ab_test.py` (370 lines) - A/B testing framework
- `backend/tune_retrieval.py` (420 lines) - Retrieval tuning
- `backend/test_event_scenario.py` (450 lines) - Party scenario test
- `DAY5_IMPLEMENTATION_PLAN.md` (this file)

### Output Directories
- `results/` - Test results and comparisons (JSON)
- `data/` - Test databases (temporary)

### No Changes Needed
- `config.yml` - Already has `use_compiled: true`
- `backend/dspy_modules.py` - Already supports compiled modules
- `compiled/compiled_scorer.json` - Ready to use

---

## Success Criteria Checklist

Day 5 is complete when:

- [ ] Retrieval tuning identifies optimal weights per scenario
- [ ] A/B test shows measurable improvement (observation/reflection rates)
- [ ] Party scenario achieves ≥60% event coherence (compiled)
- [ ] No critical errors or crashes during testing
- [ ] Results documented in `DAY5_RESULTS.md`

**Decision Point**: If improvement <10%, consider re-compilation (Day 4 iteration). Otherwise proceed to Day 6.

---

## Quick Reference Commands

```bash
# Retrieval tuning (15 min)
cd backend && python tune_retrieval.py

# Quick A/B test (10 min)
cd backend && python ab_test.py --duration 5

# Full A/B test (40 min)
cd backend && python ab_test.py --duration 20

# Quick party test (20 min)
cd backend && python test_event_scenario.py --duration 20

# Full party test (60 min)
cd backend && python test_event_scenario.py --duration 30

# View results
ls -lh results/
cat results/comparison.json | jq .
```

---

**Status**: ✅ **IMPLEMENTATION COMPLETE**
**Next Action**: Run tests (estimated 2 hours total)
**Date**: 2025-10-11
