# Compiled Modules Directory

This directory stores compiled DSPy programs after GEPA/MIPROv2 optimization.

## Files (After Day 4 Compilation)

### `compiled_scorer.json`
Optimized ScoreImportance module.

**How to Use**:
```python
from backend.dspy_modules import configure_dspy, load_compiled_modules, use_compiled

# Configure DSPy first
configure_dspy()

# Load compiled modules
load_compiled_modules()

# Enable compiled mode
use_compiled(True)

# Now score_observation() uses compiled module
```

**Performance** (after compilation):
- Baseline: 77.5% ±2 accuracy
- Target: 85%+ ±2 accuracy
- Expected improvement: +10-15%

### `compilation_results.json`
Performance metrics and comparison.

**Format**:
```json
{
  "compilation_time_hours": 4.5,
  "uncompiled": {
    "accuracy_within_2": 77.5,
    "mean_error": 1.45
  },
  "compiled": {
    "accuracy_within_2": 87.5,
    "mean_error": 1.1
  },
  "improvement": {
    "accuracy_delta": 10.0,
    "mae_delta": 0.35
  }
}
```

### `prompt_scorer.txt`
Human-readable optimized prompts.

**Contains**:
- Instruction text (evolved by GEPA)
- Few-shot demonstrations
- Chain-of-thought examples
- Scoring heuristics

---

## Loading Compiled Modules

### Method 1: Direct Loading
```python
from backend.load_compiled import load_compiled_scorer

scorer = load_compiled_scorer('compiled/compiled_scorer.json')
result = scorer(
    observation="Alice invited me to a party",
    agent_goal="Build relationships",
    agent_personality="social, optimistic"
)
print(f"Score: {result.score}")
```

### Method 2: Auto-Loading via Config
Edit `config.yml`:
```yaml
compilation:
  use_compiled: true  # Enable compiled modules
```

Then in code:
```python
from backend.dspy_modules import configure_dspy, score_observation

configure_dspy()  # Auto-loads if use_compiled=true

# Automatically uses compiled scorer
score = await score_observation(obs, goal, personality)
```

---

## A/B Testing (Day 5)

Compare compiled vs uncompiled performance:

```python
from backend.dspy_modules import use_compiled, score_observation

# Test uncompiled baseline
use_compiled(False)
uncompiled_score = await score_observation(obs, goal, personality)

# Test compiled version
use_compiled(True)
compiled_score = await score_observation(obs, goal, personality)

print(f"Uncompiled: {uncompiled_score}, Compiled: {compiled_score}")
```

---

## File Structure

```
compiled/
├── README.md                    # This file
├── compiled_scorer.json         # Compiled ScoreImportance (Day 4)
├── compilation_results.json     # Metrics summary (Day 4)
├── prompt_scorer.txt            # Human-readable prompts (Day 4)
├── compiled_reflector.json      # Compiled Reflect (future)
└── compiled_planner.json        # Compiled PlanDay (future)
```

---

## Inspection & Debugging

### View Module Info
```python
from backend.dspy_modules import get_module_info

info = get_module_info()
print(info)
```

Output:
```python
{
  'configured': True,
  'use_compiled': True,
  'modules': {
    'scorer': {
      'type': 'ChainOfThought',
      'signature': 'ScoreImportance',
      'compiled': True,
      'available_compiled': True
    }
  }
}
```

### Get Compilation Info
```python
from backend.load_compiled import get_compilation_info

info = get_compilation_info()
print(f"Improvement: +{info['compilation_results']['improvement']['accuracy_delta']:.1f}%")
```

---

## Status

**Current**: ⏳ Waiting for Day 4 compilation
**Next**: Run `compile_scorer.ipynb` in Google Colab

After compilation completes:
1. Download files from Colab/Google Drive
2. Copy to this directory
3. Update `config.yml` to enable compiled modules
4. Test with `python backend/load_compiled.py`
5. Proceed to Day 5 A/B testing

---

**Last Updated**: 2025-10-11
