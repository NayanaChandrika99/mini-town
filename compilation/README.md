# Compilation Directory

This directory contains Jupyter notebooks for compiling DSPy modules using GEPA/MIPROv2 optimizers.

## Files

### `compile_scorer.ipynb`
Compiles the ScoreImportance module to improve observation importance scoring.

**Status**: ✅ Ready to run in Google Colab

**Requirements**:
- Google Colab account (free or Pro recommended)
- Groq API key (get from https://console.groq.com/)
- Seeds file: `../seeds/scorer_v1.json` (40 training examples)

**Expected Runtime**: 4-6 hours (with GEPA optimizer)

**Expected Improvement**: 77.5% → 85%+ accuracy (±2 tolerance)

---

## How to Run Compilation

### Quick Start

1. **Open Colab**:
   - Go to https://colab.research.google.com/
   - Upload `compile_scorer.ipynb`

2. **Upload Seeds**:
   - Click "Files" panel in Colab
   - Upload `../seeds/scorer_v1.json`

3. **Set API Key**:
   - Run the setup cells
   - Enter your GROQ_API_KEY when prompted

4. **Run All**:
   - Click "Runtime" → "Run all"
   - Wait 4-6 hours for completion

5. **Download Results**:
   - Download `compiled_scorer.json` from Google Drive
   - Copy to `../compiled/` directory

### Alternative: Use Google Drive

If you prefer to keep files in Google Drive:

1. Create folder structure:
   ```
   /MyDrive/mini-town/
   ├── seeds/
   │   └── scorer_v1.json
   └── compiled/
   ```

2. Upload `scorer_v1.json` to Google Drive

3. In Colab notebook, mount Drive (already included in notebook)

4. Results will auto-save to `/MyDrive/mini-town/compiled/`

---

## Optimizer Options

### GEPA (Primary Choice)
**Advantages**:
- Faster convergence (40 rollouts vs 50-100+)
- 30-40% less compilation time than MIPROv2
- More efficient for budget constraints

**Configuration**:
```python
optimizer = GEPA(
    metric=importance_metric,
    budget=40,
    verbose=True
)
```

### MIPROv2 (Fallback)
**Use If**: GEPA has issues or doesn't improve scores

**Advantages**:
- Well-documented, proven optimizer
- Jointly optimizes instructions + few-shot examples
- More community support

**Configuration**:
```python
optimizer = MIPROv2(
    metric=importance_metric,
    auto="medium",
    num_trials=10
)
```

**Runtime**: 6-8 hours (slower than GEPA)

---

## Troubleshooting

### Issue: Colab Disconnects
**Solution**: Use Colab Pro for longer runtime limits, or enable checkpointing:
```python
if iteration % 10 == 0:
    compiled.save(f'/content/drive/MyDrive/checkpoint_{iteration}.json')
```

### Issue: Out of Memory
**Solution**: Reduce budget from 40 to 30:
```python
optimizer = GEPA(metric=importance_metric, budget=30)
```

### Issue: Compilation Doesn't Improve Scores
**Diagnosis**: Check seed quality, metric definition
**Solutions**:
1. Review worst-performing seeds from Day 3
2. Add more diverse examples (especially mundane category)
3. Try MIPROv2 optimizer
4. Adjust metric tolerances

### Issue: API Rate Limits
**Solution**: Groq free tier allows 30 req/min. Add delays if needed:
```python
import time
time.sleep(2)  # Between predictions
```

---

## Expected Output Files

After successful compilation:

### `compiled_scorer.json`
- Compiled DSPy module with optimized prompts
- Size: ~50-500 KB (varies by few-shot examples)
- Contains: Instructions, demonstrations, signature

### `compilation_results.json`
- Performance metrics summary
- Uncompiled vs compiled comparison
- Category-level breakdowns

### `prompt_scorer.txt`
- Human-readable prompts for inspection
- Shows optimized instruction text
- Few-shot examples selected by optimizer

---

## Next Notebooks (Future)

### `compile_reflector.ipynb` (Day 6-7)
Compile the Reflect module for better insight generation.

### `compile_planner.ipynb` (Day 7-8)
Compile the PlanDay module for coherent daily planning.

---

## Colab Tips

### Keep Session Alive
Colab free tier disconnects after 90 minutes of inactivity. Options:
1. Use Colab Pro (12-hour runtime)
2. Open browser console, run keep-alive script:
   ```javascript
   function ClickConnect(){
     console.log("Clicking");
     document.querySelector("colab-connect-button").click()
   }
   setInterval(ClickConnect, 60000)
   ```

### Monitor Progress
Compilation prints progress every iteration:
```
Iteration 1/40: score=0.45
Iteration 2/40: score=0.52
...
```

### Save Checkpoints
Google Drive auto-saves every 10 iterations (already in notebook).

### Resume from Checkpoint
If disconnected, load latest checkpoint:
```python
compiled.load('/content/drive/MyDrive/mini-town/checkpoint_30.json')
optimizer.resume(compiled, from_iteration=30)
```

---

## Cost Analysis

**Compilation Costs** (worst case):

| Component | Usage | Cost |
|-----------|-------|------|
| Colab GPU | 6 hours | $0 (free tier) or $0 (Pro included) |
| Groq API | ~1500 requests | $0 (free tier) |
| Google Drive | <1 MB storage | $0 |
| **Total** | | **$0.00** |

**Budget Remaining**: $5.00 / $5.00 (100%)

---

## FAQ

**Q: Can I run this locally instead of Colab?**
A: Yes, but compilation is slow without GPU. Colab provides free T4 GPU access.

**Q: How long does compilation take?**
A: GEPA: 4-6 hours, MIPROv2: 6-8 hours (depends on budget and LLM speed)

**Q: Can I reduce compilation time?**
A: Lower budget to 30 or 20, but may reduce improvement quality.

**Q: What if improvement is <10%?**
A: Iterate on seeds (add more examples), try different optimizer, or adjust metric.

**Q: Do I need Colab Pro?**
A: Not required, but recommended for longer runtime limits and fewer disconnects.

---

**Status**: ✅ Ready for Day 4 compilation
**Last Updated**: 2025-10-11
