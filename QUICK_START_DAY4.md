# Quick Start: Day 4 Compilation

**Goal**: Compile ScoreImportance module to improve from 77.5% → 85%+ accuracy
**Time**: 4-6 hours (mostly unattended)
**Cost**: $0 (Groq free tier + Colab free/Pro)

---

## Prerequisites ✅

Before starting, make sure you have:
- [x] Google account (for Colab)
- [x] Groq API key (get from https://console.groq.com/)
- [x] Day 3 seeds (`seeds/scorer_v1.json` - 40 examples)

---

## Step-by-Step Instructions

### 1. Open Google Colab (2 minutes)

1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload `compilation/compile_scorer.ipynb` from your project

### 2. Upload Seeds File (1 minute)

**Option A: Direct Upload**
1. In Colab, click the "Files" panel (folder icon on left)
2. Click "Upload" button
3. Select `seeds/scorer_v1.json` from your computer

**Option B: Use Google Drive** (recommended for persistence)
1. Run Cell 2 in the notebook (mounts Google Drive)
2. Copy `scorer_v1.json` to `/content/drive/MyDrive/mini-town/seeds/`
3. Update file path in Cell 5 if needed

### 3. Set API Key (1 minute)

When you run Cell 3, you'll be prompted:
```
Enter your GROQ_API_KEY:
```

Paste your key from https://console.groq.com/keys

**Security Tip**: Keys are not saved after session ends

### 4. Run All Cells (1 minute setup, then wait)

1. Click "Runtime" → "Run all"
2. Monitor first 2-3 cells for errors
3. When compilation starts (Cell 10), you'll see:
   ```
   STARTING GEPA COMPILATION
   Training set size: 40
   Expected runtime: 4-6 hours
   ```

### 5. Wait for Completion (4-6 hours unattended)

**What happens**:
- Colab runs GEPA optimizer with 40 rollouts
- Progress printed every iteration
- Checkpoints saved to Google Drive every 10 iterations

**You can**:
- Leave the tab open
- Close the browser (if using Colab Pro)
- Check back periodically

**Keep-alive script** (optional, for free tier):
Open browser console (F12), paste:
```javascript
function ClickConnect(){
  console.log("Keeping alive");
  document.querySelector("colab-connect-button")?.click()
}
setInterval(ClickConnect, 60000)
```

### 6. Check Results (5 minutes)

When done, you'll see:
```
✅ COMPILATION COMPLETE!
Time elapsed: 4.52 hours

📊 PERFORMANCE COMPARISON
| Metric | Uncompiled | Compiled | Improvement |
|--------|------------|----------|-------------|
| **±2** | **77.5%**  | **87.5%**| **+10.0%**  |
```

**Success**: Improvement ≥10% → Proceed to Day 5
**Needs work**: Improvement <10% → Review worst seeds, iterate

### 7. Download Compiled Files (2 minutes)

From Google Drive or Colab files panel, download:
1. `compiled_scorer.json` (main compiled module)
2. `compilation_results.json` (metrics)
3. `prompt_scorer.txt` (optimized prompts)

### 8. Copy to Project (1 minute)

```bash
# Mac/Linux
cp ~/Downloads/compiled_scorer.json ./compiled/
cp ~/Downloads/compilation_results.json ./compiled/
cp ~/Downloads/prompt_scorer.txt ./compiled/

# Windows
move %USERPROFILE%\Downloads\compiled_scorer.json .\compiled\
move %USERPROFILE%\Downloads\compilation_results.json .\compiled\
move %USERPROFILE%\Downloads\prompt_scorer.txt .\compiled\
```

### 9. Enable Compiled Module (30 seconds)

Edit `config.yml`:
```yaml
compilation:
  use_compiled: true  # Change from false to true
```

### 10. Test Locally (2 minutes)

```bash
cd backend
python load_compiled.py
```

Expected output:
```
✅ Compiled scorer loaded successfully
Test prediction: 8
Reasoning: Direct invitation to social event...

Compilation time: 4.52 hours
Uncompiled ±2 accuracy: 77.5%
Compiled ±2 accuracy:   87.5%
Improvement: +10.0%
```

---

## Troubleshooting

### Issue: "Cannot connect to runtime"
**Solution**: Wait a moment and try again, or use Colab Pro for priority access

### Issue: "GROQ_API_KEY not found"
**Solution**: Re-run Cell 3 and enter your API key

### Issue: Compilation taking >8 hours
**Diagnosis**: Groq API might be slow
**Solution**: Add delays in notebook, or reduce budget to 30

### Issue: Improvement <10%
**Diagnosis**: Seeds may need more diversity
**Solutions**:
1. Review `DAY3_SEED_COLLECTION.md` for weak categories
2. Add 5-10 more mundane category seeds
3. Try MIPROv2 optimizer (Cell 15 in notebook)
4. Reduce metric strictness

### Issue: Colab disconnected mid-compilation
**Solution**: Resume from checkpoint
1. Check Google Drive for latest checkpoint file
2. Uncomment resume code in notebook (Cell 11)
3. Load checkpoint and continue

---

## Expected Timeline

| Step | Duration |
|------|----------|
| 1-3. Setup Colab | 4 minutes |
| 4. Run notebook | 1 minute |
| 5. Wait for compilation | 4-6 hours ⏰ |
| 6-7. Check & download | 7 minutes |
| 8-10. Local setup & test | 3.5 minutes |
| **Total active time** | **~15 minutes** |
| **Total elapsed time** | **4-6 hours** |

---

## What to Do While Waiting

Compilation runs unattended, so you can:
- ☕ Take a break
- 📖 Read Day 5 plan in CLAUDE.md
- 🧪 Review Day 3 baseline results
- 🎨 Work on frontend (if applicable)
- 💤 Leave overnight if running late

---

## Success Checklist

After completion, verify:
- [ ] Compilation completed without errors
- [ ] Improvement ≥10% (77.5% → 85%+)
- [ ] Files downloaded to `compiled/` directory
- [ ] `config.yml` updated (use_compiled: true)
- [ ] Local test passed (load_compiled.py works)
- [ ] Ready for Day 5 A/B testing

---

## Next Steps (Day 5)

If compilation successful:
1. ✅ Load compiled module in full simulation
2. ✅ Run A/B test: compiled vs uncompiled agents
3. ✅ Measure town_score improvement
4. ✅ Tune retrieval weights (grid search)
5. ✅ Test event scenarios with compiled agents

If needs iteration:
1. 📋 Analyze which categories still struggling
2. 📋 Add 5-10 more seeds for weak areas
3. 📋 Try MIPROv2 optimizer
4. 📋 Re-run compilation with updated seeds

---

## Quick Links

- **Colab**: https://colab.research.google.com/
- **Groq Console**: https://console.groq.com/
- **Notebook**: `compilation/compile_scorer.ipynb`
- **Seeds**: `seeds/scorer_v1.json`
- **Results Template**: `DAY4_COMPILATION_RESULTS.md`
- **Detailed Guide**: `compilation/README.md`

---

## Need Help?

Check these resources:
1. **Compilation FAQ**: `compilation/README.md` (Section: FAQ)
2. **Troubleshooting**: `compilation/README.md` (Section: Troubleshooting)
3. **CLAUDE.md**: Day 4 section (line 464-475)
4. **Error Log**: Add any issues to `@error_log.md`

---

**Total Estimated Time**: 15 minutes active + 4-6 hours waiting
**Difficulty**: Easy (mostly automated)
**Cost**: $0.00

**Status**: ✅ Ready to start!
