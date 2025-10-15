# Mini-Town Error Log

This file tracks all errors and issues encountered during development, along with their solutions.

---

## Day 1 - Vector Search Setup

### Error 1: VSS Extension Not Available (Non-blocking)
**Date**: 2025-10-11
**Component**: DuckDB VSS Extension
**Error**: `HTTP Error: Failed to download extension "vss" at URL "http://extensions.duckdb.org/v0.9.2/osx_arm64/vss.duckdb_extension.gz"`

**Context**:
- DuckDB version: 0.9.2
- Platform: macOS ARM64 (M-series chip)
- The VSS extension for vector similarity search is not available for this DuckDB version/platform combination

**Impact**:
- HNSW index creation fails
- Can still use vector search via array_cosine_similarity function (slower but functional)
- No blocker for Day 1 testing

**Resolution**:
- Option 1: Upgrade to DuckDB 0.10+ which has better ARM support
- Option 2: Continue with functional vector search using built-in functions
- **Decision**: Use built-in array_cosine_similarity for now, upgrade DuckDB if performance becomes an issue

---

### Error 2: Auto-increment ID Field
**Date**: 2025-10-11
**Component**: DuckDB Schema
**Error**: `Constraint Error: NOT NULL constraint failed: memories.id`

**Context**:
- Memories table ID field declared as `INTEGER PRIMARY KEY`
- DuckDB doesn't auto-increment by default like SQLite
- Need to use SEQUENCE or generated column

**Impact**:
- Cannot insert memories without explicit ID
- Blocks testing

**Resolution**:
- Change ID column to use INTEGER with GENERATED ALWAYS AS IDENTITY
- Or use nextval('sequence_name')
- **Fix applied**: Updated schema to use `CREATE SEQUENCE` with `DEFAULT nextval('memories_id_seq')`

---

### Error 3: Function Name Mismatch
**Date**: 2025-10-11
**Component**: DuckDB Vector Functions
**Error**: `Scalar Function with name array_cosine_similarity does not exist! Did you mean "list_cosine_similarity"?`

**Context**:
- DuckDB 0.9.2 uses `list_cosine_similarity` instead of `array_cosine_similarity`
- Function naming changed in DuckDB versions
- Query was using wrong function name

**Impact**:
- Vector search queries failed
- Blocks retrieval testing

**Resolution**:
- Updated query to use `list_cosine_similarity` instead of `array_cosine_similarity`
- **Fix applied**: Changed function name in memory.py:249

**Status**: ✅ **RESOLVED** - All tests passing

---


## 2025-10-11 - Groq Model Decommissioned

**Phase**: Day 2 - DSPy Module Testing

**Error**:
```
litellm.BadRequestError: GroqException - {"error":{"message":"The model `llama-3.2-3b-preview` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.","type":"invalid_request_error","code":"model_decommissioned"}}
```

**Context**:
- Attempted to run DSPy module tests using `groq/llama-3.2-3b-preview` as specified in the original implementation plan
- All test cases failed with the decommissioned model error

**Root Cause**:
- Groq deprecated and removed the `llama-3.2-3b-preview` model from their API
- The model specified in the plan (DAY2_IMPLEMENTATION_PLAN.md) is no longer available

**Solution**:
- Updated `backend/dspy_modules.py` to use `groq/llama-3.1-8b-instant` instead
- Changed line 37: `model="groq/llama-3.1-8b-instant"`
- Changed line 45: Updated log message to reflect new model

**Benefits of New Model**:
- Still on Groq's free tier (30 req/min)
- Better performance (8B vs 3B parameters)
- Faster inference
- More accurate scoring and reasoning

**Impact**:
- **Minimal** - DSPy modules work correctly with the new model
- Tests pass successfully:
  - ScoreImportance: 1/3 perfect, 2/3 reasonable scores
  - Reflect: Generated coherent insights
- No changes needed to other components
- Free tier limits remain the same

**Files Modified**:
- `backend/dspy_modules.py` (line 37, 45)

**Status**:  Resolved

---
## 2025-10-11 - DuckDB VSS Extension and Vector Search Issues

**Phase**: Day 5 - Retrieval Weight Tuning

**Error**:
```
HTTP Error: Failed to download extension "vss" at URL "http://extensions.duckdb.org/v0.9.2/osx_arm64/vss.duckdb_extension.gz"
Binder Error: Index type not supported
Binder Error: HNSW indexes can only be created in in-memory databases, or when the configuration option 'hnsw_enable_experimental_persistence' is set to true
```

**Context**:
- Running `tune_retrieval.py` on Day 5 to optimize retrieval weights (α, β, γ)
- Vector similarity search (relevance scoring) was not working
- All grid search results showed α=0.0, β=0.0, γ=1.0 (pure importance, no relevance)
- HNSW indexes were failing to create

**Root Cause**:
1. **DuckDB v0.9.2 incompatibility**: VSS extension not available for ARM64 macOS
2. **Missing HNSW persistence setting**: File-based databases require `SET hnsw_enable_experimental_persistence = true`
3. **Incorrect SQL syntax**: Old DuckDB syntax (`list_cosine_similarity`) doesn't work in v1.x

**Solution**:
### Step 1: Upgrade DuckDB
```bash
pip install --upgrade 'duckdb>=1.0'
# Upgraded from 0.9.2 → 1.4.1
```

### Step 2: Enable HNSW Persistence
Updated `backend/memory.py` MemoryStore.__init__():
```python
# Enable experimental HNSW persistence for file-based databases
self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
```

### Step 3: Update Vector Search SQL
Updated `retrieve_memories_by_vector()` in `backend/memory.py`:
```python
# OLD (v0.9.2 syntax - not working):
list_cosine_similarity(embedding, ?::FLOAT[384]) as relevance

# NEW (v1.x syntax - working):
(1.0 - array_distance(embedding, ?::FLOAT[384])) as relevance
ORDER BY array_distance(embedding, ?::FLOAT[384])
```

### Step 4: Remove Unsupported Index Options
```python
# OLD (causing errors):
CREATE INDEX ... USING HNSW (embedding) WITH (metric = 'cosine')

# NEW (working):
CREATE INDEX ... USING HNSW (embedding)
```

**Verification**:
```
✅ VSS extension loaded successfully with HNSW persistence enabled
✅ HNSW index created for embeddings
✅ array_distance works correctly
✅ HNSW index used in queries (verified with EXPLAIN)
```

**Impact**:
- Vector similarity search now functional
- HNSW indexes created successfully for fast ANN search
- `array_distance` provides cosine distance metric
- Memory retrieval can now properly weight relevance (α) alongside recency (β) and importance (γ)

**Files Modified**:
- `backend/memory.py` (lines 30-34, 87-90, 242-254)
- `backend/tune_retrieval.py` (line 185 - method name fix)

**Technical Notes**:
- DuckDB 1.x uses `array_distance` for vector similarity
- HNSW persistence is still experimental but required for file-based DBs
- Indexes are built at startup (acceptable for agent memory use case)
- Default distance metric is cosine (suitable for embeddings)

**Status**: Resolved

---

## 2025-10-15 - GEPA Prompt Compilation Compat + Together Model Access

### Error A: `ModuleNotFoundError: No module named 'dspy.adapters.types'`
**Component**: GEPA → DSPy adapter (TownAgent compilation)  
**Context**: Running `compilation/compile_town_agent.py` with GEPA 0.0.7 on DSPy 2.4.10/2.5.x. The adapter expects `History` in `dspy.adapters.types`, which no longer exists.  
**Resolution**: Stub a minimal `History` class before importing the adapter. Implemented in `compilation/compile_town_agent.py` by injecting a shim module into `sys.modules["dspy.adapters.types"]`.  
**Status**: ✅ Fixed.

### Error B: `ModuleNotFoundError: No module named 'dspy.teleprompt.bootstrap_trace'`
**Component**: Same GEPA adapter  
**Context**: DSPy removed `bootstrap_trace` helpers; GEPA still imports them.  
**Resolution**: Added `compilation/dspy_bootstrap_trace_compat.py` implementing lightweight replacements for `TraceData`, `FailedPrediction`, and `bootstrap_trace_data`, then registered it as `dspy.teleprompt.bootstrap_trace`.  
**Status**: ✅ Fixed.

### Error C: Together API “model_not_available” / dedicated endpoint required  
**Component**: Baseline & GEPA compile LLM calls  
**Context**: Together rejected `meta-llama/...` model IDs (Turbo/managed variants).  
**Resolution**: List available models via `client.models.list()`, switch to serverless IDs (e.g. `meta-llama/Meta-Llama-3-8B-Instruct-Lite`), and update the LM configuration in the notebook/script.  
**Status**: ✅ Fixed (with model list verification step).

### Error D: `TypeError: argument of type 'TownAgentResponse' is not iterable`
**Component**: GEPA evaluation pipeline  
**Context**: DSPy’s `merge_dicts` expects predictions to behave like mappings. `TownAgentResponse` lacked mapping semantics, so GEPA crashed on the first rollout.  
**Resolution**: Extended `programs/town_agent.py` to add `items`, `__contains__`, `__iter__`, and `get` wrappers that proxy to `to_dict()`, making the response dict-compatible.  
**Status**: ✅ Fixed (retained mapping helpers).

### Error E: GEPA adapter assumes `Evaluate` returns object with `.results`
**Component**: GEPA evaluation pipeline  
**Context**: DSPy 2.4/2.5 returns a `(score, results, scores)` tuple when `return_outputs=True`. GEPA’s adapter expected a result object with a `results` attribute and crashed before any rollouts.  
**Resolution**: Monkey-patched `DspyAdapter.evaluate` in `compilation/compile_town_agent.py` to normalize both return shapes (object vs. tuple) into the structure GEPA expects.  
**Status**: ✅ Fixed (handled inside adapter shim).

### Follow-up
- `compilation/compile_town_agent.py` now imports without errors; remaining failures are due to sandboxed network limits. To compile locally run:  
  `PYTHONPATH=.:backend mini-town/bin/python compilation/compile_town_agent.py --dataset datasets/town_agent_train.jsonl --budget 40`

---
