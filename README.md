# Mini-Town with DSPy & GEPA

Mini-Town is a FastAPI + DSPy simulation of autonomous agents that perceive, reflect, plan, and act. Each cognitive step is a typed DSPy program, and we use GEPA (Generalized Evolutionary Prompt Acceleration) to automatically refine the TownAgent prompt against the data we log during runs.

---

## Key Directories

| Path | Purpose |
| --- | --- |
| `backend/` | FastAPI server, simulation loop, telemetry, vector search helpers. |
| `programs/` | DSPy program definitions (e.g., `town_agent.py`). |
| `compilation/` | GEPA tooling (`compile_town_agent.py`, compat shims, notebooks). |
| `datasets/` | Logged plan corpus + train/dev/test splits. |
| `evaluation/` | CLI utilities to compare baseline vs compiled prompts. |
| `ai-town/` | Next.js admin UI for agent/system monitoring. |
| `compiled/` | Latest compiled prompts and result summaries. |

---

## Prerequisites

- Python 3.11 (conda recommended)
- Node.js 18+
- Together AI (or other LLM) API key for DSPy (`TOGETHER_API_KEY`)

---

## Quick Start

### 1. Backend
```bash
# create and activate environment
conda create -n mini-town-env python=3.11
conda activate mini-town-env

# install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# run the FastAPI server (from project root)
PYTHONPATH=backend:. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
The backend serves REST/WebSocket on `http://localhost:8000`.

### 2. Frontend (AI Town)
```bash
cd ai-town
npm install
npm run dev
```
Open `http://localhost:3000` to inspect the agents, system panel, and logs. Set `NEXT_PUBLIC_WS_URL` in `ai-town/.env.local` if the backend is on a different host/port.

---

## Evaluating TownAgent

After the backend has generated data:
```bash
python -m evaluation.town_agent_eval \
  --dataset datasets/town_agent_dev.jsonl \
  --output results/town_agent_baseline_dev.json
```
The JSON output lists mean/median/min/max scores and validation feedback per example.

---

## Running GEPA (Optional)

GEPA tries to improve `TownAgentProgram` using the corpus in `datasets/`:
```bash
export TOGETHER_API_KEY=sk-...  # or other DSPy provider key
PYTHONPATH=.:backend python compilation/compile_town_agent.py \
  --dataset datasets/town_agent_train.jsonl \
  --budget 80
```
Artifacts are written to `compiled/` after the run. Compare baseline vs compiled with:
```bash
python -m evaluation.town_agent_eval \
  --dataset datasets/town_agent_dev.jsonl \
  --output results/town_agent_compiled_dev.json
python -m evaluation.town_agent_eval \
  --dataset datasets/town_agent_test.jsonl \
  --output results/town_agent_compiled_test.json
```
If the compiled mean score is higher than the baseline (~0.150), swap the prompt in deployment.

---

## Useful Docs

- `DSPY_AI_TOWN_EXECUTION_PLAN.md` – original roadmap and milestones.
- `DAY*_*.md` – day-by-day progress notes.
- `error_log.md` – compatibility fixes and known issues.

---

## Notes & Tips

- Keep virtual environments and build artifacts out of git (`mini-town-env/`, `node_modules/`, etc.).
- The dataset splits (`town_agent_train/dev/test.jsonl`) are small; collect more log data for stronger GEPA results.
- The compatibility shim in `compilation/dspy_compat.py` installs DSPy helpers at runtime—no need to patch site-packages.
- The frontend UI is decoupled (Next.js). Run it separately for diagnostics.

Happy hacking!
