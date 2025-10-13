# DSPy Optimizer Service

This microservice exposes Mini-Town's compiled DSPy modules over HTTP. It
is intended for use by the AI Town Convex backend (or other clients) when
they need planner or scorer decisions.

## Quickstart

```bash
cd /Users/nainy/Documents/Personal/mini-town
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the service
uvicorn dspy_optimizer.main:app --reload --port 8001
```

`dspy_optimizer` will automatically look for compiled artifacts inside
`compiled/compiled_planner.json` and `compiled/compiled_scorer.json`. On
startup it also configures DSPy using the provider/model values from
`config.yml` (unless overridden via environment variables).

## Configuration

Environment variables are prefixed with `DSPY_OPT_`. For local testing you
can drop them into `dspy_optimizer/.env`.

| Variable | Purpose |
|----------|---------|
| `DSPY_OPT_COMPILED_DIR` | Override path to compiled artifacts (defaults to `<project>/compiled`). |
| `DSPY_OPT_PROVIDER` / `DSPY_OPT_MODEL` | Override LLM provider/model. |
| `DSPY_OPT_GROQ_API_KEY`, `DSPY_OPT_TOGETHER_API_KEY` | Optional API keys; falls back to global environment variables. |
| `DSPY_OPT_AUTH_TOKEN` | Shared secret required via `X-Optimizer-Token` header. |
| `DSPY_OPT_ALLOW_COMPILE` | Enable future `/compile/*` endpoints (off by default). |

## Sample Requests

```bash
TOKEN=dev-secret

# Health check
curl -s -H "X-Optimizer-Token: $TOKEN" \
     http://localhost:8001/healthz

# Planner request
curl -s -X POST http://localhost:8001/planner/plan_day \
     -H "Content-Type: application/json" \
     -H "X-Optimizer-Token: $TOKEN" \
     -d '{
           "agent_name": "Alice",
           "agent_goal": "Prepare the community garden",
           "agent_personality": "Helpful, collaborative",
           "current_time": "8:45 AM",
           "current_location": "(200, 150)",
           "recent_events": ["09:15 AM - Garden meetup at (260, 180)"],
           "relevant_memories": ["Bob promised new seeds yesterday"]
         }'

# Importance scoring request
curl -s -X POST http://localhost:8001/scorer/score_importance \
     -H "Content-Type: application/json" \
     -H "X-Optimizer-Token: $TOKEN" \
     -d '{
           "agent_name": "Alice",
           "observation": "Carol invited me to a picnic at 12:30 PM",
           "agent_goal": "Build friendships",
           "agent_personality": "Friendly"
         }'
```

> Note: replace `$TOKEN` with the same value configured in
> `DSPY_OPT_AUTH_TOKEN`. For development you can skip the token by leaving
> the environment variable unset.

## Next Steps
- Add endpoints for reflection and future DSPy modules.
- Harden authentication (mTLS, OAuth) before exposing publicly.
- Integrate with AI Town Convex actions and add automated smoke tests.

