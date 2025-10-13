# Mini-Town: Compiled Generative Agents

A simulation of 5 autonomous NPCs that perceive, reflect, plan, and act—where each cognitive skill is a typed DSPy program compiled against measurable metrics.

**Current Status**: Day 1 - Vector Search ✅

## Progress

### Day 0.5 - Hardcoded Validation ✅
- ✅ Project structure created
- ✅ DuckDB schema implemented (agents, memories)
- ✅ Agent class with random walk behavior
- ✅ FastAPI server with WebSocket broadcasting
- ✅ Next.js frontend with Canvas map visualization
- ✅ Perception logic (agents detect nearby agents)

### Day 1 - Vector Search ✅
- ✅ DuckDB schema upgraded with 384-dim embeddings
- ✅ Sentence-transformers integration (all-MiniLM-L6-v2)
- ✅ Vector similarity search using list_cosine_similarity
- ✅ Triad scoring implemented (α=relevance, β=recency, γ=importance)
- ✅ Tested with 100 diverse phrases across 5 categories
- ✅ Retrieval quality validated with different weight configurations
- ⚠️ HNSW index not available (DuckDB 0.9.2 ARM limitation, functional without it)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip and npm

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server (from project root)
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

**Note for Day 1+**: The DuckDB `vss` extension will be needed for vector search. It will be installed and loaded automatically when you add vector functionality.

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# (Optional) Configure backend URL
# Copy .env.local.example to .env.local and edit if needed
# Default is ws://localhost:8000/ws

# Run the development server
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. Start the backend server (port 8000)
2. Start the frontend dev server (port 3000)
3. Open `http://localhost:3000` in your browser
4. Watch 3 agents (Alice, Bob, Carol) perform random walks
5. Click on any agent to see their details and observations
6. When agents are within 50 pixels of each other, they perceive and log observations

## Project Structure

```
mini-town/
├── backend/
│   ├── main.py          # FastAPI server + simulation loop
│   ├── agents.py        # Agent class with behaviors
│   └── memory.py        # DuckDB integration
├── frontend/
│   ├── components/
│   │   └── Map.tsx      # Canvas visualization
│   └── pages/
│       └── index.tsx    # Main page
├── data/               # DuckDB database (auto-generated)
├── logs/              # Log files (auto-generated)
├── config.yml         # Configuration
├── requirements.txt   # Python dependencies
└── CLAUDE.md         # Full project plan
```

## Next Steps

**Day 2: Latency Baseline + Uncompiled DSPy** (6-8 hours)
- [ ] Integrate Groq API (Llama-3.2-3B)
- [ ] Create DSPy signatures (ScoreImportance, Reflect)
- [ ] Wire uncompiled modules into simulation loop
- [ ] **Add latency tracking wrapper** (critical for tick interval decision)
- [ ] Run 20-tick baseline, measure p50/p95 latency
- [ ] **Decision: keep 2s ticks or adjust to 3s?**
- [ ] Create benchmark scenarios for retrieval testing

**Day 3: Seed Collection** (6-8 hours) ⚠️ CRITICAL
- [ ] Collect 30-40 diverse observations across all categories
- [ ] Get 2-3 people to independently rate 10 examples
- [ ] Calculate inter-rater agreement (Cohen's kappa > 0.6)
- [ ] Add edge cases + rationale for each seed
- [ ] Document scoring rubric
- [ ] Measure baseline town_score

**Day 4+: Compilation**
- [ ] Set up Colab notebook with GEPA optimizer
- [ ] Run compilation (4-6 hours)
- [ ] A/B test compiled vs uncompiled

## Architecture

- **Backend**: FastAPI + WebSockets for real-time updates
- **Frontend**: Next.js 14 + TypeScript + Canvas
- **Database**: DuckDB (single-file, serverless)
- **Simulation**: 2-second tick interval with random walk
- **Perception**: 50-pixel radius for agent-to-agent detection

## API Endpoints

- `GET /` - API status
- `GET /agents` - List all agents
- `GET /agents/{id}` - Get agent details with memories
- `WS /ws` - WebSocket for real-time updates

## Configuration

See `config.yml` for:
- Simulation parameters (tick interval, map size, perception radius)
- LLM configuration (provider, model, API key)
- Database settings
- Logging configuration

### Environment Variables

**Backend**: Configure via `.env` file in project root (optional for Day 0.5)
```bash
GROQ_API_KEY=your_api_key_here  # Needed for Day 2+
```

**Frontend**: Configure via `frontend/.env.local` (optional)
```bash
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws  # Default value
```

## Troubleshooting

**Backend won't start**:
- Ensure you're running from the `backend/` directory
- Check that `config.yml` exists in the project root
- Verify Python virtual environment is activated

**Frontend can't connect**:
- Ensure backend is running on port 8000
- Check browser console for WebSocket errors
- Verify `NEXT_PUBLIC_WS_URL` if using custom configuration

**Database issues**:
- The database is auto-created at `data/town.db` on first run
- Agents are loaded from DB on restart (persistent state)
- To reset: delete `data/town.db` and restart backend

## Resources

- Full project plan: See `CLAUDE.md`
- DSPy documentation: https://dspy.ai
- Budget: $5 total
- Timeline: 7-10 days

---

**Built with**: Python, FastAPI, DuckDB, DSPy, Next.js, TypeScript
