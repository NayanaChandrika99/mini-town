"""
DuckDB integration for Mini-Town.
Handles agents and memories storage (Day 0.5: no embeddings yet).
"""

import duckdb
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages DuckDB connection and operations."""

    def __init__(self, db_path: str = "data/town.db"):
        """Initialize database connection and create schema."""
        self.db_path = db_path

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(db_path)

        # Install and load vss extension for vector search
        try:
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
            # Enable experimental HNSW persistence for file-based databases
            self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
            logger.info("VSS extension loaded successfully with HNSW persistence enabled")
        except Exception as e:
            logger.warning(f"VSS extension setup: {e}")
            # Extension might already be installed/loaded

        # Initialize schema
        self._create_schema()

        logger.info(f"MemoryStore initialized with database at {db_path}")

    def _create_schema(self):
        """Create tables for agents and memories with vector embeddings."""

        # Agents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                x REAL DEFAULT 0,
                y REAL DEFAULT 0,
                goal TEXT,
                personality TEXT,
                current_plan TEXT,
                plan_source TEXT,
                plan_updated_at TIMESTAMP,
                state TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure new columns exist for legacy databases
        self.conn.execute("""
            ALTER TABLE agents ADD COLUMN IF NOT EXISTS plan_source TEXT
        """)
        self.conn.execute("""
            ALTER TABLE agents ADD COLUMN IF NOT EXISTS plan_updated_at TIMESTAMP
        """)

        # Memories table with embeddings (auto-incrementing ID)
        # Note: DuckDB requires explicit sequence for auto-increment
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS memories_id_seq START 1;
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY DEFAULT nextval('memories_id_seq'),
                agent_id INTEGER,
                ts TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
        """)

        # Create indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_agent_ts
            ON memories(agent_id, ts DESC)
        """)

        # Create HNSW index for vector similarity search
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING HNSW (embedding)
            """)
            logger.info("HNSW index created for embeddings")
        except Exception as e:
            logger.warning(f"HNSW index creation: {e}")
            # Index might already exist or VSS extension issue

        logger.info("Database schema created successfully")

    # ============ Agent Operations ============

    def create_agent(
        self,
        agent_id: int,
        name: str,
        x: float = 0,
        y: float = 0,
        goal: str = "",
        personality: str = "",
        state: str = "active"
    ) -> None:
        """Create a new agent in the database."""
        self.conn.execute(
            """
            INSERT INTO agents (id, name, x, y, goal, personality, state, current_plan, plan_source, plan_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [agent_id, name, x, y, goal, personality, state, None, "unknown", None],
        )
        logger.info(f"Created agent {agent_id}: {name} at ({x}, {y})")

    def update_agent_position(self, agent_id: int, x: float, y: float) -> None:
        """Update agent's position."""
        self.conn.execute("""
            UPDATE agents SET x = ?, y = ? WHERE id = ?
        """, [x, y, agent_id])

    def update_agent_state(self, agent_id: int, state: str) -> None:
        """Update agent's state."""
        self.conn.execute("""
            UPDATE agents SET state = ? WHERE id = ?
        """, [state, agent_id])

    def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        result = self.conn.execute("""
            SELECT id, name, x, y, goal, personality, current_plan, plan_source, plan_updated_at, state
            FROM agents WHERE id = ?
        """, [agent_id]).fetchone()

        if result:
            return {
                "id": result[0],
                "name": result[1],
                "x": result[2],
                "y": result[3],
                "goal": result[4],
                "personality": result[5],
                "current_plan": result[6],
                "plan_source": result[7] or "unknown",
                "plan_updated_at": result[8],
                "state": result[9]
            }
        return None

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents."""
        results = self.conn.execute("""
            SELECT id, name, x, y, goal, personality, current_plan, plan_source, plan_updated_at, state
            FROM agents
        """).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "x": row[2],
                "y": row[3],
                "goal": row[4],
                "personality": row[5],
                "current_plan": row[6],
                "plan_source": row[7] or "unknown",
                "plan_updated_at": row[8],
                "state": row[9]
            }
            for row in results
        ]

    def update_agent_plan(
        self,
        agent_id: int,
        plan: Optional[str],
        plan_source: Optional[str] = None,
        plan_updated_at: Optional[datetime] = None,
    ) -> None:
        """Persist plan details for an agent."""
        if plan_updated_at is None:
            plan_updated_at = datetime.now()

        self.conn.execute(
            """
            UPDATE agents
            SET current_plan = ?, plan_source = ?, plan_updated_at = ?
            WHERE id = ?
            """,
            [plan, plan_source or "unknown", plan_updated_at, agent_id],
        )

    # ============ Memory Operations ============

    def store_memory(
        self,
        agent_id: int,
        content: str,
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """Store a memory for an agent with optional embedding."""
        if timestamp is None:
            timestamp = datetime.now()

        result = self.conn.execute("""
            INSERT INTO memories (agent_id, ts, content, importance, embedding)
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
        """, [agent_id, timestamp, content, importance, embedding]).fetchone()

        memory_id = result[0]
        logger.debug(f"Stored memory {memory_id} for agent {agent_id}: {content[:50]}...")
        return memory_id

    def get_agent_memories(
        self,
        agent_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent memories for an agent."""
        results = self.conn.execute("""
            SELECT id, ts, content, importance
            FROM memories
            WHERE agent_id = ?
            ORDER BY ts DESC
            LIMIT ?
        """, [agent_id, limit]).fetchall()

        return [
            {
                "id": row[0],
                "ts": row[1],
                "content": row[2],
                "importance": row[3]
            }
            for row in results
        ]

    def get_latest_reflection(
        self,
        agent_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent reflection memory (prefixed with [REFLECTION])."""
        result = self.conn.execute("""
            SELECT ts, content
            FROM memories
            WHERE agent_id = ?
              AND content LIKE '[REFLECTION] %'
            ORDER BY ts DESC
            LIMIT 1
        """, [agent_id]).fetchone()

        if not result:
            return None

        ts, content = result
        # Strip the prefix to expose the human-readable insight.
        insight = content[len("[REFLECTION] "):] if content.startswith("[REFLECTION] ") else content

        return {
            "ts": ts,
            "content": insight
        }

    def retrieve_memories_by_vector(
        self,
        agent_id: int,
        query_embedding: List[float],
        top_k: int = 10,
        alpha: float = 0.5,  # relevance weight
        beta: float = 0.3,   # recency weight
        gamma: float = 0.2   # importance weight
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using triad scoring: relevance + recency + importance.

        Args:
            agent_id: Agent whose memories to search
            query_embedding: Query vector (384-dim)
            top_k: Number of results to return
            alpha, beta, gamma: Weights for relevance, recency, importance

        Returns:
            List of memories with computed scores
        """
        # Get current timestamp for recency calculation
        current_time = datetime.now()

        results = self.conn.execute("""
            SELECT
                id,
                ts,
                content,
                importance,
                embedding,
                list_cosine_similarity(embedding, ?) AS relevance
            FROM memories
            WHERE agent_id = ? AND embedding IS NOT NULL
            ORDER BY relevance DESC
            LIMIT ?
        """, [query_embedding, agent_id, top_k * 3]).fetchall()  # Get more, then re-rank

        # Compute triad scores
        memories = []
        for row in results:
            mem_id, ts, content, importance, embedding, relevance = row
            if relevance is None:
                relevance = 0.0

            # Recency score (exponential decay, 1.0 = now, 0.0 = very old)
            time_diff_hours = (current_time - ts).total_seconds() / 3600
            recency = max(0.0, 1.0 - (time_diff_hours / 240))  # 240 hours = 10 days (matches tuning test data)

            # Compute final score
            score = alpha * relevance + beta * recency + gamma * importance

            memories.append({
                "id": mem_id,
                "ts": ts,
                "content": content,
                "importance": importance,
                "relevance": relevance,
                "recency": recency,
                "score": score
            })

        # Sort by final score and return top_k
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:top_k]

    # ============ Utility Methods ============

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
