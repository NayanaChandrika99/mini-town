"""
Utility script to seed shared daily plans for agents.
Run manually before starting the backend to ensure agents
share overlapping events for loitering/conversation tests.
"""

import textwrap
from datetime import datetime
from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "town.db"

    plans = {
        1: textwrap.dedent(
            """
            08:45 AM - 09:15 AM: Morning meetup at plaza (200, 150)
            09:15 AM - 09:45 AM: Coffee chat with Bob (210, 155)
            11:00 AM - 11:30 AM: Lunch plan with Carol (205, 160)
            """
        ).strip(),
        2: textwrap.dedent(
            """
            08:45 AM - 09:15 AM: Morning meetup at plaza (200, 150)
            09:30 AM - 10:00 AM: Walk the market loop (230, 140)
            11:00 AM - 11:30 AM: Lunch plan with Carol (205, 160)
            """
        ).strip(),
        3: textwrap.dedent(
            """
            08:45 AM - 09:15 AM: Morning meetup at plaza (200, 150)
            09:30 AM - 10:00 AM: Garden check-in (260, 180)
            11:00 AM - 11:30 AM: Lunch plan with Carol (205, 160)
            """
        ).strip(),
    }

    conn = duckdb.connect(database=str(db_path))
    try:
        for agent_id, plan in plans.items():
            conn.execute(
                "UPDATE agents SET current_plan = ? WHERE id = ?",
                [plan, agent_id],
            )
        conn.execute("CHECKPOINT")  # ensure changes are flushed
    finally:
        conn.close()

    print("✅ Seeded plans for agents 1-3 at", datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
