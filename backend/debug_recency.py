"""
Debug recency scoring to understand why β=0 for all scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta

# Simulate the recency calculation from memory.py line 265
current_time = datetime.now()

print("Recency calculation for 10 memories spread over 10 days:")
print(f"Current time: {current_time}")
print()

for i in range(10):
    days_ago = 9 - i  # Memory 0 is 9 days old, Memory 9 is today
    timestamp = current_time - timedelta(days=days_ago)

    # From memory.py line 265
    time_diff_hours = (current_time - timestamp).total_seconds() / 3600
    recency = max(0.0, 1.0 - (time_diff_hours / 240))  # 240 hours = 10 days

    print(f"Memory {i}: {days_ago} days ago")
    print(f"  Timestamp: {timestamp}")
    print(f"  Hours ago: {time_diff_hours:.1f}")
    print(f"  Recency score: {recency:.4f}")
    print()
