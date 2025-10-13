"""
Quick diagnostic to check embedding diversity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import generate_embedding
import numpy as np

# Test memories from emergency scenario
test_memories = [
    "Fire safety procedures: exit building immediately, use stairs not elevator, go to assembly point",
    "Emergency contacts: fire department 911, building security 555-0100",
    "Fire extinguisher locations: near kitchen, main hallway, by back exit",
    "Yesterday's lunch meeting with colleagues was delicious pizza",
    "Meeting scheduled for tomorrow at 2pm in conference room A",
]

print("Generating embeddings...")
embeddings = [generate_embedding(mem) for mem in test_memories]

print("\nCosine similarities:")
print("(fire_safety vs fire_contacts):", np.dot(embeddings[0], embeddings[1]))
print("(fire_safety vs lunch):", np.dot(embeddings[0], embeddings[3]))
print("(fire_contacts vs lunch):", np.dot(embeddings[1], embeddings[3]))
print("(fire_extinguisher vs fire_safety):", np.dot(embeddings[2], embeddings[0]))
print("(lunch vs meeting):", np.dot(embeddings[3], embeddings[4]))

# Query embedding
query = "What should I do right now about the fire?"
query_emb = generate_embedding(query)

print("\nQuery similarities:")
for i, mem in enumerate(test_memories):
    similarity = np.dot(query_emb, embeddings[i])
    print(f"  Query <-> Memory {i}: {similarity:.4f} - {mem[:60]}")
