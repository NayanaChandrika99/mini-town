"""
Utility functions for Mini-Town.
Day 1: Embedding generation and helper functions.
Day 2: Latency tracking for LLM calls.
"""

import logging
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from datetime import datetime
import asyncio
import time

logger = logging.getLogger(__name__)

# Global embedding model (initialized on first use)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or initialize the embedding model (singleton pattern).

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Loaded SentenceTransformer model
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded (dimension: {_embedding_model.get_sentence_embedding_dimension()})")

    return _embedding_model


def generate_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    Generate embedding vector for a text string.

    Args:
        text: Input text to embed
        model_name: Name of the sentence-transformers model

    Returns:
        384-dimensional embedding vector as list
    """
    model = get_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_embeddings_batch(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of input texts
        model_name: Name of the sentence-transformers model
        batch_size: Batch size for encoding

    Returns:
        List of 384-dimensional embedding vectors
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1, vec2: Embedding vectors

    Returns:
        Cosine similarity score (0-1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def format_memory_for_display(memory: dict) -> str:
    """
    Format a memory object for human-readable display.

    Args:
        memory: Memory dict with id, ts, content, importance, etc.

    Returns:
        Formatted string
    """
    ts_str = memory['ts'].strftime('%Y-%m-%d %H:%M:%S')
    importance = memory.get('importance', 0.5)
    score = memory.get('score', None)

    output = f"[{ts_str}] (importance: {importance:.2f}"
    if score is not None:
        output += f", score: {score:.3f}"
    output += f") {memory['content']}"

    return output


# ============ Latency Tracking ============


class LatencyTracker:
    """Tracks LLM call latencies and computes statistics."""

    def __init__(self):
        """Initialize tracker."""
        self.calls = defaultdict(list)  # signature_name -> list of (timestamp, latency, success)
        self.lock = asyncio.Lock()

    async def record(self, signature_name: str, latency_seconds: float, success: bool = True):
        """Record a call."""
        async with self.lock:
            self.calls[signature_name].append((datetime.now(), latency_seconds, success))

    def get_stats(self, signature_name: str = None) -> dict:
        """
        Get statistics for a signature (or all if None).

        Returns:
            Dict with p50, p95, p99, count, success_rate
        """
        if signature_name:
            signatures = [signature_name]
        else:
            signatures = list(self.calls.keys())

        stats = {}
        for sig in signatures:
            calls = self.calls[sig]
            if not calls:
                stats[sig] = {
                    "count": 0,
                    "success_rate": 0.0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0
                }
                continue

            latencies = [lat for _, lat, _ in calls]
            successes = [suc for _, _, suc in calls]

            stats[sig] = {
                "count": len(calls),
                "success_rate": sum(successes) / len(successes) * 100,
                "p50_ms": int(np.percentile(latencies, 50) * 1000),
                "p95_ms": int(np.percentile(latencies, 95) * 1000),
                "p99_ms": int(np.percentile(latencies, 99) * 1000),
                "mean_ms": int(np.mean(latencies) * 1000)
            }

        return stats

    def reset(self):
        """Clear all tracked calls."""
        self.calls.clear()


# Global latency tracker instance
_latency_tracker = LatencyTracker()


def get_latency_tracker() -> LatencyTracker:
    """Get global latency tracker instance."""
    return _latency_tracker


async def timed_llm_call(func, signature_name: str, timeout: float = 5.0, **kwargs):
    """
    Wrapper for LLM calls with latency tracking and timeout.

    Args:
        func: Async function to call (e.g., score_observation)
        signature_name: Name of signature for tracking
        timeout: Timeout in seconds
        **kwargs: Arguments to pass to func

    Returns:
        Result from func

    Raises:
        asyncio.TimeoutError: If call exceeds timeout
    """
    tracker = get_latency_tracker()
    start = time.time()

    try:
        # Wrap in timeout
        result = await asyncio.wait_for(func(**kwargs), timeout=timeout)

        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=True)

        logger.debug(f"{signature_name} completed in {elapsed*1000:.0f}ms")

        return result

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=False)

        logger.warning(f"{signature_name} TIMEOUT after {elapsed*1000:.0f}ms")
        raise

    except Exception as e:
        elapsed = time.time() - start
        await tracker.record(signature_name, elapsed, success=False)

        logger.error(f"{signature_name} FAILED after {elapsed*1000:.0f}ms: {e}")
        raise
