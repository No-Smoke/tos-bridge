"""
Embedding service using Ollama.

Provides async embedding generation for document text.
Default model: mxbai-embed-large (1024 dimensions)
"""
import os
import time
import asyncio
import logging
from typing import List, Optional

import httpx

# Set up logging
logger = logging.getLogger("tos-bridge.embedding")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# Circuit breaker for Ollama
class EmbeddingCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                logger.info("Embedding circuit breaker half-open")
            else:
                raise Exception("Embedding service circuit breaker open - too many failures")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Embedding circuit breaker closed - service recovered")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Embedding circuit breaker opened after {self.failure_count} failures")

            raise e

# Global circuit breaker instance
embedding_circuit_breaker = EmbeddingCircuitBreaker()


async def _get_embedding_raw(
    text: str,
    model: str,
    timeout: float = 30.0
) -> List[float]:
    """Raw embedding function without circuit breaker (internal use)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]


async def get_embedding(
    text: str,
    model: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 3
) -> List[float]:
    """
    Generate embedding using Ollama with circuit breaker and retry logic.

    Args:
        text: Text to embed
        model: Model name (default: mxbai-embed-large)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts

    Returns:
        1024-dimensional embedding vector (mxbai-embed-large)

    Raises:
        Exception: On circuit breaker open or max retries exceeded
    """
    model = model or OLLAMA_EMBED_MODEL

    for attempt in range(max_retries):
        try:
            result = await embedding_circuit_breaker.call(
                _get_embedding_raw, text, model, timeout
            )
            if attempt > 0:
                logger.info(f"Embedding succeeded on retry {attempt}")
            return result

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise e

            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)


async def get_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    timeout: float = 30.0
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Note: Ollama doesn't have native batch support,
    so this processes texts sequentially.
    
    Args:
        texts: List of texts to embed
        model: Model name
        timeout: Per-request timeout
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    for text in texts:
        emb = await get_embedding(text, model=model, timeout=timeout)
        embeddings.append(emb)
    return embeddings


def get_embedding_sync(
    text: str,
    model: Optional[str] = None,
    timeout: float = 30.0
) -> List[float]:
    """
    Synchronous version of get_embedding.
    
    Args:
        text: Text to embed
        model: Model name
        timeout: Request timeout
        
    Returns:
        Embedding vector
    """
    model = model or OLLAMA_EMBED_MODEL
    
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
