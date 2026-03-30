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
    def __init__(self, failure_threshold: int = 8, reset_timeout: int = 120):
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


def _truncate_for_embedding(text: str, max_chars: int = 1800) -> str:
    """
    Truncate text to fit within mxbai-embed-large's 512-token context.
    
    BERT tokenizers average ~4 chars/token. 1800 chars ≈ 450 tokens,
    leaving headroom for special tokens ([CLS], [SEP]).
    Truncates at word boundary to avoid splitting mid-word.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Cut at last space to preserve word boundaries
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    logger.debug(f"Truncated embedding input from {len(text)} to {len(truncated)} chars")
    return truncated


async def _get_embedding_raw(
    text: str,
    model: str,
    timeout: float = 60.0
) -> List[float]:
    """Raw embedding function without circuit breaker (internal use)."""
    text = _truncate_for_embedding(text)
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
    timeout: float = 60.0,
    max_retries: int = 5
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

    # Check circuit breaker state before entering retry loop.
    # This prevents retries from cascading into breaker trips —
    # the breaker only counts one failure per top-level call,
    # not one per retry attempt.
    if embedding_circuit_breaker.state == "open":
        if embedding_circuit_breaker.last_failure_time and \
           time.time() - embedding_circuit_breaker.last_failure_time > embedding_circuit_breaker.reset_timeout:
            embedding_circuit_breaker.state = "half-open"
            logger.info("Embedding circuit breaker half-open (pre-retry check)")
        else:
            raise Exception("Embedding service circuit breaker open - too many failures")

    last_error = None
    for attempt in range(max_retries):
        try:
            result = await _get_embedding_raw(text, model, timeout)
            # Success — reset breaker if it was half-open
            if embedding_circuit_breaker.state == "half-open":
                embedding_circuit_breaker.state = "closed"
                embedding_circuit_breaker.failure_count = 0
                logger.info("Embedding circuit breaker closed - service recovered")
            elif embedding_circuit_breaker.failure_count > 0:
                # Successful call after some failures — decay the counter
                embedding_circuit_breaker.failure_count = max(0, embedding_circuit_breaker.failure_count - 1)
            if attempt > 0:
                logger.info(f"Embedding succeeded on retry {attempt}")
            return result

        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                # All retries exhausted — NOW record one failure on the breaker
                embedding_circuit_breaker.failure_count += 1
                embedding_circuit_breaker.last_failure_time = time.time()
                if embedding_circuit_breaker.failure_count >= embedding_circuit_breaker.failure_threshold:
                    embedding_circuit_breaker.state = "open"
                    logger.error(f"Embedding circuit breaker opened after {embedding_circuit_breaker.failure_count} top-level failures")
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise e

            # Exponential backoff: 1s, 2s, 4s, 8s
            wait_time = min(2 ** attempt, 8)
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)


async def get_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    timeout: float = 60.0
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
    timeout: float = 60.0
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
    text = _truncate_for_embedding(text)
    
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


async def warmup_ollama(model: Optional[str] = None) -> bool:
    """
    Pre-warm Ollama by loading the embedding model into memory.
    
    Sends a trivial embedding request on startup so the model is
    already loaded when real requests arrive. Cold loads on NUC
    can take 10-15s, causing timeouts on first real request.
    
    Returns:
        True if warmup succeeded, False otherwise
    """
    model = model or OLLAMA_EMBED_MODEL
    
    try:
        # First check Ollama is reachable
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = await client.get(f"{OLLAMA_URL}/api/tags")
            health.raise_for_status()
            logger.info(f"Ollama reachable at {OLLAMA_URL}")
        
        # Send a trivial embedding to force model load
        logger.info(f"Warming up Ollama model '{model}'...")
        embedding = await _get_embedding_raw("warmup", model, timeout=120.0)
        dim = len(embedding)
        logger.info(f"Ollama warmup complete — model '{model}' loaded ({dim}-dim embeddings)")
        
        # Reset circuit breaker to clean state after warmup
        embedding_circuit_breaker.state = "closed"
        embedding_circuit_breaker.failure_count = 0
        embedding_circuit_breaker.last_failure_time = None
        
        return True
        
    except Exception as e:
        logger.warning(f"Ollama warmup failed (non-fatal): {e}")
        return False
