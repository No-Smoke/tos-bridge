"""
Embedding service using Ollama.

Provides async embedding generation for document text.
Default model: mxbai-embed-large (1024 dimensions)

v0.4.0 additions:
- In-process LRU cache keyed by sha256(text||model). Most repeated queries
  in a session re-hit the same texts; 5-50x speedup on follow-up searches.
- True batch endpoint via Ollama /api/embed list input (Ollama >= 0.1.40),
  with per-batch chunking and the same circuit-breaker semantics.
"""
import os
import time
import hashlib
import asyncio
import logging
from collections import OrderedDict
from typing import List, Optional

import httpx

# Set up logging
logger = logging.getLogger("tos-bridge.embedding")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# Bounded LRU cache for single-text embeddings.
# `OrderedDict.move_to_end` gives us O(1) recency without a separate LRU library.
# Cap is tunable via OLLAMA_EMBED_CACHE_SIZE.
_EMBED_CACHE_MAX = int(os.getenv("OLLAMA_EMBED_CACHE_SIZE", "2048"))
_embed_cache: "OrderedDict[str, List[float]]" = OrderedDict()
_embed_cache_hits = 0
_embed_cache_misses = 0


def _cache_key(text: str, model: str) -> str:
    """sha256 of (model || \\x00 || text) — stable across processes."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[List[float]]:
    global _embed_cache_hits, _embed_cache_misses
    if key in _embed_cache:
        _embed_cache.move_to_end(key)
        _embed_cache_hits += 1
        return _embed_cache[key]
    _embed_cache_misses += 1
    return None


def _cache_put(key: str, vec: List[float]) -> None:
    _embed_cache[key] = vec
    _embed_cache.move_to_end(key)
    while len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)


def embed_cache_stats() -> dict:
    """Expose cache stats for diagnostic tooling."""
    total = _embed_cache_hits + _embed_cache_misses
    return {
        "size": len(_embed_cache),
        "max": _EMBED_CACHE_MAX,
        "hits": _embed_cache_hits,
        "misses": _embed_cache_misses,
        "hit_rate": (_embed_cache_hits / total) if total else 0.0,
    }

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


def _truncate_for_embedding(text: str, max_chars: int = 1400) -> str:
    """
    Client-side pre-filter before Ollama's server-side truncation.

    Ollama's /api/embed with truncate:true is authoritative, but capping
    here reduces wire bytes and tokenizer work on pathological inputs.
    1400 chars is safe even for short-token text (~2 chars/token) against
    mxbai-embed-large's 512-token window. Cuts at word boundary.
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


def _connection_error_hint(underlying: Exception) -> str:
    """Format a clear error message when Ollama is unreachable.

    The bare httpx.ConnectError message is "All connection attempts failed"
    with no URL or remediation hint, which is opaque when an MCP server's
    env block drops OLLAMA_URL and the embedder silently falls back to
    localhost:11434. Surface the URL we tried and point operators at the
    config they need to fix.
    """
    return (
        f"Cannot reach Ollama at {OLLAMA_URL}. "
        f"If OLLAMA_URL is unset, the embedding module falls back to "
        f"http://localhost:11434 — set OLLAMA_URL in the MCP server's "
        f"env (e.g. ~/.claude.json mcpServers.<name>.env) and restart "
        f"the MCP process so the new env is inherited. "
        f"Underlying: {underlying!r}"
    )


async def _get_embedding_raw(
    text: str,
    model: str,
    timeout: float = 60.0
) -> List[float]:
    """Raw embedding function without circuit breaker (internal use)."""
    text = _truncate_for_embedding(text)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={
                    "model": model,
                    "input": text,
                    "truncate": True,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
    except httpx.ConnectError as e:
        raise httpx.ConnectError(_connection_error_hint(e)) from e


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

    # Cache fast path — most session-internal queries are duplicates.
    cache_key = _cache_key(text, model)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

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
            _cache_put(cache_key, result)
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
    timeout: float = 60.0,
    batch_size: int = 32,
) -> List[List[float]]:
    """Generate embeddings for many texts using Ollama's list input.

    Ollama's /api/embed accepts a list of strings (`input: [...]`) since
    v0.1.40 and returns one embedding per input in order. We chunk into
    `batch_size` payloads so a single huge call cannot lock the server,
    and we cache each individual result so future single-text calls hit
    the same LRU cache.

    Cached entries are returned without hitting Ollama; only uncached
    texts are batched.

    Args:
        texts: List of texts to embed (order preserved in return value).
        model: Override model name.
        timeout: Per-batch request timeout.
        batch_size: Max texts per HTTP call. 32 is empirically safe for
            mxbai-embed-large on a single 24GB GPU; tune for your hardware.

    Returns:
        List of embedding vectors in the same order as `texts`.
    """
    model = model or OLLAMA_EMBED_MODEL
    n = len(texts)
    if n == 0:
        return []

    # Resolve cache first; collect indices and truncated text for the misses.
    results: List[Optional[List[float]]] = [None] * n
    miss_indices: List[int] = []
    miss_texts: List[str] = []
    miss_keys: List[str] = []
    for i, raw in enumerate(texts):
        truncated = _truncate_for_embedding(raw)
        key = _cache_key(truncated, model)
        cached = _cache_get(key)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)
            miss_texts.append(truncated)
            miss_keys.append(key)

    if not miss_texts:
        return [r for r in results if r is not None]  # type: ignore[misc]

    # Honor the circuit breaker pre-call (same shape as single-text path).
    if embedding_circuit_breaker.state == "open":
        if embedding_circuit_breaker.last_failure_time and \
           time.time() - embedding_circuit_breaker.last_failure_time > embedding_circuit_breaker.reset_timeout:
            embedding_circuit_breaker.state = "half-open"
        else:
            raise Exception("Embedding service circuit breaker open - too many failures")

    # Chunked batch call to Ollama. Failures here propagate; the circuit
    # breaker is updated by the eventual exception path.
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for chunk_start in range(0, len(miss_texts), batch_size):
                chunk = miss_texts[chunk_start:chunk_start + batch_size]
                chunk_keys = miss_keys[chunk_start:chunk_start + batch_size]
                chunk_indices = miss_indices[chunk_start:chunk_start + batch_size]

                response = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": model, "input": chunk, "truncate": True},
                )
                response.raise_for_status()
                payload = response.json()
                embeddings = payload.get("embeddings") or []
                if len(embeddings) != len(chunk):
                    raise RuntimeError(
                        f"Ollama returned {len(embeddings)} embeddings for {len(chunk)} inputs"
                    )

                for i, key, vec in zip(chunk_indices, chunk_keys, embeddings):
                    _cache_put(key, vec)
                    results[i] = vec
    except httpx.ConnectError as e:
        raise httpx.ConnectError(_connection_error_hint(e)) from e

    # All slots filled; pacify the type checker.
    return [r for r in results if r is not None]  # type: ignore[misc]


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
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": model,
                "input": text,
                "truncate": True,
            }
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


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
        logger.warning(
            f"Ollama warmup failed at {OLLAMA_URL} (non-fatal): {e!r}"
        )
        return False
