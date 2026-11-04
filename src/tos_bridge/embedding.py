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
from typing import Awaitable, Callable, Iterator, List, Optional, Tuple, TypeVar

import httpx

# Set up logging
logger = logging.getLogger("tos-bridge.embedding")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# Ollama's `prompt_eval_count` stops at the model's context window for input it
# truncated silently (measured: mxbai-embed-large reports exactly 512), so
# reaching this value means the end of the text never reached the model.
OLLAMA_EMBED_CONTEXT_TOKENS = int(os.getenv("OLLAMA_EMBED_CONTEXT_TOKENS", "512"))

# Measured against mxbai-embed-large on Ollama 0.20.5: 1,400 chars of English is
# 303 tokens, so this fits ordinary prose. Denser text is split further by
# embed_document, on the model's own signal rather than a character estimate.
CHUNK_MAX_CHARS = 1200
_MIN_SPLIT_CHARS = 64

SHRINK_RATIOS = (1.0, 0.75, 0.5, 0.35, 0.25)
_CONTEXT_ERROR = "exceeds the context length"

T = TypeVar("T")


class EmbeddingContextExceeded(Exception):
    """Ollama rejected the input as longer than the model's context window."""

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
    Client-side cap on single-string embedding, for queries and patterns.

    This bounds wire bytes and tokenizer work; it does NOT make the input safe.
    Ollama returns HTTP 400 on token-dense text well under this cap even with
    truncate:true, so callers go through the SHRINK_RATIOS retry. Documents do
    not use this path at all: see embed_document. Cuts at a word boundary.
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


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Split text into contiguous spans of at most max_chars.

    Prefers paragraph, line, sentence, then word boundaries in the back half of
    each window. ``"".join(split_text(t)) == t`` always holds: a document is read
    back by concatenating its spans, so no character may be dropped or repeated.
    """
    spans: List[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max_chars
        for boundary in ("\n\n", "\n", ". ", " "):
            found = window.rfind(boundary)
            if found >= max_chars // 2:
                cut = found + len(boundary)
                break
        spans.append(rest[:cut])
        rest = rest[cut:]
    spans.append(rest)
    return spans


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


def _shrink_candidates(text: str) -> Iterator[str]:
    """Yield `text`, then ever-shorter prefixes, for retrying on a context error."""
    for ratio in SHRINK_RATIOS:
        yield text if ratio == 1.0 else text[:max(1, int(len(text) * ratio))]


def _raise_for_embed_status(response: httpx.Response) -> None:
    """Raise EmbeddingContextExceeded for an over-long input, else the HTTP error."""
    if response.status_code in (400, 500) and _CONTEXT_ERROR in response.text:
        raise EmbeddingContextExceeded(response.text)
    response.raise_for_status()


async def _post_embed(
    text: str,
    model: str,
    timeout: float,
) -> Tuple[List[float], Optional[int]]:
    """One /api/embed call. Returns the vector and Ollama's prompt token count."""
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
    except httpx.ConnectError as e:
        raise httpx.ConnectError(_connection_error_hint(e)) from e
    _raise_for_embed_status(response)
    data = response.json()
    return data["embeddings"][0], data.get("prompt_eval_count")


async def _get_embedding_raw(
    text: str,
    model: str,
    timeout: float = 60.0
) -> List[float]:
    """Embed one string, shrinking on Ollama's context-length error.

    truncate:true does not reliably truncate: token density, not character
    count, decides whether Ollama returns 400, so no fixed cap is safe. A query
    must still produce a vector, so this retries on a shorter prefix.
    """
    text = _truncate_for_embedding(text)
    for payload in _shrink_candidates(text):
        try:
            vector, _ = await _post_embed(payload, model, timeout)
        except EmbeddingContextExceeded:
            logger.info("Embedding input of %d chars over context; shrinking", len(payload))
            continue
        return vector
    raise EmbeddingContextExceeded(
        f"Input of {len(text)} chars exceeds the context of {model} at every shrink step"
    )


async def _with_retries(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 5,
) -> T:
    """Run an Ollama call with the circuit breaker and exponential backoff.

    EmbeddingContextExceeded is a verdict on the input, not on the service, so it
    propagates at once: retrying is pointless and it must not trip the breaker.
    """
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

    for attempt in range(max_retries):
        try:
            result = await operation()
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

        except EmbeddingContextExceeded:
            raise

        except Exception as e:
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

    raise ValueError(f"max_retries must be at least 1, got {max_retries}")


async def get_embedding(
    text: str,
    model: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 5
) -> List[float]:
    """
    Generate embedding using Ollama with circuit breaker and retry logic.

    Over-long input is shrunk, so the vector may cover only a prefix of `text`.
    That suits a query; use embed_document for content that must stay findable.

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

    result = await _with_retries(
        lambda: _get_embedding_raw(text, model, timeout), max_retries
    )
    _cache_put(cache_key, result)
    return result


async def _embed_span(
    span: str,
    model: str,
    timeout: float,
) -> List[Tuple[str, List[float]]]:
    """Embed a span whole, splitting it wherever the model could not take it all."""
    fits = True
    vector: List[float] = []
    try:
        vector, tokens = await _with_retries(lambda: _post_embed(span, model, timeout))
        # Silent truncation returns 200; only the token count gives it away.
        fits = tokens is None or tokens < OLLAMA_EMBED_CONTEXT_TOKENS
    except EmbeddingContextExceeded:
        fits = False

    if fits:
        return [(span, vector)]
    if len(span) <= _MIN_SPLIT_CHARS:
        return [(span, vector or await _get_embedding_raw(span, model, timeout))]

    pairs: List[Tuple[str, List[float]]] = []
    for part in split_text(span, max_chars=len(span) * 2 // 3):
        pairs.extend(await _embed_span(part, model, timeout))
    return pairs


async def embed_document(
    text: str,
    model: Optional[str] = None,
    timeout: float = 60.0,
) -> List[Tuple[str, List[float]]]:
    """Embed a whole document so that every character reaches the model.

    Returns (span, vector) pairs in document order, and the spans concatenate to
    exactly `text`. Results skip the LRU cache, which get_embedding fills with
    vectors that may cover only a shrunk prefix.
    """
    model = model or OLLAMA_EMBED_MODEL
    pairs: List[Tuple[str, List[float]]] = []
    for span in split_text(text):
        pairs.extend(await _embed_span(span, model, timeout))
    return pairs


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
                try:
                    _raise_for_embed_status(response)
                except EmbeddingContextExceeded:
                    # One dense text fails the whole request, so isolate it.
                    embeddings = [
                        await _get_embedding_raw(item, model, timeout) for item in chunk
                    ]
                else:
                    embeddings = response.json().get("embeddings") or []
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
        for payload in _shrink_candidates(text):
            response = client.post(
                f"{OLLAMA_URL}/api/embed",
                json={
                    "model": model,
                    "input": payload,
                    "truncate": True,
                }
            )
            try:
                _raise_for_embed_status(response)
            except EmbeddingContextExceeded:
                continue
            return response.json()["embeddings"][0]
    raise EmbeddingContextExceeded(
        f"Input of {len(text)} chars exceeds the context of {model} at every shrink step"
    )


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
