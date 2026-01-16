"""
Embedding service using Ollama.

Provides async embedding generation for document text.
Default model: nomic-embed-text (768 dimensions)
"""
import os
from typing import List, Optional

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


async def get_embedding(
    text: str, 
    model: Optional[str] = None,
    timeout: float = 30.0
) -> List[float]:
    """
    Generate embedding using Ollama.
    
    Args:
        text: Text to embed
        model: Model name (default: nomic-embed-text)
        timeout: Request timeout in seconds
        
    Returns:
        768-dimensional embedding vector
        
    Raises:
        httpx.HTTPError: On request failure
    """
    model = model or OLLAMA_EMBED_MODEL
    
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
