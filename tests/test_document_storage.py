"""Document bodies survive storage, reach the index whole, and read back exactly.

Qdrant here is qdrant-client's real local mode, so filters, deletes and paging
behave as they do on the server. Only Ollama and Neo4j are faked. The fake Ollama
reproduces what mxbai-embed-large did when measured on 2026-11-05: HTTP 400 on
over-long CJK, silent truncation to 512 tokens otherwise, and a vector built only
from the tokens it kept, so content lost to truncation is genuinely unfindable.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastmcp import Client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from tos_bridge import embedding, graph_tools
from tos_bridge.server import mcp

DIM = 8192
COLLECTION = "test_docs"
CONTEXT = 512


def _tokens(text: str) -> list[str]:
    """Words are one token, CJK one per char, and other runs one per 2 chars.

    Those rates match the measurement: 500 chars of CJK was 502 tokens, and
    random hex reached the 512 cap by 800 chars.
    """
    out: list[str] = []
    for word in re.findall(r"[\u4e00-\u9fff]|[^\W\d_]+|\d+|\S", text):
        if re.fullmatch(r"[\u4e00-\u9fff]", word):
            out.append(word)
        elif re.fullmatch(r"[a-z]+", word.lower()) and re.search(r"[aeiouy]", word.lower()):
            out.append(word.lower())
        else:
            out.extend(word[i:i + 2] for i in range(0, len(word), 2))
    return out


def _vector(tokens: list[str]) -> list[float]:
    vec = [0.0] * DIM
    for token in tokens:
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[digest % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed_response(text: str) -> httpx.Response:
    request = httpx.Request("POST", "http://ollama/api/embed")
    tokens = _tokens(text)
    if len(tokens) > CONTEXT and re.search(r"[一-鿿]", text):
        return httpx.Response(
            400,
            json={"error": "the input length exceeds the context length"},
            request=request,
        )
    kept = tokens[:CONTEXT]
    return httpx.Response(
        200,
        json={"embeddings": [_vector(kept)], "prompt_eval_count": len(kept)},
        request=request,
    )


class FakeOllama:
    def __init__(self):
        self.inputs: list[str] = []

    async def post(self, _client, _url, json):
        self.inputs.append(json["input"])
        if isinstance(json["input"], list):
            if any(_embed_response(t).status_code == 400 for t in json["input"]):
                return _embed_response(next(
                    t for t in json["input"] if _embed_response(t).status_code == 400
                ))
            return httpx.Response(
                200,
                json={"embeddings": [_vector(_tokens(t)[:CONTEXT]) for t in json["input"]]},
                request=httpx.Request("POST", "http://ollama/api/embed"),
            )
        return _embed_response(json["input"])


@pytest.fixture
def ollama():
    fake = FakeOllama()

    async def post(self, url, json=None, **_):
        return await fake.post(self, url, json)

    embedding._embed_cache.clear()
    embedding.embedding_circuit_breaker.state = "closed"
    embedding.embedding_circuit_breaker.failure_count = 0
    with patch("httpx.AsyncClient.post", new=post):
        yield fake


async def _no_graph_cypher(_session, query, params=None):
    """Writes succeed; reads find no entities, so search stays purely vector."""
    return [{"neo4j_id": "n1"}] if "MERGE (d:Document" in query else []


@pytest.fixture
def qdrant(ollama):
    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    @asynccontextmanager
    async def fake_session():
        yield object()

    with (
        patch.object(graph_tools, "get_qdrant_client", return_value=client),
        patch.object(graph_tools, "neo4j_session", fake_session),
        patch.object(graph_tools, "_run_cypher", new=_no_graph_cypher),
    ):
        yield client


def _prose(words: int, seed: int = 1) -> str:
    """Paragraphed text of distinct pseudo-words, so a needle is unambiguous."""
    rng = random.Random(seed)
    syllables = ["ba", "de", "ki", "lo", "mu", "ne", "pi", "ro", "su", "ta", "ve", "zo"]
    tokens = ["".join(rng.choices(syllables, k=4)) for _ in range(words)]
    paragraphs = [" ".join(tokens[i:i + 60]) + "." for i in range(0, words, 60)]
    return "\n\n".join(paragraphs)


def _points(client: QdrantClient) -> list:
    return client.scroll(COLLECTION, limit=10_000, with_payload=True)[0]


async def _store(text: str, path: str = "docs/a.md", title: str = "A", **kwargs) -> dict:
    return await graph_tools.store_document_with_graph(
        text=text, collection=COLLECTION, title=title, path=path, **kwargs
    )


NEEDLE = "xylophone quokka zeppelin marmalade"


# -- 1. The body can be read back, and search returns an excerpt ---------------


async def test_short_document_reads_back_exactly(qdrant):
    text = "A short document.\n\nWith two paragraphs."
    stored = await _store(text)
    assert stored["chunks"] == 1

    doc = await graph_tools.get_document_tool(COLLECTION, path="docs/a.md")

    assert doc["text"] == text
    assert doc["complete"] is True
    assert doc["body_source"] == "text"


async def test_long_document_reads_back_exactly_and_payloads_stay_small(qdrant):
    text = _prose(12_000)
    stored = await _store(text, summary="s", metadata={"category": "note"})
    assert stored["chunks"] > 10
    assert stored["body_chars"] == len(text)

    doc = await graph_tools.get_document_tool(COLLECTION, path="docs/a.md")

    assert doc["text"] == text
    assert doc["chunk_count"] == stored["chunks"]
    assert doc["payload"]["category"] == "note"
    points = _points(qdrant)
    assert len(points) == stored["chunks"]
    assert max(len(p.payload["text"]) for p in points) <= embedding.CHUNK_MAX_CHARS


async def test_read_back_by_title_id_and_chunk_id(qdrant):
    text = _prose(3_000)
    stored = await graph_tools.store_document_with_graph(text=text, collection=COLLECTION, title="No path")

    assert (await graph_tools.get_document_tool(COLLECTION, title="No path"))["text"] == text
    assert (await graph_tools.get_document_tool(COLLECTION, qdrant_id=stored["qdrant_id"]))["text"] == text
    chunk_id = graph_tools._chunk_point_id(stored["qdrant_id"], 2)
    by_chunk = await graph_tools.get_document_tool(COLLECTION, qdrant_id=chunk_id)
    assert by_chunk["qdrant_id"] == stored["qdrant_id"]
    assert by_chunk["text"] == text


async def test_missing_document_and_missing_arguments(qdrant):
    assert (await graph_tools.get_document_tool(COLLECTION, path="nope"))["status"] == "not_found"
    with pytest.raises(ValueError):
        await graph_tools.get_document_tool(COLLECTION)


async def test_a_lost_chunk_is_reported_not_hidden(qdrant):
    stored = await _store(_prose(3_000))
    qdrant.delete(COLLECTION, points_selector=[graph_tools._chunk_point_id(stored["qdrant_id"], 1)])

    doc = await graph_tools.get_document_tool(COLLECTION, path="docs/a.md")

    assert doc["complete"] is False


async def test_empty_text_is_refused_before_anything_is_written(qdrant):
    with pytest.raises(ValueError, match="empty text"):
        await _store("")
    assert _points(qdrant) == []


# -- 2. Token-dense text does not fail ------------------------------------------


async def test_get_embedding_shrinks_on_400_without_backoff(ollama):
    dense = "漢" * 1390
    with patch("asyncio.sleep", new=AsyncMock(side_effect=AssertionError("backed off"))):
        vector = await embedding.get_embedding(dense)

    assert len(vector) == DIM
    assert [len(i) for i in ollama.inputs] == [1390, 1042, 695, 486]
    assert embedding.embedding_circuit_breaker.failure_count == 0


async def test_batch_isolates_a_dense_text(ollama):
    vectors = await embedding.get_embeddings_batch(["plain words here", "漢" * 1390])
    assert [len(v) for v in vectors] == [DIM, DIM]


def test_sync_path_shrinks_on_400():
    calls: list[int] = []

    def post(self, url, json=None, **_):
        calls.append(len(json["input"]))
        return _embed_response(json["input"])

    with patch("httpx.Client.post", new=post):
        vector = embedding.get_embedding_sync("漢" * 1390)

    assert len(vector) == DIM
    assert calls == [1390, 1042, 695, 486]


async def test_an_unrelated_400_is_still_an_error(ollama):
    async def bad(self, url, json=None, **_):
        return httpx.Response(400, json={"error": "invalid model"}, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", new=bad), patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(httpx.HTTPStatusError):
            await embedding.get_embedding("anything", max_retries=2)


@pytest.mark.parametrize(
    "dense",
    [
        pytest.param("漢字仮名" * 500, id="cjk-400"),
        pytest.param("".join(random.Random(2).choices("0123456789abcdef", k=2000)), id="hex-silent-truncation"),
    ],
)
async def test_dense_document_is_split_until_every_token_reaches_the_model(qdrant, ollama, dense):
    await _store(dense)

    doc = await graph_tools.get_document_tool(COLLECTION, path="docs/a.md")
    assert doc["text"] == dense
    for point in _points(qdrant):
        assert len(_tokens(point.payload["text"])) < CONTEXT


async def test_content_at_the_end_of_a_dense_span_is_findable(qdrant):
    """Shrinking a chunk would pass the 400 test and still lose this needle.

    The score is the assertion: with one document in the collection, the top hit
    is that document whether or not its vector saw the needle.
    """
    rng = random.Random(4)
    dense = "".join(chr(rng.randint(0x4e00, 0x9fff)) for _ in range(1100)) + " " + NEEDLE
    await _store(dense)

    found = await graph_tools.hybrid_search_tool(NEEDLE, COLLECTION, limit=1)

    assert NEEDLE in found["results"][0]["excerpt"]
    assert found["results"][0]["score"] > 0.05


# -- 3. Content past 1,400 characters is findable; no duplicates, no stale chunks


async def test_content_past_1400_chars_is_findable_by_both_search_tools(qdrant):
    text = _prose(4_000) + "\n\n" + NEEDLE + ".\n\n" + _prose(2_000, seed=3)
    assert text.index(NEEDLE) > 1400
    stored = await _store(text, title="Long")
    await _store(_prose(500, seed=9), path="docs/other.md", title="Other")

    hybrid = await graph_tools.hybrid_search_tool(NEEDLE, COLLECTION, limit=5)
    assert hybrid["results"][0]["id"] == stored["qdrant_id"]
    assert NEEDLE in hybrid["results"][0]["excerpt"]
    assert hybrid["results"][0]["chunk_index"] > 0

    graph = await graph_tools.graph_enhanced_search(NEEDLE, COLLECTION, limit=5)
    assert graph["results"][0]["qdrant_id"] == stored["qdrant_id"]
    assert NEEDLE in graph["results"][0]["excerpt"]
    assert graph["results"][0]["title"] == "Long"


async def test_search_returns_each_document_once_even_when_one_fills_a_page(qdrant):
    long_doc = " ".join(f"{NEEDLE} {w}." for w in _prose(4_000).split())
    stored_long = await _store(long_doc, title="Repeats")
    assert stored_long["chunks"] > 150, "must exceed several pages of hits"
    stored_other = await _store(_prose(200, seed=5) + " " + NEEDLE, path="docs/b.md", title="Other")

    hybrid = await graph_tools.hybrid_search_tool(NEEDLE, COLLECTION, limit=2)
    assert [r["id"] for r in hybrid["results"]] == [stored_long["qdrant_id"], stored_other["qdrant_id"]]

    graph = await graph_tools.graph_enhanced_search(NEEDLE, COLLECTION, limit=2)
    assert {r["qdrant_id"] for r in graph["results"]} == {stored_long["qdrant_id"], stored_other["qdrant_id"]}


async def test_restoring_a_shorter_version_leaves_no_stale_chunks(qdrant):
    first = await _store(_prose(6_000) + "\n\n" + NEEDLE)
    assert first["chunks"] > 5
    second = await _store("Now a single short sentence.")

    assert second["qdrant_id"] == first["qdrant_id"]
    assert [str(p.id) for p in _points(qdrant)] == [first["qdrant_id"]]
    doc = await graph_tools.get_document_tool(COLLECTION, path="docs/a.md")
    assert doc["text"] == "Now a single short sentence."
    found = await graph_tools.hybrid_search_tool(NEEDLE, COLLECTION, limit=5)
    assert NEEDLE not in found["results"][0]["excerpt"]


async def test_restoring_at_intermediate_length_keeps_only_the_new_chunks(qdrant):
    await _store(_prose(6_000))
    shorter = _prose(2_000, seed=7)
    second = await _store(shorter)

    assert len(_points(qdrant)) == second["chunks"]
    assert (await graph_tools.get_document_tool(COLLECTION, path="docs/a.md"))["text"] == shorter


async def test_restoring_does_not_touch_another_documents_chunks(qdrant):
    other = _prose(3_000, seed=8)
    await _store(other, path="docs/other.md")
    await _store(_prose(3_000))
    await _store("short")

    assert (await graph_tools.get_document_tool(COLLECTION, path="docs/other.md"))["text"] == other


async def test_payload_filters_match_content_in_later_chunks(qdrant):
    text = _prose(3_000) + "\n\n" + NEEDLE
    await _store(text, title="Filter me", metadata={"category": "wanted"})
    await _store(NEEDLE, path="docs/b.md", title="Other", metadata={"category": "unwanted"})

    found = await graph_tools.hybrid_search_tool(
        NEEDLE, COLLECTION, limit=5, payload_filter={"category": "wanted"}
    )

    assert [r["title"] for r in found["results"]] == ["Filter me"]
    assert NEEDLE in found["results"][0]["excerpt"]


async def test_named_vector_collection(qdrant):
    qdrant.create_collection("named", vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    text = _prose(3_000) + "\n\n" + NEEDLE
    await graph_tools.store_document_with_graph(text=text, collection="named", title="T", path="p")

    found = await graph_tools.hybrid_search_tool(NEEDLE, "named", limit=1)

    assert NEEDLE in found["results"][0]["excerpt"]
    assert (await graph_tools.get_document_tool("named", path="p"))["text"] == text


async def test_delete_point_removes_the_documents_chunks(qdrant):
    stored = await _store(_prose(3_000))
    other = await _store("Keep me.", path="docs/b.md")

    await graph_tools.delete_point_tool(COLLECTION, stored["qdrant_id"])

    assert [str(p.id) for p in _points(qdrant)] == [other["qdrant_id"]]


async def test_metadata_cannot_corrupt_chunk_bookkeeping_and_body_is_not_copied(qdrant):
    text = _prose(3_000)
    stored = await _store(
        text,
        metadata={"text": "wrong", "parent_id": "wrong", "chunk_count": 99, "chunk_index": 7, "body": "workaround copy"},
    )

    points = {str(p.id): p.payload for p in _points(qdrant)}
    parent = points.pop(stored["qdrant_id"])
    assert "parent_id" not in parent
    assert parent["chunk_count"] == stored["chunks"]
    assert parent["body"] == "workaround copy"
    assert points and all("body" not in p and p["parent_id"] == stored["qdrant_id"] for p in points.values())
    assert (await graph_tools.get_document_tool(COLLECTION, path="docs/a.md"))["text"] == text


async def test_graph_discovered_document_gets_an_excerpt(qdrant):
    stored = await _store("Body of the graph neighbour.", path="docs/n.md", title="Neighbour")
    seed = await _store(NEEDLE, path="docs/s.md", title="Seed")

    async def run_cypher(_session, query, params=None):
        if "collect(DISTINCT e.name) AS entities" in query:
            return [{"doc_id": seed["qdrant_id"], "entities": ["E"]}]
        return [{
            "qdrant_id": stored["qdrant_id"],
            "title": "Neighbour",
            "summary": "from neo4j",
            "shared_entities": ["E"],
            "entity_count": 1,
        }]

    # Keep the neighbour out of the vector hits so only the graph can surface it.
    seed_hit = qdrant.query_points(
        COLLECTION,
        query=qdrant.retrieve(COLLECTION, ids=[seed["qdrant_id"]], with_vectors=True)[0].vector,
        limit=1,
        with_payload=True,
    ).points
    with (
        patch.object(graph_tools, "_run_cypher", new=run_cypher),
        patch.object(graph_tools, "_search_documents", return_value=seed_hit),
    ):
        graph = await graph_tools.graph_enhanced_search(NEEDLE, COLLECTION, limit=5)

    neighbour = next(r for r in graph["results"] if r["discovered_via_graph"])
    assert neighbour["excerpt"] == "Body of the graph neighbour."
    assert neighbour["title"] == "Neighbour"


# -- Legacy points must keep reading -------------------------------------------


def _legacy(client: QdrantClient, path: str, payload: dict, seed_text: str) -> str:
    point_id = graph_tools._stable_doc_id(path, payload.get("title", ""))
    client.upsert(COLLECTION, points=[PointStruct(
        id=point_id, vector=_vector(_tokens(seed_text)), payload={"path": path, **payload},
    )])
    return point_id


async def test_memory_workaround_body_with_part_path_still_reads(qdrant):
    path = "memory://-home-vanya-git-IWO/some-slug#part-2"
    body = "Part two of the memory. " + NEEDLE
    point_id = _legacy(qdrant, path, {
        "title": "some-slug (part 2 of 4)", "summary": "hook", "memory_slug": "some-slug",
        "part": 2, "parts": 4, "body": body,
    }, body)

    doc = await graph_tools.get_document_tool(COLLECTION, path=path)
    assert doc["text"] == body
    assert doc["body_source"] == "body"
    assert doc["chunk_count"] == 1
    assert (doc["payload"]["part"], doc["payload"]["parts"]) == (2, 4)

    found = await graph_tools.hybrid_search_tool(NEEDLE, COLLECTION, limit=1)
    hit = found["results"][0]
    assert hit["id"] == point_id
    assert hit["excerpt"] == body
    assert hit["payload"]["body"] == body, "callers read the workaround body from the payload"

    graph = await graph_tools.graph_enhanced_search(NEEDLE, COLLECTION, limit=1)
    assert graph["results"][0]["excerpt"] == body


async def test_nested_metadata_body_and_legacy_text_read(qdrant):
    _legacy(qdrant, "n", {"title": "N", "metadata": {"body": "nested body"}}, "nested body")
    _legacy(qdrant, "t", {"title": "T", "text": "legacy text"}, "legacy text")

    nested = await graph_tools.get_document_tool(COLLECTION, path="n")
    assert (nested["text"], nested["body_source"]) == ("nested body", "metadata.body")
    legacy = await graph_tools.get_document_tool(COLLECTION, path="t")
    assert (legacy["text"], legacy["body_source"]) == ("legacy text", "text")

    found = await graph_tools.hybrid_search_tool("legacy text", COLLECTION, limit=1)
    assert found["results"][0]["payload"]["text"] == "legacy text"


async def test_bodyless_legacy_point_reports_that_rather_than_failing(qdrant):
    _legacy(qdrant, "old", {"title": "Old", "summary": "Only this survived."}, "only this survived")

    doc = await graph_tools.get_document_tool(COLLECTION, path="old")
    assert (doc["status"], doc["text"], doc["body_source"]) == ("success", "", None)
    assert doc["summary"] == "Only this survived."

    found = await graph_tools.hybrid_search_tool("only this survived", COLLECTION, limit=1)
    assert found["results"][0]["excerpt"] == ""
    assert found["results"][0]["summary"] == "Only this survived."


async def test_restoring_over_a_legacy_point_keeps_its_id(qdrant):
    point_id = _legacy(qdrant, "docs/a.md", {"title": "A", "summary": "s", "body": "old"}, "old")
    stored = await _store("New text.")
    assert stored["qdrant_id"] == point_id
    assert (await graph_tools.get_document_tool(COLLECTION, path="docs/a.md"))["text"] == "New text."


# -- Identity and the MCP surface ----------------------------------------------


def test_chunk_ids_are_stable_and_derive_from_the_document():
    doc_id = graph_tools._stable_doc_id("docs/a.md", "ignored title")
    assert doc_id == graph_tools._stable_doc_id("docs/a.md", "other title")
    assert graph_tools._chunk_point_id(doc_id, 3) == graph_tools._chunk_point_id(doc_id, 3)
    assert graph_tools._chunk_point_id(doc_id, 3) != graph_tools._chunk_point_id(doc_id, 4)
    uuid.UUID(graph_tools._chunk_point_id(doc_id, 3))


@pytest.mark.parametrize("seed", range(20))
def test_split_text_is_lossless_and_bounded(seed):
    rng = random.Random(seed)
    alphabet = ["word ", "x", "\n", "\n\n", ". ", "   ", "漢", "long" * 40]
    text = "".join(rng.choices(alphabet, k=rng.randint(0, 900)))

    spans = embedding.split_text(text, max_chars=200)

    assert "".join(spans) == text
    assert all(len(s) <= 200 for s in spans)


def test_split_text_prefers_a_paragraph_break():
    first = "a" * 300 + ". " + "b" * 400
    spans = embedding.split_text(first + "\n\n" + "c" * 900)
    assert spans[0] == first + "\n\n"


async def test_store_and_get_doc_through_the_mcp_client(qdrant):
    text = _prose(3_000) + "\n\n" + NEEDLE
    async with Client(mcp) as client:
        stored = await client.call_tool("store_doc_with_graph", {
            "text": text, "collection": COLLECTION, "title": "Via MCP", "path": "docs/mcp.md",
        })
        doc = await client.call_tool("get_doc", {"collection": COLLECTION, "path": "docs/mcp.md"})
        found = await client.call_tool("hybrid_search", {"query": NEEDLE, "collection": COLLECTION, "limit": 1})

    assert stored.data["chunks"] > 1
    assert doc.data["text"] == text
    assert NEEDLE in found.data["results"][0]["excerpt"]
