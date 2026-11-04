#!/usr/bin/env python3
"""Find documents with no stored body and the transcript calls that can restore them.

tos-bridge before v0.6.0 embedded a document's text and then discarded it. Claude Code
transcripts still hold the text argument of each successful store_doc_with_graph call,
so a point can be matched to its call by Qdrant id and collection.

Reads Qdrant and the transcripts; writes only to --workdir. plan.json holds full document
text, so keep --workdir out of any repository.

Environment: QDRANT_URL, QDRANT_API_KEY.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable

TOOL = "mcp__tos-bridge__store_doc_with_graph"
DEFAULT_COLLECTIONS = ("project_memory_v2", "system_configuration", "memories_1024")
BODY_KEYS = ("text", "body", "content")
GRAPH_ONLY_KEYS = {"title", "path", "summary", "neo4j_entities", "synced_at"}
QDRANT_ID = re.compile(r'"qdrant_id"\s*:\s*"([0-9a-f-]{36})"')
LEAKED_ENVELOPE = re.compile(r"</text>|<collection>|</parameter>|<parameter name")
SECRET = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
    r"|(?:password|passwd|api[_-]?key|secret|token)\s*[=:]\s*['\"]?[A-Za-z0-9+/_-]{24,}",
    re.I,
)


def _squash(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def as_list(value: Any) -> list:
    """Some transcripts hold entities and relationships as JSON strings."""
    if isinstance(value, str):
        value = json.loads(value)
    return value or []


def has_body(payload: dict) -> bool:
    return any(payload.get(key) for key in BODY_KEYS)


def scroll(url: str, api_key: str, collection: str) -> Iterable[dict]:
    """Every point of a collection, without vectors."""
    offset = None
    while True:
        body: dict[str, Any] = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        request = urllib.request.Request(
            f"{url}/collections/{collection}/points/scroll",
            json.dumps(body).encode(),
            {"api-key": api_key, "content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)["result"]
        yield from result["points"]
        offset = result.get("next_page_offset")
        if not offset:
            return


def _message_blocks(path: Path) -> Iterable[tuple[dict, str]]:
    """Each content block of a transcript that could concern a store call, with its timestamp."""
    with open(path, errors="replace") as handle:
        for line in handle:
            if "store_doc_with_graph" not in line and "qdrant_id" not in line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            content = (entry.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict):
                    yield block, entry.get("timestamp") or ""


def _stored_point_id(result_block: dict) -> str | None:
    """The Qdrant id a tool result reports, when the result reports success."""
    text = result_block.get("content")
    if not isinstance(text, str):
        text = " ".join(part.get("text", "") for part in text or [] if isinstance(part, dict))
    found = QDRANT_ID.search(text or "")
    return found.group(1) if found and '"status":"success"' in text.replace(" ", "") else None


def read_calls(files: Iterable[Path]) -> dict[tuple[str, str], list[tuple[str, dict]]]:
    """Successful store calls, keyed by (collection, qdrant_id), as (timestamp, input)."""
    calls: dict[str, tuple[dict, str]] = {}
    stored: dict[str, str] = {}
    for path in files:
        for block, timestamp in _message_blocks(path):
            if block.get("type") == "tool_use" and block.get("name") == TOOL:
                calls[block["id"]] = (block["input"], timestamp)
            elif block.get("type") == "tool_result" and (point_id := _stored_point_id(block)):
                stored[block["tool_use_id"]] = point_id
    matched: dict[tuple[str, str], list[tuple[str, dict]]] = collections.defaultdict(list)
    for call_id, qdrant_id in stored.items():
        if call_id in calls:
            arguments, timestamp = calls[call_id]
            matched[(arguments.get("collection"), qdrant_id)].append((timestamp, arguments))
    return matched


def select_documents(
    points: dict[tuple[str, str], dict],
    calls: dict[tuple[str, str], list[tuple[str, dict]]],
) -> tuple[list[dict], dict[str, list[str]], list[tuple[str, str]]]:
    """Return the restore plan, the refused points by reason, and the unmatched points.

    The newest call wins. A call must carry the point's title, and either its summary or the
    first 200 characters the old code kept as one. Text with a leaked tool envelope, or
    that looks like it holds a secret, is refused and left for a person to read.
    """
    plan: list[dict] = []
    refused: dict[str, list[str]] = collections.defaultdict(list)
    unmatched: list[tuple[str, str]] = []
    for (collection, qdrant_id), payload in points.items():
        found = calls.get((collection, qdrant_id))
        if not found:
            unmatched.append((collection, qdrant_id))
            continue
        timestamp, arguments = max(found, key=lambda call: call[0])
        text = arguments.get("text") or ""
        name = f"{collection}/{qdrant_id}"
        reason = None
        if not text.strip():
            reason = "empty"
        elif LEAKED_ENVELOPE.search(text):
            reason = "leaked_envelope"
        elif _squash(payload.get("title")) != _squash(arguments.get("title")):
            reason = "title_mismatch"
        elif _squash(payload.get("summary")) not in {
            _squash(arguments.get("summary")), _squash(text[:200])}:
            reason = "summary_mismatch"
        elif SECRET.search(text):
            reason = "looks_like_secret"
        if reason:
            refused[reason].append(name)
            continue
        plan.append({
            "collection": collection,
            "qdrant_id": qdrant_id,
            "title": arguments.get("title"),
            "path": payload.get("path") or arguments.get("path") or None,
            "summary": payload.get("summary"),
            "text": text,
            "metadata": {k: v for k, v in payload.items() if k not in GRAPH_ONLY_KEYS},
            "entities": as_list(arguments.get("entities")),
            "relationships": as_list(arguments.get("relationships")),
            "call_timestamp": timestamp,
        })
    return plan, dict(refused), unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, default=Path.home() / ".claude/projects")
    parser.add_argument("--collection", action="append", dest="collections")
    args = parser.parse_args()
    url, api_key = os.environ["QDRANT_URL"].rstrip("/"), os.environ["QDRANT_API_KEY"]
    args.workdir.mkdir(parents=True, exist_ok=True)

    points = {}
    for collection in args.collections or DEFAULT_COLLECTIONS:
        for point in scroll(url, api_key, collection):
            payload = point["payload"]
            if not payload.get("chunk_index") and not has_body(payload):
                points[(collection, str(point["id"]))] = payload

    files = [p for p in args.transcripts.rglob("*.jsonl") if "store_doc_with_graph" in p.read_text(errors="replace")]
    plan, refused, unmatched = select_documents(points, read_calls(files))
    (args.workdir / "plan.json").write_text(json.dumps(plan))
    (args.workdir / "refused.json").write_text(json.dumps(refused, indent=1))
    (args.workdir / "unmatched.json").write_text(json.dumps(unmatched, indent=1))
    print(f"no body: {len(points)}  plan: {len(plan)}  unmatched: {len(unmatched)}")
    print("refused:", {reason: len(names) for reason, names in refused.items()})
    print("plan by collection:", dict(collections.Counter(d["collection"] for d in plan)))


if __name__ == "__main__":
    main()
