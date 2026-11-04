#!/usr/bin/env python3
"""Re-store the documents in plan.json through a tos-bridge v0.6.0 or later server.

Each document goes through store_doc_with_graph with its original path, title, summary,
payload fields, entities and relationships, then get_doc must return the same text. The id
the server returns must match the point being restored. Progress is kept in done.json, so a
second run retries only what failed.

Start with --sample, and read the result before running the rest. Snapshot the collections
first. Environment: everything the server needs (QDRANT_URL, QDRANT_API_KEY, NEO4J_URI,
NEO4J_USER, NEO4J_PASSWORD, OLLAMA_URL, OLLAMA_EMBED_MODEL).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

SERVER_ENV = ("QDRANT_URL", "QDRANT_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
              "OLLAMA_URL", "OLLAMA_EMBED_MODEL")
CALL_TIMEOUT_SECONDS = 180
SAVE_EVERY = 25


class Server:
    """A tos-bridge server over stdio."""

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.last_id = 0
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                     "clientInfo": {"name": "tos-bridge-backfill", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, message: dict) -> None:
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        self.last_id += 1
        wanted = self.last_id
        self._send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params})
        reply: dict = {}

        def read() -> None:
            while line := self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("id") == wanted:
                    reply.update(message)
                    return

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(CALL_TIMEOUT_SECONDS)
        if not reply:
            raise TimeoutError(method)
        return reply

    def call(self, tool: str, **arguments: Any) -> dict:
        result = self._request("tools/call", {"name": tool, "arguments": arguments})["result"]
        text = result["content"][0]["text"]
        if result.get("isError"):
            return {"status": "error", "error": text[:240]}
        try:
            return json.loads(text)
        except ValueError:
            return {"status": "error", "error": text[:240]}

    def close(self) -> None:
        self.process.stdin.close()
        self.process.terminate()


def choose_sample(plan: list[dict]) -> list[dict]:
    """A small sample that exercises each collection, chunking and graph writes."""
    sample: dict[str, dict] = {}
    for document in plan:
        sample.setdefault(document["collection"], document)
    with_graph = [d for d in plan if d["relationships"]]
    for document in sorted(with_graph, key=lambda d: -len(d["text"]))[:2]:
        sample.setdefault(document["qdrant_id"], document)
    return list(sample.values())


def restore(server: Server, document: dict) -> str | None:
    """Store one document and read it back. Returns an error, or None on success."""
    stored = server.call(
        "store_doc_with_graph", text=document["text"], collection=document["collection"],
        title=document["title"], path=document["path"], summary=document["summary"],
        metadata=document["metadata"], entities=document["entities"],
        relationships=document["relationships"])
    if stored.get("status") != "success" or stored.get("qdrant_id") != document["qdrant_id"]:
        return json.dumps({k: stored.get(k) for k in ("status", "qdrant_id", "error")})[:300]
    read = server.call("get_doc", collection=document["collection"], qdrant_id=document["qdrant_id"])
    if (read.get("text") or "").strip() != document["text"].strip():
        return f"read-back differs: {len(read.get('text') or '')} v {len(document['text'])} chars"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--image", default="vanya/tos-bridge:v0.6.0")
    parser.add_argument("--wrapper", default="", help="command that supplies the server's environment")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    plan = json.loads((args.workdir / "plan.json").read_text())
    done_path = args.workdir / "done.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    command = [*shlex.split(args.wrapper), "docker", "run", "-i", "--rm"]
    for name in SERVER_ENV:
        command += ["-e", name]
    command.append(args.image)

    server = Server(command)
    succeeded = failed = 0
    try:
        for document in choose_sample(plan) if args.sample else plan:
            key = f"{document['collection']}/{document['qdrant_id']}"
            if done.get(key) == "ok":
                continue
            try:
                error = restore(server, document)
            except TimeoutError as timed_out:
                done[key] = f"timeout: {timed_out}"
                failed += 1
                print("STOP", key, done[key], flush=True)
                break
            done[key] = error or "ok"
            if error:
                failed += 1
                print("FAIL", key, error, flush=True)
            else:
                succeeded += 1
            if (succeeded + failed) % SAVE_EVERY == 0:
                done_path.write_text(json.dumps(done))
    finally:
        done_path.write_text(json.dumps(done))
        server.close()
    print(f"this run: ok {succeeded} failed {failed}; all runs: ok {sum(v == 'ok' for v in done.values())}")


if __name__ == "__main__":
    main()
