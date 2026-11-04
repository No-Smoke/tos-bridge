"""The backfill must restore only text it can tie to the point it is restoring."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "match_transcripts", Path(__file__).parents[1] / "scripts/backfill/match_transcripts.py")
match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match)

KEY = ("docs", "11111111-1111-1111-1111-111111111111")
PAYLOAD = {"title": "A note", "path": "p", "summary": "short", "category": "memory",
           "neo4j_entities": [], "synced_at": "2026-01-01"}


def call(text="Body text.", timestamp="2026-01-01", **overrides):
    return (timestamp, {"text": text, "title": "A note", "summary": "short", "collection": "docs",
                        **overrides})


def test_plan_keeps_payload_fields_and_uses_the_newest_call():
    plan, refused, unmatched = match.select_documents(
        {KEY: PAYLOAD}, {KEY: [call("old"), call("new", timestamp="2026-02-01")]})
    assert [d["text"] for d in plan] == ["new"]
    assert plan[0]["metadata"] == {"category": "memory"}
    assert not refused and not unmatched


def test_summary_may_be_the_first_200_characters_of_the_text():
    text = "x" * 300
    plan, _, _ = match.select_documents({KEY: {**PAYLOAD, "summary": text[:200]}},
                                        {KEY: [call(text, summary=None)]})
    assert len(plan) == 1


def test_entities_stored_as_json_strings_are_decoded():
    entities = [{"name": "A"}]
    plan, _, _ = match.select_documents(
        {KEY: PAYLOAD}, {KEY: [call(entities=json.dumps(entities), relationships="[]")]})
    assert plan[0]["entities"] == entities
    assert plan[0]["relationships"] == []


def test_refusals():
    cases = {
        "empty": call(" "),
        "leaked_envelope": call("Body</text>\n<collection>docs</collection>"),
        "title_mismatch": call(title="Another"),
        "summary_mismatch": call(summary="different"),
        "looks_like_secret": call("api_key = " + "a" * 30),
    }
    for reason, found in cases.items():
        plan, refused, _ = match.select_documents({KEY: PAYLOAD}, {KEY: [found]})
        assert not plan, reason
        assert list(refused) == [reason]


def test_a_point_with_no_call_is_unmatched():
    assert match.select_documents({KEY: PAYLOAD}, {}) == ([], {}, [KEY])


def test_read_calls_keeps_only_successful_calls(tmp_path):
    qdrant_id = KEY[1]
    lines = [
        {"timestamp": "t1", "message": {"content": [
            {"type": "tool_use", "id": "good", "name": match.TOOL, "input": {"collection": "docs", "text": "a"}},
            {"type": "tool_use", "id": "bad", "name": match.TOOL, "input": {"collection": "docs", "text": "b"}}]}},
        {"message": {"content": [
            {"type": "tool_result", "tool_use_id": "good",
             "content": json.dumps({"status": "success", "qdrant_id": qdrant_id})},
            {"type": "tool_result", "tool_use_id": "bad",
             "content": json.dumps({"status": "error", "qdrant_id": qdrant_id})}]}},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    found = match.read_calls([path])
    assert [args["text"] for _, args in found[KEY]] == ["a"]
