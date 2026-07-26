"""Unit tests for recovering a leaked tool-call envelope from `text`.

A malformed Claude tool call closes the `text` parameter with a plain tag
instead of the real delimiter, so the harness folds every sibling parameter
into `text` and the call arrives with `collection`/`title` genuinely absent.
Observed 2026-07-26 on 6 of 7 consecutive `store_doc_with_graph` calls.
"""

from __future__ import annotations

from tos_bridge.arg_recovery import split_leaked_envelope


# Verbatim from session 54ad559c, tool_use toolu_0156wY8TxPfQR225QcCStLFA.
CORRUPTED_SAMPLE = (
    "claude-opus-5 EXISTS: 5/25 USD per MTok, same as claude-opus-4-8. "
    "I wrongly claimed it did not, from a cached catalog. Rule: verify models "
    "against the live API, never a cached catalog."
    "</text>\n<collection>project_memory_v2</collection>\n"
    "<title>iwo-dbos Opus 5 retraction 2026-07-26</title>\n"
)


def test_clean_text_is_returned_untouched():
    text = "An ordinary document with a / and a $ and\nseveral lines."
    assert split_leaked_envelope(text) == (text, {})


def test_recovers_collection_and_title_from_real_sample():
    clean, recovered = split_leaked_envelope(CORRUPTED_SAMPLE)

    assert clean.endswith("never a cached catalog.")
    assert "</text>" not in clean
    assert "<collection>" not in clean
    assert recovered == {
        "collection": "project_memory_v2",
        "title": "iwo-dbos Opus 5 retraction 2026-07-26",
    }


def test_recovers_json_params_and_trailing_invoke_closer():
    text = (
        "body</text>\n"
        "<collection>c</collection>\n<title>t</title>\n"
        '<metadata>{"category": "session"}</metadata>\n'
        '<entities>[{"name": "iwo-dbos", "type": "project"}]</entities>\n'
        "<relationships>[]</relationships>\n</invoke>\n"
    )
    clean, recovered = split_leaked_envelope(text)

    assert clean == "body"
    assert recovered["metadata"] == {"category": "session"}
    assert recovered["entities"] == [{"name": "iwo-dbos", "type": "project"}]
    assert recovered["relationships"] == []


def test_marker_without_a_valid_envelope_is_not_treated_as_a_leak():
    text = "Prose that quotes a </text> tag and then keeps going normally."
    assert split_leaked_envelope(text) == (text, {})


def test_unknown_parameter_name_in_tail_aborts_recovery():
    text = "body</text>\n<collection>c</collection>\n<not_a_param>x</not_a_param>\n"
    assert split_leaked_envelope(text) == (text, {})


def test_unparseable_json_param_is_dropped_but_strings_survive():
    text = "body</text>\n<collection>c</collection>\n<title>t</title>\n<metadata>{oops</metadata>\n"
    clean, recovered = split_leaked_envelope(text)

    assert clean == "body"
    assert recovered == {"collection": "c", "title": "t"}


def test_envelope_with_no_recoverable_pairs_is_left_alone():
    text = "body</text>\n</invoke>\n"
    assert split_leaked_envelope(text) == (text, {})
