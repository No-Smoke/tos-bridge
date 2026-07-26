"""Recover tool-call parameters that leaked into a string argument.

A malformed Claude tool call can close a parameter with a plain tag
(``</text>``) instead of the real delimiter. The harness then folds every
sibling parameter into that first value, so the call arrives with one
oversized ``text`` and no ``collection``/``title`` at all — and the resulting
"expected string, received undefined" error points at the wrong fields.

This module reverses that fold. It is deliberately fail-safe: the tail after
the marker must parse *completely* into known parameters, or the text is
returned untouched.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Tuple

log = logging.getLogger("tos_bridge.arg_recovery")

STRING_PARAMS = frozenset({"collection", "title", "path", "summary"})
JSON_PARAMS = frozenset({"metadata", "entities", "relationships"})
RECOVERABLE = STRING_PARAMS | JSON_PARAMS

_MARKER = re.compile(r"</(?:[a-z]+:)?text>")
_PAIR = re.compile(r"\s*<(?P<key>[a-z_]{2,20})>(?P<val>.*?)</(?P=key)>", re.DOTALL)
_CLOSER = re.compile(r"\s*</(?:[a-z]+:)?(?:parameter|invoke|function_calls)>")


def _parse_tail(tail: str) -> Dict[str, Any] | None:
    """Parse an envelope tail into parameters, or None if it isn't one."""
    recovered: Dict[str, Any] = {}
    pos = 0

    while pos < len(tail):
        pair = _PAIR.match(tail, pos)
        if pair:
            key = pair.group("key")
            if key not in RECOVERABLE:
                return None
            value = pair.group("val")
            if key in JSON_PARAMS:
                try:
                    recovered[key] = json.loads(value)
                except json.JSONDecodeError:
                    log.warning(
                        "Dropped unparseable JSON parameter from leaked envelope",
                        extra={"event": "envelope_json_dropped", "param": key},
                    )
            else:
                recovered[key] = value
            pos = pair.end()
            continue

        closer = _CLOSER.match(tail, pos)
        if closer:
            pos = closer.end()
            continue

        if not tail[pos:].strip():
            break
        return None

    return recovered or None


def split_leaked_envelope(text: str) -> Tuple[str, Dict[str, Any]]:
    """Split a leaked tool-call envelope off the end of ``text``.

    Returns the cleaned text and the parameters recovered from the envelope.
    When ``text`` carries no envelope it is returned unchanged with an empty
    mapping, so callers can apply this unconditionally.
    """
    match = _MARKER.search(text)
    while match:
        recovered = _parse_tail(text[match.end():])
        if recovered is not None:
            return text[:match.start()], recovered
        match = _MARKER.search(text, match.start() + 1)

    return text, {}
