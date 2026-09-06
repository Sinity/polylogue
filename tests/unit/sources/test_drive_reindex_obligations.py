"""Standing obligations for the AI Studio Drive cohort ahead of a fresh reindex.

Three properties the live corpus depends on, each previously a defect and each
unprotected until now:

* **Attachment ownership** (polylogue-prjai). AI Studio exports carry runs of
  id-less, text-less turns sharing one timestamp; without the media block's
  ``metadata`` in the owner-comparison payload they collapse to one key and
  every attachment becomes unownable.
* **Schema currency** (polylogue-tu1f). The committed Gemini package must
  resolve the current ``chunkedPrompt`` export to a real candidate rather than
  falling back to the package default, which is what ``unseen_shape`` drift
  means.
* **Thread eligibility** (polylogue-n2blt). A zero-message Drive document is
  refused by one predicate on the production admission route, so it never
  reaches the derived thread projection.
"""

from __future__ import annotations

from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.ids import message_owner_resolution
from polylogue.schemas.drift_sentinel import UNSEEN_SHAPE, classify_schema_drift
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.sources.dispatch import require_positive_conversational_evidence
from polylogue.sources.parsers.drive import parse_chunked_prompt


def _document_turn(file_id: str) -> dict[str, Any]:
    """One id-less, text-less AI Studio turn citing a Drive file."""
    return {
        "role": "user",
        "text": "",
        "driveDocument": {"id": file_id},
    }


def _export(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runSettings": {"temperature": 1.0, "model": "models/synthetic"},
        "systemInstruction": {},
        "chunkedPrompt": {"chunks": chunks, "pendingInputs": []},
    }


def test_same_timestamp_document_turns_keep_distinguishable_owners() -> None:
    """polylogue-prjai: two id-less turns citing different files are two turns."""
    session = parse_chunked_prompt(
        Provider.GEMINI,
        _export([_document_turn("synthetic-file-a"), _document_turn("synthetic-file-b")]),
        "synthetic-drive-session",
    )

    resolution = message_owner_resolution(list(session.messages))

    assert len(set(resolution.keys)) == len(resolution.keys)
    assert not resolution.ambiguous_keys


def _drifts_as_unseen(payload: dict[str, Any]) -> bool:
    resolution = SchemaRegistry().resolve_payload(Provider.GEMINI.value, payload)
    assert resolution is not None
    return (
        classify_schema_drift(
            resolution_reason=resolution.reason,
            is_valid=True,
            drift_warnings=(),
        )
        is UNSEEN_SHAPE
    )


# Structural variants a census of the live AI Studio export surface shows in
# circulation: optional envelope siblings (citations, applets), the four Drive
# attachment kinds, thought chunks, and per-chunk delivery metadata. Values are
# synthetic; only the key shape is taken from the wire.
_CURRENT_SHAPE_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("bare envelope", _export([{"role": "user", "text": "synthetic turn"}])),
    (
        "citations sibling",
        _export([{"role": "user", "text": "t"}]) | {"citations": [{"uri": "https://example.invalid/c"}]},
    ),
    ("applets sibling", _export([{"role": "user", "text": "t"}]) | {"applets": []}),
    ("drive document turn", _export([_document_turn("synthetic-file")])),
    ("drive image turn", _export([{"role": "user", "text": "", "driveImage": {"id": "synthetic-image"}}])),
    ("drive audio turn", _export([{"role": "user", "text": "", "driveAudio": {"id": "synthetic-audio"}}])),
    ("drive video turn", _export([{"role": "user", "text": "", "driveVideo": {"id": "synthetic-video"}}])),
    ("thought chunk", _export([{"role": "model", "text": "t", "isThought": True, "thoughtSignatures": [""]}])),
    ("delivery metadata", _export([{"role": "model", "text": "t", "finishReason": "STOP", "tokenCount": 12}])),
    (
        "systemInstruction absent",
        {k: v for k, v in _export([{"role": "user", "text": "t"}]).items() if k != "systemInstruction"},
    ),
]

# The currency lock above is worthless if the package answers "known" to
# everything. Each of these must stay unseen_shape: the committed package
# recognizes the current export through its ``chunkedPrompt`` anchor, so
# removing that anchor -- or presenting a different provider's document -- must
# fall back to the package default.
_STILL_UNSEEN: list[tuple[str, dict[str, Any]]] = [
    ("unshipped shape", {"somethingGoogleHasNotShippedYet": {"nested": [1, 2, 3]}}),
    ("anchor omitted", {k: v for k, v in _export([{"role": "user", "text": "t"}]).items() if k != "chunkedPrompt"}),
    ("anchor not an object", _export([{"role": "user", "text": "t"}]) | {"chunkedPrompt": []}),
    ("empty document", {}),
    ("a ChatGPT export", {"title": "t", "create_time": 1.0, "mapping": {}, "moderation_results": []}),
]


@pytest.mark.parametrize("label,payload", _CURRENT_SHAPE_VARIANTS, ids=[label for label, _ in _CURRENT_SHAPE_VARIANTS])
def test_current_chunked_prompt_shape_resolves_to_a_committed_candidate(label: str, payload: dict[str, Any]) -> None:
    """polylogue-tu1f: the committed package knows the current export shape."""
    resolution = SchemaRegistry().resolve_payload(Provider.GEMINI.value, payload)

    assert resolution is not None
    assert resolution.reason != "package_default"
    assert not _drifts_as_unseen(payload)


@pytest.mark.parametrize("label,payload", _STILL_UNSEEN, ids=[label for label, _ in _STILL_UNSEEN])
def test_a_shape_without_the_anchor_still_classifies_as_unseen(label: str, payload: dict[str, Any]) -> None:
    """The currency lock above must not have been bought by calling everything known."""
    assert _drifts_as_unseen(payload)


def test_zero_message_drive_document_never_becomes_a_session() -> None:
    """polylogue-n2blt: one eligibility predicate, applied on the production route."""
    stub = parse_chunked_prompt(Provider.GEMINI, _export([]), "synthetic-drive-stub")
    real = parse_chunked_prompt(
        Provider.GEMINI,
        _export([{"role": "user", "text": "synthetic turn"}]),
        "synthetic-drive-session",
    )

    kept = require_positive_conversational_evidence(
        [stub, real],
        provider=Provider.GEMINI,
        source_path="/drive-cache/gemini/synthetic.json",
    )

    assert [session.provider_session_id for session in kept] == [real.provider_session_id]
