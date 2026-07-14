"""Hermes archival export contract (fs1.7): checked-in fixture + schema tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONValue
from polylogue.schemas.hermes_export_contract import (
    HERMES_EXPORT_SCHEMA_VERSION,
    HermesArchivalExportV1,
    HermesExportSchemaError,
    parse_hermes_export,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "hermes" / "archival_export" / "v1" / "example-session.json"
)


def _load_fixture() -> dict[str, JSONValue]:
    return cast("dict[str, JSONValue]", json.loads(_FIXTURE_PATH.read_text(encoding="utf-8")))


def test_fixture_exists_and_parses() -> None:
    payload = _load_fixture()
    export = parse_hermes_export(payload)
    assert export.schema_version == HERMES_EXPORT_SCHEMA_VERSION
    assert export.session_id == "hermes-session-example-001"


def test_fixture_covers_every_required_message_state() -> None:
    """2026-07-10 Nous refinement: active/inactive/compacted/rewound/observed."""

    export = parse_hermes_export(_load_fixture())
    counts = export.message_counts_by_state()
    # "inactive" is represented by the state-db "rewound" bucket in this
    # fixture (Hermes's own vocabulary distinguishes rewound-during-session
    # from never-active; both map onto this schema's closed state set).
    assert counts.get("active", 0) >= 2
    assert counts.get("compacted", 0) >= 1
    assert counts.get("rewound", 0) >= 1
    assert counts.get("observed", 0) >= 1


def test_fixture_covers_cost_and_parent_relationship_and_handoff() -> None:
    export = parse_hermes_export(_load_fixture())
    assert export.cost.billing_provider == "anthropic"
    assert export.cost.cost_status is not None
    assert export.parent_relationship == "resume"
    assert export.archive_state == "handoff-complete"
    assert export.finalized is True


def test_round_trip_to_dict_from_dict_is_lossless() -> None:
    export = parse_hermes_export(_load_fixture())
    round_tripped = HermesArchivalExportV1.from_dict(export.to_dict())
    assert round_tripped == export


def test_identical_revision_hash_is_the_dedup_key() -> None:
    """Two exports with the same session_revision_hash are the same revision."""

    payload = _load_fixture()
    first = parse_hermes_export(payload)
    second = parse_hermes_export(payload)
    assert first.session_revision_hash == second.session_revision_hash
    assert first == second

    changed = dict(payload)
    changed["session_revision_hash"] = "sha256:" + "0" * 64
    third = parse_hermes_export(changed)
    assert third.session_revision_hash != first.session_revision_hash


def test_unsupported_schema_version_fails_loudly_not_silently() -> None:
    """A future Hermes-side migration bumping schema_version must not be
    silently coerced into this parser's v1 shape (fs1.7 AC)."""

    payload = dict(_load_fixture())
    payload["schema_version"] = 99
    with pytest.raises(HermesExportSchemaError, match="schema_version"):
        parse_hermes_export(payload)


def test_export_never_carries_a_second_full_copy_via_tool_output_preview() -> None:
    """tool_results carry a bounded preview, never the full duplicated output."""

    export = parse_hermes_export(_load_fixture())
    for message in export.messages:
        for result in message.tool_results:
            if result.output_preview is not None:
                assert len(result.output_preview) < 500
