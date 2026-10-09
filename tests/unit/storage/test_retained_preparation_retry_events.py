"""Retained preparation reports its guarded retries without private operands."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.logging import capture
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
from tests.infra.archive_templates import bootstrap_archive_root
from tests.unit.storage.test_raw_observation_derivation import (
    _fixture_archive,
    _publish_to_valid,
    _run_raw_law,
)


@pytest.mark.parametrize("changed", ["reference", "carried_membership", "parser_operands"])
def test_actual_preparation_retry_reports_stable_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Removing retry emission leaves a successful replay with no reason witness."""

    def exercise(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        private_path = str(tmp_path / "private-operand.jsonl")
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"neutral","timestamp":"2026-06-02T00:00:00Z"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user","content":[{"type":"input_text","text":"neutral prose"}]}}\n',
                source_path=private_path,
                canonical_source_path=private_path,
                acquired_at_ms=0,
            )
        changed_once = False
        original = RawObservationDerivation._compute_prepared
        if changed in {"reference", "carried_membership"}:

            def prepare(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> RawObservationReplacement:
                nonlocal changed_once
                if changed == "carried_membership" and not changed_once:
                    changed_once = True
                    kwargs["carry"].raw_ids = ("another-neutral-key",)
                result = original(self, *args, **kwargs)
                if changed == "reference" and not changed_once:
                    changed_once = True
                    # A real external commit changes the retained observer's
                    # data_version while keeping durable logical contents equal.
                    with sqlite3.connect(tmp_path / "user.db") as conn:
                        conn.execute("PRAGMA user_version=1")
                return result

            monkeypatch.setattr(RawObservationDerivation, "_compute_prepared", prepare)
        else:
            import polylogue.sources.prepared_jsonl as prepared_jsonl

            original_parse = prepared_jsonl.prepare_jsonl_blob

            def parse(*args: Any, **kwargs: Any) -> prepared_jsonl.PreparedJsonl:
                nonlocal changed_once
                result = original_parse(*args, **kwargs)
                if not changed_once:
                    changed_once = True
                    with sqlite3.connect(tmp_path / "source.db") as conn:
                        conn.execute(
                            "UPDATE raw_sessions SET source_path=? WHERE raw_id=?",
                            (private_path + ".changed.jsonl", raw_id),
                        )
                return result

            monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", parse)
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        with capture() as events:
            replacement = adapter.compute(frame, raw_id, replay_current=True)
            assert _publish_to_valid(adapter, frame, replacement)
        retries = [e for e in events if e.get("event") == "storage.raw_observation.preparation_retry"]
        assert changed_once and len(retries) == 1
        event = retries[0]
        assert event["outcome"] == "degraded"
        assert (
            event["reason"]
            == {
                "reference": "reference_seal_stale",
                "carried_membership": "carried_membership_changed",
                "parser_operands": "parser_operands_changed",
            }[changed]
        )
        assert event["error_type"] == (
            "ReferenceSealStaleError" if changed == "reference" else "_CarryInvalidatedError"
        )
        assert event["attempts"] == 1 and event["phase"] == "source_preparation"
        assert event["raws"] == 0
        assert private_path not in json.dumps(event) and "neutral prose" not in json.dumps(event)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    _run_raw_law(tmp_path, exercise)
