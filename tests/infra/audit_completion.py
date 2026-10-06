"""Neutral begun-plan and Source-command controls for continuity tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path
from typing import Any

from polylogue.storage.sqlite.audit_continuity import (
    EXCISION_SOURCE_COMMIT_KIND,
    AuditMutation,
    CanonicalAuditLiteral,
    prepared_audit_continuity_command,
)


def absent_embeddings_intent() -> dict[str, object]:
    return {"incarnation": None, "namespace": None, "outputs": [], "present": False, "rows": [], "schema_version": None}


def make_source_completion_control(tmp_path: Path) -> tuple[Path, Any, dict[str, Any], Callable[..., Any]]:
    """Begin a real audited Excision before installing its Source command."""
    from polylogue.operations.audit import AuditRepository
    from polylogue.storage.sqlite.audit_leaf import open_verified_sqlite_write_connection
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.excision_execution import begin_excision_control
    from tests.infra.storage_records import SessionBuilder

    with write_lease("test.continuity-completion-begin", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        builder = SessionBuilder(tmp_path / "index.db", "completion").provider("codex").add_message(text="Neutral")
        builder.save()
        started, args = begin_excision_control(tmp_path, builder.native_session_id(), reason="synthetic")
    assert started.operation_id is not None
    with closing(sqlite3.connect(tmp_path / "audit.db")) as audit:
        attempt_id = audit.execute(
            "SELECT attempt_id FROM operation_attempts WHERE operation_id=? AND target_ordinal=0",
            (started.operation_id,),
        ).fetchone()[0]
    counts = {
        key: ordinal
        for ordinal, key in enumerate(
            (
                "source_blob_refs",
                "source_raw_rows",
                "source_raw_existence_changes",
                "source_hook_events",
                "source_fact_rows",
                "source_sidecar_rows",
                "source_container_members",
                "source_container_items",
                "source_materials",
                "source_marker_inputs_pending",
                "source_marker_inputs_accepted",
                "source_publication_reservations",
            )
        )
    }
    payload = {
        "operation_id": started.operation_id,
        "attempt_id": attempt_id,
        "embeddings_intent": absent_embeddings_intent(),
        "plan_hash": started.plan.plan_hash,
        "targets": [
            {
                "session_id": args.session_id,
                "counts": counts,
                "removed_blob_hashes": ["a" * 64],
                "shared_blob_hashes": ["b" * 64],
            }
        ],
    }
    repository = AuditRepository.for_archive_root(tmp_path)

    def install(value: dict[str, object], *, rollback: bool = False) -> Any:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

        def chunks() -> Generator[bytes, None, None]:
            for offset in range(0, len(raw), 7):
                yield raw[offset : offset + 7]

        literal = CanonicalAuditLiteral(len(raw), hashlib.sha256(raw).hexdigest(), chunks)
        mutation = AuditMutation(EXCISION_SOURCE_COMMIT_KIND, f"excision-source:{value['attempt_id']}", 1, literal)
        with write_lease("test.continuity-completion-source", archive_root=tmp_path):
            with open_verified_sqlite_write_connection(tmp_path / "source.db") as source:
                source.execute("BEGIN IMMEDIATE")
                generation, head = source.execute(
                    "SELECT committed_generation,committed_head_sha256 FROM audit_continuity_control"
                ).fetchone()
                prepared = prepared_audit_continuity_command(
                    mutation, prior_generation=generation, prior_head_sha256=head
                )
                command = b"".join(prepared.chunks()).decode()
                source.execute(
                    "UPDATE audit_continuity_control SET pending_mutation_id=?,pending_payload_json=?,"
                    "pending_payload_sha256=?,prepared_at_ms=? WHERE singleton=1",
                    (mutation.mutation_id, command, prepared.sha256, mutation.created_at_ms),
                )
                if rollback:
                    source.rollback()
                else:
                    source.commit()
        return prepared

    return tmp_path, repository, payload, install


def present_embeddings_intent_example() -> dict[str, object]:
    """One neutral retiring vec0 output, retaining every original literal byte."""
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION

    def cell(kind: str, value: bytes) -> dict[str, object]:
        return {"byte_length": len(value), "literal_hex": value.hex(), "storage_class": kind}

    return {
        "incarnation": [(1 << 64) - 1, 4],
        "namespace": [3, 4, 0o100600, None],
        "outputs": [
            {"meta_present": False, "retire": True, "vector_derivation_hash": "a" * 64, "vector_present": True}
        ],
        "present": True,
        "rows": [
            {
                "cells": [
                    cell("text", b"synthetic-session"),
                    cell("text", b"codex-session"),
                    cell("text", b"synthetic-generation"),
                    cell("text", b"synthetic-key"),
                    cell("text", b"a" * 64),
                    cell("text", b"b" * 64),
                    cell("text", b"c" * 64),
                    cell("text", b"completed"),
                    cell("integer", (1).to_bytes(8, "big", signed=True)),
                    cell("integer", (1).to_bytes(8, "big", signed=True)),
                ],
                "row_address": {"physical_rowid": -(1 << 63)},
                "table": "embedding_derivation_state",
            },
            {
                "cells": [
                    cell("text", b"a" * 64),
                    cell("blob", bytes(range(256)) * 16),
                    cell("text", b"synthetic\xff\x00model"),
                ],
                "row_address": {"vector_derivation_hash": "a" * 64},
                "table": "message_embeddings",
            },
        ],
        "schema_version": EMBEDDINGS_SCHEMA_VERSION,
    }
