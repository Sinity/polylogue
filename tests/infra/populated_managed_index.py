"""A real retained transcript whose active managed Index fingerprint has changed."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from tests.infra.empty_managed_index import make_empty_managed_index, mutate_fixture_database
from tests.infra.retained_replay import publish_retained_payload


def make_populated_stale_index(root: Path, source: Path) -> tuple[Path, str, tuple[str, ...]]:
    old = make_empty_managed_index(root, stale=False)
    source.parent.mkdir()
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"retained-reconvergence",'
        b'"timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"neutral-message",'
        b'"role":"user","content":[{"type":"input_text","text":"neutral retained prose"}]}}\n'
    )
    raw_id, session_ids = asyncio.run(
        publish_retained_payload(
            root, provider=Provider.CODEX, payload=source.read_bytes(), source_path=str(source), acquired_at_ms=0
        )
    )
    source.unlink()
    with closing(sqlite3.connect(old)) as conn:
        message_id = str(conn.execute("SELECT message_id FROM messages").fetchone()[0])
    mutate_fixture_database(
        root / "user.db",
        "INSERT INTO assertions(assertion_id,target_ref,kind,body_text,created_at_ms,updated_at_ms) VALUES ('neutral-note',?,'note','neutral note',0,0)",
        ("message:" + message_id,),
    )

    def prepare_audit_reference() -> None:
        from polylogue.operations.audit import AuditRepository
        from polylogue.operations.bindings import runtime_operation_binding
        from polylogue.operations.mutation_actuators import IdentityResetActuator, IdentityResetArgs
        from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor

        binding = runtime_operation_binding(IdentityResetActuator())
        principal = MutationPrincipal(
            "user:neutral",
            frozenset(
                capability for policy in binding.spec.target_authority for capability in policy.required_capabilities
            ),
            "cli",
        )
        executor = OperationExecutor(audit=AuditRepository.for_archive_root(root), archive_root=root)
        executor.prepare_bound_for_archive(
            binding, IdentityResetArgs(root, session_ids, "neutral pending preview"), principal, archive_root=root
        )

    from tests.infra.archive_templates import run_archive_fixture_write

    asyncio.run(run_archive_fixture_write(root, prepare_audit_reference))
    with closing(sqlite3.connect(root / "embeddings.db")) as conn, conn:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding

        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_ids[0],
            origin="codex-cli",
            embedding=[0.25] * 1024,
            model="neutral-model",
            embedded_at_ms=0,
            vector_derivation_hash=b"\0" * 32,
        )
    mutate_fixture_database(old, "UPDATE schema_identity SET identity='prior-runtime' WHERE tier='index'")
    return old, raw_id, session_ids


def logical_rows(path: Path) -> tuple[str, ...]:
    with closing(sqlite3.connect(path)) as conn:
        if path.name == "embeddings.db":
            loaded, error = try_load_sqlite_vec(conn)
            assert loaded, error
        return tuple(conn.iterdump())
