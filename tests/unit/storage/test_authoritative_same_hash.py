"""Authoritative identical content keeps truthful counts and current topology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.topology.edge import HOOK_AUTHORITATIVE_LINK_METHOD
from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.authoritative_replay_payloads import codex_single_message_bytes
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_jsonl import retained_raw_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("next_parent", ("parent-b", "missing-parent"))
@pytest.mark.parametrize("inherits_prefix", (False, True))
async def test_same_hash_authoritative_replay_consumes_changed_hook_parent(
    tmp_path: Path, next_parent: str, inherits_prefix: bool
) -> None:
    root = tmp_path / "archive"

    def acquire() -> dict[str, str]:
        bootstrap_archive_root(root)
        acquired: dict[str, str] = {}
        for native_id, text in (
            ("parent-a", "shared neutral prefix"),
            ("parent-b", "shared neutral prefix"),
            ("child", "shared neutral prefix" if inherits_prefix else "independent neutral child"),
        ):
            payload = codex_single_message_bytes(
                native_id, text, tail_text="neutral child tail" if native_id == "child" and inherits_prefix else None
            )
            digest, _size = BlobStore(root / "blob").write_from_bytes(payload)
            with retained_raw_fixture(
                root=root,
                provider=Provider.CODEX,
                blob_hash=digest,
                source_path=str(root / ".codex" / "sessions" / f"{native_id}.jsonl"),
            ) as (_reader, raw_id):
                acquired[native_id] = raw_id
        return acquired

    acquired = await run_archive_fixture_write(root, acquire)

    def hook(parent: str, order: int) -> None:
        payload: dict[str, object] = {"parent_thread_id": parent, "child_thread_id": "child"}
        path = f"neutral-hooks-{order}.ndjson"
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            archive.write_hook_event(
                provider=Provider.CODEX,
                payload=json.dumps(payload).encode(),
                source_path=path,
                acquired_at_ms=order,
                hook_event=ArchiveHookEvent(
                    hook_event_id=f"neutral-hook:{order}",
                    origin=Origin.CODEX_SESSION,
                    source_path=path,
                    event_type="codex_thread_spawn_edge",
                    payload=payload,
                    observed_at_ms=order,
                    native_id=f"hook-{order}",
                    session_native_id="child",
                ),
            )

    async with prepared_live_convergence_owner(root) as owner:
        (await owner.ingest_retained_raw_ids((acquired["parent-a"], acquired["parent-b"]))).require_complete()
        await run_archive_fixture_write(root, lambda: hook("parent-a", 1))
        first = (await owner.ingest_retained_raw_ids((acquired["child"],))).require_complete()
        assert sum(receipt.written_message_count for receipt in first) == (2 if inherits_prefix else 1)
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            index = archive.index_connection
            assert index is not None
            original = tuple(
                index.execute("SELECT session_id,content_hash FROM sessions WHERE native_id='child'").fetchone()
            )
            assert (
                index.execute(
                    "SELECT dst_native_id FROM session_links WHERE src_session_id=? AND method=?",
                    (original[0], HOOK_AUTHORITATIVE_LINK_METHOD),
                ).fetchone()[0]
                == "parent-a"
            )
        stable = (await owner.ingest_retained_raw_ids((acquired["child"],))).require_complete()
        assert sum(receipt.written_message_count for receipt in stable) == 0
        assert sum(receipt.written_counts.get("skipped_sessions", 0) for receipt in stable) == 1
        await run_archive_fixture_write(root, lambda: hook(next_parent, 2))
        changed = (await owner.ingest_retained_raw_ids((acquired["child"],))).require_complete()
        assert sum(receipt.written_message_count for receipt in changed) == (2 if inherits_prefix else 0)
        assert sum(receipt.written_counts.get("skipped_sessions", 0) for receipt in changed) == (
            0 if inherits_prefix else 1
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert (
            tuple(index.execute("SELECT session_id,content_hash FROM sessions WHERE native_id='child'").fetchone())
            == original
        )
        current = index.execute(
            "SELECT dst_native_id,resolved_dst_session_id,status FROM session_links WHERE src_session_id=? AND method=? AND dst_native_id=?",
            (original[0], HOOK_AUTHORITATIVE_LINK_METHOD, next_parent),
        ).fetchone()
        assert current is not None
        assert current[0] == next_parent
        assert (current[1] is None) is (next_parent == "missing-parent")
        assert index.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (original[0],)).fetchone()[0] == (
            2 if inherits_prefix and next_parent == "missing-parent" else 1
        )
        envelope = read_archive_session_envelope(index, original[0])
        assert ["".join(block.text or "" for block in message.blocks) for message in envelope.messages] == (
            ["shared neutral prefix", "neutral child tail"] if inherits_prefix else ["independent neutral child"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_difference", ("hash", "alias", "parser", "lowering"))
async def test_authoritative_same_hash_skip_preserves_real_replacement_obligations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_difference: str
) -> None:
    from polylogue.sources.assembly_codex import CodexAssemblySpec
    from polylogue.sources.dispatch import parse_payload
    from polylogue.storage.sqlite.archive_tiers import write as session_writer
    from tests.infra.index_writer import write_fixture_index_session

    root = tmp_path / "archive"
    raw_payload = codex_single_message_bytes("current-child", "current neutral content")
    prior_payload = (
        codex_single_message_bytes("current-child", "prior neutral content")
        if prior_difference == "hash"
        else raw_payload
    )

    def acquire(payload: bytes, order: int) -> str:
        bootstrap_archive_root(root)
        if prior_difference == "hash":
            import hashlib

            from polylogue.archive.revision_authority import (
                RawRevisionAuthority,
                RawRevisionEnvelope,
                RawRevisionKind,
            )
            from polylogue.sources.acquisition_boundary import (
                bound_profile_identity,
                bound_source_observation,
                open_bound_path,
            )

            path = root / ".codex" / "sessions" / "current-child.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with open_bound_path(path, None) as original:
                profile = bound_profile_identity(original)
                canonical, observation = bound_source_observation(original)
                captured = original.read()
                assert captured == payload and canonical is not None and observation is not None
                with ArchiveStore.open_existing(root, read_only=False) as archive:
                    raw_id = archive.write_raw_payload(
                        provider=Provider.CODEX,
                        payload=captured,
                        source_path=str(path),
                        canonical_source_path=canonical,
                        captured_profile_key=profile.key if profile else None,
                        native_id="current-child",
                        acquired_at_ms=order,
                        file_mtime_ms=observation[3] // 1_000_000,
                        revision=RawRevisionEnvelope(
                            "codex-session:current-child",
                            RawRevisionKind.FULL,
                            hashlib.sha256(captured).hexdigest(),
                            order - 1,
                            authority=RawRevisionAuthority.BYTE_PROVEN,
                        ),
                    )
                    archive.commit()
                    return raw_id
        digest, _size = BlobStore(root / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=root,
            provider=Provider.CODEX,
            blob_hash=digest,
            acquired_at_ms=order,
            source_path=str(root / ".codex" / "sessions" / "current-child.jsonl"),
        ) as (_reader, raw_id):
            return raw_id

    prior_raw_id = await run_archive_fixture_write(root, lambda: acquire(prior_payload, 1))
    async with prepared_live_convergence_owner(root) as owner:
        baseline = (await owner.ingest_retained_raw_ids((prior_raw_id,))).require_complete()
        assert sum(receipt.written_message_count for receipt in baseline) == 1
    raw_id = (
        await run_archive_fixture_write(root, lambda: acquire(raw_payload, 2))
        if prior_difference == "hash"
        else prior_raw_id
    )

    def seed() -> None:
        prior_payload = (
            codex_single_message_bytes("current-child", "prior neutral content")
            if prior_difference == "hash"
            else raw_payload
        )
        records = [json.loads(line) for line in prior_payload.splitlines()]
        sessions = parse_payload(Provider.CODEX, records, "current-child")
        assert len(sessions) == 1
        prior = CodexAssemblySpec().enrich_session(sessions[0], {})
        if prior_difference == "alias":
            prior = prior.model_copy(update={"provider_session_aliases": ["prior-alias"]})
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            index = archive.index_connection
            assert index is not None
            # The prior session stays bound to the raw its accepted head names;
            # only its stored identity (parser, lowering, aliases) differs.
            write_fixture_index_session(index, prior, archive_root=root, raw_id=prior_raw_id)

    with monkeypatch.context() as patch:
        if prior_difference == "parser":
            patch.setattr(session_writer, "parser_fingerprint_for_origin", lambda _origin: "0" * 64)
        elif prior_difference == "lowering":
            patch.setattr(session_writer, "lowering_fingerprint", lambda: "0" * 64)
        if prior_difference != "hash":
            await run_archive_fixture_write(root, seed)

    async with prepared_live_convergence_owner(root) as owner:
        replacement = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(receipt.written_message_count for receipt in replacement) == 1, replacement
        assert sum(receipt.written_counts.get("skipped_sessions", 0) for receipt in replacement) == 0
        repeated = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(receipt.written_message_count for receipt in repeated) == 0
        assert sum(receipt.written_counts.get("skipped_sessions", 0) for receipt in repeated) == 1
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        claims = {
            row[0]
            for row in index.execute(
                "SELECT provider_value FROM session_identity_claims WHERE identity_namespace='provider-session'"
            )
        }
        assert claims == {"current-child"}
        session = index.execute("SELECT session_id FROM sessions WHERE native_id='current-child'").fetchone()
        assert session is not None
        assert [
            row[0]
            for row in index.execute(
                "SELECT text FROM blocks WHERE session_id=? AND text IS NOT NULL ORDER BY position", (session[0],)
            )
        ] == ["current neutral content"]
