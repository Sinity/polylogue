"""Cache upgrades preserve evidence only when its semantics remain sufficient."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONValue
from polylogue.schemas import source_inference as source
from polylogue.schemas.field_stats import detection
from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.source_recipe import SourceEvidenceRecipe


def write_source(root: Path, name: str, *, extra: dict[str, object] | None = None, width: int = 0) -> None:
    records = [
        {
            "type": "user",
            "sessionId": name,
            "message": {"role": "user", "content": "synthetic"},
            **(extra or {}),
            **({f"optional_field_{index}": index} if width else {}),
        }
        for index in range(width or 1)
    ]
    (root / f"{name}.jsonl").write_text("\n".join(json.dumps(record) for record in records))


def run(root: Path, cache: Path) -> source.SourceInferenceResult:
    return source.infer_sources((source.SchemaSourceInput("claude-code", root),), cache_path=cache, max_workers=1)


def write_codex(
    root: Path,
    name: str,
    session_id: str,
    timestamp: str | None,
    *,
    nonce: str = "",
    marker: str | None = None,
    legacy: bool = False,
) -> Path:
    header = {"id": session_id, "nonce": nonce, **({"timestamp": timestamp} if timestamp is not None else {})}
    if legacy:
        records: list[object] = [
            {**header, **({marker: True} if marker else {})},
            {
                "type": "message",
                "id": f"{name}-message",
                "role": "user",
                "content": [{"type": "input_text", "text": "synthetic"}],
            },
        ]
    else:
        records = [
            {
                "type": "session_meta",
                "payload": {"id": session_id},
                "nonce": nonce,
                **({"timestamp": timestamp} if timestamp is not None else {}),
                **({marker: True} if marker else {}),
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{name}-message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "synthetic"}],
                },
            },
        ]
    path = root / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def run_codex(root: Path, cache: Path) -> source.SourceInferenceResult:
    return source.infer_sources((source.SchemaSourceInput("codex", root),), cache_path=cache, max_workers=1)


def chatgpt_conversation(session_id: str, updated: float, *, marker: str | None = None) -> dict[str, object]:
    conversation: dict[str, object] = {
        "id": session_id,
        "conversation_id": session_id,
        "update_time": updated,
        "mapping": {
            "root": {"id": "root", "parent": None, "message": None},
            "user": {
                "id": "user",
                "parent": "root",
                "message": {
                    "id": "user",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["synthetic"]},
                    "create_time": 1.0,
                },
            },
        },
    }
    if marker is not None:
        conversation[marker] = True
    return conversation


def run_chatgpt(root: Path, cache: Path) -> source.SourceInferenceResult:
    return source.infer_sources((source.SchemaSourceInput("chatgpt", root),), cache_path=cache, max_workers=1)


def result_evidence(result: source.SourceInferenceResult) -> SchemaEvidence:
    return merge_evidence(SchemaEvidence.from_json(row) for rows in result.evidence_by_element.values() for row in rows)


def old_declared_update_key(_provider: Provider, _payload: JSONValue) -> tuple[int, str] | None:
    return None


def old_metadata_refresh(
    _cache: source.SourceContributionCache, descriptor: source._CandidateDescriptor
) -> source._CandidateDescriptor:
    return descriptor


@pytest.fixture
def local_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Policy-change tests need the real collector to see each run's selected parameters.
    monkeypatch.setattr(source, "ProcessPoolExecutor", ThreadPoolExecutor)


def test_identity_upgrade_replaces_legacy_codex_path_fallback_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """Old path-keyed Codex rows double-count duplicate legacy exports.

    Anti-vacuity: permit the old fallback contribution and the warm upgrade
    retains two current sources instead of regrouping the exports by their
    declared legacy session id.
    """
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "first", "shared", "2026-01-01T00:00:00Z", legacy=True)
    write_codex(root, "second", "shared", "2026-01-02T00:00:00Z", legacy=True)
    cache = tmp_path / "cache.sqlite"

    def old_native_source_id(_provider: Provider, _payload: JSONValue, fallback: str, *, source_path: Path) -> str:
        del source_path
        return fallback

    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_native_source_id", old_native_source_id)
        old = run_codex(root, cache)
    upgraded = run_codex(root, cache)
    fresh = run_codex(root, tmp_path / "fresh.sqlite")
    assert old.evidence_by_element != fresh.evidence_by_element
    assert upgraded.evidence_by_element == fresh.evidence_by_element


def test_modern_codex_identity_reuses_warm_cache(tmp_path: Path, local_workers: None) -> None:
    """Native Codex cache rows remain reusable across the identity upgrade."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "session", "modern", "2026-01-01T00:00:00Z")
    cache = tmp_path / "cache.sqlite"
    cold = run_codex(root, cache)
    warm = run_codex(root, cache)
    assert warm.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert warm.evidence_by_element == cold.evidence_by_element


def test_headerless_claude_code_recollects_legacy_path_fallback_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """Anti-vacuity: a legacy path fallback must not supply current revision-keyed Claude Code evidence."""
    root = tmp_path / "inputs"
    root.mkdir()
    source_file = root / "headerless.jsonl"
    source_file.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "synthetic"}}), encoding="utf-8"
    )
    cache_path = tmp_path / "cache.sqlite"
    run(root, cache_path)
    candidate = source.inventory_schema_sources((source.SchemaSourceInput("claude-code", root),))[0]
    digest, byte_count = source._stable_file_digest(source_file)
    recipe = SourceEvidenceRecipe()
    manifest_key = source._cache_key(
        candidate,
        digest,
        dynamic_paths_by_element=None,
        recipe_fingerprint=recipe.fingerprint("structure"),
    )
    with source.SourceContributionCache(cache_path) as cache:
        manifest = cache.get(manifest_key)
        assert manifest is not None
        descriptor = source._cached_descriptors(manifest.evidence)[0]
        legacy_id = hash_payload({"source": candidate.logical_source_id})
        legacy_descriptor = replace(descriptor, logical_source_id=legacy_id)
        cache.put(replace(manifest, evidence=source._serialize_descriptors((legacy_descriptor,))))
        contribution_key = source._cache_key(
            candidate,
            descriptor.revision_sha256,
            dynamic_paths_by_element=None,
            recipe_fingerprint=recipe.fingerprint("structure"),
            logical_source_id=descriptor.logical_source_id,
        )
        contribution = cache.get(contribution_key)
        assert contribution is not None
        address = contribution.metadata.get("address")
        assert isinstance(address, dict)
        legacy_contribution = replace(
            next(source._cached_contributions(contribution.evidence)), logical_source_id=legacy_id
        )
        cache.put(
            source.CachedContribution(
                cache_key=source._cache_key(
                    candidate,
                    descriptor.revision_sha256,
                    dynamic_paths_by_element=None,
                    recipe_fingerprint=recipe.fingerprint("structure"),
                    logical_source_id=legacy_id,
                ),
                evidence=source._serialize_contributions((legacy_contribution,)),
                input_bytes=byte_count,
                record_count=legacy_contribution.record_count,
                metadata={
                    **contribution.metadata,
                    "address": {
                        **address,
                        "source_context": hash_payload({"logical_source_id": legacy_id}),
                    },
                },
            )
        )

    with monkeypatch.context() as bypass:
        bypass.setattr(source, "_old_path_fallback", lambda *_args: False)
        with pytest.raises(source.SourceInferenceError, match="final source identities changed"):
            run(root, cache_path)
    warm = run(root, cache_path)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert warm.cache_phase_misses == {"structure": 1}
    assert warm.evidence_by_element == fresh.evidence_by_element


def reject_source_recollection(*_args: object, **_kwargs: object) -> source._CollectedCandidate:
    pytest.fail("warm evidence must not recollect source records")


@pytest.mark.parametrize("legacy", [False, True])
def test_codex_metadata_refresh_orders_reserialized_legacy_exports_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None, legacy: bool
) -> None:
    """Timestamp-free old rows must not use byte hashes to choose a stale export.

    Anti-vacuity: skip descriptor refresh and the deliberately larger old
    digest wins even though the later export declares the newer timestamp.
    """
    root = tmp_path / "inputs"
    root.mkdir()
    for nonce in range(1_000):
        older = write_codex(
            root, "older", "shared", "2026-01-01T00:00:00Z", nonce=f"older-{nonce}", marker="older_only", legacy=legacy
        )
        newer = write_codex(
            root, "newer", "shared", "2026-02-01T00:00:00Z", nonce=f"newer-{nonce}", marker="newer_only", legacy=legacy
        )
        if sha256(older.read_bytes()).hexdigest() > sha256(newer.read_bytes()).hexdigest():
            break
    else:
        pytest.fail("could not construct the digest-order control")
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_declared_update_key", old_declared_update_key)
        old_code.setattr(source, "_refreshed_codex_descriptor", old_metadata_refresh)
        run_codex(root, cache)
    refreshes = 0

    original_refresh = source._refreshed_codex_descriptor

    def count_refresh(
        metadata_cache: source.SourceContributionCache, descriptor: source._CandidateDescriptor
    ) -> source._CandidateDescriptor:
        nonlocal refreshes
        refreshes += 1
        with monkeypatch.context() as metadata_scan:

            def reject_evidence_reduction(*_args: object, **_kwargs: object) -> None:
                pytest.fail("ordering recovery must not reduce schema evidence")

            metadata_scan.setattr(
                "polylogue.schemas.generation.evidence.collect_source_evidence", reject_evidence_reduction
            )
            return original_refresh(metadata_cache, descriptor)

    monkeypatch.setattr(source, "_refreshed_codex_descriptor", count_refresh)
    refreshed = run_codex(root, cache)
    assert refreshes == 2
    current_properties = result_evidence(refreshed).current_structure.get("properties")
    assert isinstance(current_properties, dict)
    assert "newer_only" in current_properties
    assert "older_only" not in current_properties
    monkeypatch.setattr(source, "_collect_candidate", reject_source_recollection)
    warm = run_codex(root, cache)
    assert warm.evidence_by_element == refreshed.evidence_by_element


def test_codex_metadata_refresh_caches_a_computed_null_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """A completed refresh with no timestamp is evidence, not a cache miss forever."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "first", "shared", None)
    write_codex(root, "second", "shared", None)
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_declared_update_key", old_declared_update_key)
        old_code.setattr(source, "_refreshed_codex_descriptor", old_metadata_refresh)
        run_codex(root, cache)
    refreshes = 0

    original_refresh = source._refreshed_codex_descriptor

    def count_refresh(
        metadata_cache: source.SourceContributionCache, descriptor: source._CandidateDescriptor
    ) -> source._CandidateDescriptor:
        nonlocal refreshes
        refreshes += 1
        return original_refresh(metadata_cache, descriptor)

    monkeypatch.setattr(source, "_refreshed_codex_descriptor", count_refresh)
    run_codex(root, cache)
    assert refreshes == 2
    monkeypatch.setattr(source, "_collect_candidate", reject_source_recollection)
    run_codex(root, cache)


def test_codex_metadata_refresh_only_recollects_missing_maximum_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """Known ordering metadata remains reusable while a competing legacy row is refreshed."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "older", "shared", "2026-01-01T00:00:00Z")
    write_codex(root, "newer", "shared", "2026-02-01T00:00:00Z")
    cache = tmp_path / "cache.sqlite"
    declared_update_key = source._declared_update_key

    def old_update_key(provider: Provider, payload: JSONValue) -> tuple[int, str] | None:
        if isinstance(payload, dict) and payload.get("timestamp") == "2026-02-01T00:00:00Z":
            return declared_update_key(provider, payload)
        return None

    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_declared_update_key", old_update_key)
        old_code.setattr(source, "_refreshed_codex_descriptor", old_metadata_refresh)
        run_codex(root, cache)
    refreshed_paths: list[Path] = []
    original_refresh = source._refreshed_codex_descriptor

    def count_refresh(
        metadata_cache: source.SourceContributionCache, descriptor: source._CandidateDescriptor
    ) -> source._CandidateDescriptor:
        refreshed_paths.append(descriptor.candidate.path)
        return original_refresh(metadata_cache, descriptor)

    monkeypatch.setattr(source, "_refreshed_codex_descriptor", count_refresh)
    run_codex(root, cache)
    assert refreshed_paths == [root / "older.jsonl"]


def test_codex_metadata_refresh_refuses_a_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """Refresh metadata may not be attached to bytes that changed after collection."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "first", "shared", "2026-01-01T00:00:00Z")
    changed = write_codex(root, "second", "shared", "2026-02-01T00:00:00Z")
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_declared_update_key", old_declared_update_key)
        old_code.setattr(source, "_refreshed_codex_descriptor", old_metadata_refresh)
        run_codex(root, cache)
    original_collect = source._collect_candidate

    def mutate_after_collect(
        candidate: source._SourceCandidate,
        dynamic_paths_by_element: dict[str, tuple[str, ...]] | None = None,
        *,
        include_statistics: bool = True,
        metadata_only: bool = False,
        spool_path: Path | None = None,
    ) -> source._CollectedCandidate:
        collected = original_collect(
            candidate,
            dynamic_paths_by_element,
            include_statistics=include_statistics,
            metadata_only=metadata_only,
            spool_path=spool_path,
        )
        if candidate.path == changed:
            changed.write_text(changed.read_text() + "\n")
        return collected

    monkeypatch.setattr(source, "_collect_candidate", mutate_after_collect)
    with pytest.raises(source.SourceInferenceError, match="metadata changed"):
        run_codex(root, cache)


def test_identity_only_upgrade_reuses_collapsed_native_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """An identity-only upgrade does not require cardinality recovery."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "wide", width=160)
    cache = tmp_path / "cache.sqlite"
    monkeypatch.setattr(detection, "_HIGH_CARDINALITY_KEY_THRESHOLD", 128)
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        run(root, cache)
    upgraded = run(root, cache)
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}


def test_failed_codex_metadata_refresh_does_not_cache_completed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected metadata scan must not turn an incompatible row into a reusable completion."""
    path = tmp_path / "rollout.jsonl"
    path.write_text("{}")
    digest, byte_count = source._stable_file_digest(path)
    candidate = source._SourceCandidate("codex", tmp_path, path, "candidate")
    descriptor = source._CandidateDescriptor(
        candidate,
        digest,
        byte_count,
        (source._ContributionDescriptor("codex:shared", digest, 1, None),),
        (),
        False,
    )
    mismatch = source._CollectedCandidate(
        candidate,
        source.SourceRevision("codex", path, "candidate", digest, byte_count),
        source.SourceTerminal("included", byte_count, 2),
        (source._SourceContribution("codex:shared", digest, {}, 2, None),),
    )
    monkeypatch.setattr(source, "_collect_candidate", lambda *_args, **_kwargs: mismatch)
    cache_path = tmp_path / "cache.sqlite"
    with source.SourceContributionCache(cache_path) as cache:
        with pytest.raises(source.SourceInferenceError, match="source, revision, or record count"):
            source._refreshed_codex_descriptor(cache, descriptor)
        assert cache.get(source._descriptor_metadata_cache_key(candidate, digest)) is None


def test_statistics_miss_preserves_codex_timestamp_discovered_after_old_preliminary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """A statistics retry may update ordering metadata without failing descriptor validation."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_codex(root, "older", "shared", "2026-01-01T00:00:00Z", marker="older_only")
    write_codex(root, "newer", "shared", "2026-02-01T00:00:00Z", marker="newer_only")
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        old_code.setattr(source, "_declared_update_key", old_declared_update_key)
        old_code.setattr(source, "_refreshed_codex_descriptor", old_metadata_refresh)
        run_codex(root, cache)
    monkeypatch.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(statistics_revision=2))
    upgraded = run_codex(root, cache)
    assert upgraded.cache_phase_misses == {"statistics": 2}
    current_properties = result_evidence(upgraded).current_structure.get("properties")
    assert isinstance(current_properties, dict)
    assert "newer_only" in current_properties
    assert "older_only" not in current_properties


def test_legacy_codex_zip_cache_is_recollected_on_identity_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """An old ZIP has unresolved member identity, so its cache contribution is never upgraded in place."""
    root = tmp_path / "inputs"
    root.mkdir()
    archive = root / "rollout.zip"
    member = write_codex(root, "rollout", "shared", "2026-01-01T00:00:00Z")
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("rollout.jsonl", member.read_text())
    member.unlink()
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(identity_revision=1))
        run_codex(root, cache)
    upgraded = run_codex(root, cache)
    fresh = run_codex(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element


def test_zip_members_with_one_chatgpt_identity_keep_current_and_historical_revisions(
    tmp_path: Path, local_workers: None
) -> None:
    """ZIP members with one native id must not merge before revision selection.

    Anti-vacuity: a spool keyed only by native id combines both member records,
    so the newer payload is counted twice and the older revision disappears.
    """
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("older.json", json.dumps(chatgpt_conversation("shared", 1, marker="older_only")))
        zip_file.writestr("newer.json", json.dumps(chatgpt_conversation("shared", 2, marker="newer_only")))
    result = run_chatgpt(archive, tmp_path / "cache.sqlite")
    evidence = result_evidence(result)
    assert result.record_count == 1
    assert result.included_native_source_revision_count == 2
    current_properties = evidence.current_structure.get("properties")
    historical_properties = evidence.historical_structure.get("properties")
    assert isinstance(current_properties, dict)
    assert isinstance(historical_properties, dict)
    assert "newer_only" in current_properties
    assert "older_only" not in current_properties
    assert "older_only" in historical_properties


def test_identical_zip_members_are_deduplicated_by_native_revision(tmp_path: Path, local_workers: None) -> None:
    """A member partition preserves distinct entries without defeating revision deduplication."""
    archive = tmp_path / "captures.zip"
    payload = json.dumps(chatgpt_conversation("shared", 1))
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("one.json", payload)
        zip_file.writestr("two.json", payload)
    result = run_chatgpt(archive, tmp_path / "cache.sqlite")
    assert result.record_count == 1
    assert result.included_native_source_revision_count == 1


def test_zip_and_loose_chatgpt_member_have_equal_evidence(tmp_path: Path, local_workers: None) -> None:
    """Member partitioning changes spool ownership, not the resulting source evidence."""
    payload = json.dumps(chatgpt_conversation("shared", 1))
    loose = tmp_path / "loose.json"
    loose.write_text(payload)
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("member.json", payload)
    loose_result = run_chatgpt(loose, tmp_path / "loose-cache.sqlite")
    zip_result = run_chatgpt(archive, tmp_path / "zip-cache.sqlite")
    assert zip_result.evidence_by_element == loose_result.evidence_by_element


def test_zip_and_loose_chatgpt_revisions_have_equal_current_and_historical_evidence(
    tmp_path: Path, local_workers: None
) -> None:
    """A ZIP member partition preserves the same revision selection as loose source files."""
    older = json.dumps(chatgpt_conversation("shared", 1, marker="older_only"))
    newer = json.dumps(chatgpt_conversation("shared", 2, marker="newer_only"))
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "older.json").write_text(older)
    (loose / "newer.json").write_text(newer)
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("older.json", older)
        zip_file.writestr("newer.json", newer)
    loose_result = run_chatgpt(loose, tmp_path / "loose-cache.sqlite")
    zip_result = run_chatgpt(archive, tmp_path / "zip-cache.sqlite")
    assert zip_result.evidence_by_element == loose_result.evidence_by_element


def test_singleton_zip_cache_row_reuses_across_member_partition_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """One reduced record cannot have merged two ZIP members."""
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("member.json", json.dumps(chatgpt_conversation("shared", 1)))
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(zip_member_revision=1))
        run_chatgpt(archive, cache)
    upgraded = run_chatgpt(archive, cache)
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}


def test_multi_record_legacy_zip_cache_row_is_recollected_for_member_partitioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """A multi-record legacy ZIP row may merge members and must be replaced."""
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("first.json", json.dumps(chatgpt_conversation("shared", 1)))
        zip_file.writestr("second.json", json.dumps(chatgpt_conversation("shared", 2)))
    cache = tmp_path / "cache.sqlite"
    collect_payload_evidence = source._collect_payload_evidence

    def old_zip_collect(
        candidate: source._SourceCandidate,
        revision: source.SourceRevision,
        payloads: Iterable[JSONValue | source._SizedPayload],
        *,
        dynamic_paths_by_element: dict[str, tuple[str, ...]],
        include_statistics: bool = True,
        metadata_only: bool = False,
        chunk_record_limit: int = 32,
        spool_path: Path | None = None,
        spool_partition: str = "",
        replay_payloads: Callable[[], Iterable[JSONValue | source._SizedPayload]] | None = None,
    ) -> tuple[tuple[source._SourceContribution, ...], int, tuple[str, ...], bool]:
        del spool_partition
        return collect_payload_evidence(
            candidate,
            revision,
            payloads,
            dynamic_paths_by_element=dynamic_paths_by_element,
            include_statistics=include_statistics,
            metadata_only=metadata_only,
            chunk_record_limit=chunk_record_limit,
            spool_path=spool_path,
            spool_partition="",
            replay_payloads=replay_payloads,
        )

    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(zip_member_revision=1))
        old_code.setattr(source, "_collect_payload_evidence", old_zip_collect)
        run_chatgpt(archive, cache)
    upgraded = run_chatgpt(archive, cache)
    fresh = run_chatgpt(archive, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element


def test_flatfile_equal_count_cache_row_reuses_across_zip_member_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """The ZIP revision does not invalidate an equal-count loose-file contribution."""
    loose = tmp_path / "loose.json"
    loose.write_text(json.dumps(chatgpt_conversation("shared", 1)))
    cache = tmp_path / "cache.sqlite"
    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(zip_member_revision=1))
        run_chatgpt(loose, cache)
    upgraded = run_chatgpt(loose, cache)
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}


def test_old_zip_count_poison_becomes_a_flatfile_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_workers: None
) -> None:
    """A merged old ZIP contribution must not make a matching loose member fail cache validation."""
    payload = json.dumps(chatgpt_conversation("shared", 1))
    archive = tmp_path / "captures.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("first.json", payload)
        zip_file.writestr("second.json", json.dumps(chatgpt_conversation("shared", 2)))
    cache = tmp_path / "cache.sqlite"
    collect_payload_evidence = source._collect_payload_evidence

    def old_zip_collect(
        candidate: source._SourceCandidate,
        revision: source.SourceRevision,
        payloads: Iterable[JSONValue | source._SizedPayload],
        *,
        dynamic_paths_by_element: dict[str, tuple[str, ...]],
        include_statistics: bool = True,
        metadata_only: bool = False,
        chunk_record_limit: int = 32,
        spool_path: Path | None = None,
        spool_partition: str = "",
        replay_payloads: Callable[[], Iterable[JSONValue | source._SizedPayload]] | None = None,
    ) -> tuple[tuple[source._SourceContribution, ...], int, tuple[str, ...], bool]:
        del spool_partition
        return collect_payload_evidence(
            candidate,
            revision,
            payloads,
            dynamic_paths_by_element=dynamic_paths_by_element,
            include_statistics=include_statistics,
            metadata_only=metadata_only,
            chunk_record_limit=chunk_record_limit,
            spool_path=spool_path,
            spool_partition="",
            replay_payloads=replay_payloads,
        )

    with monkeypatch.context() as old_code:
        old_code.setattr(source, "SourceEvidenceRecipe", lambda: SourceEvidenceRecipe(zip_member_revision=1))
        old_code.setattr(source, "_collect_payload_evidence", old_zip_collect)
        run_chatgpt(archive, cache)
    loose = tmp_path / "loose.json"
    loose.write_text(payload)
    upgraded = run_chatgpt(loose, cache)
    fresh = run_chatgpt(loose, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element


def test_implementation_provenance_does_not_invalidate_semantically_unchanged_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
) -> None:
    """Tying the cache key to reporting/import changes repeats both source passes."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "session")
    monkeypatch.setattr(source, "_source_recipe_fingerprint", lambda: "a" * 64)
    cold = run(root, tmp_path / "cache.sqlite")
    monkeypatch.setattr(source, "_source_recipe_fingerprint", lambda: "b" * 64)
    warm = run(root, tmp_path / "cache.sqlite")
    assert warm.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert warm.cache_misses == 0
    assert warm.evidence_by_element == cold.evidence_by_element
    assert warm.input_manifest_digest == cold.input_manifest_digest
    assert warm.recipe["implementation_fingerprint"] != cold.recipe["implementation_fingerprint"]


@pytest.mark.parametrize(
    ("recipe", "hits", "misses"),
    [
        (SourceEvidenceRecipe(statistics_revision=2), {"structure": 1}, {"statistics": 1}),
        (SourceEvidenceRecipe(structure_revision=2), {"statistics": 1}, {"structure": 1}),
        (SourceEvidenceRecipe(admission_revision=2), {}, {"structure": 1, "statistics": 1}),
        (SourceEvidenceRecipe(identity_revision=3), {}, {"structure": 1, "statistics": 1}),
    ],
)
def test_semantic_revision_invalidates_only_the_dependent_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
    recipe: SourceEvidenceRecipe,
    hits: dict[str, int],
    misses: dict[str, int],
) -> None:
    """A statistics change must neither reuse old counters nor discard structural evidence."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "session")
    cache = tmp_path / "cache.sqlite"
    cold = run(root, cache)
    monkeypatch.setattr(source, "SourceEvidenceRecipe", lambda: recipe)
    upgraded = run(root, cache)
    assert upgraded.cache_phase_hits == hits
    assert upgraded.cache_phase_misses == misses
    assert upgraded.evidence_by_element == cold.evidence_by_element


def test_key_limit_upgrade_recovers_collapsed_fields_and_reuses_other_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
) -> None:
    """A wildcard summary cannot supply the erased field names or separate field counters."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "wide", width=160)
    write_source(root, "narrow")
    cache = tmp_path / "cache.sqlite"
    monkeypatch.setattr(detection, "_HIGH_CARDINALITY_KEY_THRESHOLD", 128)
    old = run(root, cache)
    monkeypatch.setattr(detection, "_HIGH_CARDINALITY_KEY_THRESHOLD", 256)
    upgraded = run(root, cache)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_hits == {"structure": 1}
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 2}
    assert upgraded.evidence_by_element == fresh.evidence_by_element
    assert upgraded.evidence_by_element != old.evidence_by_element
    assert "optional_field_159" in json.dumps(upgraded.evidence_by_element)


def test_new_dynamic_path_only_reprocesses_sources_containing_that_path(
    tmp_path: Path,
    local_workers: None,
) -> None:
    """A global normalization-map key must not invalidate unrelated source contributions."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "existing")
    cache = tmp_path / "cache.sqlite"
    run(root, cache)
    write_source(root, "new", extra={"extra": {"question?": {"answer": "synthetic"}}})
    upgraded = run(root, cache)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element


@pytest.mark.parametrize("name", ["a.b", "a[*]", "*"])
def test_literal_path_punctuation_invalidates_affected_statistics(
    tmp_path: Path,
    local_workers: None,
    name: str,
) -> None:
    """Parsing a field name as a path incorrectly reuses its pre-normalization counters."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "existing", extra={name: {"field": 1}})
    cache = tmp_path / "cache.sqlite"
    run(root, cache)
    write_source(root, "new", extra={name: {"question?": 2}})
    upgraded = run(root, cache)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_misses["statistics"] == 2
    assert upgraded.cache_phase_hits.get("statistics", 0) == 0
    assert upgraded.evidence_by_element == fresh.evidence_by_element


def test_normalization_below_a_collapsed_ancestor_is_relevant() -> None:
    """A selected parent changes the collector's path for each literal child."""
    from polylogue.schemas.generation.dynamic_keys import observed_structure_schema
    from polylogue.schemas.source_recipe import relevant_normalization_paths

    structure = observed_structure_schema({"map": {"literal": {"nested": {"field": 1}}}})
    paths = ("$.map", "$.map.*.nested")
    assert relevant_normalization_paths(paths, structure) == paths
