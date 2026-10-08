from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from devtools import verify_schema_manifest
from polylogue.storage.sqlite import durable_change_train, migration_runner
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.schema_identity import _normalize_schema_sql
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    DurableChangeRider,
    DurableRuntimeConsumer,
    MigrationError,
    declare_durable_change_train,
    durable_change_train_to_payload,
    durable_migration_claim_for_sql,
    durable_migration_claims,
)
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def test_schema_manifest_checks_all_canonical_tiers() -> None:
    assert verify_schema_manifest.main([]) == 0


def test_schema_manifest_rejects_a_target_file_with_schema_drift(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    for tier in ArchiveTier:
        if tier is ArchiveTier.EMBEDDINGS:
            continue
        if tier is ArchiveTier.SOURCE:
            initialize_runtime_source_fixture(root / f"{tier.value}.db")
        else:
            initialize_archive_database(root / f"{tier.value}.db", tier)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DROP INDEX idx_sessions_origin_sort")
        conn.commit()
    assert verify_schema_manifest.main(["--archive-root", str(root)]) == 1


def test_schema_manifest_rejects_missing_archive_tier(tmp_path: Path) -> None:
    """Anti-vacuity: explicit comparison cannot pass by skipping absent tier files."""
    root = tmp_path / "partial"
    root.mkdir()
    assert verify_schema_manifest.main(["--archive-root", str(root)]) == 1


def test_schema_manifest_read_uri_encodes_legal_path_characters(tmp_path: Path) -> None:
    """Anti-vacuity: '?' and '#' in a valid directory name must not alter SQLite URI parsing."""
    root = tmp_path / "archive?copy#1"
    root.mkdir()
    path = root / "source.db"
    initialize_runtime_source_fixture(path)
    result = verify_schema_manifest._check_tier(ArchiveTier.SOURCE, path)
    assert result["ok"] is True


def test_schema_manifest_uses_the_resolved_active_index_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: a stale root index shadow must not be the compared index."""
    active = tmp_path / "generation" / "index.db"
    monkeypatch.setattr(
        verify_schema_manifest.ArchiveLocation,
        "resolve",
        lambda _root: type("Location", (), {"active_index_path": active})(),
    )
    checked: dict[str, Path | None] = {}

    def fake_check_tier(tier: ArchiveTier, path: Path | None) -> dict[str, object]:
        checked[tier.value] = path
        return {"tier": tier.value, "ok": True, "version": 1}

    monkeypatch.setattr(verify_schema_manifest, "_check_tier", fake_check_tier)
    monkeypatch.setattr(verify_schema_manifest, "_provider_named_index_objects", lambda: [])
    assert verify_schema_manifest.main(["--archive-root", str(tmp_path)]) == 0
    assert checked["index"] == active


def test_migration_type_change_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: replacing a migration with a symlink is not an allowed edit."""
    change = verify_schema_manifest._MigrationChange(
        "T",
        "polylogue/storage/sqlite/migrations/source/001_init.sql",
        "polylogue/storage/sqlite/migrations/source/001_init.sql",
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: (change,))
    assert any(
        "required migration was modified" in item
        for item in verify_schema_manifest._migration_integrity_violations("base", ArchiveTier.SOURCE)
    )


def test_schema_manifest_normalization_keeps_escaped_literal_values_exact() -> None:
    """Harmless SQL layout is normalized without rewriting quoted values."""
    compact = "CREATE TABLE sample(value TEXT DEFAULT 'a''b' CHECK(value='A  B'));"
    formatted = """
        CREATE TABLE sample (
            value TEXT DEFAULT 'a''b'
            CHECK (value = 'A  B')
        );
    """
    assert _normalize_schema_sql(compact) == _normalize_schema_sql(formatted)
    assert "'a''b'" in _normalize_schema_sql(compact)
    assert "'A  B'" in _normalize_schema_sql(compact)
    assert _normalize_schema_sql(compact.replace("'A  B'", "'a b'")) != _normalize_schema_sql(compact)


def _schema_state(
    *, source_version: int = 1, source_ddl: str = "source", lineage: str = "polylogue.archive-format.v4"
) -> verify_schema_manifest._SchemaState:
    ddl = {tier: tier.value for tier in ArchiveTier}
    ddl[ArchiveTier.SOURCE] = source_ddl
    versions = dict.fromkeys(ArchiveTier, 1)
    versions[ArchiveTier.SOURCE] = source_version
    return verify_schema_manifest._SchemaState(ddl=ddl, versions=versions, lineage=lineage)


def test_durable_evolution_requires_every_numbered_migration_for_a_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing v1 to v2 without adding 002 must make the gate red."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 2),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert any("source" in violation and "missing added migrations for v2" in violation for violation in violations)


def test_durable_evolution_compares_rendered_ddl_transformations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing rendered DDL through a post-assignment transform must make the gate red."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(
            source_ddl=(
                "CREATE TABLE sample (value TEXT NOT NULL) STRICT;"
                if ref == "base"
                else "CREATE TABLE sample (value TEXT) STRICT;"
            )
        ),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert "source: rendered DDL changed without a schema-version bump" in violations


def test_new_fresh_lineage_allows_floor_ddl_change_without_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new marker fences old v1 files while all durable counters remain one."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(
            source_ddl="before" if ref == "base" else "after",
            lineage="polylogue.archive-format.v3" if ref == "base" else "polylogue.archive-format.v4",
        ),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []


@pytest.mark.parametrize("new_lineage", [True, False], ids=["new-lineage", "same-lineage"])
def test_only_a_new_lineage_folds_one_tiers_chain_into_its_floor(
    new_lineage: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source v6 may return to a v1 floor that absorbs 002-006 only under a new marker lineage.

    Anti-vacuity: without the lineage change the same fold is a backwards move,
    because an archive already at the old floor would otherwise admit the new one.
    """
    base_lineage = "polylogue.archive-format.v5"
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(
            source_version=6 if ref == "base" else 1,
            source_ddl="CREATE TABLE folded (a TEXT) STRICT;",
            lineage=base_lineage if ref == "base" or not new_lineage else "polylogue.archive-format.v6",
        ),
    )
    deleted = tuple(
        verify_schema_manifest._MigrationChange("D", f"polylogue/storage/sqlite/migrations/source/{name}", "")
        for name in ("002_folded.sql", "002.train.json")
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: deleted if tier is ArchiveTier.SOURCE else (),
    )

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    if new_lineage:
        assert violations == []
    else:
        assert violations == ["source: schema version moved backwards from v6 to v1"]


def test_durable_evolution_rejects_removing_an_object_without_a_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a durable object from fresh DDL is an ordinary durable change."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(
            source_ddl=(
                "CREATE TABLE keeper (a TEXT PRIMARY KEY) STRICT;\n"
                "CREATE TABLE raw_membership_writeback_receipts (raw_id TEXT PRIMARY KEY) STRICT;"
            )
            if ref == "base"
            else "CREATE TABLE keeper (a TEXT PRIMARY KEY) STRICT;"
        ),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert "source: rendered DDL changed without a schema-version bump" in violations


def test_durable_evolution_ignores_standalone_ddl_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: a raw-DDL comparison would reject an unchanged schema."""
    before = "CREATE TABLE keeper (a TEXT PRIMARY KEY) STRICT;"
    after = "-- metadata-only note\nCREATE TABLE keeper (a TEXT PRIMARY KEY) STRICT;"
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_ddl=before if ref == "base" else after),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())
    assert verify_schema_manifest._durable_ddl_evolution_violations() == []


def test_durable_evolution_fails_closed_when_schema_manifest_cannot_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: two invalid DDL states must not collapse to equal None manifests."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_ddl="CREATE TABLE sample (value TEXT" if ref == "base" else "nonsense DDL"),
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())
    assert (
        "source: rendered DDL changed without a schema-version bump"
        in verify_schema_manifest._durable_ddl_evolution_violations()
    )


def test_durable_evolution_accepts_a_complete_contiguous_migration_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding 002 and 003 for a v1 to v3 bump must keep the gate green."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 3),
    )
    changes = tuple(
        verify_schema_manifest._MigrationChange(
            "A",
            f"polylogue/storage/sqlite/migrations/source/{version:03d}_step.sql",
            f"polylogue/storage/sqlite/migrations/source/{version:03d}_step.sql",
        )
        for version in (2, 3)
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: changes if tier is ArchiveTier.SOURCE else (),
    )

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []


def test_durable_evolution_fixture_covers_every_durable_tier() -> None:
    """Removing a durable tier from the fixture must make coverage fail."""
    assert set(verify_schema_manifest._DURABLE_TIERS) == {
        ArchiveTier.SOURCE,
        ArchiveTier.USER,
        ArchiveTier.AUDIT,
    }
    state = verify_schema_manifest._current_schema_state()
    assert all(tier in state.ddl and tier in state.versions for tier in verify_schema_manifest._DURABLE_TIERS)


def _isolated_git(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Keep the operator's git configuration and CI base-ref hints out of the gate."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.delenv("POLYLOGUE_SCHEMA_MERGE_BASE", raising=False)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "file.txt").write_text(text, encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-q", "-m", text)
    return _git(repo, "rev-parse", "HEAD")


def _upstream(tmp_path: Path) -> tuple[Path, list[str]]:
    """A default branch with three commits and a feature branch off its tip."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "master")
    master = [_commit(upstream, f"master {index}") for index in range(3)]
    _git(upstream, "switch", "-q", "-c", "feature")
    _commit(upstream, "feature")
    _git(upstream, "switch", "-q", "master")
    return upstream, master


def test_merge_base_is_the_fork_point_of_a_feature_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: comparing against HEAD would return the feature commit."""
    _isolated_git(monkeypatch, tmp_path)
    upstream, master = _upstream(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--branch", "feature", upstream.as_uri(), str(clone))
    monkeypatch.setattr(verify_schema_manifest, "ROOT", clone)

    assert verify_schema_manifest._merge_base() == master[-1]


def test_merge_base_on_the_default_branch_is_the_first_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A default-branch push is its own merge base; the gate compares its parent.

    Anti-vacuity: returning the merge base unchanged compares HEAD with HEAD.
    """
    _isolated_git(monkeypatch, tmp_path)
    upstream, master = _upstream(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", upstream.as_uri(), str(clone))
    monkeypatch.setattr(verify_schema_manifest, "ROOT", clone)

    assert verify_schema_manifest._merge_base() == master[-2]


@pytest.mark.parametrize("branch", ["master", "feature"])
def test_merge_base_refuses_a_shallow_clone_without_comparison_history(
    branch: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A depth-1 checkout has no parent and no fork point to compare against.

    Anti-vacuity: the former fallback to HEAD makes both branches pass.
    """
    _isolated_git(monkeypatch, tmp_path)
    upstream, _master = _upstream(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--single-branch", "--branch", branch, upstream.as_uri(), str(clone))
    assert _git(clone, "rev-parse", "--is-shallow-repository") == "true"
    monkeypatch.setattr(verify_schema_manifest, "ROOT", clone)

    with pytest.raises(RuntimeError, match="no schema comparison base") as refused:
        verify_schema_manifest._merge_base()
    assert "shallow" in str(refused.value)


def test_check_evolution_fails_in_a_shallow_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's exit status, not only the helper, refuses the missing base."""
    _isolated_git(monkeypatch, tmp_path)
    upstream, _master = _upstream(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--depth", "1", upstream.as_uri(), str(clone))
    monkeypatch.setattr(verify_schema_manifest, "ROOT", clone)

    assert verify_schema_manifest.main(["--check-evolution", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["violations"][0].startswith("cannot compare durable schema evolution: no schema comparison base")


_FUTURE_SQL = "-- migration-safety: additive-no-backup\nCREATE TABLE future_items (id INTEGER PRIMARY KEY) STRICT;\n"


def _install_source_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sidecar_sql: str | None) -> None:
    """Ship ``002_future_items.sql`` in a real package, with a sidecar bound to *sidecar_sql*."""
    package = "fixture_gate_migrations_" + "".join(ch if ch.isalnum() else "_" for ch in tmp_path.name)
    tier_package = tmp_path / package / ArchiveTier.SOURCE.value
    tier_package.mkdir(parents=True)
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    name = "002_future_items.sql"
    (tier_package / name).write_text(_FUTURE_SQL, encoding="utf-8")
    if sidecar_sql is not None:
        claim = durable_migration_claim_for_sql(ArchiveTier.SOURCE, name, sidecar_sql, owner_ref="owner:gate")
        rider = DurableChangeRider(
            rider_id="rider:gate",
            owner_ref="owner:gate-rider",
            schema_objects=("table:future_items",),
            runtime_consumers=(
                DurableRuntimeConsumer(
                    "bootstrap",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                    "proof:bootstrap",
                    ("write",),
                ),
                DurableRuntimeConsumer(
                    "daemon-health",
                    "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                    "proof:daemon-health",
                    ("read",),
                ),
            ),
            behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
        )
        declared = declare_durable_change_train(
            train_id="train:source:v2",
            tier=ArchiveTier.SOURCE,
            current_version=1,
            target_version=2,
            slot=2,
            owner_ref="owner:gate",
            migration=claim,
            riders=(rider,),
            declared_at_ms=1,
        )
        (tier_package / "002.train.json").write_text(
            json.dumps(durable_change_train_to_payload(declared)), encoding="utf-8"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(migration_runner, "_migration_package", lambda tier: f"{package}.{tier.value}")
    monkeypatch.setattr(durable_change_train, "_migration_package", lambda tier: f"{package}.{tier.value}")
    # The gate compares the source tier's v1->v2 bump against the added SQL.
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 2),
    )
    added = verify_schema_manifest._MigrationChange(
        "A", f"polylogue/storage/sqlite/migrations/source/{name}", f"polylogue/storage/sqlite/migrations/source/{name}"
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: (added,) if tier is ArchiveTier.SOURCE else (),
    )


@pytest.mark.parametrize(
    ("sidecar_sql", "expected"),
    [
        (None, "missing durable migration train sidecar: source/002.train.json"),
        (_FUTURE_SQL.replace("future_items", "other_items"), "SQL SHA-256 mismatch: 002.train.json"),
    ],
    ids=["missing-sidecar", "sidecar-bound-to-other-sql"],
)
def test_durable_evolution_rejects_an_added_slot_the_runtime_refuses(
    sidecar_sql: str | None, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete SQL chain is not enough: the runtime loader requires a bound sidecar.

    Anti-vacuity: without the gate's admission check the chain alone passes.
    """
    _install_source_migration(tmp_path, monkeypatch, sidecar_sql=sidecar_sql)
    with pytest.raises(MigrationError, match=expected):
        durable_migration_claims(ArchiveTier.SOURCE)

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert len(violations) == 1
    assert violations[0].startswith("source: shipped migrations fail runtime admission:")
    assert expected in violations[0]


def test_durable_evolution_accepts_an_added_slot_with_its_bound_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validator the runtime uses admits the slot, so the gate stays green."""
    _install_source_migration(tmp_path, monkeypatch, sidecar_sql=_FUTURE_SQL)
    assert [claim.slot for claim in durable_migration_claims(ArchiveTier.SOURCE)] == [2]

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []


def test_durable_evolution_rejects_editing_a_shipped_train_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen sidecar is part of the migration contract, like its SQL."""
    change = verify_schema_manifest._MigrationChange(
        "M",
        "polylogue/storage/sqlite/migrations/source/002.train.json",
        "polylogue/storage/sqlite/migrations/source/002.train.json",
    )
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: (change,))

    assert verify_schema_manifest._migration_integrity_violations("base", ArchiveTier.SOURCE) == [
        "source: required migration train sidecar was modified: "
        "polylogue/storage/sqlite/migrations/source/002.train.json"
    ]


@pytest.mark.parametrize("status", ["D", "M"])
def test_durable_evolution_rejects_deleted_or_modified_required_migrations(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting or editing a required migration must remain independently visible."""
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: _schema_state(source_version=1 if ref == "base" else 2),
    )
    change = verify_schema_manifest._MigrationChange(
        status,
        "polylogue/storage/sqlite/migrations/source/002_initial.sql",
        "polylogue/storage/sqlite/migrations/source/002_initial.sql",
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: (change,) if tier is ArchiveTier.SOURCE else (),
    )

    violations = verify_schema_manifest._durable_ddl_evolution_violations()

    assert any("source: required migration was" in violation for violation in violations)


def test_durable_evolution_allows_predecessor_retirement_only_for_the_complete_v1_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reset marker's repository proof retires all old chains together.

    Anti-vacuity: removing the reset exception reports each deleted predecessor
    as a required migration; applying it to just one tier leaves this red.
    """
    old_versions = {ArchiveTier.SOURCE: 47, ArchiveTier.USER: 11, ArchiveTier.AUDIT: 3}
    new_versions = dict.fromkeys(ArchiveTier, 1)
    old_versions_by_tier = dict(new_versions)
    old_versions_by_tier.update(old_versions)
    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(
        verify_schema_manifest,
        "_render_schema_state",
        lambda ref: verify_schema_manifest._SchemaState(
            ddl=dict.fromkeys(ArchiveTier, "ddl"),
            versions=old_versions_by_tier if ref == "base" else new_versions,
        ),
    )
    monkeypatch.setattr(
        verify_schema_manifest,
        "_migration_changes",
        lambda _base, tier: (
            (
                verify_schema_manifest._MigrationChange(
                    "D",
                    f"polylogue/storage/sqlite/migrations/{tier.value}/002_predecessor.sql",
                    "",
                ),
            )
            if tier in old_versions
            else ()
        ),
    )

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []

    new_versions[ArchiveTier.SOURCE] = 2
    violations = verify_schema_manifest._durable_ddl_evolution_violations()
    assert any("source: required migration was deleted" in violation for violation in violations)


def test_a_commit_schema_state_is_rendered_once_per_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: read the base state without consulting the commit-keyed
    cache and the second render extracts the commit again."""
    renders: list[str] = []

    def render(commit: str) -> dict[str, object]:
        renders.append(commit)
        return {"ddl": {"source": "CREATE TABLE t(x)"}, "versions": {"source": 1}, "lineage": "v1"}

    monkeypatch.setattr(verify_schema_manifest, "_SCHEMA_STATE_CACHE", tmp_path / "schema-state")
    monkeypatch.setattr(verify_schema_manifest, "_render_commit_payload", render)
    monkeypatch.setattr(verify_schema_manifest, "_git_text", lambda *_args: "a" * 40 + "\n")

    first = verify_schema_manifest._render_schema_state("origin/master")
    second = verify_schema_manifest._render_schema_state("origin/master")

    assert renders == ["a" * 40]
    assert first == second
    assert first.versions == {ArchiveTier.SOURCE: 1}


def test_an_unchanged_package_compares_against_itself_without_rendering_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: drop the unchanged-package shortcut and the base is rendered."""
    rendered: list[str | None] = []

    def render(ref: str | None) -> object:
        rendered.append(ref)
        return _schema_state(source_version=1)

    monkeypatch.setattr(verify_schema_manifest, "_merge_base", lambda _explicit=None: "base")
    monkeypatch.setattr(verify_schema_manifest, "_package_unchanged_since", lambda _base: True)
    monkeypatch.setattr(verify_schema_manifest, "_render_schema_state", render)
    monkeypatch.setattr(verify_schema_manifest, "_migration_changes", lambda _base, _tier: ())

    assert verify_schema_manifest._durable_ddl_evolution_violations() == []
    assert rendered == [None]
