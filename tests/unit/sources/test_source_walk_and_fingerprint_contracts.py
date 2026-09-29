"""Source discovery, source walk, and derived-identity fingerprint contracts."""

import json
from collections.abc import Iterator
from functools import partial
from io import BytesIO
from pathlib import Path

import pytest

from devtools import schema_closure
from polylogue.core.enums import Provider
from polylogue.sources import origin_specs
from polylogue.sources.decoder_json import hermes_snapshot_envelope
from polylogue.sources.live.discovery import _bounded_source_paths
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.parsers.antigravity import AntigravitySourceInspection, census_source
from polylogue.sources.parsers.local_agent import looks_like_hermes
from polylogue.sources.source_walk import _iter_source_entries
from tests.infra.source_parser_cases import case, ordered_scandir


def test_alias_enumeration_has_one_stable_order(tmp_path: Path) -> None:
    """Assigning alias ownership before sorting changes the cursor sequence."""
    target = tmp_path / "zz-real"
    target.mkdir()
    (target / "one.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "aa-link").symlink_to(target, target_is_directory=True)
    (tmp_path / "bb-link").symlink_to(target, target_is_directory=True)
    source = WatchSource(name="claude-code", root=tmp_path)
    forward = _bounded_source_paths(source, (source,), limit=100, after=None, scandir=ordered_scandir)
    reverse = _bounded_source_paths(
        source, (source,), limit=100, after=None, scandir=partial(ordered_scandir, reverse=True)
    )
    assert forward == reverse
    assert forward
    assert forward[0] == tmp_path / "aa-link/one.jsonl"


@pytest.fixture
def fingerprint_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the production fingerprint route without importing the test module."""
    source = tmp_path / "candidate.py"
    source.write_text(case("fingerprints", "before"), encoding="utf-8")
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(origin_specs, "_LOWERING_FINGERPRINT_PATHS", ("candidate.py",))
    monkeypatch.setattr(origin_specs, "_IMPORT_EDGES", None)
    monkeypatch.setattr(origin_specs, "_IMPORT_EDGES_ADDED", False)
    origin_specs._invalidate_source_signatures()
    origin_specs._fingerprint_sources_cached.cache_clear()
    try:
        yield source
    finally:
        origin_specs._invalidate_source_signatures()
        origin_specs._fingerprint_sources_cached.cache_clear()


def test_changed_source_cannot_poison_fingerprint_memo(
    fingerprint_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old signature could key an AST computed from newer source bytes."""
    original_signature = origin_specs._source_signature
    changed = False

    def observe_then_change(path: Path) -> tuple[str, str, int]:
        nonlocal changed
        signature = original_signature(path)
        if not changed:
            changed = True
            fingerprint_source.write_text(case("fingerprints", "after"), encoding="utf-8")
        return signature

    with monkeypatch.context() as local_patch:
        local_patch.setattr(origin_specs, "_source_signature", observe_then_change)
        with pytest.raises(origin_specs.FingerprintSourceChangedError):
            origin_specs.lowering_fingerprint()
    assert not list((fingerprint_source.parent / ".cache/source-fingerprints").glob("*.txt"))


def test_sql_replacement_operands_move_fingerprint(fingerprint_source: Path) -> None:
    """SQL normalization formerly erased case-sensitive Python call semantics."""
    fingerprint_source.write_text(case("fingerprints", "ddl_before"), encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    before = origin_specs.lowering_fingerprint()
    fingerprint_source.write_text(case("fingerprints", "ddl_after"), encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    assert origin_specs.lowering_fingerprint() != before


def test_streaming_hermes_envelope_keeps_container_presence() -> None:
    """Dropping a container-valued ``platform`` refused a snapshot ``looks_like_hermes`` admits."""
    payload = case("hermes_presence")
    assert looks_like_hermes(payload)
    envelope = hermes_snapshot_envelope(BytesIO(json.dumps(payload).encode()))
    assert envelope is not None
    assert envelope["session_id"] == "hermes-presence"
    assert envelope["platform"] == {}


def test_closure_paths_are_relative_to_callers_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid relative member was formerly tested against the wrong root."""
    member = tmp_path / "member.py"
    member.write_text("pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(origin_specs, "derived_identity_source_closure", lambda: (member,))
    assert origin_specs.in_derived_identity_closure("member.py")
    assert schema_closure.main(["--json", "member.py"]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == [{"path": "member.py", "in_closure": True}]
    with pytest.raises(FileNotFoundError):
        origin_specs.in_derived_identity_closure("missing.py")
    with pytest.raises(SystemExit) as refusal:
        schema_closure.main(["missing.py"])
    assert refusal.value.code == 2


def test_observation_contracts_cover_every_provider_wire() -> None:
    """Choosing only provider_wires[0] omitted the Drive wire contract."""
    contracts = origin_specs.artifact_observation_contracts()
    for spec in origin_specs.ORIGIN_SPECS:
        if not spec.artifact_rules and spec.database_capability is None:
            continue
        for wire in spec.provider_wires or (Provider.UNKNOWN,):
            assert any(row.origin is spec.origin and row.provider is wire for row in contracts)


def test_census_records_directory_link_without_following_cycle(tmp_path: Path) -> None:
    """Following a self-link multiplied candidates before the nonregular check."""
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / "note.md").write_text("neutral note", encoding="utf-8")
    (brain / "cycle").symlink_to(tmp_path, target_is_directory=True)
    census = census_source(tmp_path)
    assert len(census.items) == 2
    link = next(item for item in census.items if item.relative_path == "brain/cycle")
    assert link.inspection is AntigravitySourceInspection.NON_REGULAR


def test_source_walk_follows_a_linked_tree_and_stops_at_a_cycle(tmp_path: Path) -> None:
    """Not following links drops ``linked/``; following without a visited set never ends."""
    export = tmp_path / "export"
    export.mkdir()
    (export / "session.jsonl").write_text("{}\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(export, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    assert _iter_source_entries(root) == [root / "linked/session.jsonl", root / "loop"]
