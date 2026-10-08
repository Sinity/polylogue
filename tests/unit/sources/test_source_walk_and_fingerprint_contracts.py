"""Source discovery, source walk, and derived-identity fingerprint contracts."""

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from threading import Barrier

import pytest

from devtools import schema_closure
from polylogue.core.enums import Provider
from polylogue.sources import origin_specs
from polylogue.sources.decoder_json import hermes_snapshot_envelope
from polylogue.sources.live.discovery import _bounded_source_paths
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.parsers.antigravity import AntigravitySourceInspection, census_source
from polylogue.sources.parsers.local_agent import looks_like_hermes
from polylogue.sources.source_walk import layout_source_paths
from tests.infra.source_parser_cases import case, ordered_scandir


def test_alias_enumeration_has_one_stable_order(tmp_path: Path) -> None:
    """Assigning alias ownership before sorting changes the cursor sequence."""
    target = tmp_path / "-zz-real"
    target.mkdir()
    (target / "one.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "-aa-link").symlink_to(target, target_is_directory=True)
    (tmp_path / "-bb-link").symlink_to(target, target_is_directory=True)
    source = WatchSource(name="claude-code", root=tmp_path)
    forward = _bounded_source_paths(source, (source,), limit=100, after=None, scandir=ordered_scandir)
    reverse = _bounded_source_paths(
        source, (source,), limit=100, after=None, scandir=partial(ordered_scandir, reverse=True)
    )
    assert forward == reverse
    assert forward
    assert forward[0] == tmp_path / "-aa-link/one.jsonl"


@pytest.fixture
def fingerprint_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the production fingerprint route without importing the test module."""
    source = tmp_path / "candidate.py"
    source.write_text(case("fingerprints", "before"), encoding="utf-8")
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(origin_specs, "_LOWERING_FINGERPRINT_PATHS", ("candidate.py",))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
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
    memo_root = origin_specs._source_memo_root()
    assert memo_root is not None
    assert not list(memo_root.glob("fingerprint-*.txt"))
    assert not list(memo_root.glob("edges-*.json"))


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


def test_census_never_follows_a_directory_link_outside_the_layout(tmp_path: Path) -> None:
    """Following a self-link multiplied candidates; the layout never reaches it."""
    brain = tmp_path / "brain" / "c1"
    brain.mkdir(parents=True)
    (brain / "note.md").write_text("neutral note", encoding="utf-8")
    (brain / "cycle").symlink_to(tmp_path, target_is_directory=True)
    census = census_source(tmp_path)
    assert [item.relative_path for item in census.items] == ["brain/c1/note.md"]
    assert census.items[0].inspection is AntigravitySourceInspection.REGULAR


def test_source_walk_follows_a_linked_tree_and_stops_at_a_cycle(tmp_path: Path) -> None:
    """Not following links drops ``linked/``; following without a visited set never ends.

    The target stays inside the root (in a directory the walk itself never
    enters), the containment discovery requires of every followed link.
    """
    root = tmp_path / "root"
    export = root / ".export"
    export.mkdir(parents=True)
    (export / "session.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "linked").symlink_to(export, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    assert layout_source_paths("inbox", root) == [root / "linked/session.jsonl"]


def test_equal_checkouts_share_fingerprint_and_lexical_edge_memos(
    fingerprint_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute checkout keys or missing edge reuse make the second read reparse."""
    first = origin_specs.lowering_fingerprint()
    second_root = tmp_path / "second"
    second_root.mkdir()
    (second_root / fingerprint_source.name).write_bytes(fingerprint_source.read_bytes())
    origin_specs._invalidate_source_signatures()
    origin_specs._fingerprint_sources_cached.cache_clear()
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", second_root)

    def refuse(*_args: object) -> str:
        raise AssertionError("reparsed identical checkout bytes")

    with monkeypatch.context() as guarded:
        guarded.setattr(origin_specs, "_fingerprint_sources_compute", refuse)
        guarded.setattr(origin_specs, "_import_bases", refuse)
        assert origin_specs.lowering_fingerprint() == first
    (second_root / fingerprint_source.name).write_text(case("fingerprints", "after"), encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    assert origin_specs.lowering_fingerprint() != first


def test_memo_keys_cover_path_namespace_version_and_python(
    fingerprint_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping any coordinate would share results between incompatible requests."""
    signature = origin_specs._source_signature(fingerprint_source)
    key = origin_specs._source_memo_key((signature,), "first", 1)
    assert origin_specs._source_memo_key((signature,), "second", 1) != key
    assert origin_specs._source_memo_key((signature,), "first", 2) != key
    other_path = (str(fingerprint_source.with_name("other.py")), *signature[1:])
    assert origin_specs._source_memo_key((other_path,), "first", 1) != key
    monkeypatch.setattr(sys.implementation, "cache_tag", "different-python-ast")
    assert origin_specs._source_memo_key((signature,), "first", 1) != key


@pytest.mark.parametrize("fault", ["unwritable", "publish", "invalid-text", "invalid-digest"])
def test_advisory_cache_faults_preserve_fingerprint(
    fingerprint_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A cache refusal or malformed entry must compute the same semantic result."""
    expected = origin_specs.lowering_fingerprint()
    signature = origin_specs._source_signature(fingerprint_source)
    memo = origin_specs._fingerprint_memo_path((signature,), "lowering")
    assert memo is not None
    origin_specs._fingerprint_sources_cached.cache_clear()
    if fault == "unwritable":
        blocked = tmp_path / "blocked-cache"
        blocked.write_text("file", encoding="utf-8")
        monkeypatch.setenv("XDG_CACHE_HOME", str(blocked))
        origin_specs._invalidate_source_signatures()
    elif fault == "publish":
        memo.unlink()

        def refuse_publication(*_args: object) -> None:
            raise PermissionError("advisory publication unavailable")

        monkeypatch.setattr(Path, "replace", refuse_publication)
    elif fault == "invalid-text":
        memo.write_bytes(b"\xff")
    else:
        memo.write_text("x" * 64, encoding="utf-8")
    assert origin_specs.lowering_fingerprint() == expected
    root = origin_specs._source_memo_root()
    if root is not None:
        assert not list(root.glob(".memo-*"))


def test_concurrent_memo_publications_keep_whole_entries(
    fingerprint_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Colliding temporary names or aggregate overwrites lose concurrent publications."""
    signature = origin_specs._source_signature(fingerprint_source)
    barrier = Barrier(4)
    publish = origin_specs._publish_source_memo

    def concurrent_publish(memo: Path, payload: str) -> None:
        barrier.wait()
        publish(memo, payload)

    monkeypatch.setattr(origin_specs, "_publish_source_memo", concurrent_publish)
    # Two writers agree on one key; two others publish independent keys.
    namespaces = ("one", "one", "two", "three")
    origin_specs._fingerprint_sources_cached.cache_clear()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(lambda name: origin_specs._fingerprint_sources_cached((signature,), name), namespaces)
        )
    assert results[0] == results[1]
    root = origin_specs._source_memo_root()
    assert root is not None
    for namespace, expected in zip(namespaces, results, strict=True):
        memo = origin_specs._fingerprint_memo_path((signature,), namespace)
        assert memo is not None
        assert memo.read_text(encoding="utf-8") == expected
    assert not list(root.glob(".memo-*"))
