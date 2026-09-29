"""Every acquisition route reads bound source bytes through one boundary.

``polylogue.sources.acquisition_boundary`` validates each byte of a bound
source as it is read. These tests hold the boundary to that claim in two
ways: the validator refuses a foreign record wherever it sits in the stream,
and every route that retains or parses source material refuses the same
foreign tail -- with a structural check that no new route can read source
bytes into the archive without passing the boundary.
"""

from __future__ import annotations

import ast
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONDocumentList
from polylogue.sources.acquisition_boundary import (
    BoundRecordValidator,
    bind_stream,
    capture_bound_path,
    refuse_foreign_path,
)
from polylogue.sources.dispatch import ForeignOriginContentError
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

_SESSION_ID = "bad69218-73bd-490a-869a-2b3a30bf421b"
_CLAUDE: JSONDocumentList = [
    {
        "type": "user",
        "uuid": "u1",
        "sessionId": _SESSION_ID,
        "timestamp": "2025-06-13T17:40:00.000Z",
        "cwd": "/home/user/project",
        "message": {"role": "user", "content": "Search for ad-hoc solutions."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "parentUuid": "u1",
        "sessionId": _SESSION_ID,
        "timestamp": "2025-06-13T17:40:05.000Z",
        "cwd": "/home/user/project",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Looking now."}]},
    },
]
_CODEX: JSONDocumentList = [
    {"type": "session_meta", "payload": {"id": "codex-session-1", "timestamp": "2026-01-01T10:00:00Z"}},
    {
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Run checks."}]},
    },
]
#: Larger than every prefix or record window acquisition ever used (8 KiB,
#: then 16 MiB), so a window anywhere in the boundary would miss it.
_PAST_ANY_WINDOW = 17 * 1024 * 1024


def _jsonl(records: JSONDocumentList) -> bytes:
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")


def _padded_claude(pad: int) -> JSONDocument:
    first = dict(_CLAUDE[0])
    first["message"] = {"role": "user", "content": "x" * pad}
    return first


def _validate(name: str, data: bytes) -> None:
    validator = BoundRecordValidator(name, Provider.CLAUDE_CODE)
    for start in range(0, len(data), 1 << 20):
        validator.feed(data[start : start + (1 << 20)])
    validator.finish()


@pytest.mark.parametrize(
    ("name", "document"),
    [
        # A foreign record after an own-origin record larger than any window.
        ("tail.jsonl", lambda: _jsonl([_padded_claude(_PAST_ANY_WINDOW), *_CODEX])),
        # One JSONL record whose discriminator sits past any window.
        ("line.jsonl", lambda: _jsonl([{"pad": "x" * _PAST_ANY_WINDOW, **_CODEX[0]}])),
        # A trailing record with no newline is validated at end of stream.
        ("unterminated.jsonl", lambda: _jsonl(_CLAUDE) + json.dumps(_CODEX[0]).encode()),
        # A top-level array document whose foreign element sits after a padded one.
        ("array.json", lambda: json.dumps([_padded_claude(64 * 1024), *_CODEX]).encode()),
        # A top-level array of foreign records is not nested one level deeper.
        ("codex-array.json", lambda: json.dumps([{"pad": "x" * 64 * 1024, **_CODEX[0]}, _CODEX[1]]).encode()),
        # An object document whose discriminator sits past any window.
        ("object.json", lambda: json.dumps({"pad": "x" * _PAST_ANY_WINDOW, **_CODEX[0]}).encode()),
        # A record whose origin declares only a record detector (Antigravity's
        # language-server export envelope), inside a JSONL stream.
        (
            "envelope.jsonl",
            lambda: _jsonl(
                [
                    _CLAUDE[0],
                    {"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "# Chat"},
                ]
            ),
        ),
    ],
)
def test_validator_refuses_a_foreign_record_wherever_it_sits(name: str, document: Callable[[], bytes]) -> None:
    """Anti-vacuity: any prefix, first-record or fixed record window admits one of these."""
    with pytest.raises(ForeignOriginContentError):
        _validate(name, document())


def test_a_record_of_many_values_is_classified_from_a_bounded_view() -> None:
    """A record's classification view stays bounded however many values it holds.

    The Codex discriminator (``payload``) follows an array far longer than
    any container budget, so the record is still refused, while the view
    built for it keeps at most the record budget of values.

    Anti-vacuity: a builder that copies every value retains the whole
    array; one that stops reading at a budget never sees ``payload``.
    """
    import ijson

    from polylogue.sources import acquisition_boundary

    record = {"type": "session_meta", "pad": list(range(1 << 19)), "payload": _CODEX[0]["payload"]}
    line = json.dumps(record).encode()
    with pytest.raises(ForeignOriginContentError):
        _validate("big.jsonl", line + b"\n")

    builder = acquisition_boundary._EvidenceBuilder()
    for event, value in ijson.basic_parse(line, use_float=True):
        builder.event(event, value)

    def retained(value: object) -> int:
        if isinstance(value, dict):
            return 1 + sum(retained(item) for item in value.values())
        if isinstance(value, list):
            return 1 + sum(retained(item) for item in value)
        return 1

    view = builder.value
    assert isinstance(view, dict)
    assert view["payload"] == _CODEX[0]["payload"]
    assert retained(view) <= acquisition_boundary._RECORD_KEEP_VALUES
    assert len(view["pad"]) == acquisition_boundary._CONTAINER_KEEP_ENTRIES


def test_validator_admits_own_origin_material_of_any_size() -> None:
    """The same shapes of Claude Code's own records pass, however large.

    Anti-vacuity: a validator that refused on size or on an inconclusive
    record would refuse these too.
    """
    _validate("tail.jsonl", _jsonl([_padded_claude(_PAST_ANY_WINDOW), *_CLAUDE[1:]]))
    _validate("array.json", json.dumps([_padded_claude(64 * 1024), *_CLAUDE[1:]]).encode())
    _validate("malformed.jsonl", b'{"type": "user", "broken\n' + _jsonl(_CLAUDE))
    # Unbound (inbox), raw-only and non-JSON material is not validated.
    BoundRecordValidator("a.jsonl", Provider.UNKNOWN).feed(_jsonl(_CODEX))
    assert not BoundRecordValidator("history.jsonl", Provider.CLAUDE_CODE).active
    assert not BoundRecordValidator("notes.md", Provider.CLAUDE_CODE).active


def test_bound_stream_validates_bytes_once_across_seeks() -> None:
    """Decoders re-read ``.json`` documents after seeking; bytes are validated once, in order.

    Anti-vacuity: re-feeding re-read bytes corrupts the incremental parse
    into a spurious malformed document; skipping a forward seek leaves the
    skipped foreign record unvalidated.
    """
    from io import BytesIO

    document = json.dumps([*_CLAUDE, *_CLAUDE]).encode()
    stream = bind_stream(BytesIO(document), "doc.json", Provider.CLAUDE_CODE)
    stream.read(100)
    stream.seek(0)
    assert stream.read() == document

    foreign = _jsonl([*_CLAUDE, *_CODEX])
    stream = bind_stream(BytesIO(foreign), "doc.jsonl", Provider.CLAUDE_CODE)
    with pytest.raises(ForeignOriginContentError):
        stream.seek(len(foreign))


# -- every acquisition route --------------------------------------------------

_FOREIGN_TAIL = _jsonl([_padded_claude(64 * 1024), *_CODEX])
_NAME = f"{_SESSION_ID}.jsonl"


def _plain(tmp_path: Path) -> Path:
    path = tmp_path / ".claude" / "projects" / "proj" / _NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(_FOREIGN_TAIL)
    return path


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / ".claude" / "projects" / "bundle.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_NAME, _FOREIGN_TAIL)
    return path


def _refused(cursor_state: CursorStatePayload) -> bool:
    return "foreign_origin_content" in str(cursor_state.get("failed_files"))


def _route_parse_only(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.source_parsing import parse_one_source_path

    with pytest.raises(ForeignOriginContentError):
        list(
            parse_one_source_path(
                str(_plain(tmp_path)), file_mtime=None, source_name="claude-code", sidecar_data={}, capture_raw=False
            )
        )
    return True


def _route_parse_with_capture(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.source_parsing import parse_one_source_path

    with pytest.raises(ForeignOriginContentError):
        list(
            parse_one_source_path(
                str(_plain(tmp_path)),
                file_mtime=None,
                source_name="claude-code",
                sidecar_data={},
                capture_raw=True,
                blob_store=store,
            )
        )
    return True


def _route_zip_parse(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.decoder_zip import process_zip

    cursor_state: CursorStatePayload = {"failed_count": 0, "failed_files": []}
    sessions = list(
        process_zip(
            _archive(tmp_path),
            provider_hint=Provider.CLAUDE_CODE,
            should_group=True,
            file_mtime=None,
            capture_raw=True,
            cursor_state=cursor_state,
            blob_store=store,
        )
    )
    return not sessions and _refused(cursor_state)


def _route_acquire(path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.config import Source
    from polylogue.sources.source_acquisition import iter_source_raw_data

    cursor_state: CursorStatePayload = {"failed_count": 0, "failed_files": []}
    items = list(
        iter_source_raw_data(Source(name="claude-code", path=path), blob_store=store, cursor_state=cursor_state)
    )
    return not items and _refused(cursor_state)


def _route_acquire_file(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    return _route_acquire(_plain(tmp_path), store)


def _route_acquire_zip_member(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    return _route_acquire(_archive(tmp_path), store)


def _route_retained(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.retained_acquisition import iter_retained_source_records

    path = _plain(tmp_path)
    # The retained route decodes a physical blob frozen before the location
    # is known; the boundary refuses it at decode.
    raw = BlobStore(tmp_path / "physical")
    blob_hash, blob_size = raw.write_from_bytes(path.read_bytes())
    with pytest.raises(ForeignOriginContentError):
        list(
            iter_retained_source_records(
                source_path=str(path),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=raw,
                source_name="claude-code",
            )
        )
    return True


def _route_zip_replay(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.config import Source
    from polylogue.sources.source_acquisition_components import (
        ZipEntryReadContext,
        replay_zip_entry_acquisition_revisions,
    )

    archive = _archive(tmp_path)
    with zipfile.ZipFile(archive) as zf:
        context = ZipEntryReadContext(
            Source(name="claude-code", path=archive),
            archive,
            zf.infolist()[0],
            None,
            Provider.CLAUDE_CODE,
            store,
            bound_provider=Provider.CLAUDE_CODE,
        )
        with pytest.raises(ForeignOriginContentError):
            list(replay_zip_entry_acquisition_revisions(zf, context))
    return True


def _route_production_baseline(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import WatchSource

    path = _plain(tmp_path)
    baseline = capture_production_source_baseline(
        (WatchSource(name="claude-code", root=path.parent.parent, suffixes=(".jsonl",)),),
        operation_id="op-test",
    )
    [decision] = [decision for decision in baseline.decisions if Path(decision.path).name == _NAME]
    return decision.disposition == "excluded" and "foreign_origin_content" in decision.reason


def _route_live_capture(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    # The live batch retains every plain source file through this call.
    with pytest.raises(ForeignOriginContentError):
        capture_bound_path(store, _plain(tmp_path), Provider.CLAUDE_CODE)
    return True


def _route_live_append(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    from polylogue.sources.acquisition_boundary import admit_bound_bytes

    # The live batch admits an appended delta through this call.
    with pytest.raises(ForeignOriginContentError):
        admit_bound_bytes(_jsonl(_CODEX), _NAME, Provider.CLAUDE_CODE)
    return True


def _route_non_session_candidate(tmp_path: Path, store: ArchiveBlobPublisher) -> bool:
    # Candidates admitted to no parser are still refused when foreign.
    with pytest.raises(ForeignOriginContentError):
        refuse_foreign_path(_plain(tmp_path), Provider.CLAUDE_CODE)
    return True


ACQUISITION_ROUTES: dict[str, Callable[[Path, ArchiveBlobPublisher], bool]] = {
    "one-shot parse, parse-only": _route_parse_only,
    "one-shot parse, capturing": _route_parse_with_capture,
    "one-shot ZIP parse": _route_zip_parse,
    "acquisition, plain file": _route_acquire_file,
    "acquisition, ZIP member": _route_acquire_zip_member,
    "retained physical blob decode": _route_retained,
    "ZIP member replay": _route_zip_replay,
    "production baseline": _route_production_baseline,
    "live full capture": _route_live_capture,
    "live append delta": _route_live_append,
    "non-session candidate": _route_non_session_candidate,
}


@pytest.mark.parametrize("route", sorted(ACQUISITION_ROUTES))
def test_every_acquisition_route_refuses_a_foreign_tail(route: str, tmp_path: Path) -> None:
    """A Codex record after a large Claude Code record, at Claude Code's location.

    Each route must refuse it and leave nothing queued for publication.
    Anti-vacuity: a route that validates a prefix or its first record, or
    reads the file outside the boundary, admits this file.
    """
    store = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    assert ACQUISITION_ROUTES[route](tmp_path, store)
    assert not store.has_pending


# -- no route bypasses the boundary --------------------------------------------

#: Calls that read source bytes into the archive or open an archive member.
_SOURCE_BYTE_SINKS = frozenset(
    {
        "write_from_path",
        "write_from_fileobj",
        "write_from_writer",
        "prepare_from_path",
        "prepare_from_fileobj",
        "prepare_from_writer",
        "open_bounded_zip_entry",
    }
)

#: Call sites outside the boundary, each with the reason it is not an
#: acquisition of bound session material. A new route is not added here: it
#: reads through ``acquisition_boundary``.
_DECLARED_NON_ACQUISITION_SITES: dict[tuple[str, str], str] = {
    ("polylogue/storage/source_blob_restoration.py", "stage_exact_blob"): (
        "Exact-byte restoration of an already-retained raw: verifies the existing SHA-256 and size "
        "before staging, and publishes through the archive reservation owner; no new acquisition."
    ),
    ("polylogue/sources/acquisition_boundary.py", "open_bound_member"): "the boundary itself",
    ("polylogue/sources/acquisition_boundary.py", "capture_bound_stream"): "the boundary itself",
    ("polylogue/storage/blob_store.py", "BlobStore.write_from_path"): "blob store implementation",
    ("polylogue/storage/blob_store.py", "BlobStore.write_from_fileobj"): "blob store implementation",
    ("polylogue/storage/blob_store.py", "BlobStore.write_from_writer"): "blob store implementation",
    ("polylogue/storage/blob_publication.py", "ArchiveBlobPublisher.write_from_path"): "publisher delegation",
    ("polylogue/storage/blob_publication.py", "ArchiveBlobPublisher.write_from_fileobj"): "publisher delegation",
    ("polylogue/storage/blob_publication.py", "ArchiveBlobPublisher.write_from_writer"): "publisher delegation",
    ("polylogue/operations/attachment_convergence.py", "_download_prepared"): (
        "downloads a provider-hosted attachment, not session records"
    ),
    ("polylogue/pipeline/services/ingest_worker.py", "_browser_capture_payload.spill"): (
        "decodes a retained browser capture's attachment carriers into blobs, not session records"
    ),
    ("polylogue/operations/ingest_inputs.py", "retain_input_page"): (
        "freezes a declared input's physical bytes; its decode "
        "(retained_acquisition) reads them through the boundary before any raw"
    ),
    ("polylogue/sources/sqlite_snapshot.py", "snapshot_sqlite_to_blob"): (
        "retains a declared database's logical export; the declaration is refused by refuse_declared_foreign"
    ),
    ("polylogue/sources/assembly_chatgpt.py", "_read_json_zip_member"): "export sidecar read, not session records",
    ("polylogue/sources/assembly_chatgpt.py", "_read_chatgpt_zip_sidecars"): "export asset attachments",
    ("polylogue/sources/assembly_chatgpt.py", "_acquire_asset_blobs_from_directory"): "export asset attachments",
    (
        "polylogue/sources/decoder_zip.py",
        "zip_entry_session_artifact",
    ): "classification probe; parse reads via boundary",
    ("polylogue/sources/source_acquisition_components.py", "sniff_zip_provider"): "unbound provider sniff",
    ("polylogue/sources/import_explain.py", "_explain_zip"): "read-only diagnostics",
    ("polylogue/sources/import_preflight.py", "_preflight_zip"): "read-only diagnostics",
    ("polylogue/schemas/source_inference.py", "_collect_zip_candidate"): "schema inference tooling",
}


class _SinkVisitor(ast.NodeVisitor):
    """Record the enclosing qualified name of every source-byte sink call."""

    def __init__(self, module: str, sites: set[tuple[str, str]]) -> None:
        self._module = module
        self._sites = sites
        self._scope: list[str] = []

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name in _SOURCE_BYTE_SINKS:
            self._sites.add((self._module, ".".join(self._scope)))
        self.generic_visit(node)


def _sink_call_sites() -> set[tuple[str, str]]:
    repo = Path(__file__).resolve().parents[3]
    sites: set[tuple[str, str]] = set()
    for module in sorted((repo / "polylogue").rglob("*.py")):
        visitor = _SinkVisitor(module.relative_to(repo).as_posix(), sites)
        visitor.visit(ast.parse(module.read_text(encoding="utf-8")))
    return sites


def test_no_route_reads_source_bytes_around_the_boundary() -> None:
    """Every retention or member open outside the boundary is declared and reasoned.

    Anti-vacuity: a new acquisition route calling ``write_from_path`` or
    ``open_bounded_zip_entry`` directly appears here as undeclared; a stale
    declaration (its site removed) appears as unused.
    """
    sites = _sink_call_sites()
    assert sites - _DECLARED_NON_ACQUISITION_SITES.keys() == set()
    assert _DECLARED_NON_ACQUISITION_SITES.keys() - sites == set()
