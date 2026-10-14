"""Exact public depth grammars, retained and loaded one native row at a time."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import weakref
from collections.abc import Generator, Iterator, Mapping, MutableMapping
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import ijson
from ijson.common import ObjectBuilder

from polylogue.core.digest import RECEIPT, REFERENCE, canonical_bytes, nfc
from polylogue.core.json import JSONDocument, json_document, require_json_document
from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction, open_scratch_connection


class DepthGrammars(Mapping[str, JSONDocument]):
    """A serialized public row tape, independent of the retiring private journal.

    Rows are produced by the original grammar pass, never reconstructed here.
    A profile, registry cache, or selected corpus retains this carrier while it
    borrows rows. Last-reference cleanup closes its native reader before files.
    """

    def __init__(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="workload-depth-")
        owner = open_scratch_connection(Path(directory.name) / "rows.db", scratch_directory=directory)
        connection = owner.require_connection()
        try:
            connection.execute("CREATE TABLE depth_grammars(depth TEXT PRIMARY KEY, grammar BLOB NOT NULL)").close()
        except BaseException as primary:
            _close_failed_native_construction(owner, primary)
            raise
        self._connection = connection
        self._sql_owner = owner
        self._cleanup = weakref.finalize(self, owner.close)

    def close(self) -> None:
        self._sql_owner.close()
        self._cleanup.detach()

    def append(self, depth: str, grammar: JSONDocument) -> None:
        if not depth.isdecimal() or str(int(depth)) != depth or int(depth) < 1:
            raise ValueError("child grammar requires an exact positive depth")
        # Custody preserves the original public row. NFC belongs to the chosen
        # receipt serializer, not to acquisition of a lossless reference row.
        try:
            self._connection.execute(
                "INSERT INTO depth_grammars VALUES (?,?)", (depth, canonical_bytes(grammar, REFERENCE))
            ).close()
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate workload profile depth") from exc

    def __getitem__(self, depth: str) -> JSONDocument:
        with closing(self._connection.execute("SELECT grammar FROM depth_grammars WHERE depth=?", (depth,))) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise KeyError(depth)
        return json_document(json.loads(row[0]))

    def __len__(self) -> int:
        with closing(self._connection.execute("SELECT count(*) FROM depth_grammars")) as cursor:
            row = cursor.fetchone()
        assert row is not None
        return int(row[0])

    def encoded_rows(self) -> Iterator[tuple[str, bytes]]:
        # Lexical key order is the existing canonical JSON object order.
        after = ""
        while True:
            with closing(
                self._connection.execute(
                    "SELECT depth,grammar FROM depth_grammars WHERE depth>? ORDER BY depth LIMIT 1", (after,)
                )
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                return
            after, encoded = row
            yield after, encoded

    def __iter__(self) -> Iterator[str]:
        for depth, _ in self.encoded_rows():
            yield depth


class WorkloadProfile(MutableMapping[str, object]):
    """Bounded profile fields plus the original serialized depth row carrier."""

    def __init__(self, fields: Mapping[str, object], depths: DepthGrammars | None = None) -> None:
        self._fields = dict(fields)
        self.depth_grammars = depths

    def __getitem__(self, key: str) -> object:
        return self._fields[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        del self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False

        def encoded_identity(value: Mapping[str, object]) -> bytes:
            hasher = hashlib.sha256()
            for chunk in iter_workload_profile_chunks(value):
                hasher.update(chunk)
            return hasher.digest()

        return encoded_identity(self) == encoded_identity(other)


class _BinaryRead(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def iter_workload_profile_chunks(
    profile: Mapping[str, object], *, identity: bool = False, normalize_unicode: bool = True
) -> Generator[bytes, None, None]:
    """Frame profile rows and fixture entries around the existing canonical encoder.

    Schemas and each already-owned bounded profile body use the original encoder
    once. Only the exact-depth native tape and manifest entry sequence are framed
    here, so serialization neither expands the tape nor visits every scalar in
    Python again.
    """
    encoding = RECEIPT if normalize_unicode else REFERENCE

    def chunks(value: object, path: tuple[str, ...] = (), depths: DepthGrammars | None = None) -> Iterator[bytes]:
        if isinstance(value, WorkloadProfile):
            depths, path = value.depth_grammars, ()
        if isinstance(value, DepthGrammars):
            yield b"{"
            for ordinal, (depth, encoded) in enumerate(value.encoded_rows()):
                if ordinal:
                    yield b","
                yield canonical_bytes(depth, encoding) + b":"
                yield canonical_bytes(json.loads(encoded), encoding) if normalize_unicode else encoded
            yield b"}"
        elif isinstance(value, Mapping) and path in {
            (),
            ("elements",),
            ("elements", "agent_transcript"),
            ("entries", "item"),
        }:
            if path == ("elements", "agent_transcript") and depths is not None:
                value = {**value, "record_structures_by_depth": depths}
            keys: dict[str, str] = {}
            for key in value:
                if not isinstance(key, str):
                    raise ValueError("workload profile keys must be strings")
                normalized = nfc(key) if normalize_unicode else key
                if normalized in keys:
                    raise ValueError("workload profile keys collide under NFC")
                keys[normalized] = key
            yield b"{"
            for ordinal, key in enumerate(sorted(keys)):
                if ordinal:
                    yield b","
                yield canonical_bytes(key, encoding) + b":"
                yield from chunks(value[keys[key]], (*path, key), depths)
            yield b"}"
        elif isinstance(value, list | tuple) and path == ("entries",):
            yield b"["
            for ordinal, item in enumerate(value):
                if ordinal:
                    yield b","
                yield from chunks(item, (*path, "item"), depths)
            yield b"]"
        else:
            yield canonical_bytes(value, encoding)

    fields = {key: value for key, value in profile.items() if not identity or key != "profile_id"}
    yield from chunks(fields, depths=profile.depth_grammars if isinstance(profile, WorkloadProfile) else None)


def workload_profile_digest(profile: Mapping[str, object]) -> str:
    hasher = hashlib.sha256()
    for chunk in iter_workload_profile_chunks(profile, identity=True):
        hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _read_workload_profile_document(handle: _BinaryRead) -> object:
    """Read a profile or fixture envelope using the same exact row diversion."""
    suffix = "elements.agent_transcript.record_structures_by_depth"
    prefix: str | None = None
    builder = ObjectBuilder()
    rows: DepthGrammars | None = None
    row_builder: ObjectBuilder | None = None
    row_depth = ""
    row_level = 0
    map_keys: list[set[str] | None] = []
    for path, event, value in ijson.parse(handle, use_float=False):
        if isinstance(value, Decimal):
            value = float(value)
        if event in {"start_map", "start_array"}:
            map_keys.append(
                set()
                if event == "start_map" and not (path == suffix or path.endswith(".workload_profile." + suffix))
                else None
            )
        elif event == "map_key":
            keys = map_keys[-1]
            # Exact depth-key uniqueness is proved by the native primary key;
            # retaining all those keys in Python would defeat row streaming.
            if keys is not None:
                if value in keys:
                    raise ValueError("duplicate workload profile key")
                keys.add(value)
        elif event in {"end_map", "end_array"}:
            map_keys.pop()
        if event == "start_map" and (path == suffix or path.endswith(".workload_profile." + suffix)):
            rows, prefix = DepthGrammars(), path
            builder.event("null", rows)
            continue
        if prefix is not None and (path == prefix or path.startswith(prefix + ".")):
            assert rows is not None
            if path == prefix:
                if event == "map_key":
                    row_depth = str(value)
                elif event == "end_map":
                    prefix = None
                continue
            if row_builder is None:
                if event != "start_map" or path != prefix + "." + row_depth:
                    raise ValueError("depth grammar must be a public object")
                row_builder = ObjectBuilder()
            row_builder.event(event, value)
            row_level += int(event in {"start_map", "start_array"}) - int(event in {"end_map", "end_array"})
            if row_level == 0:
                rows.append(row_depth, require_json_document(row_builder.value, context="depth grammar"))
                row_builder = None
            continue
        builder.event(event, value)

    def profile(fields: dict[str, object]) -> WorkloadProfile:
        elements = fields.get("elements")
        agent = elements.get("agent_transcript") if isinstance(elements, dict) else None
        depths = agent.pop("record_structures_by_depth", None) if isinstance(agent, dict) else None
        if depths is not None and not isinstance(depths, DepthGrammars):
            raise ValueError("depth grammars lost their original row carrier")
        return WorkloadProfile(fields, depths)

    root = builder.value
    # Fixture envelopes already retain their ordinary bounded manifest fields;
    # only this profile row family is diverted, never expanded during traversal.
    pending = [root]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if key == "workload_profile" and isinstance(item, dict):
                    current[key] = profile(item)
                else:
                    pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    if isinstance(root, dict) and "elements" in root:
        return profile(root)
    return root


def read_workload_profile_document(handle: _BinaryRead) -> object:
    try:
        return _read_workload_profile_document(handle)
    except ijson.JSONError as exc:
        raise ValueError("invalid or non-finite workload profile JSON") from exc


def read_workload_profile(handle: _BinaryRead) -> WorkloadProfile:
    value = read_workload_profile_document(handle)
    if isinstance(value, WorkloadProfile):
        return value
    return WorkloadProfile(require_json_document(value, context="workload profile"))
