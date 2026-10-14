"""Relational constraint satisfaction for synthetic generation.

Uses ``x-polylogue-foreign-keys``, ``x-polylogue-time-deltas`` and
``x-polylogue-mutually-exclusive`` annotations on schemas to produce structurally coherent synthetic data.

These constraints are read from the schema root and applied during the
object-generation phase to enforce cross-field consistency.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Generator
from contextlib import closing, suppress
from dataclasses import dataclass, field
from typing import cast

from polylogue.schemas.field_stats.distributions import _bucket_index
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.models import (
    GeneratedRecord,
    GeneratedString,
    GeneratedValue,
    SchemaRecord,
    SchemaValue,
)
from polylogue.schemas.synthetic.relation_solver_runtime import RelationConstraintSolverRuntimeMixin
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
)

_TOOL_RELATION_PREDICATES = {
    "call": "calls>0 AND results=0",
    "result": "calls=0 AND results>0",
    "both": "calls>0 AND results>0",
}


@dataclass
class ForeignKeyGraph:
    """Reference graph built from x-polylogue-foreign-keys.

    Tracks generated ID values so foreign-key fields can reference
    previously generated IDs.
    """

    # source_path -> target_path
    references: dict[str, str] = field(default_factory=dict)
    _connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)
    _sql_owner: NativeSQLCustodyOwner | None = field(default=None, init=False, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _missing_tool_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _generation: int = field(default=0, init=False, repr=False)
    _next_array_id: int = field(default=0, init=False, repr=False)
    _next_array_revision: int = field(default=0, init=False, repr=False)
    _captured_schema_nodes: dict[int, tuple[SchemaValue, int]] = field(default_factory=dict, init=False, repr=False)

    _captured_schema_values: dict[int, SchemaValue] = field(default_factory=dict, init=False, repr=False)

    def _owner(self) -> sqlite3.Connection:
        if self._connection is None:
            # SQLite's temporary main database is file-backed and deleted by
            # its own physical close. It never opens an archive tier.
            connection = connect_measured("")
            owner = NativeSQLCustodyOwner(connection)
            try:
                connection.create_function(
                    "generated_count_bucket", 1, lambda value: _bucket_index(float(value)), deterministic=True
                )
                connection.create_function(
                    "generated_string_length",
                    1,
                    lambda state: len(_string_from_state(json.loads(state))),
                    deterministic=True,
                )
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("PRAGMA cache_size = -2048")
                connection.execute(
                    "CREATE TABLE generated_ids (target_path TEXT NOT NULL, ordinal INTEGER NOT NULL, "
                    "value_json TEXT NOT NULL, PRIMARY KEY(target_path, ordinal)) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_arrays (array_id INTEGER PRIMARY KEY, count INTEGER NOT NULL) STRICT"
                )
                connection.execute(
                    "CREATE TABLE generated_array_items (array_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, "
                    "value_json TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(array_id, ordinal)) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_path_layers(path_id INTEGER NOT NULL,remaining INTEGER NOT NULL,"
                    "state TEXT NOT NULL,log_mass REAL NOT NULL,required_resource INTEGER NOT NULL,count_support BLOB NOT NULL,"
                    "PRIMARY KEY(path_id,remaining,state)) STRICT, WITHOUT ROWID"
                ).close()
                connection.execute(
                    "CREATE TABLE generated_record_ids (array_id INTEGER NOT NULL, value TEXT NOT NULL, "
                    "PRIMARY KEY(array_id,value)) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_schema_nodes(schema_id INTEGER PRIMARY KEY,node_json TEXT NOT NULL UNIQUE) STRICT"
                )
                connection.execute(
                    "CREATE TABLE generated_records (record_id INTEGER PRIMARY KEY, count INTEGER NOT NULL, "
                    "next_ordinal INTEGER NOT NULL, schema_id INTEGER NOT NULL, original_path TEXT, "
                    "tool_reference_observed INTEGER NOT NULL DEFAULT 0) STRICT"
                )
                connection.execute(
                    "CREATE TABLE generated_record_fields (record_id INTEGER NOT NULL, key TEXT NOT NULL, "
                    "ordinal INTEGER NOT NULL, value_json TEXT NOT NULL, PRIMARY KEY(record_id,key), "
                    "UNIQUE(record_id,ordinal)) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_tool_results (tool_use_id TEXT PRIMARY KEY, "
                    "record_id INTEGER NOT NULL, inline_length INTEGER NOT NULL, truncated INTEGER NOT NULL) "
                    "STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_persisted_tool_names (filename TEXT PRIMARY KEY, "
                    "tool_use_id TEXT NOT NULL) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_tool_identities(value_json TEXT PRIMARY KEY,calls INTEGER NOT NULL,results INTEGER NOT NULL,schema_json TEXT NOT NULL,first_event TEXT NOT NULL,result_mode TEXT NOT NULL,tool_class TEXT NOT NULL) STRICT, WITHOUT ROWID"
                )
                # Keep inventory eligibility on a covering native index. The
                # original table also retains schema payloads; repeated count
                # and rank selection must not read those payloads for every ID.
                for relation, predicate in _TOOL_RELATION_PREDICATES.items():
                    connection.execute(
                        f"CREATE INDEX generated_tool_inventory_{relation} ON generated_tool_identities("
                        "value_json,result_mode,tool_class,calls,results,generated_count_bucket(results),"
                        "generated_string_length(value_json)) WHERE " + predicate
                    ).close()
                connection.execute(
                    "CREATE TABLE generated_tool_reference_events(ordinal INTEGER PRIMARY KEY,record_id INTEGER NOT NULL UNIQUE, tool_use_id TEXT NOT NULL,mode TEXT NOT NULL,basename_bucket INTEGER NOT NULL) STRICT"
                )
                connection.execute(
                    "CREATE INDEX generated_tool_reference_history ON generated_tool_reference_events(tool_use_id,ordinal)"
                )
                connection.execute(
                    "CREATE TABLE generated_adjuncts(ordinal INTEGER PRIMARY KEY,name TEXT NOT NULL UNIQUE,kind TEXT NOT NULL,byte_length INTEGER NOT NULL) STRICT"
                )
                connection.execute(
                    "CREATE TABLE generated_agent_children(ordinal INTEGER PRIMARY KEY,record_id INTEGER NOT NULL UNIQUE,"
                    "transcripts INTEGER NOT NULL,record_count INTEGER NOT NULL,metadata INTEGER NOT NULL,depth INTEGER NOT NULL,metadata_json TEXT) STRICT"
                )
                connection.execute("CREATE TABLE generated_agent_names(stem TEXT PRIMARY KEY) STRICT, WITHOUT ROWID")
                connection.execute(
                    "CREATE TABLE generated_identity_shapes(kind TEXT NOT NULL,length TEXT NOT NULL,newlines TEXT NOT NULL,"
                    "prefix TEXT NOT NULL,phase INTEGER NOT NULL,ordinal TEXT NOT NULL,"
                    "PRIMARY KEY(kind,length,newlines,prefix)) STRICT, WITHOUT ROWID"
                ).close()
                connection.execute(
                    "CREATE TABLE generated_structured_adjuncts(ordinal INTEGER PRIMARY KEY,name TEXT NOT NULL UNIQUE,kind TEXT NOT NULL,byte_length INTEGER NOT NULL,cell_kind TEXT NOT NULL,cell_id INTEGER NOT NULL) STRICT"
                )
                connection.execute(
                    "CREATE TABLE generated_structured_adjunct_chunks(ordinal INTEGER NOT NULL,chunk INTEGER NOT NULL,data BLOB NOT NULL,PRIMARY KEY(ordinal,chunk)) STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_session_prefixes(prefix_id INTEGER PRIMARY KEY, native_id TEXT NOT NULL UNIQUE, "
                    "parent_prefix INTEGER, inherited_messages INTEGER NOT NULL, own_messages INTEGER NOT NULL, "
                    "record_count INTEGER NOT NULL, depth INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0, "
                    "selection_ordinal INTEGER) STRICT"
                )
                connection.execute(
                    "CREATE UNIQUE INDEX generated_session_prefix_selection ON generated_session_prefixes("
                    "record_count,inherited_messages+own_messages,selection_ordinal) WHERE completed=1"
                )
                connection.execute(
                    "CREATE TABLE generated_session_prefix_chunks(prefix_id INTEGER NOT NULL,message_ordinal INTEGER NOT NULL, "
                    "chunk_ordinal INTEGER NOT NULL,data BLOB NOT NULL,PRIMARY KEY(prefix_id,message_ordinal,chunk_ordinal)) "
                    "STRICT, WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE generated_session_prefix_tools(prefix_id INTEGER NOT NULL,event TEXT NOT NULL, "
                    "message_ordinal INTEGER NOT NULL,ordinal INTEGER NOT NULL,identity TEXT NOT NULL, "
                    "schema_json TEXT,result_mode TEXT NOT NULL,tool_class TEXT NOT NULL, "
                    "PRIMARY KEY(prefix_id,event,message_ordinal,ordinal)) STRICT, WITHOUT ROWID"
                )
                connection.create_collation("generated_numeric_order", _compare_numeric_states)
            except BaseException as primary:
                _close_failed_native_construction(owner, primary)
                raise
            self._sql_owner = owner
            self._connection = connection
        return self._connection

    def _live_tool_owner(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("tool identity has lost its original native owner")
        return self._connection

    def tool_identity_is_new(self, value: GeneratedString) -> bool:
        if self._connection is None:
            raise RuntimeError("tool identity has lost its original native owner")
        state = json.dumps(_string_state(value), separators=(",", ":"), ensure_ascii=True)
        cursor = self._connection.execute("SELECT 1 FROM generated_tool_identities WHERE value_json=?", (state,))
        try:
            return cursor.fetchone() is None
        finally:
            cursor.close()

    def agent_identity_is_new(self, value: GeneratedString) -> bool:
        from polylogue.core.claude_code_children import claude_agent_transcript_stem

        if not isinstance(value, str) or not value or "/" in value or "\x00" in value:
            return False
        with closing(
            self._live_tool_owner().execute(
                "SELECT 1 FROM generated_agent_names WHERE stem=?", (claude_agent_transcript_stem(value),)
            )
        ) as cursor:
            return cursor.fetchone() is None

    def fresh_shape_identity(
        self,
        kind: str,
        length: int,
        newlines: int,
        *,
        prefix: str = "",
        excluded: frozenset[str] = frozenset(),
        character_class: str | None = None,
    ) -> GeneratedString | None:
        """Skip occupied members or permanently invalid agent spellings.

        A free pending or excluded candidate remains at the checkpoint: it can
        still be eligible on a later draw. Scope retirement clears checkpoint
        rows with their original membership inventory.
        """
        from polylogue.schemas.synthetic.categorical_values import _string_shape_candidates
        from polylogue.schemas.synthetic.text_value import text_equals

        if kind not in {"tool", "agent"}:
            raise ValueError("unknown original identity inventory")
        connection = self._live_tool_owner()
        if character_class not in (None, "ascii", "non_ascii"):
            raise ValueError("unknown anonymous character class")
        shape_kind = kind if character_class is None else f"{kind}:{character_class}"
        key = (shape_kind, str(length), str(newlines), prefix)
        with closing(
            connection.execute(
                "SELECT phase,ordinal FROM generated_identity_shapes WHERE kind=? AND length=? AND newlines=? AND prefix=?",
                key,
            )
        ) as cursor:
            saved = cursor.fetchone()
        phase, ordinal = (0, 0) if saved is None else (saved[0], int(saved[1]))
        checkpoint = (phase, ordinal)
        advance = True
        selected: GeneratedString | None = None
        for candidate_phase, candidate_ordinal, value in _string_shape_candidates(
            length, newlines, prefix=prefix, phase=phase, ordinal=ordinal, character_class=character_class
        ):
            is_new = (
                self.tool_identity_is_new(value)
                if kind == "tool"
                else (
                    isinstance(value, str)
                    and value.startswith("agent-") == bool(prefix)
                    and self.agent_identity_is_new(value)
                )
            )
            if not is_new:
                if advance:
                    checkpoint = (candidate_phase, candidate_ordinal + 1)
                continue
            if any(text_equals(value, item) for item in excluded):
                advance = False
                continue
            selected = value
            break
        connection.execute(
            "INSERT INTO generated_identity_shapes VALUES (?,?,?,?,?,?) ON CONFLICT(kind,length,newlines,prefix) "
            "DO UPDATE SET phase=excluded.phase,ordinal=excluded.ordinal",
            (*key, checkpoint[0], str(checkpoint[1])),
        ).close()
        return selected

    def tool_identity_count(
        self,
        relation: str,
        prior_result_mode: str | None = None,
        prior_count_bucket: int | None = None,
        widths: tuple[int, ...] | None = None,
        tool_class: str | None = None,
    ) -> int:
        predicate = _TOOL_RELATION_PREDICATES[relation]
        cursor = self._live_tool_owner().execute(
            "SELECT count(*) FROM generated_tool_identities WHERE "
            + predicate
            + (" AND result_mode=?" if prior_result_mode is not None else "")
            + (" AND tool_class=?" if tool_class is not None else "")
            + (" AND generated_count_bucket(results)=?" if prior_count_bucket is not None else "")
            + (
                " AND generated_string_length(value_json) IN (SELECT value FROM json_each(?))"
                if widths is not None
                else ""
            ),
            (() if prior_result_mode is None else (prior_result_mode,))
            + (() if tool_class is None else (tool_class,))
            + (() if prior_count_bucket is None else (prior_count_bucket,))
            + (() if widths is None else (json.dumps(widths),)),
        )
        try:
            row = cursor.fetchone()
            assert row is not None
            return int(row[0])
        finally:
            cursor.close()

    def select_tool_identity(
        self,
        relation: str,
        rng: random.Random,
        prior_result_mode: str | None = None,
        prior_count_bucket: int | None = None,
        widths: tuple[int, ...] | None = None,
        tool_class: str | None = None,
    ) -> tuple[GeneratedString, SchemaRecord]:
        predicate = _TOOL_RELATION_PREDICATES[relation]
        count = self.tool_identity_count(relation, prior_result_mode, prior_count_bucket, widths, tool_class)
        if not count:
            raise ValueError("tool relation has no original eligible identity")
        cursor = self._live_tool_owner().execute(
            "SELECT value_json FROM generated_tool_identities WHERE "
            + predicate
            + (" AND result_mode=?" if prior_result_mode is not None else "")
            + (" AND tool_class=?" if tool_class is not None else "")
            + (" AND generated_count_bucket(results)=?" if prior_count_bucket is not None else "")
            + (
                " AND generated_string_length(value_json) IN (SELECT value FROM json_each(?))"
                if widths is not None
                else ""
            )
            + " ORDER BY value_json LIMIT 1 OFFSET ?",
            (() if prior_result_mode is None else (prior_result_mode,))
            + (() if tool_class is None else (tool_class,))
            + (() if prior_count_bucket is None else (prior_count_bucket,))
            + (() if widths is None else (json.dumps(widths),))
            + (rng.randrange(count),),
        )
        try:
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("tool relation lost its original identity")
            selected = row[0]
        finally:
            cursor.close()
        # Fetch the selected original schema only after the exact same sorted
        # inventory rank has been chosen. No skipped candidate payload is read.
        with closing(
            self._live_tool_owner().execute(
                "SELECT schema_json FROM generated_tool_identities WHERE value_json=?", (selected,)
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("tool relation lost its original selected schema")
        return _string_from_state(json.loads(selected)), json.loads(row[0])

    def observe_tool_identity(
        self,
        value: GeneratedString,
        event: str,
        relation: str,
        schema: SchemaRecord,
        result_mode: str = "",
        tool_class: str | None = None,
    ) -> None:
        state = json.dumps(_string_state(value), separators=(",", ":"), ensure_ascii=True)
        cursor = self._live_tool_owner().execute(
            "SELECT calls,results,result_mode,tool_class FROM generated_tool_identities WHERE value_json=?", (state,)
        )
        try:
            before = cursor.fetchone()
        finally:
            cursor.close()
        calls, results, prior_mode, prior_class = (0, 0, "", "") if before is None else before
        actual = "both" if calls and results else "call" if calls else "result" if results else "new"
        if actual != relation:
            raise ValueError("sampled tool identity contradicts the selected original relation")
        self._live_tool_owner().execute(
            "INSERT INTO generated_tool_identities VALUES (?,?,?,?,?,?,?) ON CONFLICT(value_json) DO UPDATE SET calls=excluded.calls,results=excluded.results,result_mode=excluded.result_mode,tool_class=excluded.tool_class",
            (
                state,
                calls + (event == "call"),
                results + (event == "result"),
                json.dumps(schema, ensure_ascii=True, separators=(",", ":")),
                event,
                result_mode if event == "result" else prior_mode,
                prior_class if tool_class is None else tool_class,
            ),
        ).close()

    def observe_missing_tool_identity(self, event: str) -> None:
        self._live_tool_owner()
        key = "call_without_id" if event == "call" else "result_without_id"
        self._missing_tool_counts[key] = self._missing_tool_counts.get(key, 0) + 1

    def tool_relationship_counts(self) -> dict[str, int]:
        if self._connection is None and self._generation == 0:
            # A scalar-only original generation has never enrolled any tool
            # operand or native container. Reporting zero must not open one.
            return {}
        counts = {
            "paired": 0,
            "missing": 0,
            "orphan": 0,
            "duplicate_calls": 0,
            "duplicate_results": 0,
            "out_of_order": 0,
        }
        cursor = self._live_tool_owner().execute("SELECT calls,results,first_event FROM generated_tool_identities")
        try:
            for calls, results, first in cursor:
                counts["paired"] += bool(calls and results)
                counts["missing"] += bool(calls and not results)
                counts["orphan"] += bool(results and not calls)
                counts["duplicate_calls"] += max(0, calls - 1)
                counts["duplicate_results"] += max(0, results - 1)
                counts["out_of_order"] += bool(calls and results and first == "result")
        finally:
            cursor.close()
        counts.update(self._missing_tool_counts)
        return {kind: count for kind, count in counts.items() if count}

    def register_id(self, target_path: str, value: GeneratedString) -> None:
        """Retain the generated spelling recipe at its original path/ordinal."""
        ordinal = self._counts.get(target_path, 0)
        self._owner().execute(
            "INSERT INTO generated_ids VALUES (?, ?, ?)",
            (target_path, ordinal, json.dumps(_string_state(value), ensure_ascii=True, separators=(",", ":"))),
        )
        self._counts[target_path] = ordinal + 1

    def resolve_reference(self, source_path: str, rng: random.Random) -> GeneratedString | None:
        target = self.references.get(source_path)
        count = self._counts.get(target, 0) if target is not None else 0
        if not count:
            return None
        row = (
            self._owner()
            .execute(
                "SELECT value_json FROM generated_ids WHERE target_path = ? AND ordinal = ?",
                (target, rng.randrange(count)),
            )
            .fetchone()
        )
        if row is None:
            raise RuntimeError("generated foreign-key coordinate lost its original value")
        return _string_from_state(json.loads(row[0]))

    def iter_generated_ids(self, target_path: str) -> Generator[GeneratedString, None, None]:
        if not self._counts.get(target_path, 0):
            return
        cursor = self._owner().execute(
            "SELECT value_json FROM generated_ids WHERE target_path = ? ORDER BY ordinal", (target_path,)
        )
        try:
            for row in cursor:
                yield _string_from_state(json.loads(row[0]))
        finally:
            cursor.close()

    def new_array(self) -> GeneratedArray:
        array_id = self._next_array_id
        self._owner().execute("INSERT INTO generated_arrays VALUES (?, 0)", (array_id,)).close()
        self._next_array_id += 1
        return GeneratedArray(self, array_id, self._generation)

    def _array_owner(self, generation: int) -> sqlite3.Connection:
        if generation != self._generation or self._connection is None:
            raise RuntimeError("generated array outlived its original relation owner")
        return self._connection

    def array_length(self, array_id: int, generation: int) -> int:
        cursor = self._array_owner(generation).execute(
            "SELECT count FROM generated_arrays WHERE array_id=?", (array_id,)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError("generated array lost its original coordinate")
        return int(row[0])

    def claim_record_id(self, array: GeneratedArray, value: str) -> None:
        if array.owner is not self:
            raise ValueError("generated record belongs to another relation owner")
        try:
            self._array_owner(array.generation).execute(
                "INSERT INTO generated_record_ids VALUES (?, ?)", (array.array_id, value)
            ).close()
        except sqlite3.IntegrityError as error:
            raise ValueError("record node identity must be unique") from error

    def capture_schema(self, schema: SchemaRecord) -> SchemaRecord:
        """Snapshot a selected original schema once on this physical owner.

        Only captured input containers stay in the Python coordinate index.
        Generated union overlays are interned natively without growing that index.
        """
        from polylogue.schemas.json_walk import clone_finite_json

        captured = cast(SchemaRecord, clone_finite_json(schema))
        self.schema_coordinate(captured, capture=True)
        return captured

    def schema_coordinate(self, schema: SchemaValue, *, capture: bool = False) -> int:
        """Retain exact finite schema nodes, reusing captured child coordinates."""
        connection = self._owner()
        coordinates: dict[int, int] = {}
        active: set[int] = set()
        frames: list[tuple[SchemaValue, bool]] = [(schema, False)]
        while frames:
            value, complete = frames.pop()
            identity = id(value)
            captured = self._captured_schema_nodes.get(identity)
            if captured is not None and captured[0] is value:
                coordinates[identity] = captured[1]
                continue
            if identity in coordinates:
                continue
            if isinstance(value, dict | list) and not complete:
                if identity in active:
                    raise ValueError("cyclic objects have no finite JSON representation")
                active.add(identity)
                frames.append((value, True))
                children = value.values() if isinstance(value, dict) else value
                frames.extend((child, False) for child in reversed(tuple(children)))
                continue
            if isinstance(value, dict):
                state: list[object] = ["object", [[key, coordinates[id(child)]] for key, child in value.items()]]
            elif isinstance(value, list):
                state = ["list", [coordinates[id(child)] for child in value]]
            else:
                state = ["scalar", value]
            encoded = json.dumps(state, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            connection.execute("INSERT OR IGNORE INTO generated_schema_nodes(node_json) VALUES (?)", (encoded,)).close()
            with closing(
                connection.execute("SELECT schema_id FROM generated_schema_nodes WHERE node_json=?", (encoded,))
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise RuntimeError("selected schema node lost its original native coordinate")
            coordinates[identity] = row[0]
            active.discard(identity)
            if capture and isinstance(value, dict | list):
                self._captured_schema_nodes[identity] = value, row[0]
                self._captured_schema_values[row[0]] = value
        return coordinates[id(schema)]

    def new_record(self, schema: SchemaRecord, *, path: str | None = None) -> GeneratedRecord:
        record_id = self._next_array_id
        schema_id = self.schema_coordinate(schema)
        self._owner().execute(
            "INSERT INTO generated_records(record_id,count,next_ordinal,schema_id,original_path) VALUES (?,0,0,?,?)",
            (record_id, schema_id, path),
        ).close()
        self._next_array_id += 1
        return GeneratedRecord(self, record_id, self._generation)

    def _record_owner(self, record: GeneratedRecord) -> sqlite3.Connection:
        if record.owner is not self:
            raise ValueError("generated record belongs to another relation owner")
        return self._array_owner(record.generation)

    def record_path(self, record: GeneratedRecord) -> str | None:
        cursor = self._record_owner(record).execute(
            "SELECT original_path FROM generated_records WHERE record_id=?", (record.record_id,)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError("generated record lost its original coordinate")
        return row[0] if isinstance(row[0], str) else None

    def record_schema(self, record: GeneratedRecord) -> SchemaRecord:
        connection = self._record_owner(record)
        with closing(
            connection.execute("SELECT schema_id FROM generated_records WHERE record_id=?", (record.record_id,))
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("generated record lost its original coordinate")
        selected = row[0]
        pending = [selected]
        states: dict[int, list[object]] = {}
        values: dict[int, SchemaValue] = {}
        while pending:
            coordinate = pending.pop()
            if coordinate in values:
                continue
            if coordinate in self._captured_schema_values:
                values[coordinate] = self._captured_schema_values[coordinate]
                continue
            with closing(
                connection.execute("SELECT node_json FROM generated_schema_nodes WHERE schema_id=?", (coordinate,))
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise RuntimeError("selected schema lost its original native coordinate")
            state = json.loads(row[0])
            states[coordinate] = state
            if state[0] == "object":
                values[coordinate] = {}
                pending.extend(child for _, child in state[1])
            elif state[0] == "list":
                values[coordinate] = []
                pending.extend(state[1])
            else:
                values[coordinate] = state[1]
        for coordinate, state in states.items():
            value = values[coordinate]
            if state[0] == "object":
                assert isinstance(value, dict)
                value.update((key, values[child]) for key, child in cast(list[tuple[str, int]], state[1]))
            elif state[0] == "list":
                assert isinstance(value, list)
                value.extend(values[child] for child in cast(list[int], state[1]))
        schema = values[selected]
        if not isinstance(schema, dict):
            raise ValueError("generated record lost its selected schema")
        return schema

    def record_length(self, record: GeneratedRecord) -> int:
        cursor = self._record_owner(record).execute(
            "SELECT count FROM generated_records WHERE record_id=?", (record.record_id,)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError("generated record lost its original coordinate")
        return int(row[0])

    def record_item(self, record: GeneratedRecord, key: str) -> GeneratedValue:
        cursor = self._record_owner(record).execute(
            "SELECT value_json FROM generated_record_fields WHERE record_id=? AND key=?", (record.record_id, key)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise KeyError(key)
        return _value_from_state(json.loads(row[0]), self)

    def set_record_item(self, record: GeneratedRecord, key: str, value: GeneratedValue) -> None:
        if not isinstance(key, str):
            raise TypeError("generated object keys must be strings")
        connection = self._record_owner(record)
        state = json.dumps(_value_state(value, self), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        cursor = connection.execute(
            "UPDATE generated_record_fields SET value_json=? WHERE record_id=? AND key=?",
            (state, record.record_id, key),
        )
        existed = cursor.rowcount > 0
        cursor.close()
        if existed:
            return
        connection.execute(
            "INSERT INTO generated_record_fields SELECT record_id, ?, next_ordinal, ? "
            "FROM generated_records WHERE record_id=?",
            (key, state, record.record_id),
        ).close()
        connection.execute(
            "UPDATE generated_records SET count=count+1,next_ordinal=next_ordinal+1 WHERE record_id=?",
            (record.record_id,),
        ).close()

    def delete_record_item(self, record: GeneratedRecord, key: str) -> None:
        connection = self._record_owner(record)
        cursor = connection.execute(
            "DELETE FROM generated_record_fields WHERE record_id=? AND key=?", (record.record_id, key)
        )
        deleted = cursor.rowcount > 0
        cursor.close()
        if not deleted:
            raise KeyError(key)
        connection.execute("UPDATE generated_records SET count=count-1 WHERE record_id=?", (record.record_id,)).close()

    def iter_record_keys(self, record: GeneratedRecord) -> Generator[str, None, None]:
        ordinal = -1
        while True:
            cursor = self._record_owner(record).execute(
                "SELECT ordinal,key FROM generated_record_fields WHERE record_id=? AND ordinal>? ORDER BY ordinal LIMIT 1",
                (record.record_id, ordinal),
            )
            try:
                row = cursor.fetchone()
            finally:
                cursor.close()
            if row is None:
                return
            ordinal = int(row[0])
            yield str(row[1])

    def _array_revision(self) -> int:
        revision = self._next_array_revision
        self._next_array_revision += 1
        return revision

    def append_array_item(self, array: GeneratedArray, value: GeneratedValue) -> None:
        if array.owner is not self:
            raise ValueError("generated array belongs to another relation owner")
        connection = self._array_owner(array.generation)
        ordinal = len(array)
        state = json.dumps(_value_state(value, self), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        connection.execute(
            "INSERT INTO generated_array_items VALUES (?, ?, ?, ?)",
            (array.array_id, ordinal, state, self._array_revision()),
        ).close()
        connection.execute("UPDATE generated_arrays SET count=count+1 WHERE array_id=?", (array.array_id,)).close()

    def array_item(self, array_id: int, generation: int, index: int) -> GeneratedValue:
        return self.borrow_array_item(array_id, generation, index)[0]

    def borrow_array_item(self, array_id: int, generation: int, index: int) -> tuple[GeneratedValue, int]:
        cursor = self._array_owner(generation).execute(
            "SELECT value_json, revision FROM generated_array_items WHERE array_id=? AND ordinal=?", (array_id, index)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise IndexError(index)
        return _value_from_state(json.loads(row[0]), self), int(row[1])

    def replace_array_item(self, array_id: int, generation: int, index: int, value: GeneratedValue) -> None:
        connection = self._array_owner(generation)
        state = json.dumps(_value_state(value, self), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        # Iteration borrows mutable child records. Preserve their canonical wire
        # adaptation, but do not issue a write for an unchanged read step.
        connection.execute(
            "UPDATE generated_array_items SET value_json=?, revision=? "
            "WHERE array_id=? AND ordinal=? AND value_json != ?",
            (state, self._array_revision(), array_id, index, state),
        ).close()

    def settle_array_item(
        self, array_id: int, generation: int, index: int, value: GeneratedValue, revision: int
    ) -> None:
        connection = self._array_owner(generation)
        state = json.dumps(_value_state(value, self), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        # An explicit replacement during this iteration owns the new child.
        # Settling the old borrowed record must not overwrite that replacement.
        connection.execute(
            "UPDATE generated_array_items SET value_json=?, revision=? "
            "WHERE array_id=? AND ordinal=? AND revision=? AND value_json != ?",
            (state, self._array_revision(), array_id, index, revision, state),
        ).close()

    def insert_array_item(self, array: GeneratedArray, index: int, value: GeneratedValue) -> None:
        if array.owner is not self:
            raise ValueError("generated array belongs to another relation owner")
        connection = self._array_owner(array.generation)
        length = len(array)
        index = min(max(0, index if index >= 0 else length + index), length)
        # Descending moves preserve the original unique ordinal relation.
        for ordinal in range(length - 1, index - 1, -1):
            connection.execute(
                "UPDATE generated_array_items SET ordinal=ordinal+1, revision=? WHERE array_id=? AND ordinal=?",
                (self._array_revision(), array.array_id, ordinal),
            ).close()
        state = json.dumps(_value_state(value, self), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        connection.execute(
            "INSERT INTO generated_array_items VALUES (?, ?, ?, ?)",
            (array.array_id, index, state, self._array_revision()),
        ).close()
        connection.execute("UPDATE generated_arrays SET count=count+1 WHERE array_id=?", (array.array_id,)).close()

    def sorted_numeric_array(self, array: GeneratedArray) -> GeneratedArray:
        connection = self._array_owner(array.generation)
        result = self.new_array()
        cursor = connection.execute(
            "SELECT value_json FROM generated_array_items WHERE array_id=? "
            "ORDER BY value_json COLLATE generated_numeric_order, ordinal",
            (array.array_id,),
        )
        count = 0
        try:
            for row in cursor:
                connection.execute(
                    "INSERT INTO generated_array_items VALUES (?, ?, ?, ?)",
                    (result.array_id, count, row[0], self._array_revision()),
                ).close()
                count += 1
        finally:
            cursor.close()
        connection.execute("UPDATE generated_arrays SET count=? WHERE array_id=?", (count, result.array_id)).close()
        return result

    def advance_scope(self) -> None:
        """Invalidate settled session operands while retaining this physical cohort owner."""
        if self._connection is not None:
            # These are the original scope-owned relations declared above.
            # Prefix cells have been physically retained before this boundary;
            # no old GeneratedRecord/Array may read their recycled coordinates.
            for table in (
                "generated_ids",
                "generated_arrays",
                "generated_array_items",
                "generated_path_layers",
                "generated_record_ids",
                "generated_records",
                "generated_record_fields",
                "generated_tool_results",
                "generated_persisted_tool_names",
                "generated_tool_identities",
                "generated_tool_reference_events",
                "generated_adjuncts",
                "generated_agent_children",
                "generated_agent_names",
                "generated_identity_shapes",
                "generated_structured_adjuncts",
                "generated_structured_adjunct_chunks",
            ):
                self._connection.execute("DELETE FROM " + table).close()
            self._generation += 1
            self._next_array_id = 0
            self._next_array_revision = 0
        self._counts.clear()
        self._missing_tool_counts.clear()

    def retain_session_prefix(
        self,
        native_id: str,
        *,
        parent_prefix: int | None,
        inherited_messages: int,
        own_messages: int,
        record_count: int,
    ) -> int:
        """Enroll one original prefix on this cohort owner, without hydrating ancestors."""
        if not native_id or any(
            type(value) is not int or value < 0 for value in (inherited_messages, own_messages, record_count)
        ):
            raise ValueError("original session prefix coordinates are malformed")
        connection = self._owner()
        depth = 0
        if parent_prefix is not None:
            cursor = connection.execute(
                "SELECT inherited_messages+own_messages,depth FROM generated_session_prefixes WHERE prefix_id=? AND completed=1",
                (parent_prefix,),
            )
            try:
                parent = cursor.fetchone()
            finally:
                cursor.close()
            if parent is None or inherited_messages > parent[0]:
                raise ValueError("selected inherited prefix exceeds its original parent")
            depth = parent[1] + 1
        elif inherited_messages:
            raise ValueError("inherited message bytes require their original parent")
        cursor = connection.execute(
            "INSERT INTO generated_session_prefixes(native_id,parent_prefix,inherited_messages,own_messages,record_count,depth) "
            "VALUES (?,?,?,?,?,?) RETURNING prefix_id",
            (native_id, parent_prefix, inherited_messages, own_messages, record_count, depth),
        )
        try:
            return int(cursor.fetchone()[0])
        finally:
            cursor.close()

    def retain_prefix_chunk(self, prefix_id: int, message: int, ordinal: int, data: bytes) -> None:
        self._live_tool_owner().execute(
            "INSERT INTO generated_session_prefix_chunks VALUES (?,?,?,?)", (prefix_id, message, ordinal, data)
        ).close()

    def retain_prefix_tool(
        self,
        prefix_id: int,
        event: str,
        message: int,
        ordinal: int,
        identity: GeneratedString,
        *,
        schema: SchemaRecord | None = None,
        result_mode: str = "",
        tool_class: str = "",
    ) -> None:
        if event not in {"call", "result"}:
            raise ValueError("original prefix tool event is malformed")
        self._live_tool_owner().execute(
            "INSERT INTO generated_session_prefix_tools VALUES (?,?,?,?,?,?,?,?)",
            (
                prefix_id,
                event,
                message,
                ordinal,
                json.dumps(_string_state(identity), ensure_ascii=True, separators=(",", ":")),
                json.dumps(schema, ensure_ascii=True, separators=(",", ":")) if schema is not None else None,
                result_mode,
                tool_class,
            ),
        ).close()

    def complete_session_prefix(self, prefix_id: int) -> None:
        connection = self._live_tool_owner()
        cursor = connection.execute(
            "SELECT own_messages,completed FROM generated_session_prefixes WHERE prefix_id=?", (prefix_id,)
        )
        try:
            prefix = cursor.fetchone()
        finally:
            cursor.close()
        if prefix is None or prefix[1]:
            raise ValueError("original prefix is absent or already completed")
        cursor = connection.execute(
            "SELECT COUNT(*),MIN(message_ordinal),MAX(message_ordinal),SUM(incomplete) FROM ("
            "SELECT message_ordinal,(MIN(chunk_ordinal)!=0 OR MAX(chunk_ordinal)!=COUNT(*)-1) AS incomplete "
            "FROM generated_session_prefix_chunks WHERE prefix_id=? GROUP BY message_ordinal)",
            (prefix_id,),
        )
        try:
            count, first, last, incomplete = cursor.fetchone()
        finally:
            cursor.close()
        if count != prefix[0] or (count and (first != 0 or last != count - 1 or incomplete)):
            raise ValueError("original prefix byte cells are incomplete")
        cursor = connection.execute(
            "SELECT COALESCE(MAX(selection_ordinal)+1,0) FROM generated_session_prefixes "
            "WHERE completed=1 AND record_count=(SELECT record_count FROM generated_session_prefixes WHERE prefix_id=?) "
            "AND inherited_messages+own_messages=(SELECT inherited_messages+own_messages "
            "FROM generated_session_prefixes WHERE prefix_id=?)",
            (prefix_id, prefix_id),
        )
        try:
            ordinal = cursor.fetchone()[0]
        finally:
            cursor.close()
        connection.execute(
            "UPDATE generated_session_prefixes SET completed=1,selection_ordinal=? WHERE prefix_id=?",
            (ordinal, prefix_id),
        ).close()

    def _require_prefix_read(self, prefix_id: int, messages: int) -> sqlite3.Connection:
        connection = self._live_tool_owner()
        if type(prefix_id) is not int or type(messages) is not int or messages < 0:
            raise ValueError("original prefix read coordinates are malformed")
        cursor = connection.execute(
            "SELECT inherited_messages+own_messages FROM generated_session_prefixes WHERE prefix_id=? AND completed=1",
            (prefix_id,),
        )
        try:
            prefix = cursor.fetchone()
        finally:
            cursor.close()
        if prefix is None or messages > prefix[0]:
            raise ValueError("selected prefix lacks its original completed bytes")
        return connection

    def _prefix_chain_sql(self) -> str:
        return (
            "WITH RECURSIVE chain(prefix_id,wanted,depth) AS ("
            "SELECT prefix_id,?,depth FROM generated_session_prefixes WHERE prefix_id=? "
            "UNION ALL SELECT parent.prefix_id,MIN(chain.wanted,child.inherited_messages),parent.depth "
            "FROM chain JOIN generated_session_prefixes child USING(prefix_id) "
            "JOIN generated_session_prefixes parent ON parent.prefix_id=child.parent_prefix "
            "WHERE MIN(chain.wanted,child.inherited_messages)>0) "
        )

    def iter_prefix_chunks(self, prefix_id: int, messages: int) -> Generator[bytes, None, None]:
        connection = self._require_prefix_read(prefix_id, messages)
        cursor = connection.execute(
            self._prefix_chain_sql()
            + "SELECT chunks.data FROM chain JOIN generated_session_prefixes prefix USING(prefix_id) "
            "JOIN generated_session_prefix_chunks chunks USING(prefix_id) "
            "WHERE chunks.message_ordinal<MAX(0,chain.wanted-prefix.inherited_messages) "
            "ORDER BY chain.depth,chunks.message_ordinal,chunks.chunk_ordinal",
            (messages, prefix_id),
        )
        try:
            while True:
                if self._connection is not connection:
                    raise RuntimeError("prefix bytes lost their original physical cohort owner")
                row = cursor.fetchone()
                if row is None:
                    return
                yield bytes(row[0])
        finally:
            if self._connection is connection:
                cursor.close()

    def iter_prefix_tools(self, prefix_id: int, messages: int, event: str) -> Generator[GeneratedString, None, None]:
        connection = self._require_prefix_read(prefix_id, messages)
        cursor = connection.execute(
            self._prefix_chain_sql()
            + "SELECT tools.identity FROM chain JOIN generated_session_prefixes prefix USING(prefix_id) "
            "JOIN generated_session_prefix_tools tools USING(prefix_id) "
            "WHERE tools.event=? AND tools.message_ordinal<MAX(0,chain.wanted-prefix.inherited_messages) "
            "ORDER BY chain.depth,tools.message_ordinal,tools.ordinal",
            (messages, prefix_id, event),
        )
        try:
            while True:
                if self._connection is not connection:
                    raise RuntimeError("prefix identities lost their original physical cohort owner")
                row = cursor.fetchone()
                if row is None:
                    return
                yield _string_from_state(json.loads(row[0]))
        finally:
            if self._connection is connection:
                cursor.close()

    def restore_prefix_tool_context(self, prefix_id: int, messages: int) -> None:
        """Replay exact inherited identity events through the original relation owner."""
        connection = self._require_prefix_read(prefix_id, messages)
        cursor = connection.execute(
            self._prefix_chain_sql()
            + "SELECT tools.identity,tools.event,tools.schema_json,tools.result_mode,tools.tool_class FROM chain "
            "JOIN generated_session_prefixes prefix USING(prefix_id) "
            "JOIN generated_session_prefix_tools tools USING(prefix_id) "
            "WHERE tools.message_ordinal<MAX(0,chain.wanted-prefix.inherited_messages) "
            "ORDER BY chain.depth,tools.message_ordinal,tools.ordinal",
            (messages, prefix_id),
        )
        try:
            for identity_state, event, schema_json, mode, tool_class in cursor:
                identity = _string_from_state(json.loads(identity_state))
                if self._connection is not connection:
                    raise RuntimeError("prefix tool context lost its original physical owner")
                if schema_json is None:
                    raise ValueError("original prefix identity field schema was not retained")
                state = json.dumps(_string_state(identity), separators=(",", ":"), ensure_ascii=True)
                before = connection.execute(
                    "SELECT calls,results FROM generated_tool_identities WHERE value_json=?", (state,)
                )
                try:
                    previous = before.fetchone()
                finally:
                    before.close()
                calls, results = (0, 0) if previous is None else previous
                relation = "both" if calls and results else "call" if calls else "result" if results else "new"
                self.observe_tool_identity(identity, event, relation, json.loads(schema_json), mode, tool_class)
        finally:
            if self._connection is connection:
                cursor.close()

    def close(self) -> None:
        if self._connection is not None:
            assert self._sql_owner is not None
            self._sql_owner.close()
            self._sql_owner = None
            self._connection = None
            self._generation += 1
            self._next_array_id = 0
            self._next_array_revision = 0
        self._counts.clear()
        self._missing_tool_counts.clear()
        self._captured_schema_nodes.clear()
        self._captured_schema_values.clear()

    def __del__(self) -> None:
        # Explicit generation scopes close this owner. A standalone solver
        # abandoned by a caller still releases its private temporary file.
        # The native census retains failed cleanup for the original creator.
        with suppress(NativeConnectionSettlementError):
            self.close()


def _string_state(value: GeneratedString) -> list[object]:
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, SampledText):
        return ["sampled", value.length, value.pool, value.offset, value.newlines, value.newline_phase]
    return ["joined", [_string_state(part) for part in value.parts]]


def _string_from_state(value: object) -> GeneratedString:
    if not isinstance(value, list) or not value:
        raise ValueError("generated foreign-key spelling recipe is malformed")
    if value[0] == "text" and len(value) == 2 and isinstance(value[1], str):
        return value[1]
    if value[0] == "sampled" and len(value) == 6 and isinstance(value[2], str):
        if any(type(value[index]) is not int for index in (1, 3, 4, 5)):
            raise ValueError("generated text coordinates are malformed")
        return SampledText(value[1], value[2], value[3], value[4], value[5])
    if value[0] == "joined" and len(value) == 2 and isinstance(value[1], list):
        return JoinedText(tuple(_string_from_state(part) for part in value[1]))
    raise ValueError("generated foreign-key spelling recipe has an unknown shape")


def _compare_numeric_states(first: str, second: str) -> int:
    left, right = json.loads(first)[0][1], json.loads(second)[0][1]
    return int(left > right) - int(left < right)


def _value_state(value: GeneratedValue, owner: ForeignKeyGraph) -> list[list[object]]:
    # Flat coordinates keep the private custody encoding independent of the
    # Python JSON encoder's recursion depth. Array children stay native views.
    nodes: list[list[object]] = []
    pending: list[tuple[GeneratedValue, int]] = []
    containers: dict[int, int] = {}

    def reserve(child: GeneratedValue) -> int:
        if isinstance(child, dict | list) and id(child) in containers:
            return containers[id(child)]
        index = len(nodes)
        if isinstance(child, dict | list):
            containers[id(child)] = index
        nodes.append([])
        pending.append((child, index))
        return index

    reserve(value)
    while pending:
        child, index = pending.pop()
        if isinstance(child, GeneratedArray):
            if child.owner is not owner:
                raise ValueError("array child belongs to another relation owner")
            owner._array_owner(child.generation)
            nodes[index] = ["array", child.array_id, child.generation]
        elif isinstance(child, SampledText | JoinedText):
            nodes[index] = ["text", _string_state(child)]
        elif isinstance(child, GeneratedRecord):
            owner._record_owner(child)
            nodes[index] = ["record", child.record_id, child.generation]
        elif isinstance(child, dict):
            entries = [[key, reserve(item)] for key, item in child.items()]
            nodes[index] = ["object", entries]
        elif isinstance(child, list):
            nodes[index] = ["list", [reserve(item) for item in child]]
        else:
            nodes[index] = ["scalar", child]
    return nodes


def _value_from_state(state: object, owner: ForeignKeyGraph) -> GeneratedValue:
    if not isinstance(state, list) or not state:
        raise ValueError("generated array child recipe is malformed")
    values: list[GeneratedValue] = []
    for node in state:
        if not isinstance(node, list) or not node:
            raise ValueError("generated child coordinate is malformed")
        tag = node[0]
        if tag == "scalar" and len(node) == 2 and (node[1] is None or isinstance(node[1], str | int | float | bool)):
            values.append(node[1])
        elif tag == "text" and len(node) == 2:
            values.append(_string_from_state(node[1]))
        elif tag == "array" and len(node) == 3 and type(node[1]) is int and type(node[2]) is int:
            owner._array_owner(node[2])
            values.append(GeneratedArray(owner, node[1], node[2]))
        elif tag == "list" and len(node) == 2:
            values.append([])
        elif tag == "object" and len(node) == 2:
            values.append({})
        elif tag == "record" and len(node) == 3 and type(node[1]) is int and type(node[2]) is int:
            owner._array_owner(node[2])
            values.append(GeneratedRecord(owner, node[1], node[2]))
        else:
            raise ValueError("generated child recipe has an unknown shape")

    def referenced(index: object) -> GeneratedValue:
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError("generated child coordinate is missing")
        return values[index]

    for index, node in enumerate(state):
        value = values[index]
        if node[0] == "list":
            if not isinstance(node[1], list) or not isinstance(value, list):
                raise ValueError("generated list recipe is malformed")
            value.extend(referenced(child) for child in node[1])
        elif node[0] == "object":
            entries = node[1]
            if not isinstance(entries, list) or not isinstance(value, dict):
                raise ValueError("generated record recipe is malformed")
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                    raise ValueError("generated object child recipe is malformed")
                value[entry[0]] = referenced(entry[1])
    return values[0]


@dataclass
class TimeDeltaConstraint:
    """Constraint between two timestamp fields."""

    field_a: str
    field_b: str
    min_delta: float
    max_delta: float
    avg_delta: float

    @property
    def stddev_approx(self) -> float:
        """Approximate stddev from the delta range."""
        return (self.max_delta - self.min_delta) / 4.0


@dataclass
class MutualExclusionGroup:
    """Group of fields that must not co-occur."""

    parent_path: str
    field_names: frozenset[str]


class RelationConstraintSolver(RelationConstraintSolverRuntimeMixin):
    """Applies relational constraints from schema annotations during generation.

    Constructed once per corpus from the schema's root-level relational
    annotations.  Methods are called during object generation to enforce
    consistency.
    """

    def __init__(self, schema: SchemaRecord) -> None:
        self.fk_graph = ForeignKeyGraph()
        self.time_deltas: list[TimeDeltaConstraint] = []
        self.mutual_exclusions: list[MutualExclusionGroup] = []
        self.mutual_exclusions_by_parent: dict[str, tuple[MutualExclusionGroup, ...]] = {}
        self._time_delta_cls = TimeDeltaConstraint
        self._mutual_exclusion_cls = MutualExclusionGroup

        self._parse_foreign_keys(schema)
        self._parse_time_deltas(schema)
        self._parse_mutual_exclusions(schema)

    def begin_scope(self, schema: SchemaRecord) -> None:
        """Compile the selected element on the same settled physical relation owner."""
        self.fk_graph.advance_scope()
        self.select_element(schema)

    def select_element(self, schema: SchemaRecord) -> None:
        """Compile an original child element while retaining parent native cells."""
        self.fk_graph.references.clear()
        self.time_deltas.clear()
        self.mutual_exclusions.clear()
        self.mutual_exclusions_by_parent.clear()
        self._parse_foreign_keys(schema)
        self._parse_time_deltas(schema)
        self._parse_mutual_exclusions(schema)

    def close(self) -> None:
        self.fk_graph.close()

    @property
    def has_constraints(self) -> bool:
        """Whether any relational constraints were parsed."""
        return bool(self.fk_graph.references or self.time_deltas or self.mutual_exclusions)


__all__ = [
    "ForeignKeyGraph",
    "MutualExclusionGroup",
    "RelationConstraintSolver",
    "TimeDeltaConstraint",
]
