"""Privacy-safe workload-profile artifacts derived from schema evidence."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from posixpath import basename

from polylogue.core.claude_code_children import (
    claude_agent_id_from_path,
    claude_agent_identity_operand,
    claude_dispatch_operand,
)
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.timestamps import parse_timestamp
from polylogue.core.tool_result_pointer import (
    PERSISTED_OUTPUT_POINTER_PATTERN,
    persisted_output_shape,
    pointer_width_relation,
)
from polylogue.schemas.field_stats.distributions import DistributionSketch, _bucket_index
from polylogue.schemas.field_stats.models import FilenameWidthStats, PointerShapeStats
from polylogue.schemas.generation.models import _PackageAccumulator, _UnitMembership
from polylogue.schemas.generation.observation_journal import JournalMemberships, ObservationJournal
from polylogue.schemas.generation.packages import _package_bundle_scope_count
from polylogue.schemas.record_structure import (
    NATIVE_CODEX_TOOL_VALUE_FIELDS,
    PUBLIC_CLAUDE_TOOL_INPUT_FIELDS,
    PUBLIC_CLAUDE_TOOL_NAMES,
    PUBLIC_CLAUDE_TOOL_RESULT_FIELDS,
    PUBLIC_CONTENT_BODY_FIELDS,
    claude_agent_metadata_field_forms,
    claude_dispatch_field_forms,
    claude_todo_token_counts,
    codex_message_record,
    codex_parent_field_forms,
    codex_session_header,
    content_prefix_kind,
    content_representation,
    initial_content_prefix,
    iter_provider_records,
    iter_public_content_kinds,
    iter_tool_records,
    public_role_carriers,
    record_content,
    record_form,
    record_kind,
    record_timestamp,
    tool_identity_operand,
    tool_result_content_mode,
)
from polylogue.schemas.tool_input_syntax import measured_tool_syntax
from polylogue.schemas.workload_profile_rows import DepthGrammars, WorkloadProfile, workload_profile_digest
from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

WORKLOAD_PROFILE_VERSION = 1


def workload_profile_identity(profile: Mapping[str, object]) -> str:
    """Return a deterministic identity for a content-only workload profile."""
    return workload_profile_digest(profile)


_PARENT_FIELD_NAMES = frozenset(
    {
        "parentUuid",
        "parent_id",
        "parentId",
        "parentSessionId",
        "parent_session_id",
        "branch_from_id",
    }
)


def require_cross_record_profile(profile: Mapping[str, object]) -> None:
    """Reject the retired duplicate field-distribution representation."""
    elements = profile.get("elements")
    if isinstance(elements, Mapping):
        for element in elements.values():
            if isinstance(element, Mapping):
                if "field_profiles" in element:
                    raise ValueError("Package workload profiles cannot own field distributions; re-infer the package")
                if "structural_variants" in element:
                    raise ValueError("Private structural token profiles are retired; re-infer the package")


def _text_token(value: object) -> str:
    return value.lower() if isinstance(value, str) else ""


def _first_text(record: Mapping[str, object], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class _ToolScope:
    connection: sqlite3.Connection
    call_count: int = 0
    result_count: int = 0
    call_without_id: int = 0
    result_without_id: int = 0
    error_results: int = 0
    functions_exec_calls: int = 0

    def _record_id(
        self, kind: int, tool_id: str, position: int, reference_mode: str | None = None, pointer_bucket: int = -1
    ) -> None:
        # BLOB identity preserves escaped surrogates and exact spelling. This
        # is private inference scratch, never a published vocabulary.
        self.connection.execute(
            "INSERT INTO workload_tool_events(tool_id,kind,position,reference_mode,pointer_bucket) VALUES (?,?,?,?,?)",
            (tool_id.encode("utf-8", "surrogatepass"), kind, position, reference_mode, pointer_bucket),
        )

    def observe(self, record: Mapping[str, object], position: int) -> None:
        kind = _text_token(record.get("type")) or _text_token(record.get("kind"))
        name = _first_text(record, ("name", "recipient_name", "tool_name"))
        namespace = _first_text(record, ("namespace",))
        is_exec = (
            name == "functions.exec"
            or (namespace == "functions" and name == "exec")
            or (kind == "custom_tool_call" and name == "exec")
        )
        is_result = kind in {
            "tool_result",
            "function_call_output",
            "tool_call_output",
            "function_result",
            "custom_tool_call_output",
        }
        is_call = is_exec or kind in {"tool_use", "function_call", "tool_call", "function", "custom_tool_call"}
        if not (is_call or is_result):
            return
        if is_exec:
            self.functions_exec_calls += 1
        if is_result:
            self.result_count += 1
            tool_id = _first_text(record, ("tool_use_id", "call_id", "tool_call_id", "id"))
            if tool_id is None:
                self.result_without_id += 1
            else:
                content = record.get("content")
                pointer = persisted_output_shape(content) if isinstance(content, str) else None
                self._record_id(
                    1,
                    tool_id,
                    position,
                    tool_result_content_mode(record),
                    _bucket_index(float(pointer[5])) if pointer is not None else -1,
                )
            exit_code = record.get("exit_code")
            if (
                record.get("is_error") is True
                or record.get("tool_result_is_error") is True
                or _text_token(record.get("status")) in {"error", "failed", "failure"}
                or (isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0)
            ):
                self.error_results += 1
            return
        self.call_count += 1
        tool_id = _first_text(record, ("id", "call_id", "tool_call_id"))
        if tool_id is None:
            self.call_without_id += 1
        else:
            self._record_id(0, tool_id, position)


@dataclass
class _RelationshipAccumulator:
    calls: int = 0
    results: int = 0
    paired: int = 0
    missing_results: int = 0
    orphan_results: int = 0
    duplicate_calls: int = 0
    duplicate_results: int = 0
    out_of_order_results: int = 0
    call_without_id: int = 0
    result_without_id: int = 0
    error_results: int = 0
    functions_exec_calls: int = 0
    scope_record_counts: DistributionSketch = field(default_factory=DistributionSketch)
    result_distance: DistributionSketch = field(default_factory=DistributionSketch)
    parent_reference_count: int = 0
    root_parent_count: int = 0
    sidechain_true_count: int = 0
    subagent_marker_count: int = 0

    def merge_scope(self, scope: _ToolScope, record_count: int) -> None:
        self.scope_record_counts.observe(record_count)
        self.calls += scope.call_count
        self.results += scope.result_count
        self.call_without_id += scope.call_without_id
        self.result_without_id += scope.result_without_id
        self.error_results += scope.error_results
        self.functions_exec_calls += scope.functions_exec_calls
        counts = scope.connection.execute(
            "SELECT SUM(MIN(calls, results)), SUM(MAX(calls-results, 0)), "
            "SUM(MAX(results-calls, 0)), SUM(MAX(calls-1, 0)), SUM(MAX(results-1, 0)) "
            "FROM (SELECT tool_id, SUM(kind=0) AS calls, SUM(kind=1) AS results "
            "FROM workload_tool_events GROUP BY tool_id)"
        ).fetchone()
        self.paired += int(counts[0] or 0)
        self.missing_results += int(counts[1] or 0)
        self.orphan_results += int(counts[2] or 0)
        self.duplicate_calls += int(counts[3] or 0)
        self.duplicate_results += int(counts[4] or 0)
        cursor = scope.connection.execute(
            "WITH calls AS (SELECT tool_id, position, "
            "ROW_NUMBER() OVER (PARTITION BY tool_id ORDER BY position, rowid) AS occurrence "
            "FROM workload_tool_events WHERE kind=0), "
            "results AS (SELECT tool_id, position, "
            "ROW_NUMBER() OVER (PARTITION BY tool_id ORDER BY position, rowid) AS occurrence "
            "FROM workload_tool_events WHERE kind=1) "
            "SELECT results.position-calls.position AS distance FROM calls "
            "JOIN results USING(tool_id, occurrence)"
        )
        try:
            for row in cursor:
                distance = int(row[0])
                self.result_distance.observe(distance)
                if distance < 0:
                    self.out_of_order_results += 1
        finally:
            cursor.close()
        scope.connection.execute("DELETE FROM workload_tool_events")

    def observe_lineage(self, record: Mapping[str, object]) -> None:
        for name in _PARENT_FIELD_NAMES:
            if name not in record:
                continue
            if record[name] is None or record[name] == "":
                self.root_parent_count += 1
            else:
                self.parent_reference_count += 1
        if record.get("isSidechain") is True or record.get("is_sidechain") is True:
            self.sidechain_true_count += 1
        kind = _text_token(record.get("type")) or _text_token(record.get("kind"))
        if "subagent" in kind or ("agent" in kind and "spawn" in kind):
            self.subagent_marker_count += 1

    def to_payload(self) -> JSONDocument:
        return json_document(
            {
                "tool_results": {
                    "calls": self.calls,
                    "results": self.results,
                    "paired": self.paired,
                    "missing": self.missing_results,
                    "orphan": self.orphan_results,
                    "duplicate_calls": self.duplicate_calls,
                    "duplicate_results": self.duplicate_results,
                    "out_of_order": self.out_of_order_results,
                    "call_without_id": self.call_without_id,
                    "result_without_id": self.result_without_id,
                    "error_results": self.error_results,
                    "functions_exec_calls": self.functions_exec_calls,
                    "result_record_distance": self.result_distance.to_payload(),
                },
                "lineage": {
                    "parent_references": self.parent_reference_count,
                    "root_parent_values": self.root_parent_count,
                    "sidechain_true": self.sidechain_true_count,
                    "subagent_markers": self.subagent_marker_count,
                },
                "scope_record_count": self.scope_record_counts.to_payload(),
            }
        )


@contextmanager
def _workload_memberships(memberships: Sequence[_UnitMembership]) -> Iterator[JournalMemberships]:
    """Use the original journal, or spool an explicitly supplied fixture once."""
    if isinstance(memberships, JournalMemberships):
        yield memberships
        return
    with ObservationJournal.create() as journal:
        for membership in memberships:
            unit_id = journal.append_unit(membership.unit, retain_cluster_payload=False)
            journal.assign_profile_family(unit_id, membership.profile_family_id)
        yield journal.memberships()


def _relationship_profile(provider: str, memberships: JournalMemberships) -> JSONDocument:
    aggregate = _RelationshipAccumulator()
    connection = memberships._journal._connection
    connection.execute(
        "CREATE TEMP TABLE workload_tool_events (tool_id BLOB NOT NULL, "
        "kind INTEGER NOT NULL, position INTEGER NOT NULL, reference_mode TEXT, pointer_bucket INTEGER NOT NULL) STRICT"
    )
    reference_owners = ExitStack()
    try:
        connection.execute(
            "CREATE TEMP TABLE workload_opaque_reference_files(tool_id BLOB NOT NULL, kind TEXT NOT NULL, "
            "ordinal INTEGER NOT NULL,base_width INTEGER NOT NULL,suffix_width INTEGER NOT NULL) STRICT"
        )
        connection.execute(
            "CREATE INDEX workload_opaque_reference_identity ON workload_opaque_reference_files(tool_id, ordinal)"
        )
        connection.execute(
            "CREATE INDEX workload_tool_events_identity ON workload_tool_events(tool_id, kind, position)"
        )
        current_scope: str | None = None
        tool_scope = _ToolScope(connection)
        reference_index = (
            reference_owners.enter_context(ToolResultIndexAccumulator(disk_backed=True))
            if provider == "claude-code"
            else None
        )
        reference_lengths = {
            kind: DistributionSketch()
            for kind in ("direct_inline", "direct_truncated", "persisted_inline", "persisted_truncated", "orphan")
        }
        reference_counts = {kind: DistributionSketch() for kind in reference_lengths}
        reference_widths = {kind: DistributionSketch() for kind in reference_lengths}
        per_result: dict[str, _OpaqueReferenceStructure] = {}

        def finish_references(scope: str) -> None:
            if reference_index is None:
                return
            counts: Counter[str] = Counter()
            with closing(memberships.iter_payload_references(scope)) as payloads:
                for ordinal, (filename, byte_count) in enumerate(payloads):
                    match = reference_index.reference_for_filename(filename)
                    kind = (
                        "orphan"
                        if match is None
                        else match.reference_kind + ("_truncated" if match.was_truncated else "_inline")
                    )
                    counts[kind] += 1
                    reference_lengths[kind].observe(byte_count)
                    reference_widths[kind].observe(len(filename))
                    if match is not None:
                        connection.execute(
                            "INSERT INTO workload_opaque_reference_files VALUES (?, ?, ?, ?, ?)",
                            (
                                match.tool_use_id.encode("utf-8", "surrogatepass"),
                                kind,
                                ordinal,
                                len(match.tool_use_id)
                                if match.reference_kind == "direct"
                                else len(filename)
                                if reference_index._by_persisted_name.get(filename) == match.tool_use_id
                                else len(filename.rsplit(".", 1)[0]),
                                len(filename) - len(match.tool_use_id)
                                if match.reference_kind == "direct"
                                else 0
                                if reference_index._by_persisted_name.get(filename) == match.tool_use_id
                                else len(filename) - len(filename.rsplit(".", 1)[0]),
                            ),
                        )
            for kind in reference_counts:
                reference_counts[kind].observe(counts[kind])
            with closing(reference_index.iter_tool_result_references()) as references:
                for reference in references:
                    mode = reference.content_representation + (":truncated" if reference.was_truncated else ":inline")
                    structure = per_result.setdefault(mode, _OpaqueReferenceStructure())
                    cursor = connection.execute(
                        "SELECT kind FROM workload_opaque_reference_files WHERE tool_id = ? ORDER BY ordinal",
                        (reference.tool_use_id.encode("utf-8", "surrogatepass"),),
                    )
                    try:
                        presence = structure.observe(
                            (str(row[0]) for row in cursor), inline_length=reference.inline_length
                        )
                    finally:
                        cursor.close()
                    cursor = connection.execute(
                        "SELECT kind,base_width,suffix_width FROM workload_opaque_reference_files WHERE tool_id=? ORDER BY ordinal",
                        (reference.tool_use_id.encode("utf-8", "surrogatepass"),),
                    )
                    try:
                        for file_kind, base_width, suffix_width in cursor:
                            structure.filename_forms.setdefault(
                                (str(file_kind), _bucket_index(float(base_width))), DistributionSketch()
                            ).observe(int(suffix_width))
                    finally:
                        cursor.close()
                    # The native insertion coordinate breaks ties between
                    # several blocks in one original record. No public ID or
                    # filename spelling leaves this original inference owner.
                    cursor = connection.execute(
                        "SELECT reference_mode FROM workload_tool_events WHERE tool_id=? AND kind=1 "
                        "ORDER BY position,rowid",
                        (reference.tool_use_id.encode("utf-8", "surrogatepass"),),
                    )
                    try:
                        history = structure.histories.setdefault(presence, _OpaqueResultHistoryStructure())
                        file_count = int(
                            connection.execute(
                                "SELECT count(*) FROM workload_opaque_reference_files WHERE tool_id=?",
                                (reference.tool_use_id.encode("utf-8", "surrogatepass"),),
                            ).fetchone()[0]
                        )
                        history.observe(
                            (str(row[0]) for row in cursor),
                            terminal_mode=mode,
                            file_count=file_count,
                            identity_width=len(reference.tool_use_id),
                            pointer_buckets=tuple(
                                int(row[0])
                                for row in connection.execute(
                                    "SELECT DISTINCT pointer_bucket FROM workload_tool_events WHERE tool_id=? AND pointer_bucket>=0 ORDER BY pointer_bucket",
                                    (reference.tool_use_id.encode("utf-8", "surrogatepass"),),
                                )
                            ),
                        )
                    finally:
                        cursor.close()
            connection.execute("DELETE FROM workload_opaque_reference_files")

        position = 0
        record_count = 0
        with closing(memberships.iter_scope_samples(order="bundle")) as samples:
            for original in samples:
                scope, sample = original.bundle_scope, original.sample
                if current_scope is not None and scope != current_scope:
                    finish_references(current_scope)
                    aggregate.merge_scope(tool_scope, record_count)
                    reference_owners.close()
                    reference_index = (
                        reference_owners.enter_context(ToolResultIndexAccumulator(disk_backed=True))
                        if provider == "claude-code"
                        else None
                    )
                    tool_scope = _ToolScope(connection)
                    position = 0
                    record_count = 0
                current_scope = scope
                if sample is None:
                    continue
                for record in iter_provider_records(provider, sample):
                    if reference_index is not None:
                        reference_index.observe(record)
                    record_count += 1
                    aggregate.observe_lineage(record)
                    for tool_record in iter_tool_records(provider, record):
                        tool_scope.observe(tool_record, position)
                    position += 1
        if current_scope is not None:
            finish_references(current_scope)
            aggregate.merge_scope(tool_scope, record_count)
        payload = aggregate.to_payload()
        if provider == "claude-code":
            payload["opaque_tool_results"] = {
                "population": "retained-raw-revisions",
                "per_tool_result": {mode: structure.payload() for mode, structure in sorted(per_result.items())},
                "reference_classes": {
                    kind: {
                        "bytes": reference_lengths[kind].to_payload(),
                        "basename_length": reference_widths[kind].to_payload(),
                        "count_per_scope": reference_counts[kind].to_payload(),
                    }
                    for kind in reference_lengths
                },
            }
        return payload
    finally:
        reference_owners.close()
        connection.execute("DROP TABLE IF EXISTS workload_opaque_reference_files")
        connection.execute("DROP TABLE workload_tool_events")


@dataclass
class _OpaqueResultHistoryStructure:
    """Original result histories conditioned on the file reference presence."""

    counts: DistributionSketch = field(default_factory=DistributionSketch)
    starts: Counter[str] = field(default_factory=Counter)
    ends: Counter[str] = field(default_factory=Counter)
    transitions: Counter[tuple[str, str]] = field(default_factory=Counter)

    file_counts: dict[tuple[int, int, tuple[int, ...]], DistributionSketch] = field(default_factory=dict)

    def observe(
        self,
        modes: Iterator[str],
        *,
        terminal_mode: str,
        file_count: int,
        identity_width: int,
        pointer_buckets: tuple[int, ...],
    ) -> None:
        count = 0
        previous: str | None = None
        for mode in modes:
            if previous is None:
                self.starts[mode] += 1
            else:
                self.transitions[(previous, mode)] += 1
            previous = mode
            count += 1
        if previous != terminal_mode:
            raise ValueError("original result history contradicts the terminal reference shape")
        self.ends[previous] += 1
        self.counts.observe(count)
        self.file_counts.setdefault(
            (_bucket_index(float(identity_width)), _bucket_index(float(count)), pointer_buckets), DistributionSketch()
        ).observe(file_count)

    def payload(self) -> JSONDocument:
        return json_document(
            {
                "file_count_by_identity_history_and_pointers": {
                    str(width) + ":" + str(count) + ":" + ",".join(map(str, pointers)): shape.to_payload()
                    for (width, count, pointers), shape in sorted(self.file_counts.items())
                },
                "record_count": self.counts.to_payload(),
                "initial_modes": dict(sorted(self.starts.items())),
                "terminal_modes": dict(sorted(self.ends.items())),
                "transitions": [
                    {"from": before, "to": after, "count": count}
                    for (before, after), count in sorted(self.transitions.items())
                ],
            }
        )


@dataclass
class _OpaqueReferenceStructure:
    """Measured finite file-class paths conditioned on original result shape."""

    counts: DistributionSketch = field(default_factory=DistributionSketch)
    inline_lengths: DistributionSketch = field(default_factory=DistributionSketch)
    starts: Counter[str] = field(default_factory=Counter)
    ends: Counter[str] = field(default_factory=Counter)
    transitions: Counter[tuple[str, str]] = field(default_factory=Counter)
    histories: dict[str, _OpaqueResultHistoryStructure] = field(default_factory=dict)
    filename_forms: dict[tuple[str, int], DistributionSketch] = field(default_factory=dict)

    def observe(self, kinds: Iterator[str], *, inline_length: int) -> str:
        self.inline_lengths.observe(inline_length)
        count = 0
        previous: str | None = None
        direct = persisted = False
        for kind in kinds:
            direct |= kind.startswith("direct_")
            persisted |= kind.startswith("persisted_")
            if previous is None:
                self.starts[kind] += 1
            else:
                self.transitions[(previous, kind)] += 1
            previous = kind
            count += 1
        if previous is not None:
            self.ends[previous] += 1
        self.counts.observe(count)
        return (
            "direct+persisted" if direct and persisted else "direct" if direct else "persisted" if persisted else "none"
        )

    def payload(self) -> JSONDocument:
        return json_document(
            {
                "file_count": self.counts.to_payload(),
                "inline_length": self.inline_lengths.to_payload(),
                "result_history": {presence: history.payload() for presence, history in sorted(self.histories.items())},
                "filename_suffix_by_base_width": {
                    kind + ":" + str(width): shape.to_payload()
                    for (kind, width), shape in sorted(self.filename_forms.items())
                },
                "initial_classes": dict(sorted(self.starts.items())),
                "terminal_classes": dict(sorted(self.ends.items())),
                "transitions": [
                    {"from": before, "to": after, "count": count}
                    for (before, after), count in sorted(self.transitions.items())
                ],
            }
        )


class AgentChildCounts(PointerShapeStats):
    """Original transcript/metadata file counts and transcript record count."""

    axes = ("transcripts", "transcript_records", "metadata", "agent_width", "stem_width")

    @staticmethod
    def _validate_point(point: object) -> None:
        if (
            not isinstance(point, list)
            or len(point) != 5
            or any(type(value) is not int or value < 0 for value in point)
            or (point[0] == 0 and point[1] != 0)
        ):
            raise ValueError("original agent child count joint is malformed")


@contextmanager
def _agent_child_population(memberships: JournalMemberships) -> Iterator[None]:
    """Join exact private child names on the same journal; publish only counts.

    Repeated/ambiguous dispatches are retained as unresolved classes. They are
    not silently assigned to a convenient parent or duplicated on emission.
    """
    owner = memberships._journal._connection
    owner.execute(
        "CREATE TEMP TABLE workload_agent_children(scope TEXT NOT NULL,stem TEXT NOT NULL,kind TEXT NOT NULL,stream TEXT NOT NULL,records INTEGER NOT NULL,metadata_json TEXT,PRIMARY KEY(scope,stem,kind,stream)) STRICT, WITHOUT ROWID"
    ).close()
    owner.execute(
        "CREATE TEMP TABLE workload_agent_dispatches(scope TEXT NOT NULL,stem TEXT NOT NULL,stream TEXT NOT NULL,position INTEGER NOT NULL,PRIMARY KEY(scope,stream,position)) STRICT, WITHOUT ROWID"
    ).close()
    owner.execute(
        "CREATE TEMP TABLE workload_agent_depths(scope TEXT NOT NULL,stream TEXT NOT NULL,depth INTEGER,PRIMARY KEY(scope,stream)) STRICT, WITHOUT ROWID"
    ).close()
    try:
        current: tuple[str, str, str] | None = None
        position = 0
        with closing(memberships.iter_scope_samples(order="stream")) as samples:
            for original in samples:
                key = original.bundle_scope, original.artifact_kind, original.stream_key
                if key != current:
                    current, position = key, 0
                if original.artifact_kind == "session_record_stream":
                    owner.execute(
                        "INSERT INTO workload_agent_depths VALUES (?,?,0) ON CONFLICT DO NOTHING",
                        (original.bundle_scope, original.stream_key),
                    ).close()
                elif original.artifact_kind == "agent_transcript":
                    owner.execute(
                        "INSERT INTO workload_agent_depths VALUES (?,?,NULL) ON CONFLICT DO NOTHING",
                        (original.bundle_scope, original.stream_key),
                    ).close()
                if original.artifact_kind in {"agent_transcript", "agent_sidecar_meta"}:
                    stem = claude_agent_id_from_path(original.source_path) if original.source_path else None
                    if stem is not None:
                        owner.execute(
                            "INSERT INTO workload_agent_children VALUES (?,?,?,?,0,NULL) ON CONFLICT DO NOTHING",
                            (original.bundle_scope, stem, original.artifact_kind, original.stream_key),
                        ).close()
                        if original.sample is not None:
                            if original.artifact_kind == "agent_sidecar_meta":
                                identity = claude_agent_identity_operand(original.sample)
                                metadata = json.dumps(
                                    {
                                        "fields": claude_agent_metadata_field_forms(original.sample),
                                        "carrier": identity[0] if identity is not None else None,
                                        "private_value": identity[1] if identity is not None else None,
                                    },
                                    ensure_ascii=True,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                owner.execute(
                                    "UPDATE workload_agent_children SET metadata_json=? WHERE scope=? AND stem=? AND kind=? AND stream=?",
                                    (
                                        metadata,
                                        original.bundle_scope,
                                        stem,
                                        original.artifact_kind,
                                        original.stream_key,
                                    ),
                                ).close()
                            count = sum(1 for _ in iter_provider_records("claude-code", original.sample))
                            owner.execute(
                                "UPDATE workload_agent_children SET records=records+? WHERE scope=? AND stem=? AND kind=? AND stream=?",
                                (count, original.bundle_scope, stem, original.artifact_kind, original.stream_key),
                            ).close()
                if original.sample is None:
                    continue
                for record in iter_provider_records("claude-code", original.sample):
                    operand = claude_dispatch_operand(record)
                    if operand is not None:
                        owner.execute(
                            "INSERT INTO workload_agent_dispatches VALUES (?,?,?,?)",
                            (original.bundle_scope, operand[1], original.stream_key, position),
                        ).close()
                    position += 1
        # Each assignment consumes one previously unassigned physical stream.
        # Cycles, repeated dispatches and competing parents remain unassigned;
        # no arbitrary depth ceiling or private sequence is reconstructed.
        while True:
            cursor = owner.execute("""
                WITH links AS (
                    SELECT child.scope,child.stream,min(parent.depth)+1 AS depth
                    FROM workload_agent_children child
                    JOIN workload_agent_dispatches dispatch ON dispatch.scope=child.scope AND dispatch.stem=child.stem
                    LEFT JOIN workload_agent_depths parent ON parent.scope=dispatch.scope AND parent.stream=dispatch.stream
                    WHERE child.kind='agent_transcript'
                    AND (SELECT count(*) FROM workload_agent_children peers WHERE peers.scope=child.scope AND peers.stem=child.stem AND peers.kind='agent_transcript')=1
                    GROUP BY child.scope,child.stream
                    HAVING count(*)=1 AND count(parent.depth)=1
                )
                UPDATE workload_agent_depths SET depth=(SELECT depth FROM links WHERE links.scope=workload_agent_depths.scope AND links.stream=workload_agent_depths.stream)
                WHERE depth IS NULL AND EXISTS(SELECT 1 FROM links WHERE links.scope=workload_agent_depths.scope AND links.stream=workload_agent_depths.stream)
            """)
            cursor.close()
            with closing(owner.execute("SELECT changes()")) as cursor:
                changed = cursor.fetchone()
            assert changed is not None
            if not changed[0]:
                break
        yield
    finally:
        owner.execute("DROP TABLE workload_agent_depths").close()
        owner.execute("DROP TABLE workload_agent_dispatches").close()
        owner.execute("DROP TABLE workload_agent_children").close()


def _original_agent_children(
    owner: sqlite3.Connection, scope: str, record: Mapping[str, object]
) -> tuple[str, tuple[int, int, int, int, int], JSONDocument | None]:
    operand = claude_dispatch_operand(record)
    if operand is None:
        return "no_dispatch", (0, 0, 0, 0, 0), None
    stem = operand[1]
    with closing(
        owner.execute("SELECT count(*) FROM workload_agent_dispatches WHERE scope=? AND stem=?", (scope, stem))
    ) as cursor:
        row = cursor.fetchone()
    assert row is not None
    dispatches = row[0]
    with closing(
        owner.execute(
            "SELECT kind,count(*),sum(records) FROM workload_agent_children WHERE scope=? AND stem=? GROUP BY kind",
            (scope, stem),
        )
    ) as cursor:
        rows = {kind: (count, records) for kind, count, records in cursor}
    transcripts, records = rows.get("agent_transcript", (0, 0))
    metadata = rows.get("agent_sidecar_meta", (0, 0))[0]
    status = (
        "ambiguous"
        if dispatches != 1 or transcripts > 1 or metadata > 1
        else "matched"
        if transcripts
        else "missing_transcript"
    )
    result = record["toolUseResult"]
    assert isinstance(result, Mapping) and isinstance(result.get("agentId"), str)
    agent_id = result["agentId"]
    assert isinstance(agent_id, str)
    metadata_class: JSONDocument | None = None
    if metadata == 1:
        with closing(
            owner.execute(
                "SELECT records,metadata_json FROM workload_agent_children WHERE scope=? AND stem=? AND kind='agent_sidecar_meta'",
                (scope, stem),
            )
        ) as cursor:
            row = cursor.fetchone()
        assert row is not None
        if row[0] != 1 or row[1] is None:
            status = "ambiguous"
        else:
            original_metadata = json.loads(row[1])
            value = original_metadata.pop("private_value")
            original_metadata["relation"] = (
                "filename_only"
                if value is None
                else "raw_agent"
                if value == agent_id
                else "stem"
                if value == stem
                else "independent"
            )
            metadata_class = json_document(original_metadata)
    return status, (transcripts, records, metadata, len(agent_id), len(stem)), metadata_class


@dataclass
class _ContentStructure:
    role_carriers: Counter[tuple[str, str]] = field(default_factory=Counter)
    dispatch_forms: Counter[str] = field(default_factory=Counter)
    agent_counts: dict[str, AgentChildCounts] = field(default_factory=dict)
    representations: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)
    discriminator_fields: dict[str, Counter[str]] = field(default_factory=dict)
    body_field_forms: dict[str, Counter[tuple[str, ...]]] = field(default_factory=dict)
    native_tool_field_forms: dict[str, Counter[tuple[str, tuple[str, ...], str]]] = field(default_factory=dict)
    native_tool_tokens: dict[tuple[str, str, tuple[str, ...], str], dict[str, Counter[str]]] = field(
        default_factory=dict
    )
    starts: Counter[tuple[str, str]] = field(default_factory=Counter)
    ends: Counter[tuple[str, str]] = field(default_factory=Counter)
    transitions: Counter[tuple[str, str, str, str]] = field(default_factory=Counter)
    counts: DistributionSketch = field(default_factory=DistributionSketch)
    tool_operands: dict[str, Counter[tuple[str, str, str, str, str, str, int, tuple[int, int, str, str] | None]]] = (
        field(default_factory=dict)
    )

    tool_identity_widths: dict[tuple[object, ...], DistributionSketch] = field(default_factory=dict)
    tool_filename_widths: dict[tuple[object, ...], FilenameWidthStats] = field(default_factory=dict)

    def _observe_todo_tokens(
        self,
        kind: str,
        mode: str,
        fields: tuple[str, ...],
        syntax: str,
        value: object,
        containers: tuple[str, ...],
    ) -> None:
        # Item frequencies aggregate under the existing form, rather than
        # turning each array cardinality into a new record-weighted form.
        counters = self.native_tool_tokens.setdefault((kind, mode, fields, syntax), {})
        for path, weights in claude_todo_token_counts(value, containers).items():
            counters.setdefault(path, Counter()).update(weights)

    def _native_tool_syntax(self, kind: str, mode: str, fields: tuple[str, ...], syntax: str) -> JSONDocument:
        result = json.loads(syntax) if syntax else {}
        key = kind, mode, fields, syntax
        if key in self.native_tool_tokens:
            result["todo_tokens"] = {
                path: dict(sorted(weights.items())) for path, weights in sorted(self.native_tool_tokens[key].items())
            }
        return json_document(result)

    def observe(
        self,
        provider: str,
        record: str,
        content: object,
        *,
        envelope: Mapping[str, object],
        tool_owner: sqlite3.Connection,
        original_scope: str,
    ) -> None:
        if provider == "claude-code":
            self.role_carriers[public_role_carriers(envelope)] += 1
            status, counts, metadata_class = _original_agent_children(tool_owner, original_scope, envelope)
            result_classes: set[str] = set()
            for tool in iter_tool_records(provider, envelope):
                operand = tool_identity_operand(tool)
                if operand is not None and operand[1] == "result" and isinstance(operand[3], str) and operand[3]:
                    with closing(
                        tool_owner.execute(
                            "SELECT tool_class FROM workload_record_tool_ids WHERE identity=? AND calls>0",
                            (operand[3].encode("utf-8", "surrogatepass"),),
                        )
                    ) as cursor:
                        prior = cursor.fetchone()
                    result_classes.add(prior[0] if prior else "")
            result_class = next(iter(result_classes)) if len(result_classes) == 1 else "mixed" if result_classes else ""
            dispatch_class = json.dumps(
                {
                    "fields": claude_dispatch_field_forms(envelope),
                    "roles": list(public_role_carriers(envelope)),
                    "children": status,
                    "metadata_class": metadata_class,
                    "tool_class": result_class,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self.dispatch_forms[dispatch_class] += 1
            self.agent_counts.setdefault(dispatch_class, AgentChildCounts()).observe(counts)
        mode = content_representation(content)
        self.representations[mode] += 1
        if isinstance(content, Mapping):
            carriers: Sequence[object] = (content,)
        elif isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
            carriers = content
        else:
            carriers = ()
        for carrier in carriers:
            if not isinstance(carrier, Mapping):
                continue
            if isinstance(carrier.get("type"), str):
                field_name = "type"
            elif isinstance(carrier.get("content_type"), str):
                field_name = "content_type"
            else:
                continue
            kind = next(iter_public_content_kinds(carrier), "block:other")
            self.discriminator_fields.setdefault(kind, Counter())[field_name] += 1
            fields = tuple(field for field in PUBLIC_CONTENT_BODY_FIELDS if field in carrier)
            self.body_field_forms.setdefault(kind, Counter())[fields] += 1
        for tool in iter_tool_records(provider, envelope):
            operand = tool_identity_operand(tool)
            if operand is None:
                continue
            kind, event, field_name, value = operand
            if provider == "codex" and kind in {
                "function_call",
                "custom_tool_call",
                "function_call_output",
                "custom_tool_call_output",
            }:
                fields = tuple(field for field in NATIVE_CODEX_TOOL_VALUE_FIELDS if field in tool)
                mode = tool_result_content_mode(tool) if kind.endswith("_output") else ""
                syntax = measured_tool_syntax(tool) if not kind.endswith("_output") else ""
                self.native_tool_field_forms.setdefault(kind, Counter())[(mode, fields, syntax)] += 1
            elif provider == "claude-code" and kind == "tool_use":
                name = tool.get("name")
                tool_input = tool.get("input")
                fields = tuple(
                    field
                    for field in PUBLIC_CLAUDE_TOOL_INPUT_FIELDS
                    if isinstance(tool_input, Mapping) and field in tool_input
                )
                syntax = (
                    json.dumps(
                        {
                            "name": name,
                            "input_mode": "absent"
                            if "input" not in tool
                            else "null"
                            if tool_input is None
                            else content_representation(tool_input),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if isinstance(name, str) and name in PUBLIC_CLAUDE_TOOL_NAMES
                    else ""
                )
                self.native_tool_field_forms.setdefault(kind, Counter())[("", fields, syntax)] += 1
                if name == "TodoWrite":
                    self._observe_todo_tokens(kind, "", fields, syntax, tool_input, ("todos",))
            presence = (
                "absent"
                if field_name not in tool
                else "null"
                if value is None
                else "string"
                if isinstance(value, str)
                else "other"
            )
            relation = "missing"
            prior_mode = ""
            tool_class = ""
            prior_count = 0
            result_mode = (
                tool_result_content_mode(tool)
                if provider == "claude-code"
                and kind == "tool_result"
                or provider == "codex"
                and kind in {"function_call_output", "custom_tool_call_output"}
                else ""
            )
            if isinstance(value, str) and value:
                identity = value.encode("utf-8", "surrogatepass")
                cursor = tool_owner.execute(
                    "SELECT calls,results,result_mode,tool_class FROM workload_record_tool_ids WHERE identity=?",
                    (identity,),
                )
                try:
                    before = cursor.fetchone()
                finally:
                    cursor.close()
                calls, results, prior_mode, prior_class = (0, 0, "", "") if before is None else before
                name = tool.get("name")
                tool_class = (
                    (name if isinstance(name, str) and name in PUBLIC_CLAUDE_TOOL_NAMES else "")
                    if event == "call" and provider == "claude-code"
                    else prior_class
                )
                prior_count = _bucket_index(float(results))
                relation = "both" if calls and results else "call" if calls else "result" if results else "new"
                tool_owner.execute(
                    "INSERT INTO workload_record_tool_ids VALUES (?,?,?,?,?) ON CONFLICT(identity) DO UPDATE SET calls=excluded.calls,results=excluded.results,result_mode=excluded.result_mode,tool_class=excluded.tool_class",
                    (
                        identity,
                        calls + (event == "call"),
                        results + (event == "result"),
                        result_mode if event == "result" else prior_mode,
                        tool_class,
                    ),
                ).close()
            if provider == "claude-code" and kind == "tool_result":
                evidence = envelope.get("toolUseResult")
                fields = tuple(
                    field
                    for field in PUBLIC_CLAUDE_TOOL_RESULT_FIELDS
                    if isinstance(evidence, Mapping) and field in evidence
                )
                syntax = json.dumps(
                    {
                        "name": tool_class,
                        "envelope_mode": "absent"
                        if "toolUseResult" not in envelope
                        else "null"
                        if evidence is None
                        else content_representation(evidence),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self.native_tool_field_forms.setdefault(kind, Counter())[(result_mode, fields, syntax)] += 1
                if tool_class == "TodoWrite":
                    self._observe_todo_tokens(kind, result_mode, fields, syntax, evidence, ("oldTodos", "newTodos"))
            pointer_widths = None
            point: tuple[int, ...] | None = None
            if (
                provider == "claude-code"
                and kind == "tool_result"
                and isinstance(value, str)
                and value
                and isinstance(content_value := tool.get("content"), str)
            ):
                point = persisted_output_shape(content_value)
                if point is not None:
                    match = PERSISTED_OUTPUT_POINTER_PATTERN.search(content_value)
                    assert match is not None
                    pointer_widths = (
                        _bucket_index(float(len(value))),
                        _bucket_index(float(point[5])),
                        pointer_width_relation(len(value), point[5]),
                        "own-direct" if basename(match.group(1)).rsplit(".", 1)[0] == value else "other",
                    )
            operand_key = event, field_name, presence, relation, result_mode, prior_mode, prior_count, pointer_widths
            self.tool_operands.setdefault(kind, Counter())[operand_key] += 1
            if pointer_widths is not None:
                assert isinstance(value, str) and point is not None
                self.tool_filename_widths.setdefault((kind, *operand_key), FilenameWidthStats()).observe(
                    (len(value), point[5])
                )
            if isinstance(value, str) and value:
                self.tool_identity_widths.setdefault((kind, *operand_key), DistributionSketch()).observe(len(value))
        previous: tuple[str, str] | None = None
        prefix = initial_content_prefix(provider, record)
        count = 0
        for kind in iter_public_content_kinds(content):
            self.kinds[kind] += 1
            prefix = content_prefix_kind(provider, record, prefix, kind)
            state = kind, prefix
            if previous is None:
                self.starts[state] += 1
            else:
                self.transitions[previous[0], previous[1], kind, prefix] += 1
            previous = state
            count += 1
        if previous is not None:
            self.ends[previous] += 1
        # Arrays have their own measured count population. A string/object
        # is not silently reinterpreted as a one-element wire array.
        if mode == "array":
            self.counts.observe(count)

    def payload(self) -> JSONDocument:
        return {
            "tool_operands": {
                kind: [
                    {
                        "event": event,
                        "field": field_name,
                        "presence": presence,
                        "relation": relation,
                        "result_mode": mode,
                        "prior_result_mode": prior_mode,
                        "prior_result_count_bucket": prior_count,
                        "identity_width": self.tool_identity_widths[
                            (kind, event, field_name, presence, relation, mode, prior_mode, prior_count, widths)
                        ].to_payload()
                        if (kind, event, field_name, presence, relation, mode, prior_mode, prior_count, widths)
                        in self.tool_identity_widths
                        else None,
                        "pointer_widths": None
                        if widths is None
                        else {
                            "identity_bucket": widths[0],
                            "basename_bucket": widths[1],
                            "relation": widths[2],
                            "basename_relation": widths[3],
                            "points": self.tool_filename_widths[
                                (kind, event, field_name, presence, relation, mode, prior_mode, prior_count, widths)
                            ].to_payload(),
                        },
                        "count": count,
                    }
                    for (event, field_name, presence, relation, mode, prior_mode, prior_count, widths), count in sorted(
                        rows.items(), key=lambda item: json.dumps(item[0], separators=(",", ":"))
                    )
                ]
                for kind, rows in sorted(self.tool_operands.items())
            },
            "dispatch_forms": [
                {**json.loads(fields), "count": count, "child_count_joint": self.agent_counts[fields].to_payload()}
                for fields, count in sorted(self.dispatch_forms.items())
            ],
            "role_carriers": [
                {"root": root, "message": message, "count": count}
                for (root, message), count in sorted(self.role_carriers.items())
            ],
            "representations": dict(sorted(self.representations.items())),
            "block_kinds": dict(sorted(self.kinds.items())),
            "native_tool_field_forms": {
                kind: [
                    {
                        "result_mode": mode,
                        "fields": list(fields),
                        "syntax": self._native_tool_syntax(kind, mode, fields, syntax),
                        "count": count,
                    }
                    for (mode, fields, syntax), count in sorted(forms.items())
                ]
                for kind, forms in sorted(self.native_tool_field_forms.items())
            },
            "block_body_field_forms": {
                kind: [{"fields": list(fields), "count": count} for fields, count in sorted(forms.items())]
                for kind, forms in sorted(self.body_field_forms.items())
            },
            "block_discriminator_fields": {
                kind: dict(sorted(fields.items())) for kind, fields in sorted(self.discriminator_fields.items())
            },
            "initial_states": [
                {"kind": kind, "seen": seen, "count": count} for (kind, seen), count in sorted(self.starts.items())
            ],
            "terminal_states": [
                {"kind": kind, "seen": seen, "count": count} for (kind, seen), count in sorted(self.ends.items())
            ],
            "transitions": [
                {"from": first, "from_seen": before, "to": second, "to_seen": after, "count": count}
                for (first, before, second, after), count in sorted(self.transitions.items())
            ],
            "array_block_count": self.counts.to_payload(),
        }


def _record_structure(
    provider: str,
    memberships: JournalMemberships,
    *,
    session_carrier: str | None = None,
    session_header_form: str | None = None,
    agent_depth: int | None = None,
) -> JSONDocument:
    tool_owner = memberships._journal._connection
    tool_owner.execute(
        "CREATE TEMP TABLE workload_record_tool_ids(identity BLOB PRIMARY KEY,calls INTEGER NOT NULL,results INTEGER NOT NULL,result_mode TEXT NOT NULL,tool_class TEXT NOT NULL) STRICT, WITHOUT ROWID"
    ).close()
    try:
        return _record_structure_with_tool_owner(
            provider,
            memberships,
            tool_owner,
            session_carrier=session_carrier,
            session_header_form=session_header_form,
            agent_depth=agent_depth,
        )
    finally:
        tool_owner.execute("DROP TABLE workload_record_tool_ids").close()


def _record_structure_with_tool_owner(
    provider: str,
    memberships: JournalMemberships,
    tool_owner: sqlite3.Connection,
    *,
    session_carrier: str | None = None,
    session_header_form: str | None = None,
    agent_depth: int | None = None,
) -> JSONDocument:
    """Aggregate ordered public record kinds without retaining source keys."""
    kinds: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    scope_counts = DistributionSketch()
    gaps = DistributionSketch()
    content_structures: dict[str, dict[str, dict[str, _ContentStructure]]] = {}
    forms: dict[str, Counter[str]] = {}
    current_scope: tuple[str, str, str] | None = None
    previous_kind: str | None = None
    previous_timestamp: float | None = None
    record_count = 0
    missing_timestamps = 0
    with closing(
        memberships.iter_scope_samples(
            order="stream",
            _session_carrier=session_carrier,
            _session_header_form=session_header_form,
            _agent_depth=agent_depth,
        )
    ) as samples:
        for original in samples:
            scope = original.bundle_scope, original.artifact_kind, original.stream_key
            sample = original.sample
            if current_scope is not None and current_scope != scope:
                if previous_kind is not None:
                    ends[previous_kind] += 1
                scope_counts.observe(record_count)
                record_count = 0
                previous_kind = None
                previous_timestamp = None
                tool_owner.execute("DELETE FROM workload_record_tool_ids").close()
            current_scope = scope
            if sample is None:
                continue
            for record in iter_provider_records(provider, sample):
                kind = record_kind(provider, record)
                kinds[kind] += 1
                content = record_content(provider, record)
                mode = content_representation(content)
                form = record_form(provider, record)
                forms.setdefault(kind, Counter())[form] += 1
                content_structures.setdefault(kind, {}).setdefault(form, {}).setdefault(
                    mode, _ContentStructure()
                ).observe(
                    provider,
                    kind,
                    content,
                    envelope=record,
                    tool_owner=tool_owner,
                    original_scope=original.bundle_scope,
                )
                if previous_kind is None:
                    starts[kind] += 1
                else:
                    transitions[previous_kind, kind] += 1
                previous_kind = kind
                value = record_timestamp(provider, record)
                timestamp = (
                    parse_timestamp(value)
                    if isinstance(value, str | int | float) and not isinstance(value, bool)
                    else None
                )
                if timestamp is None:
                    missing_timestamps += 1
                    previous_timestamp = None
                else:
                    seconds = timestamp.timestamp()
                    if previous_timestamp is not None:
                        gaps.observe(seconds - previous_timestamp)
                    previous_timestamp = seconds
                record_count += 1
    if current_scope is not None:
        if previous_kind is not None:
            ends[previous_kind] += 1
        scope_counts.observe(record_count)
    return json_document(
        {
            "record_kinds": dict(sorted(kinds.items())),
            "initial_kinds": dict(sorted(starts.items())),
            "terminal_kinds": dict(sorted(ends.items())),
            "transitions": [
                {"from": first, "to": second, "count": count} for (first, second), count in sorted(transitions.items())
            ],
            "record_forms_by_kind": {kind: dict(sorted(counts.items())) for kind, counts in sorted(forms.items())},
            "content_blocks_by_record_kind": {
                kind: {
                    form: {mode: evidence.payload() for mode, evidence in sorted(modes.items())}
                    for form, modes in sorted(by_form.items())
                }
                for kind, by_form in sorted(content_structures.items())
            },
            "scope_record_count": scope_counts.to_payload(),
            "adjacent_timestamp_gap_seconds": gaps.to_payload(),
            "missing_timestamp_count": missing_timestamps,
        }
    )


def _opaque_payload_profile(memberships: JournalMemberships) -> JSONDocument:
    """Publish only measured count/length marginals from the original scoped pass."""
    lengths: dict[str, DistributionSketch] = {}
    counts: dict[str, DistributionSketch] = {}
    current_scope: str | None = None
    scope_counts: Counter[str] = Counter()
    # The finite artifact vocabulary bounds this map; scope populations remain
    # disk-backed and are replayed twice so zero counts enter every observed kind.
    scope_count = 0
    with closing(memberships.iter_scope_payloads()) as payloads:
        for scope, kind, length in payloads:
            if scope != current_scope:
                scope_count += 1
                current_scope = scope
            if kind is not None and length is not None:
                lengths.setdefault(kind, DistributionSketch()).observe(length)
    current_scope = None

    def finish_scope() -> None:
        for kind in lengths:
            counts.setdefault(kind, DistributionSketch()).observe(scope_counts[kind])

    with closing(memberships.iter_scope_payloads()) as payloads:
        for scope, kind, _length in payloads:
            if current_scope is not None and scope != current_scope:
                finish_scope()
                scope_counts.clear()
            current_scope = scope
            if kind is not None:
                scope_counts[kind] += 1
        if current_scope is not None:
            finish_scope()
    return json_document(
        {
            "population": "retained-raw-revisions",
            "scope_count": scope_count,
            "kinds": {
                kind: {"bytes": lengths[kind].to_payload(), "count_per_scope": counts[kind].to_payload()}
                for kind in sorted(lengths)
            },
        }
    )


class PhysicalScopeCounts(PointerShapeStats):
    """The existing fixed-log joint sketch for six public file-count axes."""

    axes = (
        "session_document",
        "session_record_stream",
        "agent_transcript",
        "agent_sidecar_meta",
        "other_structured",
        "opaque_payload",
    )

    @staticmethod
    def _validate_point(point: object) -> None:
        if (
            not isinstance(point, list)
            or len(point) != 6
            or any(type(value) is not int or value < 0 for value in point)
        ):
            raise ValueError("original physical scope count joint is malformed")


def _physical_scope_profile(memberships: JournalMemberships) -> JSONDocument:
    counts = PhysicalScopeCounts()
    with closing(memberships.iter_scope_file_counts()) as scopes:
        for _scope, *coordinates in scopes:
            counts.observe(tuple(coordinates))
    return {
        "population": "retained-raw-revisions-or-declared-neutral-scopes",
        "axes": list(PhysicalScopeCounts.axes),
        "count_joint": counts.to_payload(),
    }


class SessionPrefixCounts(PointerShapeStats):
    """Original child/parent/prefix count joints on the existing numeric owner."""

    axes = ("child_records", "parent_records", "child_messages", "parent_messages", "shared_leading_messages")

    @staticmethod
    def _validate_point(point: object) -> None:
        if (
            not isinstance(point, list)
            or len(point) != 5
            or any(type(value) is not int or value < 0 for value in point)
            or point[2] > point[0]
            or point[3] > point[1]
            or point[4] > min(point[2:4])
        ):
            raise ValueError("original session prefix count joint is malformed")


def _codex_parent_operand(header: Mapping[str, object]) -> tuple[str, str | None]:
    """Measure the first header's declared parent carrier, not later replay headers."""
    source = header.get("source")
    subagent = source.get("subagent") if isinstance(source, Mapping) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, Mapping) else None
    operands = (
        ("forked_from_id", header.get("forked_from_id")),
        ("thread_spawn.parent_thread_id", spawn.get("parent_thread_id") if isinstance(spawn, Mapping) else None),
        ("parent_thread_id", header.get("parent_thread_id")),
    )
    for carrier, value in operands:
        if isinstance(value, str) and value.strip():
            return ("spawn:" if isinstance(subagent, Mapping) else "prefix:") + carrier, value.strip()
    return ("spawn:missing_parent" if isinstance(subagent, Mapping) else "no_explicit_parent"), None


def _session_lineage_profile(provider: str, memberships: JournalMemberships) -> JSONDocument:
    """Match private original identities and bytes without publishing their values.

    This is evidence for planning, not permission to manufacture a relationship
    from generic parent-field marginals. Multiple retained revisions of a parent
    remain ambiguous; no newest-revision or bundle-order guess selects one.
    """
    if provider != "codex":
        return {"population": "first-original-session-header", "supported": False}
    connection = memberships._journal._connection
    connection.execute(
        "CREATE TEMP TABLE workload_session_headers("
        "stream_ordinal INTEGER PRIMARY KEY, stream_id TEXT, parent_id TEXT, carrier TEXT NOT NULL, "
        "message_count INTEGER NOT NULL, record_count INTEGER NOT NULL, bundle_scope TEXT NOT NULL, "
        "artifact_kind TEXT NOT NULL,stream_key TEXT NOT NULL,header_form TEXT,field_forms TEXT) STRICT"
    ).close()
    try:
        connection.execute(
            "CREATE TEMP TABLE workload_session_messages("
            "stream_ordinal INTEGER NOT NULL, ordinal INTEGER NOT NULL, record_bytes BLOB NOT NULL, "
            "PRIMARY KEY(stream_ordinal,ordinal)) STRICT, WITHOUT ROWID"
        ).close()
        try:
            stream_ordinal = -1
            current_stream: tuple[str, str, str] | None = None
            header_seen = False
            message_count = 0
            with closing(memberships.iter_scope_samples(order="stream")) as samples:
                for original in samples:
                    if original.artifact_kind != "session_record_stream":
                        continue
                    coordinate = (original.bundle_scope, original.artifact_kind, original.stream_key)
                    if coordinate != current_stream:
                        stream_ordinal += 1
                        current_stream = coordinate
                        header_seen = False
                        message_count = 0
                        connection.execute(
                            "INSERT INTO workload_session_headers VALUES (?,NULL,NULL,'missing_header',0,0,?,?,?,NULL,NULL)",
                            (stream_ordinal, *coordinate),
                        ).close()
                    if original.sample is None:
                        continue
                    for record in iter_provider_records(provider, original.sample):
                        connection.execute(
                            "UPDATE workload_session_headers SET record_count=record_count+1 WHERE stream_ordinal=?",
                            (stream_ordinal,),
                        ).close()
                        header = codex_session_header(record)
                        if not header_seen and header is not None:
                            identity = header.get("id")
                            if isinstance(identity, str) and identity:
                                header_seen = True
                                carrier, parent = _codex_parent_operand(header)
                                connection.execute(
                                    "UPDATE workload_session_headers SET stream_id=?,parent_id=?,carrier=?,header_form=?,field_forms=? "
                                    "WHERE stream_ordinal=?",
                                    (
                                        identity,
                                        parent,
                                        carrier,
                                        record_form(provider, record),
                                        json.dumps(
                                            codex_parent_field_forms(header), sort_keys=True, separators=(",", ":")
                                        ),
                                        stream_ordinal,
                                    ),
                                ).close()
                        if not codex_message_record(record):
                            continue
                        exact_record = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                        connection.execute(
                            "INSERT INTO workload_session_messages VALUES (?,?,?)",
                            (stream_ordinal, message_count, exact_record.encode("ascii")),
                        ).close()
                        message_count += 1
                        connection.execute(
                            "UPDATE workload_session_headers SET message_count=? WHERE stream_ordinal=?",
                            (message_count, stream_ordinal),
                        ).close()
            connection.execute(
                "CREATE INDEX temp.workload_session_identity ON workload_session_headers(stream_id)"
            ).close()
            outcomes: Counter[str] = Counter()
            joints: dict[str, SessionPrefixCounts] = {}
            header_classes: Counter[tuple[str, str, str, str]] = Counter()
            header_joints: dict[tuple[str, str, str, str], SessionPrefixCounts] = {}
            cursor = connection.execute(
                "SELECT child.stream_ordinal,child.carrier,child.message_count,child.parent_id,"
                "COUNT(parent.stream_ordinal),MIN(parent.stream_ordinal),MIN(parent.message_count), "
                "child.record_count,MIN(parent.record_count),child.header_form,child.field_forms "
                "FROM workload_session_headers child LEFT JOIN workload_session_headers parent "
                "ON child.parent_id=parent.stream_id GROUP BY child.stream_ordinal ORDER BY child.stream_ordinal"
            )
            try:
                for (
                    child,
                    carrier,
                    child_count,
                    parent_id,
                    matches,
                    parent,
                    parent_count,
                    child_records,
                    parent_records,
                    header_form,
                    field_forms,
                ) in cursor:
                    if parent_id is None:
                        outcomes[carrier] += 1
                        key = (carrier, header_form or "absent", field_forms or "{}", carrier)
                        header_classes[key] += 1
                        header_joints.setdefault(key, SessionPrefixCounts()).observe(
                            (child_records, 0, child_count, 0, 0)
                        )
                        continue
                    if matches != 1 or parent == child:
                        outcome = "ambiguous_parent" if matches else "unretained_parent"
                        outcomes[outcome] += 1
                        header_classes[(carrier, header_form, field_forms, outcome)] += 1
                        continue
                    outcomes["matched:" + carrier] += 1
                    mismatch = connection.execute(
                        "SELECT MIN(child.ordinal) FROM workload_session_messages child "
                        "LEFT JOIN workload_session_messages parent "
                        "ON parent.stream_ordinal=? AND parent.ordinal=child.ordinal "
                        "WHERE child.stream_ordinal=? AND (parent.record_bytes IS NULL OR parent.record_bytes!=child.record_bytes)",
                        (parent, child),
                    )
                    try:
                        first_mismatch = mismatch.fetchone()[0]
                    finally:
                        mismatch.close()
                    prefix_count = child_count if first_mismatch is None else first_mismatch
                    key = (carrier, header_form, field_forms, "matched")
                    header_classes[key] += 1
                    header_joints.setdefault(key, SessionPrefixCounts()).observe(
                        (child_records, parent_records, child_count, parent_count, prefix_count)
                    )
                    joints.setdefault(carrier, SessionPrefixCounts()).observe(
                        (child_records, parent_records, child_count, parent_count, prefix_count)
                    )
            finally:
                cursor.close()
            structures: JSONDocument = {}
            carriers = connection.execute(
                "SELECT DISTINCT carrier,header_form FROM workload_session_headers ORDER BY carrier,header_form"
            )
            try:
                # The first-header carrier grammar is a fixed public field
                # vocabulary, not a retained identity population. Close its
                # native cursor before the conditional pass's TEMP DDL.
                original_carriers = tuple(carriers)
            finally:
                carriers.close()
            for carrier, form in original_carriers:
                by_form = structures.setdefault(carrier, {})
                assert isinstance(by_form, dict)
                by_form[form or "absent"] = _record_structure(
                    provider, memberships, session_carrier=carrier, session_header_form=form
                )
            return {
                "population": "first-original-session-header",
                "supported": True,
                "artifact_kinds": ["session_record_stream"],
                "prefix_equality": "canonical-original-wire-message",
                "outcomes": dict(sorted(outcomes.items())),
                "prefix_axes": list(SessionPrefixCounts.axes),
                "matched_prefix_joints": {key: joint.to_payload() for key, joint in sorted(joints.items())},
                "record_structures": structures,
                "header_classes": [
                    {
                        "carrier": carrier,
                        "form": form,
                        "fields": json.loads(fields),
                        "outcome": outcome,
                        "count": count,
                        "prefix_joint": header_joints[key].to_payload() if key in header_joints else [],
                    }
                    for key, count in sorted(header_classes.items())
                    for carrier, form, fields, outcome in (key,)
                ],
            }
        finally:
            connection.execute("DROP TABLE workload_session_messages").close()
    finally:
        connection.execute("DROP TABLE workload_session_headers").close()


def build_package_workload_profile(
    *,
    provider: str,
    version: str,
    package: _PackageAccumulator,
    element_schemas: Mapping[str, JSONDocument],
    privacy_policy: str,
    observation_outcomes: JSONDocument | None = None,
) -> WorkloadProfile:
    """Keep cross-record evidence in the package, field distributions in schemas."""
    elements: JSONDocument = {}
    depth_grammars: DepthGrammars | None = None
    with ExitStack() as stack:
        memberships = stack.enter_context(_workload_memberships(package.memberships))
        if provider == "claude-code":
            stack.enter_context(_agent_child_population(memberships))
        for element_kind in sorted(element_schemas):
            element_memberships = memberships.for_artifact(element_kind)
            element_profile = {
                "artifact_count": sum(sum(row[1:6]) for row in element_memberships.iter_scope_file_counts()),
                "sample_count": element_memberships.sample_count,
                "record_structure": _record_structure(provider, element_memberships),
            }
            if provider == "claude-code" and element_kind == "agent_transcript":
                structures = DepthGrammars()
                after_depth = 0
                while True:
                    with closing(
                        memberships._journal._connection.execute(
                            "SELECT min(depth) FROM workload_agent_depths WHERE depth>?", (after_depth,)
                        )
                    ) as cursor:
                        row = cursor.fetchone()
                    if row is None or row[0] is None:
                        break
                    after_depth = row[0]
                    # End the depth cursor before the original grammar pass
                    # creates/drops its own TEMP indexes on this same owner.
                    structures.append(
                        str(after_depth), _record_structure(provider, element_memberships, agent_depth=after_depth)
                    )
                depth_grammars = structures
            elements[element_kind] = json_document(element_profile)
        relationships = _relationship_profile(provider, memberships)
        opaque_payloads = _opaque_payload_profile(memberships)
        physical_scopes = _physical_scope_profile(memberships)
        session_lineage = _session_lineage_profile(provider, memberships)
        total_samples = memberships.sample_count
        # This population counts retained original physical main-session
        # revisions, not distinct conversations or all files in a bundle.
        main_session_exclusions: JSONDocument
        if provider == "codex":
            outcomes = session_lineage["outcomes"]
            assert isinstance(outcomes, dict)
            main_session_count = outcomes.get("no_explicit_parent", 0)
            assert type(main_session_count) is int
            main_session_exclusions = {key: value for key, value in outcomes.items() if key != "no_explicit_parent"}
        else:
            main_session_count = 0
            child_count = 0
            with closing(memberships.iter_scope_file_counts()) as scopes:
                for _scope, documents, streams, agents, _metadata, _other, _opaque in scopes:
                    main_session_count += documents + streams
                    child_count += agents
            main_session_exclusions = {"agent_transcript": child_count}

    profile = WorkloadProfile(
        {
            "profile_version": WORKLOAD_PROFILE_VERSION,
            "profile_kind": "provider-package",
            "provider": provider,
            "package_version": version,
            "inference_version": "field-distributions-v1",
            "privacy_policy": privacy_policy,
            "privacy_classification": "aggregate-structural-no-raw-content",
            "provenance": {
                "first_seen": package.first_seen,
                "last_seen": package.last_seen,
                "bundle_scope_count": _package_bundle_scope_count(package),
                "sample_count": total_samples,
                "main_session_count": main_session_count,
                "main_session_population": "retained-original-main-session-revisions-or-neutral-scopes",
                "main_session_exclusions": main_session_exclusions,
                "observation_window": {
                    "start": package.first_seen,
                    "end": package.last_seen,
                },
            },
            "relationships": relationships,
            "opaque_payloads": opaque_payloads,
            "physical_scopes": physical_scopes,
            "session_lineage": session_lineage,
            "elements": elements,
        },
        depth_grammars,
    )
    if observation_outcomes:
        provenance = profile["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("Workload profile provenance must be an object")
        provenance["observation_outcomes"] = observation_outcomes
    profile["profile_id"] = workload_profile_identity(profile)
    return profile


__all__ = ["WORKLOAD_PROFILE_VERSION", "build_package_workload_profile", "workload_profile_identity"]
