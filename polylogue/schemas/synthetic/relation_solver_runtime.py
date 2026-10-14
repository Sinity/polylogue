"""Runtime behavior mixin for synthetic relation solving."""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import OrderedDict
from collections.abc import Callable, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from polylogue.core.tool_result_pointer import (
    PERSISTED_OUTPUT_POINTER_PATTERN,
    PERSISTED_OUTPUT_PREFIX,
    SidecarReferenceMatch,
    matching_tool_result_reference,
)
from polylogue.schemas.field_stats.distributions import _bucket_index
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.models import GeneratedRecord, GeneratedString, SchemaRecord
from polylogue.schemas.synthetic.text_value import JoinedText, SampledText

_T = TypeVar("_T")

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import (
        ForeignKeyGraph,
        MutualExclusionGroup,
        TimeDeltaConstraint,
    )


@dataclass(frozen=True)
class _ForeignKeyAnnotation:
    source: str
    target: str


@dataclass(frozen=True)
class _TimeDeltaAnnotation:
    field_a: str
    field_b: str
    min_delta: float
    max_delta: float
    avg_delta: float


@dataclass(frozen=True)
class _NumericOrderAnnotation:
    lower: str
    upper: str


@dataclass(frozen=True)
class _MutualExclusionAnnotation:
    parent: str
    fields: tuple[str, ...]


# Retain three annotation kinds for the ordinary and coverage schemas. The
# content digest changes for both list edits and nested annotation mutations.
_PARSED_ANNOTATIONS: OrderedDict[tuple[str, str], tuple[Any, ...]] = OrderedDict()
_PARSED_ANNOTATIONS_LIMIT = 8
_PARSED_ANNOTATIONS_LOCK = threading.Lock()


def _parsed_annotations(
    schema: SchemaRecord, key: str, parse: Callable[[tuple[SchemaRecord, ...]], tuple[_T, ...]]
) -> tuple[_T, ...]:
    value = schema.get(key)
    if not isinstance(value, list | tuple):
        return ()
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    cache_key = (key, hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    with _PARSED_ANNOTATIONS_LOCK:
        cached = _PARSED_ANNOTATIONS.get(cache_key)
        if cached is not None:
            _PARSED_ANNOTATIONS.move_to_end(cache_key)
            return cast(tuple[_T, ...], cached)
    # Parse the same snapshot that supplied the key, not the caller's live list.
    snapshot = json.loads(serialized)
    parsed = parse(tuple(item for item in snapshot if isinstance(item, dict)))
    with _PARSED_ANNOTATIONS_LOCK:
        _PARSED_ANNOTATIONS[cache_key] = parsed
        _PARSED_ANNOTATIONS.move_to_end(cache_key)
        while len(_PARSED_ANNOTATIONS) > _PARSED_ANNOTATIONS_LIMIT:
            _PARSED_ANNOTATIONS.popitem(last=False)
    return parsed


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _int_value(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return default


def _float_value(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _foreign_key_annotations(schema: SchemaRecord) -> tuple[_ForeignKeyAnnotation, ...]:
    return _parsed_annotations(schema, "x-polylogue-foreign-keys", _parse_foreign_key_records)


def _parse_foreign_key_records(records: tuple[SchemaRecord, ...]) -> tuple[_ForeignKeyAnnotation, ...]:
    annotations: list[_ForeignKeyAnnotation] = []
    for record in records:
        source = str(record.get("source", "")).strip()
        target = str(record.get("target", "")).strip()
        if source and target:
            annotations.append(_ForeignKeyAnnotation(source=source, target=target))
    return tuple(annotations)


def _time_delta_annotations(schema: SchemaRecord) -> tuple[_TimeDeltaAnnotation, ...]:
    return _parsed_annotations(schema, "x-polylogue-time-deltas", _parse_time_delta_records)


def _parse_time_delta_records(records: tuple[SchemaRecord, ...]) -> tuple[_TimeDeltaAnnotation, ...]:
    annotations: list[_TimeDeltaAnnotation] = []
    for record in records:
        annotations.append(
            _TimeDeltaAnnotation(
                field_a=str(record.get("field_a", "")).strip(),
                field_b=str(record.get("field_b", "")).strip(),
                min_delta=_float_value(record.get("min_delta"), 0.0),
                max_delta=_float_value(record.get("max_delta"), 0.0),
                avg_delta=_float_value(record.get("avg_delta"), 0.0),
            )
        )
    return tuple(annotation for annotation in annotations if annotation.field_a and annotation.field_b)


def _mutual_exclusion_annotations(schema: SchemaRecord) -> tuple[_MutualExclusionAnnotation, ...]:
    return _parsed_annotations(schema, "x-polylogue-mutually-exclusive", _parse_mutual_exclusion_records)


def _parse_mutual_exclusion_records(records: tuple[SchemaRecord, ...]) -> tuple[_MutualExclusionAnnotation, ...]:
    annotations: list[_MutualExclusionAnnotation] = []
    for record in records:
        parent = str(record.get("parent", "")).strip()
        fields = _string_tuple(record.get("fields"))
        if parent and len(fields) >= 2:
            annotations.append(_MutualExclusionAnnotation(parent=parent, fields=fields))
    return tuple(annotations)


def _parse_numeric_order_records(records: tuple[SchemaRecord, ...]) -> tuple[_NumericOrderAnnotation, ...]:
    result: list[_NumericOrderAnnotation] = []
    seen: set[str] = set()
    for row in records:
        lower, upper = row.get("lower"), row.get("upper")
        if (
            not isinstance(lower, str)
            or not isinstance(upper, str)
            or (lower, upper)
            not in {
                ("cached_input_tokens", "input_tokens"),
                ("reasoning_output_tokens", "output_tokens"),
            }
        ):
            raise ValueError("numeric subset declaration is not a raw Codex token law")
        if lower in seen or upper in seen:
            raise ValueError("numeric subset declarations overlap")
        seen.update((lower, upper))
        result.append(_NumericOrderAnnotation(lower, upper))
    return tuple(result)


def _numeric_support(schema: SchemaRecord) -> tuple[list[tuple[int, int, float]], tuple[float, float] | None]:
    """Read the same representatives/range that the existing number draw uses."""
    from polylogue.schemas.synthetic.observed_values import histogram_value, neutral_numeric_range

    observed = schema.get("x-polylogue-observed-distribution")
    distribution = observed.get("numeric") if isinstance(observed, dict) else None
    points: list[tuple[int, int, float]] = []
    integer = schema.get("type") == "integer"
    if isinstance(distribution, dict):
        rows = distribution.get("histogram")
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, list) or len(row) != 2:
                continue
            index, count = row
            if type(index) is not int or type(count) is not int or count <= 0:
                continue
            value = histogram_value(distribution, index)
            if value is not None:
                points.append((index, count, float(int(value)) if integer else value))
        if points:
            return points, None
    declared = schema.get("x-polylogue-range")
    if isinstance(declared, list) and len(declared) >= 2:
        lo, hi = declared[:2]
        if (
            isinstance(lo, int | float)
            and not isinstance(lo, bool)
            and isinstance(hi, int | float)
            and not isinstance(hi, bool)
        ):
            lower, upper = float(lo), float(hi)
            if lower <= upper:
                return [], (lower, upper)
    if schema.get("x-polylogue-semantic-role") == "message_timestamp":
        raise ValueError("declared numeric subset has no original numeric support")
    lower, upper = neutral_numeric_range(schema)
    return [], (lower, upper)


def _condition_numeric_support(
    schema: SchemaRecord, *, minimum: float | None = None, maximum: float | None = None
) -> SchemaRecord:
    points, interval = _numeric_support(schema)
    if points:
        kept = [
            (index, count)
            for index, count, value in points
            if (minimum is None or value >= minimum) and (maximum is None or value <= maximum)
        ]
        if not kept:
            raise ValueError("declared numeric subset has no feasible original support")
        observed = schema["x-polylogue-observed-distribution"]
        assert isinstance(observed, dict)
        distribution = observed["numeric"]
        assert isinstance(distribution, dict)
        return {
            **schema,
            "x-polylogue-observed-distribution": {
                **observed,
                "numeric": {**distribution, "histogram": [list(row) for row in kept]},
            },
        }
    assert interval is not None
    lower, upper = interval
    if schema.get("type") == "integer":
        # The existing scalar draw truncates uniform real values toward zero.
        # Condition its original continuous support by the exact preimage of
        # allowed integer values, rather than truncating a fractional bound.
        first, last = int(lower), int(upper)
        if minimum is not None:
            first = max(first, math.ceil(minimum))
        if maximum is not None:
            last = min(last, math.floor(maximum))
        if first > last:
            raise ValueError("declared numeric subset has no feasible original support")
        lower = max(lower, float(first) if first > 0 else math.nextafter(float(first - 1), math.inf))
        upper = min(upper, float(last) if last < 0 else math.nextafter(float(last + 1), -math.inf))
    else:
        lower = lower if minimum is None else max(lower, minimum)
        upper = upper if maximum is None else min(upper, maximum)
    if lower > upper:
        raise ValueError("declared numeric subset has no feasible original support")
    return {**schema, "x-polylogue-range": [lower, upper]}


def _generated_pointer_filename(value: object) -> GeneratedString | None:
    if isinstance(value, str):
        match = PERSISTED_OUTPUT_POINTER_PATTERN.search(value)
        from posixpath import basename

        return basename(match.group(1)) if match is not None else None
    if isinstance(value, JoinedText):
        # The field sampler retains this exact recipe for large pointer text.
        if len(value.parts) == 5 and value.parts[0] == PERSISTED_OUTPUT_PREFIX:
            return value.parts[3]
        return None
    return None


class RelationConstraintSolverRuntimeMixin:
    fk_graph: ForeignKeyGraph
    time_deltas: list[TimeDeltaConstraint]
    mutual_exclusions: list[MutualExclusionGroup]
    mutual_exclusions_by_parent: dict[str, tuple[MutualExclusionGroup, ...]]
    _time_delta_cls: type[TimeDeltaConstraint]
    _mutual_exclusion_cls: type[MutualExclusionGroup]

    def numeric_orders(self, schema: SchemaRecord) -> tuple[_NumericOrderAnnotation, ...]:
        if "x-polylogue-numeric-order" in schema:
            records = schema["x-polylogue-numeric-order"]
            if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
                raise ValueError("numeric subset declaration must contain original field pairs")
        return _parsed_annotations(schema, "x-polylogue-numeric-order", _parse_numeric_order_records)

    def condition_numeric_field(
        self,
        name: str,
        schema: SchemaRecord,
        *,
        properties: SchemaRecord,
        present: set[str],
        values: GeneratedRecord,
        orders: tuple[_NumericOrderAnnotation, ...],
    ) -> SchemaRecord:
        for order in orders:
            if order.lower not in present or order.upper not in present:
                continue
            if name == order.upper:
                lower_schema = properties.get(order.lower)
                if not isinstance(lower_schema, dict):
                    raise ValueError("numeric subset lost its original lower field")
                points, interval = _numeric_support(lower_schema)
                minimum = min(point[2] for point in points) if points else interval[0] if interval is not None else None
                if minimum is not None and lower_schema.get("type") == "integer":
                    minimum = float(int(minimum))
                return _condition_numeric_support(schema, minimum=minimum)
            if name == order.lower:
                upper = values.get(order.upper)
                if not isinstance(upper, int | float) or isinstance(upper, bool):
                    raise ValueError("numeric subset lost its original upper draw")
                return _condition_numeric_support(schema, maximum=float(upper))
        return schema

    def _parse_foreign_keys(self, schema: SchemaRecord) -> None:
        for annotation in _foreign_key_annotations(schema):
            self.fk_graph.references[annotation.source] = annotation.target

    def _parse_time_deltas(self, schema: SchemaRecord) -> None:
        for annotation in _time_delta_annotations(schema):
            self.time_deltas.append(
                self._time_delta_cls(
                    field_a=annotation.field_a,
                    field_b=annotation.field_b,
                    min_delta=annotation.min_delta,
                    max_delta=annotation.max_delta,
                    avg_delta=annotation.avg_delta,
                )
            )

    def _parse_mutual_exclusions(self, schema: SchemaRecord) -> None:
        groups_by_parent: dict[str, list[MutualExclusionGroup]] = {}
        for annotation in _mutual_exclusion_annotations(schema):
            group = self._mutual_exclusion_cls(
                parent_path=annotation.parent,
                field_names=frozenset(annotation.fields),
            )
            self.mutual_exclusions.append(group)
            groups_by_parent.setdefault(group.parent_path, []).append(group)
        self.mutual_exclusions_by_parent = {
            parent_path: tuple(groups) for parent_path, groups in groups_by_parent.items()
        }

    def register_generated_id(self, path: str, value: GeneratedString) -> None:
        self.fk_graph.register_id(path, value)

    def resolve_foreign_key(self, path: str, rng: random.Random) -> GeneratedString | None:
        return self.fk_graph.resolve_reference(path, rng)

    def _tool_result_reference(self, tool_use_id: GeneratedString) -> tuple[int, bool] | None:
        if self.fk_graph._connection is None:
            raise RuntimeError("tool-result reference outlived its original relation owner")
        from polylogue.schemas.synthetic.relations import _string_state

        key = json.dumps(_string_state(tool_use_id), separators=(",", ":"), ensure_ascii=True)
        cursor = self.fk_graph._connection.execute(
            "SELECT inline_length,truncated FROM generated_tool_results WHERE tool_use_id=?", (key,)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        return (int(row[0]), bool(row[1])) if row is not None else None

    def _persisted_tool_reference(self, filename: str) -> GeneratedString | None:
        if self.fk_graph._connection is None:
            raise RuntimeError("tool-result reference outlived its original relation owner")
        from polylogue.schemas.synthetic.relations import _string_state

        cursor = self.fk_graph._connection.execute(
            "SELECT tool_use_id FROM generated_persisted_tool_names WHERE filename=?",
            (json.dumps(_string_state(filename), separators=(",", ":"), ensure_ascii=True),),
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        from polylogue.schemas.synthetic.relations import _string_from_state

        return _string_from_state(json.loads(row[0])) if row is not None else None

    def tool_result_reference_for_filename(self, filename: str) -> SidecarReferenceMatch[GeneratedString] | None:
        # Bind the generic pure matcher to this original recipe identity.
        matcher = cast(
            Callable[
                [str, Callable[[GeneratedString], tuple[int, bool] | None], Callable[[str], GeneratedString | None]],
                SidecarReferenceMatch[GeneratedString] | None,
            ],
            matching_tool_result_reference,
        )
        return matcher(filename, self._tool_result_reference, self._persisted_tool_reference)

    def stage_tool_result_reference(self, block: GeneratedRecord) -> None:
        self._retain_tool_result_reference(block, finalized=False)

    def observe_tool_result_reference(self, block: GeneratedRecord) -> None:
        self._retain_tool_result_reference(block, finalized=True)

    def _retain_tool_result_reference(self, block: GeneratedRecord, *, finalized: bool) -> None:
        """Retain reference history on the same owner as the sampled block.

        This is finite shape metadata, never a copy of the generated body. A
        later non-pointer result updates truncation but preserves prior names,
        exactly as the retained-input join does.
        """
        connection = self.fk_graph._record_owner(block)
        if (
            block.get("type") != "tool_result"
            or not isinstance(tool_id := block.get("tool_use_id"), str | SampledText | JoinedText)
            or not len(tool_id)
        ):
            raise ValueError("tool-result reference requires its original generated identity")
        content = block.get("content")
        pointer = _generated_pointer_filename(content)
        if isinstance(content, str | JoinedText | SampledText):
            inline_length = len(content)
        elif isinstance(content, list | GeneratedArray):
            inline_length = sum(
                len(text)
                for part in content
                if isinstance(part, dict | GeneratedRecord)
                and part.get("type") == "text"
                and isinstance(text := part.get("text"), str | JoinedText | SampledText)
            )
        else:
            inline_length = 0
        from polylogue.schemas.record_structure import tool_result_content_mode

        mode = (
            "string:" + ("truncated" if pointer is not None else "inline")
            if isinstance(content, str | JoinedText | SampledText)
            else "array:inline"
            if isinstance(content, list | GeneratedArray)
            else tool_result_content_mode(block)
        )
        from polylogue.schemas.synthetic.relations import _string_state

        identity_key = json.dumps(_string_state(tool_id), separators=(",", ":"), ensure_ascii=True)
        connection.execute(
            "INSERT INTO generated_tool_reference_events(record_id,tool_use_id,mode,basename_bucket) VALUES (?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET tool_use_id=excluded.tool_use_id,mode=excluded.mode,basename_bucket=excluded.basename_bucket",
            (block.record_id, identity_key, mode, _bucket_index(float(len(pointer))) if pointer is not None else -1),
        ).close()
        if not finalized:
            return
        connection.execute(
            "INSERT OR REPLACE INTO generated_tool_results VALUES (?,?,?,?)",
            (identity_key, block.record_id, inline_length, int(pointer is not None)),
        ).close()
        if pointer is not None:
            from polylogue.schemas.synthetic.relations import _string_state

            connection.execute(
                "INSERT OR REPLACE INTO generated_persisted_tool_names VALUES (?,?)",
                (json.dumps(_string_state(pointer), separators=(",", ":"), ensure_ascii=True), identity_key),
            ).close()
        connection.execute(
            "UPDATE generated_records SET tool_reference_observed=1 WHERE record_id=?", (block.record_id,)
        ).close()

    def bind_persisted_output_filename(
        self, block: GeneratedRecord, filename: str, *, reference_kind: Literal["direct", "persisted"] = "persisted"
    ) -> SidecarReferenceMatch[GeneratedString]:
        """Finalize only neutral spelling before yielding the original session.

        The drawn string length, newlines, format, selected schema and native
        field coordinate remain unchanged. An impossible filename joint is a
        selection error; callers must select a realizable relationship upstream.
        """
        self.fk_graph._record_owner(block)
        tool_id = block.get("tool_use_id")
        if block.get("type") != "tool_result" or not isinstance(tool_id, str | JoinedText | SampledText):
            raise ValueError("filename binding requires the original tool-result identity")
        if not filename or filename in {".", ".."} or any(c == "/" or c == "\0" or c.isspace() for c in filename):
            raise ValueError("filename binding requires a neutral basename")
        content = block.get("content")
        original = _generated_pointer_filename(content)
        if original is None or len(filename) != len(original):
            raise ValueError("filename joint cannot preserve the sampled pointer shape")
        cursor = self.fk_graph._record_owner(block).execute(
            "SELECT tool_reference_observed FROM generated_records WHERE record_id=?", (block.record_id,)
        )
        try:
            row = cursor.fetchone()
            observed = row is not None and bool(row[0])
        finally:
            cursor.close()
        if observed:
            raise ValueError("filename finalization must precede original reference observation")
        stem = filename.rsplit(".", 1)[0]
        direct = stem == tool_id or self._tool_result_reference(stem) is not None
        if filename.startswith("hook-") or (direct and (reference_kind != "direct" or stem != tool_id)):
            raise ValueError("filename joint conflicts with original direct-reference precedence")
        if not direct and reference_kind != "persisted":
            raise ValueError("filename joint lacks the selected direct reference")
        if isinstance(content, str):
            match = PERSISTED_OUTPUT_POINTER_PATTERN.search(content)
            assert match is not None
            start, end = match.span(1)
            start = end - len(original)
            replacement: GeneratedString = content[:start] + filename + content[end:]
        elif isinstance(content, JoinedText):
            replacement = JoinedText((*content.parts[:3], filename, content.parts[4]))
        else:
            raise ValueError("filename joint lacks a canonical pointer spelling")
        block["content"] = replacement
        self.observe_tool_result_reference(block)
        result = self.tool_result_reference_for_filename(filename)
        if result is None or result.tool_use_id != tool_id or result.reference_kind != reference_kind:
            raise RuntimeError("original filename binding lost its selected reference")
        return result

    def get_time_delta(
        self,
        field_a: str,
        field_b: str,
        rng: random.Random,
    ) -> float | None:
        for td in self.time_deltas:
            if (td.field_a == field_a and td.field_b == field_b) or (td.field_a == field_b and td.field_b == field_a):
                if td.stddev_approx > 0:
                    val = rng.gauss(td.avg_delta, td.stddev_approx)
                else:
                    val = rng.uniform(td.min_delta, td.max_delta)
                return max(td.min_delta, min(td.max_delta, val))
        return None

    def filter_mutually_exclusive(
        self,
        parent_path: str,
        field_names: set[str],
        rng: random.Random,
        *,
        required: Set[str] = frozenset(),
    ) -> set[str]:
        result = set(field_names)
        for group in self.mutual_exclusions_by_parent.get(parent_path, ()):
            overlap = result & group.field_names
            selected = overlap & required
            if len(selected) > 1:
                raise ValueError("conditioned public fields conflict with their mutual exclusion")
            if len(overlap) > 1:
                keeper = next(iter(selected)) if selected else rng.choice(sorted(overlap))
                result -= overlap
                result.add(keeper)
        return result

    def path_matches(self, schema_path: str, annotation_path: str) -> bool:
        return schema_path == annotation_path


__all__ = ["RelationConstraintSolverRuntimeMixin"]
