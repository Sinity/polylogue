"""Borrowed opaque files on the original generated-record relation owner.

This is the Claude filesystem adapter. Counts and byte widths are drawn by
its existing distribution owner; reference paths use the existing measured
path solver. Neutral names finalize original pointer coordinates before the
session can be serialized.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Generator, Iterable, Mapping
from contextlib import closing
from typing import Any, cast

from polylogue.schemas.field_stats.distributions import _bucket_index
from polylogue.schemas.synthetic.models import GeneratedRecord, SchemaRecord, SyntheticAdjunctChunks
from polylogue.schemas.synthetic.observed_values import observed_length_support
from polylogue.schemas.synthetic.record_profiles import OpaqueReferenceKinds
from polylogue.schemas.synthetic.relation_solver_runtime import _generated_pointer_filename
from polylogue.schemas.synthetic.relations import RelationConstraintSolver, _string_from_state, _string_state
from polylogue.schemas.synthetic.runtime import _sampled_length


def _one(connection: sqlite3.Connection, statement: str, bindings: tuple[object, ...] = ()) -> tuple[Any, ...] | None:
    with closing(connection.execute(statement, bindings)) as cursor:
        return cast(tuple[Any, ...] | None, cursor.fetchone())


def _foreign_persisted_name(connection: sqlite3.Connection, name: str, identity_key: str) -> bool:
    keys = tuple(
        json.dumps(_string_state(value), separators=(",", ":"), ensure_ascii=True)
        for value in (name, name.rsplit(".", 1)[0])
    )
    return (
        _one(
            connection,
            "SELECT 1 FROM generated_persisted_tool_names WHERE filename IN (?,?) AND tool_use_id<>? LIMIT 1",
            (*keys, identity_key),
        )
        is not None
    )


def _draw(shape: object, rng: random.Random) -> int:
    if not isinstance(shape, dict):
        raise ValueError("opaque numeric measurement is missing; re-infer the package")
    result = _sampled_length({"x-polylogue-observed-distribution": {"string_length": shape}}, "string_length", rng)
    if result is None:
        raise ValueError("opaque numeric measurement is missing; re-infer the package")
    return result


def _character_at_or_after(point: int, *, dot: bool) -> str | None:
    for value in range(max(1, point), 0x110000):
        if 0xD800 <= value <= 0xDFFF:
            continue
        char = chr(value)
        if char != "/" and not char.isspace() and (dot or char != "."):
            return char
    return None


def _filename_after(
    width: int, *, base: str | None, suffix_width: int, lower: str | None, upper: str | None, prefix: str | None = None
) -> str | None:
    """Find the least lawful neutral spelling in a declared width interval.

    Fixed provider coordinates stay fixed. This is spelling, not a second
    field sampler: width/suffix and relation are already original operands.
    Exhausting the actual finite filename domain proves nonrepresentability.
    """
    total = width + suffix_width
    fixed = dict(enumerate(base if base is not None else prefix or ""))
    if suffix_width:
        fixed[width] = "."
    minimum = cast(str, _character_at_or_after(1, dot=True))
    chars: list[str] = []
    greater = lower is None
    for position in range(total):
        previous = lower[position] if lower is not None and position < len(lower) else None
        if position in fixed:
            char = fixed[position]
        elif greater or previous is None:
            char = minimum
        else:
            next_char = _character_at_or_after(ord(previous), dot=position < width and base is None and prefix is None)
            if next_char is None:
                break
            char = next_char
        chars.append(char)
        if not greater:
            if previous is None or char > previous:
                greater = True
            elif char < previous:
                break
    if len(chars) != total or not greater:
        # A fixed coordinate blocked the lower bound, or the candidate was its
        # exact prefix. Advance the last free coordinate that has a successor.
        advanced = False
        for position in range(min(len(chars), total) - 1, -1, -1):
            if position in fixed:
                continue
            successor = _character_at_or_after(
                ord(chars[position]) + 1, dot=position < width and base is None and prefix is None
            )
            if successor is not None:
                chars = chars[:position] + [successor]
                chars.extend(fixed.get(index, minimum) for index in range(position + 1, total))
                advanced = True
                break
        if not advanced:
            return None
    result = "".join(chars)
    if lower is not None and result <= lower:
        return None
    return result if upper is None or result < upper else None


def _original_block(solver: RelationConstraintSolver, record_id: int) -> GeneratedRecord:
    owner = solver.fk_graph
    result = GeneratedRecord(owner, record_id, owner._generation)
    len(result)
    return result


def _history_supported(solver: RelationConstraintSolver, identity_key: str, shape: SchemaRecord) -> bool:
    starts, ends = shape.get("initial_modes"), shape.get("terminal_modes")
    transitions = shape.get("transitions")
    if not isinstance(starts, dict) or not isinstance(ends, dict) or not isinstance(transitions, list):
        raise ValueError("original opaque history measurements are missing; re-infer the package")
    edges = {(row.get("from"), row.get("to")) for row in transitions if isinstance(row, dict)}
    previous: str | None = None
    with closing(
        solver.fk_graph._live_tool_owner().execute(
            "SELECT mode FROM generated_tool_reference_events WHERE tool_use_id=? ORDER BY ordinal", (identity_key,)
        )
    ) as cursor:
        for (mode,) in cursor:
            if previous is None and mode not in starts or previous is not None and (previous, mode) not in edges:
                return False
            previous = mode
    return previous in ends


def prepare_opaque_adjuncts(
    solver: RelationConstraintSolver, profile: Mapping[str, object] | None, rng: random.Random
) -> None:
    if profile is None:
        return
    relationships = profile.get("relationships")
    opaque = relationships.get("opaque_tool_results") if isinstance(relationships, dict) else None
    if not isinstance(opaque, dict):
        return
    by_mode, classes = opaque.get("per_tool_result"), opaque.get("reference_classes")
    if not isinstance(by_mode, dict) or not isinstance(classes, dict):
        raise ValueError("opaque reference measurements are missing; re-infer the package")
    connection = solver.fk_graph._live_tool_owner()
    with closing(
        connection.execute(
            "SELECT tool_use_id,count(*),max(ordinal) FROM generated_tool_reference_events GROUP BY tool_use_id ORDER BY min(ordinal)"
        )
    ) as tools:
        for identity_key, history_count, terminal_ordinal in tools:
            tool_id = _string_from_state(json.loads(identity_key))
            terminal = _one(
                connection, "SELECT mode FROM generated_tool_reference_events WHERE ordinal=?", (terminal_ordinal,)
            )
            assert terminal is not None
            shape = by_mode.get(terminal[0])
            if not isinstance(shape, dict) or not isinstance(shape.get("result_history"), dict):
                raise ValueError("original result history measurement is missing; re-infer the package")
            histories = cast(SchemaRecord, shape["result_history"])
            pointer_buckets = ",".join(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT basename_bucket FROM generated_tool_reference_events WHERE tool_use_id=? AND basename_bucket>=0 ORDER BY basename_bucket",
                    (identity_key,),
                )
            )
            key = f"{_bucket_index(float(len(tool_id)))}:{_bucket_index(float(history_count))}:{pointer_buckets}"
            choices: list[tuple[str, SchemaRecord]] = []
            weights: list[int] = []
            for presence, history in histories.items():
                if not isinstance(history, dict) or not _history_supported(solver, identity_key, history):
                    continue
                counts = history.get("file_count_by_identity_history_and_pointers")
                count_shape = counts.get(key) if isinstance(counts, dict) else None
                if isinstance(count_shape, dict):
                    choices.append((presence, count_shape))
                    weight = count_shape.get("count")
                    if type(weight) is not int or weight < 1:
                        raise ValueError("opaque file-count weights are malformed")
                    weights.append(weight)
            if not choices:
                raise ValueError("original generated history has no measured opaque file joint")
            presence, count_shape = rng.choices(choices, weights=weights, k=1)[0]
            count = _draw(count_shape, rng)
            path = OpaqueReferenceKinds(str(terminal[0]), shape, count, solver.fk_graph, presence=presence).sample(rng)
            # The finite path lives on the same scratch, never in a Python
            # filename/history list. Later spelling can inspect its neighbours.
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS opaque_path(ordinal INTEGER PRIMARY KEY,kind TEXT NOT NULL) STRICT"
            ).close()
            connection.execute("DELETE FROM opaque_path").close()
            try:
                for index, kind in enumerate(path):
                    connection.execute("INSERT INTO opaque_path VALUES (?,?)", (index, kind)).close()
            finally:
                path.close()
            lower: str | None = None
            with closing(connection.execute("SELECT ordinal,kind FROM opaque_path ORDER BY ordinal")) as files:
                for ordinal, kind in files:
                    family = str(kind).split("_", 1)[0]
                    forms = shape.get("filename_suffix_by_base_width")
                    if not isinstance(forms, dict):
                        raise ValueError("original filename suffix joint is missing; re-infer the package")
                    upper = None
                    next_direct = (
                        _one(
                            connection,
                            "SELECT kind FROM opaque_path WHERE ordinal>? AND kind LIKE 'direct_%' ORDER BY ordinal LIMIT 1",
                            (ordinal,),
                        )
                        if family == "persisted"
                        else None
                    )
                    if next_direct is not None:
                        if not isinstance(tool_id, str):
                            raise ValueError("original file-order joint requires a physical direct-name coordinate")
                        next_suffix = forms.get(str(next_direct[0]) + ":" + str(_bucket_index(float(len(tool_id)))))
                        support = observed_length_support(next_suffix) if isinstance(next_suffix, dict) else {}
                        if not support:
                            raise ValueError("original following direct filename has no measured suffix width")
                        upper = tool_id + "." + chr(0x10FFFF) if any(width > 0 for width in support) else tool_id
                    connection.execute(
                        "CREATE TEMP TABLE IF NOT EXISTS opaque_name_candidates(ordinal INTEGER PRIMARY KEY,name TEXT NOT NULL,suffix_width INTEGER NOT NULL,base TEXT NOT NULL,record_id INTEGER,weight INTEGER NOT NULL) STRICT"
                    ).close()
                    connection.execute("DELETE FROM opaque_name_candidates").close()
                    total_weight = 0
                    bases: Iterable[tuple[object, int | None]]
                    if family == "direct":
                        bases = ((tool_id, None),)
                    else:
                        # Only actual sampled prior/current pointer fields can
                        # supply a persisted name. Null/inline fields stay so.
                        bases = (
                            (
                                cast(
                                    str, _generated_pointer_filename(_original_block(solver, record_id).get("content"))
                                ),
                                record_id,
                            )
                            for (record_id,) in connection.execute(
                                "SELECT record_id FROM generated_tool_reference_events WHERE tool_use_id=? AND mode='string:truncated' ORDER BY ordinal",
                                (identity_key,),
                            )
                        )
                    for base, record_id in bases:
                        if not isinstance(base, str):
                            raise ValueError("opaque filename joint lacks its original scalar pointer coordinate")
                        direct_pointer = family == "persisted" and base.rsplit(".", 1)[0] == tool_id
                        suffix = forms.get(str(kind) + ":" + str(_bucket_index(float(len(base)))))
                        if not isinstance(suffix, dict):
                            continue
                        observed_row = (
                            None
                            if record_id is None
                            else _one(
                                connection,
                                "SELECT tool_reference_observed FROM generated_records WHERE record_id=?",
                                (record_id,),
                            )
                        )
                        finalized = observed_row is not None and bool(observed_row[0])
                        class_shape = classes.get(str(kind))
                        file_widths = class_shape.get("basename_length") if isinstance(class_shape, dict) else None
                        file_width_support = (
                            observed_length_support(file_widths) if isinstance(file_widths, dict) else {}
                        )
                        for suffix_width, weight in observed_length_support(suffix).items():
                            if suffix_width < 0:
                                raise ValueError("opaque filename suffix measurement is negative")
                            if not any(
                                _bucket_index(float(len(base) + suffix_width)) == _bucket_index(float(width))
                                for width in file_width_support
                            ):
                                continue
                            name = _filename_after(
                                len(base),
                                base=base if family == "direct" or finalized else None,
                                suffix_width=suffix_width,
                                lower=lower,
                                upper=upper,
                                prefix=str(tool_id) + "." if direct_pointer else None,
                            )
                            while name is not None and (
                                _one(connection, "SELECT 1 FROM generated_adjuncts WHERE name=?", (name,)) is not None
                                or name in {".", ".."}
                                or name.startswith("hook-")
                                or _foreign_persisted_name(connection, name, identity_key)
                                or (
                                    family == "persisted"
                                    and _one(
                                        connection,
                                        "SELECT 1 FROM generated_tool_reference_events WHERE tool_use_id=? LIMIT 1",
                                        (
                                            json.dumps(
                                                _string_state(name.rsplit(".", 1)[0]),
                                                separators=(",", ":"),
                                                ensure_ascii=True,
                                            ),
                                        ),
                                    )
                                    is not None
                                )
                            ):
                                name = _filename_after(
                                    len(base),
                                    base=base if family == "direct" or finalized else None,
                                    suffix_width=suffix_width,
                                    lower=name,
                                    upper=upper,
                                    prefix=str(tool_id) + "." if direct_pointer else None,
                                )
                            if name is not None:
                                connection.execute(
                                    "INSERT INTO opaque_name_candidates(name,suffix_width,base,record_id,weight) VALUES (?,?,?,?,?)",
                                    (name, suffix_width, base, record_id, weight),
                                ).close()
                                total_weight += weight
                    if not total_weight:
                        raise ValueError("original filename/path width joint is not realizable before session yield")
                    choice = rng.randrange(total_weight)
                    with closing(
                        connection.execute(
                            "SELECT name,suffix_width,base,record_id,weight FROM opaque_name_candidates ORDER BY ordinal"
                        )
                    ) as candidates:
                        for candidate in candidates:
                            choice -= candidate[4]
                            if choice < 0:
                                name, suffix_width, base, record_id, _weight = candidate
                                break
                        else:
                            raise RuntimeError("original filename candidates lost their measured mass")
                    if record_id is not None:
                        block = _original_block(solver, record_id)
                        # An already finalized pointer can supply several real
                        # stem aliases, but cannot be repointed for another file.
                        observed = _one(
                            connection,
                            "SELECT tool_reference_observed FROM generated_records WHERE record_id=?",
                            (record_id,),
                        )
                        assert observed is not None
                        desired = name[:-suffix_width] if suffix_width else name
                        if observed[0]:
                            if base != desired:
                                raise ValueError("opaque path needs another original pointer coordinate")
                        else:
                            solver.bind_persisted_output_filename(
                                block,
                                desired,
                                reference_kind="direct" if desired.rsplit(".", 1)[0] == tool_id else "persisted",
                            )
                    sizes = classes.get(str(kind))
                    byte_length = _draw(sizes.get("bytes") if isinstance(sizes, dict) else None, rng)
                    connection.execute(
                        "INSERT INTO generated_adjuncts(name,kind,byte_length) VALUES (?,?,?)",
                        (name, kind, byte_length),
                    ).close()
                    lower = name
    with closing(
        connection.execute("SELECT record_id FROM generated_tool_reference_events ORDER BY ordinal")
    ) as cursor:
        for (record_id,) in cursor:
            solver.observe_tool_result_reference(_original_block(solver, record_id))
    orphan = classes.get("orphan")
    if not isinstance(orphan, dict):
        raise ValueError("original orphan measurements are missing; re-infer the package")
    orphan_count = _draw(orphan.get("count_per_scope"), rng)
    lower_orphan: str | None = None
    for _ in range(orphan_count):
        width = _draw(orphan.get("basename_length"), rng)
        name = _filename_after(width, base=None, suffix_width=0, lower=lower_orphan, upper=None)
        while name is not None and (
            name in {".", ".."}
            or name.startswith("hook-")
            or solver.tool_result_reference_for_filename(name) is not None
            or _one(connection, "SELECT 1 FROM generated_adjuncts WHERE name=?", (name,)) is not None
        ):
            name = _filename_after(width, base=None, suffix_width=0, lower=name, upper=None)
        if name is None:
            raise ValueError("original orphan filename width has exhausted its neutral spelling domain")
        connection.execute(
            "INSERT INTO generated_adjuncts(name,kind,byte_length) VALUES (?, 'orphan', ?)",
            (name, _draw(orphan.get("bytes"), rng)),
        ).close()
        lower_orphan = name
    with closing(connection.execute("SELECT name,kind FROM generated_adjuncts ORDER BY ordinal")) as cursor:
        for name, kind in cursor:
            match = solver.tool_result_reference_for_filename(name)
            actual = (
                None if match is None else match.reference_kind + ("_truncated" if match.was_truncated else "_inline")
            )
            if (actual is None and kind != "orphan") or (actual is not None and actual != kind):
                raise RuntimeError("original opaque filename has lost its planned reference class")


def iter_opaque_adjuncts(
    solver: RelationConstraintSolver, consumed: list[bool], emitted_bytes: list[int]
) -> Generator[SyntheticAdjunctChunks, None, None]:
    owner = solver.fk_graph
    generation = owner._generation
    after = -1
    while True:
        connection = owner._array_owner(generation)
        with closing(
            connection.execute(
                "SELECT ordinal,name,kind,byte_length FROM generated_adjuncts WHERE ordinal>? ORDER BY ordinal LIMIT 1",
                (after,),
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            consumed[0] = True
            return
        ordinal, name, kind, length = row
        body_consumed = [False]

        def chunks(byte_length: int = length, settled: list[bool] = body_consumed) -> Generator[bytes, None, None]:
            remaining = byte_length
            while remaining:
                owner._array_owner(generation)
                width = min(65536, remaining)
                emitted_bytes[0] += width
                yield b"x" * width
                remaining -= width
            owner._array_owner(generation)
            settled[0] = True

        stream = chunks()
        try:
            yield SyntheticAdjunctChunks(name, kind, length, stream, "tool-results", "opaque_payload")
            if not body_consumed[0]:
                raise ValueError("consume the original opaque body before advancing its metadata")
        finally:
            stream.close()
        after = ordinal
