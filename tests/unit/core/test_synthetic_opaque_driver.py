"""Actual schema driver files retain their original Source join and lifetime."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONDocument
from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN
from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.generation.field_annotations import annotate_schema
from polylogue.schemas.generation.models import _PackageAccumulator
from polylogue.schemas.generation.observation_journal import ObservationJournal
from polylogue.schemas.generation.workload_profiles import build_package_workload_profile
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.sources.live.tool_result_sidecars import (
    ToolResultIndexAccumulator,
    join_tool_result_sidecars,
    sidecar_files_from_directory,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarScope


def _corpus(
    tmp_path: Path,
    contents: list[str | None],
    files: tuple[tuple[str, int], ...],
    *,
    metadata_record: bool = False,
    identities: tuple[str, ...] = (),
) -> SyntheticCorpus:
    record: JSONDocument = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": identities[index] if identities else "middle",
                    "content": content,
                }
                for index, content in enumerate(contents)
            ],
        },
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["user"]},
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["user"]},
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
                                # Content intentionally precedes ID in the declared wire schema.
                                "content": {"type": ["string", "null"]},
                                "tool_use_id": {
                                    "type": "string",
                                    "x-polylogue-values": list(identities) if identities else ["middle"],
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    if metadata_record:
        message = cast(JSONDocument, record["message"])
        blocks = cast(list[JSONDocument], message["content"])
        blocks[0]["metadata"] = {"type": "tool_result", "tool_use_id": "not-a-source-block", "content": None}
        properties = cast(JSONDocument, schema["properties"])
        message_schema = cast(JSONDocument, properties["message"])
        content_schema = cast(JSONDocument, cast(JSONDocument, message_schema["properties"])["content"])
        block_schema = cast(JSONDocument, content_schema["items"])
        cast(JSONDocument, block_schema["properties"])["metadata"] = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
                "tool_use_id": {"type": "string", "x-polylogue-values": ["not-a-source-block"]},
                "content": {"type": "null"},
            },
        }
    annotate_schema(schema, _collect_field_stats([record]))
    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        unit = journal.append_unit(
            SchemaUnit(
                cluster_payload={}, schema_samples=[record], artifact_kind="session_record_stream", bundle_scope="scope"
            )
        )
        journal.assign_profile_family(unit, "family")
        for index, (name, length) in enumerate(files):
            raw_id = f"opaque-{index}"
            journal.record_payload(
                raw_id=raw_id, artifact_kind="tool_result_sidecar", bundle_scope="scope", byte_count=length
            )
            journal.record_terminal(
                raw_id=raw_id,
                status="intentionally_excluded",
                artifact_kind="tool_result_sidecar",
                source_path=f"/neutral/scope/tool-results/{name}",
                reason="opaque",
            )
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": schema},
            privacy_policy="standard",
        )
    return SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )


@pytest.mark.parametrize(
    ("contents", "files", "expected"),
    [
        (["Full output saved to: /neutral/a.txt"], (("a.txt", 10_003),), ["persisted_truncated"]),
        (
            ["Full output saved to: /neutral/a.txt", None],
            (("a.txt", 10_003), ("unowned.txt", 0)),
            ["persisted_inline", "orphan"],
        ),
        (
            ["Full output saved to: /neutral/middle.m.x"],
            (("middle.a", 10_003), ("middle.m.x", 10_003), ("middle.z", 10_003)),
            ["direct_truncated", "persisted_truncated", "direct_truncated"],
        ),
        (
            ["Full output saved to: /neutral/a.txt", "Full output saved to: /neutral/z.txt", None],
            (("a.txt", 10_003), ("middle.a", 10_003), ("z.txt", 10_003)),
            ["persisted_inline", "direct_inline", "persisted_inline"],
        ),
        (["Full output saved to: /neutral/middle.txt"], (("middle.txt", 10_003),), ["direct_truncated"]),
    ],
)
def test_driver_emits_original_current_prior_and_ambiguous_file_relations(
    tmp_path: Path, contents: list[str | None], files: tuple[tuple[str, int], ...], expected: list[str]
) -> None:
    corpus = _corpus(tmp_path, contents, files)
    for seed in range(6):
        sessions = corpus.iter_sessions(count=1, seed=seed)
        try:
            session = next(sessions)
            record = cast(JSONDocument, json.loads(b"".join(session.chunks)))
            directory = tmp_path / f"seed-{seed}" / "tool-results"
            directory.mkdir(parents=True)
            actual = []
            names = []
            for file in session.adjuncts:
                body = b"".join(file.chunks)
                assert len(body) == file.byte_length
                (directory / file.name).write_bytes(body)
                actual.append(file.reference_kind)
                names.append(file.name)
            assert actual == expected
            with ToolResultIndexAccumulator(disk_backed=True) as original:
                original.observe(record)
                for name, kind in zip(names, expected, strict=True):
                    match = original.reference_for_filename(name)
                    if kind == "orphan":
                        assert match is None
                    else:
                        assert match is not None
                        assert match.reference_kind + ("_truncated" if match.was_truncated else "_inline") == kind
            joined = join_tool_result_sidecars(
                [record],
                RetainedSidecarScope(
                    scope_key=str(directory), files=sidecar_files_from_directory(directory), available=True
                ),
            )
            assert len(joined.matched) == len(files) - expected.count("orphan")
            assert [debt.reason for debt in joined.debt] == (
                ["no_owning_tool_result_block"] if "orphan" in expected else []
            )
            blocks = cast(list[JSONDocument], cast(JSONDocument, record["message"])["content"])
            if contents[-1] is None:
                assert blocks[-1]["content"] is None
            if contents == ["Full output saved to: /neutral/middle.txt"]:
                pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(cast(str, blocks[0]["content"]))
                assert pointer is not None and Path(pointer.group(1)).name.rsplit(".", 1)[0] == "middle"
            with pytest.raises(StopIteration):
                next(sessions)
        finally:
            sessions.close()


def test_driver_refuses_advance_until_original_opaque_streams_settle(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, [None], (("middle.txt", 7),))
    sessions = corpus.iter_sessions(count=2, seed=3)
    session = next(sessions)
    b"".join(session.chunks)
    file = next(session.adjuncts)
    with pytest.raises(ValueError):
        next(sessions)
    with pytest.raises(StopIteration):
        next(file.chunks)
    assert isinstance(session.data, GeneratedArray)
    with pytest.raises(RuntimeError, match="original relation owner"):
        len(session.data)
    sessions.close()


def test_large_opaque_body_borrows_one_original_chunk_without_materialization(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, [None], (("middle.txt", 2_000_000_003),))
    tracemalloc.start()
    try:
        sessions = corpus.iter_sessions(count=1, seed=2)
        try:
            session = next(sessions)
            b"".join(session.chunks)
            file = next(session.adjuncts)
            assert file.byte_length == 2_000_000_003
            for _ in range(5):
                assert len(next(file.chunks)) == 65_536
            assert tracemalloc.get_traced_memory()[1] < 8_000_000
        finally:
            sessions.close()
        with pytest.raises(StopIteration):
            next(file.chunks)
        assert isinstance(session.data, GeneratedArray)
        with pytest.raises(RuntimeError, match="original relation owner"):
            len(session.data)
    finally:
        tracemalloc.stop()


def test_collector_fixture_writer_and_fresh_build_consume_the_same_original_files(tmp_path: Path) -> None:
    from tempfile import TemporaryDirectory

    from devtools.fresh_build_bench.corpus import corpus_from_synthetic, verify_manifest
    from polylogue.scenarios import CorpusSpec
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection

    corpus = _corpus(tmp_path, ["Full output saved to: /neutral/a.txt", None], (("a.txt", 10_003), ("unowned.txt", 0)))
    collected = corpus.generate_batch(count=2, seed=11)
    assert len(collected.artifacts) == 2
    assert all(len(item.adjuncts) == 2 for item in collected.artifacts)
    selection = SyntheticSchemaSelection(
        "claude-code", "neutral", "session_record_stream", corpus.schema, corpus.wire_format, corpus.workload_profile
    )
    written = SyntheticCorpus.write_selection_artifacts(
        selection,
        CorpusSpec(provider="claude-code", count=2, seed=11),
        tmp_path / "writer",
        prefix="session",
        index_width=6,
    )
    assert len(written.files) == 6
    assert len(written.session_files) == len(written.facts) == 2
    assert all(path.suffix == ".jsonl" and path.parent == tmp_path / "writer" for path in written.session_files)
    assert len(written.facts) == 2
    for index, artifact in enumerate(collected.artifacts):
        main = tmp_path / "writer" / f"session-{index:06d}.jsonl"
        assert main.read_bytes() == artifact.raw_bytes
        for file in artifact.adjuncts:
            assert (main.parent / main.stem / "tool-results" / file.name).read_bytes() == file.raw_bytes
    # Benchmark corpus creation owns a real /realm output. Its sealed manifest
    # must include opaque files as well as the original session byte stream.
    with TemporaryDirectory(prefix="polylogue-opaque-driver-", dir="/realm/tmp/work") as temporary:
        root = Path(temporary) / "corpus"
        manifest = corpus_from_synthetic(root, corpus=corpus, seed=11, count=2)
        verify_manifest(root, manifest)
        assert manifest["file_count"] == 6
        assert manifest["total_bytes"] == sum(
            len(item.raw_bytes) + sum(len(file.raw_bytes) for file in item.adjuncts) for item in collected.artifacts
        )
        for index, artifact in enumerate(collected.artifacts):
            main = root / "home/.claude/projects/schema" / f"session-{index:06d}.jsonl"
            assert main.read_bytes() == artifact.raw_bytes
            for file in artifact.adjuncts:
                assert (main.parent / main.stem / "tool-results" / file.name).read_bytes() == file.raw_bytes
        assert not corpus._generation_state.driver_active


def test_adjunct_plan_uses_original_public_tool_coordinate_not_arbitrary_objects(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, [None], (("middle.txt", 7),), metadata_record=True)
    sessions = corpus.iter_sessions(count=1, seed=2)
    try:
        session = next(sessions)
        connection = corpus._relation_solver.fk_graph._live_tool_owner()
        cursor = connection.execute("SELECT count(*) FROM generated_tool_reference_events")
        try:
            assert cursor.fetchone() == (1,)
        finally:
            cursor.close()
        record = json.loads(b"".join(session.chunks))
        assert record["message"]["content"][0]["metadata"]["tool_use_id"] == "not-a-source-block"
        files = [(file.name, b"".join(file.chunks)) for file in session.adjuncts]
        assert len(files) == 1 and len(files[0][1]) == 7
        with pytest.raises(StopIteration):
            next(sessions)
    finally:
        sessions.close()


def test_reference_match_borrows_huge_original_identity_without_hydration() -> None:
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver
    from polylogue.schemas.synthetic.text_value import SampledText

    solver = RelationConstraintSolver({})
    try:
        identity = SampledText(2_000_000_003, "x")
        record = solver.fk_graph.new_record({})
        record["type"] = "tool_result"
        record["tool_use_id"] = identity
        record["content"] = "Full output saved to: /neutral/original.txt"
        tracemalloc.start()
        try:
            solver.observe_tool_result_reference(record)
            match = solver.tool_result_reference_for_filename("original.txt")
            assert match is not None and match.tool_use_id == identity
            assert match.reference_kind == "persisted"
            assert tracemalloc.get_traced_memory()[1] < 1_000_000
        finally:
            tracemalloc.stop()
        solver.close()
        with pytest.raises(RuntimeError, match="original relation owner"):
            solver.tool_result_reference_for_filename("original.txt")
    finally:
        solver.close()


def test_original_filename_plan_preserves_distinct_tool_reference_owners(tmp_path: Path) -> None:
    corpus = _corpus(
        tmp_path,
        ["Full output saved to: /neutral/alpha.txt", "Full output saved to: /neutral/bravo.txt"],
        (("alpha.txt", 3), ("bravo.txt", 5)),
        identities=("middle-a", "middle-b"),
    )
    for seed in range(6):
        sessions = corpus.iter_sessions(count=1, seed=seed)
        try:
            session = next(sessions)
            data = json.loads(b"".join(session.chunks))
            blocks = data["message"]["content"]
            references = {}
            for block in blocks:
                pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(block["content"])
                assert pointer is not None
                references[Path(pointer.group(1)).name] = block["tool_use_id"]
            assert len(references) == 2 and len(set(references.values())) == 2
            files = [(file.name, b"".join(file.chunks)) for file in session.adjuncts]
            assert len(files) == 2
            for name, body in files:
                match = corpus._relation_solver.tool_result_reference_for_filename(name)
                assert match is not None
                assert match.tool_use_id == references[name.rsplit(".", 1)[0]]
                assert match.reference_kind == "persisted" and match.was_truncated
                assert len(body) in {3, 4, 5}
            with pytest.raises(StopIteration):
                next(sessions)
        finally:
            sessions.close()


def test_byte_target_counts_actual_session_and_opaque_streams_without_cutting_them(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, [None], (("middle.txt", 101), ("orphan.txt", 37)))
    first = corpus.generate_batch(count=1, seed=4).artifacts[0]
    first_bytes = len(first.raw_bytes) + sum(len(file.raw_bytes) for file in first.adjuncts)
    sessions = corpus.iter_sessions(target_bytes=first_bytes + 1, seed=4)
    total = 0
    count = 0
    report = None
    try:
        while True:
            try:
                session = next(sessions)
            except StopIteration as completed:
                report = completed.value
                break
            count += 1
            total += sum(len(chunk) for chunk in session.chunks)
            for file in session.adjuncts:
                body_bytes = sum(len(chunk) for chunk in file.chunks)
                assert body_bytes == file.byte_length
                total += body_bytes
        assert report is not None
        assert count == report.generated_count == 2
        assert total == report.generated_bytes > first_bytes + 1
        assert report.requested_bytes == first_bytes + 1
        assert report.requested_count is None
    finally:
        sessions.close()
    zero = corpus.iter_sessions(target_bytes=0, seed=4)
    try:
        with pytest.raises(StopIteration) as empty:
            next(zero)
        assert empty.value.value.generated_count == empty.value.value.generated_bytes == 0
    finally:
        zero.close()


def test_fresh_build_byte_target_uses_the_actual_shared_session_driver(tmp_path: Path) -> None:
    from tempfile import TemporaryDirectory

    from devtools.fresh_build_bench.cli import _parser
    from devtools.fresh_build_bench.corpus import corpus_from_synthetic, verify_manifest

    corpus = _corpus(tmp_path, [None], (("middle.txt", 101),))
    first = corpus.generate_batch(count=1, seed=4).artifacts[0]
    first_bytes = len(first.raw_bytes) + sum(len(file.raw_bytes) for file in first.adjuncts)
    with TemporaryDirectory(prefix="synthetic-byte-target-", dir="/realm/tmp/work") as directory:
        root = Path(directory)
        manifest = corpus_from_synthetic(root, corpus=corpus, seed=4, target_bytes=first_bytes + 1)
        verify_manifest(root, manifest)
        assert manifest["parameters"]["sessions"] == 2
        assert manifest["parameters"]["target_bytes"] == first_bytes + 1
        assert manifest["file_count"] == 4
        assert manifest["total_bytes"] > first_bytes + 1
        args = _parser().parse_args(
            ["corpus", "synthetic", "--out", str(root), "--provider", "claude-code", "--bytes", str(first_bytes + 1)]
        )
        assert args.target_bytes == first_bytes + 1 and args.sessions is None


@pytest.mark.parametrize(
    ("contents", "files"),
    [
        (["Full output saved to: /neutral/middle.txt"], (("middle.txt.extra", 5),)),
        (
            ["Full output saved to: /neutral/middle.txt"],
            (("middle.a", 3), ("middle.txt.extra", 5), ("middle.z", 7)),
        ),
        (
            ["Full output saved to: /neutral/middle.aaa", "Full output saved to: /neutral/middle.zzz", None],
            (("middle.aaa.extra", 3), ("middle.m", 5), ("middle.zzz.extra", 7)),
        ),
    ],
)
def test_original_ambiguous_pointer_can_supply_persisted_stem_aliases(
    tmp_path: Path, contents: list[str | None], files: tuple[tuple[str, int], ...]
) -> None:
    corpus = _corpus(tmp_path, contents, files)
    expected = ["persisted" if name.count(".") > 1 else "direct" for name, _ in files]
    for seed in range(6):
        sessions = corpus.iter_sessions(count=1, seed=seed)
        try:
            session = next(sessions)
            record = cast(JSONDocument, json.loads(b"".join(session.chunks)))
            actual = []
            names = []
            for file in session.adjuncts:
                b"".join(file.chunks)
                match = corpus._relation_solver.tool_result_reference_for_filename(file.name)
                assert match is not None and match.tool_use_id == "middle"
                actual.append(match.reference_kind)
                names.append(file.name)
            assert actual == expected
            with ToolResultIndexAccumulator(disk_backed=True) as original:
                original.observe(record)
                for name, kind in zip(names, expected, strict=True):
                    match = original.reference_for_filename(name)
                    assert match is not None and match.tool_use_id == "middle"
                    assert match.reference_kind == kind

            with pytest.raises(StopIteration):
                next(sessions)
        finally:
            sessions.close()


def test_original_filename_width_joint_does_not_cross_marginals_inside_one_bin(tmp_path: Path) -> None:
    from polylogue.schemas.field_stats.distributions import _bucket_index

    assert _bucket_index(21.0) == _bucket_index(22.0)
    assert _bucket_index(25.0) == _bucket_index(26.0)
    corpus = _corpus(
        tmp_path,
        ["Full output saved to: /neutral/" + "a" * 25, "Full output saved to: /neutral/" + "b" * 26],
        (),
        identities=("a" * 21, "b" * 22),
    )
    for seed in range(12):
        sessions = corpus.iter_sessions(count=1, seed=seed)
        try:
            session = next(sessions)
            record = json.loads(b"".join(session.chunks))
            pairs = []
            for block in record["message"]["content"]:
                pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(block["content"])
                assert pointer is not None
                pairs.append((len(block["tool_use_id"]), len(Path(pointer.group(1)).name)))
            assert set(pairs) == {(21, 25), (22, 26)}
            assert list(session.adjuncts) == []
            with pytest.raises(StopIteration):
                next(sessions)
        finally:
            sessions.close()


def test_filename_width_joint_preserves_actual_extrema_and_rejects_malformed_points() -> None:
    from polylogue.schemas.field_stats.models import FilenameWidthStats

    first, second = FilenameWidthStats(), FilenameWidthStats()
    for pair in [(21, 25), (22, 26), (21, 26)]:
        first.observe(pair)
    second.observe((22, 25))
    first.merge(second)
    restored = FilenameWidthStats.from_payload(first.to_payload())
    assert restored.bins == first.bins and restored.extrema == first.extrema
    witnesses = {point for rows in restored.extrema.values() for point in rows}
    assert witnesses <= {(21, 25), (22, 26), (21, 26), (22, 25)}
    assert sum(count for _point, count in restored.bins.values()) == 4
    assert min(point[0] for point in witnesses) == 21 and max(point[0] for point in witnesses) == 22
    assert min(point[1] for point in witnesses) == 25 and max(point[1] for point in witnesses) == 26
    for point in ([0, 25], [21, -1], [True, 25], [21], [21, 25, 3]):
        with pytest.raises(ValueError):
            FilenameWidthStats.from_payload([{"count": 1, "points": [point]}])
