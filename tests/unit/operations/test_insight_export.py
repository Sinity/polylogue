"""Resident exports exhaust one pinned insight relation before filesystem delivery."""

from __future__ import annotations

import errno
import io
import json
import shutil
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path
from typing import IO, Any, cast

import pytest

from polylogue.analysis.archive import (
    ArchiveInsightModel,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
)
from polylogue.analysis.archive_models import ArchiveInsightProvenance
from polylogue.analysis.export_bundle_contracts import InsightExportBundleError, InsightExportBundleRequest
from polylogue.analysis.export_bundles import export_insight_bundle
from polylogue.analysis.insight_reads import iter_insight_rows, read_insight_page
from polylogue.operations.daemon_protocol import (
    InsightExportWireRequest,
    InsightExportWireResult,
    validate_operation_result,
)
from polylogue.operations.insight_export_contracts import (
    InsightExportRequest,
    InsightExportResult,
    decode_insight_export_result,
)
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder, materialize_session_insights


def _seed(root: Path) -> None:
    SessionBuilder(root / "index.db", "export-neutral").provider("codex").title("Export evidence").add_message(
        "m1", role="user", text="Neutral export evidence."
    ).save()
    materialize_session_insights(root / "index.db")


def test_resident_export_matches_canonical_tag_pages_and_version_two(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    with open_operation_read(root) as pinned:
        expected = read_insight_page(pinned.archive, SessionTagRollupQuery(limit=None))
    with running_daemon_operations(root) as stack:
        envelope = stack.client.operation(
            "insights.export_bundle", {"request": {"output_path": str(target), "insights": ["session_tag_rollups"]}}
        )
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        value = envelope["result"]
        validate_operation_result("insights.export_bundle", value)
        result = decode_insight_export_result(value)
        assert result.outcome.state == "ok"
        assert result.bundle.manifest.bundle_version == 2
        rows = [json.loads(line) for line in (target / "insights/session_tag_rollups.jsonl").read_text().splitlines()]
        assert [row["tag"] for row in rows] == [cast(SessionTagRollupInsight, row).tag for row in expected]
        assert [row["session_count"] for row in rows] == [
            cast(SessionTagRollupInsight, row).session_count for row in expected
        ]
        assert "origin:codex-session" in {row["tag"] for row in rows}
        assert result.bundle.manifest.insights[0].row_count == len(expected)
        assert (target / "schemas/session_tag_rollups.schema.json").exists()
        assert (target / "README.md").exists()
        assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_export_wire_is_closed_and_matches_canonical_contracts(tmp_path: Path) -> None:
    from pydantic import ValidationError

    assert InsightExportWireRequest.model_json_schema() == InsightExportRequest.model_json_schema()
    assert InsightExportWireResult.model_json_schema() == InsightExportResult.model_json_schema()
    for changed in ({"overwrite": 1}, {"unknown": True}, {"output_path": 2}, {"insights": [False]}):
        with pytest.raises(ValidationError):
            InsightExportWireRequest.model_validate({"request": {"output_path": str(tmp_path / "bundle"), **changed}})


@pytest.mark.parametrize("stage", ["row", "publication", "published"])
def test_export_baseexception_closes_reader_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from polylogue.analysis import export_bundles

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"

    class Cancelled(BaseException):
        pass

    closed = False
    original = iter_insight_rows

    # Preserve the original producer and observe its owned generator settlement.
    def observed(
        archive: ArchiveStore, request: ArchiveInsightModel, *, checkpoint: Callable[[], None]
    ) -> Generator[ArchiveInsightModel, None, None]:
        nonlocal closed
        stream = original(archive, request, checkpoint=checkpoint)
        try:
            for item in stream:
                if stage == "row":
                    raise Cancelled()
                yield item
        finally:
            stream.close()
            closed = True

    monkeypatch.setattr(export_bundles, "iter_insight_rows", observed)
    if stage in {"publication", "published"}:
        original_publish = export_bundles._publish_target

        def cancel(tmp: Path, request: InsightExportBundleRequest) -> None:
            assert (tmp / "manifest.json").exists()
            assert (tmp / "insights/session_profiles.jsonl").read_text()
            if stage == "published":
                original_publish(tmp, request)
            raise Cancelled()

        monkeypatch.setattr(export_bundles, "_publish_target", cancel)
    with open_operation_read(root) as pinned:
        with pytest.raises(Cancelled):
            export_insight_bundle(
                pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",))
            )
    assert closed
    assert target.exists() == (stage == "published")
    if stage == "published":
        assert (target / "manifest.json").exists()
        assert (target / "insights/session_profiles.jsonl").read_text()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_target_created_during_staging_is_preserved_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.analysis import export_bundles

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    original = export_bundles._publish_target

    def publish(tmp: Path, request: InsightExportBundleRequest) -> None:
        target.mkdir()
        (target / "marker").write_text("keep")
        original(tmp, request)

    monkeypatch.setattr(export_bundles, "_publish_target", publish)
    with open_operation_read(root) as pinned:
        with pytest.raises(InsightExportBundleError):
            export_insight_bundle(
                pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",))
            )
    assert (target / "marker").read_text() == "keep"
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_overwrite_restores_previous_on_failure_and_replaces_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    with open_operation_read(root) as pinned:
        export_insight_bundle(pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",)))
    original_bundle = {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert original_bundle[Path("manifest.json")]
    original_replace = Path.replace
    staged_contents: dict[str, bytes] = {}
    failed_publication = False

    def fail_stage_install(path: Path, destination: Path) -> Path:
        nonlocal failed_publication
        if path.parent == target.parent and path.name.startswith(f".{target.name}.tmp-") and destination == target:
            failed_publication = True
            staged_contents.update(
                {
                    staged.relative_to(path).as_posix(): staged.read_bytes()
                    for staged in path.rglob("*")
                    if staged.is_file()
                }
            )
            raise OSError(errno.EIO, "synthetic final export rename failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_stage_install)
    with open_operation_read(root) as pinned:
        with pytest.raises(OSError) as raised:
            export_insight_bundle(
                pinned.archive,
                InsightExportBundleRequest(output_path=target, insights=("profiles",), overwrite=True),
            )

    assert raised.value.errno == errno.EIO
    assert failed_publication
    assert staged_contents["manifest.json"]
    assert staged_contents["insights/session_profiles.jsonl"]
    assert {
        path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()
    } == original_bundle
    assert not list(tmp_path.glob(".bundle.tmp-*"))
    assert not list(tmp_path.glob(".bundle.previous-*"))

    monkeypatch.undo()
    with open_operation_read(root) as pinned:
        result = export_insight_bundle(
            pinned.archive,
            InsightExportBundleRequest(
                output_path=target, insights=("profiles",), overwrite=True, include_readme=False
            ),
        )
    assert result.output_path == target
    assert (target / "manifest.json").is_file()
    assert not (target / "README.md").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))
    assert not list(tmp_path.glob(".bundle.previous-*"))


def test_overwrite_keeps_recovery_bundle_if_restore_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    with open_operation_read(root) as pinned:
        export_insight_bundle(pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",)))
    original_bundle = {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    original_replace = Path.replace

    def fail_install_and_restore(path: Path, destination: Path) -> Path:
        if path.parent == target.parent and path.name.startswith(f".{target.name}.tmp-") and destination == target:
            raise OSError(errno.EIO, "synthetic final export rename failure")
        if path.parent == target.parent and path.name.startswith(f".{target.name}.previous-") and destination == target:
            raise OSError(errno.EIO, "synthetic previous bundle restore failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_install_and_restore)
    with open_operation_read(root) as pinned:
        with pytest.raises(BaseExceptionGroup, match="previous bundle is recoverable"):
            export_insight_bundle(
                pinned.archive,
                InsightExportBundleRequest(output_path=target, insights=("profiles",), overwrite=True),
            )

    backups = list(tmp_path.glob(".bundle.previous-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert {
        path.relative_to(backup): path.read_bytes() for path in backup.rglob("*") if path.is_file()
    } == original_bundle
    assert not target.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_jsonl_writer_does_not_retain_the_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.analysis.export_bundles import _write_insight_jsonl

    references: list[weakref.ReferenceType[SessionProfileInsight]] = []
    writes = 0
    target = tmp_path / "rows.jsonl"
    original_open = Path.open

    class Observed(io.TextIOBase):
        def __init__(self, stream: IO[str]) -> None:
            self.stream = stream

        def close(self) -> None:
            self.stream.close()
            super().close()

        def write(self, text: str) -> int:
            nonlocal writes
            writes += 1
            return self.stream.write(text)

    def opened(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> IO[Any]:
        stream = original_open(path, mode, *args, **kwargs)
        return cast(IO[Any], Observed(stream)) if path == target and mode == "w" else stream

    monkeypatch.setattr(Path, "open", opened)

    def rows() -> Generator[ArchiveInsightModel, None, None]:
        for number in range(1000):
            assert sum(ref() is not None for ref in references) <= 1
            assert writes == number * 2
            item = SessionProfileInsight(
                session_id=f"codex-session:row-{number}",
                logical_session_id=f"codex-session:row-{number}",
                origin="codex-session",
                provenance=ArchiveInsightProvenance(materializer_version=1),
            )
            references.append(weakref.ref(item))
            yield item

    assert _write_insight_jsonl(target, rows(), lambda: None) == 1000
    assert all(ref() is None for ref in references)
    with target.open() as stream:
        assert sum(1 for _ in stream) == 1000


def test_failed_forward_producer_withholds_its_entire_staged_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.analysis.archive import ArchiveInsightUnavailableError

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    original = ArchiveStore.iter_session_profile_insights
    reached = False

    def fail(self: ArchiveStore, **kwargs: Any) -> Generator[SessionProfileInsight, None, None]:
        nonlocal reached
        stream = original(self, **kwargs)
        try:
            for item in stream:
                yield item
                reached = True
                raise ArchiveInsightUnavailableError("neutral producer fault")
        finally:
            stream.close()

    monkeypatch.setattr(ArchiveStore, "iter_session_profile_insights", fail)
    with open_operation_read(root) as pinned:
        result = export_insight_bundle(
            pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",))
        )
    assert reached
    assert result.outcome.state == "degraded"
    gaps = result.outcome.detail["gaps"]
    assert isinstance(gaps, list)
    assert "insight_export_read_failed" in gaps
    assert result.manifest.insights[0].row_count == 0
    assert result.manifest.insights[0].errors
    assert (target / "insights/session_profiles.jsonl").read_text() == ""
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_original_read_context_cancellation_reaches_jsonl_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.analysis import export_bundles
    from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"
    context = QueryExecutionContext.create(timeout_s=None)
    reached = False

    def cancel_after_original_row(
        archive: ArchiveStore, request: ArchiveInsightModel, *, checkpoint: Callable[[], None]
    ) -> Generator[ArchiveInsightModel, None, None]:
        nonlocal reached
        stream = iter_insight_rows(archive, request, checkpoint=checkpoint)
        try:
            for item in stream:
                reached = True
                context.cancel()
                yield item
        finally:
            stream.close()

    monkeypatch.setattr(export_bundles, "iter_insight_rows", cancel_after_original_row)
    returned = False
    with pytest.raises(QueryCancelledError):
        with open_operation_read(root, execution_context=context) as pinned:
            export_insight_bundle(
                pinned.archive,
                InsightExportBundleRequest(output_path=target, insights=("profiles",)),
                checkpoint=pinned.archive.check_operation_read,
            )
            returned = True
    assert reached and not returned
    assert not target.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


@pytest.mark.parametrize("stage", ["prepare", "producer"])
def test_export_preserves_primary_and_cleanup_failure_with_staged_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from polylogue.analysis import export_bundles

    root = tmp_path / "archive"
    _seed(root)
    target = tmp_path / "bundle"

    class Cancelled(BaseException):
        pass

    primary = Cancelled()
    cleanup_error = OSError("Synthetic staging cleanup failure")
    residue: Path | None = None
    original_mkdir = Path.mkdir

    def fail_preparation(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "schemas" and path.parent.name.startswith(".bundle.tmp-"):
            raise primary
        original_mkdir(path, *args, **kwargs)

    def fail_producer(
        archive: ArchiveStore, request: ArchiveInsightModel, *, checkpoint: Callable[[], None]
    ) -> Generator[ArchiveInsightModel, None, None]:
        with closing(iter_insight_rows(archive, request, checkpoint=checkpoint)) as rows:
            for row in rows:
                yield row
                raise primary

    def fail_cleanup(path: Path) -> None:
        nonlocal residue
        residue = path
        assert path.is_dir()
        raise cleanup_error

    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    if stage == "prepare":
        monkeypatch.setattr(Path, "mkdir", fail_preparation)
    else:
        monkeypatch.setattr(export_bundles, "iter_insight_rows", fail_producer)
    with open_operation_read(root) as pinned:
        with pytest.raises(BaseExceptionGroup) as caught:
            export_insight_bundle(
                pinned.archive, InsightExportBundleRequest(output_path=target, insights=("profiles",))
            )
    assert caught.value.exceptions == (primary, cleanup_error)
    assert not target.exists()
    assert residue is not None and residue.is_dir()
    assert list(tmp_path.glob(".bundle.tmp-*")) == [residue]
    assert (residue / "insights").is_dir()
    if stage == "producer":
        assert (residue / "insights/session_profiles.jsonl").read_text()


def test_profileless_thread_export_validates_against_its_bundled_schema(tmp_path: Path) -> None:
    import sqlite3

    import jsonschema

    from polylogue.analysis.archive import ThreadInsight

    root = tmp_path / "archive"
    _seed(root)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles")
    target = tmp_path / "bundle"
    with open_operation_read(root) as pinned:
        result = export_insight_bundle(
            pinned.archive, InsightExportBundleRequest(output_path=target, insights=("threads",))
        )
    assert result.manifest.insights[0].row_count == 1
    schema = json.loads((target / "schemas/threads.schema.json").read_text())["schema"]
    rows = [json.loads(line) for line in (target / "insights/threads.jsonl").read_text().splitlines()]
    assert rows[0]["provenance"]["materializer_version"] is None
    for row in rows:
        jsonschema.validate(row, schema)
        ThreadInsight.model_validate(row)
