"""Component benchmarks over a sealed fresh-build corpus.

The end-to-end run proves a total; these isolate one production stage each so
an optimisation can be iterated in seconds. Every component calls the same
function the daemon calls, on the corpus's own files:

* ``blob`` -- ``BlobStore.write_from_path``, the acquisition copy + hash +
  fsync + publish each admitted file pays.

Results print as one JSON document per component with per-origin seconds,
bytes and derived MiB/s. The corpus is verified against its seal before and
after the timed work; a file that changed meanwhile fails the run, because the
summary attributes throughput to the sealed byte counts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from builtins import BaseExceptionGroup
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.scenarios.workload import WorkloadPhaseObservation

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from devtools.fresh_build_bench.corpus import load_manifest, refuse_inside_checkout, verify_manifest

_PROVIDER_BY_ORIGIN = {
    "claude-code": "claude-code",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "chatgpt": "chatgpt",
    "claude-ai": "claude-ai",
}


def _corpus_files(
    corpus: Path,
    manifest: dict[str, Any],
    origins: Iterable[str] | None,
    limit: int | None,
) -> list[tuple[Path, str, int]]:
    verify_manifest(corpus, manifest)
    wanted = set(origins) if origins else None
    files = []
    for relative, size, _digest, origin in manifest["files"]:
        if origin not in _PROVIDER_BY_ORIGIN or (wanted is not None and origin not in wanted):
            continue
        path = corpus / relative
        # Acquisition stores every retained sidecar's bytes, so the blob
        # component keeps sidecars beside their transcripts.
        files.append((path, origin, size))
    files.sort(key=lambda item: str(item[0]))
    selected = files[:limit] if limit else files
    if not selected:
        # A timing over no files measured no production work.
        raise SystemExit(f"no corpus files match the requested selection (origins={sorted(wanted or ())})")
    return selected


def _summarise(rows: list[tuple[str, int, float]], wall: float, extra: dict[str, Any]) -> dict[str, Any]:
    by_origin: dict[str, dict[str, float]] = defaultdict(lambda: {"files": 0, "bytes": 0, "seconds": 0.0})
    for origin, size, seconds in rows:
        entry = by_origin[origin]
        entry["files"] += 1
        entry["bytes"] += size
        entry["seconds"] += seconds
    total_bytes = sum(size for _origin, size, _seconds in rows)
    return {
        **extra,
        "wall_s": round(wall, 3),
        "files": len(rows),
        "bytes": total_bytes,
        "mib_per_s_wall": round(total_bytes / 2**20 / wall, 3) if wall else None,
        "by_origin": {
            origin: {
                "files": int(entry["files"]),
                "mib": round(entry["bytes"] / 2**20, 2),
                "seconds": round(entry["seconds"], 3),
                "mib_per_s": round(entry["bytes"] / 2**20 / entry["seconds"], 3) if entry["seconds"] else None,
            }
            for origin, entry in sorted(by_origin.items())
        },
    }


def _timed_map(
    work: Callable[[tuple[Path, str, int]], Callable[[], dict[str, int]]],
    files: list[tuple[Path, str, int]],
    workers: int,
) -> tuple[list[tuple[str, int, float]], float, dict[str, int]]:
    """Time ``work`` per file; the callable it returns runs after the timer.

    That callable collects counts (reading prepared artifacts back, say),
    which is the caller's consumption, not the stage being measured.
    """
    counts: dict[str, int] = defaultdict(int)

    def run(item: tuple[Path, str, int]) -> tuple[str, int, float, float, Callable[[], dict[str, int]]]:
        began = time.perf_counter()
        finish = work(item)
        ended = time.perf_counter()
        return item[1], item[2], ended - began, ended, finish

    rows: list[tuple[str, int, float]] = []
    last_ended = began = time.perf_counter()

    def consume(result: tuple[str, int, float, float, Callable[[], dict[str, int]]]) -> None:
        # ``finish()`` (artifact rereads, session/message counting) is the
        # caller's own consumption, not the timed production stage. It runs
        # as each file completes, so a prepared artifact's scratch is released
        # while the rest are still being prepared, and the stage's wall is
        # read from the workers' own end times rather than from this loop.
        nonlocal last_ended
        origin, size, seconds, ended, finish = result
        rows.append((origin, size, seconds))
        last_ended = max(last_ended, ended)
        for key, value in finish().items():
            counts[key] += value

    if workers <= 1:
        for item in files:
            consume(run(item))
    else:
        # At most ``2 * workers`` files are submitted and unconsumed at once:
        # a preparation holds its attempt scratch until consumed, so faster
        # workers than one consumer must not queue the whole corpus.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="component") as pool:
            pending_items = iter(files)
            in_flight = {pool.submit(run, item) for item in itertools.islice(pending_items, 2 * workers)}
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
                    for item in itertools.islice(pending_items, 1):
                        in_flight.add(pool.submit(run, item))
    wall = last_ended - began
    return rows, wall, dict(counts)


def bench_blob(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.storage.blob_store import BlobStore

    manifest = load_manifest(corpus)
    files = _corpus_files(corpus, manifest, origins, limit)
    store = BlobStore(scratch / "blob")

    def work(item: tuple[Path, str, int]) -> Callable[[], dict[str, int]]:
        store.write_from_path(item[0])
        return dict

    rows, wall, counts = _timed_map(work, files, workers)
    verify_manifest(corpus, manifest)
    return _summarise(rows, wall, {"component": "blob", "workers": workers, "counts": counts})


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _product_tree(candidate: Path) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(candidate), "rev-parse", "HEAD:polylogue"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _runtime_identity() -> dict[str, Any]:
    return {
        "path": str(Path(sys.executable).resolve()),
        "sha256": _sha256(Path(sys.executable)),
        "version": sys.version,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
    }


def _require_runtime(expected: dict[str, Any]) -> dict[str, Any]:
    actual = _runtime_identity()
    if actual != expected:
        raise ValueError("component runtime or GIL state differs from frozen launch")
    return actual


def _write_document(path: Path, document: dict[str, Any]) -> None:
    # Never replace a constituent or its launch declaration.
    with path.open("x") as stream:
        stream.write(json.dumps(document, indent=1, sort_keys=True) + "\n")


def _validation_observation(
    timing: dict[str, float], sample_count: int, phase: str = "validation"
) -> WorkloadPhaseObservation:
    from polylogue.scenarios.workload import MeasurementScope, WorkloadPhaseObservation

    return WorkloadPhaseObservation(
        name=phase,
        measurement_scope=MeasurementScope.PROCESS,
        wall_ms=timing["wall_s"] * 1000,
        cpu_ms=timing["process_cpu_s"] * 1000,
        progress_completed=sample_count,
        progress_total=sample_count,
        unavailable=("peak_rss_bytes", "sqlite_vm_steps"),
    )


def _require_loose_jsonl_recipe(recipe: dict[str, Any], stage: str) -> None:
    if stage == "page":
        if recipe.get("route") != {"domain": "synthetic-retained-page", "validation_mode": "advisory"}:
            raise ValueError("page requires the explicit synthetic acquired retained-page domain")
        if recipe.get("provider") != "codex":
            raise ValueError("synthetic retained pages support declared Codex inputs only")
        for name in ("workers", "queue_units", "queue_bytes"):
            if type(recipe.get(name)) is not int or recipe[name] <= 0:
                raise ValueError(f"page requires explicit positive {name}")
        return
    expected = {
        "domain": "complete-loose-jsonl",
        "validation_mode": "advisory",
        "jsonl": True,
        "captured_zip_coordinate": None,
        "schema_resolution": None,
        "schema_resolution_is_explicit": False,
    }
    if recipe.get("route") != expected:
        raise ValueError("recipe must explicitly declare the supported complete advisory loose-JSONL route")
    if stage == "preparation" and (recipe.get("provider") != "codex" or recipe.get("sidecar_context") != "absent"):
        raise ValueError("standalone preparation supports only declared Codex inputs without sidecar context")


def _prepare_observation(
    owned: Path, recipe: dict[str, Any], scratch: Path, prefix: int | None
) -> tuple[dict[str, Any], dict[str, float]]:
    from contextlib import closing

    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.dispatch import is_stream_record_provider
    from polylogue.sources.fallback_identity import fallback_session_id
    from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
    from polylogue.sources.sidecar_evidence import CapturedSidecarResolver
    from polylogue.storage.sqlite.session_shard import open_session_shard

    if not is_stream_record_provider(recipe["source_path"], recipe["provider"]):
        raise ValueError("standalone preparation requires a declared native streamed transcript coordinate")
    wall, thread, cpu = time.perf_counter(), time.thread_time(), time.process_time()
    attempt = prepare_jsonl_blob(
        str(owned),
        recipe["source_path"],
        recipe["provider"],
        fallback_session_id(recipe["source_path"], recipe["raw_id"]),
        is_stream=is_stream_record_provider(recipe["source_path"], recipe["provider"]),
        shard_directory=str(scratch / "preparation"),
        source_sha256=recipe["input_sha256"],
        parse_prefix_size=prefix,
        strict_jsonl_records=True,
        sidecar_resolver=CapturedSidecarResolver({}),
    )
    timing = {
        "wall_s": time.perf_counter() - wall,
        "thread_cpu_s": time.thread_time() - thread,
        "process_cpu_s": time.process_time() - cpu,
    }
    try:
        if (
            attempt.error
            or attempt.deferred
            or attempt.unsupported_shape
            or attempt.missing_profile_identity
            or attempt.codex_state_kind is not None
        ):
            raise ValueError("production preparation did not complete the declared input")
        if attempt.blob_hash != recipe["input_sha256"] or attempt.parsed_prefix_size != prefix:
            raise ValueError("prepared result differs from the declared retained revision or accepted prefix")
        rows = open_session_shard(attempt.shard_path).sessions if attempt.shard_path is not None else ()
        # These are canonical lowered rows, not guessed counts of parser fields.
        normalized = {
            "sessions": len(rows),
            "messages": sum(row.message_row_count for row in rows),
            "blocks": sum(row.block_row_count for row in rows),
            "events": 0,
        }
        hashes = []
        with closing(attempt.iter_sessions()) as sessions:
            for session in sessions:
                hashes.append(str(session_content_hash(session)))
                normalized["events"] += len(session.session_events)
        if len(hashes) != normalized["sessions"]:
            raise ValueError("complete prepared read disagrees with the sealed session count")
        return {
            "preparation": {
                "state": "complete",
                "complete_success": True,
                "consumed_session_count": len(hashes),
                "input_sha256": attempt.blob_hash,
                "accepted_prefix_size": prefix,
                "parsed_prefix_size": attempt.parsed_prefix_size,
            },
            "normalized": {**normalized, "count_basis": "production-preparation-before-lineage"},
            "session_content_hashes": hashes,
            "parsed_prefix_size": attempt.parsed_prefix_size,
            "resolved_provider": None if attempt.resolved_provider is None else attempt.resolved_provider.value,
        }, timing
    finally:
        # The original attempt retains custody until all read consumers settle.
        attempt.discard()


def _page_observation(owned: Path, recipe: dict[str, Any], scratch: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """Use the selected retained owner, including its actual preparation/publication."""
    import asyncio
    import math

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
    from tests.infra.selected_raw_preparation import acquire_independent_codex_raws
    from tests.infra.workload_artifacts import ImmutableTreeArtifact, clone_immutable_tree, seal_fixture_tree

    generator_path = sys.modules[acquire_independent_codex_raws.__module__].__file__
    if generator_path is None:
        raise ValueError("synthetic input generator source is unbound")
    generator_digest = _sha256(Path(generator_path))
    fixture = json.loads(owned.read_text())
    if (
        not isinstance(fixture, dict)
        or set(fixture) != {"count", "messages", "text_size", "namespace"}
        or any(type(fixture[name]) is not int or fixture[name] < 1 for name in ("count", "messages", "text_size"))
        or not isinstance(fixture["namespace"], str)
        or not fixture["namespace"]
        or Path(fixture["namespace"]).name != fixture["namespace"]
        or fixture["namespace"] in {".", ".."}
    ):
        raise ValueError("page input must declare positive independent synthetic Codex fixture dimensions")

    async def run() -> tuple[dict[str, Any], dict[str, float]]:
        template, archive = scratch / "acquired-template", scratch / "archive"
        await run_archive_fixture_write(template, lambda: bootstrap_archive_root(template))
        raws = await run_archive_fixture_write(template, lambda: acquire_independent_codex_raws(template, **fixture))
        await asyncio.to_thread(seal_fixture_tree, template)
        artifact = ImmutableTreeArtifact.adopt(template, key="retained-page:" + recipe["input_sha256"])
        archive.mkdir(mode=0o700)

        def clone() -> None:
            with clone_immutable_tree(artifact, archive):
                pass

        await asyncio.to_thread(clone)
        kernel = BoundedComputeAdapter(
            max_workers=recipe["workers"], queue_units=recipe["queue_units"], queue_bytes=recipe["queue_bytes"]
        )
        coordinator = DaemonWriteCoordinator(archive_root=archive)
        primary: BaseException | None = None
        try:
            owner = RawObservationConvergenceOwner(
                archive,
                compute_adapter=kernel,
                write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
                write_coordinator=coordinator,
            )
            wall, thread, cpu = time.perf_counter(), time.thread_time(), time.process_time()
            result = await owner.materialize_retained_raw_ids(raws)
            result.outcome.require_complete()
            timing = {
                "wall_s": time.perf_counter() - wall,
                "thread_cpu_s": time.thread_time() - thread,
                "process_cpu_s": time.process_time() - cpu,
            }
        except BaseException as failure:
            primary = failure
            raise
        finally:
            failures: list[BaseException] = []
            try:
                kernel.shutdown(wait=True)
            except BaseException as failure:
                failures.append(failure)
            try:
                if not await coordinator.shutdown(timeout=math.inf):
                    raise RuntimeError("retained page writer did not physically settle")
            except BaseException as failure:
                failures.append(failure)
            if failures:
                raise BaseExceptionGroup(
                    "retained page physical cleanup failed", [*([] if primary is None else [primary]), *failures]
                )
        from devtools.fresh_build_bench.report import archive_census, output_fingerprint

        with ArchiveStore.open_existing(archive, read_only=True) as read:
            index_path = str(read.index_db_path)
        fingerprint = output_fingerprint(archive, index_path, scratch / "fingerprint")
        census = archive_census(archive, index_path)
        if (census.get("rows") or {}).get("sessions") != fixture["count"] or (census.get("rows") or {}).get(
            "messages"
        ) != fixture["count"] * fixture["messages"]:
            raise ValueError("complete independent page did not produce its declared session/message population")
        if _sha256(Path(generator_path)) != generator_digest:
            raise ValueError("synthetic input generator changed during the page")
        return {
            "route": "retained-page-materialization",
            "state": "complete",
            "raw_count": len(raws),
            "census": {name: census[name] for name in ("rows", "messages_fts_rows", "fts_indexable_rows")},
            "physical_census": {
                name: value
                for name, value in census.items()
                if name not in {"rows", "messages_fts_rows", "fts_indexable_rows"}
            },
            "output_fingerprint": fingerprint,
            "input_generator_sha256": generator_digest,
            "cleanup_scope": "owned compute and writer settled; external descendants unknown",
        }, timing

    return asyncio.run(run())


def bench_validation(
    candidate: Path,
    input_path: Path,
    recipe_path: Path,
    schema_root: Path,
    scratch: Path,
    launch_path: Path | None = None,
    ordinal: int | None = None,
    arm: str | None = None,
    stage: str = "validation",
    diagnostic: str | None = None,
) -> dict[str, Any]:
    """Complete production retained validation in a fresh isolated process.

    The recipe is a bound input, not permission to ignore output differences.
    Its source coordinate controls schema resolution and must be declared.
    """
    from devtools.fresh_build_bench.run import candidate_identity, candidate_stamp

    if stage not in {"validation", "preparation", "page"}:
        raise ValueError("unsupported retained component stage")
    if diagnostic is not None and (stage != "validation" or launch_path is not None):
        raise ValueError("diagnostics support direct validation only; matched observations remain unprofiled")
    diagnostic_capture = None
    candidate = candidate.resolve()
    identity, stamp = candidate_identity(candidate), candidate_stamp(candidate)
    if identity["dirty"]:
        raise ValueError("component candidate must be a clean checkout")
    product_tree = _product_tree(candidate)
    expected_runtime = _runtime_identity()
    launch_binding = None
    launch: dict[str, Any] = {}
    if launch_path is not None:
        launch = json.loads(launch_path.read_text())
        if arm not in {"A", "B"} or ordinal is None or ordinal < 1 or ordinal > 4 * launch["repetitions"]:
            raise ValueError("invalid component launch coordinate")
        if (
            ("ABBA" * launch["repetitions"])[ordinal - 1] != arm
            or identity != launch["candidates"][arm]
            or product_tree != launch["product_trees"][arm]
            or stage != launch.get("stage", "validation")
        ):
            raise ValueError("component candidate differs from frozen launch")
        _require_runtime(launch["runtime"])
        if _sha256(Path(__file__)) != launch["runner_sha256"]:
            raise ValueError("component runner/runtime differs from frozen launch")
        launch_binding = {"sha256": _sha256(launch_path), "ordinal": ordinal, "arm": arm}
    recipe_digest = _sha256(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    _require_loose_jsonl_recipe(recipe, stage)
    input_digest = _sha256(input_path)
    if input_digest != recipe["input_sha256"]:
        raise ValueError("input does not match the declared complete revision")
    if Path(sys.executable).resolve() != Path(recipe["expected_python"]).resolve():
        raise ValueError("component interpreter differs from declared runtime")
    if launch_path is not None and (input_digest != launch["input_sha256"] or recipe_digest != launch["recipe_sha256"]):
        raise ValueError("component inputs differ from frozen launch")
    config_path = Path(recipe["config_path"])
    if _sha256(config_path) != recipe["config_sha256"]:
        raise ValueError("declared configuration changed")
    if any(name == "polylogue" or name.startswith("polylogue.") for name in sys.modules):
        raise ValueError("component product must be imported only after candidate binding")
    sys.path.insert(0, str(candidate))
    sys.dont_write_bytecode = True
    for name in ("home", "config", "data", "cache", "state", "runtime", "tmp", "signatures"):
        (scratch / name).mkdir(mode=0o700)
    owned = scratch / "retained-input.jsonl"
    shutil.copyfile(input_path, owned)
    config = scratch / "config/polylogue.toml"
    shutil.copyfile(config_path, config)
    os.environ.update(
        {
            "HOME": str(scratch / "home"),
            "TMPDIR": str(scratch / "tmp"),
            "POLYLOGUE_CONFIG": str(config),
            "POLYLOGUE_SITE_CONFIG": "",
            "POLYLOGUE_ARCHIVE_ROOT": str(scratch / "absent-archive"),
            "POLYLOGUE_SINEX_MODE": "off",
            "POLYLOGUE_TRACE_WORK": "0",
            "POLYLOGUE_LOG_FORMAT": "json",
            "POLYLOGUE_LOG_FILE": str(scratch / "events.jsonl"),
            **{"XDG_" + name.upper() + "_HOME": str(scratch / name) for name in ("config", "data", "cache", "state")},
            "XDG_RUNTIME_DIR": str(scratch / "runtime"),
        }
    )
    for name in ("POLYLOGUE_BENCH_SAMPLER", "POLYLOGUE_BENCH_POLICY_OBSERVER", "POLYLOGUE_TRACE_COMPUTE"):
        os.environ.pop(name, None)
    import tempfile

    tempfile.tempdir = str(scratch / "tmp")
    from polylogue.core.enums import Provider, ValidationMode
    from polylogue.schemas import RetainedValidationReuse, validate_retained_document
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.sources.live.batch_support import jsonl_parse_prefix_size_of_handle
    from polylogue.sources.revision_backfill import _retained_validation_input

    provider = Provider(recipe.get("provider", "codex"))
    registry = SchemaRegistry(storage_root=schema_root)

    def inventory(snapshot: Any) -> list[dict[str, Any]]:
        return [
            {
                "files": [
                    {"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                    for name, payload in files
                ]
            }
            for _root, files in snapshot.roots
        ]

    if stage == "page":
        with registry.current_provider_snapshot(provider) as snapshot:
            schemas = inventory(snapshot)
        expected = [{"files": entry["files"]} for entry in recipe["current_schema_snapshot_roots"]]
        with SchemaRegistry().current_provider_snapshot(provider) as snapshot:
            if inventory(snapshot) != schemas or schemas != expected:
                raise ValueError("actual retained owner schema inputs differ from the declared page snapshot")
        output, timing = _page_observation(owned, recipe, scratch)
        completed = output["raw_count"]
    else:
        with owned.open("rb") as stream:
            prefix = jsonl_parse_prefix_size_of_handle(stream)
        if prefix != recipe["accepted_prefix_size"]:
            raise ValueError("declared complete accepted JSONL prefix changed")

        with registry.current_provider_snapshot(provider) as snapshot:
            schemas = inventory(snapshot)
            expected = [{"files": entry["files"]} for entry in recipe["current_schema_snapshot_roots"]]
            if schemas != expected:
                raise ValueError("schema snapshot differs from the declared input")
            reuse = RetainedValidationReuse()
            from contextlib import nullcontext

            capture_context: Any = nullcontext()
            if diagnostic is not None:
                from devtools.fresh_build_bench.diagnostics import BoundModule, capture_diagnostic

                authority = {}
                for name, module in list(sys.modules.items()):
                    loaded_path = getattr(module, "__file__", None)
                    if name.startswith("polylogue") and loaded_path:
                        path = Path(loaded_path).resolve()
                        if not path.is_relative_to(candidate):
                            raise ValueError("diagnostic module is outside the bound candidate")
                        authority[str(path)] = BoundModule(name, _sha256(path))
                capture_context = capture_diagnostic(
                    diagnostic,
                    output_directory=scratch / "diagnostic",
                    bindings={
                        "candidate": identity,
                        "product_tree": product_tree,
                        "runtime": expected_runtime,
                        "input_sha256": input_digest,
                        "recipe_sha256": recipe_digest,
                        "schema_inventory_digest": hashlib.sha256(
                            json.dumps(schemas, sort_keys=True).encode()
                        ).hexdigest(),
                        "component": "retained-validation",
                        "runner_sha256": _sha256(Path(__file__)),
                    },
                    module_authority=authority,
                )
            with capture_context as diagnostic_capture:
                wall, thread, cpu = time.perf_counter(), time.thread_time(), time.process_time()
                with _retained_validation_input(owned, prefix) as (validation_path, accepted_prefix_size):
                    verdict = validate_retained_document(
                        provider,
                        validation_path,
                        mode=ValidationMode.ADVISORY,
                        raw_id=recipe["raw_id"],
                        revision_sha256=input_digest,
                        evidence_id=recipe["evidence_id"],
                        source_path=recipe["source_path"],
                        jsonl=True,
                        accepted_prefix_size=accepted_prefix_size,
                        captured_zip_coordinate=None,
                        schema_resolution=None,
                        schema_resolution_is_explicit=False,
                        registry=snapshot.reader(),
                        signature_directory=scratch / "signatures",
                        reuse=reuse,
                    )
                timing = {
                    "wall_s": time.perf_counter() - wall,
                    "thread_cpu_s": time.thread_time() - thread,
                    "process_cpu_s": time.process_time() - cpu,
                }
        with registry.current_provider_snapshot(provider) as snapshot:
            if inventory(snapshot) != schemas:
                raise ValueError("schema inputs changed during validation")
        output = {
            "verdict": {
                name: getattr(verdict, name)
                for name in ("sample_count", "invalid_count", "error_count", "drift_count", "strict_refusal")
            },
            "status": verdict.status.value,
            "mode": verdict.mode.value,
            "schema_resolution": None
            if verdict.schema_resolution is None
            else dataclasses.asdict(verdict.schema_resolution),
            "schema_inventory": schemas,
            "reuse_outcome": reuse.outcome,
        }
        if stage == "preparation":
            # Validation remains independently bound prerequisite evidence; the
            # reported preparation timing excludes validation and read consumption.
            prepared_output, timing = _prepare_observation(owned, recipe, scratch, prefix)
            output.update(prepared_output)
        completed = verdict.sample_count if stage == "validation" else output["normalized"]["sessions"]
    if (
        candidate_identity(candidate) != identity
        or candidate_stamp(candidate) != stamp
        or _sha256(input_path) != input_digest
        or _sha256(owned) != input_digest
        or _sha256(recipe_path) != recipe_digest
        or _sha256(config_path) != recipe["config_sha256"]
    ):
        raise ValueError("component inputs or candidate changed during execution")
    with registry.current_provider_snapshot(provider) as snapshot:
        if inventory(snapshot) != schemas:
            raise ValueError("schema inputs changed during component execution")
    if stage == "page":
        with SchemaRegistry().current_provider_snapshot(provider) as snapshot:
            if inventory(snapshot) != schemas:
                raise ValueError("retained owner schema inputs changed during page execution")
    from polylogue.scenarios.workload import (
        MeasurementScope,
        WorkloadEnvelopeSpec,
        WorkloadInputRef,
        WorkloadReceipt,
        WorkloadRunStatus,
    )

    spec = WorkloadEnvelopeSpec(
        workload_id="retained-" + stage,
        family_id="fresh-build-components",
        version=1,
        inputs=(WorkloadInputRef(input_id=input_digest, profile_id=recipe_digest),),
        phases=(stage,),
        measurement_scope=MeasurementScope.PROCESS,
    )
    runtime = _require_runtime(expected_runtime)
    observation = _validation_observation(timing, completed, stage)
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id=identity["git_sha"],
        runtime_id=runtime["sha256"],
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=(observation,),
        evidence_refs=(input_digest, recipe_digest),
        notes=(
            "Complete retained component only; no whole-daemon terminal qualification.",
            "Fresh reuse object; OS cache state is uncontrolled. External cleanup is unknown.",
        ),
    )
    workload_payload = receipt.to_payload()
    physical_census = output.pop("physical_census", None)
    runtime = _require_runtime(expected_runtime)
    imports = {}
    for name, module in list(sys.modules.items()):
        path_value = getattr(module, "__file__", None)
        if name.startswith("polylogue") and path_value:
            path = Path(path_value).resolve()
            if not path.is_relative_to(candidate):
                raise ValueError(f"product module imported outside the bound candidate: {name}")
            imports[name] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "format": "polylogue.component-receipt.v1",
        "component": "retained-" + stage,
        "candidate": identity,
        "product_tree": product_tree,
        "runtime": runtime,
        "launch_binding": launch_binding,
        "imported_product_modules": imports,
        "input_sha256": input_digest,
        "recipe_sha256": recipe_digest,
        "output": output,
        "physical_census": physical_census,
        "diagnostic": None
        if diagnostic_capture is None
        else {
            "mode": diagnostic,
            "manifest_path": str(diagnostic_capture.manifest_path),
            "manifest_sha256": diagnostic_capture.manifest_sha256,
            "qualification": "instrumented observation; separate from unprofiled matched timing",
        },
        "timing": timing,
        "workload_receipt": workload_payload,
        "load_average": list(os.getloadavg()),
        "process_lifetime_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "RSS is process lifetime high-water, not invocation peak.",
            "Component-only descriptive evidence; no significance or acceptance.",
        ],
    }


def run_component_experiment(
    *,
    before: Path,
    after: Path,
    input_path: Path,
    recipe: Path,
    schema_root: Path,
    scratch: Path,
    repetitions: int = 1,
    stage: str = "validation",
) -> dict[str, Any]:
    from devtools.fresh_build_bench.experiment import matched_schedule, run_matched_sequence
    from devtools.fresh_build_bench.run import candidate_identity

    if stage not in {"validation", "preparation", "page"}:
        raise ValueError("unsupported matched component stage")
    candidates = {"A": before.resolve(), "B": after.resolve()}
    identities = {arm: candidate_identity(path) for arm, path in candidates.items()}
    if any(identity["dirty"] for identity in identities.values()):
        raise ValueError("matched components require clean candidates")
    runner = Path(__file__).resolve()
    launch = {
        "format": "polylogue.component-launch.v1",
        "order": "ABBA",
        "stage": stage,
        "repetitions": repetitions,
        "candidates": identities,
        "product_trees": {arm: _product_tree(path) for arm, path in candidates.items()},
        "input_sha256": _sha256(input_path),
        "recipe_sha256": _sha256(recipe),
        "runner_sha256": _sha256(runner),
        "runtime": _runtime_identity(),
    }
    # Validate the entire schedule before creating any output.
    schedule = list(matched_schedule("ABBA", repetitions))
    _write_document(scratch / "launch.json", launch)

    def invoke(ordinal: int, arm: str) -> Path | None:
        output = scratch / f"{ordinal:03d}-{arm}"
        command = [
            sys.executable,
            "-B",
            str(runner),
            stage,
            "--candidate",
            str(candidates[arm]),
            "--input",
            str(input_path),
            "--recipe",
            str(recipe),
            "--schema-root",
            str(schema_root),
            "--scratch",
            str(output),
            "--launch",
            str(scratch / "launch.json"),
            "--ordinal",
            str(ordinal),
            "--arm",
            arm,
        ]
        with (
            (scratch / f"{ordinal:03d}-{arm}.stdout").open("xb") as stdout,
            (scratch / f"{ordinal:03d}-{arm}.stderr").open("xb") as stderr,
        ):
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            return None
        return output / "receipt.json"

    def valid(receipt: dict[str, Any]) -> bool:
        if receipt["candidate"] not in identities.values():
            raise ValueError("executed component candidate differs from launch")
        if receipt["input_sha256"] != launch["input_sha256"] or receipt["recipe_sha256"] != launch["recipe_sha256"]:
            raise ValueError("executed component inputs differ from launch")
        return True

    samples = run_matched_sequence(order="ABBA", repetitions=repetitions, invoke=invoke, valid=valid)
    for sample in samples:
        if "receipt" in sample and sample["receipt"]["candidate"] != identities[sample["arm"]]:
            raise ValueError("executed component arm differs from launch")
    if _sha256(runner) != launch["runner_sha256"]:
        raise ValueError("component runner changed during experiment")
    complete = len(samples) == len(schedule) and all("receipt" in sample for sample in samples)
    reduction = _reduce_components(samples, complete=complete)
    document = {
        "format": "polylogue.component-experiment.v1",
        "launch_sha256": _sha256(scratch / "launch.json"),
        **reduction,
        "samples": [{key: value for key, value in sample.items() if key != "receipt"} for sample in samples],
        "inference": "descriptive only; ranges are not confidence intervals; no daemon acceptance",
    }
    _write_document(scratch / "experiment.json", document)
    return document


def _reduce_components(samples: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    outputs = {json.dumps(sample["receipt"]["output"], sort_keys=True) for sample in samples if "receipt" in sample}
    metrics = {}
    for metric in ("wall_s", "thread_cpu_s", "process_cpu_s"):
        arms = {}
        for arm in ("A", "B"):
            values = [
                sample["receipt"]["timing"][metric]
                for sample in samples
                if sample["arm"] == arm and "receipt" in sample
            ]
            median = statistics.median(values) if values else None
            arms[arm] = {
                "raw": values,
                "median": median,
                "mad": statistics.median(abs(value - median) for value in values) if median is not None else None,
                "range": [min(values), max(values)] if values else None,
            }
        metrics[metric] = arms
    product_trees = {sample["receipt"].get("product_tree") for sample in samples if "receipt" in sample}
    imported = {
        json.dumps(
            {name: row["sha256"] for name, row in sample["receipt"].get("imported_product_modules", {}).items()},
            sort_keys=True,
        )
        for sample in samples
        if "receipt" in sample
    }
    treatment = "insufficient-executed-treatment-proof"
    if None not in product_trees and len(product_trees) == 1:
        treatment = "A/A-product-content-calibration"
    elif None not in product_trees and len(product_trees) == 2 and len(imported) > 1:
        treatment = "different-executed-import-content"
    return {
        "complete": complete,
        "equivalent_outputs": complete and len(outputs) == 1,
        "metrics": metrics,
        "treatment": treatment,
    }


def refresh_component_experiment(path: Path) -> dict[str, Any]:
    """Verify original launch and constituents without rewriting evidence."""
    document = json.loads(path.read_text())
    launch_path = path.parent / "launch.json"
    if _sha256(launch_path) != document["launch_sha256"]:
        raise ValueError("component launch changed")
    launch = json.loads(launch_path.read_text())
    from devtools.fresh_build_bench.experiment import matched_schedule

    schedule = list(matched_schedule(launch["order"], launch["repetitions"]))
    samples = document["samples"]
    if len(samples) > len(schedule):
        raise ValueError("component sequence exceeds frozen launch")
    loaded = []
    for expected, sample in zip(schedule, samples, strict=False):
        if (sample["ordinal"], sample["arm"]) != expected or "path" not in sample:
            raise ValueError("component sequence incomplete or changed")
        receipt_path = Path(sample["path"])
        if (
            receipt_path.resolve()
            != (path.parent / f"{sample['ordinal']:03d}-{sample['arm']}" / "receipt.json").resolve()
        ):
            raise ValueError("component constituent outside its original launch coordinate")
        if _sha256(receipt_path) != sample["sha256"]:
            raise ValueError("component constituent changed")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["launch_binding"]
            != {"sha256": document["launch_sha256"], "ordinal": sample["ordinal"], "arm": sample["arm"]}
            or receipt["candidate"] != launch["candidates"][sample["arm"]]
            or receipt["product_tree"] != launch["product_trees"][sample["arm"]]
            or receipt["input_sha256"] != launch["input_sha256"]
            or receipt["recipe_sha256"] != launch["recipe_sha256"]
            or receipt["runtime"] != launch["runtime"]
        ):
            raise ValueError("component launch binding differs from actual receipt")
        loaded.append({**sample, "receipt": receipt})
    return {
        **document,
        **_reduce_components(loaded, complete=len(samples) == len(schedule)),
        "verified_original_constituents": True,
    }


def diagnostic_capabilities() -> dict[str, Any]:
    """Probe builtin tools on this actual runtime; never promise native coverage."""
    runtime = _runtime_identity()
    providers: dict[str, Any] = {}
    try:
        import cProfile

        profile = cProfile.Profile()
        profile.runcall(lambda: sum(range(16)))
        measured = bool(profile.getstats())
        _require_runtime(runtime)
        providers["cprofile"] = {
            "state": "available" if measured else "refused",
            "tool_version": sys.version,
            "probe": "actual callback roundtrip",
            "scope": "runtime-reported Python/C call elapsed; no TID attribution",
            "unavailable": [
                "native stacks",
                "off-CPU attribution",
                "per-thread attribution",
                "qualified concurrent nesting/elapsed",
            ],
        }
    except (ImportError, RuntimeError, ValueError) as error:
        providers["cprofile"] = {"state": "refused", "reason": type(error).__name__}
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            providers["tracemalloc"] = {"state": "refused", "reason": "existing tracing owner"}
        else:
            tracemalloc.start(1)
            try:
                before, _peak = tracemalloc.get_traced_memory()
                allocation = bytearray(4096)
                during, _peak = tracemalloc.get_traced_memory()
                del allocation
                after, _peak = tracemalloc.get_traced_memory()
                _require_runtime(runtime)
                providers["tracemalloc"] = {
                    "state": "available" if during > before and after < during else "refused",
                    "tool_version": sys.version,
                    "probe": "actual allocate/drop roundtrip",
                    "scope": "Python traced allocations only",
                    "unavailable": ["complete RSS", "SQLite/native allocations", "allocator residence attribution"],
                }
            finally:
                tracemalloc.stop()
    except (ImportError, RuntimeError, ValueError) as error:
        providers["tracemalloc"] = {"state": "refused", "reason": type(error).__name__}
    providers["live-python-stacks"] = {
        "state": "refused" if runtime["gil_enabled"] is False else "available",
        "reason": "free-threaded frame snapshots unavailable" if runtime["gil_enabled"] is False else None,
        "scope": "existing in-process sampler; no native/off-CPU coverage",
    }
    return {
        "format": "polylogue.component-capabilities.v1",
        "runtime": runtime,
        "providers": providers,
        "timing": {
            "wall": "monotonic",
            "thread_cpu": "independent thread_time",
            "process_cpu": "independent process_time",
        },
        "qualification": "builtin capability probes only; diagnostic captures are separate from unprofiled timing",
    }


def _at_least_one(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def refuse_scratch_inside_corpus(scratch: Path, corpus: Path) -> None:
    """Output under the sealed input tree dirties it for good."""
    resolved, corpus_path = scratch.resolve(), corpus.resolve()
    if resolved == corpus_path or corpus_path in resolved.parents:
        raise SystemExit(f"--scratch must lie outside the corpus ({corpus_path})")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"validation", "preparation", "page"} and __name__ != "__main__":
        # The general devtools dispatcher imports product helpers. A direct
        # observation therefore needs the same clean child as a matched arm.
        return subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *arguments], check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("component", choices=("blob", "validation", "preparation", "page", "experiment"))
    parser.add_argument("--stage", choices=("validation", "preparation", "page"), default="validation")
    parser.add_argument("--diagnostic", choices=("cprofile", "tracemalloc"), help="separate direct-validation capture")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--scratch", type=Path, required=True, help="absent or empty private output directory")
    parser.add_argument("--workers", type=_at_least_one, default=1)
    parser.add_argument("--origin", action="append", default=None)
    parser.add_argument("--limit", type=_at_least_one, default=None)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--schema-root", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--ordinal", type=_at_least_one)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--repetitions", type=_at_least_one, default=1)
    args = parser.parse_args(arguments)
    if args.diagnostic is not None and args.component != "validation":
        parser.error("--diagnostic supports direct validation only; matched experiments remain unprofiled")
    required: tuple[str, ...] = ("corpus",) if args.component == "blob" else ("input", "recipe", "schema_root")
    required += (
        ("candidate",)
        if args.component in {"validation", "preparation", "page"}
        else ("before", "after")
        if args.component == "experiment"
        else ()
    )
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{args.component} requires --{name.replace('_', '-')}")
    for name in ("corpus", "input", "recipe", "schema_root"):
        path = getattr(args, name)
        if path is not None:
            refuse_inside_checkout(path, "--" + name.replace("_", "-"))
            if path.is_dir():
                refuse_scratch_inside_corpus(args.scratch, path)
            elif args.scratch.resolve() == path.resolve():
                parser.error("scratch cannot replace an input file")
    refuse_inside_checkout(args.scratch, "--scratch")
    if args.scratch.exists() and any(args.scratch.iterdir()):
        raise SystemExit(f"--scratch must be absent or empty: {args.scratch}")
    args.scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.component == "blob":
        result = bench_blob(args.corpus, args.scratch, workers=args.workers, origins=args.origin, limit=args.limit)
    elif args.component in {"validation", "preparation", "page"}:
        result = bench_validation(
            args.candidate,
            args.input,
            args.recipe,
            args.schema_root,
            args.scratch,
            args.launch,
            args.ordinal,
            args.arm,
            stage=args.component,
            diagnostic=args.diagnostic,
        )
        _write_document(args.scratch / "receipt.json", result)
    else:
        result = run_component_experiment(
            before=args.before,
            after=args.after,
            input_path=args.input,
            recipe=args.recipe,
            schema_root=args.schema_root,
            scratch=args.scratch,
            repetitions=args.repetitions,
            stage=args.stage,
        )
    # Private paths, output coordinates and schema names stay in private receipts.
    print(
        json.dumps(
            result
            if args.component == "blob"
            else {
                key: result[key]
                for key in ("format", "component", "wall_s", "files", "bytes", "complete", "equivalent_outputs")
                if key in result
            }
        )
    )
    return (
        1
        if (result.get("counts") or {}).get("errors")
        or result.get("complete") is False
        or result.get("equivalent_outputs") is False
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
