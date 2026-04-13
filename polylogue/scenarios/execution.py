"""Shared authored execution specs for scenario-bearing surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .corpus import CorpusRequest, CorpusSourceKind


class ExecutionKind(str, Enum):
    """Authored execution substrate for scenario-bearing catalogs."""

    COMMAND = "command"
    POLYLOGUE = "polylogue"
    DEVTOOLS = "devtools"
    PIPELINE_PROBE = "pipeline-probe"
    PYTEST = "pytest"
    MEMORY_BUDGET = "memory-budget"
    COMPOSITE = "composite"
    RUNNER = "runner"


class PipelineProbeInputMode(str, Enum):
    SYNTHETIC = "synthetic"
    ARCHIVE_SUBSET = "archive-subset"
    SOURCE_SUBSET = "source-subset"


_PIPELINE_PROBE_DEFAULT_COUNT = 5
_PIPELINE_PROBE_DEFAULT_MESSAGES_MIN = 4
_PIPELINE_PROBE_DEFAULT_MESSAGES_MAX = 12
_PIPELINE_PROBE_DEFAULT_SEED = 42


@dataclass(frozen=True, slots=True)
class PipelineProbeRequest:
    """Typed devtools pipeline-probe request."""

    stage: str = "all"
    input_mode: PipelineProbeInputMode | str = PipelineProbeInputMode.SYNTHETIC
    corpus_request: CorpusRequest | None = None
    sample_per_provider: int | None = None
    source_filters: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    source_name: str = "inbox"
    source_db: str | None = None
    source_blob_root: str | None = None
    manifest_out: str | None = None
    manifest_in: str | None = None
    raw_batch_size: int | None = None
    ingest_workers: int | None = None
    measure_ingest_result_size: bool = False
    workdir: str | None = None
    json_out: str | None = None
    max_total_ms: float | None = None
    max_peak_rss_mb: float | None = None

    @property
    def input_mode_kind(self) -> PipelineProbeInputMode:
        return PipelineProbeInputMode(self.input_mode)

    def to_argv(self) -> tuple[str, ...]:
        argv: list[str] = []
        if self.input_mode_kind is not PipelineProbeInputMode.SYNTHETIC:
            argv.extend(("--input-mode", self.input_mode_kind.value))
        corpus_request = self.corpus_request
        if corpus_request is not None:
            for provider in corpus_request.providers or ():
                argv.extend(("--provider", provider))
            if corpus_request.source_kind is not CorpusSourceKind.DEFAULT:
                argv.extend(("--corpus-source", corpus_request.source_kind.value))
            if corpus_request.count != _PIPELINE_PROBE_DEFAULT_COUNT:
                argv.extend(("--count", str(corpus_request.count)))
            if corpus_request.messages_min != _PIPELINE_PROBE_DEFAULT_MESSAGES_MIN:
                argv.extend(("--messages-min", str(corpus_request.messages_min)))
            if corpus_request.messages_max != _PIPELINE_PROBE_DEFAULT_MESSAGES_MAX:
                argv.extend(("--messages-max", str(corpus_request.messages_max)))
            if corpus_request.seed is not None and corpus_request.seed != _PIPELINE_PROBE_DEFAULT_SEED:
                argv.extend(("--seed", str(corpus_request.seed)))
            if corpus_request.style != "default":
                argv.extend(("--style", corpus_request.style))
            if corpus_request.package_version != "default":
                argv.extend(("--package-version", corpus_request.package_version))
        if self.stage != "all":
            argv.extend(("--stage", self.stage))
        if self.sample_per_provider is not None:
            argv.extend(("--sample-per-provider", str(self.sample_per_provider)))
        for source in self.source_filters:
            argv.extend(("--source", source))
        for source_path in self.source_paths:
            argv.extend(("--source-path", source_path))
        if self.source_name != "inbox":
            argv.extend(("--source-name", self.source_name))
        if self.source_db is not None:
            argv.extend(("--source-db", self.source_db))
        if self.source_blob_root is not None:
            argv.extend(("--source-blob-root", self.source_blob_root))
        if self.manifest_out is not None:
            argv.extend(("--manifest-out", self.manifest_out))
        if self.manifest_in is not None:
            argv.extend(("--manifest-in", self.manifest_in))
        if self.raw_batch_size is not None:
            argv.extend(("--raw-batch-size", str(self.raw_batch_size)))
        if self.ingest_workers is not None:
            argv.extend(("--ingest-workers", str(self.ingest_workers)))
        if self.measure_ingest_result_size:
            argv.append("--measure-ingest-result-size")
        if self.workdir is not None:
            argv.extend(("--workdir", self.workdir))
        if self.json_out is not None:
            argv.extend(("--json-out", self.json_out))
        if self.max_total_ms is not None:
            argv.extend(("--max-total-ms", str(self.max_total_ms)))
        if self.max_peak_rss_mb is not None:
            argv.extend(("--max-peak-rss-mb", str(self.max_peak_rss_mb)))
        return tuple(argv)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "input_mode": self.input_mode_kind.value,
        }
        if self.corpus_request is not None:
            payload["corpus_request"] = {
                "providers": list(self.corpus_request.providers) if self.corpus_request.providers is not None else None,
                "source": self.corpus_request.source_kind.value,
                "count": self.corpus_request.count,
                "messages_min": self.corpus_request.messages_min,
                "messages_max": self.corpus_request.messages_max,
                "seed": self.corpus_request.seed,
                "style": self.corpus_request.style,
                "package_version": self.corpus_request.package_version,
            }
        if self.sample_per_provider is not None:
            payload["sample_per_provider"] = self.sample_per_provider
        if self.source_filters:
            payload["source_filters"] = list(self.source_filters)
        if self.source_paths:
            payload["source_paths"] = list(self.source_paths)
        if self.source_name != "inbox":
            payload["source_name"] = self.source_name
        if self.source_db is not None:
            payload["source_db"] = self.source_db
        if self.source_blob_root is not None:
            payload["source_blob_root"] = self.source_blob_root
        if self.manifest_out is not None:
            payload["manifest_out"] = self.manifest_out
        if self.manifest_in is not None:
            payload["manifest_in"] = self.manifest_in
        if self.raw_batch_size is not None:
            payload["raw_batch_size"] = self.raw_batch_size
        if self.ingest_workers is not None:
            payload["ingest_workers"] = self.ingest_workers
        if self.measure_ingest_result_size:
            payload["measure_ingest_result_size"] = True
        if self.workdir is not None:
            payload["workdir"] = self.workdir
        if self.json_out is not None:
            payload["json_out"] = self.json_out
        if self.max_total_ms is not None:
            payload["max_total_ms"] = self.max_total_ms
        if self.max_peak_rss_mb is not None:
            payload["max_peak_rss_mb"] = self.max_peak_rss_mb
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PipelineProbeRequest:
        corpus_payload = payload.get("corpus_request")
        corpus_request: CorpusRequest | None = None
        if isinstance(corpus_payload, Mapping):
            providers = corpus_payload.get("providers")
            corpus_request = CorpusRequest(
                providers=tuple(str(item) for item in providers) if isinstance(providers, (list, tuple)) else None,
                source=CorpusSourceKind(str(corpus_payload.get("source", CorpusSourceKind.DEFAULT.value))),
                count=int(corpus_payload.get("count", _PIPELINE_PROBE_DEFAULT_COUNT) or _PIPELINE_PROBE_DEFAULT_COUNT),
                messages_min=int(
                    corpus_payload.get("messages_min", _PIPELINE_PROBE_DEFAULT_MESSAGES_MIN)
                    or _PIPELINE_PROBE_DEFAULT_MESSAGES_MIN
                ),
                messages_max=int(
                    corpus_payload.get("messages_max", _PIPELINE_PROBE_DEFAULT_MESSAGES_MAX)
                    or _PIPELINE_PROBE_DEFAULT_MESSAGES_MAX
                ),
                seed=int(corpus_payload["seed"]) if corpus_payload.get("seed") is not None else _PIPELINE_PROBE_DEFAULT_SEED,
                style=str(corpus_payload.get("style", "default") or "default"),
                package_version=str(corpus_payload.get("package_version", "default") or "default"),
            )
        source_filters = payload.get("source_filters")
        source_paths = payload.get("source_paths")
        return cls(
            stage=str(payload.get("stage", "all") or "all"),
            input_mode=PipelineProbeInputMode(str(payload.get("input_mode", PipelineProbeInputMode.SYNTHETIC.value))),
            corpus_request=corpus_request,
            sample_per_provider=int(payload["sample_per_provider"])
            if payload.get("sample_per_provider") is not None
            else None,
            source_filters=tuple(str(item) for item in source_filters) if isinstance(source_filters, list | tuple) else (),
            source_paths=tuple(str(item) for item in source_paths) if isinstance(source_paths, list | tuple) else (),
            source_name=str(payload.get("source_name", "inbox") or "inbox"),
            source_db=str(payload["source_db"]) if payload.get("source_db") is not None else None,
            source_blob_root=str(payload["source_blob_root"]) if payload.get("source_blob_root") is not None else None,
            manifest_out=str(payload["manifest_out"]) if payload.get("manifest_out") is not None else None,
            manifest_in=str(payload["manifest_in"]) if payload.get("manifest_in") is not None else None,
            raw_batch_size=int(payload["raw_batch_size"]) if payload.get("raw_batch_size") is not None else None,
            ingest_workers=int(payload["ingest_workers"]) if payload.get("ingest_workers") is not None else None,
            measure_ingest_result_size=bool(payload.get("measure_ingest_result_size", False)),
            workdir=str(payload["workdir"]) if payload.get("workdir") is not None else None,
            json_out=str(payload["json_out"]) if payload.get("json_out") is not None else None,
            max_total_ms=float(payload["max_total_ms"]) if payload.get("max_total_ms") is not None else None,
            max_peak_rss_mb=float(payload["max_peak_rss_mb"]) if payload.get("max_peak_rss_mb") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """One authored execution workload."""

    kind: ExecutionKind
    argv: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    runner: str = ""
    subcommand: str = ""
    max_rss_mb: int = 0
    wrapped: ExecutionSpec | None = None
    pipeline_probe: PipelineProbeRequest | None = None

    @property
    def is_composite(self) -> bool:
        return self.kind is ExecutionKind.COMPOSITE

    @property
    def is_runner(self) -> bool:
        return self.kind is ExecutionKind.RUNNER

    @property
    def is_devtools(self) -> bool:
        return self.kind is ExecutionKind.DEVTOOLS

    @property
    def is_pipeline_probe(self) -> bool:
        return self.kind is ExecutionKind.PIPELINE_PROBE

    @property
    def is_memory_budget(self) -> bool:
        return self.kind is ExecutionKind.MEMORY_BUDGET

    @property
    def command(self) -> tuple[str, ...] | None:
        if self.is_composite or self.is_runner:
            return None
        if self.kind is ExecutionKind.PIPELINE_PROBE:
            if self.pipeline_probe is None:
                return None
            return ("devtools", "pipeline-probe", *self.pipeline_probe.to_argv())
        if self.kind is ExecutionKind.POLYLOGUE:
            return ("polylogue", "--plain", *self.argv)
        if self.kind is ExecutionKind.DEVTOOLS:
            from devtools.command_catalog import control_plane_argv

            return tuple(control_plane_argv(self.subcommand, *self.argv))
        if self.kind is ExecutionKind.MEMORY_BUDGET:
            if self.wrapped is None or self.max_rss_mb <= 0:
                return None
            wrapped_command = self.wrapped.command
            if wrapped_command is None:
                return None
            from devtools.command_catalog import control_plane_argv

            return tuple(
                control_plane_argv(
                    "query-memory-budget",
                    "--max-rss-mb",
                    str(self.max_rss_mb),
                    "--",
                    *wrapped_command,
                )
            )
        if self.kind is ExecutionKind.PYTEST:
            return ("pytest", *self.argv)
        return self.argv

    @property
    def display_command(self) -> tuple[str, ...] | None:
        command = self.command
        if command is None:
            return None
        if self.kind is ExecutionKind.POLYLOGUE:
            return ("polylogue", *self.argv)
        if self.kind is ExecutionKind.MEMORY_BUDGET and self.wrapped is not None:
            wrapped_display = self.wrapped.display_command or self.wrapped.command
            if wrapped_display is None:
                return command
            return (
                "devtools",
                "query-memory-budget",
                "--max-rss-mb",
                str(self.max_rss_mb),
                "--",
                *wrapped_display,
            )
        return command

    @property
    def pytest_targets(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            return ()
        return self.argv

    @property
    def polylogue_args(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.POLYLOGUE:
            return ()
        return self.argv

    @property
    def polylogue_invoke_args(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.POLYLOGUE:
            return ()
        return ("--plain", *self.argv)

    def pytest_command(self, *prefix_args: str) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            raise ValueError(f"{self.kind.value} execution cannot render a pytest command")
        return ("pytest", *prefix_args, *self.argv)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind.value}
        if self.argv:
            payload["argv"] = list(self.argv)
        if self.members:
            payload["members"] = list(self.members)
        if self.runner:
            payload["runner"] = self.runner
        if self.subcommand:
            payload["subcommand"] = self.subcommand
        if self.max_rss_mb > 0:
            payload["max_rss_mb"] = self.max_rss_mb
        if self.wrapped is not None:
            payload["wrapped"] = self.wrapped.to_payload()
        if self.pipeline_probe is not None:
            payload["pipeline_probe"] = self.pipeline_probe.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExecutionSpec:
        kind = ExecutionKind(str(payload["kind"]))
        argv = tuple(str(item) for item in payload.get("argv", ()))
        members = tuple(str(item) for item in payload.get("members", ()))
        runner = str(payload.get("runner", "")) if payload.get("runner") is not None else ""
        subcommand = str(payload.get("subcommand", "")) if payload.get("subcommand") is not None else ""
        max_rss_mb = int(payload.get("max_rss_mb", 0) or 0)
        wrapped_payload = payload.get("wrapped")
        wrapped = cls.from_payload(wrapped_payload) if isinstance(wrapped_payload, Mapping) else None
        probe_payload = payload.get("pipeline_probe")
        pipeline_probe = (
            PipelineProbeRequest.from_payload(probe_payload)
            if isinstance(probe_payload, Mapping)
            else None
        )
        return cls(
            kind=kind,
            argv=argv,
            members=members,
            runner=runner,
            subcommand=subcommand,
            max_rss_mb=max_rss_mb,
            wrapped=wrapped,
            pipeline_probe=pipeline_probe,
        )


def command_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMMAND, argv=tuple(argv))


def polylogue_execution(*argv: str) -> ExecutionSpec:
    if argv[:2] == ("polylogue", "--plain"):
        argv = argv[2:]
    elif argv[:1] == ("polylogue",):
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.POLYLOGUE, argv=tuple(argv))


def devtools_execution(subcommand: str, *argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.DEVTOOLS, subcommand=subcommand, argv=tuple(argv))


def pipeline_probe_execution(request: PipelineProbeRequest) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.PIPELINE_PROBE, pipeline_probe=request)


def pytest_execution(*argv: str) -> ExecutionSpec:
    if argv and argv[0] == "pytest":
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.PYTEST, argv=tuple(argv))


def composite_execution(*members: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMPOSITE, members=tuple(members))


def runner_execution(runner: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.RUNNER, runner=runner)


def memory_budget_execution(max_rss_mb: int, execution: ExecutionSpec) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.MEMORY_BUDGET, max_rss_mb=max_rss_mb, wrapped=execution)


__all__ = [
    "command_execution",
    "composite_execution",
    "devtools_execution",
    "ExecutionKind",
    "PipelineProbeInputMode",
    "PipelineProbeRequest",
    "ExecutionSpec",
    "memory_budget_execution",
    "pipeline_probe_execution",
    "polylogue_execution",
    "pytest_execution",
    "runner_execution",
]
