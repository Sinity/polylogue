"""CLI parsing, input selection, and request schema for pipeline probes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NotRequired

from typing_extensions import TypedDict

from devtools.regression_cases import DEFAULT_REGRESSION_CASE_DIR
from polylogue.core.json import JSONDocument, JSONValue, loads, require_json_document
from polylogue.pipeline.runner import RUN_STAGE_CHOICES
from polylogue.scenarios import (
    CorpusRequest,
    CorpusSourceKind,
    PipelineProbeInputMode,
    PipelineProbeRequest,
)
from polylogue.schemas.synthetic import SyntheticCorpus

_EXT_MAP = {
    "chatgpt": ".json",
    "claude-ai": ".json",
    "gemini": ".json",
    "claude-code": ".jsonl",
    "codex": ".jsonl",
}
_INPUT_MODES = tuple(mode.value for mode in PipelineProbeInputMode)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_BACKED_PROBE_STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "acquire": ("acquire",),
    "schema": ("acquire", "schema"),
    "parse": ("acquire", "parse"),
    "materialize": ("acquire", "parse", "materialize"),
    "render": ("acquire", "parse", "render"),
    "index": ("acquire", "parse", "index"),
    "reprocess": ("acquire", "parse", "materialize", "render", "index"),
    "all": ("acquire", "parse", "materialize", "render", "index"),
}


class RawFanoutEntry(TypedDict):
    raw_id: str
    payload_provider: str | None
    source_name: str | None
    blob_size_bytes: int
    conversation_count: int
    message_count: int
    parse_error: str | None


class PathFingerprint(TypedDict):
    path: str
    kind: str
    sha256: str
    file_count: int
    total_bytes: int


class StagedSourceEntry(TypedDict):
    input_path: str
    staged_path: str
    kind: str
    file_count: int
    bytes: int


class SourceInputsSummary(TypedDict):
    input_count: int
    staged_entry_count: int
    staged_file_count: int
    total_bytes: int
    entries: list[StagedSourceEntry]


class ProbeProvenance(TypedDict):
    git_commit: str | None
    worktree_dirty: bool | None
    manifest_sha256: NotRequired[str]
    source_input_fingerprints: NotRequired[list[PathFingerprint]]
    source_inputs_sha256: NotRequired[str]


class ArchiveManifest(TypedDict):
    input_mode: str
    source_db: str
    source_blob_root: str
    seed: int
    sample_per_provider: int
    provider_filters: list[str]
    source_filters: list[str]
    candidate_count: int
    missing_blob_count: int
    available_by_provider: dict[str, int]
    sampled_by_provider: dict[str, int]
    records: list[JSONDocument]


class ArchiveSubsetSampleSummary(TypedDict):
    selected_count: int
    copied_records: int
    copied_blob_bytes: int
    provider_counts: dict[str, int]
    source_counts: dict[str, int]
    sample_per_provider: int
    candidate_count: int
    missing_blob_count: int
    available_by_provider: dict[str, int]
    sampled_by_provider: dict[str, int]


class BudgetReport(TypedDict):
    ok: bool
    max_total_ms: float | None
    observed_total_ms: JSONValue
    max_peak_rss_mb: float | None
    observed_peak_rss_mb: JSONValue
    observed_peak_rss_self_mb: JSONValue
    observed_peak_rss_children_mb: JSONValue
    violations: list[str]


class RegressionCaseSummary(TypedDict):
    case_id: str
    name: str
    path: str
    tags: list[str]


class ProbeSummary(TypedDict):
    probe: JSONDocument
    paths: JSONDocument
    provenance: ProbeProvenance
    result: JSONDocument
    run_payload: JSONDocument
    db_stats: dict[str, int]
    raw_fanout: list[RawFanoutEntry]
    source_files: NotRequired[JSONDocument]
    source_inputs: NotRequired[SourceInputsSummary]
    sample: NotRequired[ArchiveSubsetSampleSummary]
    budgets: NotRequired[BudgetReport]
    regression_case: NotRequired[RegressionCaseSummary]


def _names(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _paths(value: object | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        return [Path(value)]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, Path) else Path(str(item)) for item in value]
    return [Path(str(value))]


def _probe_mode(args: argparse.Namespace | PipelineProbeRequest) -> str:
    if isinstance(args, PipelineProbeRequest):
        return args.input_mode_kind.value
    return str(getattr(args, "input_mode", "synthetic"))


def _resolved_corpus_request(request: PipelineProbeRequest) -> CorpusRequest:
    if request.corpus_request is not None:
        return request.corpus_request
    return CorpusRequest(
        providers=("chatgpt",),
        source=CorpusSourceKind.DEFAULT,
        count=5,
        messages_min=4,
        messages_max=12,
        seed=42,
        style="default",
        package_version="default",
    )


def _resolve_synthetic_provider(request: PipelineProbeRequest) -> str:
    provider_names = tuple(_resolved_corpus_request(request).providers or ("chatgpt",))
    provider_name = provider_names[0] if provider_names else "chatgpt"
    available = set(SyntheticCorpus.available_providers())
    if provider_name not in available:
        raise ValueError(f"--provider must be one of {sorted(available)} in synthetic mode")
    if len(provider_names) > 1:
        raise ValueError("synthetic mode accepts exactly one --provider")
    return provider_name


def _load_run_payload(run_path: str | None) -> JSONDocument:
    if not run_path:
        return {}
    return require_json_document(loads(Path(run_path).read_text(encoding="utf-8")), context="pipeline run payload")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real pipeline against a bounded synthetic, archive-subset, "
            "or staged real-source corpus and emit a JSON summary."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=_INPUT_MODES,
        default="synthetic",
        help=(
            "Probe input mode: synthetic fixture generation, archive-subset replay, "
            "or staged real-source inputs (default: synthetic)"
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help=(
            "Provider selector. In synthetic mode this must resolve to exactly one provider "
            f"({', '.join(sorted(SyntheticCorpus.available_providers()))}). "
            "In archive-subset mode it filters the sampled providers and may be repeated."
        ),
    )
    parser.add_argument(
        "--corpus-source",
        choices=[kind.value for kind in CorpusSourceKind],
        default=CorpusSourceKind.DEFAULT.value,
        help="Synthetic corpus source to use in synthetic mode (default: default)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Synthetic source files to generate (default: 5)",
    )
    parser.add_argument(
        "--messages-min",
        type=int,
        default=4,
        help="Minimum messages per conversation (default: 4)",
    )
    parser.add_argument(
        "--messages-max",
        type=int,
        default=12,
        help="Maximum messages per conversation (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synthetic corpus seed (default: 42)",
    )
    parser.add_argument(
        "--style",
        default="default",
        help="Synthetic corpus style (default: default)",
    )
    parser.add_argument(
        "--package-version",
        default="default",
        help="Synthetic package version selector (default: default)",
    )
    parser.add_argument(
        "--sample-per-provider",
        type=int,
        default=50,
        help="Archive-subset sample size per provider (default: 50)",
    )
    parser.add_argument(
        "--source",
        dest="source_filters",
        action="append",
        default=None,
        help="Archive-subset source-name filter. Repeatable.",
    )
    parser.add_argument(
        "--source-path",
        dest="source_paths",
        action="append",
        type=Path,
        default=None,
        help=(
            "Real-source input path for source-subset mode. Files or directories are copied "
            "into the isolated probe workspace. Repeatable."
        ),
    )
    parser.add_argument(
        "--source-name",
        default="inbox",
        help="Source name assigned to staged source-subset inputs (default: inbox)",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        help="Archive-subset source database path (default: current archive database)",
    )
    parser.add_argument(
        "--source-blob-root",
        type=Path,
        help="Archive-subset blob-store root (default: current blob store root)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional path for the archive-subset selection manifest.",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Replay a previously persisted archive-subset manifest instead of resampling.",
    )
    parser.add_argument(
        "--stage",
        choices=RUN_STAGE_CHOICES,
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--raw-batch-size",
        type=int,
        default=None,
        help="Use this raw-record batch size for the probe.",
    )
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=None,
        help="Use this worker count for the probe ingest stage.",
    )
    parser.add_argument(
        "--measure-ingest-result-size",
        action="store_true",
        help="Measure serialized IngestRecordResult sizes for this probe.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help=(
            "Probe workspace root. If omitted, a temporary workspace is created and removed after the run. "
            "Pass an explicit path when you want to keep the run/database artifacts."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the JSON summary.",
    )
    parser.add_argument(
        "--capture-regression",
        metavar="NAME",
        help=(
            "Capture the emitted probe summary as a durable regression case under "
            ".local/regression-cases or --regression-output-dir."
        ),
    )
    parser.add_argument(
        "--regression-output-dir",
        type=Path,
        default=DEFAULT_REGRESSION_CASE_DIR,
        help=f"Output directory for --capture-regression (default: {DEFAULT_REGRESSION_CASE_DIR}).",
    )
    parser.add_argument(
        "--regression-tag",
        action="append",
        default=[],
        help="Tag to attach to the captured regression case. Repeatable.",
    )
    parser.add_argument(
        "--regression-note",
        action="append",
        default=[],
        help="Note to attach to the captured regression case. Repeatable.",
    )
    parser.add_argument(
        "--max-total-ms",
        type=float,
        default=None,
        help="Fail if total pipeline runtime exceeds this budget in milliseconds.",
    )
    parser.add_argument(
        "--max-peak-rss-mb",
        type=float,
        default=None,
        help="Fail if peak RSS exceeds this budget in MiB.",
    )
    return parser.parse_args(argv)


def _request_from_args(args: argparse.Namespace) -> PipelineProbeRequest:
    """Project argparse input onto the authored pipeline-probe request contract."""
    provider_names = tuple(_names(getattr(args, "provider", None)))
    corpus_request: CorpusRequest | None = None
    input_mode = PipelineProbeInputMode(_probe_mode(args))
    if input_mode is PipelineProbeInputMode.SYNTHETIC:
        corpus_request = CorpusRequest(
            providers=provider_names or ("chatgpt",),
            source=CorpusSourceKind(getattr(args, "corpus_source", CorpusSourceKind.DEFAULT.value)),
            count=args.count,
            messages_min=args.messages_min,
            messages_max=args.messages_max,
            seed=args.seed,
            style=getattr(args, "style", "default"),
            package_version=getattr(args, "package_version", "default"),
        )
    return PipelineProbeRequest(
        stage=args.stage,
        input_mode=input_mode,
        corpus_request=corpus_request,
        sample_per_provider=args.sample_per_provider,
        source_filters=tuple(_names(getattr(args, "source_filters", None))),
        source_paths=tuple(str(path) for path in _paths(getattr(args, "source_paths", None))),
        source_name=str(getattr(args, "source_name", "inbox") or "inbox"),
        source_db=str(args.source_db) if args.source_db is not None else None,
        source_blob_root=str(args.source_blob_root) if args.source_blob_root is not None else None,
        manifest_out=str(args.manifest_out) if args.manifest_out is not None else None,
        manifest_in=str(args.manifest_in) if args.manifest_in is not None else None,
        raw_batch_size=args.raw_batch_size,
        ingest_workers=args.ingest_workers,
        measure_ingest_result_size=bool(args.measure_ingest_result_size),
        workdir=str(args.workdir) if args.workdir is not None else None,
        json_out=str(args.json_out) if args.json_out is not None else None,
        max_total_ms=args.max_total_ms,
        max_peak_rss_mb=args.max_peak_rss_mb,
    )


__all__ = [
    "ArchiveManifest",
    "ArchiveSubsetSampleSummary",
    "BudgetReport",
    "PathFingerprint",
    "ProbeProvenance",
    "ProbeSummary",
    "RawFanoutEntry",
    "RegressionCaseSummary",
    "SourceInputsSummary",
    "StagedSourceEntry",
    "_EXT_MAP",
    "_INPUT_MODES",
    "_REPO_ROOT",
    "_SOURCE_BACKED_PROBE_STAGE_SEQUENCES",
    "_load_run_payload",
    "_names",
    "_parse_args",
    "_paths",
    "_probe_mode",
    "_request_from_args",
    "_resolved_corpus_request",
    "_resolve_synthetic_provider",
]
