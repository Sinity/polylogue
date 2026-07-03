"""Reconcile Polylogue token accounting against private provider stores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ProbeStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    compared: int
    missing_archive: int
    missing_external: int
    median_ratio: float | None
    p90_ratio: float | None
    p99_ratio: float | None
    within_tolerance: int
    outside_tolerance: int
    samples: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeSection:
    name: str
    status: ProbeStatus
    summary: str
    required: bool = False
    comparison: ComparisonSummary | None = None
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.comparison is not None:
            payload["comparison"] = self.comparison.to_dict()
        return payload


def _archive_root_default() -> Path:
    configured = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    if configured:
        return Path(configured).expanduser()
    xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return xdg_data_home / "polylogue"


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _coerce_int(value: object) -> int:
    if isinstance(value, int | float | str | bytes | bytearray):
        return int(value or 0)
    return 0


def _ratio_summary(
    pairs: list[tuple[str, int, int, dict[str, object]]],
    *,
    tolerance: float,
    max_samples: int,
) -> ComparisonSummary:
    ratios = sorted((archive / external) for _, archive, external, _ in pairs if external > 0)
    if not ratios:
        median = p90 = p99 = None
    else:
        median = ratios[len(ratios) // 2]
        p90 = ratios[min(len(ratios) - 1, int((len(ratios) - 1) * 0.90))]
        p99 = ratios[min(len(ratios) - 1, int((len(ratios) - 1) * 0.99))]

    low = 1.0 - tolerance
    high = 1.0 + tolerance
    outside = [
        (key, archive, external, extra)
        for key, archive, external, extra in pairs
        if external <= 0 or not (low <= archive / external <= high)
    ]
    samples = tuple(
        {
            "key": key,
            "archive_tokens": archive,
            "external_tokens": external,
            "ratio": round(archive / external, 6) if external else None,
            **extra,
        }
        for key, archive, external, extra in outside[:max_samples]
    )
    return ComparisonSummary(
        compared=len(pairs),
        missing_archive=0,
        missing_external=0,
        median_ratio=round(median, 6) if median is not None else None,
        p90_ratio=round(p90, 6) if p90 is not None else None,
        p99_ratio=round(p99, 6) if p99 is not None else None,
        within_tolerance=len(pairs) - len(outside),
        outside_tolerance=len(outside),
        samples=samples,
    )


def _copy_sqlite_to_scratch(source: Path, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    fd, dest_name = tempfile.mkstemp(prefix="codex-state-", suffix=".sqlite", dir=scratch_dir)
    os.close(fd)
    dest = Path(dest_name)
    shutil.copy2(source, dest)
    return dest


def _codex_archive_totals(conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
          s.session_id,
          s.native_id,
          SUM(COALESCE(u.input_tokens, 0)) AS input_tokens,
          SUM(COALESCE(u.output_tokens, 0)) AS output_tokens,
          SUM(COALESCE(u.cache_read_tokens, 0)) AS cached_input_tokens,
          SUM(COALESCE(u.cache_write_tokens, 0)) AS cache_write_tokens
        FROM session_model_usage AS u
        JOIN sessions AS s ON s.session_id = u.session_id
        WHERE s.origin = 'codex-session'
        GROUP BY s.session_id, s.native_id
        """
    ).fetchall()
    totals: dict[str, dict[str, object]] = {}
    for row in rows:
        native_id = str(row["native_id"])
        session_id = str(row["session_id"])
        key = native_id or session_id.removeprefix("codex-session:")
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        cached_input_tokens = int(row["cached_input_tokens"] or 0)
        cache_write_tokens = int(row["cache_write_tokens"] or 0)
        totals[key] = {
            "session_id": session_id,
            "total_tokens": input_tokens + output_tokens + cached_input_tokens + cache_write_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cache_write_tokens": cache_write_tokens,
        }
    return totals


def _codex_state_totals(conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
    columns = _table_columns(conn, "threads")
    required = {"id", "tokens_used"}
    missing = required - columns
    if missing:
        raise ValueError(f"Codex state threads table missing columns: {', '.join(sorted(missing))}")
    optional_model = ", model" if "model" in columns else ""
    rows = conn.execute(f"SELECT id, tokens_used{optional_model} FROM threads").fetchall()
    totals: dict[str, dict[str, object]] = {}
    for row in rows:
        thread_id = str(row["id"])
        totals[thread_id] = {
            "tokens_used": int(row["tokens_used"] or 0),
            "model": row["model"] if "model" in columns else None,
        }
    return totals


def _probe_codex(
    *,
    archive_conn: sqlite3.Connection,
    codex_state: Path | None,
    scratch_dir: Path,
    required: bool,
    tolerance: float,
    max_samples: int,
) -> ProbeSection:
    if codex_state is None:
        return ProbeSection(
            "codex",
            "skip",
            "no Codex state path was provided",
            required=required,
            details={"reason": "missing_argument", "path": None},
        )
    if not codex_state.exists():
        return ProbeSection(
            "codex",
            "skip",
            f"Codex state path does not exist: {codex_state}",
            required=required,
            details={"reason": "missing_file", "path": str(codex_state)},
        )

    copied = _copy_sqlite_to_scratch(codex_state, scratch_dir)
    try:
        archive = _codex_archive_totals(archive_conn)
        with _open_ro(copied) as conn:
            external = _codex_state_totals(conn)
    except Exception as exc:
        return ProbeSection(
            "codex",
            "fail",
            f"failed to read Codex state: {exc}",
            required=required,
            details={"copied_path": str(copied), "error": str(exc)},
        )

    pairs: list[tuple[str, int, int, dict[str, object]]] = []
    for thread_id, ext in external.items():
        ext_tokens = _coerce_int(ext["tokens_used"])
        if thread_id in archive and ext_tokens > 0:
            arch = archive[thread_id]
            pairs.append(
                (
                    thread_id,
                    _coerce_int(arch["total_tokens"]),
                    ext_tokens,
                    {
                        "session_id": arch["session_id"],
                        "model": ext.get("model"),
                        "input_tokens": arch["input_tokens"],
                        "cached_input_tokens": arch["cached_input_tokens"],
                    },
                )
            )
    comparison = _ratio_summary(pairs, tolerance=tolerance, max_samples=max_samples)
    comparison = ComparisonSummary(
        compared=comparison.compared,
        missing_archive=sum(1 for thread_id in external if thread_id not in archive),
        missing_external=sum(1 for thread_id in archive if thread_id not in external),
        median_ratio=comparison.median_ratio,
        p90_ratio=comparison.p90_ratio,
        p99_ratio=comparison.p99_ratio,
        within_tolerance=comparison.within_tolerance,
        outside_tolerance=comparison.outside_tolerance,
        samples=comparison.samples,
    )
    status: ProbeStatus = "pass" if comparison.outside_tolerance == 0 else "fail"
    return ProbeSection(
        "codex",
        status,
        f"compared {comparison.compared} Codex thread(s) against state_5.sqlite tokens_used",
        required=required,
        comparison=comparison,
        details={
            "copied_path": str(copied),
            "archive_threads": len(archive),
            "external_threads": len(external),
            "tolerance": tolerance,
            "lane_contract": (
                "archive session_model_usage disjoint lanes and Codex threads.tokens_used both include cached input; "
                "raw token_count events remain provider-cumulative evidence and are not the reconciliation grain"
            ),
        },
    )


def _int_from_mapping(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float):
            return int(value)
    return 0


def _load_claude_stats(path: Path) -> dict[str, dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Claude stats cache root must be a JSON object")
    model_usage = raw.get("modelUsage")
    if not isinstance(model_usage, dict):
        raise ValueError("Claude stats cache missing object field modelUsage")
    result: dict[str, dict[str, int]] = {}
    for model, value in model_usage.items():
        if not isinstance(value, dict):
            continue
        result[str(model)] = {
            "input_tokens": _int_from_mapping(value, "inputTokens", "input_tokens"),
            "output_tokens": _int_from_mapping(value, "outputTokens", "output_tokens"),
            "cache_read_tokens": _int_from_mapping(value, "cacheReadInputTokens", "cache_read_tokens"),
            "cache_write_tokens": _int_from_mapping(value, "cacheCreationInputTokens", "cache_write_tokens"),
            "cost_usd": _int_from_mapping(value, "costUSD", "cost_usd"),
        }
    return result


def _claude_archive_totals(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT
          u.model_name,
          SUM(u.input_tokens) AS input_tokens,
          SUM(u.output_tokens) AS output_tokens,
          SUM(u.cache_read_tokens) AS cache_read_tokens,
          SUM(u.cache_write_tokens) AS cache_write_tokens
        FROM session_model_usage AS u
        JOIN sessions AS s ON s.session_id = u.session_id
        WHERE s.origin IN ('claude-code-session', 'claude-ai-export')
        GROUP BY u.model_name
        """
    ).fetchall()
    return {
        str(row["model_name"]): {
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "cache_read_tokens": int(row["cache_read_tokens"] or 0),
            "cache_write_tokens": int(row["cache_write_tokens"] or 0),
        }
        for row in rows
    }


def _probe_claude(
    *,
    archive_conn: sqlite3.Connection,
    stats_cache: Path | None,
    required: bool,
    tolerance: float,
    max_samples: int,
) -> ProbeSection:
    if stats_cache is None:
        return ProbeSection(
            "claude",
            "skip",
            "no Claude stats-cache path was provided",
            required=required,
            details={"reason": "missing_argument", "path": None},
        )
    if not stats_cache.exists():
        return ProbeSection(
            "claude",
            "skip",
            f"Claude stats-cache path does not exist: {stats_cache}",
            required=required,
            details={"reason": "missing_file", "path": str(stats_cache)},
        )

    try:
        archive = _claude_archive_totals(archive_conn)
        external = _load_claude_stats(stats_cache)
    except Exception as exc:
        return ProbeSection(
            "claude",
            "fail",
            f"failed to read Claude stats-cache: {exc}",
            required=required,
            details={"error": str(exc), "path": str(stats_cache)},
        )

    pairs: list[tuple[str, int, int, dict[str, object]]] = []
    for model, ext in external.items():
        arch = archive.get(model)
        if arch is None:
            continue
        for lane in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            ext_tokens = int(ext[lane])
            if ext_tokens <= 0:
                continue
            pairs.append((f"{model}:{lane}", int(arch[lane]), ext_tokens, {"model": model, "lane": lane}))
    comparison = _ratio_summary(pairs, tolerance=tolerance, max_samples=max_samples)
    comparison = ComparisonSummary(
        compared=comparison.compared,
        missing_archive=sum(1 for model in external if model not in archive),
        missing_external=sum(1 for model in archive if model not in external),
        median_ratio=comparison.median_ratio,
        p90_ratio=comparison.p90_ratio,
        p99_ratio=comparison.p99_ratio,
        within_tolerance=comparison.within_tolerance,
        outside_tolerance=comparison.outside_tolerance,
        samples=comparison.samples,
    )
    status: ProbeStatus = "pass" if comparison.outside_tolerance == 0 else "fail"
    return ProbeSection(
        "claude",
        status,
        f"compared {comparison.compared} Claude model/lane value(s) against stats-cache modelUsage",
        required=required,
        comparison=comparison,
        details={
            "archive_models": len(archive),
            "external_models": len(external),
            "tolerance": tolerance,
            "lane_contract": "stats-cache modelUsage lanes map independently; cache is not folded into input",
            "cost_reconciliation": "skipped: stats-cache costUSD is not treated as authoritative",
        },
    )


def _internal_pricing_basis(conn: sqlite3.Connection) -> ProbeSection:
    rows = conn.execute(
        """
        SELECT
          COALESCE(cost_provenance, 'unknown') AS cost_provenance,
          COUNT(*) AS rows,
          SUM(CASE WHEN cost_usd IS NOT NULL THEN 1 ELSE 0 END) AS priced_rows,
          SUM(COALESCE(cost_usd, 0.0)) AS cost_usd
        FROM session_model_usage
        GROUP BY COALESCE(cost_provenance, 'unknown')
        ORDER BY rows DESC
        """
    ).fetchall()
    return ProbeSection(
        "internal_pricing_basis",
        "pass",
        "reported archive pricing provenance distribution; external stores are token-only",
        details={
            "rows": [
                {
                    "cost_provenance": row["cost_provenance"],
                    "rows": int(row["rows"] or 0),
                    "priced_rows": int(row["priced_rows"] or 0),
                    "cost_usd": round(float(row["cost_usd"] or 0.0), 6),
                }
                for row in rows
            ]
        },
    )


def _section_check_failed(section: ProbeSection) -> bool:
    if section.status == "fail":
        return True
    return section.required and section.status == "skip"


def _build_payload(sections: tuple[ProbeSection, ...], archive_root: Path) -> dict[str, object]:
    failed = tuple(section.name for section in sections if _section_check_failed(section))
    tolerance_failures = tuple(
        section.name
        for section in sections
        if section.comparison is not None and section.comparison.outside_tolerance > 0
    )
    return {
        "ok": not failed,
        "archive_root": str(archive_root),
        "sections": [section.to_dict() for section in sections],
        "failed_sections": list(failed),
        "tolerance_failures": list(tolerance_failures),
    }


def _print_human(payload: dict[str, object]) -> None:
    print("Cost reconciliation probe")
    print(f"archive_root: {payload['archive_root']}")
    sections = payload["sections"]
    if not isinstance(sections, list):
        return
    for section in sections:
        if not isinstance(section, dict):
            continue
        print(f"- {section['name']}: {section['status']} - {section['summary']}")
        comparison = section.get("comparison")
        if isinstance(comparison, dict):
            print(
                "  compared={compared} median={median_ratio} p90={p90_ratio} outside={outside_tolerance}".format(
                    **comparison
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=_archive_root_default())
    parser.add_argument("--codex-state", type=Path)
    parser.add_argument("--claude-stats-cache", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/realm/tmp/polylogue-cost-reconciliation"))
    parser.add_argument("--codex-ratio-tolerance", type=float, default=0.10)
    parser.add_argument("--claude-ratio-tolerance", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--require-codex", action="store_true")
    parser.add_argument("--require-claude", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    archive_root = args.archive_root.expanduser()
    index_db = archive_root / "index.db"
    if not index_db.exists():
        payload = {
            "ok": False,
            "archive_root": str(archive_root),
            "sections": [],
            "failed_sections": ["archive"],
            "tolerance_failures": [],
            "error": f"archive index.db not found: {index_db}",
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload["error"], file=sys.stderr)
        return 1

    with _open_ro(index_db) as archive_conn:
        sections = (
            _probe_codex(
                archive_conn=archive_conn,
                codex_state=args.codex_state.expanduser() if args.codex_state else None,
                scratch_dir=args.scratch_dir.expanduser(),
                required=args.require_codex,
                tolerance=args.codex_ratio_tolerance,
                max_samples=args.max_samples,
            ),
            _probe_claude(
                archive_conn=archive_conn,
                stats_cache=args.claude_stats_cache.expanduser() if args.claude_stats_cache else None,
                required=args.require_claude,
                tolerance=args.claude_ratio_tolerance,
                max_samples=args.max_samples,
            ),
            _internal_pricing_basis(archive_conn),
        )
    payload = _build_payload(sections, archive_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 1 if args.check and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
