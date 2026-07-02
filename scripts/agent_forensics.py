#!/usr/bin/env python3
"""AI-Agent Forensics — mine a Polylogue archive for longitudinal usage findings.

Reads a Polylogue ``index.db`` (read-only) and emits a Markdown findings report
plus standalone SVG charts: corpus scale, token economy, cost, model evolution,
temporal rhythm, and workflow/failure signals.

Pure standard library — no third-party dependencies, no network, no writes to
the archive. A stranger can run it against any archive (including the synthetic
``polylogue demo seed`` archive) and reproduce the report.

    python scripts/agent_forensics.py --archive ~/.local/share/polylogue --out ./forensics

The archive defaults to ``$POLYLOGUE_ARCHIVE_ROOT`` then the XDG data dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# SVG charting (dependency-free)
# --------------------------------------------------------------------------- #

_PALETTE = [
    "#4f9ecf",
    "#e4754f",
    "#6cc24a",
    "#b072d6",
    "#e0b341",
    "#4ac6c2",
    "#d6606f",
    "#8a8f99",
]


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="{width // 2}" y="22" fill="#e6edf3" font-size="15" '
        f'font-weight="600" text-anchor="middle">{_esc(title)}</text>',
    ]


def bar_chart(title: str, labels: list[str], values: list[float], *, unit: str = "") -> str:
    """Vertical bar chart. labels/values are parallel; values >= 0."""
    width, height = 880, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vmax = max(values) if values and max(values) > 0 else 1.0
    n = max(len(values), 1)
    slot = plot_w / n
    bar_w = slot * 0.7
    out = _svg_header(width, height, title)
    # y gridlines
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#21262d"/>')
        out.append(
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="end">{_fmt_num(vmax * frac)}{unit}</text>'
        )
    # Show every label when few bars; thin to ~15 labels when many.
    label_every = 1 if n <= 18 else max(2, (n + 14) // 15)
    for i, (lab, val) in enumerate(zip(labels, values, strict=False)):
        x = pad_l + slot * i + (slot - bar_w) / 2
        h = plot_h * (val / vmax)
        y = pad_t + plot_h - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{_PALETTE[0]}" rx="2"/>')
        if i % label_every == 0:
            lx = x + bar_w / 2
            out.append(
                f'<text x="{lx:.1f}" y="{height - pad_b + 14}" fill="#8b949e" font-size="9" '
                f'text-anchor="end" transform="rotate(-55 {lx:.1f} {height - pad_b + 14})">{_esc(lab)}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def hbar_chart(title: str, labels: list[str], values: list[float], *, unit: str = "") -> str:
    """Horizontal bar chart, good for ranked categories."""
    row_h = 30
    width = 880
    pad_l, pad_r, pad_t, pad_b = 220, 70, 40, 20
    height = pad_t + pad_b + row_h * max(len(values), 1)
    plot_w = width - pad_l - pad_r
    vmax = max(values) if values and max(values) > 0 else 1.0
    out = _svg_header(width, height, title)
    for i, (lab, val) in enumerate(zip(labels, values, strict=False)):
        y = pad_t + row_h * i + 4
        w = plot_w * (val / vmax)
        out.append(
            f'<rect x="{pad_l}" y="{y:.1f}" width="{w:.1f}" height="{row_h - 10}" fill="{_PALETTE[i % len(_PALETTE)]}" rx="2"/>'
        )
        out.append(
            f'<text x="{pad_l - 8}" y="{y + row_h - 14:.1f}" fill="#e6edf3" font-size="11" text-anchor="end">{_esc(lab)}</text>'
        )
        out.append(
            f'<text x="{pad_l + w + 6:.1f}" y="{y + row_h - 14:.1f}" fill="#8b949e" font-size="10">{_fmt_num(val)}{unit}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def line_chart(title: str, labels: list[str], series: dict[str, list[float]], *, unit: str = "") -> str:
    """Multi-series line chart over an ordered x axis (labels)."""
    width, height = 880, 360
    pad_l, pad_r, pad_t, pad_b = 64, 120, 40, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = max(max((len(v) for v in series.values()), default=1), 1)
    vmax = max((max(v) for v in series.values() if v), default=1.0) or 1.0
    out = _svg_header(width, height, title)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#21262d"/>')
        out.append(
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" fill="#8b949e" font-size="10" text-anchor="end">{_fmt_num(vmax * frac)}{unit}</text>'
        )
    step = plot_w / max(n - 1, 1)
    for si, (name, vals) in enumerate(series.items()):
        color = _PALETTE[si % len(_PALETTE)]
        pts = []
        for i, v in enumerate(vals):
            x = pad_l + step * i
            y = pad_t + plot_h * (1 - v / vmax)
            pts.append(f"{x:.1f},{y:.1f}")
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')
        out.append(
            f'<text x="{width - pad_r + 6}" y="{pad_t + 14 + si * 16}" fill="{color}" font-size="11">{_esc(name)}</text>'
        )
    # sparse x labels
    show_every = max(n // 12, 1)
    for i, lab in enumerate(labels):
        if i % show_every:
            continue
        x = pad_l + step * i
        out.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 14}" fill="#8b949e" font-size="9" '
            f'text-anchor="end" transform="rotate(-55 {x:.1f} {height - pad_b + 14})">{_esc(lab)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def _fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 1e12:
        return f"{v / 1e12:.1f}T"
    if a >= 1e9:
        return f"{v / 1e9:.1f}B"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k"
    if a and a < 1:
        return f"{v:.2f}"
    return f"{v:.0f}"


def _fmt_int(v: int) -> str:
    return f"{v:,}"


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


# --------------------------------------------------------------------------- #
# Archive access
# --------------------------------------------------------------------------- #


def resolve_index_db(archive: str | None) -> Path:
    root = Path(
        archive
        or os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
        or (Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "polylogue")
    )
    index_db = root if root.name == "index.db" else root / "index.db"
    if not index_db.exists():
        raise SystemExit(f"index.db not found at {index_db}")
    return index_db


def connect_ro(index_db: Path) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=30.0)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError as exc:
        # A read-only open can't roll back a hot rollback journal or recover a
        # WAL whose -shm is gone (a writer exited uncleanly, or the daemon is
        # mid-write without a live shared-memory index). Fail with guidance
        # instead of a raw traceback rather than writing to the archive.
        hot = index_db.with_name(index_db.name + "-journal")
        wal = index_db.with_name(index_db.name + "-wal")
        detail = (
            "hot rollback journal" if hot.exists() else ("uncheckpointed WAL" if wal.exists() else "locked/unreadable")
        )
        raise SystemExit(
            f"Cannot open {index_db} read-only ({detail}: {exc}).\n"
            "The archive is mid-write or was left in an inconsistent state. Let the "
            "daemon finish/checkpoint (or re-run the seed to completion), then retry. "
            "This tool never writes to the archive, so it will not roll the journal "
            "back itself."
        ) from exc
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1",
            (name,),
        ).fetchone()
        is not None
    )


def _scalar(conn: sqlite3.Connection, sql: str, default: Any = 0) -> Any:
    row = conn.execute(sql).fetchone()
    if row is None or row[0] is None:
        return default
    return row[0]


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    f: dict[str, Any] = {}

    # 1. Scale & span -------------------------------------------------------- #
    f["sessions"] = int(_scalar(conn, "SELECT COUNT(*) FROM sessions"))
    f["messages"] = int(_scalar(conn, "SELECT COUNT(*) FROM messages"))
    f["blocks"] = int(_scalar(conn, "SELECT COUNT(*) FROM blocks")) if _has_table(conn, "blocks") else 0
    span = conn.execute(
        "SELECT date(MIN(sort_key_ms)/1000,'unixepoch'), date(MAX(sort_key_ms)/1000,'unixepoch') "
        "FROM sessions WHERE sort_key_ms>0"
    ).fetchone()
    f["span_start"], f["span_end"] = span[0], span[1]
    f["origins"] = conn.execute("SELECT origin, COUNT(*) c FROM sessions GROUP BY origin ORDER BY c DESC").fetchall()

    # 2. Token economy + 3. cost (session_model_usage) ----------------------- #
    if _has_table(conn, "session_model_usage"):
        tot = conn.execute(
            """
            SELECT cost_provenance,
                   COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                   COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0),
                   COALESCE(SUM(cost_usd),0), COUNT(DISTINCT session_id)
            FROM session_model_usage GROUP BY cost_provenance
            """
        ).fetchall()
        # Token accounting splits by cost provenance: 'priced' rows carry a
        # computed cost_usd; 'origin_reported' rows are provider-reported token
        # counts with NO cost. Pairing the priced cost with the all-provenance
        # token total would be wrong, so we keep them separate.
        econ: dict[str, dict[str, int | float]] = {}
        for r in tot:
            econ[str(r[0])] = {
                "input": int(r[1]),
                "output": int(r[2]),
                "cache_read": int(r[3]),
                "cache_write": int(r[4]),
                "cost": float(r[5]),
                "sessions": int(r[6]),
            }
        f["economy_by_provenance"] = econ
        f["total_cost"] = sum(float(v["cost"]) for v in econ.values())
        f["tok_input"] = sum(int(v["input"]) for v in econ.values())
        f["tok_output"] = sum(int(v["output"]) for v in econ.values())
        f["tok_cache_read"] = sum(int(v["cache_read"]) for v in econ.values())
        f["tok_cache_write"] = sum(int(v["cache_write"]) for v in econ.values())
        f["priced_sessions"] = int(econ.get("priced", {}).get("sessions", 0))
        f["cost_by_model"] = conn.execute(
            """
            SELECT model_name, COUNT(DISTINCT session_id) sess, SUM(cost_usd) cost
            FROM session_model_usage GROUP BY model_name
            HAVING cost > 0 ORDER BY cost DESC LIMIT 10
            """
        ).fetchall()
        f["cost_provenance"] = conn.execute(
            "SELECT cost_provenance, COUNT(*) c, SUM(cost_usd) cost FROM session_model_usage GROUP BY cost_provenance ORDER BY c DESC"
        ).fetchall()
        per_session = [
            float(r[0])
            for r in conn.execute(
                "SELECT SUM(cost_usd) c FROM session_model_usage GROUP BY session_id HAVING c > 0"
            ).fetchall()
        ]
        per_session.sort()
        f["cost_session_count"] = len(per_session)
        if per_session:
            f["cost_median"] = per_session[len(per_session) // 2]
            f["cost_p90"] = per_session[int(len(per_session) * 0.9)]
            f["cost_max"] = per_session[-1]
            f["cost_hist"] = _log_histogram(per_session)
        # Subscription (Claude Max/Pro) credit view. Cache reads are FREE on
        # plans; usage is metered in credits with per-family rates, cache writes
        # billed at the input rate. (Rates reverse-engineered by she-llac.com.)
        per_model_classes = conn.execute(
            """
            SELECT model_name, COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                   COALESCE(SUM(cache_write_tokens),0)
            FROM session_model_usage WHERE cost_provenance='priced' GROUP BY model_name
            """
        ).fetchall()
        f["subscription_credits"] = _subscription_credits(per_model_classes)
    if _has_table(conn, "session_provider_usage_events"):
        # Use the per-event delta (last_*), NOT the cumulative running total
        # (total_*) — summing the cumulative column across a session's events
        # over-counts reasoning tokens by orders of magnitude.
        f["tok_reasoning"] = int(
            _scalar(conn, "SELECT COALESCE(SUM(last_reasoning_output_tokens),0) FROM session_provider_usage_events")
        )

    # 4. Temporal rhythm ----------------------------------------------------- #
    f["sessions_per_month"] = conn.execute(
        "SELECT strftime('%Y-%m', sort_key_ms/1000,'unixepoch') ym, COUNT(*) c "
        "FROM sessions WHERE sort_key_ms>0 GROUP BY ym ORDER BY ym"
    ).fetchall()
    if _has_table(conn, "session_provider_usage_events"):
        f["tokens_per_month"] = conn.execute(
            "SELECT strftime('%Y-%m', occurred_at_ms/1000,'unixepoch') ym, SUM(last_total_tokens) t "
            "FROM session_provider_usage_events WHERE occurred_at_ms>0 GROUP BY ym ORDER BY ym"
        ).fetchall()

    # 5. Model evolution (tokens by month, top models) ----------------------- #
    if _has_table(conn, "session_provider_usage_events"):
        top_models = [
            r[0]
            for r in conn.execute(
                "SELECT model_name FROM session_provider_usage_events WHERE model_name IS NOT NULL "
                "GROUP BY model_name ORDER BY SUM(last_total_tokens) DESC LIMIT 5"
            ).fetchall()
        ]
        f["top_models"] = top_models
        if top_models:
            placeholders = ",".join("?" for _ in top_models)
            rows = conn.execute(
                f"""
                SELECT strftime('%Y-%m', occurred_at_ms/1000,'unixepoch') ym, model_name, SUM(last_total_tokens) t
                FROM session_provider_usage_events
                WHERE occurred_at_ms>0 AND model_name IN ({placeholders})
                GROUP BY ym, model_name ORDER BY ym
                """,
                top_models,
            ).fetchall()
            f["model_evolution"] = rows

    # 6. Workflow / failure signals ----------------------------------------- #
    if _has_table(conn, "session_work_events"):
        f["work_event_types"] = conn.execute(
            "SELECT work_event_type, COUNT(*) c FROM session_work_events GROUP BY work_event_type ORDER BY c DESC LIMIT 12"
        ).fetchall()
    msg_counts = [
        int(r[0]) for r in conn.execute("SELECT message_count FROM sessions WHERE message_count > 0").fetchall()
    ]
    msg_counts.sort()
    if msg_counts:
        f["msg_median"] = msg_counts[len(msg_counts) // 2]
        f["msg_p90"] = msg_counts[int(len(msg_counts) * 0.9)]
        f["msg_max"] = msg_counts[-1]
        f["msg_hist"] = _log_histogram([float(x) for x in msg_counts])
    if _has_table(conn, "actions") and _has_table(conn, "blocks"):
        f["structured_failure_followups"] = _structured_failure_followups(conn)
    return f


_ACKNOWLEDGMENT_MARKERS = (
    "failed",
    "failure",
    "error",
    "errored",
    "exit code",
    "non-zero",
    "nonzero",
    "exception",
    "traceback",
    "permission denied",
    "not found",
    "timed out",
    "timeout",
    "fix",
    "failing",
)


def _classify_failed_followup(text: str | None) -> str:
    """Classify the next assistant turn after a structured tool failure.

    This deliberately avoids mining a positive success claim from prose. The
    structured failure is the anchor; the next assistant turn is only checked
    for explicit acknowledgment markers. Missing/short follow-up stays
    ambiguous rather than being counted as silent proceed.
    """

    if text is None:
        return "ambiguous"
    normalized = " ".join(text.lower().split())
    if len(normalized) < 20:
        return "ambiguous"
    if any(marker in normalized for marker in _ACKNOWLEDGMENT_MARKERS):
        return "acknowledged"
    return "silent_proceed"


def _structured_failure_followups(conn: sqlite3.Connection, *, sample_limit: int = 20) -> dict[str, object]:
    """Return adjacency-anchored follow-up stats for failed tool outcomes."""

    rows = conn.execute(
        """
        WITH failed AS (
            SELECT
                a.session_id,
                a.message_id,
                a.tool_name,
                a.tool_command,
                a.is_error,
                a.exit_code,
                m.position,
                m.model_name AS tool_message_model
            FROM actions AS a
            JOIN messages AS m ON m.message_id = a.message_id
            WHERE COALESCE(a.is_error, 0) = 1
               OR (a.exit_code IS NOT NULL AND a.exit_code != 0)
        ),
        next_message AS (
            SELECT
                f.*,
                (
                    SELECT nm.message_id
                    FROM messages AS nm
                    WHERE nm.session_id = f.session_id
                      AND nm.role = 'assistant'
                      AND nm.position > f.position
                    ORDER BY nm.position
                    LIMIT 1
                ) AS next_message_id
            FROM failed AS f
        )
        SELECT
            n.session_id,
            n.message_id,
            n.tool_name,
            n.tool_command,
            n.is_error,
            n.exit_code,
            COALESCE(nm.model_name, n.tool_message_model, '') AS model_name,
            n.next_message_id,
            substr(group_concat(COALESCE(b.text, ''), '\n'), 1, 1200) AS next_text
        FROM next_message AS n
        LEFT JOIN messages AS nm ON nm.message_id = n.next_message_id
        LEFT JOIN blocks AS b ON b.message_id = n.next_message_id
        GROUP BY
            n.session_id,
            n.message_id,
            n.tool_name,
            n.tool_command,
            n.is_error,
            n.exit_code,
            model_name,
            n.next_message_id
        """,
    ).fetchall()

    by_tool: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    samples: list[dict[str, object]] = []
    totals = {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0}
    for row in rows:
        classification = _classify_failed_followup(row["next_text"])
        tool = str(row["tool_name"] or "unknown")
        model = str(row["model_name"] or "unknown")
        totals["failed_outcomes"] += 1
        totals[classification] += 1
        by_tool.setdefault(tool, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_model.setdefault(model, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_tool[tool]["failed_outcomes"] += 1
        by_tool[tool][classification] += 1
        by_model[model]["failed_outcomes"] += 1
        by_model[model][classification] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "classification": classification,
                    "session_ref": f"session:{row['session_id']}",
                    "tool_message_ref": f"message:{row['message_id']}",
                    "next_message_ref": f"message:{row['next_message_id']}" if row["next_message_id"] else None,
                    "tool_name": tool,
                    "exit_code": row["exit_code"],
                    "is_error": row["is_error"],
                    "tool_command_preview": str(row["tool_command"] or "")[:160],
                }
            )

    def ranked(mapping: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, counts in mapping.items():
            failed = counts["failed_outcomes"]
            silent = counts["silent_proceed"]
            out.append(
                {
                    "name": key,
                    **counts,
                    "silent_rate": (silent / failed) if failed else 0.0,
                }
            )
        return sorted(out, key=lambda item: (-int(item["failed_outcomes"]), str(item["name"])))

    return {
        "definition": (
            "failed tool outcomes are structured rows where is_error=1 or exit_code!=0; "
            "classification inspects only the next assistant turn for explicit failure acknowledgment markers"
        ),
        "totals": totals,
        "by_tool": ranked(by_tool),
        "by_model": ranked(by_model),
        "samples": samples,
    }


def _log_histogram(values: list[float]) -> list[tuple[str, int]]:
    """Bucket positive values into log10 bands."""
    import math

    buckets: dict[int, int] = {}
    for v in values:
        if v <= 0:
            continue
        b = int(math.floor(math.log10(v)))
        buckets[b] = buckets.get(b, 0) + 1
    out: list[tuple[str, int]] = []
    for b in range(min(buckets), max(buckets) + 1):
        lo, hi = 10.0**b, 10.0 ** (b + 1)
        out.append((f"{_fmt_num(lo)}-{_fmt_num(hi)}", buckets.get(b, 0)))
    return out


# Claude subscription credit rates (credits/token), reverse-engineered by
# she-llac.com/claude-limits. Cache reads are FREE on plans; cache writes are
# billed at the input rate. Output is 5× input; Opus is 5× Haiku.
_CREDIT_RATES = {
    "opus": (10 / 15, 50 / 15),
    "sonnet": (6 / 15, 30 / 15),
    "haiku": (2 / 15, 10 / 15),
}
# Max-20× plan capacity (credits): 11M/5h, ~83.333M/week, ~361.1M/month.
_MAX20_CREDITS_PER_MONTH = 361_100_000.0


def _model_family(model_name: str) -> str | None:
    m = model_name.lower()
    for fam in _CREDIT_RATES:
        if fam in m:
            return fam
    return None


def _subscription_credits(per_model_classes: list[Any]) -> dict[str, Any]:
    """Estimate Claude subscription credits from priced per-model token classes.

    credits = (input + cache_write) × in_rate + output × out_rate; cache reads
    are free. Rows are (model_name, input, output, cache_write).
    """
    total = 0.0
    by_family: dict[str, float] = {}
    matched = False
    for model_name, inp, outp, cwrite in per_model_classes:
        fam = _model_family(str(model_name))
        if fam is None:
            continue
        matched = True
        in_rate, out_rate = _CREDIT_RATES[fam]
        credits = (int(inp) + int(cwrite)) * in_rate + int(outp) * out_rate
        by_family[fam] = by_family.get(fam, 0.0) + credits
        total += credits
    return {"total": total, "by_family": by_family, "matched": matched}


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def build_report(f: dict[str, Any], out_dir: Path, *, archive_label: str) -> str:
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    def write_chart(name: str, svg: str) -> str:
        (charts / name).write_text(svg, encoding="utf-8")
        return f"charts/{name}"

    md: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    md.append("# AI-Agent Forensics")
    md.append("")
    md.append(
        f"*Generated {now} by `scripts/agent_forensics.py` against `{archive_label}` "
        f"(read-only). Aggregate statistics only — no message content.*"
    )
    md.append("")

    # Headline
    md.append("## Headline")
    md.append("")
    md.append(f"- **Span:** {f.get('span_start')} → {f.get('span_end')}")
    md.append(f"- **Sessions:** {_fmt_int(int(f.get('sessions', 0)))}")
    md.append(f"- **Messages:** {_fmt_int(int(f.get('messages', 0)))}")
    if "tok_input" in f:
        total_tok = int(f["tok_input"]) + int(f["tok_output"]) + int(f["tok_cache_read"]) + int(f["tok_cache_write"])
        md.append(f"- **Total tokens accounted:** {_fmt_num(total_tok)} (all providers)")
    if "total_cost" in f:
        md.append(
            f"- **Priced cost (Claude Code only):** {_fmt_usd(float(f['total_cost']))} "
            f"— list-price equivalent for the priced subset; other providers report token "
            f"counts without per-token cost (see Token economy)."
        )
    md.append("")

    # Origins
    origins = f.get("origins") or []
    if origins:
        md.append("## Corpus by origin")
        md.append("")
        labels = [str(r[0]) for r in origins]
        vals = [float(r[1]) for r in origins]
        md.append(f"![origins]({write_chart('origins.svg', hbar_chart('Sessions by origin', labels, vals))})")
        md.append("")

    # Temporal rhythm
    spm = f.get("sessions_per_month") or []
    if spm:
        labels = [str(r[0]) for r in spm]
        vals = [float(r[1]) for r in spm]
        md.append("## Temporal rhythm")
        md.append("")
        md.append(
            f"![sessions/month]({write_chart('sessions_per_month.svg', bar_chart('Sessions per month', labels, vals))})"
        )
        md.append("")
        peak = max(spm, key=lambda r: r[1])
        md.append(f"Peak month: **{peak[0]}** with {_fmt_int(int(peak[1]))} sessions.")
        md.append("")
    tpm = f.get("tokens_per_month") or []
    if tpm:
        labels = [str(r[0]) for r in tpm]
        vals = [float(r[1] or 0) for r in tpm]
        md.append(
            f"![tokens/month]({write_chart('tokens_per_month.svg', bar_chart('Tokens per month (last-event totals)', labels, vals))})"
        )
        md.append("")

    # Token economy
    econ = f.get("economy_by_provenance") or {}
    if "tok_input" in f:
        ci, co = int(f["tok_input"]), int(f["tok_output"])
        cr, cw = int(f["tok_cache_read"]), int(f["tok_cache_write"])
        md.append("## Token economy")
        md.append("")
        md.append(
            "Token classes are priced very differently and are accounted by two "
            "provenances: **priced** rows (Claude Code, carrying a computed "
            "`cost_usd`) and **origin_reported** rows (other providers report token "
            "counts but no per-token cost). They are kept separate below — pairing "
            "the priced cost with the all-provider token total would misstate both."
        )
        md.append("")
        md.append("| provenance | input | output | cache read | cache write | cost |")
        md.append("|---|---|---|---|---|---|")
        for prov in ("priced", "origin_reported"):
            v = econ.get(prov)
            if not v:
                continue
            cost_cell = _fmt_usd(float(v["cost"])) if float(v["cost"]) > 0 else "—"
            md.append(
                f"| {prov} | {_fmt_num(int(v['input']))} | {_fmt_num(int(v['output']))} | "
                f"{_fmt_num(int(v['cache_read']))} | {_fmt_num(int(v['cache_write']))} | {cost_cell} |"
            )
        md.append(
            f"| **all** | {_fmt_num(ci)} | {_fmt_num(co)} | {_fmt_num(cr)} | {_fmt_num(cw)} | "
            f"**{_fmt_usd(float(f.get('total_cost', 0)))}** |"
        )
        md.append("")
        # The headline efficiency finding lives in the priced (Claude Code) subset,
        # where caching is tracked: cache reads vs fresh input.
        pv = econ.get("priced")
        if pv and int(pv["input"]) > 0:
            ratio = int(pv["cache_read"]) / int(pv["input"])
            priced_tok = int(pv["input"]) + int(pv["output"]) + int(pv["cache_read"]) + int(pv["cache_write"])
            blended = float(pv["cost"]) / priced_tok * 1e6 if priced_tok else 0.0
            md.append(
                f"**Cache amplification (Claude Code): {ratio:.0f}×.** Prompt caching served "
                f"{_fmt_num(int(pv['cache_read']))} cache-read tokens against only "
                f"{_fmt_num(int(pv['input']))} fresh input — the model re-reads cached context "
                f"~{ratio:.0f}× more than it ingests fresh. Effective blended rate "
                f"**${blended:.3f}/M** (vs $15+/M list for fresh Opus input), because cache "
                f"reads dominate the volume and are cheap. A token counter that ignores "
                f"cache reads understates real usage by orders of magnitude."
            )
        if "tok_reasoning" in f:
            md.append("")
            md.append(f"Reasoning output (provider events, per-event deltas): **{_fmt_num(int(f['tok_reasoning']))}**.")
        md.append("")
        mix = hbar_chart(
            "Token mix",
            ["cache read", "fresh input", "cache write", "output"] + (["reasoning"] if "tok_reasoning" in f else []),
            [float(cr), float(ci), float(cw), float(co)]
            + ([float(f["tok_reasoning"])] if "tok_reasoning" in f else []),
        )
        md.append(f"![token mix]({write_chart('token_mix.svg', mix)})")
        md.append("")

    # Cost
    if "cost_by_model" in f:
        md.append("## Cost")
        md.append("")
        cbm = f["cost_by_model"]
        labels = [str(r[0]) for r in cbm]
        vals = [float(r[2]) for r in cbm]
        md.append(
            f"![cost by model]({write_chart('cost_by_model.svg', hbar_chart('Cost by model (USD)', labels, vals, unit=''))})"
        )
        md.append("")
        md.append("| model | sessions | cost |")
        md.append("|-------|----------|------|")
        for r in cbm:
            md.append(f"| {_esc(str(r[0]))} | {_fmt_int(int(r[1]))} | {_fmt_usd(float(r[2]))} |")
        md.append("")
        if "cost_median" in f:
            md.append(
                f"Per-session cost (priced sessions: {_fmt_int(int(f.get('cost_session_count', 0)))}): "
                f"median **{_fmt_usd(float(f['cost_median']))}**, "
                f"p90 **{_fmt_usd(float(f['cost_p90']))}**, "
                f"max **{_fmt_usd(float(f['cost_max']))}**."
            )
            md.append("")
        if "cost_hist" in f:
            ch = f["cost_hist"]
            md.append(
                f"![cost distribution]({write_chart('cost_hist.svg', bar_chart('Per-session cost distribution (USD, log bands)', [b[0] for b in ch], [float(b[1]) for b in ch]))})"
            )
            md.append("")
        cost_provenance = f.get("cost_provenance") or []
        if cost_provenance:
            md.append(
                "Cost provenance: "
                + ", ".join(f"{_esc(str(r[0]))} ({_fmt_int(int(r[1]))})" for r in cost_provenance)
                + "."
            )
            md.append("")

    # Subscription reality (the priced cost above is API-list-equivalent)
    sub = f.get("subscription_credits") or {}
    if sub.get("matched") and float(sub.get("total", 0)) > 0:
        credits = float(sub["total"])
        api_cost = float(f.get("total_cost", 0))
        md.append("## Subscription reality")
        md.append("")
        md.append(
            "The priced cost above is **API-list-equivalent**. The operator runs "
            "Claude Code on a **Max subscription**, where the economics differ "
            "sharply: **cache reads are free** (the API charges 10% of input), and "
            "usage is metered in *credits* — `(input + cache_write) × in_rate + "
            "output × out_rate`, cache reads excluded. (Credit rates "
            "reverse-engineered by [she-llac.com](https://she-llac.com/claude-limits); "
            "estimate, not official.)"
        )
        md.append("")
        bf = sub.get("by_family") or {}
        md.append(
            "Estimated subscription credits consumed: **"
            + _fmt_num(credits)
            + "**"
            + (" (" + ", ".join(f"{k} {_fmt_num(v)}" for k, v in sorted(bf.items())) + ")" if bf else "")
            + "."
        )
        md.append("")
        plan_months = credits / _MAX20_CREDITS_PER_MONTH
        md.append(
            f"At the Max-20× cap (~{_fmt_num(_MAX20_CREDITS_PER_MONTH)} credits/month, $200/mo), "
            f"that is **~{plan_months:.0f} plan-months** of capacity — but cache reads, which "
            f"dominate the token volume ({_fmt_num(int(f.get('tok_cache_read', 0)))} read), cost "
            f"**zero** credits. The same workload on the API would bill "
            f"**{_fmt_usd(api_cost)}** (list-equivalent), so the subscription captures the bulk of "
            f"that as value: the free-cache-read effect is exactly why plan pricing beats the API "
            f"by ~13–37× for agentic, cache-heavy use."
        )
        md.append("")

    # Model evolution
    me = f.get("model_evolution")
    if me:
        months = sorted({str(r[0]) for r in me})
        idx = {m: i for i, m in enumerate(months)}
        series: dict[str, list[float]] = {m: [0.0] * len(months) for m in f.get("top_models", [])}
        for ym, model, tok in me:
            if str(model) in series:
                series[str(model)][idx[str(ym)]] = float(tok or 0)
        md.append("## Model evolution")
        md.append("")
        md.append(
            f"![model evolution]({write_chart('model_evolution.svg', line_chart('Tokens per month by model (top 5)', months, series))})"
        )
        md.append("")

    # Workflow / failure
    wet = f.get("work_event_types")
    if wet:
        md.append("## Workflow shape")
        md.append("")
        labels = [str(r[0]) for r in wet]
        vals = [float(r[1]) for r in wet]
        md.append(f"![work events]({write_chart('work_events.svg', hbar_chart('Work-event types', labels, vals))})")
        md.append("")
    if "msg_median" in f:
        md.append(
            f"Session length: median **{_fmt_int(int(f['msg_median']))}** messages, "
            f"p90 **{_fmt_int(int(f['msg_p90']))}**, max **{_fmt_int(int(f['msg_max']))}**."
        )
        md.append("")
        if "msg_hist" in f:
            mh = f["msg_hist"]
            md.append(
                f"![session length]({write_chart('session_length.svg', bar_chart('Session length distribution (messages, log bands)', [b[0] for b in mh], [float(b[1]) for b in mh]))})"
            )
            md.append("")

    followups = f.get("structured_failure_followups")
    if isinstance(followups, dict):
        totals = followups.get("totals", {})
        failed_total = int(totals.get("failed_outcomes", 0)) if isinstance(totals, dict) else 0
        if failed_total:
            md.append("## Structured failure follow-up")
            md.append("")
            md.append(
                "This section is the claim-vs-evidence core: it anchors on structured tool "
                "outcomes, not prose-mined success claims. A failed outcome is any action row "
                "with `is_error=1` or non-zero `exit_code`. The next assistant turn is then "
                "classified as `acknowledged`, `silent_proceed`, or `ambiguous` by a small, "
                "auditable acknowledgment-marker rule. Treat this as the structured core plus "
                "a lexical acknowledgment heuristic, not an LLM judgment."
            )
            md.append("")
            ack = int(totals.get("acknowledged", 0)) if isinstance(totals, dict) else 0
            silent = int(totals.get("silent_proceed", 0)) if isinstance(totals, dict) else 0
            amb = int(totals.get("ambiguous", 0)) if isinstance(totals, dict) else 0
            md.append(
                f"Failed structured outcomes: **{_fmt_int(failed_total)}**. "
                f"Acknowledged: **{_fmt_int(ack)}**; silent-proceed: **{_fmt_int(silent)}** "
                f"({silent / failed_total:.1%}); ambiguous: **{_fmt_int(amb)}**."
            )
            md.append("")
            by_tool = [item for item in followups.get("by_tool", []) if isinstance(item, dict)]
            if by_tool:
                md.append("| tool | failed outcomes | acknowledged | silent-proceed | ambiguous | silent rate |")
                md.append("|---|---:|---:|---:|---:|---:|")
                for item in by_tool[:12]:
                    failed = int(item["failed_outcomes"])
                    md.append(
                        f"| {_esc(str(item['name']))} | {_fmt_int(failed)} | "
                        f"{_fmt_int(int(item['acknowledged']))} | {_fmt_int(int(item['silent_proceed']))} | "
                        f"{_fmt_int(int(item['ambiguous']))} | {float(item['silent_rate']):.1%} |"
                    )
                md.append("")
            by_model = [item for item in followups.get("by_model", []) if isinstance(item, dict)]
            if by_model:
                md.append("| model | failed outcomes | acknowledged | silent-proceed | ambiguous | silent rate |")
                md.append("|---|---:|---:|---:|---:|---:|")
                for item in by_model[:12]:
                    failed = int(item["failed_outcomes"])
                    md.append(
                        f"| {_esc(str(item['name']))} | {_fmt_int(failed)} | "
                        f"{_fmt_int(int(item['acknowledged']))} | {_fmt_int(int(item['silent_proceed']))} | "
                        f"{_fmt_int(int(item['ambiguous']))} | {float(item['silent_rate']):.1%} |"
                    )
                md.append("")
            samples = [item for item in followups.get("samples", []) if isinstance(item, dict)]
            if samples:
                md.append("Sample ref-backed instances:")
                md.append("")
                for item in samples[:8]:
                    next_ref = item.get("next_message_ref") or "no-next-assistant-turn"
                    command = str(item.get("tool_command_preview") or "").replace("\n", " ")
                    md.append(
                        f"- `{item.get('classification')}` `{item.get('tool_name')}` "
                        f"exit=`{item.get('exit_code')}` error=`{item.get('is_error')}` "
                        f"[{item.get('tool_message_ref')}] -> [{next_ref}]"
                        + (f" — `{_esc(command)}`" if command else "")
                    )
                md.append("")

    md.append("---")
    md.append("")
    md.append(
        "*Reproduce: `python scripts/agent_forensics.py --archive <root> --out <dir>`. "
        "Numbers are read directly from the archive's materialized analytics and action tables "
        "(`session_model_usage`, `session_provider_usage_events`, `session_work_events`, "
        "`sessions`, `actions`, `messages`, `blocks`).*"
    )
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine a Polylogue archive for usage findings.")
    ap.add_argument(
        "--archive", default=None, help="Archive root or index.db path (default: $POLYLOGUE_ARCHIVE_ROOT or XDG)."
    )
    ap.add_argument("--out", default="agent-forensics", help="Output directory for report.md and charts/.")
    args = ap.parse_args()

    index_db = resolve_index_db(args.archive)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_ro(index_db)
    try:
        findings = analyze(conn)
    finally:
        conn.close()
    followups = findings.get("structured_failure_followups")
    if isinstance(followups, dict):
        (out_dir / "structured_failure_followups.json").write_text(
            json.dumps(followups, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = build_report(findings, out_dir, archive_label=str(index_db.parent))
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir / 'report.md'} ({int(findings.get('sessions', 0)):,} sessions analyzed)")


if __name__ == "__main__":
    main()
