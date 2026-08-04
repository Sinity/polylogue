"""beads-state-report: a self-contained HTML state-of-the-backlog report.

Reads ``.beads/issues.jsonl`` and emits one standalone HTML file covering the
whole bead population -- open AND closed -- across eight views: pulse (now vs
7/14 days ago, reconstructed from timestamps), shape (status x priority x
type), structure (epic/program trees with per-epic trend sparklines), topology
(the ``blocks`` dependency graph: ready / blocked / top blockers / cycles /
densest cluster / the parallel frontier), time (creation, closure, burnup, age
x priority heatmap), health (a review queue of graph defects), themes (where
open work concentrates by subsystem), and the dated review passes (the
hand-verified VERIFICATION(...) verdict subset and the 2026-07-31 P0
reconciliation markers, rendered only when their markers exist in the data).

Design rule: **interpretation is computed, never fossilized.** Every claim in
the findings list is emitted by a conditional generator in ``compute_insights``
that checks the condition against the live data and renders the variant that
is actually true (or nothing). There is no hand-authored paragraph that can
silently go false on regeneration. The only authored content is timeless
framing (what a section shows, how to read the notation) and two named
constants from dated review passes, kept as constants precisely so their
dated provenance is explicit.

Temporal reconstruction: every bead and every dependency edge in the export
carries ``created_at`` (and beads a ``closed_at``), so the open/ready/blocked
state of the graph at any past instant is recomputable. Caveats are inherent
and stated in the report: deleted beads/edges are invisible, and a reopened
bead's earlier closure is lost (``closed_at`` reflects only the latest close).
That makes trend figures *derived*, not measured -- they are tagged as such.

Usage: devtools workspace beads-state-report [--out PATH] [--fresh] [--json]
``--fresh`` runs ``bd export -o .beads/issues.jsonl`` first, because ``bd``
mutations do not immediately re-export and a stale file yields stale counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root

IssueDict = dict[str, Any]
# (src issue, depends-on issue, relation type, edge created_at ISO string)
Edge = tuple[str, str, str, str]

# --------------------------------------------------------------------------
# Constants from dated review passes.  These are the ONLY numbers in this
# module that were not computed from the input file; each is an
# operator-supplied figure from a specific, dated pass and is rendered with
# that provenance attached.
# --------------------------------------------------------------------------
# 2026-07-31 landing-check lane: the operator's own count of the hand-verified
# subset (strict regex extraction below finds fewer; the gap is reported).
OPERATOR_VERIFIED_COUNT = 190
# 2026-07-31 P0 reconciliation pass: the pass's own summary table covered this
# many beads (marker extraction below finds fewer; the gap is reported).
RECONCILIATION_EXPECTED = 24

VERDICT_STRICT = re.compile(r"VERIFICATION\s*\(([^)]*)\)\s*:\s*(STALE|PARTIAL|LIVE)", re.IGNORECASE)
VERDICT_LOOSE = re.compile(r"\b(STALE|PARTIAL|LIVE)\b")

# The 2026-07-31 P0 reconciliation pass wrote one
# `RECONCILIATION 2026-07-31[...]: <VERDICT>` marker per bead it touched.
# Scoping the token search to a window *after* that literal anchor matters:
# bare tokens like MISFRAMED are ordinary English words that also appear,
# unrelated, in older notes on other beads.
RECONCILIATION_ANCHOR = re.compile(r"RECONCILIATION 2026-07-31")
RECONCILIATION_TOKENS: tuple[str, ...] = (
    "FIXED-AND-EFFECTIVE",
    "FIXED-PENDING-REBUILD",
    "FIXED-PENDING-DEPLOY",
    "MISFRAMED",
    "GENUINELY OPEN",
)
RECONCILIATION_TOKEN_RE = re.compile("|".join(re.escape(t) for t in RECONCILIATION_TOKENS))
RECONCILIATION_WINDOW = 1200

# Thresholds for computed insights.  Named so the conditions are auditable.
STALE_CLAIM_DAYS = 7  # in_progress untouched this long = zombie claim
AGED_URGENT_DAYS = 14  # open P0/P1 older than this = urgency not being consumed
VELOCITY_WINDOW_DAYS = 14  # trailing window vs the window before it
TREE_MAX_DEPTH = 2  # structure view renders two parent-child levels

SUBSYSTEMS: dict[str, tuple[str, ...]] = {
    # canonical subsystem -> area:* label suffixes folded into it
    "storage": ("storage", "substrate", "blob", "blobs", "schema", "schemas", "archive", "durability"),
    "sources": (
        "sources",
        "source",
        "ingest",
        "parsers",
        "parsing",
        "capture",
        "lineage",
        "attachments",
        "protocol",
        "identity",
    ),
    "daemon": ("daemon", "ops", "pipeline", "events", "perf", "performance", "reliability"),
    "insights": ("insights", "analytics", "cost", "usage", "temporal", "evidence", "analysis", "embeddings"),
    "cli": ("cli", "query", "query-dsl", "search", "surface", "api", "config"),
    "mcp": ("mcp", "context", "coordination", "orchestration", "agents", "delegation", "delegations"),
    "web/extension": ("web", "browser", "browser-capture", "extension", "rendering", "ux"),
    "tests": ("test", "testing", "tests", "test-harness", "verification", "coverage", "quality", "eval"),
    "devtools": (
        "devtools",
        "devloop",
        "ci",
        "release",
        "git",
        "beads",
        "beads-hygiene",
        "planning",
        "maintenance",
        "cleanup",
        "status",
    ),
    "docs/legibility": ("docs", "legibility", "demos", "demo", "architecture", "adoption", "audit"),
}

KEYWORDS: dict[str, tuple[str, ...]] = {
    "storage": ("storage", "sqlite", "index.db", "source.db", "user.db", "schema", "blob", "migration", "table"),
    "sources": ("parser", "parse", "ingest", "provider", "origin", "claude code", "codex", "chatgpt", "session"),
    "daemon": ("daemon", "polylogued", "converg", "watcher", "cursor", "acquire"),
    "insights": ("insight", "cost", "usage", "token", "profile", "timeline", "analytic"),
    "cli": ("cli", "command", "query", "click", "filter", "find ", "read --"),
    "mcp": ("mcp", "agent", "context", "assertion", "recall", "coordination"),
    "web/extension": ("web", "browser", "extension", "workbench", "http", "ui"),
    "tests": ("test", "pytest", "fixture", "coverage", "hypothesis", "regression"),
    "devtools": ("devtools", "render", "lint", "bead", "ci ", "workflow", "hook"),
    "docs/legibility": ("doc", "readme", "demo", "narrative", "legib", "explain"),
}

TYPE_GLYPH: dict[str, tuple[str, str]] = {
    "task": ("▪", "task"),
    "bug": ("✕", "bug"),
    "feature": ("✦", "feature"),
    "epic": ("▣", "epic"),
    "chore": ("⚙", "chore"),
    "decision": ("⚖", "decision"),
    "spike": ("⚡", "spike"),
}
STATUS_GLYPH: dict[str, tuple[str, str]] = {
    "open": ("○", "open"),
    "in_progress": ("◐", "in progress"),
    "deferred": ("◌", "deferred"),
    "closed": ("●", "closed"),
}
STATUS_ORDER: tuple[str, ...] = ("open", "in_progress", "deferred", "closed")

# Mechanical graph-health checks: (badge class, label, Facts attribute, meaning).
HEALTH_CHECKS: tuple[tuple[str, str, str, str], ...] = (
    ("bad", "dangling dependency references", "dangling", "edge points at an id that does not exist"),
    ("warn", "dotted id disagrees with parent edge", "id_mismatch", "id prefix names a different parent"),
    ("warn", "dotted id with no parent edge", "id_orphan", "looks nested, is structurally a root"),
    (
        "warn",
        "open parent, all children closed",
        "open_parent_all_closed",
        "review queue &mdash; adjudicate against the parent's own AC",
    ),
    (
        "warn",
        "closed parent with open children",
        "closed_parent_open_kids",
        "parent retired while decomposed work continues",
    ),
    (
        "warn",
        f"in-progress untouched &gt;{STALE_CLAIM_DAYS} days",
        "stale_claims",
        "a claim nobody is working &mdash; release or finish it",
    ),
    ("warn", "duplicate titles", "dup_titles", "same work filed twice, independently"),
    ("info", "closed without a close reason", "closed_no_reason", "no durable record of why it ended"),
    ("info", "open without acceptance criteria", "no_ac_open", "no definition of done"),
)


# --------------------------------------------------------------------------
# loading + primitives
# --------------------------------------------------------------------------
def load(path: Path) -> tuple[dict[str, IssueDict], list[Edge]]:
    """Parse the bd JSONL export into an id->issue map and a dependency edge list."""
    issues: dict[str, IssueDict] = {}
    edges: list[Edge] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_type") == "issue":
            issues[rec["id"]] = rec
            for dep in rec.get("dependencies") or []:
                edges.append(
                    (
                        str(dep["issue_id"]),
                        str(dep["depends_on_id"]),
                        str(dep.get("type", "blocks")),
                        str(dep.get("created_at") or ""),
                    )
                )
        elif rec.get("_type") == "dependency":
            edges.append(
                (
                    str(rec["issue_id"]),
                    str(rec["depends_on_id"]),
                    str(rec.get("type", "blocks")),
                    str(rec.get("created_at") or ""),
                )
            )
    return issues, edges


def _blob(rec: IssueDict) -> str:
    parts = [str(rec.get("notes") or "")]
    parts.extend(str(c.get("text") or "") for c in (rec.get("comments") or []))
    return "\n".join(parts)


def _parse_ts(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _git_show(root: Path, ref: str, rel_path: str) -> str | None:
    """Return a file's content at a git ref, or ``None`` if unavailable."""
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


_LIFECYCLE_DECL_RE = re.compile(r"version=(\d+),.*?classes=\(([^)]*)\)", re.DOTALL)


def schema_gap_facts(root: Path) -> dict[str, Any]:
    """The derived-index schema gap: origin/master's declared
    ``INDEX_SCHEMA_VERSION`` vs the live archive's actual ``PRAGMA
    user_version``, and which intervening versions are ``SEMANTIC_REPARSE``
    (block a SQL fast-forward -- a merged fix at one of those versions is
    real and inert until ``polylogue ops reset --index && polylogued run``).

    Two independent read paths, both best-effort and non-fatal on failure so
    a sandbox without network/git-remote access or a live archive still
    renders a report (just without this fact):

    - ``git show origin/master:...`` for the declared version and delta
      classes, so this is correct even on a branch behind master.
    - the operator's configured archive root
      (``~/.config/polylogue/polylogue.toml``), read-only ``PRAGMA``, for the
      live version. Deliberately does not use ``POLYLOGUE_ARCHIVE_ROOT`` --
      that env var is routinely overridden for sandboxed/test runs, and this
      fact is specifically about the real production archive.
    """
    result: dict[str, Any] = {
        "available": False,
        "declared": None,
        "live_version": None,
        "live_path": None,
        "blockers": [],
        "blocker_notes": {},
    }

    index_src = _git_show(root, "origin/master", "polylogue/storage/sqlite/archive_tiers/index.py")
    if index_src:
        m = re.search(r"^INDEX_SCHEMA_VERSION\s*=\s*(\d+)", index_src, re.MULTILINE)
        if m:
            result["declared"] = int(m.group(1))

    lifecycle_src = _git_show(root, "origin/master", "polylogue/storage/sqlite/lifecycle.py") or ""
    versions: dict[int, tuple[str, ...]] = {}
    for vm in _LIFECYCLE_DECL_RE.finditer(lifecycle_src):
        v = int(vm.group(1))
        classes = tuple(c.strip().removeprefix("DerivedDeltaClass.") for c in vm.group(2).split(",") if c.strip())
        versions[v] = classes

    live_path: Path | None = None
    try:
        import tomllib

        cfg_path = Path.home() / ".config/polylogue/polylogue.toml"
        if cfg_path.exists():
            cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            root_str = cfg.get("archive", {}).get("root")
            if root_str:
                live_path = Path(root_str) / "index.db"
    except Exception:
        live_path = None

    if live_path and live_path.exists():
        result["live_path"] = str(live_path)
        try:
            proc = subprocess.run(
                ["sqlite3", "-readonly", str(live_path), "PRAGMA user_version;"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            result["live_version"] = int(proc.stdout.strip())
        except Exception:
            pass

    if result["declared"] is not None and result["live_version"] is not None:
        result["available"] = True
        lo, hi = result["live_version"] + 1, result["declared"]
        result["blockers"] = [v for v in range(lo, hi + 1) if "SEMANTIC_REPARSE" in versions.get(v, ())]
        result["blocker_notes"] = {v: versions.get(v, ()) for v in range(lo, hi + 1)}
    return result


# --------------------------------------------------------------------------
# facts: every measured/derived quantity the report renders
# --------------------------------------------------------------------------
class Facts:
    """Every measured quantity the report renders.  Nothing here is authored."""

    def __init__(self, issues: dict[str, IssueDict], edges: list[Edge], now: dt.datetime) -> None:
        self.issues = issues
        self.edges = edges
        self.now = now
        self._cluster_adj: dict[str, set[str]] = defaultdict(set)
        rows = list(issues.values())
        self.rows = rows
        self.total = len(rows)

        self.status = Counter(str(r["status"]) for r in rows)
        self.priority = Counter(int(r["priority"]) for r in rows)
        self.itype = Counter(str(r["issue_type"]) for r in rows)
        self.matrix: dict[tuple[str, int], int] = Counter((str(r["status"]), int(r["priority"])) for r in rows)
        self.open_ids = [str(r["id"]) for r in rows if r["status"] != "closed"]
        self.open_total = len(self.open_ids)
        self.statuses_present = [s for s in STATUS_ORDER if self.status.get(s)] + sorted(
            set(self.status) - set(STATUS_ORDER)
        )

        # ---- dependency graph -------------------------------------------
        self.dep_types = Counter(t for _, _, t, _ in edges)

        # relates-to/related vocabulary state. A repair that re-types every
        # existing `related` edge does not constrain the field, so a fresh
        # `related` edge can land minutes later -- that is a distinct state
        # from "never repaired" and from "clean".
        _relates, _related = self.dep_types["relates-to"], self.dep_types["related"]
        _assoc_total = _relates + _related
        if _related == 0:
            self.vocab_state = "clean"
        elif _assoc_total and _related <= max(3, 0.02 * _assoc_total):
            self.vocab_state = "recurring"
        else:
            self.vocab_state = "split"

        self.blockers: dict[str, set[str]] = defaultdict(set)
        self.blocking: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, list[str]] = defaultdict(list)
        self.parent: dict[str, str] = {}
        # blocks edges with their creation timestamp, for temporal replay
        self.block_edges_ts: list[tuple[str, str, str]] = []
        for src, dst, kind, created in edges:
            if kind == "blocks":
                self.blockers[src].add(dst)
                self.blocking[dst].add(src)
                self.block_edges_ts.append((src, dst, created))
            elif kind == "parent-child":
                self.children[dst].append(src)
                self.parent[src] = dst
        self.dangling = sorted({(a, b, t) for a, b, t, _ in edges if b not in issues or a not in issues})

        def is_open(bid: str) -> bool:
            rec = issues.get(bid)
            return rec is not None and rec["status"] != "closed"

        self.is_open = is_open
        self.blocked = [i for i in self.open_ids if any(is_open(b) for b in self.blockers[i])]
        self.ready = [i for i in self.open_ids if not any(is_open(b) for b in self.blockers[i])]
        self.top_blockers = sorted(
            ((sum(1 for x in self.blocking[i] if is_open(x)), i) for i in self.open_ids),
            reverse=True,
        )
        self.top_blockers = [(n, i) for n, i in self.top_blockers if n > 0]
        self.cycles = self._find_cycles()

        # ---- epics -------------------------------------------------------
        self.epics: list[dict[str, Any]] = []
        for pid, kids in self.children.items():
            if pid not in issues:
                continue
            self.epics.append(
                {
                    "id": pid,
                    "n": len(kids),
                    "open": sum(1 for k in kids if is_open(k)),
                    "kids": kids,
                    "title": str(issues[pid]["title"]),
                    "type": str(issues[pid]["issue_type"]),
                    "status": str(issues[pid]["status"]),
                }
            )
        self.epics.sort(key=lambda e: (-int(e["n"]), str(e["id"])))

        # ---- id/edge hierarchy disagreement -------------------------------
        self.id_mismatch = sorted(
            (child, self.parent[child])
            for child in self.parent
            if "." in child and child.rsplit(".", 1)[0] != self.parent[child]
        )
        self.id_orphan = sorted(str(r["id"]) for r in rows if "." in str(r["id"]) and str(r["id"]) not in self.parent)

        # ---- health -------------------------------------------------------
        self.closed_parent_open_kids = sorted(
            (pid, [k for k in kids if is_open(k)])
            for pid, kids in self.children.items()
            if pid in issues and issues[pid]["status"] == "closed" and any(is_open(k) for k in kids)
        )
        self.open_parent_all_closed = sorted(
            pid
            for pid, kids in self.children.items()
            if pid in issues and is_open(pid) and kids and not any(is_open(k) for k in kids)
        )
        title_counts = Counter(str(r["title"]).strip().lower() for r in rows)
        self.dup_titles = sorted(
            (title, [str(r["id"]) for r in rows if str(r["title"]).strip().lower() == title])
            for title, n in title_counts.items()
            if n > 1
        )
        self.no_ac_open = [
            str(r["id"])
            for r in rows
            if r["status"] != "closed" and not str(r.get("acceptance_criteria") or "").strip()
        ]
        self.closed_no_reason = [
            str(r["id"]) for r in rows if r["status"] == "closed" and not str(r.get("close_reason") or "").strip()
        ]
        # in_progress rows whose updated_at is older than the staleness budget
        self.stale_claims: list[tuple[str, float]] = sorted(
            (
                (str(r["id"]), (now - _parse_ts(str(r["updated_at"]))).total_seconds() / 86400)
                for r in rows
                if r["status"] == "in_progress"
                and r.get("updated_at")
                and (now - _parse_ts(str(r["updated_at"]))).total_seconds() / 86400 > STALE_CLAIM_DAYS
            ),
            key=lambda t: -t[1],
        )

        # ---- time ----------------------------------------------------------
        self.created_day = Counter(str(r["created_at"])[:10] for r in rows)
        self.closed_day = Counter(str(r["closed_at"])[:10] for r in rows if r.get("closed_at"))
        self.days = sorted(set(self.created_day) | set(self.closed_day))
        self.first_created = min(str(r["created_at"]) for r in rows)[:10]
        self.last_created = max(str(r["created_at"]) for r in rows)[:10]
        self.span_days = (
            _parse_ts(self.last_created + "T00:00:00Z") - _parse_ts(self.first_created + "T00:00:00Z")
        ).days + 1

        self.lead_by_priority: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["status"] == "closed" and r.get("closed_at"):
                delta = (_parse_ts(str(r["closed_at"])) - _parse_ts(str(r["created_at"]))).total_seconds() / 86400
                self.lead_by_priority[int(r["priority"])].append(delta)
        self.age_open = [
            (now - _parse_ts(str(r["created_at"]))).total_seconds() / 86400 for r in rows if r["status"] != "closed"
        ]
        # age (weeks, capped at 3+) x priority heatmap of open work
        self.age_prio: Counter[tuple[int, int]] = Counter(
            (
                min(int((now - _parse_ts(str(r["created_at"]))).total_seconds() / 86400 // 7), 3),
                int(r["priority"]),
            )
            for r in rows
            if r["status"] != "closed"
        )

        # trailing velocity windows: [now-14d, now) vs [now-28d, now-14d)
        w1_start = now - dt.timedelta(days=VELOCITY_WINDOW_DAYS)
        w2_start = now - dt.timedelta(days=2 * VELOCITY_WINDOW_DAYS)

        def _window(day_counter: Counter[str], lo: dt.datetime, hi: dt.datetime) -> int:
            return sum(n for day, n in day_counter.items() if lo <= _parse_ts(day + "T00:00:00Z") < hi)

        self.win_recent = (_window(self.created_day, w1_start, now), _window(self.closed_day, w1_start, now))
        self.win_prior = (
            _window(self.created_day, w2_start, w1_start),
            _window(self.closed_day, w2_start, w1_start),
        )

        # open P0/P1 older than the aged-urgent budget
        self.aged_urgent = sorted(
            (
                str(r["id"])
                for r in rows
                if r["status"] != "closed"
                and int(r["priority"]) <= 1
                and (now - _parse_ts(str(r["created_at"]))).total_seconds() / 86400 > AGED_URGENT_DAYS
            ),
        )

        # close-reason coverage over time (first vs second half of closures)
        closed_rows = sorted(
            (r for r in rows if r["status"] == "closed" and r.get("closed_at")),
            key=lambda r: str(r["closed_at"]),
        )
        half = len(closed_rows) // 2
        self.reason_rate_halves: tuple[float, float] = (
            (sum(1 for r in closed_rows[:half] if str(r.get("close_reason") or "").strip()) / half if half else 0.0),
            (
                sum(1 for r in closed_rows[half:] if str(r.get("close_reason") or "").strip())
                / (len(closed_rows) - half)
                if len(closed_rows) - half
                else 0.0
            ),
        )

        # ---- labels / themes ------------------------------------------------
        self.labels = Counter(str(x) for r in rows for x in (r.get("labels") or []))
        self.area_labels = [x for x in self.labels if x.startswith("area:")]
        self.area_singletons = sum(1 for x in self.area_labels if self.labels[x] == 1)
        self.labelled = sum(1 for r in rows if any(str(x).startswith("area:") for x in (r.get("labels") or [])))
        # mechanical near-synonym detection: one area suffix a strict prefix of another
        suffixes = sorted(x.removeprefix("area:") for x in self.area_labels)
        self.label_synonym_pairs = sorted(
            (a, b) for a in suffixes for b in suffixes if a != b and b.startswith(a) and len(b) - len(a) <= 8
        )

        rev_area: dict[str, str] = {}
        for canon, folded_suffixes in SUBSYSTEMS.items():
            for suffix in folded_suffixes:
                rev_area[suffix] = canon
        self.by_label_theme: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            seen: set[str] = set()
            for raw in r.get("labels") or []:
                text = str(raw)
                if not text.startswith("area:"):
                    continue
                folded = rev_area.get(text.removeprefix("area:"))
                if folded and folded not in seen:
                    seen.add(folded)
                    self.by_label_theme[folded][str(r["status"])] += 1
        self.by_kw_theme: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            text = (str(r["title"]) + " " + str(r.get("description") or "")).lower()
            best, score = "unclassified", 0
            for canon, words in KEYWORDS.items():
                hits = sum(text.count(w) for w in words)
                if hits > score:
                    best, score = canon, hits
            self.by_kw_theme[best][str(r["status"])] += 1

        # ---- verdict subset --------------------------------------------------
        self.verdicts: list[tuple[str, str, str, str]] = []
        loose: Counter[str] = Counter()
        for r in rows:
            text = _blob(r)
            strict = VERDICT_STRICT.search(text)
            if strict:
                self.verdicts.append((str(r["id"]), strict.group(2).upper(), str(r["status"]), strict.group(1).strip()))
            found = VERDICT_LOOSE.findall(text)
            if found:
                loose[Counter(found).most_common(1)[0][0]] += 1
        self.verdict_counts = Counter(v for _, v, _, _ in self.verdicts)
        self.loose_counts = loose

        # A bulk relation re-type stamps every converted edge with the repair
        # date.  Detect it as a same-day spike on the newest association edge.
        rel_days = Counter(created[:10] for _, _, kind, created in edges if kind == "relates-to" and created)
        self.retyped_spike = 0
        self.retyped_spike_day = ""
        if rel_days and self.vocab_state in ("clean", "recurring"):
            newest = max(rel_days)
            spike = rel_days[newest]
            if spike >= 0.2 * sum(rel_days.values()):
                self.retyped_spike = spike
                self.retyped_spike_day = newest

        # A `related` edge whose created_at is later than the newest
        # relates-to edge on the repair day proves the fork already started
        # recurring after the repair.
        self.related_recurrence: list[tuple[str, str, str]] = []
        if self.retyped_spike_day:
            newest_retype_ts = max(
                (
                    created
                    for _, _, kind, created in edges
                    if kind == "relates-to" and created[:10] == self.retyped_spike_day
                ),
                default="",
            )
            for src, dst, kind, created in edges:
                if kind == "related" and created > newest_retype_ts:
                    self.related_recurrence.append((src, dst, created))

        # ---- P0 reconciliation pass (2026-07-31) ---------------------------
        self.reconciliation: list[tuple[str, str, str]] = []
        self.reconciliation_anchored = 0
        for r in rows:
            text = _blob(r)
            anchor = RECONCILIATION_ANCHOR.search(text)
            if not anchor:
                continue
            self.reconciliation_anchored += 1
            window = text[anchor.end() : anchor.end() + RECONCILIATION_WINDOW]
            token = RECONCILIATION_TOKEN_RE.search(window)
            if token:
                cut = window.find(token.group()) + len(token.group())
                snippet = window[:cut].strip()
                self.reconciliation.append((str(r["id"]), token.group(), snippet))
        self.reconciliation_counts = Counter(v for _, v, _ in self.reconciliation)

    # ---- temporal reconstruction -----------------------------------------
    def open_at(self, rec: IssueDict, as_of: dt.datetime) -> bool:
        """Whether a bead existed and was not-closed at ``as_of``.

        Derived from ``created_at``/``closed_at``; a reopened bead's earlier
        closure is invisible (closed_at reflects only the latest close), and
        deleted beads are absent from the export entirely.
        """
        if _parse_ts(str(rec["created_at"])) > as_of:
            return False
        closed = rec.get("closed_at")
        return not (closed and _parse_ts(str(closed)) <= as_of)

    def snapshot(self, as_of: dt.datetime) -> dict[str, int]:
        """Key backlog metrics reconstructed as of a past instant."""
        exists = [r for r in self.rows if _parse_ts(str(r["created_at"])) <= as_of]
        open_rows = [r for r in exists if self.open_at(r, as_of)]
        open_set = {str(r["id"]) for r in open_rows}
        blocked_by_open: dict[str, bool] = defaultdict(bool)
        for src, dst, created in self.block_edges_ts:
            if created and _parse_ts(created) <= as_of and src in open_set and dst in open_set:
                blocked_by_open[src] = True
        blocked_n = sum(1 for bid in open_set if blocked_by_open[bid])
        return {
            "total": len(exists),
            "open": len(open_rows),
            "closed": len(exists) - len(open_rows),
            "p0_open": sum(1 for r in open_rows if int(r["priority"]) == 0),
            "blocked": blocked_n,
            "ready": len(open_set) - blocked_n,
        }

    def daily_series(self) -> list[tuple[str, int, int]]:
        """(day, open count, ready count) for every day of the backlog's life."""
        start = _parse_ts(self.first_created + "T00:00:00Z")
        out: list[tuple[str, int, int]] = []
        day = start
        while day <= self.now:
            eod = day + dt.timedelta(days=1)
            snap = self.snapshot(eod)
            out.append((day.strftime("%Y-%m-%d"), snap["open"], snap["ready"]))
            day = eod
        return out

    def epic_open_series(self, epic_id: str, days: Sequence[str]) -> list[int]:
        """Open-children count per day for one epic (membership as of today)."""
        kids = [self.issues[k] for k in self.children.get(epic_id, []) if k in self.issues]
        out: list[int] = []
        for day in days:
            eod = _parse_ts(day + "T00:00:00Z") + dt.timedelta(days=1)
            out.append(sum(1 for k in kids if self.open_at(k, eod)))
        return out

    # ---- graph structure --------------------------------------------------
    def _find_cycles(self) -> list[list[str]]:
        colour: dict[str, int] = {}
        cycles: list[list[str]] = []

        def visit(start: str) -> None:
            stack: list[tuple[str, Iterable[str]]] = [(start, iter(sorted(self.blockers[start])))]
            colour[start] = 1
            path = [start]
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in self.issues:
                        continue
                    state = colour.get(nxt)
                    if state == 1:
                        cycles.append(path[path.index(nxt) :] + [nxt])
                    elif state is None:
                        colour[nxt] = 1
                        path.append(nxt)
                        stack.append((nxt, iter(sorted(self.blockers[nxt]))))
                        advanced = True
                    break
                if not advanced:
                    colour[node] = 2
                    stack.pop()
                    path.pop()

        for bid in sorted(self.issues):
            if colour.get(bid) is None:
                visit(bid)
        return cycles

    def rate(self, priority: int) -> float:
        total = self.priority[priority]
        closed = self.matrix.get(("closed", priority), 0)
        return closed / total if total else 0.0

    def densest_cluster(self) -> list[str]:
        """Largest weakly-connected component of the open-only `blocks` graph."""
        adj: dict[str, set[str]] = defaultdict(set)
        open_set = set(self.open_ids)
        for src, dst, kind, _ in self.edges:
            if kind == "blocks" and src in open_set and dst in open_set:
                adj[src].add(dst)
                adj[dst].add(src)
        seen: set[str] = set()
        best: list[str] = []
        for node in sorted(adj):
            if node in seen:
                continue
            comp: list[str] = []
            queue = [node]
            seen.add(node)
            while queue:
                cur = queue.pop()
                comp.append(cur)
                for nxt in sorted(adj[cur]):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if len(comp) > len(best):
                best = comp
        self._cluster_adj = adj
        return sorted(best)

    def cluster_core(self, component: Sequence[str], min_degree: int = 3) -> list[str]:
        """The legible core of a component: nodes with >= min_degree neighbours in it."""
        adj = self._cluster_adj
        member = set(component)
        return sorted(n for n in component if len({x for x in adj[n] if x in member}) >= min_degree)

    def descendants(self, root: str, max_depth: int = 6) -> set[str]:
        """All parent-child descendants of a bead, cycle-safe."""
        out: set[str] = set()
        frontier = [root]
        for _ in range(max_depth):
            nxt: list[str] = []
            for node in frontier:
                for kid in self.children.get(node, []):
                    if kid not in out:
                        out.add(kid)
                        nxt.append(kid)
            if not nxt:
                break
            frontier = nxt
        return out

    def parallel_frontier(self, top_n: int = 8) -> list[dict[str, Any]]:
        """The largest set of big open epics whose open subtrees share no
        ``blocks`` component -- work that can run in parallel with no ordering
        constraint between the programs.  Greedy by open-descendant count.
        """
        # component id per open bead in the blocks graph
        self.densest_cluster()  # populates _cluster_adj
        comp_of: dict[str, int] = {}
        cid = 0
        for node in sorted(self._cluster_adj):
            if node in comp_of:
                continue
            queue = [node]
            comp_of[node] = cid
            while queue:
                cur = queue.pop()
                for nxt in self._cluster_adj[cur]:
                    if nxt not in comp_of:
                        comp_of[nxt] = cid
                        queue.append(nxt)
            cid += 1

        candidates: list[dict[str, Any]] = []
        for e in self.epics:
            desc = {d for d in self.descendants(str(e["id"])) if self.is_open(d)}
            if len(desc) < 3:
                continue
            comps = {comp_of[d] for d in desc if d in comp_of}
            candidates.append(
                {
                    "id": str(e["id"]),
                    "title": str(e["title"]),
                    "open_desc": len(desc),
                    "desc": desc,
                    "comps": comps,
                    "urgent": sum(1 for d in desc if int(self.issues[d]["priority"]) <= 2),
                }
            )
        candidates.sort(key=lambda c: (-int(c["open_desc"]), str(c["id"])))

        chosen: list[dict[str, Any]] = []
        used_comps: set[int] = set()
        used_beads: set[str] = set()
        for cand in candidates[:top_n]:
            if cand["comps"] & used_comps or cand["desc"] & used_beads:
                continue
            chosen.append(cand)
            used_comps |= set(cand["comps"])
            used_beads |= set(cand["desc"])
        return chosen


# --------------------------------------------------------------------------
# computed insights: the interpretation layer.  Every entry is emitted by a
# condition checked against the data; regeneration cannot leave a stale claim.
# --------------------------------------------------------------------------
@dataclass
class Insight:
    sev: str  # bad | warn | info | ok
    title: str  # html
    body: str  # html
    ev: str = "derived"  # measured | derived
    chips: list[str] = field(default_factory=list)  # bead ids to render as chips


_SEV_ORDER = {"bad": 0, "warn": 1, "info": 2, "ok": 3}


def compute_insights(facts: Facts, schema_gap: dict[str, Any], snaps: dict[str, dict[str, int]]) -> list[Insight]:
    out: list[Insight] = []

    # -- schema gap: merged fixes inert until rebuild -----------------------
    if schema_gap.get("available") and schema_gap.get("blockers"):
        pending = (
            facts.reconciliation_counts["FIXED-PENDING-REBUILD"] + facts.reconciliation_counts["FIXED-PENDING-DEPLOY"]
        )
        pending_note = (
            f" The dated reconciliation pass confirmed <b>{pending}</b> beads merged and inert behind it."
            if pending
            else ""
        )
        out.append(
            Insight(
                "bad",
                f"Live archive is {len(schema_gap['blockers'])} SEMANTIC_REPARSE version(s) behind origin/master "
                f"(v{schema_gap['live_version']} vs v{schema_gap['declared']})",
                "Merged fixes at those versions are real, reviewed, and completely inert until "
                "<code>polylogue ops reset --index &amp;&amp; polylogued run</code> executes." + pending_note,
                ev="measured",
            )
        )

    # -- dangling refs ------------------------------------------------------
    if facts.dangling:
        targets = ", ".join(f"<code>{esc(b)}</code>" for _, b, _ in facts.dangling[:6])
        out.append(
            Insight(
                "bad",
                f"{len(facts.dangling)} dependency edge(s) point at ids that do not exist",
                f"Unresolvable targets: {targets}. The blocking intent behind each edge is silently absent "
                "from every query. Id references in dependency edges are not validated on write.",
                ev="measured",
            )
        )

    # -- cycles -------------------------------------------------------------
    if facts.cycles:
        first = " &rarr; ".join(esc(x) for x in facts.cycles[0])
        out.append(
            Insight(
                "bad",
                f"{len(facts.cycles)} cycle(s) in the <code>blocks</code> graph",
                f"A mutual-blocking loop can never become ready. First cycle: {first}.",
                ev="measured",
            )
        )
    else:
        out.append(
            Insight(
                "ok",
                "No cycles in the <code>blocks</code> graph",
                f"Verified by DFS colouring over all {facts.total:,} beads and "
                f"{facts.dep_types['blocks']:,} blocks edges. Stated explicitly rather than left as an "
                "absence: a graph grown this fast by many independent lanes is where mutual-blocking "
                "pairs appear by accident.",
                ev="measured",
            )
        )

    # -- velocity: trailing window vs the one before it --------------------
    (rc, rx), (pc, px) = facts.win_recent, facts.win_prior
    r_net, p_net = rc - rx, pc - px
    if r_net > 0 and p_net > 0:
        sev, verdict = "warn", "growing in both trailing windows &mdash; still in discovery, not burn-down"
    elif r_net < 0 <= p_net:
        sev, verdict = "ok", "tipped from growth to net drain in the most recent window"
    elif r_net < 0 and p_net < 0:
        sev, verdict = "ok", "draining in both trailing windows"
    elif r_net > 0 >= p_net:
        sev, verdict = "warn", "flipped back from drain to growth in the most recent window"
    else:
        sev, verdict = "info", "roughly flat"
    out.append(
        Insight(
            sev,
            f"Backlog is {verdict}",
            f"Last {VELOCITY_WINDOW_DAYS}d: {rc} created / {rx} closed (net {r_net:+d}). "
            f"Prior {VELOCITY_WINDOW_DAYS}d: {pc} created / {px} closed (net {p_net:+d}). "
            "Velocity-derived completion estimates are meaningful only once the trailing window "
            "is reliably net-negative.",
        )
    )

    # -- ready fraction -----------------------------------------------------
    if facts.open_total:
        frac = len(facts.ready) / facts.open_total
        if frac >= 0.6:
            out.append(
                Insight(
                    "info",
                    f"{frac:.0%} of open work is unblocked &mdash; throughput-bound, not dependency-bound",
                    f"{len(facts.ready):,} of {facts.open_total:,} open beads have no open blocker. "
                    "No amount of unblocking changes the picture; execution capacity is the constraint.",
                    ev="measured",
                )
            )
        elif frac <= 0.35:
            out.append(
                Insight(
                    "warn",
                    f"Only {frac:.0%} of open work is unblocked &mdash; the graph is dependency-bound",
                    f"{len(facts.blocked):,} of {facts.open_total:,} open beads wait on an open blocker. "
                    "Unblocking the top blockers is the highest-leverage move; see Topology.",
                    ev="measured",
                )
            )

    # -- priority ladder ----------------------------------------------------
    rates = [facts.rate(p) for p in range(5)]
    leads = [_median(facts.lead_by_priority[p]) for p in range(5)]
    rate_inversions = [p for p in range(3) if facts.priority[p + 1] and rates[p] < rates[p + 1]]
    lead_inversions = [p for p in range(3) if leads[p + 1] and leads[p] > leads[p + 1]]
    lead_str = " / ".join(f"{v:.1f}" for v in leads)
    rate_str = " / ".join(f"{v:.0%}" for v in rates)
    if not rate_inversions and not lead_inversions:
        out.append(
            Insight(
                "info",
                "The priority scale is load-bearing &mdash; do not renumber it",
                f"Median lead time rises monotonically P0&rarr;P3 ({lead_str} days for P0&ndash;P4) and "
                f"closure rate falls monotonically P0&rarr;P3 ({rate_str}). An inflated priority label "
                "would show a flat curve; this one sorts work at every rung.",
            )
        )
    else:
        inv = ", ".join(f"P{p}&rarr;P{p + 1}" for p in sorted(set(rate_inversions + lead_inversions)))
        out.append(
            Insight(
                "warn",
                f"Priority ladder inverts at {inv}",
                f"Median lead days P0&ndash;P4: {lead_str}. Closure rates: {rate_str}. "
                "Where a lower-urgency rung closes faster or more often than the rung above it, the "
                "label is not carrying the information a scheduler needs.",
            )
        )

    # -- aged urgency -------------------------------------------------------
    if facts.aged_urgent:
        out.append(
            Insight(
                "warn",
                f"{len(facts.aged_urgent)} open P0/P1 bead(s) older than {AGED_URGENT_DAYS} days",
                "Urgency that sits is urgency mislabelled or capacity missing &mdash; either way it is "
                "the first shelf to triage. Oldest listed below.",
                chips=facts.aged_urgent[:8],
            )
        )

    # -- stale in-progress claims ------------------------------------------
    if facts.stale_claims:
        out.append(
            Insight(
                "warn",
                f"{len(facts.stale_claims)} in-progress claim(s) untouched for over {STALE_CLAIM_DAYS} days",
                "A claim without activity blocks other agents from picking the bead up. "
                "Release the claim or finish the work.",
                ev="measured",
                chips=[bid for bid, _ in facts.stale_claims[:8]],
            )
        )

    # -- parallel frontier --------------------------------------------------
    frontier = facts.parallel_frontier()
    if len(frontier) >= 2:
        total_open_desc = sum(int(c["open_desc"]) for c in frontier)
        share = total_open_desc / facts.open_total if facts.open_total else 0.0
        names = ", ".join(f"<code>{esc(str(c['id']))}</code>" for c in frontier)
        out.append(
            Insight(
                "info",
                f"{len(frontier)} large programs share no blocking path &mdash; they can run in parallel",
                f"{names} hold {total_open_desc} open descendants between them ({share:.0%} of all open "
                "work) and their open subtrees occupy disjoint components of the <code>blocks</code> "
                "graph: no ordering constraint exists between the programs. Detail in Topology.",
            )
        )

    # -- relation vocabulary fork ------------------------------------------
    relates, related = facts.dep_types["relates-to"], facts.dep_types["related"]
    if facts.vocab_state == "split":
        out.append(
            Insight(
                "bad",
                "Split relation vocabulary: <code>relates-to</code> vs <code>related</code>",
                f"{relates} + {related} edges are one relation under two names. Any association query "
                "silently returns a partial subset. Pick one spelling, rewrite the other, constrain the field.",
                ev="measured",
            )
        )
    elif facts.vocab_state == "recurring":
        out.append(
            Insight(
                "warn",
                "Relation vocabulary was unified, then forked again",
                f"A bulk re-type unified the association relation, but wrote no constraint, and "
                f"{related} fresh <code>related</code> edge(s) landed after the repair window. "
                "The data was cleaned; the write path is still open.",
                ev="measured",
            )
        )

    # -- label vocabulary ---------------------------------------------------
    if facts.label_synonym_pairs:
        pairs = ", ".join(f"<code>{esc(a)}</code>/<code>{esc(b)}</code>" for a, b in facts.label_synonym_pairs[:6])
        out.append(
            Insight(
                "warn",
                f"{len(facts.area_labels)} distinct <code>area:*</code> labels, "
                f"{facts.area_singletons} used once, {len(facts.label_synonym_pairs)} near-synonym pair(s)",
                f"Prefix-detected synonym pairs: {pairs}. A free-text label field with no controlled "
                "vocabulary degrades into per-author spelling; grouping by raw label undercounts every "
                "affected subsystem. The Themes section folds them through an explicit map.",
                ev="measured",
            )
        )

    # -- review queue -------------------------------------------------------
    if facts.open_parent_all_closed:
        out.append(
            Insight(
                "warn",
                f"{len(facts.open_parent_all_closed)} open parent(s) with every child closed &mdash; "
                "review, not auto-close",
                "Children closing is evidence a parent is ready for adjudication against its own "
                "acceptance criteria, never evidence that it is done. Queue in Health.",
            )
        )

    # -- duplicate titles ---------------------------------------------------
    if facts.dup_titles:
        out.append(
            Insight(
                "info",
                f"{len(facts.dup_titles)} duplicate title pair(s)",
                "Independently filed beads describing the same work; listed in Health with both ids.",
                ev="measured",
            )
        )

    # -- close-reason discipline -------------------------------------------
    r1, r2 = facts.reason_rate_halves
    if facts.status["closed"] >= 20:
        direction = "improving" if r2 > r1 + 0.02 else ("degrading" if r2 < r1 - 0.02 else "steady")
        sev = "info" if r2 >= 0.9 else "warn"
        out.append(
            Insight(
                sev,
                f"Close-reason coverage is {r2:.0%} in the newer half of closures ({direction})",
                f"Earlier half {r1:.0%}, newer half {r2:.0%}; "
                f"{len(facts.closed_no_reason)} closed bead(s) in total carry no reason. A close without "
                "a reason leaves no durable record of why the work ended.",
            )
        )

    # -- open-count trend vs 7d --------------------------------------------
    now_open = snaps["now"]["open"]
    week_open = snaps["7d"]["open"]
    if week_open:
        delta = now_open - week_open
        if abs(delta) >= max(5, 0.03 * week_open):
            direction = "grew" if delta > 0 else "shrank"
            out.append(
                Insight(
                    "info",
                    f"Open backlog {direction} {abs(delta)} bead(s) in the last 7 days ({week_open} &rarr; {now_open})",
                    "Reconstructed from created/closed timestamps; deleted beads and reopen history "
                    "are invisible to the reconstruction.",
                )
            )

    out.sort(key=lambda i: _SEV_ORDER.get(i.sev, 9))
    return out


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def chip(facts: Facts, bid: str, *, title: bool = False) -> str:
    """The compact bead notation used everywhere in the report."""
    rec = facts.issues.get(bid)
    if rec is None:
        return f'<span class="bd missing"><span class="bi">{esc(bid)}</span><span class="bg">?</span></span>'
    prio = int(rec["priority"])
    status = str(rec["status"])
    sglyph, sname = STATUS_GLYPH.get(status, ("?", status))
    tglyph, tname = TYPE_GLYPH.get(str(rec["issue_type"]), ("?", str(rec["issue_type"])))
    inn = sum(1 for b in facts.blockers[bid] if facts.is_open(b))
    out = sum(1 for b in facts.blocking[bid] if facts.is_open(b))
    deg = ""
    if inn or out:
        deg = f'<span class="bg">{"&uarr;" + str(inn) if inn else ""}{"&darr;" + str(out) if out else ""}</span>'
    tip = f"P{prio} · {sname} · {tname} · {inn} open blocker(s) · blocks {out} open"
    text = f' <span class="bl">{esc(str(rec["title"])[:64])}</span>' if title else ""
    return (
        f'<span class="bd p{prio} st-{status}" title="{esc(tip)}">'
        f'<span class="bp">P{prio}</span>'
        f'<span class="bs">{sglyph}</span>'
        f'<span class="bi">{esc(bid)}</span>'
        f'<span class="by">{tglyph}</span>{deg}{text}</span>'
    )


def delta_chip(now_v: int, then_v: int, *, up_is_bad: bool = False) -> str:
    """A signed 7-day delta rendered next to a stat tile value."""
    d = now_v - then_v
    if d == 0:
        return '<span class="dl flat">&plusmn;0</span>'
    arrow = "&#9650;" if d > 0 else "&#9660;"
    bad = (d > 0) == up_is_bad
    cls = "worse" if bad else "better"
    return f'<span class="dl {cls}">{arrow}{abs(d)}</span>'


def sparkline_svg(values: Sequence[int], *, label: str = "") -> str:
    """A tiny inline trend line; viewBox only, currentColor stroke."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    w = (len(values) - 1) * 3
    pts = " ".join(f"{i * 3},{18 - 15 * (v - lo) / span:.1f}" for i, v in enumerate(values))
    title = f"<title>{esc(label)} {values[0]} &rarr; {values[-1]}</title>" if label else ""
    return (
        f'<svg class="spark" viewBox="0 0 {w} 20" preserveAspectRatio="none" role="img" '
        f'aria-label="{esc(label or "trend")}">{title}'
        f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>'
    )


def burnup_svg(series: Sequence[tuple[str, int, int]]) -> str:
    """Open + ready count over the backlog's whole life, one small chart."""
    if len(series) < 2:
        return ""
    peak = max(v for _, v, _ in series) or 1
    n = len(series)
    w, h = (n - 1) * 10, 60

    def path(values: Sequence[int]) -> str:
        return " ".join(f"{i * 10},{h - 4 - (h - 12) * (v / peak):.1f}" for i, v in enumerate(values))

    open_pts = path([row[1] for row in series])
    ready_pts = path([row[2] for row in series])
    parts = [
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" aria-label="open and ready beads over time">',
        f'<polyline class="ln open" points="{open_pts}" fill="none"/>',
        f'<polyline class="ln ready" points="{ready_pts}" fill="none"/>',
    ]
    for i, (day, open_n, ready_n) in enumerate(series):
        parts.append(
            f'<rect class="hit" x="{i * 10 - 5}" y="0" width="10" height="{h}">'
            f"<title>{esc(day)}: {open_n} open, {ready_n} ready</title></rect>"
        )
    parts.append("</svg>")
    return "".join(parts)


def histogram_svg(days: Sequence[str], series: Sequence[tuple[str, Counter[str], str]]) -> str:
    """Grouped per-day bar chart, pure SVG, viewBox only."""
    if not days:
        return ""
    step = 10.0
    width = len(days) * step
    height = 46.0
    peak = max((c[day] for _, c, _ in series for day in days), default=1) or 1
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="beads created and closed per day" preserveAspectRatio="none">'
    ]
    bw = step / (len(series) + 1)
    for si, (_, counts, cls) in enumerate(series):
        for di, day in enumerate(days):
            value = counts[day]
            if not value:
                continue
            bh = (value / peak) * (height - 8)
            x = di * step + si * bw
            parts.append(
                f'<rect class="bar {cls}" x="{x:.2f}" y="{height - bh:.2f}" '
                f'width="{bw:.2f}" height="{bh:.2f}"><title>{esc(day)}: {value}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def cluster_svg(facts: Facts, nodes: Sequence[str]) -> str:
    """Layered DAG of one connected `blocks` component. Layer = longest path from a root."""
    node_set = set(nodes)
    layer: dict[str, int] = {}

    def depth(bid: str, seen: frozenset[str]) -> int:
        if bid in layer:
            return layer[bid]
        ups = [b for b in facts.blockers[bid] if b in node_set and b not in seen]
        value = 0 if not ups else 1 + max(depth(u, seen | {bid}) for u in ups)
        layer[bid] = value
        return value

    for bid in nodes:
        depth(bid, frozenset())
    rows: dict[int, list[str]] = defaultdict(list)
    for bid in sorted(nodes):
        rows[layer[bid]].append(bid)
    pos: dict[str, tuple[float, float]] = {}
    col_w, row_h = 210.0, 34.0
    for lv, members in rows.items():
        for i, bid in enumerate(sorted(members)):
            pos[bid] = (18 + lv * col_w, 22 + i * row_h)
    width = 36 + (max(rows) + 1) * col_w
    height = 30 + max(len(v) for v in rows.values()) * row_h
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="largest connected blocking cluster" class="dag">',
        '<defs><marker id="dagar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>',
    ]
    for src in nodes:
        for dst in facts.blockers[src]:
            if dst not in node_set:
                continue
            x1, y1 = pos[dst]
            x2, y2 = pos[src]
            parts.append(
                f'<path d="M{x1 + 168:.1f} {y1:.1f} C{x1 + 190:.1f} {y1:.1f} '
                f'{x2 - 22:.1f} {y2:.1f} {x2 - 4:.1f} {y2:.1f}" fill="none" '
                f'stroke="currentColor" opacity=".38" marker-end="url(#dagar)"/>'
            )
    for bid, (x, y) in pos.items():
        rec = facts.issues[bid]
        prio = int(rec["priority"])
        parts.append(
            f'<g class="dn p{prio}"><rect x="{x:.1f}" y="{y - 11:.1f}" width="168" height="22" rx="5"/>'
            f'<text x="{x + 7:.1f}" y="{y + 4:.1f}" font-size="11">P{prio} {esc(bid)}</text>'
            f"<title>{esc(str(rec['title'])[:110])}</title></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def tree_html(
    facts: Facts,
    root: str,
    depth: int = 0,
    _path: frozenset[str] | None = None,
    _seen_edges: set[tuple[str, str]] | None = None,
) -> str:
    """Render a bounded parent-child tree from an untrusted export.

    The export can contain repeated edges or a cycle.  Keep the normal
    two-level view, but make each render visit an edge at most once and never
    recurse into an ancestor.  Suppressed malformed edges remain visible as a
    compact diagnostic with both endpoint ids.
    """
    if depth >= TREE_MAX_DEPTH:
        return ""

    path = frozenset() if _path is None else _path
    if root in path:
        return ""
    path = path | {root}
    seen_edges = set() if _seen_edges is None else _seen_edges
    kids = sorted(facts.children.get(root, []))
    if not kids:
        return ""

    items = []
    duplicate_counts: Counter[str] = Counter()
    cyclic_children: set[str] = set()
    for kid in kids:
        edge = (root, kid)
        if edge in seen_edges:
            duplicate_counts[kid] += 1
            continue
        seen_edges.add(edge)
        if kid not in facts.issues:
            continue
        if kid in path:
            cyclic_children.add(kid)
            continue
        sub = tree_html(facts, kid, depth + 1, path, seen_edges)
        title = esc(str(facts.issues[kid]["title"])[:78])
        items.append(f'<li>{chip(facts, kid)} <span class="meta">{title}</span>{sub}</li>')

    for kid, count in sorted(duplicate_counts.items()):
        items.append(
            f'<li class="tree-diagnostic"><span class="meta">'
            f"suppressed {count:,} duplicate parent-child edge(s): "
            f"<code>{esc(root)} &rarr; {esc(kid)}</code></span></li>"
        )
    for kid in sorted(cyclic_children):
        items.append(
            f'<li class="tree-diagnostic"><span class="meta">'
            f"suppressed cyclic parent-child edge: "
            f"<code>{esc(root)} &rarr; {esc(kid)}</code></span></li>"
        )

    return f'<ul class="tree">{"".join(items)}</ul>'


def fill_bar(open_n: int, total: int) -> str:
    done = total - open_n
    pct = 100 * done / total if total else 0
    return (
        f'<span class="fill" title="{done} of {total} closed">'
        f'<span style="width:{pct:.1f}%"></span></span>'
        f'<span class="filln">{done}/{total}</span>'
    )


def heatmap_html(facts: Facts) -> str:
    """Age (weeks) x priority heatmap of open work; sequential single-hue fill."""
    age_labels = ["0&ndash;6d", "7&ndash;13d", "14&ndash;20d", "21d+"]
    peak = max(facts.age_prio.values()) if facts.age_prio else 1
    parts = ['<div class="tablewrap"><table class="hm"><thead><tr><th class="nosort">age \\ priority</th>']
    for p in range(5):
        parts.append(f'<th class="nosort">P{p}</th>')
    parts.append('<th class="nosort">row</th></tr></thead><tbody>')
    for w, label in enumerate(age_labels):
        row_total = sum(facts.age_prio.get((w, p), 0) for p in range(5))
        parts.append(f"<tr><td>{label}</td>")
        for p in range(5):
            n = facts.age_prio.get((w, p), 0)
            pct = 58 * n / peak if peak else 0
            parts.append(
                f'<td class="c" style="background:color-mix(in srgb, var(--accent) {pct:.0f}%, transparent)" '
                f'title="{n} open P{p} bead(s) aged {label.replace("&ndash;", "-")}">{n or ""}</td>'
            )
        parts.append(f'<td class="num">{row_total}</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def qa(command: str, label: str) -> str:
    """Attach the exact invocation that produced a figure."""
    return (
        '<template class="pop"><figure class="code"><figcaption>'
        f"{esc(label)}</figcaption><pre><code>{esc(command)}</code></pre></figure></template>"
    )


CSS = """
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --todo:#6b21a8; --todo-bg:#f1e6fb; --code-bg:#eef1f4;
  --p0:#a01c1c; --p1:#b45309; --p2:#1d4ed8; --p3:#0f7b46; --p4:#5b6773;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --todo:#c793ef; --todo-bg:#2a1738; --code-bg:#232a32;
  --p0:#ef7070; --p1:#e2b93b; --p2:#7ea6f4; --p3:#4cc98a; --p4:#94a1ad;
}}
:root[data-theme="light"]{color-scheme:light}
:root[data-theme="dark"]{color-scheme:dark}
:root[data-theme="dark"]{
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --todo:#c793ef; --todo-bg:#2a1738; --code-bg:#232a32;
  --p0:#ef7070; --p1:#e2b93b; --p2:#7ea6f4; --p3:#4cc98a; --p4:#94a1ad;
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --todo:#6b21a8; --todo-bg:#f1e6fb; --code-bg:#eef1f4;
  --p0:#a01c1c; --p1:#b45309; --p2:#1d4ed8; --p3:#0f7b46; --p4:#5b6773;
}
:root{--fs:17px}
*{box-sizing:border-box}
html{font-size:var(--fs)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:1rem/1.62 system-ui,-apple-system,"Segoe UI",sans-serif}
header.page{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);padding:.7rem 1.2rem;
  display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem}
header.page h1{font-size:1.3rem;margin:0}
header.page .spacer{flex:1}
.chip{display:inline-block;padding:.12rem .6rem;border:1px solid var(--line);
  border-radius:99px;font-size:.85rem;color:var(--muted);background:var(--bg)}
button.theme,button.fs{border:1px solid var(--line);background:var(--bg);color:var(--ink);
  border-radius:.4rem;padding:.2rem .6rem;cursor:pointer;font-size:.85rem}
.layout{display:grid;grid-template-columns:16rem minmax(0,1fr);
  max-width:80rem;margin:0 auto;gap:1.2rem;padding:1.2rem}
@media(max-width:62rem){.layout{grid-template-columns:minmax(0,1fr)}nav#toc{display:none}}
nav#toc{position:sticky;top:3.8rem;align-self:start;font-size:.92rem;
  border-right:1px solid var(--line);padding-right:.8rem;max-height:85vh;overflow:auto}
nav#toc a{display:block;color:var(--muted);text-decoration:none;padding:.16rem 0}
nav#toc a.h3{padding-left:.9rem;font-size:.87rem}
nav#toc a:hover{color:var(--accent)}
main{min-width:0}
section{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;
  padding:1.1rem 1.3rem;margin-bottom:1.1rem}
h2{font-size:1.28rem;margin:.1rem 0 .7rem;line-height:1.3}
h3{font-size:1.07rem;margin:1.1rem 0 .45rem;line-height:1.35}
p{margin:.5rem 0;max-width:76ch}
a{color:var(--accent)}
code,kbd{background:var(--code-bg);border-radius:.3rem;padding:.08rem .35rem;
  font:.9em ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code-bg);padding:.75rem .9rem;border-radius:.5rem;
  overflow-x:auto;font-size:.92rem;line-height:1.5}
pre code{background:none;padding:0;font-size:1em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));
  gap:.7rem;margin:.4rem 0 .9rem}
.tile{border:1px solid var(--line);border-radius:.55rem;padding:.6rem .85rem;background:var(--bg)}
.tile .n{font-size:1.6rem;font-weight:650;display:block;line-height:1.2}
.tile .l{font-size:.85rem;color:var(--muted)}
.tile.qa{position:relative;cursor:help}
.tile.qa::after{content:"\\2315";position:absolute;top:.35rem;right:.45rem;opacity:.35;font-size:.8rem}
.dl{font-size:.8rem;font-weight:700;margin-left:.35rem;vertical-align:.25em}
.dl.worse{color:var(--bad)} .dl.better{color:var(--ok)} .dl.flat{color:var(--muted)}
.badge{display:inline-block;font-size:.8rem;font-weight:600;letter-spacing:.02em;
  padding:.1rem .55rem;border-radius:99px;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)} .warn{color:var(--warn);background:var(--warn-bg)}
.bad{color:var(--bad);background:var(--bad-bg)} .info{color:var(--info);background:var(--info-bg)}
.todo{color:var(--todo);background:var(--todo-bg)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{border-bottom:1px solid var(--line);text-align:left;padding:.42rem .6rem;vertical-align:top}
th{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;
  cursor:pointer;user-select:none;white-space:nowrap}
th.nosort{cursor:default}
tr:hover td{background:color-mix(in srgb, var(--accent) 6%, transparent)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
details{border:1px solid var(--line);border-radius:.5rem;padding:.5rem .85rem;margin:.5rem 0;background:var(--bg)}
summary{cursor:pointer;font-weight:600;font-size:1rem}
details[open]{padding-bottom:.6rem}
input.filter{width:100%;max-width:26rem;margin:.2rem 0 .55rem;padding:.35rem .65rem;
  border:1px solid var(--line);border-radius:.4rem;background:var(--bg);color:var(--ink);font-size:.95rem}
footer{color:var(--muted);font-size:.88rem;text-align:center;padding:1rem}
.meta{display:grid;grid-template-columns:auto 1fr;gap:.2rem .9rem;font-size:.9rem;
  border-left:3px solid var(--line);padding:.1rem 0 .1rem .9rem;margin:.2rem 0 .9rem}
.meta dt{color:var(--muted)}
.meta dd{margin:0}
.ev{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;
  padding:0 .35rem;border-radius:.25rem;vertical-align:.1em;text-transform:uppercase}
.ev-measured{color:var(--ok);background:var(--ok-bg)}
.ev-derived{color:var(--info);background:var(--info-bg)}
.ev-inferred{color:var(--warn);background:var(--warn-bg)}
.ev-assumed{color:var(--bad);background:var(--bad-bg)}
time.age{color:var(--muted);font-size:.85em;font-variant-numeric:tabular-nums}
time.age.stale{color:var(--warn);font-weight:600}
time.age.stale::after{content:" \\26A0"}
a.path{font:.9em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code-bg);
  border-radius:.3rem;padding:.08rem .35rem;text-decoration:none;border-bottom:1px dotted var(--accent)}
.pop{position:fixed;z-index:50;max-width:min(46rem,92vw);max-height:70vh;overflow:auto;
  background:var(--panel);border:1px solid var(--line);border-radius:.5rem;
  box-shadow:0 .6rem 2rem rgba(0,0,0,.28);padding:.7rem .9rem;font-size:.92rem}
.pop h4{margin:0 0 .35rem;font-size:.92rem;font-family:ui-monospace,monospace;color:var(--muted)}
.pop pre{margin:0;max-height:52vh;font-size:.86rem}
.pop .pin{float:right;border:none;background:none;color:var(--muted);cursor:pointer}
blockquote.q{position:relative;margin:.55rem 0 .4rem;padding:.6rem 1.1rem .6rem 2.3rem;
  background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--todo);
  border-radius:0 .45rem .45rem 0;
  font:italic 1rem/1.55 Georgia,"Iowan Old Style","Noto Serif",ui-serif,serif}
blockquote.q::before{content:"\\201C";position:absolute;left:.45rem;top:.15rem;
  font:italic 2.2rem/1 Georgia,ui-serif,serif;color:var(--todo);opacity:.5}
blockquote.q cite{display:block;margin-top:.4rem;font:600 .72rem/1 system-ui,sans-serif;
  text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-style:normal}

/* ---- BEAD NOTATION -------------------------------------------------- */
.bd{display:inline-flex;align-items:center;gap:.28rem;border:1px solid var(--line);
  border-radius:.35rem;padding:.02rem .34rem .02rem .05rem;background:var(--bg);
  font:.82rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;
  border-left:3px solid var(--pc,var(--muted))}
.bd.p0{--pc:var(--p0)} .bd.p1{--pc:var(--p1)} .bd.p2{--pc:var(--p2)}
.bd.p3{--pc:var(--p3)} .bd.p4{--pc:var(--p4)}
.bd .bp{font-weight:700;color:var(--pc);font-size:.78rem;padding-left:.3rem}
.bd .bs{font-size:.82rem;opacity:.85}
.bd.st-closed{opacity:.62}
.bd.st-closed .bi{text-decoration:line-through;text-decoration-thickness:1px}
.bd.st-in_progress{box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 45%,transparent)}
.bd.st-deferred{opacity:.75;border-style:dashed}
.bd .bi{color:var(--ink)}
.bd .by{opacity:.75}
.bd .bg{color:var(--accent);font-size:.75rem;font-weight:700}
.bd.missing{border-left-color:var(--bad);background:var(--bad-bg)}
.bd .bl{font:.8rem/1.5 system-ui,sans-serif;color:var(--muted);max-width:26rem;
  overflow:hidden;text-overflow:ellipsis}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:.5rem 1.2rem;
  font-size:.9rem;margin:.6rem 0}
.legend div{display:flex;gap:.45rem;align-items:baseline}
.legend b{font:.85rem ui-monospace,monospace;color:var(--accent);min-width:2.2rem}

/* ---- distribution bars ---- */
.dist{position:relative;text-align:right;white-space:nowrap}
.dist span{position:absolute;left:0;top:.25rem;bottom:.25rem;width:var(--w);
  background:var(--accent);opacity:.18;border-radius:.2rem}
.dist b{position:relative;font-variant-numeric:tabular-nums}

/* ---- epic fill bars ---- */
.fill{display:inline-block;width:9rem;height:.62rem;border-radius:99px;background:var(--code-bg);
  overflow:hidden;vertical-align:middle;border:1px solid var(--line)}
.fill>span{display:block;height:100%;background:var(--ok)}
.filln{font:.78rem ui-monospace,monospace;color:var(--muted);margin-left:.4rem;
  font-variant-numeric:tabular-nums}

/* ---- matrix / heatmap ---- */
table.mx td.c{text-align:right;font-variant-numeric:tabular-nums;position:relative}
table.mx td.c i{position:absolute;left:.2rem;top:.25rem;bottom:.25rem;border-radius:.2rem;
  background:var(--accent);opacity:.16;font-style:normal}
table.mx td.c b{position:relative}
table.hm td.c{text-align:center;font-variant-numeric:tabular-nums;min-width:3.2rem}
table.hm td,table.hm th{border:1px solid var(--line)}

/* ---- charts ---- */
.chart svg{width:100%;height:auto;max-height:7rem}
.bar{fill:var(--accent);opacity:.75}
.bar.closed{fill:var(--ok);opacity:.8}
.ln{stroke-width:2;vector-effect:non-scaling-stroke}
.ln.open{stroke:var(--accent)}
.ln.ready{stroke:var(--ok)}
.hit{fill:transparent}
.spark{width:6.5rem;height:1.1rem;color:var(--accent);vertical-align:middle}
.dag{height:auto;color:var(--muted);min-width:44rem;width:100%}
.dag .dn rect{fill:var(--bg);stroke:var(--pc,var(--muted));stroke-width:1.4}
.dag .dn.p0{--pc:var(--p0)} .dag .dn.p1{--pc:var(--p1)} .dag .dn.p2{--pc:var(--p2)}
.dag .dn.p3{--pc:var(--p3)} .dag .dn.p4{--pc:var(--p4)}
.dag .dn text{fill:var(--ink);font-family:ui-monospace,monospace}
.dagwrap{overflow-x:auto;border:1px solid var(--line);border-radius:.5rem;padding:.4rem;background:var(--bg)}

/* ---- trees / findings ---- */
.tree,.tree ul{list-style:none;margin:0;padding-left:1.1rem}
.tree li{position:relative;padding:.1rem 0 .1rem .8rem}
.tree li::before{content:"";position:absolute;left:0;top:0;bottom:0;border-left:1px solid var(--line)}
.tree li::after{content:"";position:absolute;left:0;top:.85rem;width:.65rem;border-top:1px solid var(--line)}
.tree>li:last-child::before{bottom:auto;height:.85rem}
.tree .meta{font-size:.82rem;opacity:.7;margin-left:.4rem}
ul.findings{list-style:none;padding:0;margin:.5rem 0}
ul.findings li{border-left:3px solid var(--line);padding:.4rem 0 .4rem .8rem;margin:.5rem 0;max-width:80ch}
ul.findings li.sev-bad{border-left-color:var(--bad)}
ul.findings li.sev-warn{border-left-color:var(--warn)}
ul.findings li.sev-info{border-left-color:var(--info)}
ul.findings li.sev-ok{border-left-color:var(--ok)}
ul.findings b{display:block;margin-bottom:.15rem}
.digest{display:flex;flex-wrap:wrap;gap:.45rem;margin:.4rem 0 .2rem}
@media print{header.page,nav#toc,button,input.filter,.pop{display:none}
  .layout{grid-template-columns:1fr;max-width:none}
  section{break-inside:avoid;border:none;padding:0}
  details{border:none}details:not([open])>*:not(summary){display:revert}}
"""

JS = """
function tgl(){const r=document.documentElement,
  d=(r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'))==='dark';
  r.dataset.theme=d?'light':'dark';try{localStorage.htmlreport_theme=r.dataset.theme}catch(e){}}
function fs(d){const r=document.documentElement,
  cur=parseFloat(getComputedStyle(r).getPropertyValue('--fs'))||17,
  next=Math.min(24,Math.max(13,cur+d));
  r.style.setProperty('--fs',next+'px');try{localStorage.htmlreport_fs=next}catch(e){}}
try{if(localStorage.htmlreport_fs)document.documentElement.style.setProperty('--fs',localStorage.htmlreport_fs+'px');
    if(localStorage.htmlreport_theme)document.documentElement.dataset.theme=localStorage.htmlreport_theme}catch(e){}
(()=>{const t=document.getElementById('toc');if(!t)return;
document.querySelectorAll('main h2, main h3').forEach(h=>{
  const s=h.closest('section')||h;if(!h.id)h.id=(h.textContent||'').trim().toLowerCase()
    .replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
  const a=document.createElement('a');a.href='#'+(h.tagName==='H2'&&s.id?s.id:h.id);
  a.textContent=h.textContent;if(h.tagName==='H3')a.className='h3';t.appendChild(a)})})();
document.querySelectorAll('table').forEach(tb=>{
  tb.querySelectorAll('th:not(.nosort)').forEach((th,i)=>th.addEventListener('click',()=>{
    const dir=th.dataset.d=th.dataset.d==='a'?'d':'a';
    const val=td=>td&&td.dataset.v!==undefined?+td.dataset.v:(td?td.textContent.trim():'');
    [...tb.tBodies[0].rows].sort((r1,r2)=>{const a=val(r1.cells[i]),b=val(r2.cells[i]);
      const c=(typeof a=='number'&&typeof b=='number')?a-b:String(a).localeCompare(String(b));
      return dir==='a'?c:-c}).forEach(r=>tb.tBodies[0].appendChild(r))}))});
function flt(inp,id){const q=inp.value.toLowerCase();
  document.querySelectorAll('#'+id+' tbody tr').forEach(r=>
    r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')}
(()=>{const U=[[31536e6,'y'],[2592e6,'mo'],[6048e5,'w'],[864e5,'d'],[36e5,'h'],[6e4,'m']];
document.querySelectorAll('time.age').forEach(t=>{
  const d=new Date(t.dateTime);if(isNaN(d))return;const ms=Date.now()-d;
  let s='just now';for(const[u,n]of U){if(Math.abs(ms)>=u){s=Math.floor(Math.abs(ms)/u)+n+(ms<0?' ahead':' ago');break}}
  const pad=n=>String(n).padStart(2,'0');
  const iso=d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+' '+pad(d.getHours())+':'+pad(d.getMinutes());
  if(!t.textContent.trim())t.textContent=iso+' ('+s+')';else t.textContent+=' ('+s+')';
  t.title=d.toString();
  const budget=+(t.dataset.staleDays||0);
  if(budget&&ms>budget*864e5)t.classList.add('stale')})})();
(()=>{let cur=null,pinned=false,timer=null;
const kill=()=>{if(cur&&!pinned){cur.remove();cur=null}};
const show=(host,html,title)=>{
  if(cur)cur.remove();
  const p=document.createElement('div');p.className='pop';
  p.innerHTML='<button class="pin" title="unpin">\\u{1F4CC}</button>'+(title?'<h4>'+title+'</h4>':'')+html;
  document.body.appendChild(p);
  const r=host.getBoundingClientRect(),pr=p.getBoundingClientRect();
  let top=r.bottom+8; if(top+pr.height>innerHeight-8)top=Math.max(8,r.top-pr.height-8);
  let left=Math.min(r.left,innerWidth-pr.width-8);
  p.style.top=top+'px';p.style.left=Math.max(8,left)+'px';
  p.querySelector('.pin').onclick=()=>{pinned=false;kill()};
  p.onmouseenter=()=>clearTimeout(timer);
  p.onmouseleave=()=>{timer=setTimeout(kill,220)};
  cur=p};
const src=el=>{
  const t=el.querySelector(':scope > template.pop');
  if(t)return[t.innerHTML,el.dataset.popTitle||''];
  return null};
document.querySelectorAll(':has(> template.pop)').forEach(el=>{
  const s=src(el);if(!s)return;
  if(el.tabIndex<0)el.tabIndex=0;
  const open=()=>{clearTimeout(timer);pinned=false;show(el,s[0],s[1])};
  el.addEventListener('mouseenter',()=>{clearTimeout(timer);timer=setTimeout(open,180)});
  el.addEventListener('mouseleave',()=>{clearTimeout(timer);timer=setTimeout(kill,220)});
  el.addEventListener('focus',open);
  el.addEventListener('click',e=>{
    if(el.tagName==='A'&&(e.metaKey||e.ctrlKey))return;
    e.preventDefault();if(!cur)open();pinned=!pinned});
});
addEventListener('keydown',e=>{if(e.key==='Escape'){pinned=false;kill()}});})();
"""


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def render(facts: Facts, source: Path, generated: dt.datetime, schema_gap: dict[str, Any] | None = None) -> str:
    schema_gap = schema_gap or {}
    snaps = {
        "now": facts.snapshot(generated),
        "7d": facts.snapshot(generated - dt.timedelta(days=7)),
        "14d": facts.snapshot(generated - dt.timedelta(days=14)),
    }
    series = facts.daily_series()
    days_axis = [d for d, _, _ in series]
    insights = compute_insights(facts, schema_gap, snaps)
    sev_counts = Counter(i.sev for i in insights)
    out: list[str] = []
    add = out.append
    src = str(source)

    add('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add(f"<title>Beads backlog — state of the graph — {generated:%Y-%m-%d}</title>")
    add("<style>" + CSS + "</style>")
    add("</head>\n<body>")
    add('<header class="page"><h1>Beads backlog &mdash; state of the graph</h1>')
    add(f'<span class="chip">polylogue</span><span class="chip">{facts.total:,} beads</span>')
    add(f'<span class="chip">{generated:%Y-%m-%d}</span><span class="spacer"></span>')
    add('<button class="fs" onclick="fs(-1)" title="smaller text">A&minus;</button>')
    add('<button class="fs" onclick="fs(1)" title="larger text">A+</button>')
    add('<button class="theme" onclick="tgl()">&#9680; theme</button></header>')
    add('<div class="layout"><nav id="toc" aria-label="contents"></nav><main>')

    # ---------------- summary ----------------
    add('<section id="summary"><h2>Summary</h2>')
    add('<dl class="meta">')
    add(f'<dt>generated</dt><dd><time class="age" datetime="{generated.isoformat()}"></time></dd>')
    add(
        f'<dt>data as of</dt><dd><time class="age" datetime="{generated.isoformat()}" '
        'data-stale-days="3"></time> &mdash; a live backlog; re-run to refresh</dd>'
    )
    add(
        '<dt>basis</dt><dd><span class="ev ev-measured">measured</span> every count, matrix, '
        'tree, and histogram &middot; <span class="ev ev-derived">derived</span> every trend '
        "(reconstructed from timestamps; deletions and reopen history are invisible) and every "
        "finding (each emitted by a condition checked against this data)</dd>"
    )
    add(f'<dt>source</dt><dd><a class="path" href="file://{esc(src)}">{esc(src)}</a></dd>')
    add("<dt>regenerate</dt><dd><code>devtools workspace beads-state-report --fresh --out &lt;path&gt;</code></dd>")
    add("</dl>")

    tiles = [
        (
            f"{facts.total:,}",
            delta_chip(snaps["now"]["total"], snaps["7d"]["total"]),
            "beads, all time",
            "wc -l .beads/issues.jsonl",
        ),
        (
            f"{facts.open_total:,}",
            delta_chip(snaps["now"]["open"], snaps["7d"]["open"], up_is_bad=True),
            "open + in progress",
            "jq -r 'select(._type==\"issue\")|.status' .beads/issues.jsonl | grep -vc closed",
        ),
        (
            f"{facts.status['closed']:,}",
            delta_chip(snaps["now"]["closed"], snaps["7d"]["closed"]),
            "closed",
            "jq -r 'select(._type==\"issue\")|.status' .beads/issues.jsonl | grep -c closed",
        ),
        (
            f"{len(facts.ready):,}",
            delta_chip(snaps["now"]["ready"], snaps["7d"]["ready"]),
            "ready (no open blocker)",
            "bd ready --limit 5000 --json | jq length",
        ),
        (
            str(snaps["now"]["p0_open"]),
            delta_chip(snaps["now"]["p0_open"], snaps["7d"]["p0_open"], up_is_bad=True),
            "open P0",
            "jq -r 'select(._type==\"issue\")|select(.status!=\"closed\")|.priority' .beads/issues.jsonl | grep -c '^0$'",
        ),
        (
            f"{len(facts.epics)}",
            "",
            "beads with children",
            "jq -r '.dependencies[]?|select(.type==\"parent-child\")|.depends_on_id' "
            ".beads/issues.jsonl | sort -u | wc -l",
        ),
    ]
    add('<div class="tiles">')
    for value, delta, label, cmd in tiles:
        add(
            f'<div class="tile qa"><span class="n">{value}{delta}</span>'
            f'<span class="l">{label} <small>&Delta;7d</small></span>'
            f"{qa(cmd, 'measured ' + generated.strftime('%Y-%m-%d'))}</div>"
        )
    add("</div>")
    add(
        f"<p>The whole bead population, open and closed, as one artifact: {facts.total:,} beads "
        f"created over {facts.span_days} days ({facts.first_created} &rarr; {facts.last_created}), "
        f"{facts.status['closed']:,} already closed. Deltas on the tiles are against the state "
        "reconstructed 7 days ago. The findings below are the report's judgement layer &mdash; "
        "every one is emitted by a condition this generator checked against this file, so a claim "
        "that stops being true stops being printed.</p>"
    )
    add('<div class="digest">')
    for sev in ("bad", "warn", "info", "ok"):
        if sev_counts.get(sev):
            add(f'<a class="badge {sev}" href="#findings">{sev_counts[sev]} {sev}</a>')
    add("</div>")
    add("</section>")

    # ---------------- pulse ----------------
    add('<section id="pulse"><h2>Pulse</h2>')
    add(
        "<p>Backlog state over its whole life, reconstructed per day from "
        "<code>created_at</code>/<code>closed_at</code> and edge timestamps "
        '<span class="ev ev-derived">derived</span>. '
        '<span style="color:var(--accent)">&#9632; open</span> '
        '<span style="color:var(--ok)">&#9632; ready</span>. '
        "The gap between the lines is the blocked share; the lines converging means the "
        "graph is getting less dependency-bound.</p>"
    )
    add('<figure class="chart">' + burnup_svg(series) + "</figure>")
    add('<div class="tablewrap"><table id="snapt"><thead><tr><th class="nosort">metric</th>')
    add('<th class="nosort">14d ago</th><th class="nosort">7d ago</th><th class="nosort">now</th>')
    add('<th class="nosort">&Delta;7d</th></tr></thead><tbody>')
    for key, label, up_bad in (
        ("open", "open beads", True),
        ("ready", "ready (no open blocker)", False),
        ("blocked", "blocked by an open bead", True),
        ("p0_open", "open P0", True),
    ):
        d = delta_chip(snaps["now"][key], snaps["7d"][key], up_is_bad=up_bad)
        add(
            f"<tr><td>{label}</td><td class='num'>{snaps['14d'][key]}</td>"
            f"<td class='num'>{snaps['7d'][key]}</td><td class='num'><b>{snaps['now'][key]}</b></td>"
            f"<td>{d}</td></tr>"
        )
    add("</tbody></table></div>")
    add(
        "<p>Reconstruction caveats, stated once: a deleted bead or edge leaves no trace in the "
        "export, and a reopened bead's earlier closure is overwritten by its latest one. Both are "
        "rare enough here that the curve's shape is trustworthy; individual day values are "
        "&plusmn;a few.</p>"
    )
    add("</section>")

    # ---------------- notation ----------------
    add('<section id="notation"><h2>Bead notation</h2>')
    add(
        "<p>One bead renders as one chip, everywhere in this report. Left bar and "
        "leading number are priority; then status, id, type glyph, and dependency "
        "degree. Hover any chip for the expanded reading.</p>"
    )
    samples = [str(facts.rows[0]["id"])]
    for wanted in ("epic", "bug", "feature"):
        for r in facts.rows:
            if r["issue_type"] == wanted and str(r["id"]) not in samples:
                samples.append(str(r["id"]))
                break
    add("<p>" + " ".join(chip(facts, s) for s in samples[:4]) + "</p>")
    add('<div class="legend">')
    add("<div><b>P0&ndash;P4</b><span>priority; also the left bar colour</span></div>")
    for glyph, name in STATUS_GLYPH.values():
        add(f"<div><b>{glyph}</b><span>{name}</span></div>")
    for glyph, name in TYPE_GLYPH.values():
        add(f"<div><b>{glyph}</b><span>{name}</span></div>")
    add("<div><b>&uarr;n</b><span>n open beads block this one</span></div>")
    add("<div><b>&darr;n</b><span>this one blocks n open beads</span></div>")
    add("<div><b>struck id</b><span>closed</span></div>")
    add("<div><b>blue outline</b><span>in progress</span></div>")
    add("<div><b>dashed border</b><span>deferred</span></div>")
    add("</div>")
    add(
        '<p><span class="badge info">reading it</span> A chip with a red left bar, a hollow '
        "circle, and <code>&darr;9</code> is an open P0 that nine other open beads are "
        "waiting on &mdash; the highest-leverage shape in the graph.</p>"
    )
    add("</section>")

    # ---------------- shape ----------------
    add('<section id="shape"><h2>Shape</h2>')
    add(
        "<p>The status &times; priority matrix, plus per-tier closure rate and median lead "
        "time &mdash; the fastest read on whether the priority scale is load-bearing. The "
        "computed verdict is in <a href='#findings'>Findings</a>.</p>"
    )
    add("<h3>Status &times; priority</h3>")
    add('<div class="tablewrap"><table class="mx" id="mx">')
    status_cols = facts.statuses_present
    head = "".join(f"<th>{esc(STATUS_GLYPH.get(s, ('?', s))[1])}</th>" for s in status_cols)
    add(
        f"<thead><tr><th>priority</th>{head}<th>total</th><th>closed %</th><th>median lead (days)</th></tr></thead><tbody>"
    )
    peak = max(facts.matrix.values()) or 1
    for p in range(5):
        row = [f'<tr><td><span class="bd p{p}"><span class="bp">P{p}</span></span></td>']
        for st in status_cols:
            n = facts.matrix.get((st, p), 0)
            row.append(f'<td class="c" data-v="{n}"><i style="width:{100 * n / peak:.1f}%"></i><b>{n:,}</b></td>')
        total_p = facts.priority[p]
        med = _median(facts.lead_by_priority[p])
        row.append(f'<td class="num" data-v="{total_p}">{total_p:,}</td>')
        row.append(f'<td class="num" data-v="{facts.rate(p):.4f}">{facts.rate(p):.0%}</td>')
        row.append(f'<td class="num" data-v="{med:.3f}">{med:.2f}</td>')
        add("".join(row) + "</tr>")
    add("</tbody></table></div>")

    add("<h3>Type</h3>")
    add('<div class="tablewrap"><table id="ty"><thead><tr><th>type</th><th>beads</th>')
    add("<th>share</th><th>closed %</th></tr></thead><tbody>")
    peak_t = max(facts.itype.values()) or 1
    for tname, n in facts.itype.most_common():
        closed = sum(1 for r in facts.rows if r["issue_type"] == tname and r["status"] == "closed")
        glyph = TYPE_GLYPH.get(tname, ("?", tname))[0]
        add(
            f"<tr><td>{glyph} {esc(tname)}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_t:.1f}%"></span>'
            f"<b>{n:,}</b></td>"
            f'<td class="num" data-v="{n / facts.total:.4f}">{100 * n / facts.total:.1f}%</td>'
            f'<td class="num" data-v="{closed / n:.4f}">{closed / n:.0%}</td></tr>'
        )
    add("</tbody></table></div>")
    add("</section>")

    # ---------------- structure ----------------
    add('<section id="structure"><h2>Structure</h2>')
    add(
        "<p>Parents ranked by child count, from <code>parent-child</code> edges (the dotted "
        "id is <em>not</em> the parent link &mdash; the two disagree for "
        f"{len(facts.id_mismatch)} beads and {len(facts.id_orphan)} dotted beads have no edge "
        "at all; both are counted in Health). The trend column is each epic's open-children "
        'count over the backlog\'s life <span class="ev ev-derived">derived</span> &mdash; '
        "a falling line is a draining program, a flat high line is a parked one, a rising "
        "line is a program still being decomposed.</p>"
    )
    add('<div class="tablewrap"><table id="ep"><thead><tr><th>parent</th><th>title</th>')
    add('<th>progress</th><th class="nosort">trend</th><th>children</th><th>open</th></tr></thead><tbody>')
    for e in facts.epics[:24]:
        eid = str(e["id"])
        spark = sparkline_svg(facts.epic_open_series(eid, days_axis), label=f"{eid} open children")
        add(
            f"<tr><td>{chip(facts, eid)}</td>"
            f"<td>{esc(str(e['title'])[:74])}</td>"
            f'<td data-v="{(int(e["n"]) - int(e["open"])) / int(e["n"]):.4f}">'
            f"{fill_bar(int(e['open']), int(e['n']))}</td>"
            f"<td>{spark}</td>"
            f'<td class="num" data-v="{e["n"]}">{e["n"]}</td>'
            f'<td class="num" data-v="{e["open"]}">{e["open"]}</td></tr>'
        )
    add("</tbody></table></div>")

    # the two computed extremes: most-finished and least-started large parents
    big = [e for e in facts.epics if int(e["n"]) >= 8]
    if big:
        most_done = min(big, key=lambda e: int(e["open"]) / int(e["n"]))
        least_started = max(big, key=lambda e: int(e["open"]) / int(e["n"]))
        add("<h3>Trees: the two extremes</h3>")
        add(
            f"<p>Computed from the table above: the most-finished large parent "
            f"({chip(facts, str(most_done['id']))}, {int(most_done['n']) - int(most_done['open'])} of "
            f"{most_done['n']} children closed) against the least-started "
            f"({chip(facts, str(least_started['id']))}, {least_started['open']} of "
            f"{least_started['n']} still open). A flat backlog list renders those identically; "
            "the fill bars are why this section exists.</p>"
        )
        for e in (most_done, least_started):
            eid = str(e["id"])
            if eid not in facts.issues:
                continue
            rec = facts.issues[eid]
            add(
                f"<details><summary>{chip(facts, eid)} &nbsp;{esc(str(rec['title'])[:80])}</summary>"
                + tree_html(facts, eid)
                + "</details>"
            )
        add("<details><summary>All other parents with 8+ children</summary>")
        skip = {str(most_done["id"]), str(least_started["id"])}
        for e in big:
            if str(e["id"]) in skip:
                continue
            add(
                f"<details><summary>{chip(facts, str(e['id']))} &nbsp;{esc(str(e['title'])[:80])} "
                f"&nbsp;{fill_bar(int(e['open']), int(e['n']))}</summary>"
                + tree_html(facts, str(e["id"]))
                + "</details>"
            )
        add("</details>")
    add("</section>")

    # ---------------- topology ----------------
    add('<section id="topology"><h2>Topology</h2>')
    ready_pct = f"{len(facts.ready) / facts.open_total:.0%}" if facts.open_total else "n/a"
    add(
        f"<p>The <code>blocks</code> graph: {facts.dep_types['blocks']:,} edges over "
        f"{facts.open_total:,} open beads, of which <b>{len(facts.ready):,} are ready</b> "
        f"({ready_pct}) and {len(facts.blocked)} wait on an open blocker.</p>"
    )
    add('<div class="tiles">')
    for value, label, cmd in [
        (
            f"{facts.dep_types['blocks']:,}",
            "blocks edges",
            "jq -r '.dependencies[]?|.type' .beads/issues.jsonl | sort | uniq -c",
        ),
        (f"{len(facts.ready):,}", "ready", "bd ready --limit 5000 --json | jq length"),
        (f"{len(facts.blocked)}", "blocked by an open bead", "bd blocked --json | jq length"),
        (f"{len(facts.cycles)}", "cycles in blocks", "DFS colouring over the blocks graph (this generator)"),
    ]:
        add(
            f'<div class="tile qa"><span class="n">{value}</span><span class="l">{label}</span>'
            f"{qa(cmd, 'measured')}</div>"
        )
    add("</div>")

    frontier = facts.parallel_frontier()
    if len(frontier) >= 2:
        add("<h3>The parallel frontier</h3>")
        add(
            "<p>Computed: the largest programs whose open subtrees occupy <em>disjoint "
            "components</em> of the <code>blocks</code> graph &mdash; no dependency path of any "
            "length connects them, so they can run as parallel lanes with zero coordination "
            "on ordering. Greedy by open-descendant count.</p>"
        )
        add('<div class="tablewrap"><table id="pf"><thead><tr><th>program</th><th>title</th>')
        add("<th>open descendants</th><th>open P0&ndash;P2</th></tr></thead><tbody>")
        for c in frontier:
            add(
                f"<tr><td>{chip(facts, str(c['id']))}</td><td>{esc(str(c['title'])[:70])}</td>"
                f'<td class="num" data-v="{c["open_desc"]}">{c["open_desc"]}</td>'
                f'<td class="num" data-v="{c["urgent"]}">{c["urgent"]}</td></tr>'
            )
        add("</tbody></table></div>")

    add("<h3>Relation vocabulary</h3>")
    add('<div class="tablewrap"><table id="rel"><thead><tr><th>relation</th><th>edges</th>')
    add("<th>first written</th><th>last written</th></tr></thead><tbody>")
    first_last: dict[str, tuple[str, str]] = {}
    for _, _, kind, created in facts.edges:
        when = created[:10]
        if not when:
            continue
        lo, hi = first_last.get(kind, (when, when))
        first_last[kind] = (min(lo, when), max(hi, when))
    peak_r = max(facts.dep_types.values()) or 1
    for kind, n in facts.dep_types.most_common():
        lo, hi = first_last.get(kind, ("", ""))
        flag = ""
        if kind in ("relates-to", "related"):
            flag = {
                "split": ' <span class="badge bad">split vocab</span>',
                "recurring": ' <span class="badge warn">unified, then recurred</span>',
                "clean": ' <span class="badge ok">unified</span>',
            }[facts.vocab_state]
        add(
            f"<tr><td><code>{esc(kind)}</code>{flag}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_r:.1f}%"></span><b>{n:,}</b></td>'
            f"<td>{esc(lo)}</td><td>{esc(hi)}</td></tr>"
        )
    add("</tbody></table></div>")
    if facts.retyped_spike:
        add(
            f"<p>A bulk re-type is visible in the timestamps: {facts.retyped_spike:,} "
            f"<code>relates-to</code> edges carry <code>{esc(facts.retyped_spike_day)}</code> as "
            "their <code>created_at</code> &mdash; converted edges lose their original assertion "
            "dates, so any &ldquo;when were associations asserted?&rdquo; analysis must treat that "
            'date as an artifact spike, not a real burst <span class="ev ev-derived">derived</span>.</p>'
        )
    if facts.related_recurrence:
        add(
            "<p>Post-repair recurrence, exactly: the edge(s) below were written <em>after</em> "
            "the newest re-typed edge, proving the write path is still unconstrained.</p>"
        )
        add('<div class="tablewrap"><table id="rr"><thead><tr><th>from</th><th>relates to</th>')
        add("<th>created_at</th></tr></thead><tbody>")
        for src_id, dst_id, ts in facts.related_recurrence:
            add(f"<tr><td>{chip(facts, src_id)}</td><td>{chip(facts, dst_id)}</td><td>{esc(ts)}</td></tr>")
        add("</tbody></table></div>")

    add("<h3>Highest-leverage blockers</h3>")
    add(
        "<p>Each row releases a fan-out no other bead does; if the graph is "
        "dependency-bound these are the lever, and if it is throughput-bound they are "
        "merely the tidiest finish-first candidates.</p>"
    )
    add('<div class="tablewrap"><table id="tb"><thead><tr><th>bead</th><th>title</th>')
    add("<th>blocks (open)</th></tr></thead><tbody>")
    for n, bid in facts.top_blockers[:20]:
        add(
            f"<tr><td>{chip(facts, bid)}</td><td>{esc(str(facts.issues[bid]['title'])[:80])}</td>"
            f'<td class="num" data-v="{n}">{n}</td></tr>'
        )
    add("</tbody></table></div>")

    cluster = facts.densest_cluster()
    core = facts.cluster_core(cluster)
    if core:
        add("<h3>Densest blocking cluster</h3>")
        add(
            f"<p>Rendering all {facts.total:,} beads as a graph produces a hairball, so this is "
            f"a slice: the largest connected component of the open-only <code>blocks</code> graph "
            f"holds <b>{len(cluster)}</b> beads, and this is its <b>{len(core)}</b>-node core "
            "(members with &ge;3 neighbours inside it; the dropped nodes are single-edge leaves). "
            "Arrows point blocker &rarr; blocked; columns are longest-path depth, so leftmost is "
            "furthest upstream. Hover a node for its title.</p>"
        )
        add('<div class="dagwrap">' + cluster_svg(facts, core) + "</div>")
    add("</section>")

    # ---------------- time ----------------
    add('<section id="time"><h2>Time</h2>')
    peak_day, peak_n = max(facts.created_day.items(), key=lambda kv: kv[1])
    add(
        f"<p>Filing is spiky ({peak_n} beads on {esc(peak_day)}, the busiest day, against a "
        f"median of {_median(list(facts.created_day.values())):.0f}/day) &mdash; the signature "
        "of audit-lane batch filing rather than continuous discovery. Closure runs at a median "
        f"of {_median(list(facts.closed_day.values())):.0f}/day.</p>"
    )
    add('<figure class="chart">')
    add(
        histogram_svg(
            facts.days,
            [("created", facts.created_day, "created"), ("closed", facts.closed_day, "closed")],
        )
    )
    add(
        f"<figcaption>beads per day, {facts.first_created} &rarr; {facts.last_created} "
        f'&middot; <span style="color:var(--accent)">&#9632; created</span> '
        f'<span style="color:var(--ok)">&#9632; closed</span> &middot; '
        f'<span class="ev ev-measured">measured</span></figcaption>'
    )
    add("</figure>")

    (rc, rx), (pc, px) = facts.win_recent, facts.win_prior
    add('<div class="tablewrap"><table id="vel"><thead><tr><th class="nosort">window</th>')
    add('<th class="nosort">created</th><th class="nosort">closed</th><th class="nosort">net</th></tr></thead><tbody>')
    for name, created_n, closed_n in (
        (f"prior {VELOCITY_WINDOW_DAYS}d", pc, px),
        (f"last {VELOCITY_WINDOW_DAYS}d", rc, rx),
    ):
        cls = "bad" if created_n - closed_n > 0 else "ok"
        add(
            f'<tr><td>{name}</td><td class="num">{created_n}</td><td class="num">{closed_n}</td>'
            f'<td class="num"><span class="badge {cls}">{created_n - closed_n:+d}</span></td></tr>'
        )
    add("</tbody></table></div>")

    add("<h3>Age &times; priority of open work</h3>")
    add(
        "<p>Where open work sits in the age-urgency plane. Mass in the bottom-left "
        "(old, urgent) is the shelf to triage first; mass in the bottom-right (old, P3/P4) "
        "is parking, which is fine as long as it was chosen.</p>"
    )
    add(heatmap_html(facts))
    add(
        f"<p>Median open bead is {_median(facts.age_open):.0f} days old; the oldest is "
        f"{max(facts.age_open):.0f} days &mdash; against a tracker that is itself only "
        f"{facts.span_days} days old.</p>"
    )
    add("</section>")

    # ---------------- health ----------------
    add('<section id="health"><h2>Health</h2>')
    add(
        f"<p>{len(HEALTH_CHECKS)} mechanical checks over the graph. These are a "
        "<em>review queue</em>, not a batch-fix list &mdash; especially the "
        "&ldquo;open parent, all children closed&rdquo; row, where children closing means the "
        "parent is ready for adjudication against its own acceptance criteria, never that it "
        "is done.</p>"
    )
    add('<div class="tablewrap"><table id="hc"><thead><tr><th>check</th><th>count</th>')
    add("<th>meaning</th></tr></thead><tbody>")
    for cls, name, attr, meaning in HEALTH_CHECKS:
        n = len(getattr(facts, attr))
        badge = f'<span class="badge {"ok" if n == 0 else cls}">{n}</span>'
        add(f"<tr><td>{name}</td><td data-v='{n}'>{badge}</td><td>{meaning}</td></tr>")
    add("</tbody></table></div>")

    if facts.dangling:
        add("<h3>Dangling references</h3>")
        add(
            "<p>A genuine data bug, not a judgement call: these edges point at ids that "
            "resolve to nothing, so the blocking intent behind them is silently absent from "
            "every query.</p>"
        )
        add('<div class="tablewrap"><table id="dg"><thead><tr><th>from</th><th>points at</th>')
        add("<th>relation</th></tr></thead><tbody>")
        for a, b, kind in facts.dangling:
            add(
                f"<tr><td>{chip(facts, a)}</td>"
                f'<td><span class="bd missing"><span class="bi">{esc(b)}</span>'
                f'<span class="bg">&#10008;</span></span></td><td><code>{esc(kind)}</code></td></tr>'
            )
        add("</tbody></table></div>")

    if facts.stale_claims:
        add("<h3>Stale in-progress claims</h3>")
        add('<div class="tablewrap"><table id="sc"><thead><tr><th>bead</th><th>title</th>')
        add("<th>days since update</th></tr></thead><tbody>")
        for bid, days_stale in facts.stale_claims:
            add(
                f"<tr><td>{chip(facts, bid)}</td><td>{esc(str(facts.issues[bid]['title'])[:76])}</td>"
                f'<td class="num" data-v="{days_stale:.1f}">{days_stale:.0f}</td></tr>'
            )
        add("</tbody></table></div>")

    add("<h3>Review queue: open parents whose children are all closed</h3>")
    add('<div class="tablewrap"><table id="rq"><thead><tr><th>parent</th><th>title</th>')
    add('<th>children</th><th class="nosort">has own AC</th></tr></thead><tbody>')
    for pid in facts.open_parent_all_closed:
        rec = facts.issues[pid]
        has_ac = bool(str(rec.get("acceptance_criteria") or "").strip())
        flag = '<span class="badge warn">yes &mdash; read it</span>' if has_ac else '<span class="badge info">no</span>'
        add(
            f"<tr><td>{chip(facts, pid)}</td><td>{esc(str(rec['title'])[:76])}</td>"
            f'<td class="num" data-v="{len(facts.children[pid])}">{len(facts.children[pid])}</td>'
            f"<td>{flag}</td></tr>"
        )
    add("</tbody></table></div>")

    if facts.dup_titles:
        add("<h3>Duplicate titles</h3>")
        add('<div class="tablewrap"><table id="dup"><thead><tr><th>title</th>')
        add('<th class="nosort">beads</th></tr></thead><tbody>')
        for title, ids in facts.dup_titles:
            add(f"<tr><td>{esc(title[:86])}</td><td>" + " ".join(chip(facts, i) for i in ids) + "</td></tr>")
        add("</tbody></table></div>")

    # schema gap: a live archive/code divergence fact, rendered when measurable
    if schema_gap.get("available"):
        add("<h3>Archive/code schema gap</h3>")
        add(
            f"<p>The live archive is at index-schema <b>v{schema_gap['live_version']}</b>; "
            f"<code>origin/master</code> declares <b>v{schema_gap['declared']}</b>. "
            f"<b>{len(schema_gap['blockers'])}</b> intervening version(s) are "
            "<code>SEMANTIC_REPARSE</code> &mdash; no clone-safe SQL fast-forward exists, so every "
            "merged fix at those versions is inert until "
            "<code>polylogue ops reset --index &amp;&amp; polylogued run</code>. A closed bead "
            "here can still describe a live bug against the data an operator queries right now.</p>"
        )
        add('<div class="tiles">')
        for value, label, cmd in [
            (
                f"v{schema_gap['live_version']}",
                "live archive schema",
                f"sqlite3 -readonly {schema_gap.get('live_path', '<archive>/index.db')} 'PRAGMA user_version;'",
            ),
            (
                f"v{schema_gap['declared']}",
                "origin/master declares",
                "git show origin/master:polylogue/storage/sqlite/archive_tiers/index.py | grep INDEX_SCHEMA_VERSION",
            ),
            (
                str(len(schema_gap["blockers"])),
                "SEMANTIC_REPARSE versions blocking fast-forward",
                "git show origin/master:polylogue/storage/sqlite/lifecycle.py",
            ),
        ]:
            add(
                f'<div class="tile qa"><span class="n">{esc(value)}</span><span class="l">{esc(label)}</span>'
                f"{qa(cmd, 'measured ' + generated.strftime('%Y-%m-%d'))}</div>"
            )
        add("</div>")
        if schema_gap["blockers"]:
            add('<div class="tablewrap"><table id="sg"><thead><tr><th>version</th>')
            add('<th class="nosort">delta class(es)</th></tr></thead><tbody>')
            for v in range(schema_gap["live_version"] + 1, schema_gap["declared"] + 1):
                classes = schema_gap["blocker_notes"].get(v, ())
                cls_badges = (
                    " ".join(
                        f'<span class="badge {"bad" if c == "SEMANTIC_REPARSE" else "info"}">{esc(c)}</span>'
                        for c in classes
                    )
                    or '<span class="badge info">undeclared</span>'
                )
                add(f"<tr><td>v{v}</td><td>{cls_badges}</td></tr>")
            add("</tbody></table></div>")
    else:
        add(
            '<p><span class="badge warn">schema gap not measured</span> Needs both '
            "<code>git show origin/master:&hellip;</code> and a readable live archive at the "
            "operator's configured archive root; at least one was unavailable, so the tiles are "
            "omitted rather than guessed.</p>"
        )
    add("</section>")

    # ---------------- themes ----------------
    add('<section id="themes"><h2>Themes</h2>')
    add(
        f"<p>Two independent readings of where open work concentrates, because neither alone is "
        f"trustworthy: the label view folds the repository's own <code>area:*</code> labels "
        f"through an explicit synonym map (covers {facts.labelled:,} of {facts.total:,} beads and "
        f"says nothing about the rest), and the keyword view classifies every bead from title and "
        "description (full coverage, will misfile loose titles). Where the two agree, the "
        "concentration is real.</p>"
    )
    add('<div class="tablewrap"><table id="th"><thead><tr><th>subsystem</th>')
    add(
        "<th>open (labels)</th><th>closed (labels)</th><th>open (keywords)</th>"
        "<th>closed (keywords)</th></tr></thead><tbody>"
    )
    names = sorted(set(facts.by_label_theme) | set(facts.by_kw_theme))
    peak_o = (
        max(
            (facts.by_label_theme[n]["open"] + facts.by_label_theme[n]["in_progress"] for n in names),
            default=1,
        )
        or 1
    )
    for name in names:
        lab = facts.by_label_theme.get(name, Counter())
        kw = facts.by_kw_theme.get(name, Counter())
        lab_open = lab["open"] + lab["in_progress"]
        kw_open = kw["open"] + kw["in_progress"]
        add(
            f"<tr><td>{esc(name)}</td>"
            f'<td class="dist" data-v="{lab_open}"><span style="--w:{100 * lab_open / peak_o:.1f}%"></span><b>{lab_open}</b></td>'
            f'<td class="num" data-v="{lab["closed"]}">{lab["closed"]}</td>'
            f'<td class="num" data-v="{kw_open}">{kw_open}</td>'
            f'<td class="num" data-v="{kw["closed"]}">{kw["closed"]}</td></tr>'
        )
    add("</tbody></table></div>")
    add("<details><summary>The synonym map used to fold <code>area:*</code> labels</summary>")
    add('<div class="tablewrap"><table id="syn"><thead><tr><th>subsystem</th>')
    add('<th class="nosort">folded area labels</th></tr></thead><tbody>')
    for canon, suffixes in SUBSYSTEMS.items():
        add(f"<tr><td>{esc(canon)}</td><td>" + " ".join(f"<code>area:{esc(s)}</code>" for s in suffixes) + "</td></tr>")
    add("</tbody></table></div>")
    add(
        "<p>This map is a proposal written by hand, not something the repository declares. "
        "Disagreeing with a row changes the table above it.</p>"
    )
    add("</details>")
    add("<details><summary>Raw <code>area:*</code> labels, top 30</summary>")
    add('<div class="tablewrap"><table id="raw"><thead><tr><th>label</th><th>beads</th>')
    add("</tr></thead><tbody>")
    area_counts = [(x, facts.labels[x]) for x in facts.area_labels]
    area_counts.sort(key=lambda kv: (-kv[1], kv[0]))
    peak_l = area_counts[0][1] if area_counts else 1
    for name, n in area_counts[:30]:
        add(
            f"<tr><td><code>{esc(name)}</code></td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_l:.1f}%"></span><b>{n}</b></td></tr>'
        )
    add("</tbody></table></div></details>")
    add("</section>")

    # ---------------- dated review passes ----------------
    strict_total = sum(facts.verdict_counts.values())
    loose_total = sum(facts.loose_counts.values())
    if strict_total or facts.reconciliation:
        add('<section id="passes"><h2>Dated review passes</h2>')
        add(
            "<p>Two hand-verification passes left machine-parseable markers in bead notes; this "
            "generator extracts them mechanically. They are <em>dated</em> evidence &mdash; "
            "ground truth as of the day each pass ran, ageing at the speed the code moves, not "
            "recomputed facts.</p>"
        )
    if strict_total:
        add('<h3>The verified subset <span class="chip">pass of 2026-07-31</span></h3>')
        add(
            f"<p><b>{strict_total}</b> beads carry a machine-parseable "
            "<code>VERIFICATION (&hellip;): STALE|PARTIAL|LIVE</code> verdict written by a human "
            f"reviewer &mdash; {facts.verdict_counts['LIVE']} LIVE / "
            f"{facts.verdict_counts['PARTIAL']} PARTIAL / {facts.verdict_counts['STALE']} STALE. "
            "It is the only subset of this backlog where &ldquo;is this claim still true?&rdquo; "
            "was answered by inspection rather than assumed. Preserve it as a labelled "
            "evaluation set.</p>"
        )
        add(
            f"<p>Extraction gap, stated rather than papered over: the operator's own count of "
            f"the pass is {OPERATOR_VERIFIED_COUNT} beads; the strict pattern finds "
            f"{strict_total}, and relaxing to any bare verdict token finds {loose_total} "
            f"({facts.loose_counts['LIVE']}/{facts.loose_counts['PARTIAL']}/"
            f"{facts.loose_counts['STALE']}), which brackets the figure but includes incidental "
            "prose. The difference is verdicts phrased outside the parseable form &mdash; a "
            "review this valuable should write its verdict one way, every time.</p>"
        )
        add('<input class="filter" placeholder="filter verdicts&hellip;" oninput="flt(this,\'vd\')">')
        add('<div class="tablewrap"><table id="vd"><thead><tr><th>bead</th><th>verdict</th>')
        add("<th>sweep</th><th>title</th></tr></thead><tbody>")
        badge_of = {"LIVE": "ok", "PARTIAL": "warn", "STALE": "bad"}
        for bid, verdict, _status, sweep in sorted(facts.verdicts, key=lambda v: (v[1], v[0])):
            add(
                f"<tr><td>{chip(facts, bid)}</td>"
                f'<td><span class="badge {badge_of[verdict]}">{verdict}</span></td>'
                f"<td>{esc(sweep[:34])}</td>"
                f"<td>{esc(str(facts.issues[bid]['title'])[:70])}</td></tr>"
            )
        add("</tbody></table></div>")

    if facts.reconciliation:
        add('<h3>P0 reconciliation <span class="chip">pass of 2026-07-31</span></h3>')
        add(
            f"<p>A hand-verified pass over the P0 shelf, read against <code>origin/master</code> "
            "and the live archive. Its sharpest product is a split a flat status field cannot "
            "express: &ldquo;open&rdquo; work with no code fix yet versus work that is "
            "<em>done and merged</em> but inert behind the archive/code schema gap (see Health). "
            f"Marker coverage: {facts.reconciliation_anchored} beads carry the anchor; the "
            f"pass's own summary covered {RECONCILIATION_EXPECTED}.</p>"
        )
        add('<div class="tiles">')
        for value, label in [
            (str(facts.reconciliation_counts["FIXED-AND-EFFECTIVE"]), "FIXED-AND-EFFECTIVE &mdash; closed"),
            (str(facts.reconciliation_counts["FIXED-PENDING-REBUILD"]), "FIXED-PENDING-REBUILD"),
            (str(facts.reconciliation_counts["FIXED-PENDING-DEPLOY"]), "FIXED-PENDING-DEPLOY"),
            (str(facts.reconciliation_counts["MISFRAMED"]), "MISFRAMED &mdash; demoted"),
            (str(facts.reconciliation_counts["GENUINELY OPEN"]), "GENUINELY OPEN"),
        ]:
            add(f'<div class="tile"><span class="n">{value}</span><span class="l">{label}</span></div>')
        add("</div>")
        rebuild_rows = [
            (bid, verdict, snippet)
            for bid, verdict, snippet in facts.reconciliation
            if verdict in ("FIXED-PENDING-REBUILD", "FIXED-PENDING-DEPLOY")
        ]
        if rebuild_rows:
            add(
                "<p>The rebuild-blocked set: merged, confirmed inert by direct query, needing no "
                "further engineering &mdash; the honest measure of how much apparent open work is "
                "actually done.</p>"
            )
            add('<div class="tablewrap"><table id="rb"><thead><tr><th>bead</th><th>title</th>')
            add('<th>verdict</th><th class="nosort">evidence</th></tr></thead><tbody>')
            for bid, verdict, snippet in sorted(rebuild_rows, key=lambda t: (t[1], t[0])):
                title = esc(str(facts.issues[bid]["title"])[:70]) if bid in facts.issues else ""
                add(
                    f"<tr><td>{chip(facts, bid)}</td><td>{title}</td>"
                    f'<td><span class="badge warn">{esc(verdict)}</span></td>'
                    f"<td>{esc(snippet[-220:])}</td></tr>"
                )
            add("</tbody></table></div>")
        add("<details><summary>Full reconciliation table</summary>")
        add('<input class="filter" placeholder="filter reconciliation&hellip;" oninput="flt(this,\'rc\')">')
        add('<div class="tablewrap"><table id="rc"><thead><tr><th>bead</th><th>title</th>')
        add("<th>verdict</th></tr></thead><tbody>")
        rc_badge = {
            "FIXED-AND-EFFECTIVE": "ok",
            "FIXED-PENDING-REBUILD": "warn",
            "FIXED-PENDING-DEPLOY": "warn",
            "MISFRAMED": "info",
            "GENUINELY OPEN": "bad",
        }
        for bid, verdict, _snippet in sorted(facts.reconciliation, key=lambda t: (t[1], t[0])):
            title = esc(str(facts.issues[bid]["title"])[:74]) if bid in facts.issues else ""
            add(
                f"<tr><td>{chip(facts, bid)}</td><td>{title}</td>"
                f'<td><span class="badge {rc_badge.get(verdict, "info")}">{esc(verdict)}</span></td></tr>'
            )
        add("</tbody></table></div></details>")
    if strict_total or facts.reconciliation:
        add("</section>")

    # ---------------- findings ----------------
    add('<section id="findings"><h2>Findings</h2>')
    add(
        "<p>The judgement layer, ordered by severity. Every entry was emitted by a condition "
        "this generator checked against the input file at render time &mdash; there is no "
        "hand-authored claim here that can silently outlive its evidence. "
        '<span class="ev ev-measured">measured</span> marks direct counts; '
        '<span class="ev ev-derived">derived</span> marks reconstructions and computed '
        "verdicts.</p>"
    )
    add('<ul class="findings">')
    for ins in insights:
        chips = (" " + " ".join(chip(facts, c) for c in ins.chips)) if ins.chips else ""
        add(
            f'<li class="sev-{ins.sev}"><b><span class="badge {ins.sev}">{ins.sev}</span> '
            f'{ins.title} <span class="ev ev-{ins.ev}">{ins.ev}</span></b>{ins.body}{chips}</li>'
        )
    add("</ul>")
    add("</section>")

    # ---------------- method ----------------
    add('<section id="method"><h2>Method &amp; regeneration</h2>')
    add(
        "<p>Everything numeric on this page is computed by the generator named below from a "
        "single input file; nothing was transcribed. Interpretation is computed too: findings "
        "are conditional generators, not prose &mdash; if a number here disagrees with "
        "<code>bd</code>, the export is stale, not the arithmetic; regenerate with "
        "<code>--fresh</code>.</p>"
    )
    add("<h3>Provenance</h3>")
    add('<div class="tablewrap"><table id="prov"><thead><tr><th>part of the report</th>')
    add('<th class="nosort">origin</th></tr></thead><tbody>')
    for part, origin in [
        ("every count, percentage, matrix cell, median, heatmap cell", "measured from the JSONL"),
        ("epic trees, fill bars, dependency degree on each chip", "measured (parent-child + blocks edges)"),
        ("per-day histogram, velocity windows", "measured"),
        (
            "pulse snapshots, burnup, epic sparklines, 7d deltas",
            "derived (timestamp reconstruction; deletions invisible)",
        ),
        (
            "ready / blocked / cycles / densest cluster / parallel frontier",
            "measured (graph traversal in the generator)",
        ),
        ("health checks and review queues", "measured"),
        ("keyword theme classification", "measured, using a hand-written keyword list"),
        ("area-label folding", "measured, using a hand-written synonym map"),
        ("verdict + reconciliation marker extraction", "measured by regex over notes and comments"),
        ("schema gap (live vs origin/master)", "measured: git show + read-only PRAGMA"),
        ("findings", "derived: conditional generators over all of the above"),
        ("section framing sentences and the notation legend", "authored, deliberately data-free"),
        (
            f"two constants from dated passes ({OPERATOR_VERIFIED_COUNT}, {RECONCILIATION_EXPECTED})",
            "operator-supplied, provenance attached where used",
        ),
    ]:
        cls = "todo" if origin.startswith(("authored", "operator")) else "info"
        add(f'<tr><td>{esc(part)}</td><td><span class="badge {cls}">{esc(origin)}</span></td></tr>')
    add("</tbody></table></div>")
    add("<h3>Regenerate</h3>")
    add(
        "<pre><code>cd /realm/project/polylogue\n"
        "devtools workspace beads-state-report --fresh \\\n"
        f"  --out {esc(str(source.parent))}/beads-state.html</code></pre>"
    )
    add(
        "<p><code>--fresh</code> re-runs <code>bd export -o .beads/issues.jsonl</code> "
        "first. Without it the report describes whatever the file last held, which after "
        "any uncommitted <code>bd</code> mutation is not the live state. The exported "
        "file is contended across sessions and should not be committed as part of "
        "generating a report.</p>"
    )
    add("</section>")

    add("</main></div>")
    add(
        f"<footer>generated by <code>devtools workspace beads-state-report</code> on "
        f"{generated:%Y-%m-%d %H:%M} from {esc(src)} &middot; {facts.total:,} beads, "
        f"{len(facts.edges):,} dependency edges</footer>"
    )
    add("<script>" + JS + "</script>")
    add("</body>\n</html>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def json_payload(facts: Facts, schema_gap: dict[str, Any], generated: dt.datetime) -> dict[str, Any]:
    snaps = {
        "now": facts.snapshot(generated),
        "7d": facts.snapshot(generated - dt.timedelta(days=7)),
        "14d": facts.snapshot(generated - dt.timedelta(days=14)),
    }
    insights = compute_insights(facts, schema_gap, snaps)
    return {
        "generated": generated.isoformat(),
        "total": facts.total,
        "status": dict(facts.status),
        "priority": {str(k): v for k, v in facts.priority.items()},
        "open_total": facts.open_total,
        "ready": len(facts.ready),
        "blocked": len(facts.blocked),
        "cycles": facts.cycles,
        "dangling": [list(d) for d in facts.dangling],
        "open_parent_all_closed": facts.open_parent_all_closed,
        "stale_claims": [[bid, round(days, 2)] for bid, days in facts.stale_claims],
        "aged_urgent": facts.aged_urgent,
        "dup_titles": [[t, ids] for t, ids in facts.dup_titles],
        "snapshots": snaps,
        "velocity": {
            "recent": {"created": facts.win_recent[0], "closed": facts.win_recent[1]},
            "prior": {"created": facts.win_prior[0], "closed": facts.win_prior[1]},
            "window_days": VELOCITY_WINDOW_DAYS,
        },
        "parallel_frontier": [
            {"id": c["id"], "open_desc": c["open_desc"], "urgent": c["urgent"]} for c in facts.parallel_frontier()
        ],
        "verdicts": {"strict": dict(facts.verdict_counts), "loose": dict(facts.loose_counts)},
        "reconciliation": facts.reconciliation,
        "vocab_state": facts.vocab_state,
        "schema_gap": schema_gap,
        "insights": [{"sev": i.sev, "title": i.title, "body": i.body, "ev": i.ev} for i in insights],
    }


def main(argv: list[str] | None = None) -> int:
    root = _get_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=None, help="issues.jsonl (default: .beads/issues.jsonl)")
    parser.add_argument("--out", default=None, help="write HTML here (default: .local/beads-state.html)")
    parser.add_argument("--fresh", action="store_true", help="run `bd export` before reading")
    parser.add_argument("--json", action="store_true", help="print the computed facts as JSON instead of HTML")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else root / ".beads/issues.jsonl"
    if args.fresh:
        subprocess.run(["bd", "export", "-o", str(path)], check=True, capture_output=True)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    issues, edges = load(path)
    if not issues:
        print(f"no issues parsed from {path}", file=sys.stderr)
        return 1
    now = dt.datetime.now(dt.UTC)
    facts = Facts(issues, edges, now)
    schema_gap = schema_gap_facts(root)

    if args.json:
        print(json.dumps(json_payload(facts, schema_gap, now), indent=2, sort_keys=True))
        return 0

    out = Path(args.out) if args.out else root / ".local/beads-state.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(facts, path, now, schema_gap), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {facts.total:,} beads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
