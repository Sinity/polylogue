"""Verify Beads backlog structure invariants (`.beads/issues.jsonl`).

Background
----------

Backlog structure trails filing unless an invariant lint enforces it: the
2026-07-06 session needed a 41-agent sweep to recover from accumulated drift
(missing acceptance criteria, dangling dependency refs, unlabeled beads,
stale "adopted" decisions left open). This is the backlog equivalent of
`devtools lab policy schema-versioning` / `docs-drift` / `timestamp-doctrine`:
a mechanical check that fails a gate instead of drift silently accumulating
until an archaeology session (polylogue-8jg9.1).

Checks over `.beads/issues.jsonl`:

  D1  no dangling dependency refs
  D2  no dependency cycles among blocks-edges
  H1  open tech-tree bead has a horizon label (frontier/mid/vision)
  H2  horizon:vision => priority P3/P4 (keeps `bd ready` clean)
  H3  open horizon:frontier bead has acceptance criteria (field or notes sidecar)
  H4  open horizon:frontier bead has design content (field, notes, or description with file paths)
  P1  open P0/P1 bead has acceptance criteria
  E1  epic has members: id-prefix children, dep edges, or bead ids named in its text
  E2  epic has a non-empty description (WHY + member map)
  T1  no ephemeral-path ground truth: /realm/inbox/ or /tmp/ cited outside provenance context
  X1  duplicate open titles (exact, case-folded)
  X2  bead id named in an open bead's text does not exist
  R1  READY bead (open, all blocks-deps closed) at P1/P2 lacking AC — the fast-execution gap
  A1  open non-epic bead has at least one area:* label
  B1  open decision-type bead whose text declares Status: adopted/decided should be closed
  S1  the most recent bd JSONL sync receipt (``.cache/bd-sync-receipts/``,
      written by ``devtools/bd_reimport_guard.py``) is missing required
      fields, fails to parse, or reports a conflicted/unauthorized-downgrade
      row — the portfolio gate's consumption of polylogue-gxjh.1's monotonic
      sync receipts (polylogue-8jg9.1). No receipt on disk is not a
      violation (nothing has synced through the guard yet in this
      checkout); a *present but unclean* receipt is.
  F1  an open bead carrying ``metadata.frontier == "active"`` (an admitted
      active leaf) is itself an epic — epics are not work; the epic should
      instead carry ``metadata.frontier_program == "active"`` and its
      member leaves should be the ones admitted.
  F2  an open active leaf (non-epic; F1 already covers epics) has no
      ``frontier_program_ref``, or the ref is dangling, or the referenced
      bead does not itself carry ``metadata.frontier_program == "active"``
      — the leaf cannot be grouped into a valid program.
  F3  an ``in_progress`` bead has had no recorded activity
      (``updated_at``) within the configurable stale-claim window (default
      7 days, ``--stale-claim-days``) — a likely abandoned/session-killed
      claim that should be re-verified or released.
  F4  an open bead marked ``metadata.frontier_program == "active"`` has
      zero open active leaves whose ``frontier_program_ref`` points back
      at it — a program admitted with no admitted members ("program
      grouping derives frontier_program=active from its member leaves",
      polylogue-8jg9.1 AC3).

Three-view policy (polylogue-8jg9.1 AC1): *full ambition* is every open bead
in the export regardless of admission (unaffected by this module — nothing
here demotes or hides a bead); the *active set* is every open non-epic leaf
carrying ``metadata.frontier == "active"`` (F1/F2/F4 police its structural
validity, ``compute_active_set_summary()`` reports its size); *execution
focus* is the smaller ``status=in_progress`` claim-backed subset (F3 polices
claim liveness). Active-set *size* is soft operating guidance, never a hard
cap: ``compute_active_set_summary()`` compares the current active-leaf count
to a configurable target (default 30) and warn threshold (default 50) and
returns informational diagnostics only — it is not part of ``Finding``/
``collect_findings()`` and can never fail the gate, truncate the set, or
hide a bead. Only F1/F2/F3/F4 structural violations are hard findings.

(8jg9.1's design names five conceptual classes: (a) P0/P1 missing AC = P1;
(b) decision-type bead stuck past adopted/decided = B1; (c) no area:* label =
A1; (d) orphan beads with no epic parent — covered by the native `bd orphans`
command, not duplicated here; (e) a blocks-edge pointing at a closed bead —
not representable, since bd computes "blocked" live from dependency status
rather than persisting a blocked flag that could go stale.)

This module supersedes the standalone `.agent/tools/bead-lint.py` script
(same algorithm, ported so the check runs through the standing `devtools
verify --lab` gate instead of requiring a manual invocation). The allowlist
lives at `devtools/data/bead-lint-allow.txt` (format:
`CHECK<TAB>bead-id<TAB>reason` per line; moved from
`.agent/tools/bead-lint-allow.txt` by polylogue-kapb).

S1 is this module's named consumption of `devtools/bd_reimport_guard.py`'s
`SyncReceipt` (polylogue-gxjh.1): the sync layer classifies every merged row
as new/updated/equal/skipped_downgrade/conflicted/recovered_downgrade and
writes a receipt to `.cache/bd-sync-receipts/`; this gate reads the most
recent one and refuses to treat a corrupt, incomplete, conflicted, or
silently-downgraded synchronization as clean, instead of trusting bare `bd`
command exit status. Per 8jg9.1's division of labor, this module does not
reimplement merge/conflict resolution — it only consumes the receipt gxjh.1
already writes.

Wired into ``devtools verify --lab`` alongside the other policy checks, since
this is a repo-hygiene boundary check rather than a per-edit gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from devtools import repo_root as _get_root

_HORIZONS = {"horizon:frontier", "horizon:mid", "horizon:vision"}
_EPHEMERAL_RE = re.compile(r"(/realm/inbox/|(?<![\w.])/tmp/)")
# Provenance-ish context that legitimizes an ephemeral path mention.
_PROVENANCE_HINTS = ("verbatim spec", "preserved as", "provenance", "escrow", "was in /realm/inbox", "corpus")
# A Bead ref cannot be only the prefix of a longer external/request id such as
# ``polylogue-ext-mrhjgnkn``.  The negative lookahead prevents that token from
# being misreported as a dangling ``polylogue-ext`` Bead.
_BEAD_REF_RE = re.compile(r"polylogue-[a-z0-9]+(?:\.[0-9]+)?(?![a-z0-9-])")

_DEFAULT_ISSUES_RELPATH = ".beads/issues.jsonl"
_DEFAULT_ALLOWLIST_RELPATH = "devtools/data/bead-lint-allow.txt"
_DEFAULT_RECEIPTS_RELPATH = ".cache/bd-sync-receipts"

# Active-set soft operating bands (polylogue-8jg9.1 AC1): informational only,
# never a hard cap. See compute_active_set_summary().
_DEFAULT_ACTIVE_TARGET = 30
_DEFAULT_ACTIVE_WARN = 50
# F3 stale-claim window: an in_progress bead with no updated_at activity in
# this many days is flagged for re-verification, not auto-released.
_DEFAULT_STALE_CLAIM_DAYS = 7

# The unclean outcome kinds a SyncReceipt.to_dict() row can carry (see
# devtools/bd_reimport_guard.py:RowOutcome/SyncReceipt). Duplicated here as
# string literals rather than imported, since this module reads the receipt
# purely as data (a downstream consumer, per 8jg9.1/gxjh.1's division of
# labor) and should not depend on gxjh.1's internal dataclasses.
_UNCLEAN_OUTCOMES = {"conflicted", "skipped_downgrade"}


@dataclass(frozen=True, slots=True)
class Finding:
    check: str
    bead_id: str
    message: str


def _load(path: Path) -> tuple[dict[str, dict[str, object]], list[tuple[str, str, str]]]:
    """Parse the exported jsonl snapshot into an in-memory issue/dep index.

    Bounded-enumeration note (polylogue-8jg9.1 remaining scope item 2): this
    reads one already-exported, finite local file (``.beads/issues.jsonl``,
    ~1,100 rows / ~5 MB at present backlog scale) with a single
    ``read_text().splitlines()`` pass, never a live, potentially-unbounded
    `bd list`/`bd ready` query. That distinction matters — the 2026-07-15
    incident recorded on this bead was a read-only `bd list --status open
    --limit 500` process reaching 7.88 GB RSS + 20.3 GB swap against the live
    Dolt-backed `bd` server, not against this exported-file path. Every
    check in this module (including the new F1-F4 active-set/execution-focus
    checks below) is derived from this same bounded, already-materialized
    dict — none of them shell out to `bd` per-issue or re-query a live
    unbounded source. If the exported jsonl itself grows to a size where a
    single-pass in-memory dict is no longer bounded for realistic backlog
    scale, that is the trigger to switch this loader to a streaming/paged
    parse; at ~5 MB for ~1,100 issues it is not currently warranted.
    """
    issues: dict[str, dict[str, object]] = {}
    deps: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("_type") == "issue":
            issues[d["id"]] = d
            for dep in d.get("dependencies") or []:
                deps.append((d["id"], dep.get("depends_on_id"), dep.get("type", "blocks")))
        elif d.get("_type") == "dependency":
            deps.append((d.get("issue_id"), d.get("depends_on_id"), d.get("type", "blocks")))
    return issues, deps


def _text_of(d: dict[str, object]) -> str:
    return " ".join(str(d.get(k) or "") for k in ("description", "design", "acceptance_criteria", "notes"))


def _has_ac(d: dict[str, object]) -> bool:
    if str(d.get("acceptance_criteria") or "").strip():
        return True
    notes = str(d.get("notes") or "").lower()
    return "acceptance" in notes or "verify:" in notes or "ac:" in notes


def _has_design(d: dict[str, object]) -> bool:
    if str(d.get("design") or "").strip():
        return True
    blob = str(d.get("description") or "") + str(d.get("notes") or "")
    # A description that names concrete code surfaces counts as design-bearing.
    return bool(re.search(r"\w+\.py|\w+/\w+\.|::|polylogue/", blob))


def _labels_of(d: dict[str, object]) -> set[str]:
    raw = d.get("labels") or []
    return {str(lab) for lab in raw} if isinstance(raw, list) else set()


def _priority_of(d: dict[str, object]) -> int:
    prio = d.get("priority", 2)
    return int(prio) if isinstance(prio, int | float | str) and str(prio).lstrip("-").isdigit() else 2


def _metadata_of(d: dict[str, object]) -> dict[str, object]:
    raw = d.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _is_active_leaf(d: dict[str, object]) -> bool:
    return _metadata_of(d).get("frontier") == "active"


def _is_active_program(d: dict[str, object]) -> bool:
    return _metadata_of(d).get("frontier_program") == "active"


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_allowlist(allow_path: Path) -> set[tuple[str, str]]:
    allow: set[tuple[str, str]] = set()
    if allow_path.exists():
        for line in allow_path.read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                allow.add((parts[0], parts[1]))
    return allow


def _latest_receipt_path(receipts_dir: Path) -> Path | None:
    if not receipts_dir.is_dir():
        return None
    # Receipt filenames are `<created_at-token><-source>.json` with the
    # created_at token stripped of `:`/`-`, so lexicographic sort of
    # basenames is chronological (see bd_reimport_guard.write_receipt).
    candidates = sorted(p for p in receipts_dir.glob("*.json") if p.is_file())
    return candidates[-1] if candidates else None


def _check_sync_receipt(receipts_dir: Path, add: Callable[[str, str, str], None]) -> None:
    """S1: the most recent bd sync receipt, if any, must be clean.

    No receipt on disk is not a finding — `.cache/` is disposable local
    state and nothing may have synced through the guard yet in this
    checkout. A *present* receipt that fails to parse, is missing required
    fields, or reports a conflicted/unauthorized-downgrade row is a hard
    finding: it means the last JSONL synchronization was not proven clean,
    and bare command exit status cannot be trusted instead (polylogue-8jg9.1
    consuming polylogue-gxjh.1's SyncReceipt contract).
    """
    latest = _latest_receipt_path(receipts_dir)
    if latest is None:
        return

    try:
        payload = json.loads(latest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add("S1", "<sync-receipt>", f"corrupt sync receipt {latest.name}: {exc}")
        return

    if not isinstance(payload, dict) or "is_clean" not in payload or "outcomes" not in payload:
        add("S1", "<sync-receipt>", f"incomplete sync receipt {latest.name}: missing is_clean/outcomes")
        return

    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        add("S1", "<sync-receipt>", f"incomplete sync receipt {latest.name}: outcomes is not a list")
        return

    if payload.get("is_clean") is True:
        return

    unclean_rows = [o for o in outcomes if isinstance(o, dict) and o.get("outcome") in _UNCLEAN_OUTCOMES]
    if not unclean_rows:
        # is_clean is False but no row explains why (schema drift on gxjh.1's
        # side, or a payload we don't fully understand) -- still a finding,
        # since a receipt claiming uncleanliness must be actionable, not silent.
        add("S1", "<sync-receipt>", f"sync receipt {latest.name} reports is_clean=false with no explanatory rows")
        return

    for row in unclean_rows:
        bead_id = str(row.get("id") or "<unknown>")
        kind = row.get("outcome")
        current_rev = row.get("current_revision")
        candidate_rev = row.get("candidate_revision")
        if kind == "conflicted":
            add(
                "S1",
                bead_id,
                f"sync receipt {latest.name}: incomparable/conflicted row "
                f"(current={current_rev!r}, candidate={candidate_rev!r})",
            )
        else:  # skipped_downgrade
            add(
                "S1",
                bead_id,
                f"sync receipt {latest.name}: unauthorized downgrade skipped "
                f"(current={current_rev!r}, candidate={candidate_rev!r}) -- "
                "recover explicitly via `bd_reimport_guard.py reconcile --allow-downgrade` "
                "with an actor/reason if this candidate should win",
            )


def collect_findings(
    path: Path | None = None,
    allow_path: Path | None = None,
    checks: set[str] | None = None,
    receipts_path: Path | None = None,
    stale_claim_days: int = _DEFAULT_STALE_CLAIM_DAYS,
    now: datetime | None = None,
) -> list[Finding]:
    """Run all 20 backlog-hygiene checks against a Beads jsonl export.

    ``path`` defaults to ``.beads/issues.jsonl`` under the repo root;
    ``allow_path`` defaults to ``devtools/data/bead-lint-allow.txt``;
    ``receipts_path`` (S1) defaults to ``.cache/bd-sync-receipts/`` under the
    repo root. ``stale_claim_days`` (F3) defaults to
    ``_DEFAULT_STALE_CLAIM_DAYS``; ``now`` (F3) defaults to the current UTC
    time and exists so tests can pin a fixed clock instead of the host wall
    clock.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    root = _get_root()
    if path is None:
        path = root / _DEFAULT_ISSUES_RELPATH
    if allow_path is None:
        allow_path = root / _DEFAULT_ALLOWLIST_RELPATH
    if receipts_path is None:
        receipts_path = root / _DEFAULT_RECEIPTS_RELPATH
    allow = _load_allowlist(allow_path)

    issues, deps = _load(path)
    open_ids = {i for i, d in issues.items() if d.get("status") in ("open", "in_progress")}
    findings: list[Finding] = []

    def add(check: str, bid: str, msg: str) -> None:
        if (checks is None or check in checks) and (check, bid) not in allow:
            findings.append(Finding(check, bid, msg))

    _check_sync_receipt(receipts_path, add)

    # D1 dangling deps
    for src, dst, typ in deps:
        if dst not in issues:
            add("D1", src, f"dangling dep -> {dst} ({typ})")

    # D2 cycles among blocks deps (open issues only)
    graph: dict[str, set[str]] = defaultdict(set)
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            graph[src].add(dst)
    white, gray, black = 0, 1, 2
    color: dict[str, int] = defaultdict(int)

    def dfs(n: str, stack: list[str]) -> None:
        color[n] = gray
        for m in graph[n]:
            if color[m] == gray:
                cyc = stack[stack.index(m) :] + [m] if m in stack else [n, m]
                add("D2", n, "blocks-cycle: " + " -> ".join(cyc))
            elif color[m] == white:
                dfs(m, stack + [m])
        color[n] = black

    for n in list(graph):
        if color[n] == white:
            dfs(n, [n])

    # F1/F2/F4: active-set (frontier=active) and program-grouping structure.
    # active_leaf_ids are open beads admitted into the active set; epics among
    # them are F1 violations (epics are not work) rather than valid leaves.
    active_leaf_ids = [i for i in open_ids if _is_active_leaf(issues[i])]
    # Reverse index: program id -> admitted non-epic leaves referencing it,
    # used by F4 to check a program's frontier_program=active claim is backed
    # by at least one member (polylogue-8jg9.1 AC3: program grouping is
    # derived from member leaves, not asserted independently of them).
    program_leaves: dict[str, list[str]] = defaultdict(list)
    for i in active_leaf_ids:
        d = issues[i]
        if d.get("issue_type") == "epic":
            add(
                "F1",
                i,
                "epic carries frontier=active as a leaf (epics are not work; mark the epic "
                "frontier_program=active instead and admit its member leaves)",
            )
            continue
        ref = _metadata_of(d).get("frontier_program_ref")
        if not isinstance(ref, str) or not ref:
            add("F2", i, "active leaf has no frontier_program_ref (cannot be grouped into a program)")
        elif ref not in issues:
            add("F2", i, f"active leaf's frontier_program_ref -> {ref} does not exist")
        elif not _is_active_program(issues[ref]):
            add("F2", i, f"active leaf's frontier_program_ref -> {ref} is not itself frontier_program=active")
        else:
            program_leaves[ref].append(i)

    for i in open_ids:
        d = issues[i]
        if _is_active_program(d) and not program_leaves.get(i):
            add(
                "F4",
                i,
                "frontier_program=active with no open active leaf referencing it via "
                "frontier_program_ref (stale program admission -- derive frontier_program=active "
                "from member leaves, not the reverse)",
            )

    # F3: stale in_progress claims (no recorded updated_at activity within
    # the configured window). Defined purely from bd's own updated_at field
    # -- bd does not persist a separate "last activity" timestamp, and
    # updated_at already advances on every field/status/note mutation.
    for i, d in issues.items():
        if d.get("status") != "in_progress":
            continue
        updated = _parse_timestamp(d.get("updated_at"))
        if updated is None:
            continue
        age_days = (now - updated).total_seconds() / 86400
        if age_days > stale_claim_days:
            add(
                "F3",
                i,
                f"in_progress with no recorded activity for {age_days:.1f}d "
                f"(> {stale_claim_days}d stale-claim threshold) -- re-verify claim liveness or release it",
            )

    # Per-issue checks.
    titles: dict[str, list[str]] = defaultdict(list)
    children: dict[str, int] = defaultdict(int)
    for i in issues:
        if "." in i.removeprefix("polylogue-"):
            children[i.rsplit(".", 1)[0]] += 1
    dep_touch: dict[str, int] = defaultdict(int)  # epics may group members via dep edges instead of id-prefix
    for src, dst, _typ in deps:
        dep_touch[src] += 1
        dep_touch[dst] += 1

    for i, d in issues.items():
        if d.get("status") not in ("open", "in_progress"):
            continue
        labels = _labels_of(d)
        prio = _priority_of(d)
        horizon = labels & _HORIZONS
        titles[str(d.get("title", "")).strip().casefold()].append(i)

        if "tech-tree" in labels and not horizon:
            add("H1", i, "tech-tree bead without horizon label")
        if "horizon:vision" in labels and prio < 3:
            add("H2", i, f"vision bead at P{prio} (should be P3/P4)")
        if "horizon:frontier" in labels and not _has_ac(d):
            add("H3", i, "frontier bead without acceptance criteria")
        if "horizon:frontier" in labels and not _has_design(d):
            add("H4", i, "frontier bead without design content")
        if prio <= 1 and d.get("issue_type") != "epic" and not _has_ac(d):
            add("P1", i, f"P{prio} bead without acceptance criteria")
        if d.get("issue_type") != "epic" and not any(lab.startswith("area:") for lab in labels):
            add("A1", i, "open non-epic bead without an area:* label")
        if d.get("issue_type") == "decision" and re.search(r"status:\s*(adopted|decided)", _text_of(d), re.IGNORECASE):
            add("B1", i, "decision bead declares adopted/decided but is still open")
        if d.get("issue_type") == "epic":
            named_members = [r for r in _BEAD_REF_RE.findall(_text_of(d)) if r != i and r in issues]
            if children[i] == 0 and dep_touch[i] == 0 and not named_members:
                add("E1", i, "epic with no members (no children, no dep edges, no named bead ids)")
            if not str(d.get("description") or "").strip():
                add("E2", i, "epic without description")
        blob = _text_of(d)
        for ref in set(_BEAD_REF_RE.findall(blob)):
            token = ref.removeprefix("polylogue-").split(".", 1)[0]
            # id-shaped tokens only: pure-alpha words >=4 chars are English compounds
            # ("polylogue-substrate intake"); pure-numeric are #N-style refs.
            if token.isalpha() and len(token) >= 4:
                continue
            if token.isdigit():
                continue
            # Tolerate .N suffix references to a future child of an existing bead.
            if ref not in issues and ref.rsplit(".", 1)[0] not in issues:
                add("X2", i, f"names nonexistent bead {ref}")
        if _EPHEMERAL_RE.search(blob):
            low = blob.lower()
            if not any(h in low for h in _PROVENANCE_HINTS):
                add("T1", i, "ephemeral path (/realm/inbox or /tmp) cited without provenance framing")

    for _t, ids in titles.items():
        if _t and len(ids) > 1:
            for i in ids:
                add("X1", i, f"duplicate open title with {[x for x in ids if x != i]}")

    # R1 ready-queue executable check.
    blocked: set[str] = set()
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            blocked.add(src)
    for i in sorted(open_ids - blocked):
        d = issues[i]
        if _priority_of(d) <= 2 and d.get("issue_type") not in ("epic",) and not _has_ac(d):
            add("R1", i, f"READY P{_priority_of(d)} bead without AC (cold agent cannot execute fast)")

    return findings


@dataclass(frozen=True, slots=True)
class ActiveSetSummary:
    """Soft-band size report over the active set (polylogue-8jg9.1 AC1).

    Never a `Finding`: exceeding ``target``/``warn`` is informational
    guidance, not a structural violation, a truncation, or a gate failure.
    """

    active_leaf_count: int
    target: int
    warn: int
    band: str
    programs: dict[str, int]
    diagnostics: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "active_leaf_count": self.active_leaf_count,
            "target": self.target,
            "warn": self.warn,
            "band": self.band,
            "programs": self.programs,
            "diagnostics": self.diagnostics,
        }


def compute_active_set_summary(
    path: Path | None = None,
    *,
    target: int = _DEFAULT_ACTIVE_TARGET,
    warn: int = _DEFAULT_ACTIVE_WARN,
) -> ActiveSetSummary:
    """Report active-set size against soft target/warn bands.

    Reads the same bounded, already-exported jsonl as `collect_findings`
    (see `_load`'s bounded-enumeration note) -- no additional `bd` query.
    Non-epic open leaves only (epics-as-leaves are F1 findings, not counted
    here as legitimate active work).
    """
    root = _get_root()
    if path is None:
        path = root / _DEFAULT_ISSUES_RELPATH
    issues, _deps = _load(path)
    open_ids = {i for i, d in issues.items() if d.get("status") in ("open", "in_progress")}
    active_leaf_ids = [i for i in open_ids if _is_active_leaf(issues[i]) and issues[i].get("issue_type") != "epic"]

    programs: dict[str, int] = defaultdict(int)
    for i in active_leaf_ids:
        ref = _metadata_of(issues[i]).get("frontier_program_ref")
        programs[str(ref)] += 1

    count = len(active_leaf_ids)
    diagnostics: list[str] = []
    if count > warn:
        band = "above-warn"
        diagnostics.append(
            f"{count} active leaves exceeds the soft warn threshold of {warn}. This is a diagnostic, "
            "not a failure -- review whether the growth is explained (a new program admitted, a "
            "reconciliation sweep) or is unexplained backlog drift worth pruning."
        )
    elif count > target:
        band = "above-target"
        diagnostics.append(
            f"{count} active leaves exceeds the soft target of {target}. Informational only; no action "
            f"required unless growth continues unexplained toward the warn threshold of {warn}."
        )
    else:
        band = "within-target"
    return ActiveSetSummary(
        active_leaf_count=count, target=target, warn=warn, band=band, programs=dict(programs), diagnostics=diagnostics
    )


def _format_report(findings: list[Finding], *, issues_scanned: int) -> str:
    if not findings:
        return f"backlog hygiene: zero unhandled findings across {issues_scanned} issues scanned."
    by: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by[f.check].append(f)
    lines: list[str] = []
    for check in sorted(by):
        lines.append(f"[{check}] {len(by[check])} finding(s)")
        for f in by[check]:
            lines.append(f"    {f.bead_id}: {f.message}")
    lines.append(f"\n{len(findings)} finding(s) across {len(by)} check(s); {issues_scanned} issues scanned.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--checks",
        type=lambda value: {item.strip() for item in value.split(",") if item.strip()},
        default=None,
        help="run only the named comma-separated checks (for example D1,D2)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="run `bd export -o <path>` first (bd updates do not immediately re-export the jsonl)",
    )
    parser.add_argument(
        "--stale-claim-days",
        type=int,
        default=_DEFAULT_STALE_CLAIM_DAYS,
        help=f"F3 stale-claim window in days (default {_DEFAULT_STALE_CLAIM_DAYS})",
    )
    parser.add_argument(
        "--active-target",
        type=int,
        default=_DEFAULT_ACTIVE_TARGET,
        help=f"active-set soft target leaf count, informational only (default {_DEFAULT_ACTIVE_TARGET})",
    )
    parser.add_argument(
        "--active-warn",
        type=int,
        default=_DEFAULT_ACTIVE_WARN,
        help=f"active-set soft warn leaf count, informational only (default {_DEFAULT_ACTIVE_WARN})",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="path to issues.jsonl (default: .beads/issues.jsonl under the repo root)",
    )
    args = parser.parse_args(argv)

    root = _get_root()
    path = Path(args.path) if args.path else root / _DEFAULT_ISSUES_RELPATH

    if args.fresh:
        subprocess.run(["bd", "export", "-o", str(path)], check=True, capture_output=True)

    if not path.exists():
        message = f"backlog hygiene: {path} does not exist (no Beads workspace to check)."
        if args.json:
            print(json.dumps({"ok": True, "findings": [], "issues_scanned": 0, "skipped": message}, indent=2))
        else:
            print(message)
        return 0

    findings = collect_findings(
        path=path,
        allow_path=root / _DEFAULT_ALLOWLIST_RELPATH,
        checks=args.checks,
        stale_claim_days=args.stale_claim_days,
    )
    issues_scanned = len(_load(path)[0])
    active_set = compute_active_set_summary(path=path, target=args.active_target, warn=args.active_warn)

    if args.json:
        payload = {
            "ok": not findings,
            "issues_scanned": issues_scanned,
            "findings": [{"check": f.check, "id": f.bead_id, "msg": f.message} for f in findings],
            "active_set": active_set.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(findings, issues_scanned=issues_scanned))
        print(
            f"\nactive set: {active_set.active_leaf_count} leaf/leaves "
            f"(target~{active_set.target}, warn~{active_set.warn}, band={active_set.band}) "
            f"across {len(active_set.programs)} program(s)"
        )
        for diag in active_set.diagnostics:
            print(f"  note: {diag}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
