"""Run continuity scenarios, work-effect reconciliation, and query discovery.

This module wires three executable surfaces without reimplementing them:

- :func:`check_discovery_coverage` reuses the real
  :data:`polylogue.archive.query.discovery.QUERY_DISCOVERY_EXAMPLES` catalog
  to prove every ``query``-tool route step any continuity scenario executes
  has a declared positive example of the same unit-source/route shape --
  i.e. a cold model relying on discovery alone could have found that plan
  family, not just executed it once the runner already knew it.
- :func:`build_repository_claim_graph` and :func:`run_work_evidence_effect_proof`
  reuse the real, production :mod:`polylogue.insights.work_effects` adapters
  (``GitCommitEffectAdapter``, ``BeadsIssueEffectAdapter``,
  ``GitHubPullRequestEffectAdapter``) against *this repository's own* git
  history and committed ``.beads/interactions.jsonl`` ledger -- real,
  full-scale, and privacy-safe because both are already public/committed
  project artifacts, not private chat archive content. Claims are built
  independently from the same ledger's "issue closed" transitions, so the
  effect adapters are proving something over real evidence, not reconciling
  a graph against its own construction.
- :func:`run_mandate_continuity_replay` calls
  :func:`devtools.continuity_replay.replay_archive` unmodified against either
  a supplied archive root (an authorized live-scale replay) or a freshly
  seeded synthetic corpus (the default, privacy-safe CI lane), then combines
  all three executable lanes into one JSON
  artifact. :func:`redact_report` strips raw evidence prose from that
  artifact (keeping refs/hashes/counts) for the live-archive lane; the
  synthetic lane never touches private content so redaction there is a no-op
  proof of the same mechanism, not a load-bearing privacy boundary.

The default synthetic run does not claim to replay a private production
incident. Supplying an authorized archive root runs the same mechanism over
that archive. Mutation coverage remains owned by the continuity replay tests,
not by a duplicated status matrix in this report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TextIO, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.continuity_replay import replay_archive
from polylogue.archive.artifact_taxonomy.support import looks_like_beads_interaction
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.core.refs import ObjectRef
from polylogue.insights.work_effects import (
    DEFAULT_WORK_ITEM_ID_PATTERN,
    BeadsIssueEffectAdapter,
    GitCommitEffectAdapter,
    GitHubPullRequestEffectAdapter,
    RepositoryEffectAdapter,
    collect_repository_effects,
    derive_direct_identifier_judgments,
)
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.insights.work_reconciliation import reconcile_work_effects
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS, ContinuityScenarioSpec, continuity_scenario
from tests.infra.continuity import load_continuity_catalog, seed_continuity_archive

#: Repo root -- this file lives at ``devtools/mandate_continuity_replay.py``.
DEFAULT_REPO_PATH = Path(__file__).resolve().parents[1]
DEFAULT_BEADS_LEDGER_RELATIVE = Path(".beads") / "interactions.jsonl"
_GITHUB_REPO_SLUG = "Sinity/polylogue"

# ── Discovery-coverage lane ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DiscoveryCoverageGap:
    """One continuity route step whose plan family has no discovery example."""

    scenario_id: str
    step_id: str
    plan_atom: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscoveryCoverageReport:
    """Whether every continuity-scenario query plan is independently discoverable."""

    checked_steps: int
    covered_steps: int
    gaps: tuple[DiscoveryCoverageGap, ...]

    @property
    def status(self) -> Literal["pass", "fail"]:
        return "pass" if not self.gaps else "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checked_steps": self.checked_steps,
            "covered_steps": self.covered_steps,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def check_discovery_coverage(
    scenarios: Sequence[ContinuityScenarioSpec],
) -> DiscoveryCoverageReport:
    """Prove every ``query``-tool continuity route step is independently discoverable.

    A continuity scenario's route steps prove the runner *can execute* a
    plan; they say nothing about whether a cold model, given only the
    published query-discovery catalog (``archive/query/discovery.py``), could
    have *formulated* that same plan family without hidden knowledge. This
    cross-checks each ``query``-tool step's unit-source against
    ``QUERY_DISCOVERY_EXAMPLES`` -- the same catalog z9gh.3 generates MCP
    schemas/completions from -- so a shipped scenario whose plan family the
    discovery catalog does not teach shows up as a named gap rather than a
    silent success.
    """

    catalog_atoms = {f"query:{example.unit_source}" for example in QUERY_DISCOVERY_EXAMPLES if example.route == "query"}
    gaps: list[DiscoveryCoverageGap] = []
    checked = 0
    for scenario in scenarios:
        for step in scenario.route_steps:
            if step.tool != "query":
                continue
            checked += 1
            atom = step.plan_atom
            if atom not in catalog_atoms:
                gaps.append(
                    DiscoveryCoverageGap(
                        scenario_id=scenario.scenario_id,
                        step_id=step.step_id,
                        plan_atom=atom,
                        reason=f"no declared query-discovery example teaches {atom!r}",
                    )
                )
    return DiscoveryCoverageReport(checked_steps=checked, covered_steps=checked - len(gaps), gaps=tuple(gaps))


# ── Work-evidence effect-reconciliation lane (this repo's own git+Beads) ──


def _file_identity(path: Path) -> str:
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]


def build_repository_claim_graph(
    jsonl_path: Path,
    *,
    graph_id: str = "mandate-repository-claims",
    id_pattern: re.Pattern[str] = DEFAULT_WORK_ITEM_ID_PATTERN,
) -> WorkEvidenceGraph:
    """Build one claim node per Beads issue independently observed as closed.

    Reads the *same* interaction ledger :class:`BeadsIssueEffectAdapter` reads
    as an effect source, but here as an independent claim source: "issue X
    was closed" is a claim about work completion, distinct from whether
    observed git/PR/Beads evidence actually supports it. Every claim's own
    identity is the issue id; the reconciliation lane below never treats this
    ledger's row as its own confirming effect for the same interaction --
    corroboration must come from an independently matched effect (typically a
    git commit citing the same issue id, or a distinct Beads interaction).
    """

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Beads interaction ledger not found: {jsonl_path}")
    snapshot_ref = ObjectRef(kind="context-snapshot", object_id=f"beads-claims:{_file_identity(jsonl_path)}")
    nodes: dict[str, WorkEvidenceNode] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not looks_like_beads_interaction(record):
            continue
        if str(record.get("kind")) != "field_change":
            continue
        extra = record.get("extra")
        if not isinstance(extra, dict) or extra.get("field") != "status" or extra.get("new_value") != "closed":
            continue
        issue_id = str(record["issue_id"])
        if not id_pattern.fullmatch(issue_id):
            continue
        ref = ObjectRef(kind="work-claim", object_id=f"claimed-closed:{issue_id}")
        if ref.format() in nodes:
            continue
        nodes[ref.format()] = WorkEvidenceNode(
            ref=ref,
            kind="claim",
            label=f"{issue_id} claimed closed",
            claim_text=f"Beads issue {issue_id} was recorded as closed.",
            evidence_refs=(ObjectRef(kind="artifact", object_id=f"beads-interaction:{issue_id}:{record['id']}"),),
            corpus_snapshot_ref=snapshot_ref,
            authority="operator",
            confidence=1.0,
        )
    return WorkEvidenceGraph(graph_id=graph_id, corpus_snapshot_ref=snapshot_ref, nodes=tuple(nodes.values()), edges=())


@dataclass(frozen=True, slots=True)
class WorkEvidenceEffectProof:
    """Quantified result of reconciling repository claims against real effects."""

    graph_id: str
    claims_total: int
    claims_evaluated: int
    claims_unevaluated: int
    effect_count_by_authority: dict[str, int]
    judgment_count_by_evaluation: dict[str, int]
    adapter_failures: tuple[dict[str, str], ...]

    @property
    def status(self) -> Literal["pass", "fail"]:
        # A live-scale proof over a real, non-empty ledger must actually
        # evaluate at least one claim through a real effect adapter, or the
        # wiring has silently stopped matching anything.
        return "pass" if self.claims_total > 0 and self.claims_evaluated > 0 else "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "claims_total": self.claims_total,
            "claims_evaluated": self.claims_evaluated,
            "claims_unevaluated": self.claims_unevaluated,
            "effect_count_by_authority": dict(self.effect_count_by_authority),
            "judgment_count_by_evaluation": dict(self.judgment_count_by_evaluation),
            "adapter_failures": [dict(failure) for failure in self.adapter_failures],
            "status": self.status,
        }


def run_work_evidence_effect_proof(
    *,
    repo_path: Path,
    beads_ledger_path: Path,
    since_ms: int | None = None,
    until_ms: int | None = None,
    adapters: Sequence[RepositoryEffectAdapter] | None = None,
) -> WorkEvidenceEffectProof:
    """Reconcile real repository claims against real git/GitHub/Beads effects.

    Runs the production adapters from ``polylogue.insights.work_effects``
    against this checkout's own git history and Beads ledger -- real,
    full-repository scale, and privacy-safe because both are already public,
    committed project artifacts. The GitHub adapter is included and is
    expected to fail explicitly (no ``gh``/network assumption here): that
    failure is itself the honest "cites ... with uncertainty" PR-evidence
    citation the mandate AC asks for, not an omission.
    """

    graph = build_repository_claim_graph(beads_ledger_path)
    resolved_adapters: Sequence[RepositoryEffectAdapter] = adapters or (
        GitCommitEffectAdapter(repo_path=repo_path),
        BeadsIssueEffectAdapter(jsonl_path=beads_ledger_path),
        GitHubPullRequestEffectAdapter(repo=_GITHUB_REPO_SLUG),
    )
    collection = collect_repository_effects(resolved_adapters, since_ms=since_ms, until_ms=until_ms)
    judgments = derive_direct_identifier_judgments(graph, collection.effects)
    reconciled = reconcile_work_effects(graph, effects=collection.effects, judgments=judgments)

    claim_refs = {node.ref.format() for node in graph.nodes if node.kind == "claim"}
    evaluated_claim_refs = {edge.source_ref.format() for edge in reconciled.edges if edge.kind == "claimed"}
    evaluated = claim_refs & evaluated_claim_refs

    return WorkEvidenceEffectProof(
        graph_id=graph.graph_id,
        claims_total=len(claim_refs),
        claims_evaluated=len(evaluated),
        claims_unevaluated=len(claim_refs - evaluated),
        effect_count_by_authority=dict(sorted(Counter(effect.authority for effect in collection.effects).items())),
        judgment_count_by_evaluation=dict(sorted(Counter(judgment.evaluation for judgment in judgments).items())),
        adapter_failures=tuple(
            {"authority": failure.authority, "reason": failure.reason} for failure in collection.unavailable
        ),
    )


# ── Redaction ─────────────────────────────────────────────────────────

_REDACTABLE_KEYS = frozenset({"label", "claim_text", "reason", "response_sha256"})


def _redact_value(key: str, value: JSONValue) -> JSONValue:
    if key in _REDACTABLE_KEYS and isinstance(value, str) and value:
        return f"redacted:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    return value


def redact_report(document: JSONValue) -> JSONValue:
    """Strip raw evidence prose from a mandate report, keeping refs/counts/hashes.

    Recursively walks the report replacing any string value stored under a
    label/claim-text/reason-shaped key with a stable hash of itself. Refs,
    ids, statuses, and counts (everything the mandate AC actually needs
    cited) pass through unchanged -- only free-text prose that could carry
    private archive content is hashed.
    """

    if isinstance(document, dict):
        return {key: _redact_value(key, redact_report(value)) for key, value in document.items()}
    if isinstance(document, list):
        return [redact_report(item) for item in document]
    return document


# Executable continuity lanes


# ── Orchestration ──────────────────────────────────────────────────────


async def run_mandate_continuity_replay(
    *,
    archive_root: Path | None = None,
    repo_path: Path = DEFAULT_REPO_PATH,
    beads_ledger_path: Path | None = None,
    scenario_names: Sequence[str] | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    redact: bool = True,
    keep_archive: bool = False,
) -> JSONDocument:
    """Run the continuity, discovery, and work-evidence lanes as one artifact.

    When ``archive_root`` is ``None`` (the default), a fresh, privacy-safe
    synthetic continuity corpus is seeded and torn down automatically -- the
    CI/deterministic lane. Passing an authorized live archive root runs the
    identical mechanism against it (the live-scale lane); pair that with
    ``redact=True`` (the default) so evidence prose never leaves this
    process's stdout/artifact file.
    """

    beads_ledger_path = beads_ledger_path or (repo_path / DEFAULT_BEADS_LEDGER_RELATIVE)
    started_ns = time.perf_counter_ns()
    catalog = load_continuity_catalog()
    live_archive = archive_root is not None
    workdir: TemporaryDirectory[str] | None = None

    resolved_root: Path
    if archive_root is None:
        workdir = TemporaryDirectory(prefix="continuity-replay-")
        resolved_root = Path(workdir.name) / "archive"
        seed_continuity_archive(resolved_root, catalog=catalog)
    else:
        resolved_root = archive_root

    try:
        continuity_report = await replay_archive(resolved_root, catalog, scenario_names=scenario_names)
    finally:
        if workdir is not None and not keep_archive:
            workdir.cleanup()

    scenarios = (
        CONTINUITY_SCENARIOS if scenario_names is None else tuple(continuity_scenario(name) for name in scenario_names)
    )
    discovery_report = check_discovery_coverage(scenarios)
    effect_proof = run_work_evidence_effect_proof(
        repo_path=repo_path,
        beads_ledger_path=beads_ledger_path,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    lane_statuses = (
        str(continuity_report.get("status")),
        discovery_report.status,
        effect_proof.status,
    )
    overall_status: Literal["pass", "fail"] = "pass" if all(status == "pass" for status in lane_statuses) else "fail"
    report: dict[str, object] = {
        "schema_version": 1,
        "live_archive": live_archive,
        "archive_root": str(resolved_root.resolve()) if keep_archive or live_archive else None,
        "elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "status": overall_status,
        "continuity": continuity_report,
        "discovery_coverage": discovery_report.to_dict(),
        "work_evidence_effect_proof": effect_proof.to_dict(),
    }
    document = require_json_document(report, context="mandate continuity replay report")
    return cast(JSONDocument, redact_report(document)) if redact else document


def _scenario_names(value: str) -> tuple[str, ...] | None:
    if value == "all":
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Authorized live archive to replay against; omit for the default synthetic CI lane.",
    )
    parser.add_argument("--repo-path", type=Path, default=DEFAULT_REPO_PATH)
    parser.add_argument("--beads-ledger", type=Path, default=None)
    parser.add_argument("--scenario", default="all", help="all or a comma-separated scenario id list")
    parser.add_argument("--since-ms", type=int, default=None)
    parser.add_argument("--until-ms", type=int, default=None)
    parser.add_argument("--no-redact", action="store_true", help="Disable evidence redaction (CI/synthetic lane only)")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = asyncio.run(
        run_mandate_continuity_replay(
            archive_root=args.archive_root,
            repo_path=args.repo_path,
            beads_ledger_path=args.beads_ledger,
            scenario_names=_scenario_names(args.scenario),
            since_ms=args.since_ms,
            until_ms=args.until_ms,
            redact=not args.no_redact,
            keep_archive=args.keep_archive,
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    out = stdout or sys.stdout
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, file=out)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BEADS_LEDGER_RELATIVE",
    "DEFAULT_REPO_PATH",
    "DiscoveryCoverageGap",
    "DiscoveryCoverageReport",
    "WorkEvidenceEffectProof",
    "build_repository_claim_graph",
    "check_discovery_coverage",
    "main",
    "redact_report",
    "run_mandate_continuity_replay",
    "run_work_evidence_effect_proof",
]
