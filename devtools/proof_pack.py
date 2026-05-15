"""Diff-shaped proof-pack report for PR confidence.

Consumes the assurance-domain manifests, proof catalog, and affected-obligation
routing to produce an operator-facing confidence answer.

The report classifies every affected obligation by evidence taxonomy so agents
can distinguish executable checks from static declarations, metadata
completeness assertions, and advisory noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root
from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.catalog import (
    VerificationCatalog,
    build_verification_catalog,
)
from polylogue.proof.diffing import (
    AffectedObligation,
    RecommendedCheck,
    build_affected_obligation_report,
    changed_paths_between_refs,
    obligation_ids_for_ref,
)
from polylogue.proof.models import EvidenceTaxonomy, RunnerBinding, SubjectRef

# Taxonomy human-facing descriptions and actionability guidance.
_TAXONOMY_DESCRIPTIONS: dict[EvidenceTaxonomy, str] = {
    "executable_behavior": "Gates backed by executable checks (tests, probes, smoke runs).",
    "observability": "Runtime trace or diagnostic observability evidence.",
    "architectural_static": "Static boundary checks (topology, layering, file budgets).",
    "metadata_spec": "Metadata and specification completeness -- does not block by itself.",
    "manual_review": "Requires human agent judgment artifact.",
    "advisory": "Advisory / noise -- tracking only, not a gate.",
}

_TAXONOMY_ACTIONABLE: frozenset[EvidenceTaxonomy] = frozenset(
    {"executable_behavior", "observability", "architectural_static"}
)

_TAXONOMY_SORT_ORDER: tuple[EvidenceTaxonomy, ...] = (
    "executable_behavior",
    "observability",
    "architectural_static",
    "metadata_spec",
    "manual_review",
    "advisory",
)


def _load_assurance_domains(root: Path) -> dict[str, dict[str, Any]]:
    import yaml

    path = root / "docs" / "plans" / "assurance-domains.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    domains = data.get("domains", {}) if isinstance(data, dict) else {}
    return domains if isinstance(domains, dict) else {}


def _classify_obligation_taxonomy(
    claim_oracle: str,
    claim_independence: str,
    runner_evidence_class: str,
) -> EvidenceTaxonomy:
    """Classify an obligation by its evidence taxonomy.

    Derives the taxonomy from existing oracle, independence level, and
    evidence class fields -- no new catalog metadata required.
    """
    if claim_oracle == "manual_review":
        return "manual_review"
    if claim_oracle == "ceremonial" or claim_independence in ("ceremonial", "self_attesting"):
        return "advisory"
    if claim_oracle in ("proof", "smoke") or runner_evidence_class in ("smoke", "semantic", "performance"):
        return "executable_behavior"
    if runner_evidence_class == "trace":
        return "observability"
    if claim_oracle == "drift_check":
        return "architectural_static"
    if claim_oracle == "construction_sanity":
        return "metadata_spec"
    return "advisory"


def _taxonomy_for_obligation(
    obligation_id: str,
    catalog: VerificationCatalog,
    *,
    cache: dict[str, EvidenceTaxonomy] | None = None,
) -> EvidenceTaxonomy:
    """Look up an obligation by ID and return its evidence taxonomy."""
    if cache is not None and obligation_id in cache:
        return cache[obligation_id]
    for obligation in catalog.obligations:
        if obligation.id == obligation_id:
            taxonomy = _classify_obligation_taxonomy(
                claim_oracle=obligation.claim.oracle,
                claim_independence=obligation.claim.independence_level,
                runner_evidence_class=obligation.runner.evidence_class,
            )
            if cache is not None:
                cache[obligation_id] = taxonomy
            return taxonomy
    result: EvidenceTaxonomy = "advisory"
    if cache is not None:
        cache[obligation_id] = result
    return result


def _build_taxonomy_groups(
    affected_obligations: tuple[AffectedObligation, ...],
    *,
    catalog: VerificationCatalog,
    cache: dict[str, EvidenceTaxonomy],
) -> dict[str, Any]:
    """Group affected obligations by evidence taxonomy."""
    grouped: dict[EvidenceTaxonomy, list[dict[str, Any]]] = defaultdict(list)
    for item in affected_obligations:
        taxonomy = _taxonomy_for_obligation(item.obligation_id, catalog, cache=cache)
        grouped[taxonomy].append(
            {
                "obligation_id": item.obligation_id,
                "claim_id": item.claim_id,
                "subject_id": item.subject_id,
                "reasons": list(item.reasons),
            }
        )

    payload: dict[str, Any] = {}
    for taxonomy in _TAXONOMY_SORT_ORDER:
        items = grouped.get(taxonomy, [])
        payload[taxonomy] = {
            "description": _TAXONOMY_DESCRIPTIONS.get(taxonomy, ""),
            "actionable": taxonomy in _TAXONOMY_ACTIONABLE,
            "obligation_count": len(items),
            "obligations": items,
        }
    return payload


def build_proof_pack(
    root: Path,
    *,
    base_ref: str = "origin/master",
    head_ref: str = "HEAD",
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a diff-shaped proof-pack report with evidence taxonomy."""
    catalog = build_verification_catalog()
    domains_data = _load_assurance_domains(root)
    paths = tuple(sorted(dict.fromkeys(changed_paths or changed_paths_between_refs(base_ref, head_ref))))
    base_ids = () if changed_paths else obligation_ids_for_ref(base_ref)
    head_ids = tuple(obligation.id for obligation in catalog.obligations) if head_ref in {"", "HEAD"} else None
    if head_ids is None:
        head_ids = obligation_ids_for_ref(head_ref)
    affected = build_affected_obligation_report(
        paths,
        base_ref=base_ref,
        head_ref=head_ref,
        catalog=catalog,
        base_obligation_ids=base_ids,
        head_obligation_ids=head_ids,
    )
    claims_by_id = {claim.id: claim for claim in catalog.claims}
    subjects_by_id = {subject.id: subject for subject in catalog.subjects}
    affected_claims = [
        claims_by_id[item.claim_id] for item in affected.affected_obligations if item.claim_id in claims_by_id
    ]
    affected_subjects = [
        subjects_by_id[item.subject_id] for item in affected.affected_obligations if item.subject_id in subjects_by_id
    ]
    affected_domain_names = sorted(
        {claim.assurance_domain for claim in affected_claims}
        | {domain for subject in affected_subjects for domain in (_subject_assurance_domain(subject),) if domain}
    )
    affected_domains = [
        _domain_payload(domain, domains_data.get(domain, {}), affected_claims) for domain in affected_domain_names
    ]
    gate_groups = {
        "focused": _checks_payload(affected.inner_loop_checks),
        "required": _checks_payload(affected.pr_gates),
        "optional_confidence": _checks_payload(affected.deployment_gates),
    }
    visible_gap_domains = {domain["domain"] for domain in affected_domains if domain["claim_count"] > 0}
    known_gaps, additional_known_gaps = _known_gaps(
        domains_data,
        catalog,
        affected_domain_names,
        visible_domains=visible_gap_domains,
    )
    taxonomy_cache: dict[str, EvidenceTaxonomy] = {}
    taxonomy_groups = _build_taxonomy_groups(
        affected.affected_obligations,
        catalog=catalog,
        cache=taxonomy_cache,
    )
    clean_tree = len(paths) == 0
    stable_count = len(affected.obligation_diff.stable_affected)
    affected_count = len(affected.affected_obligations)
    return {
        "refs": {"base_ref": base_ref, "head_ref": head_ref},
        "changed_paths": list(paths),
        "clean_tree": clean_tree,
        "catalog_headline": {
            "claim_count": len(catalog.claims),
            "subject_count": len(catalog.subjects),
            "obligation_count": len(catalog.obligations),
        },
        "catalog_quality_checks": [check.to_dict() for check in catalog.quality_checks],
        "affected": affected.to_payload(),
        "affected_domains": affected_domains,
        "domain_coverage": _domain_coverage(domains_data, catalog),
        "gate_groups": gate_groups,
        "required_gates": _checks_payload((*affected.inner_loop_checks, *affected.pr_gates)),
        "agent_judgment_cells": _agent_judgment_cells(affected_claims),
        "stable_affected_obligations": list(affected.obligation_diff.stable_affected),
        "taxonomy": taxonomy_groups,
        "known_gaps": known_gaps,
        "additional_known_gaps": additional_known_gaps,
        "oracle_mix": dict(Counter(claim.oracle for claim in affected_claims)),
        "cost_tier": _cost_tier_counts(affected, catalog),
        "_context": (
            "No changed paths -- zero obligations are change-specific. Catalog stats reflect the baseline, not a diff."
            if clean_tree
            else (
                f"{len(paths)} changed path(s); {affected_count} affected obligations, "
                f"{stable_count} stable affected. "
                "Taxonomy distinguishes executable gates from static/metadata obligations."
            )
        ),
    }


def evaluate_check_policy(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate proof-pack blocking policy over an already-built report."""
    errors: list[str] = []
    warnings: list[str] = []
    for check in report.get("catalog_quality_checks", []):
        if not isinstance(check, dict):
            continue
        name = str(check.get("name", "catalog.check"))
        summary = str(check.get("summary", "")).strip()
        line = f"{name}: {summary}" if summary else name
        status = str(check.get("status", "")).strip().lower()
        if status == OutcomeStatus.ERROR.value:
            errors.append(line)
        elif status == OutcomeStatus.WARNING.value:
            warnings.append(line)

    serious_judgment_cells = [
        cell
        for cell in report.get("agent_judgment_cells", [])
        if isinstance(cell, dict) and cell.get("severity") == "serious" and not _judgment_cell_satisfied(cell)
    ]
    for cell in serious_judgment_cells:
        errors.append(
            "affected serious claim needs agent judgment artifact: "
            f"{cell.get('claim_id')} ({cell.get('oracle')}, {cell.get('independence_level')})"
        )

    suppressed = _suppressed_change_subjects(report)
    if suppressed:
        warnings.append("change-subject IDs were suppressed from obligation routing: " + ", ".join(suppressed[:10]))
    known_gaps = report.get("known_gaps", [])
    if known_gaps:
        warnings.append(f"known coverage gaps intersect affected domains: {len(known_gaps)} domain(s)")
    if report.get("additional_known_gaps"):
        warnings.append("additional known gaps exist in zero-claim routed domains")

    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
    }


def _judgment_cell_satisfied(cell: dict[str, Any]) -> bool:
    if cell.get("tracked_exception"):
        return True
    result = str(cell.get("result") or "").strip().lower()
    if result not in {"accepted", "passed", "pass", "ok"}:
        return False
    return all(str(cell.get(field) or "").strip() for field in ("artifact", "reviewer", "produced_at", "freshness"))


def _suppressed_change_subjects(report: dict[str, Any]) -> list[str]:
    affected = report.get("affected")
    if not isinstance(affected, dict):
        return []
    diff = affected.get("obligation_diff")
    if not isinstance(diff, dict):
        return []
    suppressed = diff.get("suppressed")
    if not isinstance(suppressed, list):
        return []
    return [str(item) for item in suppressed]


def _domain_payload(domain: str, info: object, claims: list[Any]) -> dict[str, Any]:
    domain_claims = [claim for claim in claims if claim.assurance_domain == domain]
    info_dict = info if isinstance(info, dict) else {}
    return {
        "domain": domain,
        "description": info_dict.get("description", ""),
        "maturity": info_dict.get("maturity", "seed"),
        "claim_count": len({claim.id for claim in domain_claims}),
        "oracle_counts": dict(Counter(claim.oracle for claim in domain_claims)),
    }


def _domain_coverage(domains_data: dict[str, dict[str, Any]], catalog: Any) -> list[dict[str, Any]]:
    claims_by_domain: dict[str, list[Any]] = defaultdict(list)
    obligations_by_domain: Counter[str] = Counter()
    for claim in catalog.claims:
        claims_by_domain[claim.assurance_domain].append(claim)
    for obligation in catalog.obligations:
        obligations_by_domain[obligation.claim.assurance_domain] += 1
    coverage: list[dict[str, Any]] = []
    for domain, info in sorted(domains_data.items()):
        info_dict = info if isinstance(info, dict) else {}
        claims = claims_by_domain.get(domain, [])
        coverage.append(
            {
                "domain": domain,
                "description": info_dict.get("description", ""),
                "maturity": info_dict.get("maturity", "seed"),
                "claim_count": len(claims),
                "obligation_count": obligations_by_domain[domain],
                "oracle_counts": dict(Counter(claim.oracle for claim in claims)),
                "gaps": list(info_dict.get("coverage_gaps", []) or []),
            }
        )
    return coverage


def _checks_payload(checks: tuple[RecommendedCheck, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    payload: list[dict[str, Any]] = []
    for check in checks:
        if check.command in seen:
            continue
        seen.add(check.command)
        payload.append(check.to_payload())
    return payload


def _agent_judgment_cells(claims: list[Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for claim in claims:
        if claim.oracle == "manual_review" or claim.independence_level in {"same_source", "self_attesting"}:
            cells.append(
                {
                    "claim_id": claim.id,
                    "oracle": claim.oracle,
                    "independence_level": claim.independence_level,
                    "severity": claim.severity,
                    "tracked_exception": claim.tracked_exception,
                    "artifact": None,
                    "reviewer": None,
                    "produced_at": None,
                    "freshness": None,
                    "result": "tracked_exception" if claim.tracked_exception else "missing",
                    "reason": "requires agent judgment artifact or independent evidence upgrade",
                }
            )
    return cells


def _known_gaps(
    domains_data: dict[str, dict[str, Any]],
    catalog: Any,
    affected_domains: list[str],
    *,
    visible_domains: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gaps_by_domain: dict[str, list[str]] = defaultdict(list)
    for subject in catalog.subjects:
        if subject.kind != "assurance.coverage_gap":
            continue
        domain = _subject_assurance_domain(subject)
        if not domain or domain not in affected_domains:
            continue
        gap = subject.attrs.get("gap")
        axis = subject.attrs.get("axis")
        owner = subject.attrs.get("owner")
        next_evidence = subject.attrs.get("next_evidence")
        rendered = str(gap or "").strip()
        if axis:
            rendered = f"{axis}: {rendered}"
        details = [f"owner: {owner}" if owner else "", f"next: {next_evidence}" if next_evidence else ""]
        details = [detail for detail in details if detail]
        if details:
            rendered = f"{rendered} [{'; '.join(details)}]"
        if rendered:
            gaps_by_domain[domain].append(rendered)

    for domain in affected_domains:
        info = domains_data.get(domain, {})
        raw_gaps = info.get("coverage_gaps", []) if isinstance(info, dict) else []
        for raw_gap in raw_gaps:
            gaps_by_domain[domain].append(str(raw_gap))
    gap_payloads = [
        {"domain": domain, "gaps": sorted(dict.fromkeys(gaps))}
        for domain, gaps in sorted(gaps_by_domain.items())
        if gaps
    ]
    visible = [item for item in gap_payloads if item["domain"] in visible_domains]
    additional = [item for item in gap_payloads if item["domain"] not in visible_domains]
    return visible, additional


def _subject_assurance_domain(subject: Any) -> str | None:
    domain = subject.attrs.get("assurance_domain")
    return domain if isinstance(domain, str) and domain.strip() else None


def build_slop_dashboard(catalog: VerificationCatalog) -> dict[str, Any]:
    """Analyze the catalog for anti-vacuity violations — "slop" claims.

    A claim is slop if ANY of:
    - No breaker defined and no tracked exception
    - No runner binding (no executable check)
    - Zero subjects matched (no compiled obligations)
    - Its runner bindings have stale or missing freshness
    - Its assertion source is the same as the manifest (self-referential)
    """
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    runners_by_claim: dict[str, list[RunnerBinding]] = defaultdict(list)
    for runner in catalog.runner_bindings:
        runners_by_claim[runner.claim_id].append(runner)

    obligations_by_claim: dict[str, int] = defaultdict(int)
    for obligation in catalog.obligations:
        obligations_by_claim[obligation.claim.id] += 1

    rows: list[dict[str, Any]] = []
    for claim in sorted(catalog.claims, key=lambda c: c.id):
        claim_runners = runners_by_claim.get(claim.id, [])
        obligation_count = obligations_by_claim.get(claim.id, 0)

        missing_breaker = claim.breaker is None and claim.tracked_exception is None
        missing_runner = len(claim_runners) == 0
        zero_subjects = obligation_count == 0 and not claim.abstract
        # Self-referential: evidence source is the same manifest
        self_referential = claim.assertion_source in {
            "same_source_manifest",
            "same_source",
        }

        stale_evidence: list[str] = []
        for runner in claim_runners:
            freshness = runner.trust.freshness
            if freshness is None or not freshness.strip():
                stale_evidence.append(f"{runner.id}: no freshness")
            elif freshness in (
                "static review refreshed with generated catalog",
                "catalog-reviewed",
            ):
                pass  # static freshness is by design for structural claims
            elif freshness.startswith("ci:") or freshness.startswith("local:"):
                try:
                    ts_str = freshness.split(":", 1)[1]
                    ts = datetime.fromisoformat(ts_str)
                    if (now - ts).days > 30:
                        stale_evidence.append(f"{runner.id}: freshness >30d old")
                except (ValueError, IndexError):
                    stale_evidence.append(f"{runner.id}: unparseable freshness {freshness!r}")

        reasons: dict[str, bool] = {
            "missing_breaker": missing_breaker,
            "missing_runner": missing_runner,
            "zero_subjects": zero_subjects,
            "self_referential": self_referential,
            "stale_evidence": bool(stale_evidence),
        }
        is_slop = any(reasons.values())
        if is_slop:
            rows.append(
                {
                    "claim_id": claim.id,
                    "assurance_domain": str(claim.assurance_domain),
                    "severity": str(claim.severity),
                    "missing_breaker": missing_breaker,
                    "missing_runner": missing_runner,
                    "zero_subjects": zero_subjects,
                    "self_referential": self_referential,
                    "stale_evidence": bool(stale_evidence),
                    "stale_evidence_details": stale_evidence,
                    "runner_count": len(claim_runners),
                    "obligation_count": obligation_count,
                    "abstract": claim.abstract,
                }
            )

    return {
        "total_claims": len(catalog.claims),
        "slop_count": len(rows),
        "rows": sorted(rows, key=lambda r: r["claim_id"]),
        "by_reason": {
            "missing_breaker": sum(1 for r in rows if r["missing_breaker"]),
            "missing_runner": sum(1 for r in rows if r["missing_runner"]),
            "zero_subjects": sum(1 for r in rows if r["zero_subjects"]),
            "self_referential": sum(1 for r in rows if r["self_referential"]),
            "stale_evidence": sum(1 for r in rows if r["stale_evidence"]),
        },
    }


def render_slop_markdown(dashboard: dict[str, Any]) -> str:
    """Render the slop dashboard as markdown."""
    lines = [
        "## Anti-Vacuity Slop Dashboard",
        "",
        f"**Claims analyzed:** {dashboard['total_claims']}  **Slop claims:** {dashboard['slop_count']}",
        "",
        "### By Reason",
        "",
    ]
    by_reason = dashboard["by_reason"]
    for reason in ("missing_breaker", "missing_runner", "zero_subjects", "self_referential", "stale_evidence"):
        count = by_reason.get(reason, 0)
        lines.append(f"- **{reason}:** {count}")
    lines.append("")

    rows = dashboard["rows"]
    if not rows:
        lines.append("_No slop claims found._")
        return "\n".join(lines)

    lines.extend(
        [
            "| Claim ID | Domain | Severity | Missing Breaker | Missing Runner | Zero Subjects | Self-Referential | Stale Evidence",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['claim_id']} "
            f"| {row['assurance_domain']} "
            f"| {row['severity']} "
            f"| {'X' if row['missing_breaker'] else ''} "
            f"| {'X' if row['missing_runner'] else ''} "
            f"| {'X' if row['zero_subjects'] else ''} "
            f"| {'X' if row['self_referential'] else ''} "
            f"| {'X' if row['stale_evidence'] else ''} |"
        )

    lines.append("")
    lines.append("### X = slop condition detected. Empty = OK.")
    return "\n".join(lines)


def discover_click_json_subjects(
    cli_group: Any = None,
) -> list[SubjectRef]:
    """Discover CLI commands with ``--format json`` and generate proof subjects.

    Each command that accepts ``--format json`` (or ``--json``) produces a
    ``SubjectRef`` with kind ``cli.json_command``.  The subject's attrs
    carry the command path and help text so downstream claims can match
    on ``Kind("cli.json_command")``.
    """

    try:
        from polylogue.cli.click_app import cli as click_group
    except ImportError:
        return []

    subjects: list[SubjectRef] = []
    _walk_json_commands(click_group, (), subjects)
    return subjects


def _walk_json_commands(
    group: Any,
    parent_path: tuple[str, ...],
    subjects: list[SubjectRef],
) -> None:
    """Recursively walk a Click group and collect JSON-format commands."""
    for cmd_name in getattr(group, "commands", {}):
        cmd = group.commands[cmd_name]
        cmd_path = (*parent_path, cmd_name)
        full_name = " ".join(cmd_path)

        has_json = _click_has_json_flag(cmd)
        is_group = hasattr(cmd, "commands")
        if is_group:
            _walk_json_commands(cmd, cmd_path, subjects)
        if has_json:
            subjects.append(
                SubjectRef(
                    kind="cli.json_command",
                    id=full_name,
                    attrs={
                        "command_path": full_name,
                        "help": (cmd.help or "").strip(),
                    },
                )
            )


def _click_has_json_flag(cmd: Any) -> bool:
    """Check if a Click command has a --json or --format json option."""
    if not hasattr(cmd, "params"):
        return False
    for param in cmd.params:
        if not hasattr(param, "opts"):
            continue
        opts = [str(o).lstrip("-") for o in param.opts]
        if "json" in opts:
            return True
        if (
            "format" in opts
            and hasattr(param, "type")
            and hasattr(param.type, "choices")
            and "json" in [str(c).lower() for c in param.type.choices]
        ):
            return True
    return False


def _print_slop_dashboard(dashboard: dict[str, Any]) -> None:
    """Print a human-readable slop dashboard to stdout."""
    print("Anti-Vacuity Slop Dashboard")
    print(f"{'=' * 40}")
    print(f"Claims analyzed: {dashboard['total_claims']}")
    print(f"Slop claims:     {dashboard['slop_count']}")
    print()
    print("By reason:")
    by_reason = dashboard["by_reason"]
    for reason in ("missing_breaker", "missing_runner", "zero_subjects", "self_referential", "stale_evidence"):
        count = by_reason.get(reason, 0)
        print(f"  {reason:<25} {count}")
    print()

    rows = dashboard["rows"]
    if not rows:
        print("No slop claims found.")
        return

    print(f"{'Claim ID':<55} {'Domain':<25} {'Brk':>3} {'Run':>3} {'Sub':>3} {'Slf':>3} {'Stl':>3}")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['claim_id']:<55} "
            f"{row['assurance_domain']:<25} "
            f"{'X' if row['missing_breaker'] else '.':>3} "
            f"{'X' if row['missing_runner'] else '.':>3} "
            f"{'X' if row['zero_subjects'] else '.':>3} "
            f"{'X' if row['self_referential'] else '.':>3} "
            f"{'X' if row['stale_evidence'] else '.':>3}"
        )
    print()
    print("Brk=missing_breaker  Run=missing_runner  Sub=zero_subjects")
    print("Slf=self_referential  Stl=stale_evidence")


def _cost_tier_counts(report: Any, catalog: VerificationCatalog) -> dict[str, int]:
    counts: Counter[str] = Counter()
    tiers_by_obligation = {obligation.id: obligation.runner.cost_tier for obligation in catalog.obligations}
    for item in report.affected_obligations:
        counts[tiers_by_obligation.get(item.obligation_id, "static")] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff-shaped verification impact report for PR review.")
    parser.add_argument("--base-ref", default="origin/master", help="Base git ref for changed-path discovery.")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref for changed-path discovery.")
    parser.add_argument("--path", action="append", dest="paths", default=None, help="Changed file path (repeatable).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--markdown", action="store_true", help="Emit PR-comment-ready Markdown.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when report blocking policy fails.")
    parser.add_argument(
        "--slop",
        action="store_true",
        help="Emit anti-vacuity slop dashboard (claims lacking breakers, runners, subjects, or with stale evidence).",
    )
    parser.add_argument(
        "--discover-subjects",
        action="store_true",
        help="Discover and print subjects from Click introspection (CLI commands with --format json).",
    )
    args = parser.parse_args(argv)

    root = _get_root()

    if args.slop:
        catalog = build_verification_catalog()
        dashboard = build_slop_dashboard(catalog)
        if args.json:
            json.dump(dashboard, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        elif args.markdown:
            sys.stdout.write(render_slop_markdown(dashboard))
            sys.stdout.write("\n")
        else:
            _print_slop_dashboard(dashboard)
        return 1 if dashboard["slop_count"] > 0 and args.check else 0

    if args.discover_subjects:
        subjects = discover_click_json_subjects()
        if args.json:
            payload = [s.to_payload() for s in subjects]
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        else:
            print(f"Discovered {len(subjects)} JSON-capable CLI subjects:")
            for s in subjects:
                print(f"  {s.id:50} {str(s.attrs.get('help', ''))[:60]}")
        return 0

    report = build_proof_pack(
        root,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        changed_paths=args.paths,
    )
    check_result = evaluate_check_policy(report)
    if args.check:
        report["check"] = check_result

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.markdown:
        sys.stdout.write(render_markdown(report))
        sys.stdout.write("\n")
    else:
        _print_human_report(report)

    if args.check and check_result["status"] != "ok":
        return 1
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    """Render a taxonomy-aware report as PR-comment-ready markdown."""
    refs = report["refs"]
    context = str(report.get("_context", ""))
    changed_paths_count = len(report.get("changed_paths", []))
    lines = [
        "## Polylogue Verification Impact",
        "",
        f"**Refs:** `{refs['base_ref']}..{refs['head_ref']}`",
        f"**Changed paths:** {changed_paths_count}",
    ]

    if context:
        lines.extend(["", f"> {context}"])

    check_result = report.get("check")
    if isinstance(check_result, dict):
        lines.extend(["", "### Check Policy"])
        if check_result.get("status") == "ok":
            lines.append("- status: `ok`")
        else:
            lines.append("- status: `error`")
        for error in check_result.get("errors", []) or []:
            lines.append(f"- blocking: {error}")
        for warning in check_result.get("warnings", []) or []:
            lines.append(f"- warning: {warning}")

    # Required Gates
    gate_groups = report.get("gate_groups", {})
    focused = gate_groups.get("focused", [])
    combined_required = _checks_payload_merge(focused, gate_groups.get("required", []))
    lines.extend(["", "### Required Gates (run now)"])
    if combined_required:
        for gate in combined_required:
            command = " ".join(gate["command"])
            lines.append(f"- `{command}` -- {gate['reason']}")
    else:
        lines.append("- none beyond always-on PR gates")
    lines.extend(["", "### Always-On PR Gates"])
    _append_gate_lines(lines, gate_groups.get("required", []), empty_text="- none")

    # Confidence Gates
    lines.extend(["", "### Confidence Gates (optional)"])
    _append_gate_lines(lines, gate_groups.get("optional_confidence", []), empty_text="- none")

    # Evidence Taxonomy
    lines.extend(["", "### Affected Evidence by Taxonomy"])
    taxonomy = report.get("taxonomy", {})
    _render_taxonomy_section_markdown(lines, taxonomy)

    # Agent Judgment Cells
    agent_judgment_cells = report.get("agent_judgment_cells", [])
    if agent_judgment_cells:
        lines.extend(["", "### Agent Judgment Cells (manual review)"])
        lines.append(f"_{len(agent_judgment_cells)} cell(s) -- only actionable if severity is 'serious'_")
        for cell in agent_judgment_cells:
            lines.append(
                f"- `{cell['claim_id']}` -- result `{cell['result']}`; "
                f"artifact `{cell['artifact'] or 'missing'}`; "
                f"reviewer `{cell['reviewer'] or 'missing'}`"
            )

    # Known Gaps
    lines.extend(["", "### Known Gaps (tracking, not blocking)"])
    known_gaps = report.get("known_gaps", [])
    if known_gaps:
        for item in known_gaps:
            lines.append(f"- `{item['domain']}`: {', '.join(item['gaps'])}")
    else:
        lines.append("- none for domains with affected claims")
    additional_known_gaps = report.get("additional_known_gaps", [])
    if additional_known_gaps:
        lines.extend(["", "<details>", "<summary>Additional gaps (zero-claim routed domains)</summary>", ""])
        for item in additional_known_gaps:
            lines.append(f"- `{item['domain']}`: {', '.join(item['gaps'])}")
        lines.extend(["", "</details>"])

    # Catalog quality summary
    lines.extend(["", "### Catalog Checks"])
    oracle_mix = report.get("oracle_mix", {})
    cost_tier = report.get("cost_tier", {})
    catalog_headline = report.get("catalog_headline", {})
    lines.append(
        f"- claims: {catalog_headline.get('claim_count', '?')}, "
        f"obligations: {catalog_headline.get('obligation_count', '?')}"
    )
    oracle_str = json.dumps(oracle_mix, sort_keys=True) if oracle_mix else "(none affected)"
    lines.append(f"- oracle mix: `{oracle_str}`")
    cost_str = json.dumps(cost_tier, sort_keys=True) if cost_tier else "(none)"
    lines.append(f"- cost tier: `{cost_str}`")
    lines.append(f"- stable affected obligations: {len(report.get('stable_affected_obligations', []))}")
    lines.append(f"- agent judgment cells: {len(agent_judgment_cells)}")

    return "\n".join(lines)


def _render_taxonomy_section_markdown(lines: list[str], taxonomy: dict[str, Any]) -> None:
    """Append taxonomy-grouped obligation tables to markdown lines."""
    if not taxonomy:
        lines.append("- no affected obligations")
        return

    for taxonomy_name in _TAXONOMY_SORT_ORDER:
        group = taxonomy.get(taxonomy_name)
        if not isinstance(group, dict) or group.get("obligation_count", 0) == 0:
            continue
        count = group["obligation_count"]
        description = str(group.get("description", ""))
        actionable = bool(group.get("actionable", False))
        label = "actionable" if actionable else "does not block"
        lines.append(f"- **{taxonomy_name}** ({label}): {count} obligation(s) -- {description}")


def _checks_payload_merge(
    *check_lists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge and deduplicate check payload lists."""
    seen: set[tuple[str, ...]] = set()
    merged: list[dict[str, Any]] = []
    for check_list in check_lists:
        for check in check_list:
            command = tuple(check.get("command", []))
            if command in seen:
                continue
            seen.add(command)
            merged.append(check)
    return merged


def _append_gate_lines(lines: list[str], gates: list[dict[str, Any]], *, empty_text: str) -> None:
    if not gates:
        lines.append(empty_text)
        return
    for gate in gates:
        command = " ".join(gate["command"])
        lines.append(f"- `{command}` -- {gate['reason']}")


def _print_human_report(report: dict[str, Any]) -> None:
    """Print a taxonomy-aware human-readable verification report."""
    headline = report["catalog_headline"]
    print(
        f"Verification catalog: {headline['claim_count']} claims, "
        f"{headline['obligation_count']} obligations, {headline['subject_count']} subjects"
    )
    print(f"Refs: {report['refs']['base_ref']}..{report['refs']['head_ref']}")
    print(f"Changed paths: {len(report['changed_paths'])}")
    context = report.get("_context", "")
    if context:
        print(f"  {context}")
    print()

    gate_groups = report.get("gate_groups", {})
    print("Required gates (run now):")
    focused = gate_groups.get("focused", [])
    required = gate_groups.get("required", [])
    all_required = _checks_payload_merge(focused, required)
    if all_required:
        for gate in all_required:
            print(f"  {' '.join(gate['command'])} -- {gate['reason']}")
    else:
        print("  none beyond always-on PR gates")
    print()

    print("Evidence taxonomy:")
    taxonomy = report.get("taxonomy", {})
    if taxonomy:
        for taxonomy_name in _TAXONOMY_SORT_ORDER:
            group = taxonomy.get(taxonomy_name)
            if not isinstance(group, dict) or group.get("obligation_count", 0) == 0:
                continue
            count = group["obligation_count"]
            actionable = "RUN" if group.get("actionable") else "INFO"
            print(f"  {taxonomy_name:<25} {count:>4} obligations [{actionable}]")
            print(f"    {group.get('description', '')}")
    else:
        print("  no affected obligations")
    print()

    print("Confidence gates (optional):")
    for gate in gate_groups.get("optional_confidence", []):
        print(f"  {' '.join(gate['command'])} -- {gate['reason']}")
    print()
    print(f"Agent judgment cells: {len(report.get('agent_judgment_cells', []))}")
    print(f"Stable affected obligations: {len(report.get('stable_affected_obligations', []))}")
    print(f"Known gaps: {len(report.get('known_gaps', []))}")

    check_result = report.get("check")
    if isinstance(check_result, dict):
        print(f"Check policy: {check_result.get('status', 'unknown')}")
        for error in check_result.get("errors", []) or []:
            print(f"  blocking: {error}")
        for warning in check_result.get("warnings", []) or []:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    sys.exit(main())
