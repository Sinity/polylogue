"""Diff-shaped proof-pack report for PR confidence.

Consumes the assurance-domain manifests, proof catalog, and affected-obligation
routing to produce an operator-facing confidence answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from polylogue.core.outcomes import OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.diffing import (
    AffectedObligationReport,
    RecommendedCheck,
    build_affected_obligation_report,
    changed_paths_between_refs,
    obligation_ids_for_ref,
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


def build_proof_pack(
    root: Path,
    *,
    base_ref: str = "origin/master",
    head_ref: str = "HEAD",
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a diff-shaped proof-pack report."""
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
    return {
        "refs": {"base_ref": base_ref, "head_ref": head_ref},
        "changed_paths": list(paths),
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
        "manual_review_cells": _manual_review_cells(affected_claims),
        "stable_affected_obligations": list(affected.obligation_diff.stable_affected),
        "known_gaps": known_gaps,
        "additional_known_gaps": additional_known_gaps,
        "oracle_mix": dict(Counter(claim.oracle for claim in affected_claims)),
        "cost_tier": _cost_tier_counts(affected, catalog),
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

    serious_manual_cells = [
        cell
        for cell in report.get("manual_review_cells", [])
        if isinstance(cell, dict) and cell.get("severity") == "serious" and not cell.get("tracked_exception")
    ]
    for cell in serious_manual_cells:
        errors.append(
            "affected serious claim needs manual/same-source review: "
            f"{cell.get('claim_id')} ({cell.get('oracle')}, {cell.get('independence_level')})"
        )

    suppressed = _suppressed_change_subjects(report)
    if suppressed:
        warnings.append("changed paths were suppressed from obligation routing: " + ", ".join(suppressed[:10]))
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


def _manual_review_cells(claims: list[Any]) -> list[dict[str, Any]]:
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
                    "reason": "requires human confirmation or independent evidence upgrade",
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


def _cost_tier_counts(report: AffectedObligationReport, catalog: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    tiers_by_obligation = {obligation.id: obligation.runner.cost_tier for obligation in catalog.obligations}
    for item in report.affected_obligations:
        counts[tiers_by_obligation.get(item.obligation_id, "static")] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff-shaped proof-pack report for PR confidence.")
    parser.add_argument("--base-ref", default="origin/master", help="Base git ref for changed-path discovery.")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref for changed-path discovery.")
    parser.add_argument("--path", action="append", dest="paths", default=None, help="Changed file path (repeatable).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--markdown", action="store_true", help="Emit PR-comment-ready Markdown.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when proof-pack blocking policy fails.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
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
    refs = report["refs"]
    lines = [
        "## Polylogue Proof Pack",
        "",
        f"**Refs:** `{refs['base_ref']}..{refs['head_ref']}`",
        f"**Changed paths:** {len(report['changed_paths'])}",
    ]
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
    lines.extend(["", "### Affected Domains"])
    affected_domains = report.get("affected_domains", [])
    visible_domains = [domain for domain in affected_domains if domain.get("claim_count", 0) > 0]
    zero_claim_domains = [domain for domain in affected_domains if domain.get("claim_count", 0) == 0]
    if visible_domains:
        for domain in visible_domains:
            lines.append(f"- `{domain['domain']}` — {domain['claim_count']} claim(s), maturity `{domain['maturity']}`")
    else:
        lines.append("- none with affected claims")
    if zero_claim_domains:
        lines.extend(
            [
                "",
                "<details>",
                "<summary>Additional routed domains with zero affected claims</summary>",
                "",
            ]
        )
        for domain in zero_claim_domains:
            lines.append(f"- `{domain['domain']}` — maturity `{domain['maturity']}`")
        lines.extend(["", "</details>"])

    gate_groups = report.get("gate_groups", {})
    lines.extend(["", "### Focused Gates"])
    _append_gate_lines(lines, gate_groups.get("focused", []), empty_text="- none")
    lines.extend(["", "### Required PR Gates"])
    _append_gate_lines(lines, gate_groups.get("required", report.get("required_gates", [])), empty_text="- none")
    lines.extend(["", "### Optional Confidence Gates"])
    _append_gate_lines(lines, gate_groups.get("optional_confidence", []), empty_text="- none")

    lines.extend(["", "### Known Gaps"])
    known_gaps = report.get("known_gaps", [])
    if known_gaps:
        for item in known_gaps:
            lines.append(f"- `{item['domain']}`: {', '.join(item['gaps'])}")
    else:
        lines.append("- none for domains with affected claims")
    additional_known_gaps = report.get("additional_known_gaps", [])
    if additional_known_gaps:
        lines.extend(["", "<details>", "<summary>Additional gaps in zero-claim routed domains</summary>", ""])
        for item in additional_known_gaps:
            lines.append(f"- `{item['domain']}`: {', '.join(item['gaps'])}")
        lines.extend(["", "</details>"])
    lines.extend(["", "### Oracle / Cost"])
    lines.append(f"- oracle mix: `{json.dumps(report.get('oracle_mix', {}), sort_keys=True)}`")
    lines.append(f"- cost tier: `{json.dumps(report.get('cost_tier', {}), sort_keys=True)}`")
    lines.append(f"- stable affected obligations: {len(report.get('stable_affected_obligations', []))}")
    lines.append(f"- manual review cells: {len(report.get('manual_review_cells', []))}")
    return "\n".join(lines)


def _append_gate_lines(lines: list[str], gates: list[dict[str, Any]], *, empty_text: str) -> None:
    if not gates:
        lines.append(empty_text)
        return
    for gate in gates:
        command = " ".join(gate["command"])
        lines.append(f"- `{command}` — {gate['reason']}")


def _print_human_report(report: dict[str, Any]) -> None:
    headline = report["catalog_headline"]
    print(
        f"Proof catalog: {headline['claim_count']} claims, "
        f"{headline['obligation_count']} obligations, {headline['subject_count']} subjects"
    )
    print(f"Refs: {report['refs']['base_ref']}..{report['refs']['head_ref']}")
    print(f"Changed paths: {len(report['changed_paths'])}")
    print()
    print("Affected domains:")
    affected_domains = [domain for domain in report.get("affected_domains", []) if domain.get("claim_count", 0) > 0]
    if not affected_domains:
        print("  none with affected claims")
    for domain in affected_domains:
        print(f"  {domain['domain']:<30} {domain['claim_count']:>3} claims  {domain['oracle_counts']}")
    print()
    gate_groups = report.get("gate_groups", {})
    print("Focused gates:")
    for gate in gate_groups.get("focused", []):
        print(f"  {' '.join(gate['command'])} — {gate['reason']}")
    print("Required PR gates:")
    for gate in gate_groups.get("required", []):
        print(f"  {' '.join(gate['command'])} — {gate['reason']}")
    print("Optional confidence gates:")
    for gate in gate_groups.get("optional_confidence", []):
        print(f"  {' '.join(gate['command'])} — {gate['reason']}")
    print()
    print(f"Manual review cells: {len(report.get('manual_review_cells', []))}")
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
