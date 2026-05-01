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
    affected_domains = sorted(
        {claim.assurance_domain for claim in affected_claims}
        | {domain for subject in affected_subjects for domain in (_subject_assurance_domain(subject),) if domain}
    )
    return {
        "refs": {"base_ref": base_ref, "head_ref": head_ref},
        "changed_paths": list(paths),
        "catalog_headline": {
            "claim_count": len(catalog.claims),
            "subject_count": len(catalog.subjects),
            "obligation_count": len(catalog.obligations),
        },
        "affected": affected.to_payload(),
        "affected_domains": [
            _domain_payload(domain, domains_data.get(domain, {}), affected_claims) for domain in affected_domains
        ],
        "domain_coverage": _domain_coverage(domains_data, catalog),
        "required_gates": _checks_payload(
            (*affected.inner_loop_checks, *affected.pr_gates, *affected.deployment_gates)
        ),
        "manual_review_cells": _manual_review_cells(affected_claims),
        "stale_evidence": list(affected.obligation_diff.stale_evidence),
        "known_gaps": _known_gaps(domains_data, catalog, affected_domains),
        "oracle_mix": dict(Counter(claim.oracle for claim in affected_claims)),
        "cost_tier": _cost_tier_counts(affected, catalog),
    }


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


def _manual_review_cells(claims: list[Any]) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for claim in claims:
        if claim.oracle == "manual_review" or claim.independence_level in {"same_source", "self_attesting"}:
            cells.append(
                {
                    "claim_id": claim.id,
                    "oracle": claim.oracle,
                    "independence_level": claim.independence_level,
                    "reason": "requires human confirmation or independent evidence upgrade",
                }
            )
    return cells


def _known_gaps(
    domains_data: dict[str, dict[str, Any]],
    catalog: Any,
    affected_domains: list[str],
) -> list[dict[str, Any]]:
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
    return [
        {"domain": domain, "gaps": sorted(dict.fromkeys(gaps))}
        for domain, gaps in sorted(gaps_by_domain.items())
        if gaps
    ]


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
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    report = build_proof_pack(
        root,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        changed_paths=args.paths,
    )

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.markdown:
        sys.stdout.write(render_markdown(report))
        sys.stdout.write("\n")
    else:
        _print_human_report(report)

    return 0


def render_markdown(report: dict[str, Any]) -> str:
    refs = report["refs"]
    lines = [
        "## Polylogue Proof Pack",
        "",
        f"**Refs:** `{refs['base_ref']}..{refs['head_ref']}`",
        f"**Changed paths:** {len(report['changed_paths'])}",
        "",
        "### Affected Domains",
    ]
    affected_domains = report.get("affected_domains", [])
    if affected_domains:
        for domain in affected_domains:
            lines.append(f"- `{domain['domain']}` — {domain['claim_count']} claim(s), maturity `{domain['maturity']}`")
    else:
        lines.append("- none")
    lines.extend(["", "### Required Gates"])
    for gate in report.get("required_gates", []):
        command = " ".join(gate["command"])
        lines.append(f"- `{command}` — {gate['reason']}")
    lines.extend(["", "### Known Gaps"])
    known_gaps = report.get("known_gaps", [])
    if known_gaps:
        for item in known_gaps:
            lines.append(f"- `{item['domain']}`: {', '.join(item['gaps'])}")
    else:
        lines.append("- none for affected domains")
    lines.extend(["", "### Oracle / Cost"])
    lines.append(f"- oracle mix: `{json.dumps(report.get('oracle_mix', {}), sort_keys=True)}`")
    lines.append(f"- cost tier: `{json.dumps(report.get('cost_tier', {}), sort_keys=True)}`")
    lines.append(f"- stale evidence: {len(report.get('stale_evidence', []))}")
    lines.append(f"- manual review cells: {len(report.get('manual_review_cells', []))}")
    return "\n".join(lines)


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
    affected_domains = report.get("affected_domains", [])
    if not affected_domains:
        print("  none")
    for domain in affected_domains:
        print(f"  {domain['domain']:<30} {domain['claim_count']:>3} claims  {domain['oracle_counts']}")
    print()
    print("Required gates:")
    for gate in report.get("required_gates", []):
        print(f"  {' '.join(gate['command'])} — {gate['reason']}")
    print()
    print(f"Manual review cells: {len(report.get('manual_review_cells', []))}")
    print(f"Stale evidence: {len(report.get('stale_evidence', []))}")
    print(f"Known gaps: {len(report.get('known_gaps', []))}")


if __name__ == "__main__":
    sys.exit(main())
