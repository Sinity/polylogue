"""Domain-grouped affected coverage report for vibecode confidence.

Consumes the assurance-domain manifests and proof catalog to produce a
domain-aware confidence answer instead of an undifferentiated obligation count.

Part of #510.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.proof.catalog import build_verification_catalog


@dataclass
class DomainCoverage:
    domain: str
    description: str
    maturity: str
    claim_count: int
    obligation_count: int
    oracle_counts: dict[str, int] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


def _load_assurance_domains(root: Path) -> dict[str, dict[str, Any]]:
    import yaml

    path = root / "docs" / "plans" / "assurance-domains.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    domains = data.get("domains", {})
    return domains if isinstance(domains, dict) else {}


def build_proof_pack(
    root: Path,
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a domain-grouped coverage report.

    Args:
        root: Project root directory.
        changed_paths: Optional list of changed file paths to contextualize.

    Returns:
        A JSON-serializable dict with domain coverage, affected domains,
        and actionable gaps.
    """
    catalog = build_verification_catalog()
    domains_data = _load_assurance_domains(root)

    # Map claims to domains via Claim.assurance_domain
    domain_claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in catalog.claims:
        dom = claim.assurance_domain
        domain_claims[dom].append(
            {
                "claim_id": claim.id,
                "description": claim.description,
                "oracle": claim.oracle,
                "severity": claim.severity,
            }
        )

    # Build per-domain coverage
    coverage: list[dict[str, Any]] = []
    for dom_name, dom_info in sorted(domains_data.items()):
        if not isinstance(dom_info, dict):
            continue
        claims = domain_claims.get(dom_name, [])
        oracle_counts: dict[str, int] = defaultdict(int)
        for c in claims:
            oracle_counts[str(c.get("oracle", "construction_sanity"))] += 1

        coverage.append(
            {
                "domain": dom_name,
                "description": dom_info.get("description", ""),
                "maturity": dom_info.get("maturity", "seed"),
                "claim_count": len(claims),
                "oracle_counts": dict(oracle_counts),
                "oracle_present": dom_info.get("oracle_present", False),
                "gaps": _collect_gaps(dom_name, dom_info),
            }
        )

    # Compute domain-level impact if changed_paths provided
    affected_domains: list[dict[str, Any]] = []
    if changed_paths:
        affected_domains = _route_paths_to_domains(changed_paths, domains_data, domain_claims)

    # Overall confidence assessment
    proof_obligations = sum(
        sum(1 for o in catalog.obligations if o.claim.assurance_domain == dom_name) for dom_name in domains_data
    )
    smoky_obligations = sum(1 for o in catalog.obligations if o.claim.oracle == "proof")
    ceremonial_obligations = sum(1 for o in catalog.obligations if o.claim.oracle == "ceremonial")

    return {
        "catalog_headline": {
            "total_obligations": len(catalog.obligations),
            "proof_obligations": proof_obligations,
            "smoky_obligations": smoky_obligations,
            "ceremonial_obligations": ceremonial_obligations,
            "claim_count": len(catalog.claims),
            "subject_count": len(catalog.subjects),
            "domain_count": len(coverage),
        },
        "domain_coverage": coverage,
        "affected_domains": affected_domains,
        "changed_paths": changed_paths or [],
        "gates": [
            {"command": ["devtools", "verify", "--quick"], "when": "every_change"},
            {"command": ["devtools", "verify"], "when": "before_pr"},
        ],
    }


def _collect_gaps(dom_name: str, dom_info: dict[str, Any]) -> list[str]:
    gaps = dom_info.get("coverage_gaps", [])
    return list(gaps) if isinstance(gaps, list) else []


_HIGH_IMPACT_PATHS = {
    "polylogue/storage/": "storage_correctness",
    "polylogue/sources/": "parser_correctness",
    "polylogue/pipeline/": "pipeline_correctness",
    "polylogue/cli/": "cli_surface",
    "polylogue/mcp/": "mcp_surface",
    "polylogue/site/": "site_publication",
    "polylogue/lib/security.py": "security_privacy",
    "polylogue/schemas/": "schema_correctness",
    "tests/": "test_quality",
    "docs/": "docs_media",
    "devtools/": "operational_resilience",
    "pyproject.toml": "distribution",
    "flake.nix": "distribution",
    "polylogue/products/": "surface_parity",
}


def _route_paths_to_domains(
    paths: list[str],
    domains_data: dict[str, dict[str, Any]],
    domain_claims: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    affected: dict[str, dict[str, Any]] = {}
    for p in paths:
        for prefix, dom in _HIGH_IMPACT_PATHS.items():
            if p.startswith(prefix):
                if dom not in affected:
                    dom_info = domains_data.get(dom, {})
                    affected[dom] = {
                        "domain": dom,
                        "description": dom_info.get("description", "") if isinstance(dom_info, dict) else "",
                        "paths": [],
                        "impact": _impact_level(dom, domains_data),
                        "claims_affected": len(domain_claims.get(dom, [])),
                    }
                affected[dom]["paths"].append(p)
                break

    return list(affected.values())


def _impact_level(dom: str, domains_data: dict[str, dict[str, Any]]) -> str:
    dom_info = domains_data.get(dom, {})
    if not isinstance(dom_info, dict):
        return "unknown"
    maturity = dom_info.get("maturity", "seed")
    if maturity == "complete":
        return "low"
    if maturity == "established":
        return "medium"
    if maturity == "nascent":
        return "high"
    return "high"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Domain-grouped affected coverage report for vibecode confidence.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--path", action="append", dest="paths", default=None, help="Changed file path (repeatable).")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    report = build_proof_pack(root, changed_paths=args.paths)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human_report(report)

    return 0


def _print_human_report(report: dict[str, Any]) -> None:
    headline = report["catalog_headline"]
    print(
        f"Proof catalog: {headline['claim_count']} claims, "
        f"{headline['total_obligations']} obligations "
        f"({headline['proof_obligations']} proof, "
        f"{headline['ceremonial_obligations']} ceremonial)"
    )
    print()

    coverage = report.get("domain_coverage", [])
    if isinstance(coverage, list):
        for d in coverage:
            if not isinstance(d, dict):
                continue
            ocounts = d.get("oracle_counts", {})
            oracle_str = ", ".join(f"{k}:{v}" for k, v in sorted(ocounts.items()) if v > 0)
            print(f"  {d['domain']:<30} {d['maturity']:<11} {d['claim_count']:>3} claims  oracles=[{oracle_str}]")
            gaps = d.get("gaps", [])
            if gaps:
                for g in gaps:
                    print(f"    ─ gap: {g}")

    affected = report.get("affected_domains", [])
    if affected and isinstance(affected, list):
        print("\nAffected domains:")
        for a in affected:
            if not isinstance(a, dict):
                continue
            paths = a.get("paths", [])
            print(
                f"  {a['domain']} ({a.get('impact', 'unknown')} impact): "
                f"{len(paths)} path(s), {a.get('claims_affected', 0)} claims"
            )


if __name__ == "__main__":
    sys.exit(main())
