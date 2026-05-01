"""Coverage-manifest discovery for proof catalog subjects."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from polylogue.lib.json import JSONDocument, JSONValue, require_json_value
from polylogue.proof.models import SourceSpan, SubjectRef

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANS_DIR = _REPO_ROOT / "docs" / "plans"
_COVERAGE_MANIFEST_NAMES = (
    "assurance-domains.yaml",
    "campaign-coverage.yaml",
    "distribution-coverage.yaml",
    "docs-media-coverage.yaml",
    "evidence-freshness.yaml",
    "oracle-quality.yaml",
    "scenario-coverage.yaml",
    "security-privacy-coverage.yaml",
    "test-quality-coverage.yaml",
)
_DOMAIN_BY_MANIFEST = {
    "assurance-domains": "spec_completeness",
    "campaign-coverage": "mutation_coverage",
    "distribution-coverage": "distribution",
    "docs-media-coverage": "docs_media",
    "evidence-freshness": "spec_accuracy",
    "oracle-quality": "spec_accuracy",
    "scenario-coverage": "scenario_coverage",
    "security-privacy-coverage": "security_privacy",
    "test-quality-coverage": "test_quality",
}
_DOMAIN_BY_GAP_AXIS = {
    "benchmark": "benchmark_coverage",
    "benchmark_coverage": "benchmark_coverage",
    "dependency": "dependency_closure",
    "dependency_audit": "security_privacy",
    "direct_coverage": "test_quality",
    "distribution": "distribution",
    "docs_media": "docs_media",
    "flakiness": "test_quality",
    "fuzz": "test_quality",
    "migration_safety": "migration_safety",
    "mock_depth": "test_quality",
    "performance": "performance",
    "security_privacy": "security_privacy",
    "site_publication": "site_publication",
    "storage_correctness": "storage_correctness",
}
_ITEM_SECTIONS = (
    "artifacts",
    "areas",
    "benchmark_campaigns",
    "dimensions",
    "domains",
    "families",
    "freshness_policies",
    "mutation_campaigns",
    "oracles",
    "surfaces",
)


@dataclass(frozen=True, slots=True)
class CoverageManifestEntry:
    """A parsed assurance coverage manifest."""

    manifest_id: str
    path: Path
    payload: Mapping[str, Any]

    @property
    def repo_path(self) -> str:
        return _repo_relative(self.path)

    @property
    def domain(self) -> str:
        return _DOMAIN_BY_MANIFEST.get(self.manifest_id, "spec_completeness")

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(section for section in _ITEM_SECTIONS if section in self.payload)

    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(_iter_manifest_items(self))

    def gaps(self) -> tuple[dict[str, Any], ...]:
        return tuple(_iter_manifest_gaps(self))


def coverage_manifest_entries(plans_dir: Path = _PLANS_DIR) -> tuple[CoverageManifestEntry, ...]:
    """Parse coverage manifests that feed the assurance proof catalog."""
    entries: list[CoverageManifestEntry] = []
    for name in _COVERAGE_MANIFEST_NAMES:
        path = plans_dir / name
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            payload = {}
        entries.append(CoverageManifestEntry(manifest_id=path.stem, path=path, payload=payload))
    return tuple(entries)


def coverage_manifest_subjects(plans_dir: Path = _PLANS_DIR) -> tuple[SubjectRef, ...]:
    """Compile assurance coverage manifests, items, and known gaps into subjects."""
    subjects: list[SubjectRef] = []
    for entry in coverage_manifest_entries(plans_dir):
        items = entry.items()
        gaps = entry.gaps()
        subjects.append(
            SubjectRef(
                kind="assurance.coverage_manifest",
                id=f"assurance.coverage_manifest.{entry.manifest_id}",
                attrs=_json_document(
                    {
                        "manifest_id": entry.manifest_id,
                        "assurance_domain": entry.domain,
                        "path": entry.repo_path,
                        "sections": list(entry.sections),
                        "item_count": len(items),
                        "coverage_gap_count": len(gaps),
                    }
                ),
                source_span=SourceSpan(path=entry.repo_path, line=1, symbol=entry.manifest_id),
            )
        )
        for item in items:
            subjects.append(_coverage_item_subject(entry, item))
        for gap in gaps:
            subjects.append(_coverage_gap_subject(entry, gap))
    return tuple(sorted(subjects, key=lambda subject: subject.id))


def _iter_manifest_items(entry: CoverageManifestEntry) -> Iterator[dict[str, Any]]:
    for section in _ITEM_SECTIONS:
        raw = entry.payload.get(section)
        if isinstance(raw, Mapping):
            for name, payload in raw.items():
                if isinstance(payload, Mapping):
                    yield {"section": section, "name": str(name), "payload": dict(payload)}
                else:
                    yield {"section": section, "name": str(name), "payload": {"value": payload}}
        elif isinstance(raw, list):
            for index, payload in enumerate(raw):
                if isinstance(payload, Mapping):
                    name = str(payload.get("name") or payload.get("artifact") or payload.get("area") or index)
                    yield {"section": section, "name": name, "payload": dict(payload)}
                else:
                    yield {"section": section, "name": str(index), "payload": {"value": payload}}


def _iter_manifest_gaps(entry: CoverageManifestEntry) -> Iterator[dict[str, Any]]:
    raw = entry.payload.get("coverage_gaps")
    if not isinstance(raw, list):
        return
    for index, item in enumerate(raw):
        payload = dict(item) if isinstance(item, Mapping) else {"gap": item}
        axis = _gap_axis(payload)
        yield {
            "index": index,
            "axis": axis,
            "domain": _gap_domain(entry, axis, payload),
            "gap": str(payload.get("gap") or payload.get("note") or payload.get("value") or ""),
            "owner": str(payload.get("owner") or ""),
            "next_evidence": str(payload.get("next_evidence") or ""),
            "payload": payload,
        }


def _coverage_item_subject(entry: CoverageManifestEntry, item: Mapping[str, Any]) -> SubjectRef:
    section = str(item.get("section") or "items")
    name = str(item.get("name") or "unnamed")
    raw_payload = item.get("payload")
    payload: Mapping[str, Any] = (
        {str(key): value for key, value in raw_payload.items()} if isinstance(raw_payload, Mapping) else {}
    )
    domain = _item_domain(entry, section, payload)
    source_path = _source_path_for_payload(payload) or entry.repo_path
    subject_id = f"assurance.coverage_item.{entry.manifest_id}.{section}.{_slug(name)}"
    attrs = _json_document(
        {
            "manifest_id": entry.manifest_id,
            "assurance_domain": domain,
            "section": section,
            "name": name,
            "path": source_path,
            "status": _status_for_payload(payload),
            "has_automated_gate": _has_automated_gate(payload),
            "declared_gap": bool(_string_value(payload.get("gap")) or _string_value(payload.get("notes"))),
        }
    )
    return SubjectRef(
        kind="assurance.coverage_item",
        id=subject_id,
        attrs=attrs,
        source_span=SourceSpan(path=source_path, line=1, symbol=f"{entry.manifest_id}.{section}.{name}"),
    )


def _coverage_gap_subject(entry: CoverageManifestEntry, gap: Mapping[str, Any]) -> SubjectRef:
    index = int(gap.get("index") or 0)
    axis = str(gap.get("axis") or "gap")
    domain = str(gap.get("domain") or entry.domain)
    subject_id = f"assurance.coverage_gap.{entry.manifest_id}.{index:03d}.{_slug(axis)}"
    attrs = _json_document(
        {
            "manifest_id": entry.manifest_id,
            "assurance_domain": domain,
            "axis": axis,
            "gap": str(gap.get("gap") or ""),
            "owner": str(gap.get("owner") or ""),
            "next_evidence": str(gap.get("next_evidence") or ""),
            "path": entry.repo_path,
        }
    )
    return SubjectRef(
        kind="assurance.coverage_gap",
        id=subject_id,
        attrs=attrs,
        source_span=SourceSpan(path=entry.repo_path, line=1, symbol=subject_id),
    )


def _item_domain(entry: CoverageManifestEntry, section: str, payload: Mapping[str, Any]) -> str:
    if section == "benchmark_campaigns":
        return "benchmark_coverage"
    if section == "mutation_campaigns":
        return "mutation_coverage"
    raw_domain = payload.get("domain") or payload.get("subject") or payload.get("area") or payload.get("dimension")
    if isinstance(raw_domain, str):
        normalized = raw_domain.replace("-", "_")
        return _DOMAIN_BY_GAP_AXIS.get(normalized, normalized)
    return entry.domain


def _gap_domain(entry: CoverageManifestEntry, axis: str, payload: Mapping[str, Any]) -> str:
    explicit = payload.get("domain")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().replace("-", "_")
    return _DOMAIN_BY_GAP_AXIS.get(axis.replace("-", "_"), entry.domain)


def _gap_axis(payload: Mapping[str, Any]) -> str:
    for key in ("domain", "subject", "area", "dimension", "artifact", "platform", "concern"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("-", "_")
    return "gap"


def _source_path_for_payload(payload: Mapping[str, Any]) -> str | None:
    for key in ("path", "location", "config_location"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.split(" ", maxsplit=1)[0].strip()
        if candidate and candidate != "null":
            return candidate
    tests = payload.get("tests")
    if isinstance(tests, list) and tests:
        first = tests[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _status_for_payload(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("status"), str):
        return str(payload["status"])
    if payload.get("implemented") is True:
        return "implemented"
    if payload.get("implemented") is False:
        return "missing"
    if payload.get("measured") is True:
        return "measured"
    if payload.get("measured") is False:
        return "unmeasured"
    if payload.get("oracle_present") is True:
        return "oracle-present"
    if payload.get("oracle_present") is False:
        return "oracle-missing"
    return "declared"


def _has_automated_gate(payload: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get(key))
        for key in (
            "ci_build",
            "ci_gate",
            "ci_present",
            "ci_test",
            "generated_by",
            "verified_by",
            "test_coverage",
            "tests",
            "tool",
        )
    )


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-") or "unnamed"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _json_document(items: Mapping[str, object]) -> JSONDocument:
    return {str(key): _json_value(value) for key, value in items.items()}


def _json_value(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(child) for child in value]
    return require_json_value(value, context="coverage manifest")


__all__ = [
    "CoverageManifestEntry",
    "coverage_manifest_entries",
    "coverage_manifest_subjects",
]
