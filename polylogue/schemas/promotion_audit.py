"""Promotion-time privacy and validity audit for staged schema artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import jsonschema

from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.field_stats.detection import is_dynamic_key

AuditSeverity: TypeAlias = Literal["blocker", "review"]

_REVIEW_FIELDS = frozenset(
    {
        "bundle_scopes",
        "dominant_keys",
        "privacy_approved_values",
        "profile_tokens",
        "representative_paths",
        "x-polylogue-values",
    }
)
_SECRET_PATTERNS = {
    "anthropic_api_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential_url": re.compile(r"(?i)https?://[^\s/:]+:[^\s/@]+@"),
}
_EMAIL = re.compile(r"(?i)^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$")
_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
_URL = re.compile(r"(?i)^https?://")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)")
_IDENTIFIER = re.compile(r"^(?:[0-9a-f]{16,}|[0-9a-f]{8}-[0-9a-f-]{27,}|(?:rollout|session|agent)-)", re.I)


@dataclass(frozen=True, order=True)
class PromotionAuditFinding:
    """One deterministic blocker or operator-review observation."""

    severity: AuditSeverity
    category: str
    artifact: str
    json_path: str
    value: str


@dataclass(frozen=True)
class PromotionAuditReport:
    """Complete promotion verdict and review inventory."""

    root: str
    artifact_count: int
    findings: tuple[PromotionAuditFinding, ...]

    @property
    def blockers(self) -> tuple[PromotionAuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "blocker")

    @property
    def review_items(self) -> tuple[PromotionAuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "review")

    def to_payload(self) -> JSONDocument:
        category_counts = Counter(f"{item.severity}:{item.category}" for item in self.findings)
        return json_document(
            {
                "audit_version": 1,
                "root": self.root,
                "artifact_count": self.artifact_count,
                "verdict": "blocked" if self.blockers else "review_required",
                "blocker_count": len(self.blockers),
                "review_count": len(self.review_items),
                "category_counts": dict(sorted(category_counts.items())),
                "findings": [asdict(item) for item in self.findings],
            }
        )


def _load_artifact(path: Path) -> JSONValue:
    opener = gzip.open if path.name.endswith(".json.gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value: JSONValue = json.load(stream)
    return value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for child in value for text in _strings(child)]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _strings(child)]
    return []


def _review_category(field: str, value: str) -> str:
    if field == "representative_paths" or _ABSOLUTE_PATH.search(value):
        return "filesystem_path"
    if field == "bundle_scopes" or _IDENTIFIER.search(value):
        return "identifier"
    if _EMAIL.fullmatch(value):
        return "email_or_account"
    if _URL.search(value):
        return "url_or_domain"
    if _DATE.search(value):
        return "date_or_time"
    if field in {"dominant_keys", "profile_tokens"}:
        return "structural_vocabulary"
    return "approved_readable_value"


def _unsafe_profile_token(value: str) -> bool:
    if ":" not in value:
        return False
    token_kind, _, observed_name = value.rpartition(":")
    return token_kind.startswith(("child:", "field:", "item:")) and is_dynamic_key(observed_name)


def _secret_findings(*, artifact: str, json_path: str, value: str) -> list[PromotionAuditFinding]:
    findings = []
    for category, pattern in _SECRET_PATTERNS.items():
        if pattern.search(value):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category=category,
                    artifact=artifact,
                    json_path=json_path,
                    value=f"sha256:{digest};length={len(value)}",
                )
            )
    return findings


def _walk_artifact(
    value: object,
    *,
    artifact: str,
    json_path: str,
    findings: list[PromotionAuditFinding],
) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for name in properties:
                property_path = f"{json_path}.properties[{name!r}]"
                secret_findings = _secret_findings(artifact=artifact, json_path=property_path, value=name)
                findings.extend(secret_findings)
                if is_dynamic_key(name):
                    findings.append(
                        PromotionAuditFinding(
                            severity="blocker",
                            category="unsafe_property_name",
                            artifact=artifact,
                            json_path=property_path,
                            value=secret_findings[0].value if secret_findings else name,
                        )
                    )
        for key, child in value.items():
            child_path = f"{json_path}.{key}"
            if key in _REVIEW_FIELDS:
                for text in _strings(child):
                    secret_findings = _secret_findings(artifact=artifact, json_path=child_path, value=text)
                    findings.extend(secret_findings)
                    if secret_findings:
                        continue
                    if key == "profile_tokens" and _unsafe_profile_token(text):
                        findings.append(
                            PromotionAuditFinding(
                                severity="blocker",
                                category="unsafe_structural_identifier",
                                artifact=artifact,
                                json_path=child_path,
                                value=text,
                            )
                        )
                        continue
                    findings.append(
                        PromotionAuditFinding(
                            severity="review",
                            category=_review_category(key, text),
                            artifact=artifact,
                            json_path=child_path,
                            value=text,
                        )
                    )
            _walk_artifact(child, artifact=artifact, json_path=child_path, findings=findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_artifact(child, artifact=artifact, json_path=f"{json_path}[{index}]", findings=findings)


def audit_schema_artifacts(root: Path) -> PromotionAuditReport:
    """Audit every JSON/gzip-JSON artifact below ``root`` without mutating it."""
    resolved = root.expanduser().resolve()
    artifacts = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz"))
    )
    findings: list[PromotionAuditFinding] = []
    for path in artifacts:
        relative = str(path.relative_to(resolved))
        try:
            payload = _load_artifact(path)
        except Exception as error:
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category="malformed_artifact",
                    artifact=relative,
                    json_path="$",
                    value=f"{type(error).__name__}: {error}",
                )
            )
            continue
        if path.name.endswith(".schema.json.gz"):
            try:
                jsonschema.Draft202012Validator.check_schema(payload)
            except Exception as error:
                findings.append(
                    PromotionAuditFinding(
                        severity="blocker",
                        category="invalid_json_schema",
                        artifact=relative,
                        json_path="$",
                        value=f"{type(error).__name__}: {error}",
                    )
                )
        _walk_artifact(payload, artifact=relative, json_path="$", findings=findings)
    return PromotionAuditReport(
        root=str(resolved),
        artifact_count=len(artifacts),
        findings=tuple(sorted(set(findings))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit staged provider schemas before promotion")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_schema_artifacts(args.root)
    rendered = json.dumps(report.to_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if report.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PromotionAuditFinding", "PromotionAuditReport", "audit_schema_artifacts", "main"]
