"""Verify committed provider schema package roundtrips with optional corpus-backed validation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from polylogue.schemas.packages import SchemaVersionPackage
from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    jsonschema = None
    Draft202012Validator = None

try:
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS
except ImportError:  # pragma: no cover
    SyntheticCorpus = None  # type: ignore[misc,assignment]
    PROVIDER_WIRE_FORMATS = {}


@dataclass
class ProviderRoundtripResult:
    provider: str
    package_count: int = 0
    element_count: int = 0
    failures: list[str] = field(default_factory=list)
    corpus_failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.corpus_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "package_count": self.package_count,
            "element_count": self.element_count,
            "failures": self.failures,
            "corpus_failures": self.corpus_failures,
        }


def verify_provider_roundtrip(provider: str, *, registry: SchemaRegistry | None = None) -> ProviderRoundtripResult:
    """Verify that one committed provider schema package roundtrips."""
    schema_registry = registry or SchemaRegistry(storage_root=SCHEMA_DIR)
    result = ProviderRoundtripResult(provider=provider)
    versions = schema_registry.list_versions(provider)
    if not versions:
        result.failures.append("no package versions registered")
        return result

    for version in versions:
        package = schema_registry.get_package(provider, version=version)
        if package is None:
            result.failures.append(f"{version}: package not loadable")
            continue
        result.package_count += 1
        reparsed = SchemaVersionPackage.from_dict(package.to_dict())
        if reparsed.to_dict() != package.to_dict():
            result.failures.append(f"{version}: manifest roundtrip changed payload")
        if not package.elements:
            result.failures.append(f"{version}: package has no elements")
        for element in package.elements:
            result.element_count += 1
            if not element.schema_file:
                if element.supported:
                    result.failures.append(f"{version}/{element.element_kind}: supported element has no schema_file")
                continue
            schema = schema_registry.get_element_schema(provider, version=version, element_kind=element.element_kind)
            if schema is None:
                result.failures.append(f"{version}/{element.element_kind}: schema file not loadable")
                continue
            if not isinstance(schema.get("type"), str):
                result.failures.append(f"{version}/{element.element_kind}: schema has no root type")
    return result


def _verify_corpus_roundtrip_for_provider(
    provider: str,
    *,
    registry: SchemaRegistry | None = None,
    count: int = 3,
) -> list[str]:
    """Generate synthetic records from committed schemas and validate them.

    Returns a list of failure messages (empty if all records validate).
    """
    if SyntheticCorpus is None or jsonschema is None:
        return ["synthetic corpus or jsonschema not available (import failed)"]
    if provider not in PROVIDER_WIRE_FORMATS:
        return [f"no wire format configured for provider {provider!r}"]

    schema_registry = registry or SchemaRegistry(storage_root=SCHEMA_DIR)
    failures: list[str] = []

    try:
        corpus = SyntheticCorpus.for_provider(provider)
    except Exception as exc:
        return [f"failed to create synthetic corpus: {exc}"]

    try:
        records = corpus.generate(count=count, seed=42)
    except Exception as exc:
        return [f"failed to generate synthetic records: {exc}"]

    # Determine the wire encoding to know how to parse generated records
    is_jsonl = corpus.wire_format.encoding == "jsonl"

    # Parse each generated record into one or more JSON documents
    parsed_records: list[dict[str, Any]] = []
    for raw_record in records:
        try:
            raw_text = raw_record.decode("utf-8") if isinstance(raw_record, bytes) else str(raw_record)
            if is_jsonl:
                for line in raw_text.strip().splitlines():
                    line = line.strip()
                    if line:
                        parsed_records.append(json.loads(line))
            else:
                parsed_records.append(json.loads(raw_text))
        except Exception as exc:
            failures.append(f"failed to parse generated record: {exc}")

    if not parsed_records:
        failures.append("no parsable records generated")
        return failures

    # Validate each parsed record against the provider element schemas
    versions = schema_registry.list_versions(provider)
    for version in versions:
        package = schema_registry.get_package(provider, version=version)
        if package is None:
            continue
        for element in package.elements:
            if not element.schema_file:
                continue
            schema = schema_registry.get_element_schema(provider, version=version, element_kind=element.element_kind)
            if schema is None:
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except jsonschema.SchemaError as exc:
                failures.append(f"{version}/{element.element_kind}: schema is not valid JSON Schema: {exc.message}")
                continue

            # Validate each parsed record against this element schema
            for idx, data in enumerate(parsed_records):
                try:
                    Draft202012Validator(schema).validate(data)
                except jsonschema.ValidationError as exc:
                    failures.append(
                        f"{version}/{element.element_kind}: record {idx} failed schema validation: {exc.message}"
                    )
                except Exception as exc:
                    failures.append(f"{version}/{element.element_kind}: record {idx} unexpected error: {exc}")

    return failures


def build_report(
    *,
    provider: str | None = None,
    all_providers: bool = False,
    corpus: bool = False,
    corpus_count: int = 3,
) -> dict[str, Any]:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    providers = [provider] if provider else registry.list_providers() if all_providers else []
    if not providers:
        return {
            "ok": False,
            "provider_count": 0,
            "package_count": 0,
            "element_count": 0,
            "failures": ["choose --provider NAME or --all"],
            "corpus_failures": [],
            "providers": [],
        }
    results = [verify_provider_roundtrip(item, registry=registry) for item in providers]
    failures = [failure for item in results for failure in item.failures]

    all_corpus_failures: list[str] = []
    if corpus:
        for item in providers:
            item_failures = _verify_corpus_roundtrip_for_provider(item, registry=registry, count=corpus_count)
            all_corpus_failures.extend(item_failures)
            # Attach corpus failures to the matching result
            for result in results:
                if result.provider == item:
                    result.corpus_failures = item_failures
                    break

    return {
        "ok": not failures and not all_corpus_failures,
        "provider_count": len(results),
        "package_count": sum(item.package_count for item in results),
        "element_count": sum(item.element_count for item in results),
        "failures": failures,
        "corpus_failures": all_corpus_failures,
        "providers": [item.to_dict() for item in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify schema inference-validation roundtrip.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", help="Provider to verify.")
    group.add_argument("--all", action="store_true", help="Verify all committed provider schema packages.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="Also generate synthetic corpus records and validate them against committed element schemas.",
    )
    parser.add_argument(
        "--corpus-count",
        type=int,
        default=3,
        help="Number of synthetic records to generate per provider (default: 3, used with --corpus).",
    )
    args = parser.parse_args(argv)

    report = build_report(
        provider=args.provider,
        all_providers=bool(args.all),
        corpus=bool(args.corpus),
        corpus_count=args.corpus_count,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        status = "ok" if report["ok"] else "failed"
        print(
            f"schema roundtrip: {status}; "
            f"{report['provider_count']} provider(s), "
            f"{report['package_count']} package(s), "
            f"{report['element_count']} element(s)"
        )
        for failure in report["failures"]:
            print(f"  - {failure}")
        if report["corpus_failures"]:
            print(f"\ncorpus roundtrip failures ({len(report['corpus_failures'])}):")
            for failure in report["corpus_failures"]:
                print(f"  - {failure}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
