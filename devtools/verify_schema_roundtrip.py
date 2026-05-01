"""Verify committed provider schema package roundtrips."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from polylogue.schemas.packages import SchemaVersionPackage
from polylogue.schemas.runtime_registry import SCHEMA_DIR, SchemaRegistry


@dataclass
class ProviderRoundtripResult:
    provider: str
    package_count: int = 0
    element_count: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "package_count": self.package_count,
            "element_count": self.element_count,
            "failures": self.failures,
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


def build_report(*, provider: str | None = None, all_providers: bool = False) -> dict[str, Any]:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    providers = [provider] if provider else registry.list_providers() if all_providers else []
    if not providers:
        return {
            "ok": False,
            "provider_count": 0,
            "package_count": 0,
            "element_count": 0,
            "failures": ["choose --provider NAME or --all"],
            "providers": [],
        }
    results = [verify_provider_roundtrip(item, registry=registry) for item in providers]
    failures = [failure for item in results for failure in item.failures]
    return {
        "ok": not failures,
        "provider_count": len(results),
        "package_count": sum(item.package_count for item in results),
        "element_count": sum(item.element_count for item in results),
        "failures": failures,
        "providers": [item.to_dict() for item in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify schema inference-validation roundtrip.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", help="Provider to verify.")
    group.add_argument("--all", action="store_true", help="Verify all committed provider schema packages.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = build_report(provider=args.provider, all_providers=bool(args.all))
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
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
