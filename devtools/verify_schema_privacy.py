"""Run the required committed-schema privacy registry check."""

from __future__ import annotations

import argparse

from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify privacy of committed schema bundles.")
    parser.parse_args(argv)
    report = audit_schema_bundle_privacy()
    print(report.format_text())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
