"""Run the committed-schema provider-identity check (polylogue-n61h5)."""

from __future__ import annotations

import argparse

from polylogue.schemas.provider_identity_audit import audit_committed_provider_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify each committed schema package describes its own subject.")
    parser.parse_args(argv)
    report = audit_committed_provider_identity()
    print(report.format_text())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
