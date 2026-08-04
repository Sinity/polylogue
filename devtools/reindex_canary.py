"""Developer-tool adapter for the product reindex canary command."""

from __future__ import annotations

import sys

from devtools.cli_boundary import invoke_polylogue_cli
from polylogue.scenarios import polylogue_execution


def main(argv: list[str] | None = None) -> int:
    """Delegate to the real product CLI without reimplementing rebuild logic."""

    forwarded = list(argv or ())
    if "--json" in forwarded:
        forwarded.remove("--json")
        if "--output-format" not in forwarded:
            forwarded.extend(("--output-format", "json"))
    result = invoke_polylogue_cli(
        polylogue_execution("ops", "maintenance", "reindex-canary", *forwarded),
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
