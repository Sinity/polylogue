"""Pipeline must not import from CLI surfaces (#430).

Topology rule: surfaces present existing meaning; substrate owns it. A
pipeline module importing from ``polylogue.cli.*`` inverts that rule and
creates a substrate↔surface cycle that ``devtools verify-cluster-cohesion``
flags.
"""

from __future__ import annotations

import re
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[3] / "polylogue" / "pipeline"

_CLI_IMPORT_RE = re.compile(
    r"""(?mx)                # multiline + verbose
    ^(?:\s*)                 # leading whitespace
    (?:
        from\s+polylogue\.cli\b
      |
        import\s+polylogue\.cli\b
    )
    """
)


def _find_cli_imports(source: str) -> list[str]:
    return [match.group(0).strip() for match in _CLI_IMPORT_RE.finditer(source)]


def test_pipeline_does_not_import_cli() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(PIPELINE_ROOT.rglob("*.py")):
        rel = path.relative_to(PIPELINE_ROOT.parents[1])
        text = path.read_text(encoding="utf-8")
        violations = _find_cli_imports(text)
        if violations:
            offenders[str(rel)] = violations
    assert not offenders, "polylogue/pipeline/* must not import from polylogue/cli/*; offenders: " + repr(offenders)
