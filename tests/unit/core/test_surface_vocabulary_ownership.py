"""Each surface vocabulary has one owner that generates its SQL CHECK.

polylogue-ir1wn: "which surface" was declared six times with no generator
tying any pair together, and the two DDL copies were hand-written restatements
of the Python tokens. Two of the six declarations are now gone entirely
(``operations/candidate_build.py`` and ``maintenance/envelope.py`` no longer
exist), a third (``production_evaluator.Surface``) was dead and spelled the
daemon HTTP route ``daemon-web`` -- a token no writer ever produced.

Anti-vacuity: re-spelling either CHECK by hand, or adding a member to one
Literal without the DDL following, makes this red.
"""

from __future__ import annotations

from typing import get_args

from polylogue.core.enums import (
    PRINCIPAL_SURFACE_VALUES,
    TELEMETRY_SURFACE_VALUES,
    PrincipalSurface,
    TelemetrySurface,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL


def _check_line(ddl: str, column: str) -> str:
    for line in ddl.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{column} ") and "CHECK" in stripped:
            return stripped
    raise AssertionError(f"no CHECK on {column}")


def test_audit_principal_surface_check_is_generated_from_the_literal() -> None:
    line = _check_line(AUDIT_DDL, "principal_surface")
    for value in get_args(PrincipalSurface):
        assert f"'{value}'" in line, value
    assert line.count("'") == 2 * len(PRINCIPAL_SURFACE_VALUES)


def test_ops_surface_check_is_generated_from_the_literal() -> None:
    line = _check_line(OPS_DDL, "surface")
    for value in get_args(TelemetrySurface):
        assert f"'{value}'" in line, value
    assert line.count("'") == 2 * len(TELEMETRY_SURFACE_VALUES)


def test_the_dead_daemon_web_spelling_is_gone() -> None:
    assert "daemon-web" not in TELEMETRY_SURFACE_VALUES
    assert "daemon-http" in TELEMETRY_SURFACE_VALUES
    import polylogue.archive.query.production_evaluator as production_evaluator

    assert not hasattr(production_evaluator, "Surface")


def test_the_two_vocabularies_are_declared_distinct_on_purpose() -> None:
    """They are not joinable: one names the asking authority, one the transport."""
    assert PRINCIPAL_SURFACE_VALUES != TELEMETRY_SURFACE_VALUES
    assert "daemon" in PRINCIPAL_SURFACE_VALUES
    assert "daemon" not in TELEMETRY_SURFACE_VALUES
