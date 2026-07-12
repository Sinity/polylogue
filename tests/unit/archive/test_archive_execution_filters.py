"""Archive execution adapter filter coverage."""

from __future__ import annotations

from polylogue.archive.query.archive_execution import _plan_filter_kwargs, _provider_for_origin
from polylogue.archive.query.expression import compile_expression
from polylogue.core.enums import Provider


def test_archive_filter_kwargs_include_session_id() -> None:
    plan = compile_expression("id:abc123").to_plan()

    assert _plan_filter_kwargs(plan)["session_id"] == "abc123"


# ---------------------------------------------------------------------------
# GEMINI/DRIVE reverse-lookup disambiguation (polylogue-4rrv,
# polylogue-9e5.8 Step 5)
# ---------------------------------------------------------------------------


def test_provider_for_origin_defaults_to_canonical_gemini() -> None:
    """Unchanged default: no hint means the documented canonical choice."""
    assert _provider_for_origin("aistudio-drive") is Provider.GEMINI


def test_provider_for_origin_family_hint_disambiguates_drive() -> None:
    """When a caller has independent knowledge of the acquisition mechanism
    (e.g. Provider.DRIVE from live Google-Drive acquisition context), this
    reverse lookup now correctly recovers it instead of always collapsing to
    Provider.GEMINI."""
    assert _provider_for_origin("aistudio-drive", family_hint=Provider.DRIVE) is Provider.DRIVE
    assert _provider_for_origin("aistudio-drive", family_hint="drive-takeout") is Provider.DRIVE


def test_provider_for_origin_family_hint_confirms_gemini() -> None:
    assert _provider_for_origin("aistudio-drive", family_hint=Provider.GEMINI) is Provider.GEMINI
    assert _provider_for_origin("aistudio-drive", family_hint="gemini-export") is Provider.GEMINI


def test_provider_for_origin_unrelated_origin_ignores_hint() -> None:
    """A hint outside the target origin's fiber is ignored, not honored."""
    assert _provider_for_origin("codex-session", family_hint=Provider.GEMINI) is Provider.CODEX


def test_provider_for_origin_grok_still_resolves_correctly() -> None:
    """polylogue-9e5.8 Step 1 regression: grok-export must not silently fall
    back to Provider.UNKNOWN (the bug the dedup fixed)."""
    assert _provider_for_origin("grok-export") is Provider.GROK
