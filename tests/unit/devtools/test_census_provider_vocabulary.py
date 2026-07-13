"""Tests for the provider/origin vocabulary census (polylogue-9e5.8 Step 0).

The census is the "scripted, AST-level census (not another manual rg pass)"
the bead's design field calls for -- these tests prove it actually catches
each of the four leak categories on synthetic fixture trees (not the real
repo, so a future edit to real source can't accidentally make the fixture
tests vacuous), correctly excludes Tier-A packages and test files, honors
the allowlist, and rejects a stale allowlist entry.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from devtools import census_provider_vocabulary


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _run_json(
    root: Path, capsys: pytest.CaptureFixture[str], *, allowlist: Path | None = None, check: bool = False
) -> dict[str, Any]:
    args = ["--json", "--root", str(root)]
    if allowlist is not None:
        args += ["--allowlist", str(allowlist)]
    if check:
        args += ["--check"]
    rc = census_provider_vocabulary.main(args)
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    payload["_rc"] = rc
    return payload


def test_flags_function_parameter_named_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/api/example.py",
        """
        def list_sessions(provider: str | None = None, providers: list[str] | None = None) -> list[str]:
            return []
        """,
    )

    payload = _run_json(tmp_path, capsys)

    params = [s for s in payload["unallowlisted"] if s["category"] == "param"]
    assert {s["identifier"] for s in params} == {"provider", "providers"}
    assert payload["counts_by_category"]["param"] == 2


def test_flags_dataclass_field_named_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_models.py",
        """
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class SessionRecordQuery:
            provider: str | None = None
            unrelated: int = 0
        """,
    )

    payload = _run_json(tmp_path, capsys)

    fields = [s for s in payload["unallowlisted"] if s["category"] == "field"]
    assert len(fields) == 1
    assert fields[0]["identifier"] == "provider"
    assert fields[0]["qualname"] == "<module>.SessionRecordQuery"


def test_flags_dict_and_set_literal_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/daemon/example_http.py",
        """
        _SCOPE_FILTER_KEYS = frozenset({"session_ids", "provider", "source_family"})


        def to_dict(self):
            return {"provider": self.provider, "unrelated": 1}
        """,
    )

    payload = _run_json(tmp_path, capsys)

    keys = [s for s in payload["unallowlisted"] if s["category"] == "key"]
    assert len(keys) == 2
    assert {s["identifier"] for s in keys} == {"provider"}


def test_flags_cli_flag_and_http_route_literals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/cli/example_check_options.py",
        """
        SCHEMA_PROVIDER_FLAG = "--schema-provider"
        ROUTE = "/api/some-provider-scoped-thing"
        UNRELATED_FLAG = "--format"
        """,
    )

    payload = _run_json(tmp_path, capsys)

    literals = [s for s in payload["unallowlisted"] if s["category"] == "literal"]
    assert {s["identifier"] for s in literals} == {"--schema-provider", "/api/some-provider-scoped-thing"}


def test_ignores_explicitly_deprecated_click_option_aliases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Deprecated compatibility aliases do not keep the active CLI count high."""
    _write(
        tmp_path,
        "polylogue/cli/example_check_options.py",
        """
        import click


        ACTIVE = click.option("--schema-origin")
        LEGACY = click.option("--schema-provider", deprecated="Use --schema-origin instead.")
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["counts_by_category"]["literal"] == 0
    assert payload["unallowlisted"] == []


def test_compound_provider_identifiers_are_not_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """VectorProvider/vector_provider/create_vector_provider-style compound
    names are exact-match excluded by construction (polylogue-9e5.8 Axis-2
    exclusion #1) -- no allowlist entry needed for them."""
    _write(
        tmp_path,
        "polylogue/storage/example_search_providers.py",
        """
        class VectorProvider:
            def __init__(self, vector_provider=None):
                self.vector_provider = vector_provider


        def create_vector_provider(backend):
            return VectorProvider(vector_provider=backend)
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["sites_scanned"] == 0
    assert payload["unallowlisted"] == []


def test_sources_schemas_pipeline_browser_capture_and_tests_are_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for relative in (
        "polylogue/sources/example_parser.py",
        "polylogue/schemas/example_schema.py",
        "polylogue/pipeline/example_ids.py",
        "polylogue/browser_capture/example_models.py",
        "tests/unit/api/test_example.py",
    ):
        _write(
            tmp_path,
            relative,
            """
            def handler(provider: str) -> str:
                return provider
            """,
        )

    payload = _run_json(tmp_path, capsys)

    assert payload["sites_scanned"] == 0


def test_allowlisted_site_passes_and_check_flag_is_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/cost/example_plans.py",
        """
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class SubscriptionPlan:
            provider: str
        """,
    )
    allowlist = tmp_path / "docs" / "plans" / "provider-vocabulary-exclusions.yaml"
    _write(
        tmp_path,
        "docs/plans/provider-vocabulary-exclusions.yaml",
        """
        entries:
        - path: polylogue/cost/example_plans.py
          category: field
          identifier: provider
          occurrence: 0
          reason: 'Free-text LiteLLM billing-catalog vendor label, ordinary-English "provider".'
        """,
    )

    payload = _run_json(tmp_path, capsys, allowlist=allowlist, check=True)

    assert payload["_rc"] == 0
    assert payload["ok"] is True
    assert payload["allowlisted"] == 1
    assert payload["unallowlisted"] == []
    assert payload["stale_allowlist_entries"] == []


def test_stale_allowlist_entry_fails_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An allowlist entry with no matching site (the field was renamed or
    removed) must fail --check -- otherwise the allowlist only ever grows."""
    _write(
        tmp_path,
        "polylogue/cost/example_plans.py",
        """
        def unrelated() -> None:
            return None
        """,
    )
    allowlist = tmp_path / "docs" / "plans" / "provider-vocabulary-exclusions.yaml"
    _write(
        tmp_path,
        "docs/plans/provider-vocabulary-exclusions.yaml",
        """
        entries:
        - path: polylogue/cost/example_plans.py
          category: field
          identifier: provider
          occurrence: 0
          reason: 'No longer matches any field in this file.'
        """,
    )

    payload = _run_json(tmp_path, capsys, allowlist=allowlist, check=True)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    assert len(payload["stale_allowlist_entries"]) == 1


def test_vacuous_zero_site_scan_fails_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A --check run that finds zero sites anywhere is far more likely a
    broken scanner (wrong root, exclusion regression) than a fully retired
    vocabulary -- the bead's own AC requires the tool prove it is "wired to
    real code, not vacuous"."""
    _write(
        tmp_path,
        "polylogue/api/example_unrelated.py",
        """
        def noop() -> None:
            return None
        """,
    )

    payload = _run_json(tmp_path, capsys, check=True)

    assert payload["sites_scanned"] == 0
    assert payload["_rc"] == 1
    assert payload["ok"] is False


def test_real_repo_allowlist_is_internally_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed docs/plans/provider-vocabulary-exclusions.yaml must
    currently match the real repo: no stale entries, and a non-zero,
    non-vacuous scan. This is the gate a CI wiring of --check would run."""
    rc = census_provider_vocabulary.main(["--json", "--check"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["stale_allowlist_entries"] == []
    assert payload["sites_scanned"] > 0
    # Sanity: the census should surface all four categories on the real repo,
    # and the documented Axis-2 exclusion classes should show up allowlisted.
    assert all(count > 0 for count in payload["counts_by_category"].values())
    assert payload["allowlisted"] >= 9
