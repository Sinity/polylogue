"""Witness discovery pytest plugin.

Captures test failures as witnesses in the local witness inbox.
Loaded by conftest.py via pytest_plugins.
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest

from polylogue.proof.witnesses import LOCAL_WITNESS_INBOX, WITNESS_SCHEMA_VERSION


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Generator[None, None, None]:
    """Capture test failures as witnesses in the local witness inbox.

    On test failure, writes a minimal witness file with the test's nodeid,
    error message, and traceback.  The witness inbox is non-empty after the
    first failure, satisfying the automated discovery path requirement.
    """
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]  # hookwrapper outcome is a pluggy._Result
    if report.when == "call" and report.failed:
        _save_failure_witness(item, call)


def _save_failure_witness(item: pytest.Item, call: pytest.CallInfo[Any]) -> None:
    """Write a lightweight witness record for a test failure."""
    # Build a filesystem-safe slug from the test nodeid
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", item.nodeid)[:96]

    error_msg = ""
    if call.excinfo is not None:
        ex = call.excinfo
        error_msg = f"{ex.type.__name__}: {ex.value}" if ex.value else str(ex.type)

    traceback_text = ""
    if call.excinfo is not None:
        import traceback as _tb

        traceback_text = "".join(_tb.format_exception(call.excinfo.type, call.excinfo.value, call.excinfo.tb))

    LOCAL_WITNESS_INBOX.mkdir(parents=True, exist_ok=True)

    witness_content = {
        "nodeid": item.nodeid,
        "error": error_msg,
        "traceback": traceback_text,
    }
    witness_path = LOCAL_WITNESS_INBOX / f"{slug}.witness.txt"
    witness_path.write_text(_json.dumps(witness_content, indent=2), encoding="utf-8")

    metadata = {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "witness_id": slug,
        "path": str(witness_path),
        "origin": "regression",
        "committed": False,
        "provenance": {
            "discovery_hook": "pytest_runtest_makereport",
            "discovery_cwd": str(item.config.rootpath),
        },
        "preserved_semantic_facts": [error_msg] if error_msg else [],
        "minimization_status": "raw",
        "privacy": {
            "private_material": "not_observed",
            "transformed": False,
            "redacted": False,
            "discarded": False,
            "retained": False,
            "notes": [],
        },
        "privacy_classification": "synthetic",
        "lifecycle": {
            "state": "discovered",
            "discovered_at": datetime.now(tz=timezone.utc).isoformat(),
            "minimized_at": None,
            "committed_at": None,
            "last_exercised_at": None,
            "retired_at": None,
            "retirement_reason": None,
        },
        "known_failing": False,
        "xfail_strict": False,
        "rejection_reason": None,
        "notes": [f"Auto-captured from test failure: {item.nodeid}"],
    }
    metadata_path = LOCAL_WITNESS_INBOX / f"{slug}.metadata.json"
    metadata_path.write_text(_json.dumps(metadata, indent=2), encoding="utf-8")
