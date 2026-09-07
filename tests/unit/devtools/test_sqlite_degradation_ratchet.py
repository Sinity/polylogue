"""The layering gate counts improvised sqlite degradation policy per file.

Anti-vacuity: drop the per-file comparison (accept any count), or stop counting
``except sqlite3.*`` handlers, and ``test_a_new_handler_fails_the_gate`` passes
a repo that grew one.
"""

from __future__ import annotations

import json
from pathlib import Path

from devtools import repo_root
from devtools.sqlite_degradation import census_sqlite_degradation_sites, load_sqlite_degradation_baseline
from devtools.verify_layering import _sqlite_degradation_findings

BASELINE_REF = "docs/plans/sqlite-degradation-baseline.json"


def _manifest(baseline: str = BASELINE_REF, roots: tuple[str, ...] = ("polylogue",)) -> dict[str, object]:
    return {"sqlite_degradation": {"baseline": baseline, "roots": list(roots)}}


def _write_module(path: Path, handlers: int) -> None:
    body = "import sqlite3\n\n\ndef read() -> int:\n"
    for index in range(handlers):
        body += f"    try:\n        pass\n    except sqlite3.Error:\n        return {index}\n"
    body += "    return -1\n"
    path.write_text(body, encoding="utf-8")


def test_census_counts_only_sqlite_handlers(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 2)
    (package / "other.py").write_text(
        "import sqlite3\n\n\ndef f() -> None:\n    try:\n        pass\n    except OSError:\n        pass\n",
        encoding="utf-8",
    )

    counts = census_sqlite_degradation_sites(tmp_path, ("pkg",))

    assert counts == {"pkg/reader.py": 2}


def test_a_new_handler_fails_the_gate(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 2)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"sites": {"pkg/reader.py": 2}}), encoding="utf-8")
    manifest = _manifest(baseline="baseline.json", roots=("pkg",))

    violations, shrunk = _sqlite_degradation_findings(tmp_path, manifest)
    assert violations == []
    assert shrunk == []

    _write_module(package / "reader.py", 3)
    violations, _ = _sqlite_degradation_findings(tmp_path, manifest)

    assert [violation["rule"] for violation in violations] == ["sqlite_degradation_sites_grew"]
    assert violations[0]["observed"] == 3
    assert violations[0]["baseline"] == 2


def test_an_unlisted_file_may_carry_none(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "fresh.py", 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"sites": {}}), encoding="utf-8")

    violations, _ = _sqlite_degradation_findings(tmp_path, _manifest(baseline="baseline.json", roots=("pkg",)))

    assert [violation["file"] for violation in violations] == ["pkg/fresh.py"]


def test_a_converted_seam_is_reported_as_ratchet_headroom(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"sites": {"pkg/reader.py": 4}}), encoding="utf-8")

    violations, shrunk = _sqlite_degradation_findings(tmp_path, _manifest(baseline="baseline.json", roots=("pkg",)))

    assert violations == []
    assert shrunk == [{"file": "pkg/reader.py", "observed": 1, "baseline": 4}]


def test_checked_in_baseline_holds_the_current_repository() -> None:
    root = repo_root()
    baseline = load_sqlite_degradation_baseline(root / BASELINE_REF)
    observed = census_sqlite_degradation_sites(root, ("polylogue",))

    assert baseline, "the ratchet baseline must exist for the gate to hold ground"
    grew = {file: count for file, count in observed.items() if count > baseline.get(file, 0)}
    assert grew == {}
