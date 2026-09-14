"""The layering gate anchors improvised sqlite degradation policy by content.

Anti-vacuity: replace the anchor comparison with a per-file count comparison
(the pre-#5089 form), or stop censusing ``except sqlite3.*`` handlers, and
``test_two_disjoint_additions_to_one_file_are_both_reported`` fails -- the count
form reports one grown file, never the two distinct sites that grew it, which is
exactly what lets two separately green branches sum to a red master.
``test_a_new_handler_fails_the_gate`` additionally fails if the comparison is
dropped altogether, and ``test_an_unchanged_neighbour_edit_keeps_the_anchor``
fails if the anchor is keyed on anything positional (line number, file digest)
rather than the handler's own text.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from devtools import repo_root
from devtools.sqlite_degradation import census_sqlite_degradation_anchors, load_sqlite_degradation_baseline
from devtools.verify_layering import _sqlite_degradation_findings

BASELINE_REF = "docs/plans/sqlite-degradation-baseline.json"


def _manifest(baseline: str = BASELINE_REF, roots: tuple[str, ...] = ("polylogue",)) -> dict[str, object]:
    return {"sqlite_degradation": {"baseline": baseline, "roots": list(roots)}}


def _module_source(returns: tuple[str, ...], *, tail: str = "") -> str:
    body = "import sqlite3\n\n\ndef read() -> int:\n"
    for value in returns:
        body += f"    try:\n        pass\n    except sqlite3.Error:\n        return {value}\n"
    body += "    return -1\n" + tail
    return body


def _write_module(path: Path, handlers: int) -> None:
    path.write_text(_module_source(tuple(str(index) for index in range(handlers))), encoding="utf-8")


def _write_baseline(path: Path, anchors: Counter[tuple[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "rule": "test",
                "anchors": [
                    {"file": file_name, "digest": digest, **({"count": count} if count != 1 else {})}
                    for (file_name, digest), count in sorted(anchors.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def test_census_ignores_explicit_failure_boundaries(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "reader.py").write_text(
        "import sqlite3\n"
        "\n"
        "def read() -> int:\n"
        "    try:\n"
        "        pass\n"
        "    except sqlite3.Error as exc:\n"
        "        raise RuntimeError('catalog unavailable') from exc\n"
        "\n"
        "    try:\n"
        "        pass\n"
        "    except sqlite3.Error:\n"
        "        return 0\n",
        encoding="utf-8",
    )

    anchors = census_sqlite_degradation_anchors(tmp_path, ("pkg",))

    assert [file_name for file_name, _digest in anchors] == ["pkg/reader.py"]
    assert sum(anchors.values()) == 1


def test_census_counts_only_sqlite_handlers(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 2)
    (package / "other.py").write_text(
        "import sqlite3\n\n\ndef f() -> None:\n    try:\n        pass\n    except OSError:\n        pass\n",
        encoding="utf-8",
    )

    anchors = census_sqlite_degradation_anchors(tmp_path, ("pkg",))

    assert {file_name for file_name, _digest in anchors} == {"pkg/reader.py"}
    assert sum(anchors.values()) == 2
    # Two textually distinct handlers are two anchors, not one count of two.
    assert len(anchors) == 2


def test_a_new_handler_fails_the_gate(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 2)
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, census_sqlite_degradation_anchors(tmp_path, ("pkg",)))
    manifest = _manifest(baseline="baseline.json", roots=("pkg",))

    violations, shrunk = _sqlite_degradation_findings(tmp_path, manifest)
    assert violations == []
    assert shrunk == []

    _write_module(package / "reader.py", 3)
    violations, _ = _sqlite_degradation_findings(tmp_path, manifest)

    assert [violation["rule"] for violation in violations] == ["sqlite_degradation_site_added"]
    assert violations[0]["file"] == "pkg/reader.py"
    assert violations[0]["added"] == 1


def test_two_disjoint_additions_to_one_file_are_both_reported(tmp_path: Path) -> None:
    """The property the count form cannot express.

    Two branches each add a different handler to one file. Each is green against
    the shared baseline on its own -- under a per-file count both would have to
    be, since each file grows by exactly one -- and their merge must not pass by
    having the count already been raised once. Anchors identify *which* handler,
    so the union reports both sites, and neither branch's raise covers the
    other's.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    reader = package / "reader.py"
    reader.write_text(_module_source(("0",)), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, census_sqlite_degradation_anchors(tmp_path, ("pkg",)))
    manifest = _manifest(baseline="baseline.json", roots=("pkg",))

    # Branch A adds one handler; branch B adds a different one. Each alone is a
    # single blocking finding against the shared baseline.
    reader.write_text(_module_source(("0", "11")), encoding="utf-8")
    branch_a, _ = _sqlite_degradation_findings(tmp_path, manifest)
    reader.write_text(_module_source(("0", "22")), encoding="utf-8")
    branch_b, _ = _sqlite_degradation_findings(tmp_path, manifest)
    assert len(branch_a) == 1
    assert len(branch_b) == 1

    # Their merge reports both distinct sites, not one aggregate "file grew".
    reader.write_text(_module_source(("0", "11", "22")), encoding="utf-8")
    merged, _ = _sqlite_degradation_findings(tmp_path, manifest)

    assert [violation["rule"] for violation in merged] == [
        "sqlite_degradation_site_added",
        "sqlite_degradation_site_added",
    ]
    assert branch_a[0]["digest"] != branch_b[0]["digest"]
    assert {str(violation["digest"]) for violation in merged} == {
        str(branch_a[0]["digest"]),
        str(branch_b[0]["digest"]),
    }
    # And a baseline raised for branch A alone still refuses branch B's site.
    _write_baseline(
        baseline,
        census_sqlite_degradation_anchors(tmp_path, ("pkg",))
        - Counter({("pkg/reader.py", str(branch_b[0]["digest"])): 1}),
    )
    remaining, _ = _sqlite_degradation_findings(tmp_path, manifest)
    assert [violation["digest"] for violation in remaining] == [branch_b[0]["digest"]]


def test_an_unchanged_neighbour_edit_keeps_the_anchor(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    reader = package / "reader.py"
    reader.write_text(_module_source(("0",)), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, census_sqlite_degradation_anchors(tmp_path, ("pkg",)))
    manifest = _manifest(baseline="baseline.json", roots=("pkg",))

    reader.write_text(
        "# an unrelated edit above the handler\n"
        + _module_source(("0",), tail="\n\ndef unrelated() -> None:\n    pass\n"),
        encoding="utf-8",
    )
    violations, shrunk = _sqlite_degradation_findings(tmp_path, manifest)

    assert violations == []
    assert shrunk == []


def test_an_unlisted_file_may_carry_none(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "fresh.py", 1)
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, Counter())

    violations, _ = _sqlite_degradation_findings(tmp_path, _manifest(baseline="baseline.json", roots=("pkg",)))

    assert [violation["file"] for violation in violations] == ["pkg/fresh.py"]


def test_a_converted_seam_is_reported_as_ratchet_headroom(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    _write_module(package / "reader.py", 2)
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, census_sqlite_degradation_anchors(tmp_path, ("pkg",)))
    _write_module(package / "reader.py", 1)

    violations, shrunk = _sqlite_degradation_findings(tmp_path, _manifest(baseline="baseline.json", roots=("pkg",)))

    assert violations == []
    assert [item["file"] for item in shrunk] == ["pkg/reader.py"]
    assert shrunk[0]["removed"] == 1


def test_checked_in_baseline_holds_the_current_repository() -> None:
    root = repo_root()
    baseline = load_sqlite_degradation_baseline(root / BASELINE_REF)
    observed = census_sqlite_degradation_anchors(root, ("polylogue",))

    assert baseline, "the ratchet baseline must exist for the gate to hold ground"
    assert observed - baseline == Counter()
