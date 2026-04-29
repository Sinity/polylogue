"""Topology invariants — fail today, pass after the topology refactors land.

Each test corresponds to a refactor issue and encodes its post-state
invariant. Marked ``xfail(strict=True)`` so:

  * the assertion runs every CI run (verifies the refactor is still
    incomplete);
  * when the refactor lands, the xfail flips to UNEXPECTED PASS and the
    test author flips it to a normal ``assert``. Any subsequent regression
    fails CI.

See `#429 <https://github.com/Sinity/polylogue/issues/429>`_ for the
broader pattern. Each xfail's ``reason`` cites the refactor issue.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


KERNEL_ROOT_FILES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "version.py",
        "errors.py",
        "types.py",
        "protocols.py",
        "config.py",
        "logging.py",
        "services.py",
        "assets.py",
        "py.typed",
    }
)


def root_files() -> set[str]:
    return {p.name for p in (ROOT / "polylogue").glob("*.py")}


# -------- #414: product modules live under polylogue/products/ --------


def test_no_archive_product_modules_at_root() -> None:
    files = root_files()
    legacy = {f for f in files if f.startswith("archive_product") or f.startswith("product_")}
    legacy.update({f for f in files if f in {"archive_resume.py", "authored_payloads.py"}})
    assert not legacy, f"legacy product modules still at root: {sorted(legacy)}"


# -------- #426: facade/sync consolidate into polylogue/api/ --------


def test_no_facade_or_sync_modules_at_root() -> None:
    files = root_files()
    legacy = {f for f in files if f.startswith("facade") or f.startswith("sync")}
    assert not legacy, f"facade/sync modules still at root: {sorted(legacy)}"


def test_polylogue_root_matches_kernel_rule() -> None:
    files = root_files()
    extra = files - KERNEL_ROOT_FILES
    assert not extra, f"non-kernel files at polylogue/ root: {sorted(extra)}"


# -------- #424: lib subpackages exist --------


def test_lib_has_query_subpackage() -> None:
    assert (ROOT / "polylogue" / "lib" / "query" / "__init__.py").exists()


def test_no_query_runtime_at_lib_root() -> None:
    files = {p.name for p in (ROOT / "polylogue" / "lib").glob("*.py")}
    legacy = {f for f in files if f.startswith("query_")}
    assert not legacy, f"query_* modules still at lib/ root: {sorted(legacy)}"


# -------- #425: storage subpackages exist --------


def test_storage_has_products_session_subpackage() -> None:
    assert (ROOT / "polylogue" / "storage" / "products" / "session" / "__init__.py").exists()


def test_no_session_product_at_storage_root() -> None:
    files = {p.name for p in (ROOT / "polylogue" / "storage").glob("*.py")}
    legacy = {f for f in files if f.startswith("session_product_") or f.startswith("store_product_")}
    assert not legacy, f"product-storage modules still at storage/ root: {sorted(legacy)}"


# -------- #413: showcase dismantled --------


def test_no_verify_showcase_shim() -> None:
    assert not (ROOT / "devtools" / "verify_showcase.py").exists()


@pytest.mark.xfail(strict=True, reason="topology refactor: legacy CLI baselines, see #413")
def test_no_retired_cli_baselines() -> None:
    legacy = {
        ROOT / "tests" / "baselines" / "showcase" / "help-audit.txt",
        ROOT / "tests" / "baselines" / "showcase" / "help-resume.txt",
    }
    present = {p for p in legacy if p.exists()}
    assert not present, f"baselines for retired commands still present: {sorted(p.name for p in present)}"


# -------- #427: architecture docs reference cross-cuts --------


@pytest.mark.xfail(strict=True, reason="topology refactor: docs lag, see #427")
def test_architecture_md_documents_cross_cuts() -> None:
    text = (ROOT / "docs" / "architecture.md").read_text()
    expected = ("async/sync", "read/write", "runtime/models")
    missing = [e for e in expected if e.lower() not in text.lower()]
    assert not missing, f"docs/architecture.md missing cross-cut documentation for: {missing}"


# -------- #429 Phase 1: lints land as devtools commands (passing) --------


def test_devtools_verify_topology_exists() -> None:
    assert (ROOT / "devtools" / "verify_topology.py").exists()


def test_devtools_verify_cluster_cohesion_exists() -> None:
    assert (ROOT / "devtools" / "verify_cluster_cohesion.py").exists()


def test_topology_projection_yaml_exists() -> None:
    assert (ROOT / "docs" / "plans" / "topology-target.yaml").exists()


# -------- #429 Phase 3: drift dashboard rendered (passing) --------


def test_topology_status_doc_rendered() -> None:
    path = ROOT / "docs" / "topology-status.md"
    assert path.exists()
    text = path.read_text()
    assert re.search(r"\b\d+\s+files declared", text), "drift dashboard missing declared-files header"
