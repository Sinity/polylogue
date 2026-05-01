"""Durable topology invariants for the realized package layout."""

from __future__ import annotations

import re
from pathlib import Path

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


# Insight modules live under polylogue/insights/.


def test_no_archive_product_modules_at_root() -> None:
    files = root_files()
    legacy = {f for f in files if f.startswith("archive_product") or f.startswith("product_")}
    legacy.update({f for f in files if f in {"archive_resume.py", "authored_payloads.py"}})
    assert not legacy, f"legacy product modules still at root: {sorted(legacy)}"


# Facade/sync surfaces live under polylogue/api/.


def test_no_facade_or_sync_modules_at_root() -> None:
    files = root_files()
    legacy = {f for f in files if f.startswith("facade") or f.startswith("sync")}
    assert not legacy, f"facade/sync modules still at root: {sorted(legacy)}"


def test_polylogue_root_matches_kernel_rule() -> None:
    files = root_files()
    extra = files - KERNEL_ROOT_FILES
    assert not extra, f"non-kernel files at polylogue/ root: {sorted(extra)}"


# Query semantics live under the archive query package.


def test_archive_has_query_subpackage() -> None:
    assert (ROOT / "polylogue" / "archive" / "query" / "__init__.py").exists()


def test_archive_package_imports() -> None:
    import polylogue.archive as archive

    assert archive.__doc__


def test_no_query_runtime_at_lib_root() -> None:
    files = {p.name for p in (ROOT / "polylogue" / "lib").glob("*.py")}
    legacy = {f for f in files if f.startswith("query_")}
    assert not legacy, f"query_* modules still at lib/ root: {sorted(legacy)}"


# Insight storage lives under the storage product package.


def test_storage_has_products_session_subpackage() -> None:
    assert (ROOT / "polylogue" / "storage" / "insights" / "session" / "__init__.py").exists()


def test_no_session_product_at_storage_root() -> None:
    files = {p.name for p in (ROOT / "polylogue" / "storage").glob("*.py")}
    legacy = {f for f in files if f.startswith("session_product_") or f.startswith("store_product_")}
    assert not legacy, f"product-storage modules still at storage/ root: {sorted(legacy)}"


# Structural verification commands exist.


def test_devtools_verify_topology_exists() -> None:
    assert (ROOT / "devtools" / "verify_topology.py").exists()


def test_devtools_verify_cluster_cohesion_exists() -> None:
    assert (ROOT / "devtools" / "verify_cluster_cohesion.py").exists()


def test_topology_projection_yaml_exists() -> None:
    assert (ROOT / "docs" / "plans" / "topology-target.yaml").exists()


# Topology drift dashboard is rendered.


def test_topology_status_doc_rendered() -> None:
    path = ROOT / "docs" / "topology-status.md"
    assert path.exists()
    text = path.read_text()
    assert re.search(r"\b\d+\s+files declared", text), "drift dashboard missing declared-files header"
