"""Generate the topology-target projection from the current tree.

Walks ``polylogue/**/*.py``, applies placement rules derived from issues
#403, #414, #419, #420, #424, #425, #426, and emits YAML covering every
file.

Output is a first-cut projection. Cells where the rule is uncertain are
marked ``target: TBD`` with a reason. The intended workflow is:

    1. Run this script to produce the initial YAML.
    2. Review TBD rows and fill them in.
    3. Use the YAML as input to ``devtools verify-topology``.

See `#429 <https://github.com/Sinity/polylogue/issues/429>`_.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "docs" / "plans" / "topology-target.yaml"


# ---------------------------------------------------------------------------
# Placement rules — derived from the seven topology issues
# ---------------------------------------------------------------------------

# polylogue/ root — kernel rule per #426
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

# polylogue/ root product-domain modules — moved by #414 to polylogue/products/
PRODUCT_ROOT_MOVES = {
    "archive_products.py": "polylogue/products/archive.py",
    "archive_product_models.py": "polylogue/products/archive_models.py",
    "archive_product_summaries.py": "polylogue/products/archive_summaries.py",
    "archive_product_rollups.py": "polylogue/products/archive_rollups.py",
    "archive_resume.py": "polylogue/products/resume.py",
    "product_export_bundles.py": "polylogue/products/export_bundles.py",
    "product_readiness.py": "polylogue/products/readiness.py",
    "authored_payloads.py": "polylogue/products/authored_payloads.py",
}

# polylogue/ root facade/sync — moved by #426 to polylogue/api/
FACADE_ROOT_MOVES = {
    "facade.py": "polylogue/api/__init__.py",
    "facade_archive.py": "polylogue/api/archive.py",
    "facade_ingest.py": "polylogue/api/ingest.py",
    "facade_products.py": "polylogue/api/products.py",
    "sync.py": "polylogue/api/sync/__init__.py",
    "sync_bridge.py": "polylogue/api/sync/bridge.py",
    "sync_conversation_queries.py": "polylogue/api/sync/conversations.py",
    "sync_product_queries.py": "polylogue/api/sync/products.py",
}

# polylogue/ root cross-ring concepts — homes per #426
CROSS_RING_ROOT_MOVES = {
    "artifacts.py": "polylogue/artifacts/__init__.py",  # provisional; #426/#425 decide
    "artifact_graph.py": "polylogue/artifacts/graph.py",
    "readiness.py": "polylogue/readiness/__init__.py",
    "surface_payloads.py": "polylogue/surfaces/payloads.py",
    "maintenance_models.py": "polylogue/maintenance/models.py",
    "maintenance_targets.py": "polylogue/maintenance/targets.py",
    "publication.py": "polylogue/publication/__init__.py",
}

# polylogue/lib/ subpackage rules per #424 — prefix → subpackage
LIB_PREFIX_TO_SUBPACKAGE = {
    "query_": "lib/query/",
    "session_profile": "lib/session/",
    "session_payload": "lib/session/",
    "session_summaries": "lib/session/",
    "viewport_": "lib/viewport/",
    "viewports": "lib/viewport/",
    "raw_payload_": "lib/raw_payload/",
    "raw_payload": "lib/raw_payload/",
    "artifact_taxonomy_": "lib/artifact_taxonomy/",
    "artifact_taxonomy": "lib/artifact_taxonomy/",
    "action_event_": "lib/action_event/",
    "action_events": "lib/action_event/",
    "message_": "lib/message/",
    "messages": "lib/message/",
    "conversation_": "lib/conversation/",
    "semantic_fact": "lib/semantic/",
    "content_projection": "lib/semantic/",
    "attachment_": "lib/conversation/",
    "branch_type": "lib/conversation/",
    "threads": "lib/conversation/",
    "neighbor_candidates": "lib/conversation/",
    "work_event": "lib/conversation/",
    "attribution": "lib/conversation/",
    "filter_": "lib/filter/",
    "filters": "lib/filter/",
}

# Lib root primitives — stay at lib/ root per #424
LIB_ROOT_PRIMITIVES = frozenset(
    {
        "__init__.py",
        "json.py",
        "hashing.py",
        "dates.py",
        "timestamps.py",
        "security.py",
        "stats.py",
        "metrics.py",
        "outcomes.py",
        "coverage.py",
        "repo_identity.py",
        "tail_overlay.py",
        "models.py",
        "roles.py",
        "search_hits.py",
        "run_activity.py",
        "provider_identity.py",
    }
)

# polylogue/storage/ subpackage rules per #425
STORAGE_PREFIX_TO_SUBPACKAGE = {
    "repository_archive_": "storage/repository/archive/",
    "repository_product_": "storage/repository/product/",
    "repository_action_": "storage/repository/archive/",
    "repository_raw": "storage/repository/raw/",
    "repository_vectors": "storage/repository/vectors/",
    "repository_write_": "storage/repository/archive/",
    "repository_writes": "storage/repository/archive/",
    "repository_contracts": "storage/repository/",
    "repository.py": "storage/repository/__init__.py",
    "session_product_": "storage/products/session/",
    "store_runtime_action_": "storage/runtime/action/",
    "store_runtime_archive_": "storage/runtime/archive/",
    "store_runtime_raw_": "storage/runtime/raw/",
    "store_product_aggregate_": "storage/products/aggregate/",
    "store_product_session_": "storage/products/session/",
    "store_product_timeline_": "storage/products/timeline/",
    "store_constants": "storage/runtime/",
    "store.py": "storage/runtime/__init__.py",
    "action_event_": "storage/action_events/",
    "embedding_stats_": "storage/embeddings/",
    "embedding_stats": "storage/embeddings/",
    "search_": "storage/search/",
    "search.py": "storage/search/__init__.py",
    "artifact_": "storage/artifacts/",
    "fts_lifecycle": "storage/fts/",
    "raw_ingest_": "storage/raw/",
    "raw_state_": "storage/raw/",
    "derived_status": "storage/derived/",
    "product_read_support": "storage/products/",
    "store_product": "storage/products/",  # catch
    "store_runtime": "storage/runtime/",  # catch
}

STORAGE_ROOT_KEEP = frozenset(
    {
        "__init__.py",
        "blob_store.py",
        "repair.py",
        "index.py",
        "hydrators.py",
        "cursor_state.py",
        "run_state.py",
        "conversation_replacement.py",
        "archive_views.py",
        "query_models.py",
    }
)

# Owning issue per target prefix
TARGET_TO_ISSUE = [
    ("polylogue/products/", "#414"),
    ("polylogue/api/", "#426"),
    ("polylogue/artifacts/", "#426"),
    ("polylogue/readiness/", "#426"),
    ("polylogue/surfaces/", "#426"),
    ("polylogue/maintenance/", "#426"),
    ("polylogue/publication/", "#426"),
    ("polylogue/lib/query/", "#424"),
    ("polylogue/lib/session/", "#424"),
    ("polylogue/lib/viewport/", "#424"),
    ("polylogue/lib/raw_payload/", "#424"),
    ("polylogue/lib/artifact_taxonomy/", "#424"),
    ("polylogue/lib/action_event/", "#424"),
    ("polylogue/lib/message/", "#424"),
    ("polylogue/lib/conversation/", "#424"),
    ("polylogue/lib/semantic/", "#424"),
    ("polylogue/lib/filter/", "#424"),
    ("polylogue/storage/repository/", "#425"),
    ("polylogue/storage/products/", "#425"),
    ("polylogue/storage/runtime/", "#425"),
    ("polylogue/storage/action_events/", "#425"),
    ("polylogue/storage/embeddings/", "#425"),
    ("polylogue/storage/search/", "#425"),
    ("polylogue/storage/artifacts/", "#425"),
    ("polylogue/storage/fts/", "#425"),
    ("polylogue/storage/raw/", "#425"),
    ("polylogue/storage/derived/", "#425"),
    ("polylogue/sources/drive/", "#403"),
    ("polylogue/sources/parsers/claude/", "#403"),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.read_text().splitlines())
    except Exception:
        return 0


def _resolve_target(name: str, sub: str, prefix: str) -> str:
    """Build a unique target path for a module that matches a given subpackage prefix.

    When the module name equals the prefix exactly, keep the full name as the
    target stem (so e.g. ``session_payload`` lands at
    ``lib/session/session_payload.py``, not at the cluster's ``__init__.py``
    where it would collide with sibling modules).

    When the prefix has a trailing underscore (``query_``, ``store_runtime_``),
    strip it from the stem so ``query_runtime`` becomes ``runtime.py``.
    """
    stem = name[len(prefix) :]
    if not stem:
        # Module name == prefix; preserve the full name to avoid colliding
        # at the subpackage __init__.py.
        return f"polylogue/{sub}{name}"
    if not stem.endswith(".py"):
        stem = stem + ".py"
    stem = stem.lstrip("_")
    if not stem or stem == ".py":
        return f"polylogue/{sub}{name}"
    return f"polylogue/{sub}{stem}"


def lib_target(name: str) -> str:
    """Return target path for a polylogue/lib/<name> file, or 'TBD'."""
    if name in LIB_ROOT_PRIMITIVES:
        return f"polylogue/lib/{name}"
    # match longest prefix first
    for prefix in sorted(LIB_PREFIX_TO_SUBPACKAGE, key=len, reverse=True):
        if name.startswith(prefix):
            return _resolve_target(name, LIB_PREFIX_TO_SUBPACKAGE[prefix], prefix)
    return "TBD"


def storage_target(name: str) -> str:
    if name in STORAGE_ROOT_KEEP:
        return f"polylogue/storage/{name}"
    for prefix in sorted(STORAGE_PREFIX_TO_SUBPACKAGE, key=len, reverse=True):
        if name.startswith(prefix):
            return _resolve_target(name, STORAGE_PREFIX_TO_SUBPACKAGE[prefix], prefix)
    return "TBD"


def owning_issue(target: str) -> str:
    if target == "TBD" or target.startswith("polylogue/lib/") and "/" not in target[len("polylogue/lib/") :]:
        return ""
    for prefix, issue in sorted(TARGET_TO_ISSUE, key=lambda x: -len(x[0])):
        if target.startswith(prefix):
            return issue
    return ""


def cross_cut_tags(name: str) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    if "_runtime" in name or name.endswith("_runtime.py"):
        tags["lifecycle"] = "runtime"
    if "_models" in name or name.endswith("_models.py"):
        tags["lifecycle"] = tags.get("lifecycle", "model")
    if "_reads" in name or "read_support" in name:
        tags["layer"] = "read"
    if "_writes" in name or "_write_" in name:
        tags["layer"] = "write"
    if name.startswith("sync_") or name == "sync.py" or name == "sync_bridge.py":
        tags["api"] = "sync"
    if name.startswith("facade"):
        tags["api"] = "async"
    return tags


def classify(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    name = path.name
    target = ""
    issue = ""
    reason = ""

    if rel.startswith("polylogue/") and "/" not in rel[len("polylogue/") :]:
        # Root-level polylogue file
        if name in KERNEL_ROOT_FILES:
            target = rel
            issue = "kernel"
            reason = "kernel rule per #426"
        elif name in PRODUCT_ROOT_MOVES:
            target = PRODUCT_ROOT_MOVES[name]
            issue = "#414"
        elif name in FACADE_ROOT_MOVES:
            target = FACADE_ROOT_MOVES[name]
            issue = "#426"
        elif name in CROSS_RING_ROOT_MOVES:
            target = CROSS_RING_ROOT_MOVES[name]
            issue = "#426"
        else:
            target = "TBD"
            reason = "root file, no rule yet"
    elif rel.startswith("polylogue/lib/"):
        suffix = rel[len("polylogue/lib/") :]
        if "/" in suffix:
            target = rel  # already in a subdir, leave for now
            issue = "stable"
        else:
            target = lib_target(suffix)
            if target.startswith("polylogue/lib/") and "/" not in target[len("polylogue/lib/") :]:
                issue = "lib-root"
                reason = "lib-root primitive per #424"
            else:
                issue = owning_issue(target) or "#424"
    elif rel.startswith("polylogue/storage/"):
        suffix = rel[len("polylogue/storage/") :]
        if "/" in suffix:
            target = rel  # already in backends/ or search_providers/
            issue = "stable"
        else:
            target = storage_target(suffix)
            if target.startswith("polylogue/storage/") and "/" not in target[len("polylogue/storage/") :]:
                issue = "storage-root"
                reason = "storage-root cross-cutting helper per #425"
            else:
                issue = owning_issue(target) or "#425"
    elif rel.startswith("polylogue/showcase/"):
        target = "TBD"
        issue = "#413"
        reason = "showcase dismantling — substrate vs lab vs proof split"
    elif rel.startswith("polylogue/sources/"):
        # Drive-specific files cluster
        suffix = rel[len("polylogue/sources/") :]
        if suffix.startswith("drive") and "/" not in suffix:
            target = f"polylogue/sources/drive/{suffix.replace('drive_', '').replace('drive.py', '__init__.py') or '__init__.py'}"
            issue = "#403"
        elif suffix.startswith("parsers/claude") and "/" not in suffix[len("parsers/") :]:
            # parsers/claude_*
            stem = suffix[len("parsers/claude_") :] or "__init__.py"
            target = f"polylogue/sources/parsers/claude/{stem}"
            issue = "#403"
        else:
            target = rel
            issue = "stable"
    else:
        target = rel
        issue = "stable"

    return {
        "path": rel,
        "loc": loc(path),
        "target": target,
        "owner": issue,
        "reason": reason,
        "cross_cut": cross_cut_tags(name),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def yaml_escape(value: str) -> str:
    if any(c in value for c in ":#'\""):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def emit_yaml(rows: list[dict[str, Any]], out: Path) -> None:
    lines: list[str] = ["# Topology projection — generated by build_projection.py", "# See #429.", "", "files:"]
    for row in rows:
        lines.append(f"  - path: {row['path']}")
        lines.append(f"    loc: {row['loc']}")
        lines.append(f"    target: {yaml_escape(row['target'])}")
        if row["owner"]:
            lines.append(f"    owner: {row['owner']}")
        if row["reason"]:
            lines.append(f"    reason: {yaml_escape(row['reason'])}")
        if row["cross_cut"]:
            tags = ", ".join(f"{k}: {v}" for k, v in sorted(row["cross_cut"].items()))
            lines.append(f"    cross_cut: {{ {tags} }}")
    out.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    files = sorted(ROOT.glob("polylogue/**/*.py"))
    rows = [classify(f) for f in files if "__pycache__" not in f.parts]
    out = PROJECTION
    out.parent.mkdir(parents=True, exist_ok=True)
    emit_yaml(rows, out)
    # Stats
    by_owner: dict[str, int] = {}
    tbd = 0
    for row in rows:
        by_owner[row["owner"] or "(none)"] = by_owner.get(row["owner"] or "(none)", 0) + 1
        if row["target"] == "TBD":
            tbd += 1
    print(f"Wrote {out} with {len(rows)} rows. TBD: {tbd}.")
    print("By owner:")
    for owner, count in sorted(by_owner.items(), key=lambda x: -x[1]):
        print(f"  {owner:>20s}  {count:>4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
