"""Generate the topology-target projection from the current tree.

Walks ``polylogue/**/*.py``, applies explicit placement rules, and emits YAML
covering every file.

Output is a first-cut projection. Cells where the rule is uncertain are
marked ``target: TBD`` with a reason. The intended workflow is:

    1. Run this script to produce the initial YAML.
    2. Review TBD rows and fill them in.
    3. Use the YAML as input to ``devtools verify-topology``.

"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "docs" / "plans" / "topology-target.yaml"


# ---------------------------------------------------------------------------
# Placement rules
# ---------------------------------------------------------------------------

# polylogue/ root kernel rule
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

# polylogue/ root product-domain modules.
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

# polylogue/ root facade/sync surfaces.
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

# polylogue/ root cross-ring concepts.
CROSS_RING_ROOT_MOVES = {
    "artifacts.py": "polylogue/artifacts/__init__.py",
    "artifact_graph.py": "polylogue/artifacts/graph.py",
    "readiness.py": "polylogue/readiness/__init__.py",
    "surface_payloads.py": "polylogue/surfaces/payloads.py",
    "maintenance_models.py": "polylogue/maintenance/models.py",
    "maintenance_targets.py": "polylogue/maintenance/targets.py",
    "publication.py": "polylogue/publication/__init__.py",
}

# Archive-domain subpackage rules — prefix → subpackage
LIB_PREFIX_TO_SUBPACKAGE = {
    "query_": "archive/query/",
    "session_profile": "lib/session/",
    "session_payload": "lib/session/",
    "session_summaries": "lib/session/",
    "viewport_": "lib/viewport/",
    "viewports": "lib/viewport/",
    "raw_payload_": "lib/raw_payload/",
    "raw_payload": "lib/raw_payload/",
    "artifact_taxonomy_": "lib/artifact_taxonomy/",
    "artifact_taxonomy": "lib/artifact_taxonomy/",
    "action_event_": "archive/action_event/",
    "action_events": "archive/action_event/",
    "message_": "lib/message/",
    "messages": "lib/message/",
    "conversation_": "lib/conversation/",
    "semantic_fact_": "lib/semantic/",
    "semantic_facts": "lib/semantic/facts.py",
    "content_projection": "lib/semantic/",
    "branch_type": "lib/conversation/",
    "threads": "lib/conversation/",
    "neighbor_candidates": "lib/conversation/",
    "work_event": "lib/conversation/",
    "attribution": "lib/conversation/",
    "filter_": "archive/filter/",
    "filters": "archive/filter/",
    "phase_": "lib/phase/",
    "projection_": "lib/projection/",
    "projections": "lib/projection/",
    "provider_": "lib/provider/",
    "attachment_": "archive/attachment/",
}

# Lib root primitives stay at lib/ root.
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
        "pricing.py",
        "payload_coercion.py",
    }
)

# polylogue/storage/ subpackage rules.
STORAGE_PREFIX_TO_SUBPACKAGE = {
    "repository_archive_": "storage/repository/archive/",
    "repository_product_": "storage/repository/product/",
    "repository_action_": "storage/repository/action/",
    "repository_raw": "storage/repository/raw/",
    "repository_vectors": "storage/repository/vectors/",
    "repository_write_": "storage/repository/archive/writes/",
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

# polylogue/showcase/ is now declared as verification-lab substrate. The
# public product CLI no longer exposes audit/qa/showcase vocabulary; lab
# scenarios remain here until a future package split is useful.

# Placement owner per target prefix.
TARGET_TO_OWNER = [
    ("polylogue/products/", "product-domain"),
    ("polylogue/api/", "api-surface"),
    ("polylogue/artifacts/", "artifact-domain"),
    ("polylogue/readiness/", "readiness-domain"),
    ("polylogue/surfaces/", "surface-shared"),
    ("polylogue/maintenance/", "maintenance-domain"),
    ("polylogue/publication/", "publication-domain"),
    ("polylogue/archive/query/", "archive-query"),
    ("polylogue/archive/filter/", "archive-filter"),
    ("polylogue/lib/session/", "lib-session"),
    ("polylogue/lib/viewport/", "lib-viewport"),
    ("polylogue/lib/raw_payload/", "lib-raw-payload"),
    ("polylogue/lib/artifact_taxonomy/", "lib-artifact-taxonomy"),
    ("polylogue/archive/action_event/", "archive-action-event"),
    ("polylogue/lib/message/", "lib-message"),
    ("polylogue/lib/conversation/", "lib-conversation"),
    ("polylogue/lib/semantic/", "lib-semantic"),
    ("polylogue/lib/phase/", "lib-phase"),
    ("polylogue/lib/projection/", "lib-projection"),
    ("polylogue/lib/provider/", "lib-provider"),
    ("polylogue/archive/attachment/", "archive-attachment"),
    ("polylogue/storage/repository/", "storage-repository"),
    ("polylogue/storage/products/", "storage-products"),
    ("polylogue/storage/runtime/", "storage-runtime"),
    ("polylogue/storage/action_events/", "storage-action-events"),
    ("polylogue/storage/embeddings/", "storage-embeddings"),
    ("polylogue/storage/search/", "storage-search"),
    ("polylogue/storage/artifacts/", "storage-artifacts"),
    ("polylogue/storage/fts/", "storage-fts"),
    ("polylogue/storage/raw/", "storage-raw"),
    ("polylogue/storage/derived/", "storage-derived"),
    ("polylogue/sources/drive/", "source-drive"),
    ("polylogue/sources/parsers/claude/", "source-claude-parser"),
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
    # Guard against creating a target that would double the __init__.py
    # suffix when the stem itself is __init__.py.
    if stem == "__init__.py" and sub.endswith("/"):
        return f"polylogue/{sub}{name}"
    return f"polylogue/{sub}{stem}"


def _apply_rule(name: str, prefix: str, sub: str) -> str:
    # Full-target rule: value ends in .py and prefix matches name (with or without .py).
    if sub.endswith(".py") and (prefix == name or prefix == name.removesuffix(".py")):
        return f"polylogue/{sub}"
    return _resolve_target(name, sub, prefix)


def lib_target(name: str) -> str:
    """Return target path for a polylogue/lib/<name> file, or 'TBD'."""
    if name in LIB_ROOT_PRIMITIVES:
        return f"polylogue/lib/{name}"
    # match longest prefix first
    for prefix in sorted(LIB_PREFIX_TO_SUBPACKAGE, key=len, reverse=True):
        if name.startswith(prefix):
            return _apply_rule(name, prefix, LIB_PREFIX_TO_SUBPACKAGE[prefix])
    return "TBD"


def storage_target(name: str) -> str:
    if name in STORAGE_ROOT_KEEP:
        return f"polylogue/storage/{name}"
    for prefix in sorted(STORAGE_PREFIX_TO_SUBPACKAGE, key=len, reverse=True):
        if name.startswith(prefix):
            return _apply_rule(name, prefix, STORAGE_PREFIX_TO_SUBPACKAGE[prefix])
    return "TBD"


def placement_owner(target: str) -> str:
    if target == "TBD" or target.startswith("polylogue/lib/") and "/" not in target[len("polylogue/lib/") :]:
        return ""
    for prefix, owner in sorted(TARGET_TO_OWNER, key=lambda x: -len(x[0])):
        if target.startswith(prefix):
            return owner
    return ""


def cross_cut_tags(name: str, rel: str = "") -> dict[str, Any]:
    tags: dict[str, Any] = {}
    if "_runtime" in name or name.endswith("_runtime.py"):
        tags["lifecycle"] = "runtime"
    if "_models" in name or name.endswith("_models.py"):
        tags["lifecycle"] = tags.get("lifecycle", "model")
    if "_reads" in name or "read_support" in name:
        tags["layer"] = "read"
    if "_writes" in name or "_write_" in name:
        tags["layer"] = "write"
    # Path-based api tagging after api-surface consolidation.
    if rel.startswith("polylogue/api/sync/"):
        tags["api"] = "sync"
    elif rel.startswith("polylogue/api/"):
        tags["api"] = "async"
    elif name.startswith("sync_") or name == "sync.py" or name == "sync_bridge.py":
        tags["api"] = "sync"
    elif name.startswith("facade"):
        tags["api"] = "async"
    return tags


def classify(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    name = path.name
    target = ""
    owner = ""
    reason = ""

    if rel.startswith("polylogue/") and "/" not in rel[len("polylogue/") :]:
        # Root-level polylogue file
        if name in KERNEL_ROOT_FILES:
            target = rel
            owner = "kernel"
            reason = "kernel root rule"
        elif name in PRODUCT_ROOT_MOVES:
            target = PRODUCT_ROOT_MOVES[name]
            owner = "product-domain"
        elif name in FACADE_ROOT_MOVES:
            target = FACADE_ROOT_MOVES[name]
            owner = "api-surface"
        elif name in CROSS_RING_ROOT_MOVES:
            target = CROSS_RING_ROOT_MOVES[name]
            owner = placement_owner(target) or "cross-ring-domain"
        else:
            target = "TBD"
            reason = "root file, no rule yet"
    elif rel.startswith("polylogue/lib/"):
        suffix = rel[len("polylogue/lib/") :]
        if "/" in suffix:
            target = rel  # already in a subdir, leave for now
            owner = "stable"
        else:
            target = lib_target(suffix)
            if target.startswith("polylogue/lib/") and "/" not in target[len("polylogue/lib/") :]:
                owner = "lib-root"
                reason = "lib-root primitive"
            else:
                owner = placement_owner(target) or "lib-domain"
    elif rel.startswith("polylogue/archive/query/"):
        target = rel
        owner = "archive-query"
        reason = "archive-domain query semantics"
    elif rel.startswith("polylogue/archive/filter/"):
        target = rel
        owner = "archive-filter"
        reason = "archive-domain filter semantics"
    elif rel.startswith("polylogue/archive/"):
        target = rel
        owner = "stable"
    elif rel.startswith("polylogue/storage/"):
        suffix = rel[len("polylogue/storage/") :]
        if "/" in suffix:
            target = rel  # already in backends/ or search_providers/
            owner = "stable"
        else:
            target = storage_target(suffix)
            if target.startswith("polylogue/storage/") and "/" not in target[len("polylogue/storage/") :]:
                owner = "storage-root"
                reason = "storage-root cross-cutting helper"
            else:
                owner = placement_owner(target) or "storage-domain"
    elif rel.startswith("polylogue/showcase/"):
        target = rel
        owner = "stable"
        reason = "verification-lab showcase substrate retained"
    elif rel.startswith("polylogue/sources/"):
        # Drive-specific files cluster
        suffix = rel[len("polylogue/sources/") :]
        if suffix.startswith("drive") and "/" not in suffix:
            target = f"polylogue/sources/drive/{suffix.replace('drive_', '').replace('drive.py', '__init__.py') or '__init__.py'}"
            owner = "source-drive"
        elif suffix.startswith("parsers/claude_") and "/" not in suffix[len("parsers/") :]:
            # parsers/claude_*
            stem = suffix[len("parsers/claude_") :] or "__init__.py"
            target = f"polylogue/sources/parsers/claude/{stem}"
            owner = "source-claude-parser"
        elif suffix == "parsers/claude.py":
            target = "polylogue/sources/parsers/claude/__init__.py"
            owner = "source-claude-parser"
        else:
            target = rel
            owner = "stable"
    else:
        target = rel
        owner = "stable"

    return {
        "path": rel,
        "loc": loc(path),
        "target": target,
        "owner": owner,
        "reason": reason,
        "cross_cut": cross_cut_tags(name, rel),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def yaml_escape(value: str) -> str:
    if any(c in value for c in ":#'\""):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def emit_yaml(rows: list[dict[str, Any]], out: Path) -> None:
    lines: list[str] = ["# Topology projection — generated by build_projection.py", "", "files:"]
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
