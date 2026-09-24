"""Freeze marker candidates from the canonical pending writer coordinates."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from polylogue.markers import parser as marker_parser
from polylogue.markers.lowering import candidates_for_block
from polylogue.markers.models import MarkerCandidate, MarkerKindSpec, MarkerMatch, MarkerProvenance, marker_provenance
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite


def marker_candidates_for_prepared_write(prepared: PreparedSessionWrite) -> list[dict[str, object]]:
    """Retain the divergent input's candidates, including unknown/malformed ones.

    Inherited prefix blocks belong to their parent's accepted input. The row
    carrier already resolved native, fallback, and append occurrence identities.
    """
    columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
    candidates: list[dict[str, object]] = []
    for values in prepared.rows.block_rows:
        row = dict(zip(columns, values, strict=True))
        text = row["text"]
        if not isinstance(text, str):
            continue
        message_id = str(row["message_id"])
        block_id = f"{message_id}:{row['position']}"
        for candidate in candidates_for_block(message_id, block_id, text):
            value = asdict(candidate)
            value["assertion_kind"] = candidate.assertion_kind.value if candidate.assertion_kind else None
            candidates.append(value)
    return candidates


def marker_recipe_fingerprint() -> str:
    """Identify candidate extraction, grammar, registry, and carrier semantics."""
    grammar = {
        name: {
            "pattern": getattr(marker_parser, name).pattern,
            "flags": getattr(marker_parser, name).flags,
        }
        for name in ("_LINE", "_INLINE", "_INLINE_OPEN", "_MALFORMED")
    }
    recipe = {
        "format": 3,
        "sources": [
            inspect.getsource(marker_candidates_for_prepared_write),
            inspect.getsource(candidates_for_block),
            inspect.getsource(marker_parser.parse_markers),
            inspect.getsource(marker_parser._args),
            inspect.getsource(marker_parser.__dict__["marker_spec"]),
            inspect.getsource(MarkerRegistry.get),
            inspect.getsource(MarkerRegistry.__contains__),
            inspect.getsource(marker_provenance),
            inspect.getsource(MarkerCandidate),
            inspect.getsource(MarkerKindSpec),
            inspect.getsource(MarkerMatch),
            inspect.getsource(MarkerProvenance),
        ],
        # These regexes are mutable module-level grammar inputs. Function
        # source alone does not change when a grammar constant is replaced.
        "grammar": grammar,
        # The prepared adapter zips values using this exact insert-column
        # order, so a change can move marker text/provenance to another field.
        "prepared_block_columns": [column.name for column in BLOCKS_SPEC.insert_columns],
        "registry": [
            {
                "kind": spec.kind,
                "payload": spec.payload,
                "lowering_target": spec.lowering_target.value if spec.lowering_target is not None else None,
                "description": spec.description,
                "authority": spec.authority,
            }
            for spec in MARKER_REGISTRY
        ],
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
