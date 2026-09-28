"""Freeze marker candidates from the canonical pending writer coordinates."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING

from polylogue.markers import parser as marker_parser
from polylogue.markers.lowering import assertion_id_for_marker, candidates_for_block
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
    seen_assertion_ids: set[str] = set()
    for values in prepared.rows.block_rows:
        row = dict(zip(columns, values, strict=True))
        text = row["text"]
        if not isinstance(text, str):
            continue
        message_id = str(row["message_id"])
        block_id = f"{message_id}:{row['position']}"
        for candidate in candidates_for_block(message_id, block_id, text):
            assertion_id = assertion_id_for_marker(candidate)
            if assertion_id is not None:
                if assertion_id in seen_assertion_ids:
                    continue
                seen_assertion_ids.add(assertion_id)
            value = asdict(candidate)
            value["assertion_kind"] = candidate.assertion_kind.value if candidate.assertion_kind else None
            candidates.append(value)
    return candidates


def retired_marker_assertion_ids(blocks: Iterable[tuple[str, int, str]]) -> list[str]:
    """Name the child-owned assertions a late parent's re-extraction supersedes.

    A child ingested before its parent seals candidates for its whole
    transcript, including the replayed prefix, under the child's message ids.
    Once the parent arrives those prefix blocks belong to the parent's
    accepted input, whose own candidates carry the parent's evidence. The
    ids are recomputed from the removed ``(message_id, position, text)`` rows
    with the same extraction the child's carrier used, so they name exactly
    the prefix candidates that carrier delivered.
    """
    retired: set[str] = set()
    for message_id, position, text in blocks:
        for candidate in candidates_for_block(message_id, f"{message_id}:{position}", text):
            assertion_id = assertion_id_for_marker(candidate)
            if assertion_id is not None:
                retired.add(assertion_id)
    return sorted(retired)


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
        "format": 4,
        "sources": [
            inspect.getsource(marker_candidates_for_prepared_write),
            inspect.getsource(retired_marker_assertion_ids),
            inspect.getsource(assertion_id_for_marker),
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
