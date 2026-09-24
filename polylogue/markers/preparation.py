"""Freeze marker candidates from the canonical pending writer coordinates."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from polylogue.markers.lowering import candidates_for_block
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
