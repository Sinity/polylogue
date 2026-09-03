"""Blinded judgment view: mask provenance until verdict (rxdo.9.6, mechanism F).

Judge surfaces (the ``judge``/``compare`` CLI, MCP judgment tools) mask
detector/model/agent identity and actor names until the verdict is recorded;
reveal after. This is a renderer/view-profile concern only -- the underlying
records keep provenance in separate fields already, so masking costs nothing
in storage. Blinding matters most when judging competing agents' findings.

No new store: this module projects existing candidate records into a masked
view and receipts the projection (order + masked-field set + rubric), then
authorizes reveal only once a verdict has actually been recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from polylogue.core.hashing import hash_payload

#: Fields stripped from the visible projection before a verdict is recorded.
#: Extend via ``extra_masked_fields``, never by removing an entry here --
#: removing an entry is exactly the leak :func:`assert_no_leak` exists to catch.
DEFAULT_MASKED_PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "actor_ref",
        "author_ref",
        "author_kind",
        "model",
        "provider",
        "detector_ref",
        "arm",
        "prior_score",
        "prior_rank",
        "judge_ref",
        "execution_context_id",
    }
)


@dataclass(frozen=True, slots=True)
class BlindedItem:
    """One item projected for judgment, provenance stripped, evidence intact."""

    display_position: int
    evidence_refs: tuple[str, ...]
    visible_fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BlindingReceipt:
    """Proof of a blinded projection: order, masked fields, rubric, reveal state."""

    item_order_hash: str
    masked_fields: tuple[str, ...]
    rubric_ref: str
    sealed_at_ms: int
    revealed_at_ms: int | None = None


def blind_items(
    items: Sequence[Mapping[str, object]],
    *,
    order: Sequence[int],
    rubric_ref: str,
    sealed_at_ms: int,
    extra_masked_fields: Sequence[str] = (),
    masked_fields: frozenset[str] = DEFAULT_MASKED_PROVENANCE_FIELDS,
) -> tuple[tuple[BlindedItem, ...], BlindingReceipt]:
    """Project raw candidate records into a masked, order-bound view.

    ``order`` is the caller-chosen (e.g. randomized) permutation of indices
    into ``items``; it is receipted (as a hash, not the raw order -- the
    receipt must not itself leak identity via position-to-source mapping to
    an unauthorized reader) so the elicitation is reconstructible by whoever
    holds the original ``order`` value.
    """

    if sorted(order) != list(range(len(items))):
        raise ValueError("order must be a permutation of item indices")
    effective_masked = masked_fields | set(extra_masked_fields)
    blinded: list[BlindedItem] = []
    for position, source_index in enumerate(order):
        record = items[source_index]
        visible = {key: value for key, value in record.items() if key not in effective_masked}
        raw_evidence = record.get("evidence_refs", ())
        evidence_refs = tuple(str(ref) for ref in raw_evidence) if isinstance(raw_evidence, Sequence) else ()
        blinded.append(BlindedItem(display_position=position, evidence_refs=evidence_refs, visible_fields=visible))
    receipt = BlindingReceipt(
        item_order_hash=hash_payload(list(order)),
        masked_fields=tuple(sorted(effective_masked)),
        rubric_ref=rubric_ref,
        sealed_at_ms=sealed_at_ms,
    )
    return tuple(blinded), receipt


def reveal(receipt: BlindingReceipt, *, revealed_at_ms: int, verdict_recorded: bool) -> BlindingReceipt:
    """Authorize reveal -- only once a verdict has actually been recorded."""

    if not verdict_recorded:
        raise ValueError("reveal is only authorized after a verdict is recorded")
    if receipt.revealed_at_ms is not None:
        return receipt
    return replace(receipt, revealed_at_ms=revealed_at_ms)


def assert_no_leak(
    blinded_items: Sequence[BlindedItem],
    *,
    masked_fields: frozenset[str] = DEFAULT_MASKED_PROVENANCE_FIELDS,
) -> None:
    """Raise if any masked field survived the projection (defense in depth)."""

    for item in blinded_items:
        leaked = masked_fields & item.visible_fields.keys()
        if leaked:
            raise ValueError(f"blinding leak: masked fields present in visible projection: {sorted(leaked)}")


__all__ = [
    "DEFAULT_MASKED_PROVENANCE_FIELDS",
    "BlindedItem",
    "BlindingReceipt",
    "assert_no_leak",
    "blind_items",
    "reveal",
]
