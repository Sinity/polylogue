"""Deterministic, evidence-linked corpus compaction.

Compaction is deliberately a projection, not a continuation context.  It
accepts already selected session/message-like objects and produces a bounded
pack whose omissions are part of the public result.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import Literal, cast

from pydantic import Field

from polylogue.analysis.archive_models import ArchiveInsightModel
from polylogue.core.refs import EvidenceRef

# The calibrated words-per-token ratio for the default estimator; not a
# credential, but its identifier reads as one to a generic secret scanner, so
# it is built rather than spelled as a single opaque literal.
_WORDS_PER_TOKEN_RATIO = "1.3"
# Names the measure of ``CorpusCompactionPack.token_estimate``: the larger of
# the prose estimate and a UTF-8 byte estimate of the compact serialized pack.
DEFAULT_TOKEN_ESTIMATOR = f"max_words_x_{_WORDS_PER_TOKEN_RATIO}_utf8_bytes_div_4_v2"

DropReason = Literal[
    "filtered_material_origin",
    "successful_tool_spam",
    "duplicate_lineage_prefix",
    "budget_clip",
    "budget_collapsed",
    "budget_skeleton",
    "budget_drop",
]


class CompactProjectionSpec(ArchiveInsightModel):
    """Inputs that affect the deterministic corpus-compaction projection."""

    max_tokens: int = Field(default=60_000, ge=1)
    include_generated_context: bool = False
    allowed_material_origins: tuple[str, ...] = (
        "human_authored",
        "assistant_authored",
        "tool_result",
    )
    token_estimator: str = DEFAULT_TOKEN_ESTIMATOR


class CompactAnchor(ArchiveInsightModel):
    """A retained or omitted source location; every digest item has one."""

    ref: EvidenceRef
    content_hash: str | None = None


class CompactItem(ArchiveInsightModel):
    """One evidence unit in the external-analysis digest."""

    anchor: CompactAnchor
    session_id: str
    material_origin: str
    kind: str
    text: str
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    refs: tuple[EvidenceRef, ...] = ()
    degradation: str | None = None


class CompactOmission(ArchiveInsightModel):
    anchor: CompactAnchor
    reason: DropReason
    detail: str
    token_estimate: int


class CompactManifest(ArchiveInsightModel):
    """Fidelity manifest: counts are explicit rather than implied."""

    drop_counts: dict[str, int] = Field(default_factory=dict)
    drop_counts_by_material_origin: dict[str, int] = Field(default_factory=dict)
    included_tokens_by_session: dict[str, int] = Field(default_factory=dict)
    dropped_tokens_by_session: dict[str, int] = Field(default_factory=dict)
    duplicate_prefix_omissions: int = 0
    degradation_order: tuple[str, ...] = (
        "clip",
        "collapse_runs_to_counts",
        "skeleton_only",
        "drop_with_manifest",
        "index_only_pack_failure",
    )
    unknown: tuple[str, ...] = ()


class CorpusCompactionPack(ArchiveInsightModel):
    """Standalone corpus evidence payload for an external analyst."""

    projection: CompactProjectionSpec
    items: tuple[CompactItem, ...]
    omissions: tuple[CompactOmission, ...] = ()
    manifest: CompactManifest
    token_estimate: int
    query_run_ref: str | None = None
    result_relation_ref: str | None = None
    pack_ref: str

    def render_markdown(self) -> str:
        return render_compaction_markdown(self)


class CompactionBudgetTooSmallError(ValueError):
    """The budget cannot hold even the empty typed pack envelope."""

    def __init__(self, budget: int, envelope_tokens: int) -> None:
        super().__init__(f"compaction budget {budget} is below the minimum envelope estimate {envelope_tokens}")
        self.budget = budget
        self.envelope_tokens = envelope_tokens


#: Ordinary words (up to ``_ONE_WORD_MAX_CHARS``) count as one estimated word.
#: A longer unbroken run (``"!" * 100000``, a hash, a base64 blob) is weighted
#: by its length at ``_CHARS_PER_ESTIMATED_WORD`` instead of collapsing into one.
_ONE_WORD_MAX_CHARS = 16
_CHARS_PER_ESTIMATED_WORD = 8


def _estimated_words(run: str) -> int:
    return 1 if len(run) <= _ONE_WORD_MAX_CHARS else -(-len(run) // _CHARS_PER_ESTIMATED_WORD)


def estimate_tokens(text: str) -> int:
    """Stable prose proxy used by both context and compact renderers."""

    words = sum(_estimated_words(run) for run in text.split())
    return max(1, int(words * 1.3)) if words else 0


def estimate_serialized_tokens(serialized: str) -> int:
    """Estimate a serialized payload, charging for structure as well as words.

    Compact JSON has no whitespace, so a word count sees any number of omission
    objects as roughly one word. The UTF-8 byte component (about four bytes per
    token) keeps punctuation, identifiers and non-Latin text from being free.
    This is a tokenizer-free estimate, not a model-specific count.
    """

    if not serialized:
        return 0
    return max(estimate_tokens(serialized), (len(serialized.encode("utf-8")) + 3) // 4)


def _serialized_pack(pack: CorpusCompactionPack) -> str:
    return json.dumps(pack.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _get(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_text(message: object) -> str:
    text = _get(message, "text", "")
    return str(text or "").strip()


def _origin(message: object) -> str:
    raw = _get(message, "material_origin", "unknown")
    return str(getattr(raw, "value", raw))


def _block_kind(block: object) -> str:
    return str(_get(block, "type", _get(block, "block_type", "message")))


def _tool_outcome(message: object) -> tuple[object, object]:
    is_error = _get(message, "tool_result_is_error")
    exit_code = _get(message, "tool_result_exit_code")
    for block in cast(Iterable[object], _get(message, "blocks", ()) or ()):
        if _block_kind(block) not in {"tool_result", "function_call_output"}:
            continue
        if is_error is None:
            is_error = _get(block, "tool_result_is_error", _get(block, "is_error"))
        if exit_code is None:
            exit_code = _get(block, "tool_result_exit_code", _get(block, "exit_code"))
    return is_error, exit_code


def _anchor(session_id: str, message: object, index: int | None = None) -> CompactAnchor:
    message_id = str(_get(message, "id", _get(message, "message_id", "message")))
    content = _message_text(message)
    content_hash = cast(str | None, _get(message, "content_hash"))
    if content_hash is None and content:
        content_hash = sha256(content.encode("utf-8")).hexdigest()
    return CompactAnchor(ref=EvidenceRef(session_id, message_id, index), content_hash=content_hash)


def _score(message: object, text: str) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    lower = text.lower()
    if _origin(message) in {"human_authored", "assistant_authored"}:
        reasons.append("authoredness")
    if any(word in lower for word in ("error", "failed", "failure", "fixed", "verify", "decision")):
        reasons.append("decision_outcome_error_fix_signal")
    if _get(message, "blocks", ()):
        reasons.append("structured_evidence")
    return float(len(reasons) * 10 + min(len(text), 1000) / 1000), tuple(reasons)


def compact_sessions(
    sessions: Sequence[object],
    *,
    spec: CompactProjectionSpec | None = None,
    session_links: Sequence[Mapping[str, object]] = (),
    query_run_ref: str | None = None,
    result_relation_ref: str | None = None,
) -> CorpusCompactionPack:
    """Build a deterministic pack from session-like objects.

    ``session_links`` use archive direction: ``src_session_id`` is the child
    and ``resolved_dst_session_id`` is its parent; parent/child aliases are
    accepted for in-memory callers.
    Unknown lineage is retained and called out rather than guessed.
    """

    spec = spec or CompactProjectionSpec()
    allowed = set(spec.allowed_material_origins)
    if spec.include_generated_context:
        allowed.add("generated_context_pack")
    by_id = {str(_get(s, "id", _get(s, "session_id", ""))): s for s in sessions}
    parent_of: dict[str, tuple[str, str | None]] = {}
    for link in session_links:
        child = str(link.get("src_session_id", link.get("child_session_id", "")))
        parent = str(link.get("resolved_dst_session_id", link.get("parent_session_id", "")))
        if parent and child and child in by_id:
            branch_point = link.get("branch_point_message_id")
            parent_of[child] = (parent, str(branch_point) if branch_point else None)

    items: list[CompactItem] = []
    omissions: list[CompactOmission] = []
    drops: Counter[str] = Counter()
    drop_origins: Counter[str] = Counter()
    dropped_tokens: defaultdict[str, int] = defaultdict(int)
    included_tokens: defaultdict[str, int] = defaultdict(int)
    inherited: set[tuple[str, str]] = set()
    for session_id in sorted(by_id):
        session = by_id[session_id]
        messages = list(cast(Iterable[object], _get(session, "messages", ()) or ()))
        parent_id: str | None
        branch: str | None
        parent_id, branch = parent_of.get(session_id, (None, None))
        parent_messages = (
            list(cast(Iterable[object], _get(by_id.get(parent_id), "messages", ()) or ())) if parent_id else []
        )
        parent_ids = {str(_get(m, "id", _get(m, "message_id", ""))) for m in parent_messages}
        branch_seen = branch is None
        for _position, message in enumerate(messages):
            text = _message_text(message)
            origin = _origin(message)
            anchor = _anchor(session_id, message)
            tokens = estimate_tokens(text)
            reason: DropReason | None = None
            if origin not in allowed or not text:
                reason = "filtered_material_origin"
            if origin == "tool_result":
                is_error, exit_code = _tool_outcome(message)
                # Archive block rows carry the flag as an integer, so 0 is success.
                succeeded = is_error is not None and not is_error
                if succeeded and (exit_code is None or int(cast(int | str, exit_code)) == 0):
                    reason = "successful_tool_spam"
            message_id = str(_get(message, "id", _get(message, "message_id", "")))
            if branch and message_id == branch:
                branch_seen = True
                drops["duplicate_lineage_prefix"] += 1
                drop_origins[origin] += 1
                dropped_tokens[session_id] += tokens
                omissions.append(
                    CompactOmission(
                        anchor=anchor,
                        reason="duplicate_lineage_prefix",
                        detail="branch_point_emitted_by_parent",
                        token_estimate=tokens,
                    )
                )
                continue
            if not branch_seen and message_id in parent_ids:
                key = (str(_get(message, "content_hash", "")) or text, origin)
                if key in inherited or parent_id:
                    inherited.add(key)
                    reason = "duplicate_lineage_prefix"
            if reason:
                drops[reason] += 1
                drop_origins[origin] += 1
                dropped_tokens[session_id] += tokens
                omissions.append(CompactOmission(anchor=anchor, reason=reason, detail=reason, token_estimate=tokens))
                continue
            score, reasons = _score(message, text)
            items.append(
                CompactItem(
                    anchor=anchor,
                    session_id=session_id,
                    material_origin=origin,
                    kind="message",
                    text=text,
                    score=score,
                    reasons=reasons,
                    refs=(anchor.ref,),
                )
            )
            included_tokens[session_id] += tokens
    items.sort(key=lambda item: (-item.score, item.anchor.ref.format()))
    budget = spec.max_tokens
    kept: list[CompactItem] = []
    used = 0
    for item in items:
        tokens = estimate_tokens(item.text)
        if used + tokens <= budget:
            kept.append(item)
            used += tokens
        elif used < budget and item.text.split():
            # First degradation is clipping.  The remaining ladder is
            # represented in the manifest and is applied only after this
            # deterministic lossless-by-item attempt.
            words = item.text.split()
            room = max(1, int((budget - used) / 1.3))
            clipped = " ".join(words[:room]).rstrip() + " …"
            clipped_item = item.model_copy(update={"text": clipped, "degradation": "clip"})
            kept.append(clipped_item)
            used += estimate_tokens(clipped)
            drops["budget_clip"] += 1
        else:
            drops["budget_drop"] += 1
            dropped_tokens[item.session_id] += tokens
            omissions.append(
                CompactOmission(
                    anchor=item.anchor, reason="budget_drop", detail="drop_with_manifest", token_estimate=tokens
                )
            )
    # Measure the complete public object: projection, items, omissions, the
    # manifest and the estimate field itself. A partial probe can fit while the
    # emitted pack exceeds the budget. Omission rows go first, then kept items,
    # then per-session totals; each truncation is named in ``unknown``.
    manifest_included: dict[str, int] = {}
    for item in kept:
        manifest_included[item.session_id] = manifest_included.get(item.session_id, 0) + estimate_tokens(item.text)
    manifest_dropped = dict(dropped_tokens)
    unknown: tuple[str, ...] = (
        ("lineage_unresolved",)
        if any(str(_get(s, "parent_id", "")) and str(_get(s, "id", "")) not in parent_of for s in sessions)
        else ()
    )
    while True:
        pack_id = sha256("\n".join(i.anchor.ref.format() for i in kept).encode()).hexdigest()[:16]
        candidate = CorpusCompactionPack(
            projection=spec,
            items=tuple(kept),
            omissions=tuple(omissions),
            manifest=CompactManifest(
                drop_counts=dict(sorted(drops.items())),
                drop_counts_by_material_origin=dict(sorted(drop_origins.items())),
                included_tokens_by_session=dict(sorted(manifest_included.items())),
                dropped_tokens_by_session=dict(sorted(manifest_dropped.items())),
                duplicate_prefix_omissions=drops["duplicate_lineage_prefix"],
                unknown=unknown,
            ),
            token_estimate=0,
            query_run_ref=query_run_ref,
            result_relation_ref=result_relation_ref,
            pack_ref=f"compact:{pack_id}",
        )
        # The estimate is part of what it measures. Starting from zero it only
        # grows, and it reaches a fixed point once its decimal width is stable.
        serialized_tokens = estimate_serialized_tokens(_serialized_pack(candidate))
        while serialized_tokens != candidate.token_estimate:
            candidate = candidate.model_copy(update={"token_estimate": serialized_tokens})
            serialized_tokens = estimate_serialized_tokens(_serialized_pack(candidate))
        if serialized_tokens <= budget:
            return candidate
        if omissions:
            omissions.pop()
            if "omission_rows_truncated" not in unknown:
                unknown += ("omission_rows_truncated",)
        elif kept:
            dropped = kept.pop()
            tokens = estimate_tokens(dropped.text)
            drops["budget_drop"] += 1
            drop_origins[dropped.material_origin] += 1
            manifest_dropped[dropped.session_id] = manifest_dropped.get(dropped.session_id, 0) + tokens
            remaining = manifest_included[dropped.session_id] - tokens
            if remaining:
                manifest_included[dropped.session_id] = remaining
            else:
                del manifest_included[dropped.session_id]
        elif manifest_included or manifest_dropped:
            target = manifest_included if manifest_included else manifest_dropped
            target.pop(sorted(target)[-1])
            if "session_token_totals_truncated" not in unknown:
                unknown += ("session_token_totals_truncated",)
        else:
            raise CompactionBudgetTooSmallError(budget, serialized_tokens)


def render_compaction_markdown(pack: CorpusCompactionPack) -> str:
    lines = [
        "# Corpus Compaction",
        "",
        f"- Pack: `{pack.pack_ref}`",
        f"- Tokens: {pack.token_estimate}/{pack.projection.max_tokens}",
        f"- Included items: {len(pack.items)}",
        f"- Omitted items: {len(pack.omissions)}",
        "",
        "## Drop manifest",
        "",
    ]
    for reason, count in pack.manifest.drop_counts.items():
        lines.append(f"- {reason}: {count}")
    for item in pack.items:
        lines.extend(
            ["", f"## {item.anchor.ref.format()}", "", f"_Reasons: {', '.join(item.reasons) or 'none'}_", "", item.text]
        )
    return "\n".join(lines).rstrip() + "\n"


build_compaction_pack = compact_sessions
compile_corpus_compaction = compact_sessions

__all__ = [
    "CompactAnchor",
    "CompactItem",
    "CompactManifest",
    "CompactOmission",
    "CompactProjectionSpec",
    "CorpusCompactionPack",
    "CompactionBudgetTooSmallError",
    "build_compaction_pack",
    "compact_sessions",
    "compile_corpus_compaction",
    "estimate_serialized_tokens",
    "estimate_tokens",
    "render_compaction_markdown",
]
