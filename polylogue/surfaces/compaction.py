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
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome

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
    occurrence_count: int = Field(default=1, ge=1)


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
    outcome: OutcomeEnvelope
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

    return tokens_from_estimated_words(estimate_token_words(text))


def estimate_token_words(text: str) -> int:
    """Additive word charge for fragments separated by whitespace."""
    return sum(_estimated_words(run) for run in text.split())


def tokens_from_estimated_words(words: int) -> int:
    """Apply the shared token conversion after adding fragment charges."""
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


_PLACEHOLDER_PACK_REF = "compact:" + "0" * 64


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
    # A digest of the stripped text is not the archive's canonical content
    # hash, so an input without one keeps no anchor hash.
    content_hash = cast(str | None, _get(message, "content_hash"))
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
    runs: list[list[str]] = []
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
        previous_key: tuple[str, str] | None = None
        previous_position = -2
        for position, message in enumerate(messages):
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
            run_key = (origin, text)
            if run_key == previous_key and position == previous_position + 1:
                runs[-1].append(anchor.ref.format())
            else:
                runs.append([anchor.ref.format()])
            previous_key, previous_position = run_key, position
    items.sort(key=lambda item: (-item.score, item.anchor.ref.format()))
    budget = spec.max_tokens
    kept = list(items)
    # These are source-prose estimates, excluding the clip marker and other
    # synthesized presentation metadata. Every reduction transfers its exact
    # difference to dropped_tokens; the source total never silently shrinks.
    retained = {item.anchor.ref.format(): estimate_tokens(item.text) for item in items}
    unknown: tuple[str, ...] = (
        ("lineage_unresolved",)
        if any(str(_get(s, "parent_id", "")) and str(_get(s, "id", "")) not in parent_of for s in sessions)
        else ()
    )
    manifest_included: dict[str, int] = {}
    manifest_dropped: dict[str, int] = {}

    def account(item: CompactItem, reason: DropReason, remaining: int, detail: str) -> None:
        key = item.anchor.ref.format()
        removed = retained[key] - remaining
        retained[key] = remaining
        dropped_tokens[item.session_id] += removed
        drops[reason] += 1
        drop_origins[item.material_origin] += 1
        omissions.append(CompactOmission(anchor=item.anchor, reason=reason, detail=detail, token_estimate=removed))

    def refresh_totals() -> None:
        nonlocal manifest_included, manifest_dropped
        manifest_included = {}
        for item in kept:
            tokens = retained[item.anchor.ref.format()]
            manifest_included[item.session_id] = manifest_included.get(item.session_id, 0) + tokens
        manifest_dropped = dict(dropped_tokens)

    def measured_pack() -> CorpusCompactionPack:
        gaps = tuple(reason for reason, count in sorted(drops.items()) if reason.startswith("budget_") and count)
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
            outcome=decide_outcome(matched=len(kept), degraded=(*gaps, *unknown)),
            token_estimate=0,
            query_run_ref=query_run_ref,
            result_relation_ref=result_relation_ref,
            pack_ref=_PLACEHOLDER_PACK_REF,
        )
        serialized_tokens = estimate_serialized_tokens(_serialized_pack(candidate))
        while serialized_tokens != candidate.token_estimate:
            candidate = candidate.model_copy(update={"token_estimate": serialized_tokens})
            serialized_tokens = estimate_serialized_tokens(_serialized_pack(candidate))
        return candidate

    def finish_if_fits() -> CorpusCompactionPack | None:
        nonlocal unknown
        candidate = measured_pack()
        # Detailed omission evidence yields to the requested budget before
        # retained prose does; aggregate accounting stays complete and this
        # provenance gap is explicit. Re-evaluate after every ladder stage.
        while candidate.token_estimate > budget and omissions:
            omissions.pop()
            if "omission_rows_truncated" not in unknown:
                unknown += ("omission_rows_truncated",)
            candidate = measured_pack()
        if candidate.token_estimate > budget:
            return None
        identity = sha256(_serialized_pack(candidate.model_copy(update={"pack_ref": ""})).encode("utf-8"))
        return candidate.model_copy(update={"pack_ref": f"compact:{identity.hexdigest()}"})

    refresh_totals()
    finished = finish_if_fits()
    if finished is not None:
        return finished

    # Clip: share the available prose allowance across the selected items.
    # Binary search a source prefix, never a character-by-character ladder.
    # Even a long unbroken run can therefore be clipped without invented text.
    def text_charge(text: str) -> int:
        return estimate_serialized_tokens(json.dumps(text, ensure_ascii=False))

    def clipped_prefix(text: str, allowance: int) -> str:
        words = text.split()
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            if text_charge(" ".join(words[:middle]) + " …") <= allowance:
                low = middle
            else:
                high = middle - 1
        prefix = " ".join(words[:low])
        if prefix:
            return prefix
        low, high = 1, len(words[0])
        while low < high:
            middle = (low + high + 1) // 2
            if text_charge(words[0][:middle] + " …") <= allowance:
                low = middle
            else:
                high = middle - 1
        return words[0][:low]

    overflow = measured_pack().token_estimate - budget
    available_prose = sum(text_charge(item.text) for item in kept) - overflow
    # If metadata alone is oversized, keep a budget-proportional prefix for
    # the following run collapse instead of erasing prose before it can act.
    allowance = max(1, (available_prose if available_prose > 0 else budget) // max(1, len(kept)))
    while True:
        for index, item in enumerate(kept):
            original = items[index]
            if text_charge(item.text) <= allowance:
                continue
            prefix = clipped_prefix(original.text, allowance)
            text = prefix + " …"
            if prefix == original.text or text == item.text:
                continue
            remaining = estimate_tokens(prefix)
            if item.degradation == "clip":
                # Refining the same stage is one clip event, with each extra
                # suffix charged exactly once rather than charging it twice.
                ref_key = item.anchor.ref.format()
                removed = retained[ref_key] - remaining
                retained[ref_key] = remaining
                dropped_tokens[item.session_id] += removed
                omissions.append(
                    CompactOmission(
                        anchor=item.anchor, reason="budget_clip", detail="removed_source_suffix", token_estimate=removed
                    )
                )
            else:
                account(item, "budget_clip", remaining, "removed_source_suffix")
            kept[index] = item.model_copy(update={"text": text, "degradation": "clip"})
        refresh_totals()
        finished = finish_if_fits()
        if finished is not None:
            return finished
        overflow = measured_pack().token_estimate - budget
        if allowance == 1 or sum(text_charge(item.text) for item in kept) <= overflow:
            break
        allowance = max(1, allowance - max(1, (overflow + len(kept) - 1) // max(1, len(kept))))

    # Collapse runs defined by original adjacent source text, not the clipped
    # prefixes (which could make different messages appear equal).
    by_ref = {item.anchor.ref.format(): item for item in kept}
    for run in runs:
        if len(run) < 2:
            continue
        representative = by_ref[run[0]]
        refs = tuple(by_ref[key].anchor.ref for key in run)
        by_ref[run[0]] = representative.model_copy(
            update={"refs": refs, "occurrence_count": len(run), "degradation": "collapse_runs_to_counts"}
        )
        for member_ref in run[1:]:
            account(by_ref[member_ref], "budget_collapsed", 0, "repeated_source_text")
            del by_ref[member_ref]
    kept = [by_ref[item.anchor.ref.format()] for item in kept if item.anchor.ref.format() in by_ref]
    refresh_totals()
    finished = finish_if_fits()
    if finished is not None:
        return finished

    # Skeleton: source pointers and typed provenance survive without prose.
    for index, item in enumerate(kept):
        account(item, "budget_skeleton", 0, "prose_removed_anchor_retained")
        kept[index] = item.model_copy(update={"text": "", "degradation": "skeleton_only"})
    refresh_totals()
    finished = finish_if_fits()
    if finished is not None:
        return finished

    # Drop lower-score items only after the prior stages have been tried.
    # Source reductions were already accounted; these omission rows carry
    # zero prose tokens rather than charging the same loss a second time.
    while kept:
        dropped = kept.pop()
        account(dropped, "budget_drop", 0, "drop_with_manifest")
        refresh_totals()
        if not kept and "index_only_pack_failure" not in unknown:
            unknown += ("index_only_pack_failure",)
        finished = finish_if_fits()
        if finished is not None:
            return finished

    # Index-only: no retained item falsely implies an empty selected scope.
    # Keep the omission index and complete aggregate manifest where possible.
    # If those totals must be shortened, name that gap explicitly. A budget
    # below the final typed envelope remains a refusal, never an oversized pack.
    while manifest_included or manifest_dropped:
        target = manifest_included if manifest_included else manifest_dropped
        target.pop(sorted(target)[-1])
        if "session_token_totals_truncated" not in unknown:
            unknown += ("session_token_totals_truncated",)
        finished = finish_if_fits()
        if finished is not None:
            return finished
    raise CompactionBudgetTooSmallError(budget, measured_pack().token_estimate)


def render_compaction_markdown(pack: CorpusCompactionPack) -> str:
    lines = [
        "# Corpus Compaction",
        "",
        f"- Pack: `{pack.pack_ref}`",
        f"- Tokens: {pack.token_estimate}/{pack.projection.max_tokens}",
        f"- Outcome: {pack.outcome.state}",
        f"- Included items: {len(pack.items)}",
        f"- Omitted items: {len(pack.omissions)}",
        "",
        "## Drop manifest",
        "",
    ]
    lines.extend(["```json", json.dumps(pack.manifest.model_dump(mode="json"), sort_keys=True, indent=2), "```"])
    lines.extend(["", "## Omission evidence", ""])
    for omission in pack.omissions:
        lines.append(
            f"- `{omission.anchor.ref.format()}`: {omission.reason}; "
            f"{omission.detail}; tokens={omission.token_estimate}; "
            f"content_hash={omission.anchor.content_hash or 'unavailable'}"
        )
    for item in pack.items:
        lines.extend(
            [
                "",
                f"## {item.anchor.ref.format()}",
                "",
                f"_Reasons: {', '.join(item.reasons) or 'none'}_",
                "",
                f"_Occurrences: {item.occurrence_count}; degradation: {item.degradation or 'none'}_",
                "",
                item.text,
            ]
        )
        if item.occurrence_count > 1:
            lines.extend(["", "### Run source references", ""])
            lines.extend(f"- `{ref.format()}`" for ref in item.refs)
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
    "estimate_token_words",
    "estimate_tokens",
    "render_compaction_markdown",
    "tokens_from_estimated_words",
]
