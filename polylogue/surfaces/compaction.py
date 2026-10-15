"""Deterministic, evidence-linked corpus compaction.

Compaction is deliberately a projection, not a continuation context.  It
accepts already selected session/message-like objects and produces a bounded
pack whose omissions are part of the public result.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from polylogue.archive.message.compaction_store import CompactionStore

from pydantic import Field

from polylogue.analysis.archive_models import ArchiveInsightModel
from polylogue.archive.message import token_measure
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


_PLACEHOLDER_PACK_REF = "compact:" + "0" * 64


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


def _successful_tool_result(message: object) -> bool:
    outcomes = (
        _get(block, "tool_outcome")
        for block in cast(Iterable[object], _get(message, "blocks", ()) or ())
        if _block_kind(block) in {"tool_result", "function_call_output"}
    )
    seen = False
    for outcome in outcomes:
        seen = True
        if getattr(outcome, "value", outcome) != "ok":
            return False
    return seen


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
    sessions: Iterable[object],
    *,
    spec: CompactProjectionSpec | None = None,
    session_links: Iterable[Mapping[str, object]] = (),
    query_run_ref: str | None = None,
    result_relation_ref: str | None = None,
    checkpoint: Callable[[], None] = lambda: None,
) -> CorpusCompactionPack:
    """Compact one-pass session inputs on a request-owned native scratch store.

    Links use archive child-to-parent direction. Only a selected parent can
    account for an inherited prefix; an unselected parent never removes it.
    """
    from polylogue.archive.message.compaction_store import compaction_store

    with compaction_store(checkpoint) as stage:
        return _compact_staged(
            stage,
            sessions,
            spec=spec or CompactProjectionSpec(),
            session_links=session_links,
            query_run_ref=query_run_ref,
            result_relation_ref=result_relation_ref,
        )


def _compact_staged(
    stage: CompactionStore,
    sessions: Iterable[object],
    *,
    spec: CompactProjectionSpec,
    session_links: Iterable[Mapping[str, object]],
    query_run_ref: str | None,
    result_relation_ref: str | None,
) -> CorpusCompactionPack:
    from polylogue.archive.message.compaction_store import compact_json
    from polylogue.archive.message.token_measure import TextMeasure

    allowed = set(spec.allowed_material_origins)
    if spec.include_generated_context:
        allowed.add("generated_context_pack")
    # Stage scalar evidence once, then release the original message before
    # requesting its successor. Parent membership is resolved from these rows.
    for session in sessions:
        session_id = str(_get(session, "id", _get(session, "session_id", "")))
        parent = _get(session, "parent_id")
        stage.add_session(session_id, str(parent) if parent else None)
        for position, message in enumerate(cast(Iterable[object], _get(session, "messages", ()) or ())):
            stage.checkpoint()
            text = _message_text(message)
            score, reasons = _score(message, text)
            value = {
                "anchor": _anchor(session_id, message).model_dump(mode="json"),
                "text": text,
                "origin": _origin(message),
                "score": score,
                "reasons": reasons,
                "successful_tool": _successful_tool_result(message),
            }
            message_id = str(_get(message, "id", _get(message, "message_id", "")))
            stage.add_message(session_id, position, message_id, value)
            del message
        embedded_link = _get(session, "lineage_link")
        if isinstance(embedded_link, Mapping) and embedded_link.get("resolved_dst_session_id"):
            branch = embedded_link.get("branch_point_message_id")
            stage.add_link(session_id, str(embedded_link["resolved_dst_session_id"]), str(branch) if branch else None)
        del session
    for link in session_links:
        child = str(link.get("src_session_id", link.get("child_session_id", "")))
        parent = str(link.get("resolved_dst_session_id", link.get("parent_session_id", "")))
        if child and parent:
            branch = link.get("branch_point_message_id")
            stage.add_link(child, parent, str(branch) if branch else None)

    drops: Counter[str] = Counter()
    drop_origins: Counter[str] = Counter()
    unknown: tuple[str, ...] = (
        ("lineage_unresolved",) if stage.scalar("SELECT 1 FROM compact_sessions WHERE unresolved=1 LIMIT 1") else ()
    )

    def omission(anchor: object, reason: DropReason, detail: str, tokens: int, origin: str) -> None:
        drops[reason] += 1
        drop_origins[origin] += 1
        stage.add_omission({"anchor": anchor, "reason": reason, "detail": detail, "token_estimate": tokens})

    for row in stage.rows("SELECT id,parent_id,branch FROM compact_sessions ORDER BY id"):
        session_id = str(row[0])
        parent = str(row[1]) if row[1] is not None else None
        branch = str(row[2]) if row[2] is not None else None
        parent_selected = bool(parent and stage.scalar("SELECT 1 FROM compact_sessions WHERE id=?", (parent,)))
        branch_seen = branch is None
        previous_key: tuple[str, str] | None = None
        previous_position = -2
        run: int | None = None
        for message_row in stage.rows(
            "SELECT position,message_id,payload FROM compact_messages WHERE session_id=? ORDER BY position",
            (session_id,),
        ):
            position, message_id = int(message_row[0]), str(message_row[1])
            value = stage.load(int(message_row[2]))
            assert isinstance(value, dict)
            text, origin = str(value["text"]), str(value["origin"])
            tokens = token_measure.estimate_tokens(text)
            reason: DropReason | None = None
            detail = ""
            if origin not in allowed or not text:
                reason = "filtered_material_origin"
            if origin == "tool_result" and value["successful_tool"]:
                reason = "successful_tool_spam"
            if branch and message_id == branch:
                branch_seen = True
                if parent_selected:
                    reason, detail = "duplicate_lineage_prefix", "branch_point_emitted_by_parent"
            elif not branch_seen and parent_selected and stage.parent_has(parent, message_id):
                reason = "duplicate_lineage_prefix"
            if reason:
                stage.drop_tokens(session_id, tokens)
                omission(value["anchor"], reason, detail or reason, tokens, origin)
                continue
            anchor = value["anchor"]
            assert isinstance(anchor, dict)
            item = CompactItem(
                anchor=CompactAnchor.model_validate(anchor),
                session_id=session_id,
                material_origin=origin,
                kind="message",
                text=text,
                score=float(value["score"]),
                reasons=tuple(cast(Iterable[str], value["reasons"])),
                refs=(CompactAnchor.model_validate(anchor).ref,),
            )
            key = (origin, text)
            run = stage.add_candidate(
                item.model_dump(mode="json"),
                reference=item.anchor.ref.format(),
                run=run if key == previous_key and position == previous_position + 1 else None,
                tokens=tokens,
            )
            previous_key, previous_position = key, position

    budget = spec.max_tokens
    estimate = 0

    def account(
        identifier: int,
        item: dict[str, object],
        reason: DropReason,
        remaining: int,
        detail: str,
        *,
        refinement: bool = False,
    ) -> None:
        removed = stage.transfer(identifier, str(item["session_id"]), remaining)
        if refinement:
            stage.add_omission(
                {"anchor": item["anchor"], "reason": reason, "detail": detail, "token_estimate": removed}
            )
        else:
            omission(item["anchor"], reason, detail, removed, str(item["material_origin"]))

    def scalar_fields(*, pack_ref: str = _PLACEHOLDER_PACK_REF) -> dict[str, object]:
        gaps = tuple(reason for reason, count in sorted(drops.items()) if reason.startswith("budget_") and count)
        return {
            "projection": spec.model_dump(mode="json"),
            "items": None,
            "omissions": None,
            "manifest": {
                "drop_counts": dict(sorted(drops.items())),
                "drop_counts_by_material_origin": dict(sorted(drop_origins.items())),
                "included_tokens_by_session": None,
                "dropped_tokens_by_session": None,
                "duplicate_prefix_omissions": drops["duplicate_lineage_prefix"],
                "degradation_order": CompactManifest().degradation_order,
                "unknown": unknown,
            },
            "outcome": decide_outcome(matched=stage.count, degraded=(*gaps, *unknown)).model_dump(mode="json"),
            "token_estimate": estimate,
            "query_run_ref": query_run_ref,
            "result_relation_ref": result_relation_ref,
            "pack_ref": pack_ref,
        }

    def pieces(fields: Mapping[str, object], *, measuring: bool = False) -> Iterator[str | TextMeasure]:
        yield "{"
        for index, key in enumerate(sorted(fields)):
            stage.checkpoint()
            if index:
                yield ","
            yield compact_json(key) + ":"
            value = fields[key]
            if key in {"items", "omissions"}:
                if measuring:
                    yield stage.array_measure(key)
                else:
                    yield from stage.array_parts(key)
            elif key in {"included_tokens_by_session", "dropped_tokens_by_session"}:
                if measuring:
                    yield stage.totals_measure(key)
                else:
                    yield from stage.totals_parts(key)
            elif key == "manifest":
                assert isinstance(value, dict)
                yield from pieces(value, measuring=measuring)
            else:
                yield compact_json(value)
        yield "}"

    def measured_tokens() -> int:
        nonlocal estimate
        while True:
            measure = TextMeasure()
            for part in pieces(scalar_fields(), measuring=True):
                measure = measure.then(part if isinstance(part, TextMeasure) else TextMeasure.text(part))
            if measure.tokens == estimate:
                return estimate
            estimate = measure.tokens

    def finish_if_fits() -> CorpusCompactionPack | None:
        nonlocal unknown
        if measured_tokens() > budget and stage.omission_count:
            unknown += () if "omission_rows_truncated" in unknown else ("omission_rows_truncated",)
            # Prefix measures make omission trimming logarithmic rather than
            # repeatedly serializing a shrinking, potentially enormous list.
            low, high = 0, stage.omission_count - 1
            while low < high:
                middle = (low + high + 1) // 2
                stage.omission_count = middle
                if measured_tokens() <= budget:
                    low = middle
                else:
                    high = middle - 1
            stage.omission_count = low
        if measured_tokens() > budget:
            return None
        identity = sha256()
        for part in pieces(scalar_fields(pack_ref="")):
            assert isinstance(part, str)
            identity.update(part.encode("utf-8"))
        fields = scalar_fields(pack_ref="compact:" + identity.hexdigest())
        wire = "".join(cast(Iterator[str], pieces(fields)))
        return CorpusCompactionPack.model_validate_json(wire)

    def text_charge(text: str) -> int:
        return token_measure.estimate_serialized_tokens(json.dumps(text, ensure_ascii=False))

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

    stage.refresh()
    finished = finish_if_fits()
    if finished is not None:
        return finished
    overflow = measured_tokens() - budget
    available_prose = sum(text_charge(str(item["text"])) for _, item in stage.candidates()) - overflow
    allowance = max(1, (available_prose if available_prose > 0 else budget) // max(1, stage.count))
    while True:
        for identifier, item in stage.candidates():
            if text_charge(str(item["text"])) <= allowance:
                continue
            original_payload = stage.scalar("SELECT original FROM compact_candidates WHERE id=?", (identifier,))
            original = stage.load(int(cast(int, original_payload)))
            assert isinstance(original, dict)
            prefix = clipped_prefix(str(original["text"]), allowance)
            text = prefix + " …"
            if prefix == original["text"] or text == item["text"]:
                continue
            account(
                identifier,
                item,
                "budget_clip",
                token_measure.estimate_tokens(prefix),
                "removed_source_suffix",
                refinement=item["degradation"] == "clip",
            )
            item.update(text=text, degradation="clip")
            stage.update_candidate(identifier, item)
        stage.refresh()
        finished = finish_if_fits()
        if finished is not None:
            return finished
        overflow = measured_tokens() - budget
        if allowance == 1 or sum(text_charge(str(item["text"])) for _, item in stage.candidates()) <= overflow:
            break
        allowance = max(1, allowance - max(1, (overflow + stage.count - 1) // max(1, stage.count)))

    for row in stage.rows("SELECT run,COUNT(*) FROM compact_candidates GROUP BY run HAVING COUNT(*)>1 ORDER BY run"):
        run, count = int(row[0]), int(row[1])
        for index, member in enumerate(
            stage.rows("SELECT id,payload FROM compact_candidates WHERE run=? ORDER BY id", (run,))
        ):
            identifier = int(member[0])
            item = stage.load(int(member[1]))
            assert isinstance(item, dict)
            if index == 0:
                item.update(occurrence_count=count, degradation="collapse_runs_to_counts")
                stage.update_candidate(identifier, item)
            else:
                account(identifier, item, "budget_collapsed", 0, "repeated_source_text")
                stage.execute("UPDATE compact_candidates SET active=0 WHERE id=?", (identifier,))
    stage.refresh()
    finished = finish_if_fits()
    if finished is not None:
        return finished
    for identifier, item in stage.candidates():
        account(identifier, item, "budget_skeleton", 0, "prose_removed_anchor_retained")
        item.update(text="", degradation="skeleton_only")
        stage.update_candidate(identifier, item)
    stage.refresh()
    finished = finish_if_fits()
    if finished is not None:
        return finished
    while stage.count:
        identifier, item = stage.pop_candidate()
        account(identifier, item, "budget_drop", 0, "drop_with_manifest")
        # Skeletons have zero retained prose. Their original per-session zero
        # entry disappears when the last retained anchor of that session goes.
        session = str(item["session_id"])
        if not stage.scalar("SELECT 1 FROM compact_candidates WHERE active=1 AND session_id=? LIMIT 1", (session,)):
            stage.execute("UPDATE compact_totals SET inc_order=NULL WHERE session_id=?", (session,))
            # Rebuild scalar total prefixes, without rescanning candidate bodies.
            stage.refresh_totals()
        if not stage.count and "index_only_pack_failure" not in unknown:
            unknown += ("index_only_pack_failure",)
        finished = finish_if_fits()
        if finished is not None:
            return finished
    while stage.included_count or stage.dropped_count:
        if stage.included_count:
            stage.included_count -= 1
        else:
            stage.dropped_count -= 1
        if "session_token_totals_truncated" not in unknown:
            unknown += ("session_token_totals_truncated",)
        finished = finish_if_fits()
        if finished is not None:
            return finished
    raise CompactionBudgetTooSmallError(budget, measured_tokens())


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
    "render_compaction_markdown",
]
