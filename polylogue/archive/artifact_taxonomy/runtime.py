"""Artifact taxonomy classification runtime."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, BinaryIO, Literal, cast

from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind
from polylogue.archive.artifact_taxonomy.support import (
    is_subagent_path,
    looks_like_beads_interaction,
    looks_like_extracted_transcript_corpus,
    looks_like_extracted_transcript_record,
    looks_like_file_history_snapshot_only_stream,
    looks_like_hook_event,
    looks_like_record_entry,
    looks_like_session_document,
    looks_metadataish_dict,
    looks_metadataish_list,
    normalize_source_path,
    path_only_sidecar_reason,
    record_carries_provider_envelope,
)
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document

_HERMES_STATE_DB_MARKER = "hermes_state_db"
_HERMES_VERIFICATION_DB_MARKER = "hermes_verification_evidence_db"

# The taxonomy layer's exclusion of self-generated agent side-output
# (polylogue-omsw / polylogue-9ykn): scratch analysis artifacts an agent
# writes into its own Claude Code project directory, e.g. an index of prior
# conversation ids. The declared source layouts never reach an ``analysis/``
# directory, but a single-file acquisition route (``Source.path`` pointing directly at one
# file, bypassing ``os.walk``) never consults it, so a path like
# ``.../analysis/problem_solutions/problems_index.jsonl`` can still reach
# payload classification, where a generic JSONL-of-dicts heuristic
# (``looks_like_record_stream``) misreads its ``{"conversation": <id>,
# "type": ...}`` pointer records as session content purely because ``type``
# is a recordish key. Declaring the same exclusion here, on the path alone,
# closes that gap for every acquisition route rather than only the walk.
_SELF_GENERATED_ARTIFACT_DIR_SEGMENTS = frozenset({"analysis"})


def _has_self_generated_artifact_dir_segment(normalized_path: str) -> bool:
    inner = normalized_path.rsplit(":", 1)[-1]
    return any(part in _SELF_GENERATED_ARTIFACT_DIR_SEGMENTS for part in Path(inner).parts[:-1])


def _self_generated_artifact_dir_classification(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Weak, content-blind path heuristic: refuse anything under an
    ``analysis/`` directory segment.

    Deliberately split out of ``classify_artifact_path`` (polylogue-6mpy):
    this heuristic exists to catch self-generated side-output that never
    carries genuine conversation evidence (e.g. a sinex
    ``conversation_relationships.jsonl`` pointer index) when no content is
    available to classify (pre-decode, path-only filtering routes such as
    ``decoder_zip``/``source_walk`` skip-listing). But it is a *location*
    guess, not conversation evidence, and a genuine Claude Code session
    JSONL file can legitimately be re-homed or replayed from a path that
    happens to include an ``analysis`` segment. ``classify_artifact`` (the
    content-aware entry point) must let positive record content override
    this heuristic rather than let it win unconditionally -- see its own
    call site for the tie-break order.
    """
    provider_token = Provider.from_string(provider)
    normalized = normalize_source_path(source_path)
    if not normalized or not _has_self_generated_artifact_dir_segment(normalized):
        return None
    return ArtifactClassification(
        provider=provider_token,
        kind=ArtifactKind.METADATA_DOCUMENT,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="self-generated analysis artifact under an 'analysis/' directory "
        "(agent side-output, not conversation content)",
    )


def classify_artifact_path(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Classify obvious sidecars using only the source path.

    Path-only callers (pre-decode filtering: ``decoder_zip``, ``source_walk``
    skip-listing, schema sampling) get the weak ``analysis/`` directory
    heuristic first, same as always -- no content is available for them to
    weigh against it. ``classify_artifact`` (content-aware) instead calls
    ``_classify_artifact_path_strong`` directly and only falls back to the
    weak heuristic when content classification finds no positive evidence;
    see that function's call site.
    """
    if weak := _self_generated_artifact_dir_classification(source_path, provider=provider):
        return weak
    return strong_path_classification(source_path, provider=provider)


def strong_path_classification(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Classify only definitive path rules.

    Live admission uses this before deciding whether a payload may enter a
    bounded streaming route. The weak ``analysis/`` location heuristic is
    deliberately excluded there because it must yield to bounded payload
    evidence or the streaming policy.
    """
    return _classify_artifact_path_strong(source_path, provider=provider)


def fact_path_admits_session_content(source_path: str | Path | None, *, provider: str | Provider) -> bool:
    """Whether decoded session records may outrank this path's refusal.

    An OriginSpec ``fact`` rule names where a family's evidence usually sits,
    not what its bytes are, so records carrying a provider's session envelope
    there still reach the parser. A ``raw-only`` rule and the content-blind
    sidecar markers stay terminal.
    """
    normalized = normalize_source_path(source_path)
    if not normalized:
        return False
    from polylogue.sources.origin_specs import artifact_rule_for_path

    rule = artifact_rule_for_path(Provider.from_string(provider), normalized)
    return rule is not None and rule.parse_policy == "fact"


def _classify_artifact_path_strong(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Classify obvious sidecars by path, excluding the weak ``analysis/``
    directory heuristic (split out so ``classify_artifact`` can let positive
    record content override that one heuristic; polylogue-6mpy)."""
    provider_token = Provider.from_string(provider)
    normalized = normalize_source_path(source_path)
    if not normalized:
        return None

    # Import lazily: ``sources`` imports decoder helpers which in turn depend
    # on this taxonomy during package bootstrap.  Classification happens after
    # that bootstrap, while OriginSpec remains the owner of the actual rules.
    from polylogue.sources.origin_specs import artifact_rule_for_path

    inner_name = Path(normalized.rsplit(":", 1)[-1]).name.lower()
    if rule := artifact_rule_for_path(provider_token, normalized):
        return ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind(rule.kind),
            parse_as_session=rule.parse_policy == "session",
            schema_eligible=rule.parse_policy == "session",
            default_priority=120 if rule.parse_policy == "session" else 80,
            reason=f"OriginSpec {provider_token.value} artifact rule: {rule.coverage_role}",
        )
    # polylogue-omsw: generic/ad-hoc acquisition routes (the daemon's "inbox"
    # drop directory, the ingest operation behind `polylogue import <path>`, and this
    # taxonomy's own `classify_artifact_path` pre-decode callers) resolve a
    # provider hint of "unknown" or a shape-detected non-Claude-Code provider
    # for these paths -- they never learn the file actually sits under a
    # watched Claude Code project tree. `tool-results/<name>.json` is a
    # directory-name pattern specific enough to Claude Code's own artifact
    # family that it is safe to check regardless of the caller-supplied
    # provider hint (a tool call's OWN output can coincidentally look like a
    # session document from a different provider -- see the
    # ``TOOL_RESULT_SIDECAR`` ``ArtifactKind`` docstring -- which is exactly
    # the scenario this closes). Scoped narrowly to the ``tool_result_sidecar``
    # rule only: other Claude Code path rules (``coordinator_session_stream``
    # in particular) match directory shapes too generic to safely check
    # provider-agnostically.
    if provider_token is not Provider.CLAUDE_CODE:
        tool_result_rule = artifact_rule_for_path(Provider.CLAUDE_CODE, normalized)
        if tool_result_rule is not None and tool_result_rule.kind == "tool_result_sidecar":
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind(tool_result_rule.kind),
                parse_as_session=False,
                schema_eligible=False,
                default_priority=80,
                reason=f"OriginSpec Claude artifact rule (provider-agnostic path match): "
                f"{tool_result_rule.coverage_role}",
            )
    if provider_token is Provider.HERMES and inner_name in {
        "verification_evidence.db",
        "verification_evidence.sqlite",
        "verification_evidence.sqlite3",
    }:
        # Path-only classification (pre-JSON-decode filtering, e.g. schema
        # sampling) must stay non-session here even though a real parser now
        # exists (polylogue-wj25): raw bytes at this path are still SQLite
        # binary, not the JSON marker payload the parser actually consumes.
        # Same split as state.db: the *positive* session classification
        # lives on the marker payload below (classify_artifact), never on
        # the raw path -- see `_HERMES_STATE_DB_MARKER` for the precedent.
        return ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="Hermes SQLite evidence sidecar",
        )
    if provider_token is Provider.ANTIGRAVITY:
        if inner_name.endswith(".metadata.json"):
            # Brain metadata is a sidecar, never a primary session: fragmenting
            # one file per artifact produced single-message sessions that were
            # noise (all real conversation content lives in the .pb trajectories
            # the language-server export route acquires directly -- polylogue-eo81,
            # GH #1764). Still accounted for via ``raw_artifacts.artifact_kind``
            # rather than silently dropped.
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind.AGENT_SIDECAR_META,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Antigravity brain-artifact metadata sidecar (superseded by "
                "language-server conversation export; polylogue-eo81)",
            )
        if inner_name.endswith((".pb", ".pbtxt", ".resolved")) or ".resolved." in inner_name:
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind.METADATA_DOCUMENT,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Antigravity opaque or resolved sidecar",
            )
        if inner_name in {
            "browserallowlist.txt",
            "installation_id",
            "knowledge.lock",
            "mcp_config.json",
            "user_settings.pb",
        }:
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind.METADATA_DOCUMENT,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Antigravity configuration sidecar",
            )
    if sidecar_reason := path_only_sidecar_reason(inner_name):
        kind = ArtifactKind.BRIDGE_POINTER if inner_name == "bridge-pointer.json" else ArtifactKind.SESSION_INDEX
        return ArtifactClassification(
            provider=provider_token,
            kind=kind,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason=sidecar_reason,
        )

    if inner_name.startswith("agent-") and inner_name.endswith(".meta.json"):
        return ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind.AGENT_SIDECAR_META,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="agent sidecar metadata path",
        )

    return None


def classify_artifact(
    payload: JSONValue,
    *,
    provider: str | Provider,
    source_path: str | Path | None = None,
) -> ArtifactClassification:
    """Classify a payload/document into a session or sidecar cohort."""
    provider_token = Provider.from_string(provider)

    # Hermes SQLite marker payloads (state.db / verification_evidence.db)
    # must win over the path-only "SQLite evidence sidecar" classification
    # below (polylogue-zoc3). The raw *.db path itself always classifies as
    # a non-session sidecar (its bytes are still opaque SQLite, not a JSON
    # marker) -- see `classify_artifact_path`'s "Hermes SQLite evidence
    # sidecar" branch -- but once the raw payload has been decoded into the
    # synthetic marker dict (`build_raw_payload_envelope`'s
    # `_hermes_sqlite_marker_payload`), `source_path` still points at the
    # same *.db filename. Checking the marker dict first, before consulting
    # `classify_artifact_path`, keeps that positive session classification
    # from being shadowed by the path-only sidecar rule for the exact same
    # filename.
    if isinstance(payload, dict):
        marker_classification = _classify_hermes_sqlite_marker(payload, provider=provider_token)
        if marker_classification is not None:
            return marker_classification

    # ``_classify_artifact_path_strong`` covers the definitive, content-blind
    # path rules (OriginSpec artifact rules, known sidecar filenames, Hermes/
    # Antigravity path markers) -- these always win regardless of content,
    # with one deliberate exception checked immediately below.
    explicit = _classify_artifact_path_strong(source_path, provider=provider_token)
    if explicit is not None and not explicit.parse_as_session:
        if (
            isinstance(payload, Sequence)
            and not isinstance(payload, str | bytes | bytearray)
            and fact_path_admits_session_content(source_path, provider=provider_token)
        ):
            # A record sequence at a fact path takes the complete record fold,
            # where decoded session evidence outranks the location.
            return classify_artifact_records(payload, provider=provider_token, source_path=source_path).classification
        return explicit

    # A path rule that admits a session (``coordinator_session_stream`` and
    # its siblings) asserts only that the file sits where the provider writes
    # transcripts. Records that name the transcript their turns were copied
    # out of are a generated derivative wherever they sit, so that evidence
    # outranks the location -- otherwise the extract becomes a session keyed
    # on its own filename stem, republishing the original's turns. Same
    # direction as ``_file_history_snapshot_override``, generalized: positive
    # content refusal beats a positive path-only verdict.
    extracted = _extracted_transcript_corpus_classification(payload, provider=provider_token)
    if extracted is not None:
        return extracted

    if explicit is not None:
        override = _file_history_snapshot_override(explicit, payload, provider=provider_token)
        return override if override is not None else explicit

    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        content_classification = _classify_list(payload, provider=provider_token, source_path=source_path)
    elif isinstance(payload, dict):
        content_classification = _classify_dict(payload, provider=provider_token, source_path=source_path)
    else:
        content_classification = ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind.UNKNOWN,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="non-object payload",
        )

    # polylogue-6mpy: positive conversational evidence in the record content
    # (recognised session/record shape) outranks the weak, content-blind
    # ``analysis/`` directory heuristic -- a genuine session record must not
    # be refused merely because its replay/backfill path happens to route
    # through a directory segment named "analysis". The heuristic still wins
    # when content classification found no positive evidence at all, which
    # is exactly the polylogue-9ykn direction: an unrecognised record stays
    # refused, never defaults to a session.
    if content_classification.parse_as_session:
        return content_classification
    weak = _self_generated_artifact_dir_classification(source_path, provider=provider_token)
    if weak is not None:
        return weak
    return content_classification


def _extracted_transcript_corpus_classification(
    payload: JSONValue,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Classify a stream of turns copied out of transcripts it names.

    Provider-agnostic: the evidence is the records' own declared provenance
    plus the absence of any provider record envelope, never a filename, a
    directory segment or a producer-specific report schema.
    """
    if isinstance(payload, dict):
        dict_items = iter((payload,))
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        dict_items = (document for item in payload if (document := json_document(item)))
    else:
        return None
    if not looks_like_extracted_transcript_corpus(dict_items):
        return None
    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.EXTRACTED_TRANSCRIPT_CORPUS,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="extracted transcript corpus: records carry copied turns and name the transcript they came from",
    )


def _is_bare_codex_session_meta_stream(payload: object) -> bool:
    """True when EVERY record of the stream is an exact bare Codex header.

    Like the surrounding stream predicates, this scans the complete payload.
    A positive result admits the stream
    as a parseable session, so a prefix of bare headers followed by real records
    would let this narrow recovery shape claim a file it was never meant to.

    The loop returns on the first record that is not an exact bare header, so a
    stream that is not this shape costs one record, and one that is costs
    exactly what it is -- a file of nothing but headers.
    """
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes | bytearray):
        return False
    seen = 0
    for item in payload:
        document = json_document(item)
        if not document:
            continue
        if document != {"type": "session_meta"}:
            return False
        seen += 1
    return seen > 1


def _file_history_snapshot_override(
    explicit: ArtifactClassification,
    payload: JSONValue,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Override a path-rule session verdict for a pure file-history stream.

    polylogue-omsw: ``coordinator_session_stream`` (``projects/<proj>/
    <uuid>.jsonl``) is a path-only rule that cannot distinguish a genuine
    Claude Code session from a session-uuid-named file whose only records
    are file-history checkpoints -- Claude Code writes both shapes under the
    identical path pattern. Positive content evidence (every record's
    ``type`` is a known non-conversational envelope kind) must win over that
    path-only positive verdict, the same direction ``classify_artifact``'s
    ``analysis/`` weak-heuristic override already takes, just the opposite
    polarity: there, weak path evidence loses to positive content; here,
    positive path evidence loses to negative (refusing) content evidence.
    """
    if provider is not Provider.CLAUDE_CODE or not explicit.parse_as_session:
        return None
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes | bytearray):
        return None
    # The complete payload, not a 32-record prefix: a positive result here
    # OVERRIDES a positive session verdict, so deciding it on a prefix dropped
    # any real session whose first records happened to be checkpoints. The
    # predicate scans lazily and exits on the first non-checkpoint record, so a
    # genuine session still costs only its first few records.
    if not looks_like_file_history_snapshot_only_stream(json_document(item) for item in payload):
        return None
    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.FILE_HISTORY_SNAPSHOT,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="Claude Code file-history-snapshot-only stream (no conversational records)",
    )


@dataclass(slots=True)
class _RecordArtifactEvidence:
    document_count: int = 0
    record_count: int = 0
    all_atof: bool = True
    all_hooks: bool = True
    all_beads: bool = True
    any_session: bool = False
    extracted: bool = False
    provider_envelope: bool = False
    all_checkpoints: bool = True
    saw_checkpoint: bool = False
    checkpoint_disqualified: bool = False
    all_bare_codex_headers: bool = True
    all_codex_session_meta: bool = True

    def observe(self, item: JSONDocument) -> None:
        from polylogue.sources.parsers.hermes_spans import looks_like_atof_payload

        if not item:
            return
        self.document_count += 1
        self.all_bare_codex_headers &= len(item) == 1 and item == {"type": "session_meta"}
        self.all_codex_session_meta &= item.get("type") == "session_meta"
        self.record_count += int(looks_like_record_entry(item))
        self.all_atof = self.all_atof and looks_like_atof_payload(item)
        self.all_hooks = self.all_hooks and looks_like_hook_event(item)
        self.all_beads = self.all_beads and looks_like_beads_interaction(item)
        self.any_session = self.any_session or looks_like_session_document(item)
        self.extracted = self.extracted or looks_like_extracted_transcript_record(item)
        self.provider_envelope = self.provider_envelope or record_carries_provider_envelope(item)
        record_type = item.get("type")
        if isinstance(record_type, str):
            self.saw_checkpoint |= record_type in {"file-history-snapshot", "progress"}
            self.checkpoint_disqualified |= record_type not in {"file-history-snapshot", "progress"}
        self.all_checkpoints = (
            self.all_checkpoints
            and isinstance(record_type, str)
            and record_type in {"file-history-snapshot", "progress"}
        )


def _record_candidacy_from_evidence(
    evidence: _RecordArtifactEvidence,
    specific_document: bool,
    *,
    provider: Provider,
    source_path: str | Path | None,
    fact_path_recovery: bool,
) -> ArtifactClassification | None:
    if not evidence.document_count or (evidence.extracted and not evidence.provider_envelope):
        return None
    explicit = strong_path_classification(source_path, provider=provider)
    if explicit is not None and not fact_path_recovery:
        if not explicit.parse_as_session:
            return None
        if provider is Provider.CLAUDE_CODE and evidence.all_checkpoints:
            return None
        return replace(explicit, schema_eligible=False)
    if evidence.all_hooks or evidence.all_beads:
        return None
    if provider is Provider.HERMES:
        positive_record = evidence.all_atof
    else:
        positive_record = evidence.record_count * 2 >= evidence.document_count
    if fact_path_recovery:
        positive_record = evidence.record_count > 0 and evidence.provider_envelope
    if not positive_record and not (evidence.any_session or specific_document):
        return None
    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.SESSION_RECORD_STREAM if positive_record else ArtifactKind.SESSION_DOCUMENT,
        parse_as_session=True,
        schema_eligible=False,
        default_priority=120,
        reason="complete artifact candidacy; full-record parser and schema validation remain unmeasured",
    )


@dataclass(frozen=True, slots=True)
class ArtifactStreamClassification:
    """Complete admission evidence, with refusal distinguished from unknown content."""

    classification: ArtifactClassification
    proved_non_session: bool
    record_count: int = 0


def classify_artifact_stream(
    handle: IO[bytes],
    *,
    provider: Provider,
    source_path: str | Path | None = None,
    wire_format: Literal["json", "jsonl"],
    check_stop: Callable[[], None] | None = None,
) -> ArtifactStreamClassification:
    """Classify complete caller-owned input, privately replaying non-seekable streams."""
    from polylogue.archive.raw_payload.streams import rewindable_byte_stream

    if (declared := declared_evidence_classification(source_path, provider=provider)) is not None:
        return ArtifactStreamClassification(declared, True, 0)
    with rewindable_byte_stream(handle, check_stop=check_stop) as stream:
        return _classify_seekable_artifact_stream(
            cast(BinaryIO, stream),
            provider=provider,
            source_path=source_path,
            wire_format=wire_format,
            check_stop=check_stop,
        )


def declared_evidence_classification(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Return the terminal classification of a declared ``raw-only`` evidence path.

    A ``raw-only`` rule states its bytes are evidence and never a session,
    and that content shape cannot decide otherwise (a hook carrier, a Markdown
    memory document, a tool-result sidecar). Its classification is the
    declaration itself: the bytes are never decoded as a session grammar, so
    non-JSON evidence or one malformed carrier line cannot become a decode
    refusal of the retained artifact.
    """
    normalized = normalize_source_path(source_path)
    if not normalized:
        return None
    classification = strong_path_classification(normalized, provider=provider)
    if classification is None or classification.parse_as_session:
        return None
    from polylogue.sources.origin_specs import path_declaration_refuses_session

    if classification.kind is not ArtifactKind.TOOL_RESULT_SIDECAR and not path_declaration_refuses_session(
        classification.provider, normalized
    ):
        # The tool-result rule is matched provider-agnostically above.
        return None
    return classification


def _classify_seekable_artifact_stream(
    handle: BinaryIO,
    *,
    provider: Provider,
    source_path: str | Path | None,
    wire_format: Literal["json", "jsonl"],
    check_stop: Callable[[], None] | None,
) -> ArtifactStreamClassification:
    """Fold the whole accepted input; the canonical parser owns session validation.

    The caller owns the handle and any accepted JSONL-prefix boundary. Syntax,
    cancellation and I/O failures propagate; no prefix sample proves refusal.
    """
    import json
    from contextlib import closing
    from itertools import chain

    import ijson

    from polylogue.archive.artifact_taxonomy.support import record_candidacy_projection
    from polylogue.sources.detection_projection import iter_projected_document_records, iter_projected_jsonl_records

    position = handle.tell()
    encoding = json.detect_encoding(handle.read(4))
    handle.seek(position)
    sequence = wire_format == "jsonl"
    callback_failure: BaseException | None = None

    def checkpoint() -> None:
        nonlocal callback_failure
        if check_stop is not None:
            try:
                check_stop()
            except BaseException as exc:
                callback_failure = exc
                raise

    def root(kind: Literal["record", "sequence"]) -> None:
        nonlocal sequence
        sequence = wire_format == "jsonl" or kind == "sequence"

    def measure(records: Generator[object, None, None]) -> ArtifactStreamClassification:
        with closing(records):
            try:
                first = next(records)
            except StopIteration:
                values: Iterator[object] = iter(())
            else:
                values = chain((first,), records)
            return _classify_artifact_records(
                values,
                provider=provider,
                source_path=source_path,
                sequence=sequence,
                empty_jsonl=wire_format == "jsonl",
                check_stop=checkpoint,
            )

    # A complete first physical value followed by later line bytes cannot be
    # one document. Prove that grammar without decoding a potentially huge
    # single document, then fold the existing strict projected record stream.
    if wire_format == "jsonl" and encoding in {"utf-8", "utf-8-sig"}:
        from polylogue.core.json_envelope import jsonl_has_record_successor

        if jsonl_has_record_successor(handle, check_stop=checkpoint):
            try:
                return measure(
                    iter_projected_jsonl_records(handle, record_candidacy_projection(), check_stop=checkpoint)
                )
            except (ijson.JSONError, UnicodeError, json.JSONDecodeError):
                if callback_failure is not None:
                    raise callback_failure from None
                handle.seek(position)
                sequence = True

    # A physical JSONL file can contain one complete document/array. Preserve
    # that grammar before treating physical lines as separate record inputs.
    try:
        return measure(
            iter_projected_document_records(
                handle,
                record_candidacy_projection(),
                encoding=encoding,
                check_stop=checkpoint,
                on_root=root,
            )
        )
    except (ijson.JSONError, UnicodeError, json.JSONDecodeError):
        if callback_failure is not None:
            raise callback_failure from None
        if wire_format == "json":
            raise
        handle.seek(position)
        sequence = True
        return measure(iter_projected_jsonl_records(handle, record_candidacy_projection(), check_stop=checkpoint))


def classify_artifact_records(
    records: Iterable[object],
    *,
    provider: Provider,
    source_path: str | Path | None = None,
    check_stop: Callable[[], None] | None = None,
) -> ArtifactStreamClassification:
    """Fold an exhausted record stream without replacing unknown with refusal."""
    return _classify_artifact_records(
        records, provider=provider, source_path=source_path, sequence=True, empty_jsonl=True, check_stop=check_stop
    )


def _classify_artifact_records(
    records: Iterable[object],
    *,
    provider: Provider,
    source_path: str | Path | None,
    sequence: bool,
    empty_jsonl: bool,
    check_stop: Callable[[], None] | None,
) -> ArtifactStreamClassification:
    # At a ``fact`` path, decoded session evidence outranks the location: the
    # rule's refusal stands only when the records carry none.
    fact_path_recovery = fact_path_admits_session_content(source_path, provider=provider)
    evidence = _RecordArtifactEvidence()
    count = 0
    all_metadata = True
    specific_document = False
    first_classification: ArtifactClassification | None = None
    codex_unsupported_record = False
    from polylogue.sources.parsers.codex import is_legacy_response_record, is_supported_outer_record

    def result(classification: ArtifactClassification, proved: bool) -> ArtifactStreamClassification:
        return ArtifactStreamClassification(classification, proved, count)

    for value in records:
        check_compute_cancelled()
        if check_stop is not None:
            check_stop()
        count += 1
        item = json_document(value)
        evidence.observe(item)
        if provider is Provider.CODEX:
            codex_unsupported_record |= not is_supported_outer_record(value)
            # An unwrapped 2025 response record (function_call, its output,
            # reasoning) carries no generic envelope marker, yet the parser
            # materializes it; count it as record evidence so a rollout made
            # mostly of tool calls still clears the record majority.
            evidence.record_count += int(is_legacy_response_record(value) and not looks_like_record_entry(item))
        all_metadata &= isinstance(value, str | int | float | bool | type(None)) or (
            isinstance(value, dict) and looks_metadataish_dict(item)
        )
        classification = classify_artifact(cast(JSONValue, value), provider=provider, source_path=source_path)
        if count == 1:
            first_classification = classification
        specific_document |= classification.parse_as_session
    if empty_jsonl and not count:
        classification = ArtifactClassification(
            provider, ArtifactKind.UNKNOWN, False, False, 0, "no complete JSONL artifact records"
        )
        return result(classification, False)
    if not sequence and first_classification is not None and not fact_path_recovery:
        classification = replace(first_classification, schema_eligible=False)
        return result(
            classification,
            not classification.parse_as_session
            and (
                classification.kind is not ArtifactKind.UNKNOWN or evidence.all_beads and bool(evidence.document_count)
            ),
        )
    explicit = strong_path_classification(source_path, provider=provider)
    if explicit is not None and not explicit.parse_as_session and not fact_path_recovery:
        return result(explicit, True)
    if (
        provider is Provider.CODEX
        and evidence.document_count
        and evidence.all_codex_session_meta
        and not (evidence.document_count > 1 and evidence.all_bare_codex_headers)
    ):
        # A Codex stream of nothing but ``session_meta`` headers carries no
        # conversation records, wherever it lives: it is not a session. The
        # narrow repeated-bare-header recovery shape below stays admitted.
        classification = ArtifactClassification(
            provider,
            ArtifactKind.METADATA_DOCUMENT,
            False,
            False,
            0,
            "Codex session-meta-only stream without conversation records",
        )
        return result(classification, True)
    if provider is Provider.CODEX and sequence and codex_unsupported_record:
        # The parser would drop such a record and report the rest as the
        # whole session. The same contract ``classify_artifact`` applies to a
        # complete payload refuses the stream instead, ahead of any path rule,
        # so the drop surfaces as a typed unsupported shape.
        classification = ArtifactClassification(
            provider,
            ArtifactKind.UNKNOWN,
            False,
            False,
            0,
            "Codex record stream contains unsupported session records",
        )
        return result(classification, False)
    if evidence.extracted and not evidence.provider_envelope:
        classification = ArtifactClassification(
            provider, ArtifactKind.EXTRACTED_TRANSCRIPT_CORPUS, False, False, 0, "extracted transcript corpus"
        )
        return result(classification, True)
    if (
        provider is Provider.CLAUDE_CODE
        and explicit is not None
        and explicit.parse_as_session
        and evidence.saw_checkpoint
        and not evidence.checkpoint_disqualified
    ):
        classification = ArtifactClassification(
            provider,
            ArtifactKind.FILE_HISTORY_SNAPSHOT,
            False,
            False,
            0,
            "Claude Code file-history-snapshot-only stream",
        )
        return result(classification, True)
    if explicit is not None and not fact_path_recovery:
        return result(replace(explicit, schema_eligible=False), False)
    if evidence.document_count and evidence.all_hooks:
        classification = ArtifactClassification(
            provider, ArtifactKind.HOOK_EVENT, False, False, 100, "hook event stream"
        )
        return result(classification, True)
    if evidence.document_count and evidence.all_beads:
        classification = ArtifactClassification(
            Provider.UNKNOWN, ArtifactKind.UNKNOWN, False, False, 0, "Beads interaction-history artifact"
        )
        return result(classification, True)
    if provider is Provider.CODEX and evidence.document_count > 1 and evidence.all_bare_codex_headers:
        classification = ArtifactClassification(
            provider,
            ArtifactKind.SESSION_RECORD_STREAM,
            True,
            False,
            120,
            "repeated bare Codex session-meta record stream",
        )
        return result(classification, False)
    candidacy = _record_candidacy_from_evidence(
        evidence,
        specific_document,
        provider=provider,
        source_path=source_path,
        fact_path_recovery=fact_path_recovery,
    )
    if candidacy is not None:
        return result(candidacy, False)
    if explicit is not None and not explicit.parse_as_session:
        return result(explicit, True)
    if all_metadata:
        classification = ArtifactClassification(
            provider,
            ArtifactKind.METADATA_DOCUMENT,
            False,
            False,
            0,
            "metadata-oriented list payload" if count else "empty list payload",
        )
        return result(classification, True)
    weak = _self_generated_artifact_dir_classification(source_path, provider=provider)
    if weak is not None:
        return result(weak, True)
    classification = ArtifactClassification(
        provider, ArtifactKind.UNKNOWN, False, False, 0, "unrecognized artifact stream"
    )
    return result(classification, False)


def _classify_list(
    payload: Sequence[JSONValue],
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    if not payload:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="empty list payload",
        )
    # Fold the complete stream without retaining its decoded records.
    evidence = _RecordArtifactEvidence()
    for value in payload:
        evidence.observe(json_document(value))

    if provider is Provider.HERMES:
        if evidence.document_count and evidence.all_atof:
            return ArtifactClassification(
                provider=provider,
                kind=ArtifactKind.SESSION_RECORD_STREAM,
                parse_as_session=True,
                schema_eligible=True,
                default_priority=110,
                reason="Hermes NeMo Relay ATOF observer event stream",
            )
        # Hermes's other durable source classes (state.db, verification
        # evidence, ATIF trajectory documents, session snapshots) are all
        # single JSON *documents*, never a bare JSON array -- so a
        # list-shaped payload under a Hermes-tagged source that isn't an
        # ATOF stream has no legitimate Hermes session shape to match.
        # Falling through to the generic ``looks_like_record_stream``
        # heuristic below let a skill prompt-prefill template
        # (``optional-skills/**/templates/*.json``, a bare array of
        # ``{"role", "content"}`` pairs) get claimed as a Hermes session
        # purely because that shape also satisfies the generic
        # role/content-key check -- the Hermes watch source recursively
        # scans its entire home directory, not just a sessions subtree, so
        # any non-session JSON array living there reaches this classifier
        # (polylogue-omsw class; dyica classification 2026-08-19 bucket B6).
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.UNKNOWN,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="Hermes source has no list-shaped session artifact other than an ATOF event stream",
        )

    if evidence.document_count and evidence.all_hooks:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.HOOK_EVENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=100,
            reason="hook event stream",
        )

    if evidence.document_count and evidence.all_beads:
        return ArtifactClassification(
            provider=Provider.UNKNOWN,
            kind=ArtifactKind.UNKNOWN,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="Beads interaction-history artifact, not a session stream",
        )

    if provider is Provider.GEMINI_CLI:
        from polylogue.sources.parsers.local_agent import is_gemini_cli_checkpoint_stream

        if is_gemini_cli_checkpoint_stream(payload):
            return ArtifactClassification(
                provider=provider,
                kind=ArtifactKind.SESSION_RECORD_STREAM,
                parse_as_session=False,
                schema_eligible=True,
                default_priority=120,
                reason="Gemini CLI checkpoint schema evidence",
            )

    if provider is Provider.CODEX:
        from polylogue.sources.parsers.codex import is_supported_session_stream

        if is_supported_session_stream(payload):
            subagent = is_subagent_path(source_path)
            kind = ArtifactKind.AGENT_TRANSCRIPT if subagent else ArtifactKind.SESSION_RECORD_STREAM
            return ArtifactClassification(
                provider=provider,
                kind=kind,
                parse_as_session=True,
                schema_eligible=True,
                default_priority=90 if subagent else 120,
                reason="parser-supported Codex session record stream",
            )

    # A Codex rollout can be truncated to repeated bare session headers while
    # still remaining a JSONL record stream.  A single bare ``type`` is too
    # weak to admit generically, but multiple exact Codex session-meta records
    # with a declared Codex origin are provider-specific structural evidence.
    # Keep this before the generic record predicate so the narrow recovery
    # shape reaches schema inference without reopening the generic type-only
    # false-positive class.
    # Decided on the COMPLETE payload, rather than a record prefix: this
    # branch admits a stream as a session, so a prefix of bare headers followed
    # by any other record would admit a file this rule was never meant to
    # claim. The scan exits on the first non-matching record.
    if provider is Provider.CODEX and _is_bare_codex_session_meta_stream(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_RECORD_STREAM,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="repeated bare Codex session-meta record stream",
        )
    if provider is Provider.CODEX and evidence.record_count:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.UNKNOWN,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="Codex record stream contains unsupported session records",
        )

    if evidence.document_count and evidence.record_count * 2 >= evidence.document_count:
        subagent = is_subagent_path(source_path)
        kind = ArtifactKind.AGENT_TRANSCRIPT if subagent else ArtifactKind.SESSION_RECORD_STREAM
        return ArtifactClassification(
            provider=provider,
            kind=kind,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=90 if subagent else 120,
            reason="record-like JSONL stream",
        )

    if evidence.any_session:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="bundle of session documents",
        )

    if looks_metadataish_list(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented list payload",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized list payload",
    )


def _classify_hermes_sqlite_marker(
    payload: JSONDocument,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Classify a decoded Hermes SQLite marker payload (state.db / verification_evidence.db).

    Shared by `classify_artifact` (checked before the path-only sidecar rule,
    polylogue-zoc3) and `_classify_dict` (the ordinary dict-classification
    fallthrough), so both call sites agree on the marker shape.
    """
    if provider is not Provider.HERMES:
        return None
    artifact_marker = payload.get("polylogue_artifact")
    if artifact_marker == _HERMES_STATE_DB_MARKER:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="Hermes state.db SQLite archive marker",
        )
    if artifact_marker == _HERMES_VERIFICATION_DB_MARKER:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="Hermes verification_evidence.db SQLite archive marker",
        )
    return None


def _classify_dict(
    payload: JSONDocument,
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    # Keep this deferred to avoid the artifact-taxonomy/sources bootstrap
    # cycle described below. List streams import the same pair locally.
    from polylogue.sources.parsers.grok import looks_like_export as looks_like_grok_export
    from polylogue.sources.parsers.hermes_spans import looks_like_atif_payload

    if provider is Provider.CHATGPT:
        from polylogue.sources.parsers.chatgpt_codex_sidecar import looks_like as looks_like_codex_task

        if looks_like_codex_task(payload):
            # bd polylogue-2m2e: codex.json Codex Cloud tasks delivered
            # inside the ChatGPT export. None of the generic session-document
            # heuristics below recognize this shape (no "mapping"/"messages"
            # list), and it also fails looks_metadataish_dict (its "turns"
            # list is not scalarish), so without this branch every task fell
            # through to UNKNOWN/parse_as_session=False and was silently
            # dropped before dispatch.py's chatgpt_codex_task lowering ever
            # ran.
            return ArtifactClassification(
                provider=provider,
                kind=ArtifactKind.SESSION_DOCUMENT,
                parse_as_session=True,
                schema_eligible=True,
                default_priority=100,
                reason="ChatGPT export codex.json Codex Cloud task",
            )

    if looks_like_beads_interaction(payload):
        return ArtifactClassification(
            provider=Provider.UNKNOWN,
            kind=ArtifactKind.UNKNOWN,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="Beads interaction-history artifact, not a session record",
        )

    if looks_like_hook_event(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.HOOK_EVENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=100,
            reason="hook event record",
        )

    if provider is Provider.GROK:
        from polylogue.sources.parsers.grok import looks_like_native_bundle

        if looks_like_native_bundle(payload):
            return ArtifactClassification(
                provider=provider,
                kind=ArtifactKind.SESSION_DOCUMENT,
                parse_as_session=True,
                schema_eligible=True,
                default_priority=120,
                reason="Grok native conversation endpoint bundle",
            )

    if provider is Provider.GROK and looks_like_grok_export(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="Grok account-data export document",
        )

    if provider is Provider.ANTIGRAVITY and _is_antigravity_markdown_export(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="Antigravity language-server Markdown export",
        )

    if (marker_classification := _classify_hermes_sqlite_marker(payload, provider=provider)) is not None:
        return marker_classification

    # Deferred import: `sources.parsers.hermes_spans` sits downstream of
    # `sources/__init__.py` (drive -> dispatch -> decoders -> decoder_zip),
    # which itself imports back from `archive.artifact_taxonomy` -- a
    # module-level import here creates a circular import the moment this
    # package is the first one initialized. See
    # `_archive_reconcile_hermes_session_lifecycle` in `api/archive.py` for
    # the same deferred-import pattern used to break an equivalent cycle.
    if provider is Provider.HERMES and looks_like_atif_payload(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=110,
            reason="Hermes NeMo Relay ATIF trajectory export (schema_version/session_id/steps)",
        )

    if looks_like_session_document(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="session-bearing document",
        )

    if is_subagent_path(source_path) and looks_like_record_entry(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.AGENT_TRANSCRIPT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=90,
            reason="subagent record payload",
        )

    if looks_metadataish_dict(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented document",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized document payload",
    )


def _is_antigravity_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )
