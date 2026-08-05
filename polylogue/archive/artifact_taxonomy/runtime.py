"""Artifact taxonomy classification runtime."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import islice
from pathlib import Path

from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind
from polylogue.archive.artifact_taxonomy.support import (
    is_subagent_path,
    looks_like_beads_interaction,
    looks_like_file_history_snapshot_only_stream,
    looks_like_hook_event,
    looks_like_hook_event_stream,
    looks_like_record_entry,
    looks_like_record_stream,
    looks_like_session_document,
    looks_metadataish_dict,
    looks_metadataish_list,
    normalize_source_path,
    path_only_sidecar_reason,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document

_HERMES_STATE_DB_MARKER = "hermes_state_db"
_HERMES_VERIFICATION_DB_MARKER = "hermes_verification_evidence_db"

# Mirrors ``polylogue.sources.source_walk._SKIP_DIRS``'s "analysis" entry at
# the taxonomy layer (polylogue-omsw / polylogue-9ykn).  The directory-name
# skip in the recursive source walk was meant to keep self-generated agent
# side-output (scratch analysis artifacts an agent writes into its own
# Claude Code project directory, e.g. an index of prior conversation ids)
# out of the archive entirely.  It only guards the recursive walk, though --
# a single-file acquisition route (``Source.path`` pointing directly at one
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
        "(agent side-output, not conversation content; mirrors source_walk _SKIP_DIRS)",
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
    return _classify_artifact_path_strong(source_path, provider=provider)


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
            reason=f"OriginSpec Claude artifact rule: {rule.coverage_role}",
        )
    # polylogue-omsw: generic/ad-hoc acquisition routes (the daemon's shared
    # "inbox" import source backing `polylogue import <path>`, and this
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
        if inner_name.endswith(".md.metadata.json"):
            # Per-artifact brain metadata is a sidecar, never a primary
            # session: fragmenting one file per artifact produced 116
            # single-message "sessions" that were 100% noise (all real
            # conversation content lives in the .pb trajectories the
            # language-server export route now acquires directly --
            # polylogue-eo81, GH #1764). Still accounted for via
            # ``raw_artifacts.artifact_kind`` rather than silently dropped.
            # The one legitimate use of this shape -- a degraded fallback
            # when the language server truly cannot be reached -- is wired
            # explicitly in ``source_parsing._iter_antigravity_brain_metadata_fallback``,
            # which calls ``parse_brain_metadata`` directly and bypasses this
            # path-only classification.
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
    dict_items = [json_document(item) for item in islice(payload, 32)]
    dict_items = [item for item in dict_items if item]
    if not looks_like_file_history_snapshot_only_stream(dict_items):
        return None
    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.FILE_HISTORY_SNAPSHOT,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="Claude Code file-history-snapshot-only stream (no conversational records)",
    )


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
    # ``islice`` bounds consumption directly. ``payload[:32]`` would resolve
    # slice bounds via ``len(payload)`` first, which for a lazy full-corpus
    # record stream (``ReplayableRecordSamples``) forces a complete rescan of
    # the backing file just to take its first 32 items.
    dict_items = [json_document(item) for item in islice(payload, 32)]
    dict_items = [item for item in dict_items if item]

    # Hermes ATOF records look superficially like generic hook events, but
    # carry a producer-defined observer session stream and must be admitted
    # before the generic hook-sidecar exclusion below.
    from polylogue.sources.parsers.hermes_spans import looks_like_atof_payload

    if provider is Provider.HERMES and dict_items and all(looks_like_atof_payload(item) for item in dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_RECORD_STREAM,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=110,
            reason="Hermes NeMo Relay ATOF observer event stream",
        )

    if dict_items and looks_like_hook_event_stream(dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.HOOK_EVENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=100,
            reason="hook event stream",
        )

    if provider is Provider.BEADS and dict_items and all(looks_like_beads_interaction(item) for item in dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_RECORD_STREAM,
            parse_as_session=True,
            schema_eligible=False,
            default_priority=120,
            reason="Beads interaction-history stream",
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
        if dict_items and any(looks_like_record_entry(item) for item in dict_items):
            return ArtifactClassification(
                provider=provider,
                kind=ArtifactKind.UNKNOWN,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Codex record stream contains unsupported session records",
            )

    if dict_items and looks_like_record_stream(dict_items):
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

    if dict_items and any(looks_like_session_document(item) for item in dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="bundle of session documents",
        )

    if looks_metadataish_list(payload):  # type: ignore[arg-type]
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

    if provider is Provider.BEADS and looks_like_beads_interaction(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_RECORD_STREAM,
            parse_as_session=True,
            schema_eligible=False,
            default_priority=120,
            reason="Beads interaction-history record",
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

    if provider is Provider.ANTIGRAVITY and _is_antigravity_brain_metadata(payload, source_path):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=100,
            reason="Antigravity brain artifact metadata with sibling Markdown",
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


def _is_antigravity_brain_metadata(payload: JSONDocument, source_path: str | Path | None) -> bool:
    normalized = normalize_source_path(source_path)
    name = Path(normalized.rsplit(":", 1)[-1]).name.lower() if normalized else ""
    return (
        (not name or name.endswith((".json", ".md.metadata.json")))
        and isinstance(payload.get("artifactType"), str)
        and ("summary" in payload or "updatedAt" in payload)
    )


def _is_antigravity_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )
