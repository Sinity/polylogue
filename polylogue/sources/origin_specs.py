"""Executable source-admission declarations for public archive origins.

Provider adapters retain record-level parsing.  This module owns only the
cross-adapter admission contract: public origin vocabulary, acquisition modes,
detector tightness, registration/fixture evidence, fidelity, and reparse
consequences.  The founding pilots (polylogue-2qx.1.1) deliberately covered a
streaming runtime (Claude Code), a document export (ChatGPT), and a reserved
origin (Grok, which had no confirmed export shape at the time). Grok has since
shipped a real parser (polylogue-611 / #3201) and is admitted here as
``lifecycle="executable"`` -- there is currently no origin genuinely in
``lifecycle="reserved"``. The reserved-lifecycle path (an origin admitted with
no parser/detector binding pending a fixture) remains part of this module's
executable contract; ``tests/unit/sources/test_origin_specs.py`` exercises it
with a synthetic declaration rather than a real Origin member, since every
current public ``Origin`` token is already either executable or
compatibility-only.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from polylogue.core.enums import Origin, Provider
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    validate_registry,
)

OriginLifecycle = Literal["executable", "reserved", "unsupported", "compatibility-only"]
OriginCompletenessMaturity = Literal["accepted", "proposed", "reserved", "unsupported"]
ArtifactParsePolicy = Literal["session", "fact", "raw-only"]


@dataclass(frozen=True, slots=True)
class OriginArtifactRule:
    """Declared admission policy for one native artifact path family.

    These rules deliberately describe paths, not a second parser registry:
    the owning ``OriginSpec`` remains the one admission declaration and
    consumers can distinguish session streams from evidence-bearing sidecars.
    """

    kind: str
    path_pattern: str
    parse_policy: ArtifactParsePolicy
    parser_path: str | None
    coverage_role: str
    fidelity_note: str
    path_suffixes: tuple[str, ...]

    def matches(self, source_path: str) -> bool:
        return re.search(self.path_pattern, source_path.replace("\\", "/")) is not None


#: Root the committed schema packages live under, resolved once here so
#: ``schema_observed_leaf_values`` and its callers do not each hand-roll the
#: ``polylogue/schemas/providers`` path (mirrors ``devtools/schema_parser_diff.py``
#: and ``polylogue.schemas.schema_parser_coverage``'s own ``_SCHEMA_ROOT``).
_SCHEMA_PROVIDERS_ROOT = Path(__file__).resolve().parents[1] / "schemas" / "providers"


@dataclass(frozen=True, slots=True)
class DroppedValueVocabulary:
    """A hand-guessed "known value" set a parser tests membership against.

    ``_SUCCESS_OUTCOMES``-shaped constants (a hardcoded set of strings a
    parser treats as equivalent, e.g. every spelling of "this tool call
    succeeded") are a real, legitimate disposition -- narrowing an open wire
    vocabulary to the values that matter for one boolean fact -- but a bare
    frozenset with no cross-check silently drifts the day a provider starts
    emitting a value nobody guessed (polylogue-2qx: "the declaration should
    be checkable against the schema's observed values rather than being
    another hand-maintained list that drifts"). This declares the field next
    to the committed schema leaf that actually observes it, so
    ``undeclared_schema_values``/``check_dropped_value_vocabularies`` can
    prove -- from the real, versioned schema, not from memory -- whether the
    guessed set still covers everything the corpus has emitted.

    Deliberately NOT a promise that every hardcoded parser vocabulary can be
    expressed this way: a vocabulary sourced from a SQLite column (no JSON
    schema inference runs over SQLite state) or from a schema branch the
    inference engine does not label with a fixed enumerable property name
    (Gemini's chunk-indexed ``functionResponse`` structure) has no schema
    leaf to check against. Those stay bare parser-local constants with a
    comment explaining why; forcing them into this shape would be exactly
    the "relocates the frozenset without making drift detectable" failure
    mode this declaration exists to avoid.
    """

    field: str
    schema_provider: str
    schema_field_path: str
    declared_values: frozenset[str]
    parser_path: str
    reason: str


def _resolve_schema_path(schema: object, segments: Sequence[str]) -> list[dict[str, object]]:
    """Walk a dotted/bracket leaf path (``"messages[].toolCalls[].status"``
    -> ``("messages[]", "toolCalls[]", "status")``) through nested JSON
    Schema ``properties``/``items``, returning every schema node reached.
    """
    nodes: list[dict[str, object]] = [schema] if isinstance(schema, dict) else []
    for segment in segments:
        is_array = segment.endswith("[]")
        name = segment[:-2] if is_array else segment
        next_nodes: list[dict[str, object]] = []
        for node in nodes:
            properties = node.get("properties")
            if not isinstance(properties, dict) or name not in properties:
                continue
            child = properties[name]
            if not isinstance(child, dict):
                continue
            if is_array:
                items = child.get("items")
                if isinstance(items, dict):
                    next_nodes.append(items)
            else:
                next_nodes.append(child)
        nodes = next_nodes
    return nodes


def schema_observed_leaf_values(provider: str, field_path: str, *, schema_root: Path | None = None) -> frozenset[str]:
    """Every ``x-polylogue-values`` entry the committed schema recorded at ``field_path``.

    Reads the same committed, gzip-compressed schema packages
    ``devtools/schema_parser_diff.py`` and
    ``polylogue.schemas.schema_parser_coverage`` read -- real, versioned
    evidence of what the provider has actually emitted, not a guess.
    """
    provider_dir = (schema_root or _SCHEMA_PROVIDERS_ROOT) / provider
    if not provider_dir.exists():
        return frozenset()
    segments = field_path.split(".")
    values: set[str] = set()
    for path in sorted(provider_dir.glob("versions/*/elements/*.schema.json.gz")):
        try:
            document = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
            continue
        for node in _resolve_schema_path(document, segments):
            observed = node.get("x-polylogue-values")
            if isinstance(observed, list):
                values.update(str(item) for item in observed)
    return frozenset(values)


def undeclared_schema_values(vocab: DroppedValueVocabulary, *, schema_root: Path | None = None) -> frozenset[str]:
    """Schema-observed values at ``vocab``'s leaf that its declared set doesn't cover."""
    observed = schema_observed_leaf_values(vocab.schema_provider, vocab.schema_field_path, schema_root=schema_root)
    return observed - vocab.declared_values


#: Concrete registered vocabularies. Each entry documents one hardcoded
#: parser constant that narrows an open wire vocabulary to a guessed set of
#: equivalent values, paired with the schema leaf that actually observes the
#: field so drift is a runnable check
#: (``tests/unit/sources/test_origin_specs.py::test_dropped_value_vocabularies_match_schema_and_parser``),
#: not a hope. See ``DroppedValueVocabulary``'s docstring for which
#: hardcoded vocabularies deliberately do NOT have an entry here yet, and why.
DROPPED_VALUE_VOCABULARIES: tuple[DroppedValueVocabulary, ...] = (
    DroppedValueVocabulary(
        field="gemini-cli tool-result status",
        schema_provider="gemini-cli",
        schema_field_path="messages[].toolCalls[].status",
        declared_values=frozenset({"success", "succeeded", "ok", "completed"}),
        parser_path="polylogue/sources/parsers/local_agent.py:_status_is_error",
        reason=(
            "_status_is_error treats these as non-error outcomes (the error "
            "case is a separate substring-marker heuristic). gemini-cli's own "
            "committed schema has only ever observed 'success' at this leaf, "
            "already covered by the declared set."
        ),
    ),
)


def check_dropped_value_vocabularies(*, schema_root: Path | None = None) -> dict[str, frozenset[str]]:
    """Per-vocabulary schema-observed values none of ``DROPPED_VALUE_VOCABULARIES`` declares.

    Returns only vocabularies with real drift (non-empty); an empty result
    means every registered guessed vocabulary still covers everything the
    committed schema has observed at its leaf.
    """
    drift: dict[str, frozenset[str]] = {}
    for vocab in DROPPED_VALUE_VOCABULARIES:
        missing = undeclared_schema_values(vocab, schema_root=schema_root)
        if missing:
            drift[vocab.field] = missing
    return drift


@dataclass(frozen=True, slots=True)
class OriginCompletenessMode:
    """One material import mode projected into provider-completeness reports."""

    package_ref: str
    capture_mode: str
    provider_wire: Provider | None
    maturity: OriginCompletenessMaturity
    detector_paths: tuple[str, ...]
    raw_model_paths: tuple[str, ...]
    parser_paths: tuple[str, ...]
    normalizer_paths: tuple[str, ...]
    fixture_paths: tuple[str, ...]
    schema_paths: tuple[str, ...]
    docs_paths: tuple[str, ...]
    privacy_paths: tuple[str, ...]
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OriginSpec:
    """One public-origin admission contract, independent of parser internals."""

    origin: Origin
    declaration: DeclarationSpec
    lifecycle: OriginLifecycle
    acquisition_modes: tuple[str, ...]
    provider_wires: tuple[Provider, ...]
    collision_policy: str | None
    detector_tightness: int | None
    parser_paths: tuple[str, ...]
    stream_parser_path: str | None
    assembly_paths: tuple[str, ...]
    fixture_paths: tuple[str, ...]
    coverage_refs: tuple[str, ...]
    fidelity_notes: tuple[str, ...]
    semantic_reparse: str
    #: Short operator-facing description used by user surfaces (for example
    #: CLI ``--origin`` shell completion). Declared here so no surface keeps a
    #: second hand-maintained per-origin description inventory.
    display_description: str
    artifact_rules: tuple[OriginArtifactRule, ...] = ()
    completeness_modes: tuple[OriginCompletenessMode, ...] = ()
    #: ``"module/path.py:ClassName"`` for the ``ProviderAssemblySpec`` this
    #: origin's provider wire binds in ``polylogue.sources.assembly.get_assembly_spec``,
    #: or ``None`` when no sidecar/title/orchestration enrichment hook exists.
    #: This is the one typed admission point for the assembly-layer
    #: orchestration/title extensions that polylogue-2qx.2, polylogue-j2zz, and
    #: polylogue-ih67 build on -- it is validated against the live
    #: ``get_assembly_spec`` registry by :func:`validate_assembly_spec_parity`
    #: rather than replacing that registry with a second one.
    assembly_spec_path: str | None = None


@dataclass(frozen=True, slots=True)
class OriginSpecDiagnostic:
    """Actionable domain diagnostic layered over the shared declaration kernel."""

    code: str
    message: str
    origin: Origin
    owner_path: str
    repair_command: str


#: Provider-wire token values. A public origin name that collides with one of
#: these would let a raw provider-wire spelling masquerade as public origin
#: vocabulary (docs/provider-origin-identity.md's doctrine). No current
#: ``Origin`` member collides with a ``Provider`` member; this check keeps
#: that true as new origins are admitted.
_PROVIDER_TOKEN_VALUES = frozenset(provider.value for provider in Provider)


class OriginSpecRegistry:
    """Register origin declarations and reject incomplete admission edges."""

    def __init__(self) -> None:
        self._kernel = DeclarationRegistry()
        self._by_origin: dict[Origin, OriginSpec] = {}

    def register(self, spec: OriginSpec) -> OriginSpec:
        if spec.origin in self._by_origin:
            raise ValueError(f"duplicate OriginSpec for {spec.origin.value!r}")
        if spec.declaration.public_name in _PROVIDER_TOKEN_VALUES:
            raise ValueError(
                f"{spec.origin.value}: public origin name {spec.declaration.public_name!r} leaks a "
                "Provider-wire token; public origin vocabulary must not collide with provider-wire spelling"
            )
        if spec.declaration.public_name != spec.origin.value:
            raise ValueError(f"{spec.origin.value}: declaration public name must equal the public Origin token")
        if not spec.acquisition_modes:
            raise ValueError(f"{spec.origin.value}: missing acquisition mode")
        if not spec.coverage_refs:
            raise ValueError(f"{spec.origin.value}: missing coverage declaration")
        if not spec.fixture_paths:
            raise ValueError(f"{spec.origin.value}: missing fixture declaration")
        if len(spec.provider_wires) > 1 and not spec.collision_policy:
            raise ValueError(f"{spec.origin.value}: multiple provider wires require an explicit collision policy")
        if spec.lifecycle == "executable":
            if spec.detector_tightness is None:
                raise ValueError(f"{spec.origin.value}: executable origin requires detector tightness")
            if not spec.parser_paths:
                raise ValueError(f"{spec.origin.value}: executable origin requires parser binding")
            for rule in spec.artifact_rules:
                if rule.parse_policy != "raw-only" and rule.parser_path is None:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires a parser binding")
                if not rule.path_suffixes:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires acquisition suffixes")
                if any(not suffix.startswith(".") or suffix != suffix.lower() for suffix in rule.path_suffixes):
                    raise ValueError(
                        f"{spec.origin.value}: {rule.kind} acquisition suffixes must be lowercase dot suffixes"
                    )
        elif spec.parser_paths or spec.stream_parser_path is not None:
            raise ValueError(f"{spec.origin.value}: non-executable origin cannot declare a parser binding")
        self._kernel.register(spec.declaration)
        self._by_origin[spec.origin] = spec
        return spec

    def specs(self) -> tuple[OriginSpec, ...]:
        return tuple(self._by_origin[origin] for origin in sorted(self._by_origin, key=lambda item: item.value))

    def diagnostics(self) -> tuple[OriginSpecDiagnostic, ...]:
        diagnostics = [
            OriginSpecDiagnostic(
                code=item.code,
                message=item.message,
                origin=Origin.from_string(item.declaration_id.removeprefix("origin.")),
                owner_path=item.owner_path,
                repair_command=item.repair_command,
            )
            for item in validate_registry(self._kernel)
        ]
        executable = [spec for spec in self.specs() if spec.lifecycle == "executable"]
        missing_origins = sorted(set(Origin) - set(self._by_origin), key=lambda origin: origin.value)
        diagnostics.extend(
            OriginSpecDiagnostic(
                code="missing_origin_spec",
                message=f"{origin.value}: public Origin has no admission declaration",
                origin=origin,
                owner_path="polylogue/sources/origin_specs.py",
                repair_command="devtools test tests/unit/sources/test_origin_specs.py",
            )
            for origin in missing_origins
        )
        tightness = [spec.detector_tightness for spec in executable]
        if len(tightness) != len(set(tightness)):
            diagnostics.extend(
                OriginSpecDiagnostic(
                    code="ambiguous_detector_tightness",
                    message=f"{spec.origin.value}: detector tightness must be unique among executable OriginSpecs",
                    origin=spec.origin,
                    owner_path=spec.declaration.owner_path,
                    repair_command=spec.declaration.repair_command,
                )
                for spec in executable
            )
        return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


def _declaration(origin: Origin, *, lifecycle: OriginLifecycle, discovery: str) -> DeclarationSpec:
    return DeclarationSpec(
        declaration_id=f"origin.{origin.value}",
        family_id=f"source-origin-admission:{lifecycle}",
        public_name=origin.value,
        owner_path="polylogue/sources/origin_specs.py",
        compatibility=CompatibilityKey(
            identity="public-origin",
            lifecycle=lifecycle,
            authority="source-admission",
            access_result_shape="normalized-session-or-honest-nonexecution",
            durability="source-evidence",
        ),
        producer="polylogue.sources.origin_specs",
        role_gate="archive:read",
        schema_ref="polylogue.core.enums.Origin",
        discovery_text=discovery,
        repair_command="devtools test tests/unit/sources/test_origin_specs.py",
        handlers=(
            HandlerBinding(
                surface="source-admission",
                owner_path="polylogue/sources/origin_specs.py",
                symbol="ORIGIN_SPECS",
                binding_key=origin.value,
            ),
        ),
        outputs=(
            OutputSpec("coverage", "completeness-row", "ProviderPackageCompletenessPayload", "provider-completeness"),
            OutputSpec("docs", "origin-catalog", "Origin", "docs/provider-origin-identity.md"),
        ),
        examples=(ExampleSpec("discover", discovery),),
        completeness_edges=(
            CompletenessEdge(
                producer=f"origin.{origin.value}",
                consumer="polylogue.sources.dispatch",
                kind="detector-parity",
                owner_path="polylogue/sources/dispatch.py",
            ),
        ),
    )


def _claude_code_spec() -> OriginSpec:
    origin = Origin.CLAUDE_CODE_SESSION
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery="Claude Code JSONL and sidecar admission."),
        lifecycle="executable",
        acquisition_modes=("export-jsonl", "live-jsonl", "sidecar", "workflow-orchestration"),
        provider_wires=(Provider.CLAUDE_CODE,),
        collision_policy=None,
        detector_tightness=60,
        parser_paths=("polylogue/sources/parsers/claude/code_parser.py",),
        stream_parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        assembly_paths=("polylogue/sources/dispatch.py:_claude_code_multiway_parse",),
        fixture_paths=(
            "tests/unit/sources/test_parsers_claude_code_artifacts.py",
            "tests/unit/sources/test_assembly_claude_code_history.py",
        ),
        coverage_refs=("provider-package:claude-code-session/export-jsonl@v1",),
        fidelity_notes=(
            "Streaming JSONL retains source ordering; sidecars require separate authority admission.",
            "polylogue-2qx.4 / polylogue-cgfy unread-wire batch: message.stop_reason, toolUseResult's "
            "structuredPatch/originalFile/oldString/newString/replaceAll/userModified (file_edits), the "
            "top-level slug (sessions.display_name), and pr-link (session_refs, kind=pull_request) are now "
            "read and persisted. parentToolUseID was NOT wired to session_links.parent_tool_use_block_id: "
            "measured against 200 recent subagent (agent-*.jsonl) transcripts, the field occurs on the "
            "DISPATCHING parent's own progress/agent_progress records, never on the child session's own "
            "records -- there is no child-side wire evidence to read. tool_result outcome_unknown_reason is "
            "NOT_REPORTED when the Anthropic-protocol segment carries no is_error, and DISTRUSTED for the "
            "background-task start acknowledgement's is_error=false (see _mark_background_task_start).",
            "polylogue-cgfy AC1: disposition of the 'other unread keys of substance' the bead's corpus "
            "enumeration named beyond the structuredPatch/file_edits cluster (already read, see the note "
            "above) and beyond stop_reason/stop_sequence/parentToolUseID/cache_creation/ttftMs/todos/"
            "toolUseResult.sandbox/filenames/numFiles (already read, see code_parser.py's "
            "_message_usage_event_payload and toolUseResult structural-fact projection). READ (this batch): "
            "requestId (1,171 sampled occurrences -- the Anthropic API per-call request id, a real "
            "cross-reference key against provider-side billing/support records) and "
            "thinkingMetadata.maxThinkingTokens (34 occurrences -- the extended-thinking token budget "
            "configured for the turn) both now ride the message_usage session-event payload as "
            "request_id/max_thinking_tokens. MEASURED NEGATIVE: userType is the literal string 'external' "
            "on every sampled record across two independent corpora (2,789 occurrences in the bead's "
            "sample, reconfirmed against a second live ~/.claude/projects corpus this pass) -- a constant, "
            "acquiring it adds nothing, same class as usage.service_tier. DELIBERATELY DROPPED, duplicate "
            "of an already-captured field: sourceToolAssistantUUID (143 occurrences) was verified against "
            "real records to equal that same record's own parentUuid (already captured as "
            "ParsedMessage.parent_message_provider_id) -- a second spelling of the same edge, not new "
            "evidence. DELIBERATELY DROPPED, duplicate of an already-captured tier: hookCount/hookInfos "
            "(22 occurrences) describe which hooks fired on a record; hook execution is already tracked "
            "durably in source.db's raw_hook_events (the 2026-07-22 hook-session-inflation fix's `write_"
            "hook_event`) with the command line and outcome -- adding a second, index-tier, less complete "
            "copy (hookInfos carries only `command`, no outcome) would be a parallel representation of "
            "the same fact, not new information. toolUseID (679 occurrences, the record's own tool-call "
            "correlation id, distinct from parentToolUseID) is already consumed for the real signal it "
            "carries: the progress/agent_progress delegation-edge disposition documented in the module "
            "docstring above _parse_code_records (claude_delegation_progress vs. six transient synthetic-"
            "tick subtypes) -- not a bare unread field.",
            "code_parser.py's _NON_MESSAGE_SIDECAR_RECORD_TYPES (14 sidecar record "
            "types) already carries a per-type disposition with corpus counts "
            "in a comment block (polylogue-pbuh/parser-diff triage, "
            "2026-07-29) -- not converted to a DroppedValueVocabulary "
            "(polylogue-2qx) because it is a record-TYPE inventory, not a "
            "value-equivalence guess, and the committed schema's top-level "
            "``.type`` x-polylogue-values only surfaces the three dominant "
            "message-shape branches (assistant/progress/user); the schema "
            "inference does not label each sidecar record type as a separate "
            "enumerable leaf the way it does a scalar status/outcome field. "
            "Making this checkable needs the schema generator to track "
            "per-branch discriminant values, not a change on this side.",
            "claude/index.py's _GIT_BRANCH_PREFIXES (title-fallback heuristic: "
            "does a bare index-summary string look like a branch name rather "
            "than a title) is also not a DroppedValueVocabulary candidate: it "
            "matches an open-ended organizational naming CONVENTION "
            "(feature/, fix/, chore/, ...), not equivalence classes of one "
            "provider-reported field's observed values -- there is no schema "
            "leaf enumerating branch-prefix conventions to check against, the "
            "same reason filesystem-shaped constants like _SUPPORTED_EXTENSIONS "
            "stay plain hardcoded sets.",
        ),
        semantic_reparse=(
            "reparse Claude Code sessions and re-inventory workflow artifacts when the Claude parser, "
            "orchestration artifact parser, or sidecar assembly fingerprint changes"
        ),
        artifact_rules=(
            OriginArtifactRule(
                kind="tool_result_sidecar",
                # ``hook-*`` files under the same directory are a distinct,
                # already-tracked capture surface (raw hook stdout,
                # polylogue-qqyg / #2781) with their own reliable
                # content-shape detector (``looks_like_hook_event``) --
                # excluded here so this rule doesn't relabel their already
                # correct ``HOOK_EVENT`` classification.
                path_pattern=r"(?:^|/)tool-results/(?!hook-)[^/]+$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="tool_result_overflow",
                fidelity_note=(
                    "Tool-result overflow content persisted verbatim; never independently parsed -- "
                    "sources/live/tool_result_sidecars.py joins it to its owning tool_result block by "
                    "tool_use_id. A tool call's own output can coincidentally reproduce a genuine "
                    "session-document shape (verified live: a real claude.ai export dumped by a prior "
                    "tool call), so content heuristics alone cannot refuse this family; this path rule "
                    "is the only reliable gate (polylogue-omsw)."
                ),
                # Deliberately NOT widened to every extension a tool result
                # can carry (a real corpus scan found genuine ``.txt``/
                # ``.html`` tool-results content too -- see the sibling
                # classification test): ``artifact_suffixes_for_provider``
                # feeds ``live/watcher.py``'s Claude Code ``WatchSource``
                # acquisition suffix filter directly, so adding a suffix here
                # would widen what gets *acquired* archive-wide, not merely
                # what this rule classifies. ``.json`` is the one extension
                # already inside the acquired/watched set today; the
                # ``path_pattern`` match above still refuses any other
                # extension's *content* once a raw row for it exists via
                # some other acquisition route (zip replay, reindex), since
                # ``matches()`` doesn't consult ``path_suffixes`` at all.
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="workflow_run_snapshot",
                path_pattern=r"(?:^|/)workflows/[^/]+\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="run_snapshot",
                fidelity_note="Authoritative mutable workflow run snapshot; every observed revision is retained.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="workflow_journal",
                path_pattern=r"(?:^|/)subagents/workflows/[^/]+/journal\.jsonl$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="journal",
                fidelity_note=(
                    "Append-only workflow journal; content keys and unresolved references remain provider evidence."
                ),
                path_suffixes=(".jsonl",),
            ),
            OriginArtifactRule(
                kind="agent_transcript",
                path_pattern=r"(?:^|/)subagents/(?:[^/]+/)*agent-[^/]+\.(?:jsonl|ndjson)$",
                parse_policy="session",
                parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
                coverage_role="attempt_transcript",
                fidelity_note="Attempt transcript is a session only when provider workflow evidence links it to a run.",
                path_suffixes=(".jsonl", ".ndjson"),
            ),
            OriginArtifactRule(
                kind="agent_sidecar_meta",
                path_pattern=r"(?:^|/)subagents/(?:[^/]+/)*agent-[^/]+\.meta\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="attempt_meta",
                fidelity_note="Agent metadata never fabricates a transcript pair; missing peers are coverage gaps.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="adopt_manifest",
                path_pattern=r"(?:^|/)jobs/[^/]+/adopt\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="adopt_manifest",
                fidelity_note="Recovery manifest preserves resume/adoption evidence without asserting completed work.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="coordinator_session_stream",
                path_pattern=r"(?:^|/)projects/[^/]+/[^/]+\.(?:jsonl|ndjson)$",
                parse_policy="session",
                parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
                coverage_role="coordinator_invocation_stream",
                fidelity_note=(
                    "Coordinator streams retain authored prompts and Workflow tool-use events; "
                    "child topology alone never establishes run membership."
                ),
                path_suffixes=(".jsonl", ".ndjson"),
            ),
            OriginArtifactRule(
                kind="todo_snapshot",
                path_pattern=r"(?:^|/)todos/[^/]+\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/todos.py:parse_claude_todo_artifact",
                coverage_role="plan_snapshot",
                fidelity_note=(
                    "~/.claude/todos/<session-id>[-agent-<agent-id>].json is Claude Code's live plan "
                    "state (polylogue-t0p): a session's own TodoWrite invocations overwrite the same "
                    "path wholesale, never append. Preserved: every task's content/status/priority/id "
                    "and the agent's own list ORDER (a real priority signal, not alphabetical/insertion "
                    "order) for every snapshot the watcher actually observed on disk. Lost: Claude Code "
                    "prunes this directory on its own schedule and each write is a full overwrite, so "
                    "intermediate transitions between two watcher-observed snapshots (e.g. pending -> "
                    "in_progress -> completed within one polling gap) are unrecoverable once overwritten "
                    "or pruned -- there is no in-file timestamp; observation time is acquisition-time, "
                    "not plan-mutation time."
                ),
                path_suffixes=(".json",),
            ),
        ),
        assembly_spec_path="polylogue/sources/assembly_claude_code.py:ClaudeCodeAssemblySpec",
        display_description="Claude Code local sessions (lab: Anthropic)",
    )


def artifact_rule_for_path(provider: Provider, source_path: str) -> OriginArtifactRule | None:
    """Return the owning OriginSpec artifact rule for a native source path."""

    for spec in ORIGIN_SPECS:
        if provider not in spec.provider_wires:
            continue
        for rule in spec.artifact_rules:
            if rule.matches(source_path):
                return rule
    return None


def artifact_suffixes_for_provider(
    provider: Provider,
    *,
    defaults: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Project live-acquisition suffixes from the owning OriginSpec rules.

    Acquisition may add a generic default, but provider artifact families must
    not maintain a second suffix inventory beside OriginSpec.
    """

    suffixes = list(defaults)
    for spec in ORIGIN_SPECS:
        if provider not in spec.provider_wires:
            continue
        for rule in spec.artifact_rules:
            suffixes.extend(rule.path_suffixes)
    return tuple(dict.fromkeys(suffix.lower() for suffix in suffixes))


def _chatgpt_spec() -> OriginSpec:
    origin = Origin.CHATGPT_EXPORT
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin, lifecycle="executable", discovery="ChatGPT document and bundled export admission."
        ),
        lifecycle="executable",
        acquisition_modes=("takeout-json", "bundle", "browser-capture"),
        provider_wires=(Provider.CHATGPT,),
        collision_policy=None,
        detector_tightness=70,
        parser_paths=("polylogue/sources/parsers/chatgpt.py",),
        stream_parser_path=None,
        assembly_paths=("polylogue/sources/dispatch.py:_lower_payload_specs",),
        assembly_spec_path="polylogue/sources/assembly_chatgpt.py:ChatGPTAssemblySpec",
        fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
        coverage_refs=("provider-package:chatgpt-export/takeout-json@v1",),
        fidelity_notes=("Browser capture remains an acquisition mode and is not a new public origin.",),
        semantic_reparse="reparse when ChatGPT document parsing fingerprints change",
        display_description="ChatGPT web exports (lab: OpenAI)",
    )


def _grok_spec() -> OriginSpec:
    return _executable_spec(
        Origin.GROK_EXPORT,
        provider=Provider.GROK,
        tightness=85,
        discovery="Grok account-data export document admission.",
        acquisition_modes=("export-json",),
        parser_paths=("polylogue/sources/parsers/grok.py",),
        fixture_paths=(
            "tests/unit/sources/parsers/test_grok.py",
            "tests/unit/sources/parsers/test_origin_regression_pack.py",
        ),
        assembly_paths=("polylogue/sources/dispatch.py:_lower_grok_export_payload",),
        fidelity_notes=(
            "No native conversation or response id is present in any confirmed export shape; "
            "provider_session_id/provider_message_id are synthesized from file-relative position.",
            "The export drops attachments/images by xAI's own documentation; only text turns are recoverable.",
        ),
        display_description="Grok account-data exports (lab: xAI)",
    )


def _executable_spec(
    origin: Origin,
    *,
    provider: Provider,
    tightness: int,
    discovery: str,
    acquisition_modes: tuple[str, ...],
    parser_paths: tuple[str, ...],
    fixture_paths: tuple[str, ...],
    display_description: str,
    stream_parser_path: str | None = None,
    assembly_paths: tuple[str, ...] = (),
    fidelity_notes: tuple[str, ...] = (),
    assembly_spec_path: str | None = None,
) -> OriginSpec:
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery=discovery),
        lifecycle="executable",
        acquisition_modes=acquisition_modes,
        provider_wires=(provider,),
        collision_policy=None,
        detector_tightness=tightness,
        parser_paths=parser_paths,
        stream_parser_path=stream_parser_path,
        assembly_paths=assembly_paths,
        fixture_paths=fixture_paths,
        coverage_refs=(f"origin:{origin.value}:admitted",),
        fidelity_notes=fidelity_notes,
        semantic_reparse=f"reparse when {origin.value} parser fingerprints change",
        assembly_spec_path=assembly_spec_path,
        display_description=display_description,
    )


def _codex_spec() -> OriginSpec:
    return _executable_spec(
        Origin.CODEX_SESSION,
        provider=Provider.CODEX,
        tightness=50,
        discovery="Codex session JSONL admission plus live Codex SQLite state.",
        acquisition_modes=("session-jsonl", "thread-state-db", "goals-db", "memories-db"),
        parser_paths=(
            "polylogue/sources/parsers/codex.py",
            "polylogue/sources/parsers/codex_state.py",
        ),
        fixture_paths=(
            "tests/unit/sources/test_parsers_codex.py",
            "tests/data/codex_event_stream",
            "tests/unit/sources/parsers/test_codex_state.py",
        ),
        stream_parser_path="polylogue/sources/parsers/codex.py:parse_codex_stream",
        assembly_spec_path="polylogue/sources/assembly_codex.py:CodexAssemblySpec",
        display_description="Codex CLI local sessions (lab: OpenAI)",
        # polylogue-0jf4 acceptance criterion 1: classify each of the five
        # live ~/.codex SQLite databases. This declaration mirrors
        # ``sources/parsers/codex_state.py``'s ``CODEX_STATE_FIDELITY`` tuple
        # verbatim (that module's docstring names this the canonical home for
        # the reasons; it is not imported here so this file stays free of
        # parser-internal imports, matching every other origin declaration).
        fidelity_notes=(
            "state_5.sqlite (thread_state, acquire): threads.title and "
            "thread_spawn_edges have no other evidence source -- no Codex "
            "rollout JSONL session_meta record carries a curated title or a "
            "parent/child spawn relationship at the orchestration level.",
            "goals_1.sqlite (goals, acquire-partial): thread_goals.objective "
            "is stated task intent unavailable anywhere else, but the table "
            "is small and low-churn; the raw snapshot is acquired for "
            "durability with no session_events wiring.",
            "memories_1.sqlite (memories, acquire-partial): stage1_outputs "
            "is Codex-side memory derived from content already ingested from "
            "the JSONL rollout; the raw snapshot is acquired for durability "
            "with no parsed/typed consumption.",
            "logs_2.sqlite (logs, out-of-scope): 627 MB of runtime tracing "
            "(level/target/module_path/file/line), not session evidence -- "
            "acquiring it by default would roughly double this archive's "
            "Codex footprint for no session-reconstruction value.",
            "codex-dev.db (automation, out-of-scope): local CLI automation "
            "scheduling config, not AI session content; empty on every "
            "install observed.",
            "patch_apply_end.changes/.success (acquired, polylogue-cgfy codex "
            "lane): per-file add/update/delete classification plus "
            "unified_diff/move_path, retained verbatim on the patch_apply_end "
            "session_event -- the structural equivalent of Claude Code's "
            "structuredPatch. stdout/stderr/aggregated_output/formatted_output "
            "on the same record are deliberately not re-stored (duplicate the "
            "paired function_call_output tool_result text).",
            "turn_context.personality/.summary/.collaboration_mode (acquired, "
            "polylogue-cgfy codex lane): agent-persona, reasoning-summary "
            "verbosity, and collaboration-mode-name knobs reported on every "
            "turn; collaboration_mode.settings itself duplicates model/effort/"
            "developer_instructions already captured from the top-level "
            "turn_context and is not re-stored.",
            "event_msg.memory_citation (measured negative, polylogue-cgfy "
            "codex lane): observed null on every sampled record across "
            "~3,200 real session files -- a constant, not an unread signal; "
            "not acquired.",
            "response_item(message).internal_chat_message_metadata_passthrough "
            "(to-acquire, deferred): carries only {turn_id}, letting a "
            "message join back to its turn -- messages presently carry no "
            "turn_id at all. Deferred because it needs a metadata channel on "
            "ParsedMessage plumbed through every codex.py message "
            "constructor, not an additive per-event change.",
        ),
    )


def _gemini_cli_spec() -> OriginSpec:
    return _executable_spec(
        Origin.GEMINI_CLI_SESSION,
        provider=Provider.GEMINI_CLI,
        tightness=10,
        discovery="Gemini CLI local-agent document admission.",
        acquisition_modes=("local-agent-document",),
        parser_paths=("polylogue/sources/parsers/local_agent.py",),
        fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
        display_description="Gemini CLI local sessions (lab: Google)",
        fidelity_notes=(
            "local_agent.py's _status_is_error guessed success-outcome set is "
            "registered as a DroppedValueVocabulary (polylogue-2qx) against "
            "the committed schema's messages[].toolCalls[].status leaf -- "
            "see DROPPED_VALUE_VOCABULARIES/check_dropped_value_vocabularies "
            "in this module.",
        ),
    )


def _hermes_spec() -> OriginSpec:
    return _executable_spec(
        Origin.HERMES_SESSION,
        provider=Provider.HERMES,
        tightness=20,
        discovery=(
            "Hermes state database plus NeMo Relay ATIF/ATOF observer and coding-verification-ledger admission."
        ),
        acquisition_modes=("state-db", "atif-spans", "atof-jsonl", "verification-evidence-db"),
        parser_paths=(
            "polylogue/sources/parsers/hermes_state.py",
            "polylogue/sources/parsers/hermes_spans.py",
            "polylogue/sources/parsers/hermes_verification.py",
        ),
        fixture_paths=(
            "tests/unit/sources/test_parsers_local_agent.py",
            "tests/unit/sources/parsers/test_hermes_spans.py",
            "tests/unit/sources/parsers/test_hermes_verification.py",
            "tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json",
            "tests/fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl",
        ),
        stream_parser_path="polylogue/sources/parsers/hermes_spans.py:parse_atof_stream",
        display_description="Hermes agent sessions",
        fidelity_notes=(
            "hermes_state.py's _COMPACTION_END_REASONS ({'compression', "
            "'compaction'}) and _REQUIRED_SESSION_COLUMNS are not "
            "DroppedValueVocabulary candidates (polylogue-2qx): both are read "
            "from the live Hermes SQLite state database's own columns, not "
            "from a JSON wire payload the schema-inference pipeline observes "
            "-- no committed schema package exists to check them against. A "
            "schema-backed check here would need SQLite-column value "
            "sampling, a different mechanism than the JSON x-polylogue-values "
            "this bead's declaration reads, not merely a different provider "
            "argument to the same function.",
        ),
    )


def _antigravity_spec() -> OriginSpec:
    return _executable_spec(
        Origin.ANTIGRAVITY_SESSION,
        provider=Provider.ANTIGRAVITY,
        tightness=30,
        discovery="Antigravity language-server export admission.",
        acquisition_modes=("language-server-export",),
        parser_paths=("polylogue/sources/parsers/antigravity.py",),
        fixture_paths=(
            "tests/unit/sources/test_antigravity_language_server.py",
            "tests/unit/sources/parsers/test_antigravity.py",
        ),
        display_description="Antigravity brain artifacts",
    )


def _beads_spec() -> OriginSpec:
    return _executable_spec(
        Origin.BEADS_ISSUE,
        provider=Provider.BEADS,
        tightness=40,
        discovery="Beads issue export admission.",
        acquisition_modes=("issue-jsonl",),
        parser_paths=("polylogue/sources/parsers/beads.py",),
        fixture_paths=("tests/unit/sources/parsers/test_beads.py",),
        # dispatch.py:parse_stream_payload routes Provider.BEADS to the
        # plain ``beads.parse`` entry point -- there is no dedicated
        # ``parse_beads_stream`` function; this declaration must name the
        # function dispatch actually calls.
        stream_parser_path="polylogue/sources/parsers/beads.py:parse",
        display_description="Beads issue exports (non-chat work artifacts)",
    )


def _claude_ai_spec() -> OriginSpec:
    return _executable_spec(
        Origin.CLAUDE_AI_EXPORT,
        provider=Provider.CLAUDE_AI,
        tightness=80,
        discovery="Claude AI document export admission.",
        acquisition_modes=("export-json",),
        parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
        fixture_paths=("tests/unit/sources/test_parsers_claude_ai_catalog.py",),
        # bd polylogue-4zqh3: sole-copy attachment-byte recovery sidecar.
        assembly_spec_path="polylogue/sources/assembly_claude_ai.py:ClaudeAIAssemblySpec",
        display_description="Claude web exports (lab: Anthropic)",
    )


def _claude_design_spec() -> OriginSpec:
    return _executable_spec(
        Origin.CLAUDE_DESIGN_SESSION,
        provider=Provider.CLAUDE_DESIGN,
        tightness=82,
        discovery="Claude Design chat document admission (design_chats/*.json in a Claude AI GDPR export).",
        acquisition_modes=("export-json",),
        parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
        fixture_paths=("tests/unit/sources/test_parsers_claude_design.py",),
        display_description="Claude Design agentic sessions (lab: Anthropic)",
        fidelity_notes=(
            "bd polylogue-tbun: measured over the 11 design_chats in the 2026-07-30 export. Claude Design is "
            "NOT claude.ai with a flag -- distinct camelCase wire shape (contentBlocks/authorAccountUuid/"
            "turnChanges), message.content is a dict not a list, admitted as its own Origin/Provider rather "
            "than folded into claude-ai-export.",
            "turnChanges (created/edited/deleted/moved file lists per turn) is stored as a "
            "claude_design_turn_changes session_event rather than a new construct type or table -- it is "
            "turn-scoped structured evidence, exactly what session_events + payload_json already models "
            "(the same mechanism ai_parser.py already uses for claude_ai_conversation_summary/"
            "claude_ai_web_tool_evidence); no schema addition needed.",
            "user_interjection (a user message nested inside an assistant turn) is NOT flattened into an "
            "ordinary same-position user message: the assistant turn's contentBlocks are split at the "
            "interjection boundary into separate ParsedMessage segments sharing incrementing positions, so "
            "the interjection lands as a real role=user message physically between the two half-turns, "
            "preserving interruption ordering instead of destroying it.",
            "attachment_kind gains 'skill' and 'folder' (attachment_kind is an open string field, not a "
            "CHECK-constrained column -- see ParsedAttachment.attachment_kind docstring -- so no schema "
            "change is needed for the new values).",
            "Only 11 real chats have been observed; the parser is deliberately strict (unrecognized "
            "contentBlocks types are logged and skipped, never guessed) rather than modeling a shape that "
            "is still moving. No committed schema-harvest package exists yet -- see the 'proposed' "
            "completeness maturity below.",
        ),
    )


def _aistudio_drive_spec() -> OriginSpec:
    origin = Origin.AISTUDIO_DRIVE
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery="AI Studio and Drive export admission."),
        lifecycle="executable",
        acquisition_modes=("drive-like-export",),
        provider_wires=(Provider.GEMINI, Provider.DRIVE),
        collision_policy="Gemini and Drive wire families intentionally normalize to one public AI Studio origin.",
        detector_tightness=90,
        parser_paths=("polylogue/sources/parsers/drive.py",),
        stream_parser_path=None,
        assembly_paths=("polylogue/sources/dispatch.py:_lower_payload_specs",),
        fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
        coverage_refs=("origin:aistudio-drive:admitted",),
        fidelity_notes=(
            "Provider reverse mapping remains intentionally non-injective.",
            "runSettings (temperature/topP/topK/maxOutputTokens/thinkingLevel/safetySettings/enable* flags) "
            "is read and stored verbatim as sessions.run_settings_json (polylogue-2qx.4 / polylogue-cgfy); "
            "deliberately not decomposed into columns so the schema stays uncoupled from one provider's knobs.",
            "chunkedPrompt.pendingInputs (unsent textbox drafts) is read and stored verbatim as "
            "sessions.pending_drafts_json (polylogue-o4j2), deliberately as a session-row field rather than a "
            "session_event: a draft is mutable current UI state, and session_events participate in "
            "session_revision_projection's append-only comparison axes (polylogue-aggz Invariant 1).",
            "drive_support_blocks.py's _SUCCESS_OUTCOMES ({'ok', 'success', "
            "'succeeded', 'completed', 'outcome_ok'}) is not (yet) a "
            "DroppedValueVocabulary (polylogue-2qx): Gemini's own committed "
            "schema never labels a fixed 'outcome'/'status' property at a "
            "stable leaf path the way gemini-cli's toolCalls[].status does -- "
            "Gemini's functionResponse content lives inside chunk-indexed, "
            "dynamically-keyed structures the schema inference does not "
            "collapse into one enumerable property. Contrast gemini-cli's "
            "local_agent.py:_status_is_error, which shares the exact same "
            "guessed value set and IS registered "
            "(DROPPED_VALUE_VOCABULARIES in origin_specs.py) because its "
            "schema leaf is stable.",
        ),
        semantic_reparse="reparse when Drive parser fingerprints change",
        assembly_spec_path="polylogue/sources/assembly_gemini.py:GeminiAssemblySpec",
        display_description="Google AI Studio / Drive exports (lab: Google)",
    )


def _unknown_spec() -> OriginSpec:
    origin = Origin.UNKNOWN_EXPORT
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin, lifecycle="compatibility-only", discovery="Unknown fallback origin admission."
        ),
        lifecycle="compatibility-only",
        acquisition_modes=("fallback", "browser-capture"),
        provider_wires=(Provider.UNKNOWN,),
        collision_policy=None,
        detector_tightness=None,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
        fixture_paths=("tests/unit/sources/test_origin_specs.py",),
        coverage_refs=("origin:unknown-export:fallback",),
        fidelity_notes=(
            "Fallback is explicit; browser capture resolves a provider-specific origin before archive materialization.",
        ),
        semantic_reparse="no direct parser; retain unknown evidence until a concrete source adapter is admitted",
        display_description="Unrecognized fallback exports",
    )


def _completeness_mode(
    package_ref: str,
    capture_mode: str,
    provider_wire: Provider | None,
    maturity: OriginCompletenessMaturity,
    *,
    detector_paths: tuple[str, ...],
    raw_model_paths: tuple[str, ...],
    parser_paths: tuple[str, ...],
    normalizer_paths: tuple[str, ...],
    fixture_paths: tuple[str, ...],
    schema_paths: tuple[str, ...],
    docs_paths: tuple[str, ...],
    privacy_paths: tuple[str, ...] = ("docs/provider-origin-identity.md",),
    caveats: tuple[str, ...] = (),
) -> OriginCompletenessMode:
    return OriginCompletenessMode(
        package_ref=package_ref,
        capture_mode=capture_mode,
        provider_wire=provider_wire,
        maturity=maturity,
        detector_paths=detector_paths,
        raw_model_paths=raw_model_paths,
        parser_paths=parser_paths,
        normalizer_paths=normalizer_paths,
        fixture_paths=fixture_paths,
        schema_paths=schema_paths,
        docs_paths=docs_paths,
        privacy_paths=privacy_paths,
        caveats=caveats,
    )


_ORIGIN_COMPLETENESS_MODES: dict[Origin, tuple[OriginCompletenessMode, ...]] = {
    Origin.CLAUDE_CODE_SESSION: (
        _completeness_mode(
            "provider-package:claude-code-session/export-jsonl@v1",
            "export-jsonl",
            Provider.CLAUDE_CODE,
            "accepted",
            detector_paths=("polylogue/sources/parsers/claude/code_detection.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/claude_code_record.py",),
            parser_paths=("polylogue/sources/parsers/claude/code_parser.py",),
            normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
            fixture_paths=(
                "tests/unit/sources/test_parsers_claude_code_artifacts.py",
                "tests/unit/sources/test_assembly_claude_code_history.py",
            ),
            schema_paths=("polylogue/schemas/providers/claude-code/catalog.json",),
            docs_paths=("docs/providers/claude-code.md",),
        ),
    ),
    Origin.CODEX_SESSION: (
        _completeness_mode(
            "provider-package:codex-session/session-jsonl@v1",
            "session-jsonl",
            Provider.CODEX,
            "accepted",
            detector_paths=("polylogue/sources/parsers/codex.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/codex.py",),
            parser_paths=("polylogue/sources/parsers/codex.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_codex.py", "tests/data/codex_event_stream"),
            schema_paths=("polylogue/schemas/providers/codex/catalog.json",),
            docs_paths=("docs/providers/openai-codex.md",),
        ),
    ),
    Origin.GEMINI_CLI_SESSION: (
        _completeness_mode(
            "provider-package:gemini-cli-session/local-agent-document@v1",
            "local-agent-document",
            Provider.GEMINI_CLI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/local_agent.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/local_agent.py",),
            parser_paths=("polylogue/sources/parsers/local_agent.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
            schema_paths=("polylogue/schemas/providers/gemini-cli/catalog.json",),
            docs_paths=("docs/providers/README.md",),
        ),
    ),
    Origin.HERMES_SESSION: (
        _completeness_mode(
            "provider-package:hermes-session/state-db@v1",
            "state-db",
            Provider.HERMES,
            "accepted",
            detector_paths=(
                "polylogue/sources/parsers/hermes_state.py",
                "polylogue/sources/dispatch.py",
                "polylogue/sources/source_parsing.py",
            ),
            raw_model_paths=("polylogue/sources/parsers/hermes_state.py",),
            parser_paths=("polylogue/sources/parsers/hermes_state.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
            schema_paths=("polylogue/schemas/providers/hermes/state_db_v16.contract.json",),
            docs_paths=("docs/providers/README.md", "docs/onboarding.md"),
        ),
    ),
    Origin.ANTIGRAVITY_SESSION: (
        _completeness_mode(
            "provider-package:antigravity-session/language-server-export@v1",
            "language-server-export",
            Provider.ANTIGRAVITY,
            "accepted",
            detector_paths=("polylogue/sources/parsers/antigravity.py", "polylogue/sources/source_parsing.py"),
            raw_model_paths=("polylogue/sources/parsers/antigravity.py",),
            parser_paths=("polylogue/sources/parsers/antigravity.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=(
                "tests/unit/sources/test_antigravity_language_server.py",
                "tests/unit/sources/parsers/test_antigravity.py",
            ),
            schema_paths=("polylogue/schemas/providers/antigravity/catalog.json",),
            docs_paths=("docs/architecture.md",),
        ),
    ),
    Origin.BEADS_ISSUE: (
        _completeness_mode(
            "provider-package:beads-issue/issue-jsonl@v1",
            "issue-jsonl",
            Provider.BEADS,
            # Kept "proposed" (not "accepted") deliberately: the detector and
            # parser are real and admitted (OriginSpec.lifecycle="executable"
            # above), but no schema-discovery harvesting pass has run against
            # a real Beads repository export sample -- the wire shape is
            # reconstructed from independent secondary sources (see
            # polylogue/sources/parsers/beads.py), not a harvested provider-
            # package catalog. "accepted" would require schema_package evidence
            # this origin does not yet have, and would fail `devtools provider-
            # completeness --check`.
            "proposed",
            detector_paths=("polylogue/sources/parsers/beads.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/beads.py",),
            parser_paths=("polylogue/sources/parsers/beads.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/parsers/test_beads.py",),
            schema_paths=(),
            docs_paths=("docs/architecture.md",),
            caveats=(
                "No schema-discovery harvesting pass has run against a real Beads repository export; "
                "the wire shape is reconstructed from independent secondary sources (see "
                "polylogue/sources/parsers/beads.py), not a harvested provider-package catalog. "
                "Promote to accepted once real-sample schema evidence exists.",
            ),
        ),
    ),
    Origin.GROK_EXPORT: (
        _completeness_mode(
            "provider-package:grok-export/export-json@v1",
            "export-json",
            Provider.GROK,
            # Kept "proposed" (not "accepted") deliberately: the detector and
            # parser are real and admitted (OriginSpec.lifecycle="executable"
            # above), but no schema-discovery harvesting pass has run against
            # a real xAI export sample -- the wire shape is reconstructed from
            # independent secondary sources (see polylogue/sources/parsers/
            # grok.py), not a harvested provider-package catalog. "accepted"
            # would require schema_package evidence this origin does not yet
            # have, and would fail `devtools provider-completeness --check`.
            "proposed",
            detector_paths=("polylogue/sources/parsers/grok.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/grok.py",),
            parser_paths=("polylogue/sources/parsers/grok.py",),
            normalizer_paths=("polylogue/sources/parsers/grok.py",),
            fixture_paths=(
                "tests/unit/sources/parsers/test_grok.py",
                "tests/unit/sources/parsers/test_origin_regression_pack.py",
            ),
            schema_paths=(),
            docs_paths=("docs/provider-origin-identity.md", "docs/architecture.md"),
            caveats=(
                "No schema-discovery harvesting pass has run against a real xAI export; the wire shape is "
                "reconstructed from independent secondary sources (see polylogue/sources/parsers/grok.py), "
                "not a harvested provider-package catalog. Promote to accepted once real-sample schema "
                "evidence exists.",
            ),
        ),
    ),
    Origin.CHATGPT_EXPORT: (
        _completeness_mode(
            "provider-package:chatgpt-export/takeout-json@v1",
            "takeout-json",
            Provider.CHATGPT,
            "accepted",
            detector_paths=("polylogue/sources/parsers/chatgpt.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/chatgpt.py",),
            parser_paths=("polylogue/sources/parsers/chatgpt.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
            schema_paths=("polylogue/schemas/providers/chatgpt/catalog.json",),
            docs_paths=("docs/providers/chatgpt.md",),
        ),
    ),
    Origin.CLAUDE_AI_EXPORT: (
        _completeness_mode(
            "provider-package:claude-ai-export/export-json@v1",
            "export-json",
            Provider.CLAUDE_AI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/claude/ai_parser.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/claude_ai.py",),
            parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
            fixture_paths=("tests/unit/sources/test_parsers_claude_ai_catalog.py",),
            schema_paths=("polylogue/schemas/providers/claude-ai/catalog.json",),
            docs_paths=("docs/providers/claude-ai.md",),
        ),
    ),
    Origin.CLAUDE_DESIGN_SESSION: (
        _completeness_mode(
            "provider-package:claude-design-session/export-json@v1",
            "export-json",
            Provider.CLAUDE_DESIGN,
            # Kept "proposed" (not "accepted") deliberately: the detector and parser are real and admitted
            # (OriginSpec.lifecycle="executable" above), but no schema-discovery harvesting pass has run --
            # the wire shape is reconstructed from 11 real design_chats sampled from one 2026-07-30 export,
            # not a harvested provider-package catalog. "accepted" would require schema_package evidence
            # this origin does not yet have, and would fail `devtools provider-completeness --check`.
            "proposed",
            detector_paths=("polylogue/sources/parsers/claude/ai_parser.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
            fixture_paths=("tests/unit/sources/test_parsers_claude_design.py",),
            schema_paths=(),
            docs_paths=("docs/provider-origin-identity.md",),
            caveats=(
                "No schema-discovery harvesting pass has run against a broad Claude Design corpus; the wire "
                "shape is reconstructed from 11 real design_chats in one 2026-07-30 GDPR export sample. "
                "Promote to accepted once a harvested provider-package catalog exists.",
            ),
        ),
    ),
    Origin.AISTUDIO_DRIVE: (
        _completeness_mode(
            "provider-package:aistudio-drive/drive-export@v1",
            "drive-like-export",
            Provider.GEMINI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/drive.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/gemini_message.py",),
            parser_paths=("polylogue/sources/parsers/drive.py",),
            normalizer_paths=("polylogue/sources/parsers/drive_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
            schema_paths=("polylogue/schemas/providers/gemini/catalog.json",),
            docs_paths=("docs/providers/gemini.md",),
        ),
    ),
    Origin.UNKNOWN_EXPORT: (
        _completeness_mode(
            "provider-package:browser-capture/live-receiver@v1",
            "browser-capture-live-receiver",
            None,
            "accepted",
            detector_paths=("polylogue/sources/parsers/browser_capture.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/browser_capture.py",),
            parser_paths=("polylogue/sources/parsers/browser_capture.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=(
                "tests/unit/sources/test_browser_capture.py",
                "tests/data/witnesses/browser-capture-sequence.json",
            ),
            schema_paths=("polylogue/schemas/providers/browser-capture/catalog.json",),
            docs_paths=("docs/browser-capture.md",),
            privacy_paths=("docs/provider-origin-identity.md", "docs/daemon-threat-model.md"),
            caveats=(
                "Browser capture maps captured page sessions onto provider-specific origins at parse time.",
                "The schema package is generated from the first-party pydantic wire contract "
                "(polylogue.browser_capture.models.BrowserCaptureEnvelope) that the receiver enforces "
                "at ingestion, not from a harvested real-sample corpus -- unlike third-party export "
                "formats, browser-capture is a Polylogue-controlled envelope, so the validation "
                "schema itself is the authoritative shape, not an inference over observed samples.",
            ),
        ),
    ),
}


def _with_completeness_modes(spec: OriginSpec) -> OriginSpec:
    return replace(spec, completeness_modes=_ORIGIN_COMPLETENESS_MODES[spec.origin])


ORIGIN_SPEC_REGISTRY = OriginSpecRegistry()
for _spec in (
    _claude_code_spec(),
    _codex_spec(),
    _gemini_cli_spec(),
    _hermes_spec(),
    _antigravity_spec(),
    _beads_spec(),
    _grok_spec(),
    _chatgpt_spec(),
    _claude_ai_spec(),
    _claude_design_spec(),
    _aistudio_drive_spec(),
    _unknown_spec(),
):
    ORIGIN_SPEC_REGISTRY.register(_with_completeness_modes(_spec))
ORIGIN_SPECS = ORIGIN_SPEC_REGISTRY.specs()


def origin_specs() -> tuple[OriginSpec, ...]:
    """Return the stable public-origin admission projection."""

    return ORIGIN_SPECS


def validate_dispatch_precedence(provider_order: tuple[Provider, ...]) -> tuple[OriginSpecDiagnostic, ...]:
    """Check that the legacy detector branch order honors declared tightness."""

    positions = {provider: index for index, provider in enumerate(provider_order)}
    diagnostics: list[OriginSpecDiagnostic] = []
    executable = [spec for spec in ORIGIN_SPECS if spec.lifecycle == "executable"]
    for spec in executable:
        if not spec.provider_wires or spec.provider_wires[0] not in positions:
            diagnostics.append(
                OriginSpecDiagnostic(
                    code="missing_dispatch_provider",
                    message=f"{spec.origin.value}: provider is absent from dispatch precedence",
                    origin=spec.origin,
                    owner_path=spec.declaration.owner_path,
                    repair_command=spec.declaration.repair_command,
                )
            )
    ordered = sorted(executable, key=lambda spec: spec.detector_tightness or 0)
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_position = positions.get(left.provider_wires[0])
        right_position = positions.get(right.provider_wires[0])
        if left_position is not None and right_position is not None and left_position > right_position:
            diagnostics.append(
                OriginSpecDiagnostic(
                    code="dispatch_tightness_mismatch",
                    message=(
                        f"{left.origin.value}: declared tighter than {right.origin.value} but appears later in dispatch"
                    ),
                    origin=left.origin,
                    owner_path="polylogue/sources/dispatch.py",
                    repair_command="devtools test tests/unit/sources/test_origin_specs.py",
                )
            )
    return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


def validate_stream_parser_parity(stream_record_providers: frozenset[Provider]) -> tuple[OriginSpecDiagnostic, ...]:
    """Check that declared ``stream_parser_path`` presence matches dispatch's stream-record providers.

    ``sources/dispatch.py:STREAM_RECORD_PROVIDERS`` is the current production
    set of providers whose JSONL sources are parsed via a streaming record
    path rather than a fully-materialized payload. An executable OriginSpec
    whose provider wire is in that set but declares no ``stream_parser_path``
    (or vice versa) is a stream/parser binding conflict: the declaration would
    silently under- or over-promise streaming support relative to the actual
    dispatch behavior it is supposed to describe.
    """

    diagnostics: list[OriginSpecDiagnostic] = []
    for spec in ORIGIN_SPECS:
        if spec.lifecycle != "executable":
            continue
        expects_stream = any(provider in stream_record_providers for provider in spec.provider_wires)
        declares_stream = spec.stream_parser_path is not None
        if expects_stream == declares_stream:
            continue
        diagnostics.append(
            OriginSpecDiagnostic(
                code="stream_parser_parity_mismatch",
                message=(
                    f"{spec.origin.value}: dispatch expects a stream parser binding ({expects_stream}) but "
                    f"OriginSpec declares stream_parser_path={spec.stream_parser_path!r}"
                ),
                origin=spec.origin,
                owner_path=spec.declaration.owner_path,
                repair_command=spec.declaration.repair_command,
            )
        )
    return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


def validate_assembly_spec_parity(
    assembly_spec_resolver: Callable[[Provider], object | None],
) -> tuple[OriginSpecDiagnostic, ...]:
    """Check declared ``assembly_spec_path`` presence against the live assembly registry.

    ``polylogue.sources.assembly.get_assembly_spec`` is the current production
    per-provider sidecar/title/orchestration enrichment factory consumed by
    ingest (``source_walk.py``, ``emitter.py``, ``ingest_worker.py``). This is
    the one typed admission point polylogue-2qx.2, polylogue-j2zz, and
    polylogue-ih67 build their assembly/orchestration/title/action extensions
    on: a declared ``assembly_spec_path`` that must agree with whether the live
    registry actually returns an assembly spec for the origin's provider
    wire(s), rather than a second parallel provider-to-assembly-spec registry
    living in this module.
    """

    diagnostics: list[OriginSpecDiagnostic] = []
    for spec in ORIGIN_SPECS:
        if spec.lifecycle != "executable":
            continue
        declares_assembly = spec.assembly_spec_path is not None
        has_live_assembly = any(assembly_spec_resolver(provider) is not None for provider in spec.provider_wires)
        if declares_assembly == has_live_assembly:
            continue
        diagnostics.append(
            OriginSpecDiagnostic(
                code="assembly_spec_parity_mismatch",
                message=(
                    f"{spec.origin.value}: live assembly registry returns a spec "
                    f"({has_live_assembly}) but OriginSpec declares "
                    f"assembly_spec_path={spec.assembly_spec_path!r}"
                ),
                origin=spec.origin,
                owner_path=spec.declaration.owner_path,
                repair_command=spec.declaration.repair_command,
            )
        )
    return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


__all__ = [
    "DROPPED_VALUE_VOCABULARIES",
    "ORIGIN_SPECS",
    "ORIGIN_SPEC_REGISTRY",
    "ArtifactParsePolicy",
    "DroppedValueVocabulary",
    "OriginArtifactRule",
    "OriginLifecycle",
    "OriginSpec",
    "OriginSpecDiagnostic",
    "OriginSpecRegistry",
    "check_dropped_value_vocabularies",
    "origin_specs",
    "artifact_rule_for_path",
    "artifact_suffixes_for_provider",
    "schema_observed_leaf_values",
    "undeclared_schema_values",
    "validate_assembly_spec_parity",
    "validate_dispatch_precedence",
    "validate_stream_parser_parity",
]
