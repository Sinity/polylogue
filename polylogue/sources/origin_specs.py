"""Executable source-admission declarations for public archive origins.

Provider adapters retain record-level parsing.  This module owns only the
cross-adapter admission contract: public origin vocabulary, acquisition modes,
detector tightness, registration/fixture evidence, fidelity, and reparse
consequences.  The founding pilots (polylogue-2qx.1.1) deliberately covered a
streaming runtime (Claude Code), a document export (ChatGPT), and a reserved
origin (Grok, which had no confirmed export shape at the time). Grok has since
shipped a real parser (polylogue-611 / #3201) and is admitted here as
``lifecycle="executable"``. Beads is retained as a reserved origin because its
durable vocabulary predates the session-admission decision. It has no parser or
detector binding; ``tests/unit/sources/test_origin_specs.py`` is its synthetic
fixture declaration until a real wire format is intentionally adopted.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from polylogue.core.enums import Origin, Provider, ToolResultUnknownReason
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
from polylogue.sources.detection import (
    CompiledDetectorRegistry,
    DetectionMode,
    DetectorBinding,
    compile_detector_registry,
)

logger = logging.getLogger(__name__)

OriginLifecycle = Literal["executable", "reserved", "unsupported", "compatibility-only"]
OriginCompletenessMaturity = Literal["accepted", "proposed", "reserved", "unsupported"]
ArtifactParsePolicy = Literal["session", "fact", "raw-only"]
SourceClass = Literal["session", "non_session", "unsupported"]
TopologyCapabilityState = Literal["carried", "positive-derived", "structurally-absent", "unknown"]
TopologyCapabilityDimension = Literal[
    "message_parent",
    "message_branch_state",
    "session_parent_target",
    "inheritance_branch_point",
    "parent_dispatch",
]
#: ``whole-snapshot``: one logical session is written as complete files
#: that share no byte prefix, so successive observations are competing
#: snapshots rather than continuations and byte revision authority has
#: nothing to be a proof about. Such an origin is governed by membership
#: authority from its first observation.
SourceFrontierKind = Literal["exact-prefix", "claude-header-body", "whole-snapshot"]
DatabaseMemberDisposition = Literal["acquire", "acquire-partial", "out-of-scope"]
DatabaseTableDisposition = Literal[
    "retained-and-consumed",
    "retained-for-later-consumption",
    "deliberately-excluded",
]

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_LOWERING_FINGERPRINT_PATHS: tuple[str, ...] = (
    "polylogue/sources/dispatch.py",
    "polylogue/sources/detection.py",
    "polylogue/sources/emitter.py",
    "polylogue/pipeline/ids.py",
    "polylogue/storage/sqlite/archive_tiers/write.py",
    "polylogue/archive/session_revision_membership.py",
)
_REPLAY_ROUTING_FINGERPRINT_PATHS: tuple[str, ...] = ("polylogue/sources/revision_backfill.py",)
# Logging is deliberately available to parser code for diagnostics, but its
# implementation and configuration do not affect normalized parser output.
_PARSER_DIAGNOSTIC_FINGERPRINT_PATHS: frozenset[str] = frozenset({"polylogue/logging.py"})
# ``version.py`` imports this file only in an installed/package-shaped tree.
# Hatch and Nix generate it with build-specific values, so it is provenance,
# not parser/lowering/materializer/replay computation.  Keep ``version.py`` in
# the closure: its implementation remains a semantic dependency when it is
# used by one of those routes.
_GENERATED_PROVENANCE_FINGERPRINT_PATHS: frozenset[str] = frozenset({"polylogue/_build_info.py"})


class _ProjectionFingerprintStripper(ast.NodeTransformer):
    """Remove declaration-only projection syntax from semantic source stamps.

    ``OriginSpec`` is imported by executable source modules, so excluding the
    module from every transitive fingerprint would also hide meaningful
    admission/runtime changes.  Instead, normalize only the public projection
    keyword arguments and projection helper functions; parser, detector,
    replay, and materializer code remains in the source-AST closure.
    """

    _PROJECTION_KEYWORDS = frozenset({"display_description", "public_filter"})
    _PROJECTION_FUNCTIONS = frozenset({"public_origin_tokens", "public_origin_meanings", "public_origin_descriptions"})

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = cast(ast.Call, self.generic_visit(node))
        node.keywords = [keyword for keyword in node.keywords if keyword.arg not in self._PROJECTION_KEYWORDS]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node = cast(ast.ClassDef, self.generic_visit(node))
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
        if node.name in self._PROJECTION_FUNCTIONS:
            return None
        return cast(ast.FunctionDef, self.generic_visit(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef | None:
        if node.name in self._PROJECTION_FUNCTIONS:
            return None
        return cast(ast.AsyncFunctionDef, self.generic_visit(node))


_MATERIALIZER_FINGERPRINT_PATHS: tuple[str, ...] = (
    "polylogue/storage/raw_convergence.py",
    "polylogue/storage/derived/session/rebuild.py",
    "polylogue/storage/derived/session/threads.py",
    "polylogue/storage/derived/session/profiles.py",
    "polylogue/storage/derived/session/latency_profiles.py",
    "polylogue/storage/derived/session/timeline_rows.py",
    "polylogue/storage/derived/session/aggregates.py",
    "polylogue/storage/runtime/store_constants.py",
)


def _without_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


class _DocstringStripper(ast.NodeTransformer):
    """Apply the classifier-fingerprint AST convention to whole modules."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = cast(ast.Module, self.generic_visit(node))
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node = cast(ast.ClassDef, self.generic_visit(node))
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node = cast(ast.FunctionDef, self.generic_visit(node))
        node.body = _without_leading_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node = cast(ast.AsyncFunctionDef, self.generic_visit(node))
        node.body = _without_leading_docstring(node.body)
        return node


class _SchemaDdlFingerprintStripper(ast.NodeTransformer):
    """Normalize SQL source literals before hashing the transitive closure.

    Schema DDL is spread across the archive-tier declaration and its imported
    FTS/runtime-index fragments. Their SQL literals contain maintenance
    comments and formatting, which are already absent from the SQLite
    semantic manifest. Keep the source closure honest for real SQL changes
    while making those representational edits agree with the manifest used by
    derived identity.

    The ``*_DDL`` convention covers archive declarations and trigger lists.
    These explicit SQL names are DDL fragments too, even though their names
    also support non-DDL query constants in the same modules.
    """

    _in_ddl = False
    _DDL_SQL_NAMES = frozenset(
        {
            "FTS_MESSAGES_TABLE_SQL",
            "FTS_MESSAGES_IDENTITY_TABLE_SQL",
            "_FTS_BULK_GUARD_NOT_SET",
            "_TRIGRAM_BULK_GUARD_NOT_SET",
            "_RUNTIME_INDEX_SQL",
            "_DEFERRED_SECONDARY_INDEX_SQL",
        }
    )

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if any(
            isinstance(target, ast.Name) and (target.id.endswith("_DDL") or target.id in self._DDL_SQL_NAMES)
            for target in node.targets
        ):
            previous = self._in_ddl
            self._in_ddl = True
            try:
                return cast(ast.Assign, self.generic_visit(node))
            finally:
                self._in_ddl = previous
        return cast(ast.Assign, self.generic_visit(node))

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if self._in_ddl and isinstance(node.value, str):
            from polylogue.storage.sqlite.archive_tiers.schema_identity import _normalize_schema_sql

            return ast.copy_location(ast.Constant(_normalize_schema_sql(node.value)), node)
        return node


def _source_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _source_file_from_reference(reference: str) -> str:
    """Return the file portion of a declared ``path:Symbol`` reference."""
    path, separator, _symbol = reference.partition(":")
    return path if separator else reference


#: Per-process content digests, keyed by the full inode identity of the file
#: they were read from. ``st_ctime_ns`` is the load-bearing component: it
#: advances on every write and, unlike ``st_mtime_ns``, cannot be restored by
#: ``os.utime``, so a same-length rewrite under a replayed mtime misses this
#: cache and is re-read.
_SOURCE_DIGESTS: dict[tuple[str, int, int, int, int, int], str] = {}


def _source_signature(path: Path) -> tuple[str, str, int]:
    """Identify one parser source by its contents, not its stat metadata.

    The persistent fingerprint memo is keyed by these signatures. Keyed on
    (path, mtime, size) it is reused by any rewrite preserving both -- a
    same-length edit under a restored mtime, which checkout, patch
    application, and archive extraction all produce -- and the stale parser
    fingerprint then claims semantics the file no longer has.

    Digesting the bytes is the identity; the stat-keyed cache above only
    avoids re-reading a file whose inode has not been touched since.
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_dev, stat.st_ino)
    digest = _SOURCE_DIGESTS.get(key)
    if digest is None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _SOURCE_DIGESTS[key] = digest
    return str(path), digest, stat.st_size


def _fingerprint_path_label(path: Path) -> str:
    """Describe repository sources without binding a stamp to one checkout."""
    try:
        return path.relative_to(_SOURCE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _module_path(base: Path) -> Path | None:
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_init = base / "__init__.py"
    return package_init.resolve() if package_init.is_file() else None


@lru_cache(maxsize=2048)
def _local_import_paths(signature: tuple[str, str, int]) -> tuple[str, ...]:
    """Return local Python dependencies of one parser-semantic source file."""
    path = Path(signature[0])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[Path] = set()
    for node in ast.walk(tree):
        candidate: Path | None = None
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                candidate = _module_path(base / Path(*(node.module or "").split(".")))
            elif node.module and node.module.startswith("polylogue."):
                candidate = _module_path(_SOURCE_ROOT / Path(*node.module.split(".")))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("polylogue."):
                    resolved = _module_path(_SOURCE_ROOT / Path(*alias.name.split(".")))
                    if resolved is not None:
                        found.add(resolved)
        if candidate is not None:
            found.add(candidate)
    return tuple(sorted(str(item) for item in found))


@lru_cache(maxsize=64)
def _semantic_source_closure(root: Path, paths: tuple[str, ...], excluded_labels: frozenset[str]) -> tuple[Path, ...]:
    pending = [_source_path(path, root) for path in paths]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if _fingerprint_path_label(path) in excluded_labels:
            continue
        found.add(path)
        for dependency in _local_import_paths(_source_signature(path)):
            pending.append(Path(dependency))
    return tuple(sorted(found))


def _semantic_source_paths(
    paths: tuple[str, ...], *, excluded_labels: frozenset[str] = _GENERATED_PROVENANCE_FINGERPRINT_PATHS
) -> tuple[Path, ...]:
    """Return the parser-semantic import closure of ``paths``.

    Membership is walked once per process per argument set. Only the member
    *list* is memoized: every caller re-derives :func:`_source_signature` for
    each member on each call, so an edited source still changes its content
    digest and the fingerprint that digest keys.

    The memo holds a member's import graph fixed for the life of the process,
    the same assumption :func:`_local_import_paths` makes by caching edges per
    signature. ``_SOURCE_ROOT`` belongs in the key because declared paths are
    relative to it and a substituted root names different files.
    """
    return _semantic_source_closure(_SOURCE_ROOT, paths, excluded_labels)


#: Bump when the normalization below changes; it is part of the disk memo key.
_FINGERPRINT_ALGORITHM_VERSION = 4


def _fingerprint_memo_path(signatures: tuple[tuple[str, str, int], ...], namespace: str) -> Path | None:
    """Where this exact set of source signatures memoizes its fingerprint.

    Parsing and normalizing the closure of parser sources costs tens of
    seconds of CPU per process; every test worker and every CLI start paid
    it. The signatures encode each source's content digest, so a memo keyed
    by them is invalidated by any edit. Returns None where no cache
    directory can exist (installed packages), which falls back to computing.
    """
    root = _SOURCE_ROOT / ".cache" / "source-fingerprints"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    digest = hashlib.sha256(
        json.dumps(
            {"version": _FINGERPRINT_ALGORITHM_VERSION, "namespace": namespace, "signatures": signatures},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return root / f"{digest}.txt"


@lru_cache(maxsize=128)
def _fingerprint_sources_cached(signatures: tuple[tuple[str, str, int], ...], namespace: str) -> str:
    memo = _fingerprint_memo_path(signatures, namespace)
    if memo is not None:
        try:
            cached = memo.read_text(encoding="utf-8").strip()
        except OSError:
            cached = ""
        if len(cached) == 64:
            return cached
    fingerprint = _fingerprint_sources_compute(signatures, namespace)
    if memo is not None:
        try:
            scratch = memo.with_name(f"{memo.name}.{id(signatures)}.tmp")
            scratch.write_text(fingerprint, encoding="utf-8")
            scratch.replace(memo)
        except OSError:
            pass
    return fingerprint


def _fingerprint_sources_compute(signatures: tuple[tuple[str, str, int], ...], namespace: str) -> str:
    fragments: list[dict[str, str]] = []
    for path_string, _mtime_ns, _size in signatures:
        tree = ast.parse(Path(path_string).read_text(encoding="utf-8"))
        normalized = _DocstringStripper().visit(tree)
        if (
            Path(path_string).name == "origin_specs.py"
            and _fingerprint_path_label(Path(path_string)) == "polylogue/sources/origin_specs.py"
        ):
            normalized = _ProjectionFingerprintStripper().visit(normalized)
        normalized = _SchemaDdlFingerprintStripper().visit(normalized)
        fragments.append(
            {
                "path": _fingerprint_path_label(Path(path_string)),
                "ast": ast.dump(normalized, annotate_fields=True, include_attributes=False),
            }
        )
    payload = {"namespace": namespace, "sources": fragments}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fingerprint_sources(
    paths: tuple[str, ...], *, namespace: str, excluded_labels: frozenset[str] = frozenset()
) -> str:
    source_paths = _semantic_source_paths(
        paths,
        excluded_labels=excluded_labels | _GENERATED_PROVENANCE_FINGERPRINT_PATHS,
    )
    signatures = tuple(_source_signature(path) for path in source_paths)
    return _fingerprint_sources_cached(signatures, namespace)


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
    # Suffixes safe to project onto a whole watched root. Path-scoped forms
    # such as opaque sidecars stay governed by ``path_pattern``.
    watch_suffixes: tuple[str, ...] | None = None

    def matches(self, source_path: str) -> bool:
        return re.search(self.path_pattern, source_path.replace("\\", "/")) is not None


@dataclass(frozen=True, slots=True)
class SourceClassRecognition:
    """The declaration-owned result of inspecting one source candidate.

    Enumeration is deliberately separate from admission: a suffix may make a
    file observable, but only structural evidence can make it a session.
    ``unsupported`` is a typed observation and must not be sent to a parser.
    """

    source_class: SourceClass
    reason: str


@dataclass(frozen=True, slots=True)
class TopologyCapability:
    """Per-origin declaration of the topology evidence a parser may use."""

    state: TopologyCapabilityState
    evidence: tuple[str, ...]
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state == "unknown":
            raise ValueError("topology capability cannot remain unknown")
        if not self.evidence:
            raise ValueError("topology capability requires evidence or an absence reason")
        if self.state == "structurally-absent" and not self.reason:
            raise ValueError("structurally absent topology capability requires a reason")


@dataclass(frozen=True, slots=True)
class TopologyCapabilities:
    """Complete, table-driven topology disposition for one origin."""

    message_parent: TopologyCapability
    message_branch_state: TopologyCapability
    session_parent_target: TopologyCapability
    inheritance_branch_point: TopologyCapability
    parent_dispatch: TopologyCapability

    def as_dict(self) -> dict[str, TopologyCapability]:
        return {
            "message_parent": self.message_parent,
            "message_branch_state": self.message_branch_state,
            "session_parent_target": self.session_parent_target,
            "inheritance_branch_point": self.inheritance_branch_point,
            "parent_dispatch": self.parent_dispatch,
        }


def _absent_topology(reason: str) -> TopologyCapability:
    return TopologyCapability("structurally-absent", (reason,), reason)


def _no_topology_capabilities(origin: Origin) -> TopologyCapabilities:
    prefix = f"{origin.value} admits no"
    return TopologyCapabilities(
        message_parent=_absent_topology(f"{prefix} provider message-parent field"),
        message_branch_state=_absent_topology(f"{prefix} provider branch-state field"),
        session_parent_target=_absent_topology(f"{prefix} provider session-parent target"),
        inheritance_branch_point=_absent_topology(f"{prefix} provider inheritance boundary"),
        parent_dispatch=_absent_topology(f"{prefix} provider parent-dispatch identity"),
    )


def _looks_like_extracted_transcript_corpus_path(
    path: Path,
    *,
    payload: object | None = None,
) -> bool:
    """Inspect a bounded record prefix for external-transcript provenance.

    Reads at most 32 records and stops at the first record carrying a provider
    envelope, which disqualifies the stream outright -- so a genuine transcript
    costs one decoded line.
    """
    from polylogue.archive.artifact_taxonomy.support import (
        looks_like_extracted_transcript_corpus,
        record_carries_provider_envelope,
    )
    from polylogue.core.json import json_document

    if payload is not None:
        records: list[object] = list(payload) if isinstance(payload, list) else [payload]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if record_carries_provider_envelope(record):
                        return False
                    records.append(record)
                    if len(records) >= 32:
                        break
        except OSError:
            return False
    else:
        return False
    dict_items = [item for item in (json_document(item) for item in records) if item]
    return looks_like_extracted_transcript_corpus(dict_items)


def recognize_source_class(
    provider: Provider,
    source_path: str | Path,
    *,
    payload: object | None = None,
    source_only: bool = False,
) -> SourceClassRecognition | None:
    """Classify broad-root candidates before provider-session admission.

    Keep this dispatch declaration-owned and structural; callers may still
    enumerate cheaply by suffix, but may not assign a provider session from
    that suffix alone.
    """
    if provider is Provider.UNKNOWN:
        if source_only and Path(source_path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return SourceClassRecognition("unsupported", "SQLite has no declared provider source class")
        return None

    from polylogue.sources.parsers import (
        antigravity,
        codex_state,
        hermes_spans,
        hermes_state,
        hermes_verification,
        local_agent,
    )

    path = Path(source_path)
    rule = artifact_rule_for_path(provider, str(path))
    if rule is not None and rule.parse_policy != "session":
        return SourceClassRecognition("non_session", f"declared {rule.kind} artifact")
    if rule is not None and rule.parse_policy == "session":
        # A session path rule states a location, so a generated extract
        # dropped into a provider's transcript directory matches it exactly
        # as the transcripts do. Records that name the transcript their
        # turns were copied out of separate the two by their own provenance,
        # which is what keeps the derivative and its original apart in the
        # source manifest instead of both landing in one denominator.
        if _looks_like_extracted_transcript_corpus_path(path, payload=payload):
            return SourceClassRecognition("non_session", "extracted transcript corpus")
        return SourceClassRecognition("session", f"declared {rule.kind} source class")
    if provider is Provider.ANTIGRAVITY:
        classification = antigravity.classify_source_path(path)
        if classification.role.value == "conversation_protobuf":
            return SourceClassRecognition("session", "Antigravity declared conversation source class")
        if classification.role.value != "unknown":
            return SourceClassRecognition("non_session", "Antigravity declared artifact source class")
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"} and antigravity.looks_like_trajectory_db_path(path):
            return SourceClassRecognition("session", "Antigravity trajectory SQLite schema signature")

    if path.suffix.lower() == ".zip":
        return None

    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        if source_only:
            if provider is Provider.CODEX:
                declaration = codex_state.declared_codex_sqlite_classification(path)
                if declaration is None:
                    return SourceClassRecognition("unsupported", "Codex SQLite has no declared database identity")
                if declaration.disposition == "out-of-scope":
                    return SourceClassRecognition("unsupported", f"declared Codex {declaration.kind} database")
                return SourceClassRecognition("session", f"declared Codex {declaration.kind} database")
            if provider is Provider.HERMES and path.name in {"state.db", "verification_evidence.db"}:
                return SourceClassRecognition("session", "declared Hermes SQLite source class")
            return SourceClassRecognition("unsupported", f"{provider.value} SQLite has no declared source class")
        if provider is Provider.HERMES:
            if hermes_state.looks_like_state_db_path(
                path
            ) or hermes_verification.looks_like_verification_evidence_db_path(path):
                return SourceClassRecognition("session", "Hermes SQLite schema signature")
            return SourceClassRecognition("unsupported", "Hermes SQLite lacks a declared state/verification schema")
        if provider is Provider.CODEX:
            if codex_state.is_in_scope_codex_sqlite_path(path):
                return SourceClassRecognition("session", "Codex SQLite schema signature")
            return SourceClassRecognition("unsupported", "Codex SQLite lacks a declared state schema")
        return SourceClassRecognition("unsupported", f"{provider.value} SQLite has no declared source class")

    if source_only:
        return None

    if provider not in {Provider.HERMES, Provider.ANTIGRAVITY}:
        return None

    if payload is None:
        try:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                records: list[object] = []
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            records.append(json.loads(line))
                        if len(records) >= 32:
                            break
                payload = records
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return SourceClassRecognition("unsupported", "Hermes candidate is not readable JSON")

    if provider is Provider.HERMES:
        record = payload if isinstance(payload, dict) else None
        if record is not None and (
            hermes_spans.looks_like_atif_payload(record)
            or hermes_spans.looks_like_atof_payload(record)
            or local_agent.looks_like_hermes(record)
        ):
            return SourceClassRecognition("session", "Hermes declared JSON structural signature")
        if isinstance(payload, list) and any(
            isinstance(item, dict) and hermes_spans.looks_like_atof_payload(item) for item in payload
        ):
            return SourceClassRecognition("session", "Hermes ATOF JSONL structural signature")
        return SourceClassRecognition("unsupported", "Hermes candidate has no declared source-class signature")

    if provider is Provider.ANTIGRAVITY:
        if isinstance(payload, dict) and antigravity.looks_like_markdown_export(payload):
            return SourceClassRecognition("session", "Antigravity language-server export structural signature")
        return SourceClassRecognition("unsupported", "Antigravity candidate has no declared source-class signature")
    return None


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
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as error:
            logger.warning("schema_package_unreadable path=%s error=%s", path, error)
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
class DatabaseTableRule:
    """Disposition of one observed table within a declared database member.

    Member-level admission decides whether Polylogue opens a database at all.
    Table rules make the narrower export decision inspectable: a retained table
    belongs to the member's logical revision, while a deliberately excluded
    table remains visible to schema observation with its reason.
    """

    table: str
    disposition: DatabaseTableDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class DatabaseMemberRule:
    """Admission disposition for one database member of a DB-shaped origin.

    A mutable database is admitted for the logical content it carries, so an
    admitted member names that content (``logical_tables``) and who reads it
    (``consumer``). ``consumer`` is ``None`` only for ``acquire-partial``: the
    logical product is declared and retained as durable evidence with no
    typed reader yet. ``out-of-scope`` members declare neither -- nothing is
    acquired to have a product or a consumer.
    """

    filename: str
    disposition: DatabaseMemberDisposition
    kind: str
    reason: str
    logical_tables: tuple[str, ...] = ()
    consumer: str | None = None
    table_rules: tuple[DatabaseTableRule, ...] = ()

    def table_rule(self, table: str) -> DatabaseTableRule | None:
        return next((item for item in self.table_rules if item.table == table), None)


@dataclass(frozen=True, slots=True)
class DatabaseSourceCapability:
    """Declared acquisition contract for an origin backed by SQLite files."""

    snapshot_method: Literal["logical_export"]
    consistency_fence: str
    revision_identity: str
    raw_id_strategy: str
    members: tuple[DatabaseMemberRule, ...]
    full_snapshot_per_revision: bool
    snapshot_lineage_policy: str

    def member(self, filename: str) -> DatabaseMemberRule | None:
        return next((item for item in self.members if item.filename == filename), None)

    @property
    def filenames(self) -> frozenset[str]:
        return frozenset(item.filename for item in self.members)


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
    topology_capabilities: TopologyCapabilities
    #: Whether this origin is offered as a public filter/completion choice.
    #: Compatibility-only and non-session evidence origins can remain in the
    #: authoritative enum without being advertised as query choices.
    public_filter: bool = True
    #: The source-level continuation law consumed by acquisition and cuts.
    frontier_kind: SourceFrontierKind = "exact-prefix"
    #: Ordered executable detector claims. Parser modules keep their shape
    #: predicates; this declaration owns which predicates may classify input.
    detector_bindings: tuple[DetectorBinding, ...] = ()
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
    #: The closed set of :class:`ToolResultUnknownReason` members this origin's
    #: parsers can derive from its own record structures. It owns no producer
    #: semantics -- a reason is always read off the record -- but it names who
    #: owns each reason, so a reason no parser here can derive refuses the
    #: write instead of entering the archive unattributed.
    tool_outcome_unknown_reasons: frozenset[ToolResultUnknownReason] = frozenset()
    database_capability: DatabaseSourceCapability | None = None

    def parser_fingerprint(self) -> str:
        """Return the origin-scoped fingerprint of parser output semantics."""
        declared_paths = list(self.parser_paths)
        declared_paths.extend(self.assembly_paths)
        if self.stream_parser_path is not None:
            declared_paths.append(self.stream_parser_path)
        if self.assembly_spec_path is not None:
            declared_paths.append(self.assembly_spec_path)
        declared_paths.extend(rule.parser_path for rule in self.artifact_rules if rule.parser_path is not None)
        source_paths = tuple(dict.fromkeys(_source_file_from_reference(path) for path in declared_paths))
        return _fingerprint_sources(
            source_paths,
            namespace=f"parser:{self.origin.value}",
            excluded_labels=_PARSER_DIAGNOSTIC_FINGERPRINT_PATHS,
        )


def _detector_declaration_fingerprint_payload() -> tuple[dict[str, object], ...]:
    """Project every runtime detector claim that can change lowering behavior."""
    return tuple(
        {
            "origin": spec.origin.value,
            "lifecycle": spec.lifecycle,
            "detector_tightness": spec.detector_tightness,
            "bindings": tuple(
                {
                    "binding_id": binding.binding_id,
                    "mode": binding.mode.value,
                    "predicate_path": binding.predicate_path,
                    "local_rank": binding.local_rank,
                    "mode_rank": binding.mode_rank,
                    "evidence_label": binding.evidence_label,
                    "fixed_provider": binding.fixed_provider.value if binding.fixed_provider is not None else None,
                    "dynamic_provider_path": binding.dynamic_provider_path,
                    "dynamic_provider_allowlist": tuple(
                        provider.value for provider in binding.dynamic_provider_allowlist
                    ),
                }
                for binding in spec.detector_bindings
            ),
        }
        for spec in ORIGIN_SPECS
        if spec.detector_bindings
    )


def lowering_fingerprint() -> str:
    """Return the shared lowering, identity, revision, lineage, and detector fingerprint."""
    payload = {
        "source_fingerprint": _fingerprint_sources(_LOWERING_FINGERPRINT_PATHS, namespace="lowering"),
        "detector_declarations": _detector_declaration_fingerprint_payload(),
        "topology_capabilities": {
            spec.origin.value: {
                name: {
                    "state": capability.state,
                    "evidence": capability.evidence,
                    "reason": capability.reason,
                }
                for name, capability in spec.topology_capabilities.as_dict().items()
            }
            for spec in ORIGIN_SPECS
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def replay_routing_fingerprint() -> str:
    """Fingerprint the production raw replay router, including special paths."""
    return _fingerprint_sources(_REPLAY_ROUTING_FINGERPRINT_PATHS, namespace="raw-replay-routing")


def retained_enumeration_fingerprint() -> str:
    """Bind accepted input enumeration to the actual decoder/coordinate closure."""
    return _fingerprint_sources(
        ("polylogue/sources/retained_acquisition.py",), namespace="retained-source-enumeration-v1"
    )


def materializer_fingerprint() -> str:
    """Fingerprint the session-insight materializer used by index replay."""
    return _fingerprint_sources(_MATERIALIZER_FINGERPRINT_PATHS, namespace="session-materializer")


#: The fingerprint entry points that feed ``derived_schema_identity``. The
#: per-origin parser fingerprints are deliberately absent: they are not part of
#: the derived identity.
_DERIVED_IDENTITY_ENTRY_PATHS: tuple[str, ...] = tuple(
    sorted(set(_LOWERING_FINGERPRINT_PATHS + _MATERIALIZER_FINGERPRINT_PATHS + _REPLAY_ROUTING_FINGERPRINT_PATHS))
)


def derived_identity_source_closure() -> tuple[Path, ...]:
    """Return every source file whose content feeds the derived schema identity.

    The identity digests AST-normalized source, so editing any file in this
    closure moves it — a pure-performance change with no schema edit moves it
    just as surely as a new column. Membership follows the import graph, not
    directory boundaries, which is why callers must ask rather than assume.
    """
    return _semantic_source_paths(
        _DERIVED_IDENTITY_ENTRY_PATHS,
        excluded_labels=_GENERATED_PROVENANCE_FINGERPRINT_PATHS,
    )


def in_derived_identity_closure(path: Path | str) -> bool:
    """Whether editing ``path`` would move the derived schema identity."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _SOURCE_ROOT / candidate
    candidate = candidate.resolve(strict=False)
    return candidate in {member.resolve(strict=False) for member in derived_identity_source_closure()}


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
        if set(spec.topology_capabilities.as_dict()) != {
            "message_parent",
            "message_branch_state",
            "session_parent_target",
            "inheritance_branch_point",
            "parent_dispatch",
        }:
            raise ValueError(f"{spec.origin.value}: topology capability census is incomplete")
        if len(spec.provider_wires) > 1 and not spec.collision_policy:
            raise ValueError(f"{spec.origin.value}: multiple provider wires require an explicit collision policy")
        if spec.lifecycle == "executable":
            if spec.detector_tightness is None:
                raise ValueError(f"{spec.origin.value}: executable origin requires detector tightness")
            if not spec.detector_bindings:
                raise ValueError(f"{spec.origin.value}: executable origin requires detector binding")
            if not spec.parser_paths:
                raise ValueError(f"{spec.origin.value}: executable origin requires parser binding")
            for rule in spec.artifact_rules:
                if rule.parse_policy != "raw-only" and rule.parser_path is None:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires a parser binding")
                if not rule.path_suffixes:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires acquisition suffixes")
                if any(
                    suffix and (not suffix.startswith(".") or suffix != suffix.lower()) for suffix in rule.path_suffixes
                ):
                    raise ValueError(
                        f"{spec.origin.value}: {rule.kind} acquisition suffixes must be lowercase dot suffixes or empty"
                    )
                if rule.watch_suffixes is not None and any(
                    not suffix.startswith(".") or suffix != suffix.lower() for suffix in rule.watch_suffixes
                ):
                    raise ValueError(f"{spec.origin.value}: {rule.kind} watch suffixes must be lowercase dot suffixes")
            if any("db" in mode or "sqlite" in mode for mode in spec.acquisition_modes):
                if spec.database_capability is None:
                    raise ValueError(f"{spec.origin.value}: database acquisition requires a database capability")
                if not spec.database_capability.members:
                    raise ValueError(f"{spec.origin.value}: database capability requires member declarations")
                filenames = [member.filename for member in spec.database_capability.members]
                if len(filenames) != len(set(filenames)):
                    raise ValueError(f"{spec.origin.value}: database member filenames must be unique")
                if any(not member.filename or "/" in member.filename for member in spec.database_capability.members):
                    raise ValueError(f"{spec.origin.value}: database member filenames must be basenames")
                for member in spec.database_capability.members:
                    if member.disposition == "out-of-scope":
                        if member.logical_tables or member.consumer is not None or member.table_rules:
                            raise ValueError(
                                f"{spec.origin.value}: out-of-scope database member "
                                f"{member.filename} cannot declare a logical product"
                            )
                        continue
                    if not member.logical_tables:
                        raise ValueError(
                            f"{spec.origin.value}: admitted database member {member.filename} "
                            "must name the logical tables it is acquired for"
                        )
                    if member.disposition == "acquire" and not member.consumer:
                        raise ValueError(
                            f"{spec.origin.value}: acquired database member {member.filename} "
                            "must name the consumer of its logical product"
                        )
                    if member.table_rules:
                        table_names = [rule.table for rule in member.table_rules]
                        if len(table_names) != len(set(table_names)) or any(not name for name in table_names):
                            raise ValueError(
                                f"{spec.origin.value}: database member {member.filename} has duplicate or blank table rules"
                            )
                        retained_tables = {
                            rule.table for rule in member.table_rules if rule.disposition != "deliberately-excluded"
                        }
                        if retained_tables != set(member.logical_tables):
                            raise ValueError(
                                f"{spec.origin.value}: database member {member.filename} table rules must account for "
                                "exactly its logical tables"
                            )
                        if any(not rule.reason.strip() for rule in member.table_rules):
                            raise ValueError(
                                f"{spec.origin.value}: database member {member.filename} table rules require reasons"
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
    spec = OriginSpec(
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
            "NOT_REPORTED when the Anthropic-protocol segment carries no is_error, UNSUPPORTED_CONSTRUCT when "
            "it carries an is_error/exit_code the shared mapping cannot read, and DISTRUSTED for the "
            "background-task start acknowledgement's is_error=false (see _mark_background_task_start).",
            "polylogue-cgfy AC1: disposition of the 'other unread keys of substance' the bead's corpus "
            "enumeration named beyond the structuredPatch/file_edits cluster (already read, see the note "
            "above) and beyond stop_reason/stop_sequence/parentToolUseID/cache_creation/ttftMs/todos/"
            "toolUseResult.sandbox/filenames/numFiles (already read, see code_parser.py's "
            "_message_usage_event_payload and toolUseResult structural-fact projection). READ (this batch): "
            "requestId (1,171 sampled occurrences -- the Anthropic API per-call request id, a real "
            "cross-reference key against provider-side billing/support records) rides the message_usage "
            "session-event payload as request_id. thinkingMetadata (3,620 occurrences over 14,536 session "
            "files, 2026-09-06 walk) rides its own claude_thinking_budget event: the field occurs only on "
            "user records, and none of them carries message.usage, so the usage-gated message_usage payload "
            "cannot see it. maxThinkingTokens (2,118 records, 31,999 in every one) and the "
            "level/disabled/triggers shape (1,502) both land there; triggers names the span of the user's "
            "own prompt that raised the effort. MEASURED NEGATIVE: userType is the literal string 'external' "
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
            "code_parser.py's _NON_MESSAGE_SIDECAR_RECORD_TYPES already carries "
            "a per-type disposition with corpus counts "
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
            "polylogue-chemh / polylogue-esvzb (exhaustive walk of 14,536 session files, 2026-09-06): every "
            "record type the corpus carries now has a disposition entry -- atis-latch and agent-color are "
            "declared transient on measured evidence, frame-link and the two artifact ledgers persist as "
            "typed events. A record type in no table persists as claude_unclassified_record rather than "
            "vanishing at the empty-content drop, so the next CLI version's new kind is visible in the "
            "index. forkedFrom ({sessionId, messageUuid}, 10,561 records across 9 forked sessions) resolves "
            "to the session's parent edge and rides claude_forked_from with its branch point; it is adopted "
            "only when no identity-anchored route already resolved a parent.",
            "polylogue-sd7n1 (full-corpus census 2026-09-07, 14,684 session files): the capability-attribution "
            "cluster -- attributionAgent (517,881), attributionSkill (50,586), attributionMcpServer and "
            "attributionMcpTool (7,683 each, never one without the other), attributionPlugin (5,430) -- is "
            "always a top-level string on an assistant record and occurs on no other record type. All five "
            "ride claude_capability_attribution, one event per record carrying any of them, keyed to the "
            "message uuid. Read off the record rather than inside the usage-gated message_usage payload so "
            "an assistant turn with no message.usage keeps its attribution, and emitted before the "
            "empty-content drop so an API-error row still names the capability that produced the attempt. "
            "Plugin is not redundant with skill: a plugin-shipped skill carries the plugin prefix in its own "
            "value (superpowers:brainstorming) but feature-dev stamps the plugin on 541 records with no "
            "skill at all. attributionAgent is read here on the session route rather than left to "
            "orchestration.py's Workflow agent-*.meta.json allowlist: it occurs only in subagent "
            "(agent-*.jsonl) transcripts, and a transcript that replays a parent's prefix carries the "
            "parent's agent on the contiguous replayed head and its own on the divergent tail (every "
            "multi-valued file in a 300-file walk, e.g. 8 triage turns then 52 fork turns), so a "
            "session-level projection would erase the split.",
            "polylogue-jpq9x (exhaustive walk of 14,667 session files, 2026-09-07): the top-level rendered[] "
            "array is the text the provider injected into the model's context for an attachment record, and "
            "it occurs on no other record type (55,006 records, 55,025 entries, each entry exactly "
            "{content}). READ: rendered[].content rides the attachment's own session event as `rendered`. It "
            "is not a second spelling of the payload -- 6,153 of those entries appear nowhere in the "
            "attachment payload (a `file` is rendered with line numbers, a `queued_command` inside the "
            "notification framing that made it a system reminder) and 4,440 more sit on payloads with no "
            "substantive text at all. BOUNDED, not verbatim, for the six subtypes with a bounded payload "
            "builder (deferred_tools_delta, mcp_instructions_delta, agent_listing_delta, skill_listing, "
            "invoked_skills, diagnostics): there the rendered text IS the injected instruction body those "
            "builders exclude, so only rendered_char_count/rendered_count are kept -- 10.6 MB of the "
            "corpus's 42.6 MB of rendered text. DROPPED WITH ITS SUBTYPE: a subtype in "
            "_ATTACHMENT_TRANSIENT_SUBTYPES emits no event, so its rendered content has nothing to ride; "
            "total_tokens_reminder is 39,551 of those records and its remaining-token number is measured to "
            "vary, which the transient ruling for that subtype (made on the payload) does not account for.",
            "polylogue-v53yf (same walk): the record-level provenance keys outside `message` are READ as one "
            "claude_session_environment event per session, each key a value->count map. entrypoint (cli "
            "2,263,879 / sdk-cli 130,097 / sdk-py 2,664 over 8,091 files, never varying within a file) is "
            "the only in-band evidence separating a dispatched SDK lane from an operator's interactive "
            "session -- the two produce identical record shapes. version (133 distinct builds over 14,287 "
            "files; 177 files carry more than one, a resume spanning a CLI upgrade) names the producer that "
            "wrote the bytes. permissionMode (5 values, 65,746 records) and promptSource (typed/queued/"
            "system/sdk/suggestion_accepted, 9,253 records) are per-record state, so their counts are the "
            "session's distribution of them. All four ride attachment and progress records that return "
            "before ordinary message parsing, so they are read at the top of the fold. "
            "attachment.failedMcpServers[] (498 entries over the corpus, all on deferred_tools_delta, keys "
            "exactly {name, errorCode, error}) rides that subtype's bounded payload in full: it is the only "
            "record that a capability the session expected never arrived, and a missing tool is otherwise "
            "indistinguishable from one that was never configured.",
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
        frontier_kind="claude-header-body",
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
                # Tool-result output is opaque and Claude Code has emitted
                # JSON, text, HTML, and extensionless files. The path pattern
                # admits all four forms; only JSON participates in the
                # ordinary suffix projection because text and HTML are
                # path-scoped and must not widen the Claude root globally.
                path_suffixes=(".json", ".txt", ".html", ""),
                watch_suffixes=(".json",),
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
            OriginArtifactRule(
                kind="agent_memory_document",
                # ``~/.claude/projects/<project>/memory/**.md`` is Claude
                # Code's own memory directory: the harness writes and
                # rewrites these documents itself, independently of whether
                # any session quoted one. Scoped to the ``memory/`` segment
                # of a project directory so the rest of a project tree, and
                # every Markdown file elsewhere under the watched root, stays
                # outside the declaration (polylogue-rovf5).
                path_pattern=r"(?:^|/)projects/[^/]+/memory/(?:[^/]+/)*[^/]+\.md$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="agent_memory_document",
                fidelity_note=(
                    "Memory documents are retained verbatim as source bytes and never parsed into a "
                    "session or promoted to a user assertion: the harness authored them, so their "
                    "authoredness is the harness and their session ownership stays unknown. The "
                    "project directory in the path is the scope, so the same basename under two "
                    "projects or two installs is two distinct retained objects. Content change is an "
                    "ordinary newer observation of the same coordinate; disappearance retires nothing."
                ),
                path_suffixes=(".md",),
                # Path-scoped: ``.md`` must not become an admitted suffix for
                # the whole ``projects/`` root, only for ``memory/`` inside it.
                watch_suffixes=(),
            ),
            OriginArtifactRule(
                kind="session_index",
                # ``projects/<project>/sessions-index.json`` is the assembly
                # input that resolves a Claude Code session's curated title
                # and branch. Declaring it makes the acquired bytes the
                # archive's own evidence, so a reindex resolves the same
                # title with the original tree gone (polylogue-ximhz, D3).
                path_pattern=r"(?:^|/)projects/[^/]+/sessions-index\.json$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="session_index",
                fidelity_note=(
                    "Claude Code rewrites this index whole on every update, so each observation is a "
                    "competing snapshot rather than a continuation; the newest retained observation of "
                    "the same project directory is the current value and older ones stay archived. "
                    "Consumed by sources/retained_assembly.py, never parsed as a session."
                ),
                path_suffixes=(".json",),
                # ``.json`` is already an admitted Claude Code suffix; this
                # rule states a location, not a new suffix family.
                watch_suffixes=(),
            ),
            OriginArtifactRule(
                kind="prompt_history_log",
                # ``~/.claude/history.jsonl`` is global to one Claude Code
                # install and sits two levels above the project directories,
                # outside the sessions root. Its rows carry the paste
                # evidence no transcript records.
                path_pattern=r"(?:^|/)history\.jsonl$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="prompt_history_log",
                fidelity_note=(
                    "Prompt-history rows are retained verbatim and joined to a session by their own "
                    "sessionId plus a bounded timestamp window (assembly_claude_code.py); an ambiguous "
                    "row is dropped rather than fanned across candidates. The install directory is the "
                    "scope, so two installs never share one history."
                ),
                path_suffixes=(".jsonl",),
                watch_suffixes=(),
            ),
        ),
        assembly_spec_path="polylogue/sources/assembly_claude_code.py:ClaudeCodeAssemblySpec",
        display_description="Claude Code local sessions (lab: Anthropic)",
        topology_capabilities=_no_topology_capabilities(origin),
        tool_outcome_unknown_reasons=frozenset(
            {
                ToolResultUnknownReason.NOT_REPORTED,
                ToolResultUnknownReason.DISTRUSTED,
                ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT,
            }
        ),
    )
    topology_capabilities = TopologyCapabilities(
        message_parent=TopologyCapability("carried", ("claude_code.parentUuid",)),
        message_branch_state=TopologyCapability(
            "positive-derived",
            ("code_parser._finalize_code_session.branch_type",),
            "branch type is derived from record evidence",
        ),
        session_parent_target=TopologyCapability(
            "positive-derived",
            (
                "code_parser._finalize_code_session.parent_session_provider_id",
                "claude_code.forkedFrom.sessionId",
            ),
            "parent session is derived from the Claude sessionId relationship, or carried outright by "
            "forkedFrom on a forked session's own records",
        ),
        inheritance_branch_point=TopologyCapability(
            "carried",
            (
                "claude_code.forkedFrom.messageUuid",
                "claude_code fork-context-ref.parentLastUuid + .parentSessionId",
            ),
            "a forked session's records name the parent message it diverged at; a forked subagent transcript names it in fork-context-ref and never replays it",
        ),
        parent_dispatch=TopologyCapability(
            "positive-derived",
            (
                "claude_code user.toolUseResult.agentId + tool_result.tool_use_id (Agent/Task result)",
                "claude_code progress.data.type=agent_progress.parentToolUseID",
                "claude_code progress.data.childSessionId/agentId when present",
                "claude_code subagents/agent-*.meta.json toolUseId (source tier, child-bound)",
            ),
            "parent-side dispatch evidence is retained only when the wire carries an exact child identity",
        ),
    )
    return replace(spec, topology_capabilities=topology_capabilities)


def artifact_rule_for_path(provider: Provider, source_path: str) -> OriginArtifactRule | None:
    """Return the owning OriginSpec artifact rule for a native source path."""

    for spec in ORIGIN_SPECS:
        if provider not in spec.provider_wires:
            continue
        for rule in spec.artifact_rules:
            if rule.matches(source_path):
                return rule
    return None


def path_declaration_refuses_session(provider: Provider, source_path: str | Path) -> bool:
    """Whether the owning artifact rule refuses session parsing outright.

    ``parse_policy="raw-only"`` states that a family's bytes are evidence and
    never a conversation, and that content shape cannot decide it: a
    tool-result sidecar can reproduce a genuine export byte-for-byte
    (polylogue-omsw) and a prompt-history log carries the same ``sessionId``
    keys a transcript does (polylogue-ximhz). For those families the path rule
    is terminal. ``fact`` and ``session`` rules keep the ordinary behaviour
    where positive decoded session evidence may outrank a location.
    """
    rule = artifact_rule_for_path(provider, str(source_path))
    return rule is not None and rule.parse_policy == "raw-only"


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
            suffixes.extend(rule.watch_suffixes if rule.watch_suffixes is not None else rule.path_suffixes)
    return tuple(dict.fromkeys(suffix.lower() for suffix in suffixes))


def database_capability_for_provider(provider: Provider) -> DatabaseSourceCapability | None:
    """Return the declared SQLite acquisition contract for a provider wire."""

    for spec in ORIGIN_SPECS:
        if provider in spec.provider_wires:
            return spec.database_capability
    return None


@dataclass(frozen=True, slots=True)
class DatabaseMemberBinding:
    """One declared database member together with the origin that declares it."""

    origin: Origin
    provider: Provider
    member: DatabaseMemberRule


def database_member_for_filename(filename: str) -> DatabaseMemberBinding | None:
    """Return the declared member rule owning a database basename.

    Acquisition needs the declared logical product for a file it holds only a
    path to, so the lookup is by basename across every DB-shaped origin. A
    basename claimed by two origins is a declaration defect, reported by
    :func:`validate_database_member_filenames` rather than resolved here.
    """
    for spec in ORIGIN_SPECS:
        capability = spec.database_capability
        if capability is None:
            continue
        member = capability.member(filename)
        if member is not None:
            return DatabaseMemberBinding(
                origin=spec.origin,
                provider=spec.provider_wires[0],
                member=member,
            )
    return None


def validate_database_member_filenames() -> tuple[str, ...]:
    """Return every database basename declared by more than one origin."""
    seen: dict[str, list[str]] = {}
    for spec in ORIGIN_SPECS:
        capability = spec.database_capability
        if capability is None:
            continue
        for member in capability.members:
            seen.setdefault(member.filename, []).append(spec.origin.value)
    return tuple(sorted(filename for filename, origins in seen.items() if len(origins) > 1))


def _chatgpt_spec() -> OriginSpec:
    origin = Origin.CHATGPT_EXPORT
    spec = OriginSpec(
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
        artifact_rules=(
            OriginArtifactRule(
                kind="export_asset_index",
                # The two cross-conversation lookup tables a GDPR/Takeout
                # export ships beside its conversation shards. Declaring them
                # makes the acquired bytes the archive's own evidence for the
                # attachment join instead of a live sibling-file read
                # (polylogue-ximhz, D3).
                path_pattern=r"(?:^|[/:])(?:library_files|conversation_asset_file_names)\.json$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="export_asset_index",
                fidelity_note=(
                    "Asset-name and library tables are retained verbatim and rebuilt into "
                    "ChatGPTAssetIndex from the acquired bytes. They are scoped to their own export: "
                    "the same asset id in two exports names two objects and never cross-binds."
                ),
                path_suffixes=(".json",),
                watch_suffixes=(),
            ),
            OriginArtifactRule(
                kind="export_asset",
                # An export member whose basename carries a provider file id.
                # One vintage names them ``file-<id>.dat``, another ships the
                # real extension or none at all, in per-conversation
                # subdirectories -- the id in the name is the identity, never
                # the suffix (assembly_chatgpt.py's ``_member_asset_id``).
                path_pattern=r"(?:^|[/:])file[-_][A-Za-z0-9]+[^/]*$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="export_asset",
                fidelity_note=(
                    "Asset bytes are retained once, content-addressed, under the member coordinate of "
                    "the export they came from. The attachment join resolves a provider file id to "
                    "those retained bytes, so a reindex reproduces the attachment identity and payload "
                    "with the original export gone."
                ),
                # The measured vintages: a ``file-<id>.dat`` family, members
                # under their real extension, and members under none. The id
                # in the name is the claim, never the suffix, so this list
                # documents what was observed rather than gating admission.
                path_suffixes=(".dat", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".pdf", ".json", ""),
                # Path-scoped by id-bearing member name: no suffix family may
                # be projected onto a whole watched root from this rule.
                watch_suffixes=(),
            ),
        ),
        fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
        coverage_refs=("provider-package:chatgpt-export/takeout-json@v1",),
        fidelity_notes=(
            "Browser capture remains an acquisition mode and is not a new public origin.",
            "A cross-conversation memory citation "
            "(message.metadata.conversation_context_citation_metadata) is conserved as a "
            "CONTENT_REFERENCE web construct whose source_id is the cited conversation's "
            "native id, and deliberately not as a session_links row: LinkType is the "
            "lineage vocabulary, and every reader of that table composes an inherited "
            "prefix from the edge. A lateral retrieval inherits nothing, so an edge for it "
            "needs its own relation rather than a member in that one. Per-citation "
            "prompt_text, alt, refs, reason, attribution and pub_date are unread.",
            "aggregate_result conserves the executed program as the construct's text -- the "
            "only place it survives, the calling `code` node's text being measured empty or "
            "an unrelated tool-call payload -- plus the run's id, clock, timeout and "
            "exception class as a chatgpt_code_interpreter_run event. Its stream text is "
            "stored verbatim only where it differs from the result node's own text, which "
            "is also where in_kernel_exception.traceback already lands.",
            "search_result_groups[].entries[] per-result pub_date, thumbnail_url, "
            "thumbnail_source and result_source are unread: ParsedWebConstruct carries no "
            "field for them and adding one is a derived-schema change. `attribution` "
            "repeats the group's `domain`, already the construct's group title.",
            "author.metadata.real_author sets material_origin to tool_result for its `tool:` "
            "values. The one measured `onboarding` value is left to the ordinary classifier: "
            "nothing in the record names what injected it.",
            "metadata.finish_details.type maps only `stop` and `max_tokens` onto "
            "messages.stop_reason; `interrupted`, `skipped` and `unknown` name no StopReason "
            "member. finish_details.stop_tokens is the sampler's stop-token list, not a "
            "terminal state.",
        ),
        semantic_reparse="reparse when ChatGPT document parsing fingerprints change",
        display_description="ChatGPT web exports (lab: OpenAI)",
        topology_capabilities=_no_topology_capabilities(origin),
        tool_outcome_unknown_reasons=frozenset(
            {ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}
        ),
    )
    return replace(
        spec,
        topology_capabilities=TopologyCapabilities(
            message_parent=TopologyCapability("carried", ("chatgpt.mapping.parent",)),
            message_branch_state=TopologyCapability(
                "carried",
                # ``children`` states sibling order where the export ships
                # it; the reduced export shape ships none, and the ordinal
                # among the siblings naming the same ``parent`` carries the
                # same sequence.
                ("chatgpt.mapping.children", "chatgpt.mapping.parent"),
            ),
            session_parent_target=_absent_topology("ChatGPT exports carry no session-parent target"),
            inheritance_branch_point=_absent_topology(
                "ChatGPT mapping ancestry is intra-session message topology, not cross-session inheritance"
            ),
            parent_dispatch=_absent_topology("ChatGPT exports carry no parent-dispatch identity"),
        ),
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
            "provider_session_id is derived from file identity and provider_message_id from response content.",
            "The export drops attachments/images by xAI's own documentation; only text turns are recoverable.",
        ),
        display_description="Grok account-data exports (lab: xAI)",
        topology_capabilities=_no_topology_capabilities(Origin.GROK_EXPORT),
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
    public_filter: bool = True,
    stream_parser_path: str | None = None,
    assembly_paths: tuple[str, ...] = (),
    fidelity_notes: tuple[str, ...] = (),
    assembly_spec_path: str | None = None,
    tool_outcome_unknown_reasons: frozenset[ToolResultUnknownReason] = frozenset(),
    artifact_rules: tuple[OriginArtifactRule, ...] = (),
    database_capability: DatabaseSourceCapability | None = None,
    frontier_kind: SourceFrontierKind = "exact-prefix",
    topology_capabilities: TopologyCapabilities,
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
        tool_outcome_unknown_reasons=tool_outcome_unknown_reasons,
        artifact_rules=artifact_rules,
        display_description=display_description,
        public_filter=public_filter,
        topology_capabilities=topology_capabilities,
        database_capability=database_capability,
        frontier_kind=frontier_kind,
    )


def _codex_spec() -> OriginSpec:
    spec = _executable_spec(
        Origin.CODEX_SESSION,
        provider=Provider.CODEX,
        tightness=50,
        discovery="Codex session JSONL admission plus live Codex SQLite state.",
        acquisition_modes=("session-jsonl", "thread-state-db", "goals-db", "memories-db", "memory-documents"),
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
        artifact_rules=(
            OriginArtifactRule(
                kind="agent_memory_document",
                # ``~/.codex/memories/**.md``. Codex keeps its memory
                # documents in a directory beside ``sessions/``, so the
                # ``memories/`` segment is the declaration; ``vendor_imports``,
                # ``skills`` and every other Markdown family under a Codex
                # install stays outside it (polylogue-rovf5).
                path_pattern=r"(?:^|/)memories/(?:[^/]+/)*[^/]+\.md$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="agent_memory_document",
                fidelity_note=(
                    "Codex memory documents are retained verbatim as source bytes and never parsed "
                    "into a session or promoted to a user assertion. ``memories_1.sqlite`` is Codex's "
                    "own derived memory state and remains a separate declared database member; these "
                    "Markdown documents are the harness-authored text itself. The install root is the "
                    "scope, so one basename under two installs is two retained objects."
                ),
                path_suffixes=(".md",),
                # Path-scoped: the Codex roots must not admit ``.md`` globally.
                watch_suffixes=(),
            ),
        ),
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
            "turn_context.user_instructions/.developer_instructions "
            "(acquired, polylogue-4r20i): both are re-declared on every turn, "
            "so the first value fills the session's own slot "
            "(sessions.instructions_text / the codex_agent_identity event) and "
            "a value distinct from every one already seen becomes its own "
            "codex_instructions_changed session_event carrying the message "
            "position it took effect on. Measured over 596 real rollout files: "
            "16 of the 120 carrying user_instructions declare more than one "
            "distinct value.",
            "compacted.replacement_history (acquired, polylogue-6ev92): the "
            "pre-compaction records Codex re-embeds on the compaction "
            "record. Measured over the 131 rollout files carrying it in a "
            "596-file sample (195,851 text values), 97.8% are already stored "
            "from the same file's live stream and 0.98% from an ancestor "
            "session whose prefix the file replays; the remaining 1.2% -- "
            "2,351 occurrences collapsing to 397 distinct values -- exists "
            "only here (injected environment/skills/AGENTS.md context and "
            "real user turns). Each text value is resolved against everything "
            "the parsed session retains and becomes its own "
            "codex_replacement_context session_event only when it is retained "
            "nowhere else; per-entry phase/ghost_commit/image annotation "
            "stays a bounded aggregate on the compaction event.",
            "event_msg.task_complete.last_agent_message (acquired, "
            "polylogue-6ev92): the turn's final assistant text repeated on "
            "the completion marker. Measured over 270 real rollout files, all "
            "1,565 occurrences were already stored from the same file's live "
            "stream, so the event records last_agent_message_chars plus "
            "last_agent_message_retained rather than the text -- and stores "
            "the text verbatim in the case the duplication does not hold.",
            "Codex lifecycle timing on task_started/task_complete and the other "
            "named event families (started_at/completed_at/duration_ms and "
            "observed elapsed/start/end aliases) is retained as bounded scalar "
            "event evidence. inter_agent_communication_metadata and "
            "token_usage_record are classified at their top-level dispatch and "
            "retain only named delegation/counter fields; opaque siblings are "
            "excluded rather than copied as a wire-payload dump.",
            "event_msg.memory_citation (measured negative, polylogue-cgfy "
            "codex lane): observed null on every sampled record across "
            "~3,200 real session files -- a constant, not an unread signal; "
            "not acquired.",
            "response_item/event_msg phase and "
            "internal_chat_message_metadata_passthrough.turn_id (acquired, "
            "polylogue-q0vka): phase and turn-correlation evidence is retained "
            "on the normalized session_events route, anchored to the source "
            "message where available. Existing metadata.turn_id remains the "
            "compatibility authority; passthrough/direct values are retained "
            "and conflicts are explicit, so no turn identity is guessed. "
            "user_message.local_images/text_elements retain bounded provider "
            "references and placeholder ranges with acquired_bytes/path and "
            "content-policy markers; local paths never imply acquired bytes.",
        ),
        topology_capabilities=_no_topology_capabilities(Origin.CODEX_SESSION),
        tool_outcome_unknown_reasons=frozenset(
            {
                ToolResultUnknownReason.NOT_REPORTED,
                ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT,
                ToolResultUnknownReason.SOURCE_TRUNCATED,
            }
        ),
        database_capability=DatabaseSourceCapability(
            snapshot_method="logical_export",
            consistency_fence="one SQLite read transaction over a mode=ro URI",
            revision_identity=(
                "sha256 over the canonical logical export of the member's declared tables (sqlite_logical_revision)"
            ),
            raw_id_strategy="codex state raw-id domain + absolute source path + logical revision",
            members=(
                DatabaseMemberRule(
                    "state_5.sqlite",
                    "acquire",
                    "thread_state",
                    "thread state is retained as one logical export, with table-level dispositions below",
                    logical_tables=(
                        "threads",
                        "thread_spawn_edges",
                        "thread_artifacts",
                        "thread_dynamic_tools",
                        "thread_sections",
                        "projects",
                        "project_roots",
                    ),
                    consumer="polylogue/sources/codex_state_evidence.py:record_codex_state_snapshot_terminal",
                    table_rules=(
                        DatabaseTableRule(
                            "threads",
                            "retained-and-consumed",
                            "Curated titles and orchestration metadata feed the thread-state projection.",
                        ),
                        DatabaseTableRule(
                            "thread_spawn_edges",
                            "retained-and-consumed",
                            "Codex orchestration parent/child edges feed the thread-state projection.",
                        ),
                        DatabaseTableRule(
                            "thread_artifacts",
                            "retained-for-later-consumption",
                            "Artifact identity and payload are thread evidence with no typed projection yet.",
                        ),
                        DatabaseTableRule(
                            "thread_dynamic_tools",
                            "retained-for-later-consumption",
                            "Dynamic tool descriptions and schemas are thread evidence with no typed projection yet.",
                        ),
                        DatabaseTableRule(
                            "thread_sections",
                            "retained-for-later-consumption",
                            "Thread section definitions contextualize the retained thread section references.",
                        ),
                        DatabaseTableRule(
                            "projects",
                            "retained-for-later-consumption",
                            "Project identity and metadata contextualize retained thread project references.",
                        ),
                        DatabaseTableRule(
                            "project_roots",
                            "retained-for-later-consumption",
                            "Project roots contextualize retained project references without a typed projection yet.",
                        ),
                        DatabaseTableRule(
                            "_sqlx_migrations",
                            "deliberately-excluded",
                            "Database migration bookkeeping is not session evidence.",
                        ),
                        DatabaseTableRule(
                            "backfill_state",
                            "deliberately-excluded",
                            "Resumable backfill cursor state is operational bookkeeping, not session evidence.",
                        ),
                        DatabaseTableRule(
                            "external_agent_config_imports",
                            "deliberately-excluded",
                            "External-agent configuration import status is operational configuration, not session evidence.",
                        ),
                        DatabaseTableRule(
                            "project_idempotency_keys",
                            "deliberately-excluded",
                            "Project request deduplication keys are operational state, not session evidence.",
                        ),
                        DatabaseTableRule(
                            "remote_control_enrollments",
                            "deliberately-excluded",
                            "Remote-control enrollment configuration can carry connection details and is not session evidence.",
                        ),
                        DatabaseTableRule(
                            "rollout_migration_skipped_rollouts",
                            "deliberately-excluded",
                            "Rollout migration skip bookkeeping duplicates rollout discovery operational state.",
                        ),
                        DatabaseTableRule(
                            "rollout_migration_state",
                            "deliberately-excluded",
                            "Rollout migration cursors are operational bookkeeping, not session evidence.",
                        ),
                    ),
                ),
                DatabaseMemberRule(
                    "goals_1.sqlite",
                    "acquire-partial",
                    "goals",
                    "goal intent is retained as durable raw evidence",
                    logical_tables=("thread_goals", "thread_goal_continuation_deferrals"),
                ),
                DatabaseMemberRule(
                    "memories_1.sqlite",
                    "acquire-partial",
                    "memories",
                    "memory state is retained as durable raw evidence",
                    logical_tables=("stage1_outputs", "jobs"),
                ),
                DatabaseMemberRule("logs_2.sqlite", "out-of-scope", "logs", "runtime tracing is not session evidence"),
                DatabaseMemberRule(
                    "codex-dev.db", "out-of-scope", "automation", "automation scheduling is not session evidence"
                ),
                DatabaseMemberRule(
                    "thread_history_1.sqlite",
                    "out-of-scope",
                    "thread_history",
                    "a measured projection of the rollout JSONL the archive already acquires",
                ),
                DatabaseMemberRule(
                    "queue_1.sqlite",
                    "out-of-scope",
                    "queue",
                    "input queued for submission, not evidence of a session that happened",
                ),
            ),
            full_snapshot_per_revision=True,
            snapshot_lineage_policy=(
                "one export blob per logical revision; supersession and dedup receipts govern lineage retention"
            ),
        ),
    )
    carried = TopologyCapability("carried", ("codex_state.thread.parent_thread_id",))
    return replace(
        spec,
        topology_capabilities=TopologyCapabilities(
            message_parent=_absent_topology("Codex message records carry no parent-message field"),
            message_branch_state=_absent_topology("Codex message records carry no branch-state field"),
            session_parent_target=carried,
            inheritance_branch_point=_absent_topology("Codex wire carries no inheritance boundary"),
            parent_dispatch=_absent_topology("Codex wire carries no parent-dispatch identity"),
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
        # Gemini CLI rewrites a chat's checkpoint whole on every save: the
        # later payload is not a byte-prefix extension of the earlier one
        # (``lastUpdated`` sits mid-object), so the cohort has no byte
        # revision chain to accept a head from.
        frontier_kind="whole-snapshot",
        topology_capabilities=_no_topology_capabilities(Origin.GEMINI_CLI_SESSION),
        tool_outcome_unknown_reasons=frozenset(
            {ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}
        ),
        artifact_rules=(
            OriginArtifactRule(
                # Same family as Claude Code's ``tool-results/`` overflow, and
                # deliberately the same kind: an opaque tool output persisted
                # beside the transcript, joined back to its owning block by
                # ``sources/live/gemini_tool_output_sidecars.py``, never
                # independent conversation content. Declaring it is what makes
                # the bytes retained evidence rather than a live filesystem
                # lookup at parse time (polylogue-cq1ql): without this rule the
                # walk admits no ``tool-outputs/`` file at all, so a reparse
                # after the source tree moves keeps only the masked envelope's
                # first 8,000 and last 32,000 characters.
                kind="tool_result_sidecar",
                path_pattern=r"(?:^|/)tool-outputs/session-[^/]+/[^/]+$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="tool_output_overflow",
                fidelity_note=(
                    "Gemini CLI tool-output overflow content persisted verbatim under "
                    "tool-outputs/session-<sessionId>/; never independently parsed -- "
                    "sources/live/gemini_tool_output_sidecars.py joins it to its owning tool call by "
                    "filename stem, with the masking envelope's 'For full output see:' pointer as the "
                    "reverse index. A tool's own output can reproduce any document shape, so the path "
                    "rule is the gate, not content heuristics."
                ),
                # Gemini CLI writes these with the extension of whatever the
                # tool produced; the path rule admits all of them, and no
                # suffix is projected onto the watched root because the
                # directory shape is the whole admission evidence.
                path_suffixes=(".txt", ".json", ".md", ".log", ""),
                watch_suffixes=(),
            ),
        ),
        fidelity_notes=(
            "local_agent.py's _status_is_error guessed success-outcome set is "
            "registered as a DroppedValueVocabulary (polylogue-2qx) against "
            "the committed schema's messages[].toolCalls[].status leaf -- "
            "see DROPPED_VALUE_VOCABULARIES/check_dropped_value_vocabularies "
            "in this module.",
        ),
    )


def _hermes_spec() -> OriginSpec:
    spec = _executable_spec(
        Origin.HERMES_SESSION,
        provider=Provider.HERMES,
        tightness=20,
        discovery=(
            "Hermes state database plus NeMo Relay ATIF/ATOF observer and coding-verification-ledger admission."
        ),
        acquisition_modes=("state-db", "atif-spans", "atof-jsonl", "verification-evidence-db", "session-snapshot"),
        parser_paths=(
            "polylogue/sources/parsers/local_agent.py",
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
        topology_capabilities=_no_topology_capabilities(Origin.HERMES_SESSION),
        tool_outcome_unknown_reasons=frozenset(
            {
                ToolResultUnknownReason.NOT_REPORTED,
                ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT,
                ToolResultUnknownReason.SOURCE_TRUNCATED,
            }
        ),
        database_capability=DatabaseSourceCapability(
            snapshot_method="logical_export",
            consistency_fence="one SQLite read transaction over a mode=ro URI",
            revision_identity=(
                "sha256 over the canonical logical export of the member's declared tables (sqlite_logical_revision)"
            ),
            raw_id_strategy="Hermes profile raw-id domain + profile path + member filename + source index + logical revision",
            members=(
                DatabaseMemberRule(
                    "state.db",
                    "acquire",
                    "state",
                    "conversation state is the authoritative Hermes session source",
                    logical_tables=("schema_version", "sessions", "messages"),
                    consumer="polylogue/sources/parsers/hermes_state.py:parse_state_db",
                ),
                DatabaseMemberRule(
                    "verification_evidence.db",
                    "acquire",
                    "verification",
                    "verification evidence is a declared observer source",
                    logical_tables=("meta", "verification_events", "verification_state"),
                    consumer="polylogue/sources/parsers/hermes_verification.py:parse_verification_evidence_db",
                ),
            ),
            full_snapshot_per_revision=True,
            snapshot_lineage_policy=(
                "one export blob per logical revision; supersession and dedup receipts govern lineage retention"
            ),
        ),
    )
    carried = TopologyCapability(
        "carried", ("hermes_state.sessions.parent_session_id", "hermes ATIF parent_session_id")
    )
    return replace(
        spec,
        topology_capabilities=TopologyCapabilities(
            message_parent=_absent_topology("Hermes message records carry no parent-message field"),
            message_branch_state=TopologyCapability(
                "positive-derived",
                ("hermes_state._branch_type",),
                "branch type is derived only when parent state is present",
            ),
            session_parent_target=carried,
            inheritance_branch_point=_absent_topology("Hermes wire carries no inheritance boundary"),
            parent_dispatch=_absent_topology("Hermes wire carries no parent-dispatch identity"),
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
        display_description="Antigravity language-server and trajectory SQLite conversations",
        fidelity_notes=(
            "trajectory_meta and steps SQLite stores are acquired as one consistent logical export, then parsed by "
            "the same canonical session writer; unknown step formats remain retained evidence.",
        ),
        artifact_rules=(
            OriginArtifactRule(
                kind="session_document",
                path_pattern=r"(?:^|/)conversations/(?:[^/]+/)*[^/]+\.pb$",
                parse_policy="session",
                parser_path="polylogue/sources/parsers/antigravity.py:iter_language_server_exports",
                coverage_role="conversation_protobuf",
                fidelity_note="Opaque conversation protobufs are converted only by Antigravity's language server.",
                path_suffixes=(".pb",),
                watch_suffixes=(".pb", ".db", ".sqlite", ".sqlite3"),
            ),
            OriginArtifactRule(
                kind="agent_sidecar_meta",
                path_pattern=r"(?:^|/)brain/(?:[^/]+/)*[^/]+\.metadata\.json$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="brain_metadata_sidecar",
                fidelity_note="Brain metadata is retained as typed artifact evidence and never creates a session.",
                path_suffixes=(".metadata.json",),
            ),
            OriginArtifactRule(
                kind="metadata_document",
                path_pattern=r"(?:^|/)brain/(?:[^/]+/)*[^/]+\.md$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="brain_document",
                fidelity_note="Brain documents are retained as typed artifacts and never create a session.",
                path_suffixes=(".md",),
            ),
        ),
        topology_capabilities=TopologyCapabilities(
            message_parent=_absent_topology("Antigravity steps carry no reviewed message-parent identity"),
            message_branch_state=_absent_topology("Antigravity steps carry no reviewed branch-state identity"),
            session_parent_target=TopologyCapability(
                "carried",
                ("trajectory parent_references.parent_id", "trajectory parent_references.cascade_id"),
            ),
            inheritance_branch_point=_absent_topology("Antigravity trajectory stores carry no branch boundary"),
            parent_dispatch=_absent_topology("Antigravity trajectory stores carry no parent-dispatch identity"),
        ),
        tool_outcome_unknown_reasons=frozenset({ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}),
    )


def _beads_spec() -> OriginSpec:
    origin = Origin.BEADS_ISSUE
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin,
            lifecycle="reserved",
            discovery="Reserved Beads issue origin vocabulary; no session admission.",
        ),
        lifecycle="reserved",
        acquisition_modes=("reserved",),
        provider_wires=(Provider.BEADS,),
        collision_policy=None,
        detector_tightness=None,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
        fixture_paths=("tests/unit/sources/test_origin_specs.py",),
        coverage_refs=("origin:beads-issue:reserved",),
        fidelity_notes=(
            "The durable origin and provider tokens remain available for historical CHECK vocabulary, "
            "but no Beads issue record is admitted as a session.",
        ),
        semantic_reparse="no parser; retain the reserved vocabulary until a Beads session adapter is admitted",
        display_description="Reserved Beads issue origin (not admitted)",
        public_filter=False,
        topology_capabilities=_no_topology_capabilities(Origin.BEADS_ISSUE),
    )


def _claude_ai_spec() -> OriginSpec:
    spec = _executable_spec(
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
        fidelity_notes=(
            "chat_messages[].stop_reason lands on messages.stop_reason only for the tokens "
            "that name a StopReason member. The web surface also emits user_canceled, "
            "error, conversation_length_limit and tool_use_limit, which have no equivalent "
            "and leave the column NULL.",
        ),
        topology_capabilities=_no_topology_capabilities(Origin.CLAUDE_AI_EXPORT),
        tool_outcome_unknown_reasons=frozenset(
            {ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}
        ),
    )
    return replace(
        spec,
        topology_capabilities=TopologyCapabilities(
            message_parent=TopologyCapability(
                "carried", ("claude.common._message_parent_id -> ParsedMessage.parent_message_provider_id",)
            ),
            message_branch_state=TopologyCapability(
                "positive-derived",
                ("claude.common.normalize_chat_messages branch_index/variant_index/is_active_path/is_active_leaf",),
            ),
            session_parent_target=_absent_topology("Claude AI export has no session-parent target"),
            inheritance_branch_point=_absent_topology("Claude AI export has no inheritance boundary"),
            parent_dispatch=_absent_topology("Claude AI export has no parent dispatch identity"),
        ),
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
        topology_capabilities=_no_topology_capabilities(Origin.CLAUDE_DESIGN_SESSION),
        tool_outcome_unknown_reasons=frozenset(
            {ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}
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
        artifact_rules=(
            OriginArtifactRule(
                kind="metadata_document",
                # AI Studio writes the applet access log into the same Drive
                # folder as the conversation exports. Its ``{"applets": [...]}`
                # body carries no turn, so content classification can only
                # reach ``ArtifactKind.UNKNOWN`` -- a kind the coverage gate
                # refuses by construction. The path is the declaration.
                path_pattern=r"(?:^|/)applet_access_history\.json$",
                parse_policy="raw-only",
                parser_path=None,
                coverage_role="applet_access_log",
                fidelity_note=(
                    "AI Studio applet access log retained as acquired evidence and never parsed as a "
                    "session: it records which applets an account opened, not a conversation."
                ),
                path_suffixes=(".json",),
                # One named file, not a suffix family: enumeration of the
                # Drive root stays governed by ``path_pattern``.
                watch_suffixes=(),
            ),
        ),
        assembly_paths=("polylogue/sources/dispatch.py:_lower_payload_specs",),
        fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
        coverage_refs=("origin:aistudio-drive:admitted",),
        fidelity_notes=(
            "Provider reverse mapping remains intentionally non-injective.",
            "runSettings (model/temperature/topP/topK/... ) is read for the model_config session_event; "
            "the settings bag itself is not projected onto the session row.",
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
        tool_outcome_unknown_reasons=frozenset(
            {ToolResultUnknownReason.NOT_REPORTED, ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT}
        ),
        topology_capabilities=TopologyCapabilities(
            message_parent=TopologyCapability(
                "carried",
                (
                    "drive._branch_parent_message_provider_id/_branch_child_parent_map -> ParsedMessage.parent_message_provider_id; only id/messageId are local message evidence",
                ),
            ),
            message_branch_state=TopologyCapability(
                "positive-derived",
                ("drive.parse_chunked_prompt active path and fill_linear_parent_chain",),
            ),
            session_parent_target=TopologyCapability(
                "carried",
                ("drive.branchParent.promptId -> ParsedSession.parent_session_provider_id -> session_links",),
            ),
            inheritance_branch_point=TopologyCapability(
                "positive-derived",
                (
                    "Drive prompt-grain parents retain branch_point_message_id=NULL with typed unresolved evidence when no local message id is asserted",
                ),
            ),
            parent_dispatch=_absent_topology("AI Studio and Drive exports have no parent dispatch identity"),
        ),
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
        public_filter=False,
        topology_capabilities=_no_topology_capabilities(Origin.UNKNOWN_EXPORT),
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
            "provider-package:beads-issue/reserved@v1",
            "reserved",
            Provider.BEADS,
            "reserved",
            detector_paths=(),
            raw_model_paths=(),
            parser_paths=(),
            normalizer_paths=(),
            fixture_paths=("tests/unit/sources/test_origin_specs.py",),
            schema_paths=(),
            docs_paths=("docs/provider-origin-identity.md",),
            caveats=(
                "The reserved origin has durable vocabulary and declaration evidence only; no Beads session "
                "wire format, parser, detector, or schema package is admitted.",
            ),
        ),
    ),
    Origin.GROK_EXPORT: (
        _completeness_mode(
            "provider-package:grok-export/export-json@v1",
            "export-json",
            Provider.GROK,
            "accepted",
            detector_paths=("polylogue/sources/parsers/grok.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/grok.py",),
            parser_paths=("polylogue/sources/parsers/grok.py",),
            normalizer_paths=("polylogue/sources/parsers/grok.py",),
            fixture_paths=(
                "tests/unit/sources/parsers/test_grok.py",
                "tests/unit/sources/parsers/test_origin_regression_pack.py",
            ),
            schema_paths=("polylogue/schemas/providers/grok/catalog.json",),
            docs_paths=("docs/provider-origin-identity.md", "docs/architecture.md"),
            caveats=(
                "The package is a structural parser contract; broader export sampling remains separately tracked.",
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
            "accepted",
            detector_paths=("polylogue/sources/parsers/claude/ai_parser.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
            fixture_paths=("tests/unit/sources/test_parsers_claude_design.py",),
            schema_paths=("polylogue/schemas/providers/claude-design/catalog.json",),
            docs_paths=("docs/provider-origin-identity.md",),
            caveats=(
                "The package is a structural parser contract; broader export sampling remains separately tracked.",
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


_ALL_BROWSER_CAPTURE_PROVIDERS = tuple(Provider)

_ORIGIN_DETECTOR_BINDINGS: dict[Origin, tuple[DetectorBinding, ...]] = {
    Origin.CLAUDE_CODE_SESSION: (
        DetectorBinding(
            "claude-code-record-envelope",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_claude_code_record",
            0,
            "claude.looks_like_code (envelope marker, #3428)",
            fixed_provider=Provider.CLAUDE_CODE,
        ),
        DetectorBinding(
            "claude-code-record-stream",
            DetectionMode.SEQUENCE_RECORD_STREAM,
            "polylogue.sources.dispatch:_looks_like_claude_code_stream",
            0,
            "claude.looks_like_code (record stream envelope markers)",
            mode_rank=0,
            fixed_provider=Provider.CLAUDE_CODE,
        ),
    ),
    Origin.CODEX_SESSION: (
        DetectorBinding(
            "codex-record-pydantic",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_codex_record",
            0,
            "codex.looks_like (pydantic record validation)",
            fixed_provider=Provider.CODEX,
        ),
        DetectorBinding(
            "codex-record-stream",
            DetectionMode.SEQUENCE_RECORD_STREAM,
            "polylogue.sources.dispatch:_looks_like_codex_stream",
            0,
            "codex.looks_like (pydantic record stream validation)",
            mode_rank=1,
            fixed_provider=Provider.CODEX,
        ),
    ),
    Origin.GEMINI_CLI_SESSION: (
        DetectorBinding(
            "gemini-cli-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_gemini_cli_record",
            0,
            "local_agent.looks_like_gemini_cli",
            fixed_provider=Provider.GEMINI_CLI,
        ),
        DetectorBinding(
            "gemini-cli-sequence-stub",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_gemini_cli_sequence_stub",
            0,
            "local_agent.looks_like_gemini_cli (stub record)",
            fixed_provider=Provider.GEMINI_CLI,
        ),
    ),
    Origin.HERMES_SESSION: (
        DetectorBinding(
            "hermes-state-db-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_hermes_state_record",
            0,
            "hermes_state.looks_like_state_db_payload",
            fixed_provider=Provider.HERMES,
        ),
        DetectorBinding(
            "hermes-verification-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_hermes_verification_record",
            1,
            "hermes_verification.looks_like_verification_evidence_db_payload",
            fixed_provider=Provider.HERMES,
        ),
        DetectorBinding(
            "hermes-atif-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_hermes_atif_record",
            2,
            "hermes_spans.looks_like_atif_payload",
            fixed_provider=Provider.HERMES,
        ),
        DetectorBinding(
            "hermes-atof-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_hermes_atof_record",
            3,
            "hermes_spans.looks_like_atof_payload",
            fixed_provider=Provider.HERMES,
        ),
        DetectorBinding(
            "hermes-local-agent-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_hermes_local_agent_record",
            4,
            "local_agent.looks_like_hermes",
            fixed_provider=Provider.HERMES,
        ),
        DetectorBinding(
            "hermes-atof-sequence",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_hermes_atof_sequence",
            0,
            "hermes_spans.looks_like_atof_payload (sequence[0])",
            fixed_provider=Provider.HERMES,
        ),
    ),
    Origin.ANTIGRAVITY_SESSION: (
        DetectorBinding(
            "antigravity-markdown-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_antigravity_markdown_record",
            0,
            "antigravity.looks_like_markdown_export",
            fixed_provider=Provider.ANTIGRAVITY,
        ),
    ),
    Origin.CHATGPT_EXPORT: (
        DetectorBinding(
            "chatgpt-record-fragment",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_chatgpt_fragment_record",
            0,
            "chatgpt.looks_like_fragment (mapping node shape)",
            fixed_provider=Provider.CHATGPT,
        ),
        DetectorBinding(
            "chatgpt-record-shared-decode",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_chatgpt_shared_decode_record",
            1,
            "chatgpt.looks_like_shared_decode (shared-page stream decode)",
            fixed_provider=Provider.CHATGPT,
        ),
        DetectorBinding(
            "chatgpt-sequence-document",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_chatgpt_sequence_document",
            0,
            "chatgpt.looks_like (sequence[0] whole-document)",
            fixed_provider=Provider.CHATGPT,
        ),
    ),
    Origin.CLAUDE_AI_EXPORT: (
        DetectorBinding(
            "claude-ai-record-memories",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_claude_memories_record",
            0,
            "claude.looks_like_claude_memories",
            fixed_provider=Provider.CLAUDE_AI,
        ),
        DetectorBinding(
            "claude-ai-record-project",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_claude_project_record",
            1,
            "claude.looks_like_claude_project",
            fixed_provider=Provider.CLAUDE_AI,
        ),
        DetectorBinding(
            "claude-ai-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_claude_ai_record",
            2,
            "claude.looks_like_ai (non-empty plausible chat_messages)",
            fixed_provider=Provider.CLAUDE_AI,
        ),
        DetectorBinding(
            "claude-ai-sequence-chat-messages",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_claude_ai_sequence",
            0,
            "sequence[0] chat_messages dict-key present",
            fixed_provider=Provider.CLAUDE_AI,
        ),
        DetectorBinding(
            "claude-ai-sequence-memories",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_claude_memories_sequence",
            1,
            "claude.looks_like_claude_memories (sequence[0])",
            fixed_provider=Provider.CLAUDE_AI,
        ),
        DetectorBinding(
            "claude-ai-sequence-project",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_claude_project_sequence",
            2,
            "claude.looks_like_claude_project (sequence[0])",
            fixed_provider=Provider.CLAUDE_AI,
        ),
    ),
    Origin.CLAUDE_DESIGN_SESSION: (
        DetectorBinding(
            "claude-design-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_claude_design_record",
            0,
            "claude.looks_like_claude_design",
            fixed_provider=Provider.CLAUDE_DESIGN,
        ),
        DetectorBinding(
            "claude-design-sequence",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_claude_design_sequence",
            0,
            "claude.looks_like_claude_design (sequence[0])",
            fixed_provider=Provider.CLAUDE_DESIGN,
        ),
    ),
    Origin.GROK_EXPORT: (
        DetectorBinding(
            "grok-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_grok_record",
            0,
            "grok.looks_like_export",
            fixed_provider=Provider.GROK,
        ),
        DetectorBinding(
            "grok-sequence",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_grok_sequence",
            0,
            "grok.looks_like_export (sequence[0])",
            fixed_provider=Provider.GROK,
        ),
    ),
    Origin.AISTUDIO_DRIVE: (
        DetectorBinding(
            "aistudio-drive-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_gemini_mapping_record",
            0,
            "drive.looks_like (chunkedPrompt/chunks)",
            fixed_provider=Provider.GEMINI,
        ),
        DetectorBinding(
            "aistudio-drive-sequence",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_gemini_mapping_sequence",
            0,
            "drive.looks_like (sequence[0])",
            fixed_provider=Provider.GEMINI,
        ),
    ),
    Origin.UNKNOWN_EXPORT: (
        DetectorBinding(
            "browser-capture-record",
            DetectionMode.RECORD,
            "polylogue.sources.dispatch:_looks_like_browser_capture_record",
            0,
            "browser_capture.looks_like",
            dynamic_provider_path="polylogue.sources.dispatch:_browser_capture_provider",
            dynamic_provider_allowlist=_ALL_BROWSER_CAPTURE_PROVIDERS,
        ),
        DetectorBinding(
            "browser-capture-sequence",
            DetectionMode.SEQUENCE_DOCUMENT,
            "polylogue.sources.dispatch:_looks_like_browser_capture_sequence",
            0,
            "sequence[0] browser_capture.looks_like -> browser_capture.looks_like",
            dynamic_provider_path="polylogue.sources.dispatch:_browser_capture_sequence_provider",
            dynamic_provider_allowlist=_ALL_BROWSER_CAPTURE_PROVIDERS,
        ),
    ),
}


def _with_declaration_fields(spec: OriginSpec) -> OriginSpec:
    return replace(
        spec,
        completeness_modes=_ORIGIN_COMPLETENESS_MODES[spec.origin],
        detector_bindings=_ORIGIN_DETECTOR_BINDINGS.get(spec.origin, ()),
    )


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
    ORIGIN_SPEC_REGISTRY.register(_with_declaration_fields(_spec))
ORIGIN_SPECS = ORIGIN_SPEC_REGISTRY.specs()
_ORIGIN_SPECS_BY_ORIGIN = {spec.origin: spec for spec in ORIGIN_SPECS}


def public_origin_tokens(specs: Sequence[OriginSpec] | None = None) -> tuple[str, ...]:
    """Return the declared public-origin filter/completion vocabulary.

    ``Origin`` remains the closed public identity enum, while ``public_filter``
    distinguishes query choices from compatibility/evidence-only origins.  A
    caller may pass a synthetic spec tuple in tests or during rendering; this
    keeps projections tied to declaration metadata rather than a copied list.
    """
    by_origin = {spec.origin: spec for spec in (ORIGIN_SPECS if specs is None else specs)}
    return tuple(
        origin.value for origin in Origin if (spec := by_origin.get(origin)) is not None and spec.public_filter
    )


def public_origin_meanings(
    specs: Sequence[OriginSpec] | None = None, *, include_non_public: bool = False
) -> tuple[tuple[str, str], ...]:
    """Return public ``(Origin token, operator-facing description)`` rows.

    ``include_non_public`` is reserved for the agent manual, which documents
    the complete closed enum; completion and filter callers use the default
    public-only projection.
    """
    by_origin = {spec.origin: spec for spec in (ORIGIN_SPECS if specs is None else specs)}
    return tuple(
        (origin.value, by_origin[origin].display_description)
        for origin in Origin
        if (spec := by_origin.get(origin)) is not None and (include_non_public or spec.public_filter)
    )


def public_origin_descriptions(specs: Sequence[OriginSpec] | None = None) -> dict[str, str]:
    """Return public description rows for completion/help surfaces."""
    return dict(public_origin_meanings(specs))


@lru_cache(maxsize=8)
def _compiled_detector_registry(specs: tuple[OriginSpec, ...]) -> CompiledDetectorRegistry:
    return compile_detector_registry(specs)


def detector_registry() -> CompiledDetectorRegistry:
    """Return the one validated executable detector registry for current OriginSpecs."""
    return _compiled_detector_registry(ORIGIN_SPECS)


def origin_specs() -> tuple[OriginSpec, ...]:
    """Return the stable public-origin admission projection."""

    return ORIGIN_SPECS


def tool_outcome_unknown_reasons_for_origin(origin: Origin) -> frozenset[ToolResultUnknownReason]:
    """Return the unknown-outcome reasons this origin's parsers can derive."""

    return _ORIGIN_SPECS_BY_ORIGIN[origin].tool_outcome_unknown_reasons


def topology_capability_census(
    specs: Sequence[OriginSpec] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    """Project the complete topology capability census from ``OriginSpec``.

    The optional sequence is used by declaration-law tests to prove that
    omissions and malformed capability cells fail closed.
    """
    declared_specs = ORIGIN_SPECS if specs is None else tuple(specs)
    expected_dimensions = {
        "message_parent",
        "message_branch_state",
        "session_parent_target",
        "inheritance_branch_point",
        "parent_dispatch",
    }
    origins = tuple(spec.origin for spec in declared_specs)
    if len(set(origins)) != len(origins) or set(origins) != set(Origin):
        raise ValueError("topology capability census must cover every current Origin exactly once")
    for spec in declared_specs:
        capabilities = spec.topology_capabilities.as_dict()
        if set(capabilities) != expected_dimensions:
            raise ValueError(f"{spec.origin.value}: topology capability census is incomplete")
        for dimension, capability in capabilities.items():
            if capability.state not in {"carried", "positive-derived", "structurally-absent"}:
                raise ValueError(f"{spec.origin.value}.{dimension}: topology capability state is not complete")
            if not capability.evidence:
                raise ValueError(f"{spec.origin.value}.{dimension}: topology capability lacks evidence")
            if capability.state == "structurally-absent" and not capability.reason:
                raise ValueError(f"{spec.origin.value}.{dimension}: structural absence lacks a reason")
    return {
        spec.origin.value: {
            name: {
                "state": capability.state,
                "evidence": capability.evidence,
                "reason": capability.reason,
            }
            for name, capability in spec.topology_capabilities.as_dict().items()
        }
        for spec in declared_specs
    }


def parser_fingerprint_for_origin(origin: Origin | str) -> str:
    """Return the current parser fingerprint for one normalized archive origin."""
    normalized = Origin.from_string(origin)
    return _ORIGIN_SPECS_BY_ORIGIN[normalized].parser_fingerprint()


def frontier_kind_for_origin(origin: Origin | str) -> SourceFrontierKind:
    """Return the declared source continuation law for one origin."""
    return _ORIGIN_SPECS_BY_ORIGIN[Origin.from_string(origin)].frontier_kind


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
    "derived_identity_source_closure",
    "in_derived_identity_closure",
    "ORIGIN_SPECS",
    "frontier_kind_for_origin",
    "ORIGIN_SPEC_REGISTRY",
    "ArtifactParsePolicy",
    "DroppedValueVocabulary",
    "OriginArtifactRule",
    "DatabaseMemberRule",
    "DatabaseTableRule",
    "DatabaseSourceCapability",
    "DatabaseMemberDisposition",
    "DatabaseTableDisposition",
    "SourceClassRecognition",
    "SourceClass",
    "OriginLifecycle",
    "OriginSpec",
    "OriginSpecDiagnostic",
    "OriginSpecRegistry",
    "TopologyCapability",
    "TopologyCapabilities",
    "DetectorBinding",
    "check_dropped_value_vocabularies",
    "origin_specs",
    "tool_outcome_unknown_reasons_for_origin",
    "DatabaseMemberBinding",
    "database_capability_for_provider",
    "database_member_for_filename",
    "validate_database_member_filenames",
    "topology_capability_census",
    "public_origin_descriptions",
    "public_origin_meanings",
    "public_origin_tokens",
    "artifact_rule_for_path",
    "path_declaration_refuses_session",
    "artifact_suffixes_for_provider",
    "recognize_source_class",
    "schema_observed_leaf_values",
    "undeclared_schema_values",
    "lowering_fingerprint",
    "detector_registry",
    "materializer_fingerprint",
    "replay_routing_fingerprint",
    "parser_fingerprint_for_origin",
    "validate_assembly_spec_parity",
    "validate_stream_parser_parity",
]
