"""Parser and local export client for Antigravity session state."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import socket
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from glob import glob
from pathlib import Path
from types import TracebackType
from typing import Protocol
from urllib.request import Request, urlopen

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.core.json import JSONDocument, dumps_bytes, loads

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    human_authored_override,
    mark_last_occurrence_as_active_leaf,
    parser_admission,
    synthetic_message_id,
)

_SEARCH_ENDPOINT = "/exa.language_server_pb.LanguageServerService/SearchConversations"
_MARKDOWN_ENDPOINT = "/exa.language_server_pb.LanguageServerService/ConvertTrajectoryToMarkdown"
_SECTION_RE = re.compile(r"^### (?P<title>User Input|Planner Response)\s*$", re.MULTILINE)


class AntigravityExportError(RuntimeError):
    """Raised when Antigravity's local export surface cannot be queried."""


class AntigravityBinaryUnavailableError(AntigravityExportError):
    """Raised when the Antigravity language-server binary is not installed.

    This is a source coverage blocker for manifested conversation protobufs.
    Brain artifacts remain independently admitted as raw-only evidence.
    """


class AntigravityPartialExportError(AntigravityExportError):
    """Raised when the language-server export aborts mid-iteration.

    Distinct from a binary-absent condition: some sessions were already
    obtained before the failure, so the remainder is genuinely at risk of being
    dropped. Carries obtained-vs-expected counts so callers can surface the loss
    instead of silently truncating.
    """

    def __init__(self, message: str, *, obtained: int, expected: int) -> None:
        self.obtained = obtained
        self.expected = expected
        super().__init__(f"{message} (obtained {obtained} of {expected} sessions)")


class AntigravitySourceMutationError(AntigravityExportError):
    """Raised when a source item changes while read-only evidence is captured."""


class AntigravitySourceRole(StrEnum):
    """The admission role of one item below Antigravity's source root."""

    CONVERSATION_PROTOBUF = "conversation_protobuf"
    BRAIN_DOCUMENT = "brain_document"
    METADATA_SIDECAR = "metadata_sidecar"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AntigravitySourceClassification:
    """Positive source-role evidence shared by batch and resident routes."""

    role: AntigravitySourceRole
    parse_as_session: bool
    artifact_kind: ArtifactKind
    reason: str


@dataclass(frozen=True, slots=True)
class AntigravitySourceItem:
    """One immutable source-census item and its positive admission role."""

    path: Path
    relative_path: str
    classification: AntigravitySourceClassification
    size_bytes: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class AntigravitySourceCensus:
    """Complete, read-only accounting for one Antigravity source root."""

    root: Path
    items: tuple[AntigravitySourceItem, ...]

    @property
    def counts(self) -> dict[AntigravitySourceRole, int]:
        counts = Counter(item.classification.role for item in self.items)
        return {role: counts.get(role, 0) for role in AntigravitySourceRole}

    @property
    def unknown_count(self) -> int:
        return self.counts[AntigravitySourceRole.UNKNOWN]

    @property
    def unexplained_items(self) -> tuple[Path, ...]:
        roles = frozenset(AntigravitySourceRole)
        return tuple(item.path for item in self.items if item.classification.role not in roles)

    def assert_conserved(self) -> None:
        if self.unexplained_items:
            raise ValueError("Antigravity source census contains unexplained items")
        if sum(self.counts.values()) != len(self.items):
            raise ValueError("Antigravity source census does not conserve its item denominator")


def census_source(root: Path) -> AntigravitySourceCensus:
    """Capture every regular file below ``root`` and assign one source role.

    This is preparation evidence only. It does not open the archive or blob
    store, and a change during hashing is a visible source failure.
    """
    root = root.expanduser()
    items: list[AntigravitySourceItem] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if not path.is_file():
                continue
            before = path.stat()
            digest = _file_digest(path)
            after = path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise AntigravitySourceMutationError(f"Antigravity source changed during census: {path}")
            items.append(
                AntigravitySourceItem(
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    classification=classify_source_path(path),
                    size_bytes=after.st_size,
                    content_sha256=digest,
                )
            )
    census = AntigravitySourceCensus(root=root, items=tuple(items))
    census.assert_conserved()
    return census


@dataclass(frozen=True, slots=True)
class AntigravityLanguageServerInfo:
    """Identity and capabilities established by the vendor adapter handshake."""

    binary_path: Path
    version: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AntigravityExportOutcome:
    """One conservation-bearing result for one manifested conversation."""

    source_path: Path
    cascade_id: str
    session: ParsedSession | None = None
    error: str | None = None
    converter: AntigravityLanguageServerInfo | None = None

    @property
    def obtained(self) -> bool:
        return self.session is not None and self.error is None


def classify_source_path(source_path: str | Path) -> AntigravitySourceClassification:
    """Classify every Antigravity path into a session, artifact, or unknown role."""
    path = Path(source_path)
    from polylogue.sources.origin_specs import artifact_rule_for_path

    rule = artifact_rule_for_path(Provider.ANTIGRAVITY, str(path))
    role_by_coverage = {
        "conversation_protobuf": AntigravitySourceRole.CONVERSATION_PROTOBUF,
        "brain_metadata_sidecar": AntigravitySourceRole.METADATA_SIDECAR,
        "brain_document": AntigravitySourceRole.BRAIN_DOCUMENT,
    }
    if rule is not None and rule.coverage_role in role_by_coverage:
        return AntigravitySourceClassification(
            role_by_coverage[rule.coverage_role],
            rule.parse_policy == "session",
            ArtifactKind(rule.kind),
            rule.fidelity_note,
        )
    return AntigravitySourceClassification(
        AntigravitySourceRole.UNKNOWN,
        False,
        ArtifactKind.UNKNOWN,
        "unrecognized Antigravity source item",
    )


@dataclass(frozen=True, slots=True)
class AntigravitySessionSummary:
    cascade_id: str
    title: str | None = None
    workspace_name: str | None = None
    snippet: str | None = None
    last_modified_time: str | None = None

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> AntigravitySessionSummary | None:
        cascade_id = _string(payload.get("cascadeId"))
        if cascade_id is None:
            return None
        return cls(
            cascade_id=cascade_id,
            title=_string(payload.get("title")),
            workspace_name=_string(payload.get("workspaceName")),
            snippet=_string(payload.get("snippet")),
            last_modified_time=_string(payload.get("lastModifiedTime")),
        )


class _AntigravityLanguageServerExportClient(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]: ...

    def export_markdown(self, cascade_id: str) -> str: ...


class AntigravityLanguageServerClient:
    """Small client for Antigravity's own local language-server export API."""

    def __init__(
        self,
        root: Path,
        *,
        language_server_path: Path | None = None,
        startup_timeout_s: float = 6.0,
    ) -> None:
        self.root = root.expanduser()
        self.language_server_path = language_server_path
        self.startup_timeout_s = startup_timeout_s
        self.port = _free_local_port()
        self._process: subprocess.Popen[bytes] | None = None
        self.server_info: AntigravityLanguageServerInfo | None = None

    def __enter__(self) -> AntigravityLanguageServerClient:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return
        binary = self.language_server_path or discover_language_server()
        if binary is None:
            raise AntigravityBinaryUnavailableError("Antigravity language_server_linux_x64 was not found")
        version = _discover_language_server_version(binary)

        cmd = [
            str(binary),
            "-standalone",
            "-persistent_mode",
            f"-http_server_port={self.port}",
            f"-gemini_dir={self.root.parent}",
            f"-app_data_dir={self.root.name}",
            "-override_ide_name=antigravity",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._wait_until_ready()
        self.server_info = AntigravityLanguageServerInfo(
            binary_path=binary,
            version=version,
            capabilities=("SearchConversations", "ConvertTrajectoryToMarkdown"),
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        payload = self._post(_SEARCH_ENDPOINT, {"query": query, "limit": limit})
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        summaries: list[AntigravitySessionSummary] = []
        for item in results:
            if isinstance(item, dict):
                normalized = {str(key): value for key, value in item.items()}
                if summary := AntigravitySessionSummary.from_payload(normalized):
                    summaries.append(summary)
        return summaries

    def export_markdown(self, cascade_id: str) -> str:
        payload = self._post(_MARKDOWN_ENDPOINT, {"conversationId": cascade_id})
        markdown = payload.get("markdown")
        if not isinstance(markdown, str) or not markdown:
            raise AntigravityExportError(f"Antigravity returned no markdown for cascade {cascade_id}")
        return markdown

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise AntigravityExportError(f"Antigravity language server exited with code {process.returncode}")
            try:
                self._post(_SEARCH_ENDPOINT, {"query": "", "limit": 1})
                return
            except AntigravityExportError as exc:
                last_error = exc
                time.sleep(0.2)
        raise AntigravityExportError(f"Antigravity language server did not become ready: {last_error}")

    def _post(self, endpoint: str, payload: JSONDocument) -> JSONDocument:
        request = Request(
            f"http://127.0.0.1:{self.port}{endpoint}",
            data=dumps_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10.0) as response:
                loaded = loads(response.read())
        except (OSError, TimeoutError, ValueError) as exc:
            raise AntigravityExportError(str(exc)) from exc
        if not isinstance(loaded, dict):
            raise AntigravityExportError(f"Antigravity endpoint {endpoint} returned non-object JSON")
        return {str(key): value for key, value in loaded.items()}


def looks_like_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )


def _validate_language_server_markdown(markdown: str, cascade_id: str) -> None:
    """Reject a successful RPC response that is not a transcript export."""
    sections = list(_SECTION_RE.finditer(markdown))
    if not sections:
        raise AntigravityExportError(f"language server returned partial conversation export for cascade {cascade_id}")
    has_content = any(
        markdown[section.end() : (sections[index + 1].start() if index + 1 < len(sections) else len(markdown))].strip()
        for index, section in enumerate(sections)
    )
    if not has_content:
        raise AntigravityExportError(f"language server returned an empty conversation export for cascade {cascade_id}")


def markdown_export_payload(summary: AntigravitySessionSummary, markdown: str) -> JSONDocument:
    payload: JSONDocument = {
        "source": "antigravity_language_server",
        "cascadeId": summary.cascade_id,
        "markdown": markdown,
    }
    if summary.title:
        payload["title"] = summary.title
    if summary.workspace_name:
        payload["workspaceName"] = summary.workspace_name
    if summary.snippet:
        payload["snippet"] = summary.snippet
    if summary.last_modified_time:
        payload["lastModifiedTime"] = summary.last_modified_time
    return payload


@parser_admission("antigravity")
def parse_markdown_export_payload(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    summary = AntigravitySessionSummary(
        cascade_id=_string(payload.get("cascadeId")) or fallback_id,
        title=_string(payload.get("title")),
        workspace_name=_string(payload.get("workspaceName")),
        snippet=_string(payload.get("snippet")),
        last_modified_time=_string(payload.get("lastModifiedTime")),
    )
    return parse_markdown_export(_string(payload.get("markdown")) or "", summary)


def parse_markdown_export(
    markdown: str,
    summary: AntigravitySessionSummary,
) -> ParsedSession:
    messages = _mark_active_leaf(_messages_from_markdown(markdown, summary.cascade_id))

    return ParsedSession(
        source_name=Provider.ANTIGRAVITY,
        provider_session_id=summary.cascade_id,
        title=summary.title,
        title_source=TitleSource.ORIGIN if summary.title else None,
        created_at=None,
        updated_at=summary.last_modified_time,
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
    )


def iter_language_server_exports(
    root: Path,
    *,
    client: _AntigravityLanguageServerExportClient | None = None,
    only_cascade_ids: frozenset[str] | None = None,
) -> Iterable[ParsedSession]:
    """Yield successful conversions, preserving the historical strict API."""
    outcomes = iter_language_server_export_results(root, client=client, only_cascade_ids=only_cascade_ids)
    expected = len(_conversation_pb_paths(root))
    if only_cascade_ids is not None:
        expected = sum(1 for path in _conversation_pb_paths(root) if path.stem in only_cascade_ids)
    for obtained, outcome in enumerate(outcomes):
        if not outcome.obtained:
            raise AntigravityPartialExportError(
                f"Antigravity export failed for cascade {outcome.cascade_id}: {outcome.error}",
                obtained=obtained,
                expected=expected,
            )
        assert outcome.session is not None
        yield outcome.session


def iter_language_server_export_results(
    root: Path,
    *,
    client: _AntigravityLanguageServerExportClient | None = None,
    only_cascade_ids: frozenset[str] | None = None,
) -> Iterable[AntigravityExportOutcome]:
    """Yield one typed outcome for every manifested conversation protobuf.

    Conversion failures are isolated to their item so a poison trajectory
    cannot suppress unrelated progress. Startup and handshake failures remain
    raised because they invalidate the complete source route.
    """
    owned_client = client is None
    runtime_client = client or AntigravityLanguageServerClient(root)
    try:
        if owned_client:
            runtime_client.start()
        pb_paths = _conversation_pb_paths(root)
        if only_cascade_ids is not None:
            pb_paths = [pb_path for pb_path in pb_paths if pb_path.stem in only_cascade_ids]
        if not pb_paths:
            return
        try:
            summaries_by_id = {summary.cascade_id: summary for summary in runtime_client.search_sessions()}
        except Exception as exc:
            raise AntigravityExportError(f"Antigravity SearchConversations handshake failed: {exc}") from exc
        seen_ids: set[str] = set()
        for pb_path in pb_paths:
            cascade_id = pb_path.stem
            if cascade_id in seen_ids:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    error="duplicate conversation identity",
                    converter=getattr(runtime_client, "server_info", None),
                )
                continue
            seen_ids.add(cascade_id)
            try:
                before = _file_digest(pb_path)
                summary = summaries_by_id.get(cascade_id) or AntigravitySessionSummary(
                    cascade_id=cascade_id,
                    last_modified_time=_iso_mtime(pb_path),
                )
                markdown = runtime_client.export_markdown(cascade_id)
                _validate_language_server_markdown(markdown, cascade_id)
                session = parse_markdown_export_payload(markdown_export_payload(summary, markdown), cascade_id)
                if not session.messages or not any((message.text or "").strip() for message in session.messages):
                    raise AntigravityExportError("language server returned an empty or partial conversation export")
                after = _file_digest(pb_path)
                if before != after:
                    raise AntigravityExportError("conversation protobuf changed during conversion")
            except Exception as exc:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    error=str(exc),
                    converter=getattr(runtime_client, "server_info", None),
                )
            else:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    session=session,
                    converter=getattr(runtime_client, "server_info", None),
                )
    finally:
        if owned_client:
            runtime_client.close()


def _conversation_pb_paths(root: Path) -> list[Path]:
    """List real conversation trajectory files, sorted for deterministic order."""
    conversations_dir = root / "conversations"
    if not conversations_dir.is_dir():
        return []
    return sorted(conversations_dir.glob("*.pb"))


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iso_mtime(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()


def discover_language_server() -> Path | None:
    from polylogue.config import load_polylogue_config

    configured_path = load_polylogue_config().antigravity_language_server
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.is_file():
            return path

    if binary_path := shutil.which("language_server_linux_x64"):
        return Path(binary_path)

    candidates = sorted(
        Path(match)
        for match in glob(
            "/nix/store/*-antigravity-*/lib/antigravity/resources/app/extensions/antigravity/bin/language_server_linux_x64"
        )
    )
    return candidates[-1] if candidates else None


def _discover_language_server_version(binary: Path) -> str:
    """Read the vendor binary version before using its HTTP conversion API."""

    def compatible(version: str) -> str:
        if version.split(".", 1)[0] not in {"1", "2"}:
            raise AntigravityExportError(
                f"incompatible Antigravity language server version {version}; expected vendor 1.x or 2.x"
            )
        return version

    configured = os.environ.get("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER_VERSION")
    if configured and configured.strip():
        return compatible(configured.strip())
    for flag in ("--version", "-version"):
        try:
            completed = subprocess.run(
                [str(binary), flag],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AntigravityExportError(f"could not handshake with language server {binary}: {exc}") from exc
        output = completed.stdout.decode("utf-8", errors="replace").strip()
        match = re.search(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b", output)
        if match:
            return compatible(match.group(0))
    for candidate in (binary, binary.resolve()):
        package_match = re.search(r"antigravity-(\d+\.\d+\.\d+)", str(candidate))
        if package_match:
            return compatible(package_match.group(1))
    raise AntigravityExportError(f"language server {binary} did not report a compatible version")


def _messages_from_markdown(markdown: str, cascade_id: str) -> list[ParsedMessage]:
    sections = list(_SECTION_RE.finditer(markdown))
    messages: list[ParsedMessage] = []
    for index, section in enumerate(sections):
        start = section.end()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(markdown)
        text = markdown[start:end].strip()
        if not text:
            continue
        heading = section.group("title")
        role = Role.USER if heading == "User Input" else Role.ASSISTANT
        provider_message_id = synthetic_message_id(
            namespace=cascade_id,
            role=role,
            text=text,
            timestamp=None,
            kind=_message_kind(heading),
        )
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=role,
                text=text,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                position=len(messages),
                variant_index=0,
                is_active_path=True,
                # polylogue-gzgyl: an antigravity "User Input" section is
                # unambiguously a real human turn -- positive-evidence
                # override for the shared classify_material_origin
                # no-fallthrough (#2502).
                material_origin=human_authored_override(
                    role,
                    MessageType.MESSAGE,
                    classify_material_origin(role=role, message_type=MessageType.MESSAGE, text=text),
                ),
            )
        )

    if messages:
        return messages

    text = _strip_markdown_preamble(markdown)
    if not text:
        return []
    return [
        ParsedMessage(
            provider_message_id=synthetic_message_id(
                namespace=cascade_id,
                role=Role.ASSISTANT,
                text=text,
                timestamp=None,
                kind="export",
            ),
            role=Role.ASSISTANT,
            text=text,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            position=0,
            variant_index=0,
            is_active_path=True,
        )
    ]


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    # bd polylogue-2hwl: delegate to the shared position-based helper --
    # flagging by provider_message_id equality (the previous approach here)
    # flags every message sharing the final message's id, not just the true
    # leaf, whenever a retried/regenerated section reuses that id.
    return mark_last_occurrence_as_active_leaf(messages)


def _strip_markdown_preamble(markdown: str) -> str:
    lines = markdown.splitlines()
    while lines and (lines[0].startswith("# ") or lines[0].startswith("Note:") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _message_kind(heading: str) -> str:
    return heading.lower().replace(" ", "_")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "AntigravityBinaryUnavailableError",
    "AntigravitySessionSummary",
    "AntigravityExportError",
    "AntigravityLanguageServerClient",
    "AntigravityPartialExportError",
    "AntigravityExportOutcome",
    "AntigravityLanguageServerInfo",
    "AntigravitySourceClassification",
    "AntigravitySourceRole",
    "AntigravitySourceItem",
    "AntigravitySourceCensus",
    "AntigravitySourceMutationError",
    "census_source",
    "classify_source_path",
    "conversation_pb_paths",
    "discover_language_server",
    "iter_language_server_exports",
    "looks_like_markdown_export",
    "markdown_export_payload",
    "iter_language_server_export_results",
    "parse_markdown_export",
    "parse_markdown_export_payload",
]

# Public alias -- ``source_parsing.py`` needs the same disk-truth listing to
# locate the raw ``.pb`` bytes for blob snapshotting per exported session.
conversation_pb_paths = _conversation_pb_paths
