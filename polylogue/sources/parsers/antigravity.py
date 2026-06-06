"""Parser and local export client for Antigravity session state."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from types import TracebackType
from urllib.error import URLError
from urllib.request import Request, urlopen

from polylogue.archive.message.roles import Role
from polylogue.core.json import JSONDocument, dumps_bytes, loads
from polylogue.types import ContentBlockType, Provider

from .base import ParsedContentBlock, ParsedMessage, ParsedSession

_METADATA_SUFFIX = ".metadata.json"
_SEARCH_ENDPOINT = "/exa.language_server_pb.LanguageServerService/SearchConversations"
_MARKDOWN_ENDPOINT = "/exa.language_server_pb.LanguageServerService/ConvertTrajectoryToMarkdown"
_SECTION_RE = re.compile(r"^### (?P<title>User Input|Planner Response)\s*$", re.MULTILINE)


class AntigravityExportError(RuntimeError):
    """Raised when Antigravity's local export surface cannot be queried."""


class AntigravityBinaryUnavailableError(AntigravityExportError):
    """Raised when the Antigravity language-server binary is not installed.

    This is the benign case — Antigravity is simply not present — and callers
    should fall back to the brain-artifact walk at INFO level without treating
    it as data loss.
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
        except URLError as exc:
            raise AntigravityExportError(str(exc)) from exc
        if not isinstance(loaded, dict):
            raise AntigravityExportError(f"Antigravity endpoint {endpoint} returned non-object JSON")
        return {str(key): value for key, value in loaded.items()}


def looks_like_brain_metadata(payload: JSONDocument, source_path: str | Path | None) -> bool:
    name = Path(source_path).name.lower() if source_path is not None else ""
    return (
        (not name or name.endswith((".json", ".md.metadata.json")))
        and isinstance(payload.get("artifactType"), str)
        and ("summary" in payload or "updatedAt" in payload)
    )


def looks_like_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )


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


def parse_markdown_export_payload(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    summary = AntigravitySessionSummary(
        cascade_id=_string(payload.get("cascadeId")) or fallback_id,
        title=_string(payload.get("title")),
        workspace_name=_string(payload.get("workspaceName")),
        snippet=_string(payload.get("snippet")),
        last_modified_time=_string(payload.get("lastModifiedTime")),
    )
    return parse_markdown_export(_string(payload.get("markdown")) or "", summary)


def parse_brain_metadata(payload: JSONDocument, source_path: Path, fallback_id: str) -> ParsedSession:
    artifact_path = _artifact_path_for_metadata(source_path)
    session_id = artifact_path.parent.name if artifact_path.parent.name else fallback_id
    artifact_name = artifact_path.name if artifact_path.name else fallback_id
    composed_session_id = f"{session_id}:{artifact_name}"
    body = _read_text(artifact_path) or _string(payload.get("summary")) or ""
    title = _title_for_artifact(artifact_path, payload, fallback_id)
    updated_at = _string(payload.get("updatedAt"))
    provider_meta: dict[str, object] = {
        "source_family": "antigravity",
        "artifact_path": str(artifact_path),
        "session_id": session_id,
    }
    if artifact_type := _string(payload.get("artifactType")):
        provider_meta["artifact_type"] = artifact_type
    if summary := _string(payload.get("summary")):
        provider_meta["summary"] = summary
    if not artifact_path.exists():
        provider_meta["missing_markdown_body"] = True

    return ParsedSession(
        source_name=Provider.ANTIGRAVITY,
        provider_session_id=composed_session_id,
        title=title,
        created_at=None,
        updated_at=updated_at,
        messages=[
            ParsedMessage(
                provider_message_id=f"{composed_session_id}:artifact",
                role=Role.ASSISTANT,
                text=body,
                timestamp=updated_at,
                content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=body)] if body else [],
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=True,
                provider_meta={"artifact_name": artifact_name},
            )
        ],
        active_leaf_message_provider_id=f"{composed_session_id}:artifact",
        provider_meta=provider_meta,
    )


def parse_markdown_export(
    markdown: str,
    summary: AntigravitySessionSummary,
) -> ParsedSession:
    messages = _mark_active_leaf(_messages_from_markdown(markdown, summary.cascade_id))
    provider_meta: dict[str, object] = {
        "source_family": "antigravity",
        "source_format": "language_server_markdown_export",
        "cascade_id": summary.cascade_id,
        "degraded_fragmentation": True,
    }
    if summary.workspace_name:
        provider_meta["workspace_name"] = summary.workspace_name
    if summary.snippet:
        provider_meta["snippet"] = summary.snippet

    return ParsedSession(
        source_name=Provider.ANTIGRAVITY,
        provider_session_id=summary.cascade_id,
        title=summary.title,
        created_at=None,
        updated_at=summary.last_modified_time,
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
        provider_meta=provider_meta,
    )


def iter_language_server_exports(
    root: Path,
    *,
    client: AntigravityLanguageServerClient | None = None,
) -> Iterable[ParsedSession]:
    owned_client = client is None
    runtime_client = client or AntigravityLanguageServerClient(root)
    if owned_client:
        runtime_client.start()
    try:
        summaries = runtime_client.search_sessions()
        expected = len(summaries)
        for obtained, summary in enumerate(summaries):
            try:
                markdown = runtime_client.export_markdown(summary.cascade_id)
            except AntigravityExportError as exc:
                # A mid-iteration export failure would otherwise abort the
                # generator after yielding only the sessions seen so far,
                # silently dropping the remainder. Surface obtained-vs-expected
                # so the caller can distinguish partial loss from a benign
                # binary-absent fallback. ``obtained`` is the number already
                # yielded (the index of the failing summary).
                raise AntigravityPartialExportError(
                    f"Antigravity export aborted on cascade {summary.cascade_id}: {exc}",
                    obtained=obtained,
                    expected=expected,
                ) from exc
            else:
                yield parse_markdown_export_payload(markdown_export_payload(summary, markdown), summary.cascade_id)
    finally:
        if owned_client:
            runtime_client.close()


def discover_language_server() -> Path | None:
    env_path = os.environ.get("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER")
    if env_path:
        path = Path(env_path).expanduser()
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
        provider_message_id = f"{cascade_id}:{index}:{_message_kind(heading)}"
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=role,
                text=text,
                content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
                position=len(messages),
                variant_index=0,
                is_active_path=True,
                provider_meta={"antigravity_section": heading},
            )
        )

    if messages:
        return messages

    text = _strip_markdown_preamble(markdown)
    if not text:
        return []
    return [
        ParsedMessage(
            provider_message_id=f"{cascade_id}:0:export",
            role=Role.ASSISTANT,
            text=text,
            content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
            position=0,
            variant_index=0,
            is_active_path=True,
            provider_meta={"antigravity_section": "Markdown Export"},
        )
    ]


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    if not messages:
        return messages
    active_leaf_message_provider_id = messages[-1].provider_message_id
    return [
        message.model_copy(update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id})
        for message in messages
    ]


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


def _artifact_path_for_metadata(source_path: Path) -> Path:
    raw = str(source_path)
    if raw.endswith(_METADATA_SUFFIX):
        return Path(raw[: -len(_METADATA_SUFFIX)])
    return source_path


def _title_for_artifact(artifact_path: Path, payload: JSONDocument, fallback_id: str) -> str:
    if summary := _string(payload.get("summary")):
        return summary[:120]
    return artifact_path.stem.replace("_", " ").replace("-", " ").strip().title() or fallback_id


def _read_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if text else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "AntigravityBinaryUnavailableError",
    "AntigravitySessionSummary",
    "AntigravityExportError",
    "AntigravityLanguageServerClient",
    "AntigravityPartialExportError",
    "discover_language_server",
    "iter_language_server_exports",
    "looks_like_brain_metadata",
    "looks_like_markdown_export",
    "markdown_export_payload",
    "parse_brain_metadata",
    "parse_markdown_export",
    "parse_markdown_export_payload",
]
