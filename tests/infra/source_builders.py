"""Shared file/export builders for inbox and provider-source tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from polylogue.config import Source
from polylogue.sources.parsers.antigravity import AntigravitySessionSummary

JsonObject: TypeAlias = dict[str, object]
JsonObjectList: TypeAlias = list[JsonObject]
PROVIDER_SOURCE_CLASS = "provider-shaped"


@dataclass(frozen=True, slots=True)
class ProviderSourcePackage:
    """Provider-shaped input material for one law-owned candidate fixture.

    This is deliberately an input primitive, not a case record.  It carries
    enough information to reproduce and authenticate the bytes that enter the
    production admission route, but it has no expected archive rows or
    semantic verdict.  Consumers compose one package for the source traits
    their law needs.
    """

    provider: str
    source_paths: tuple[Path, ...]
    wire_hashes: tuple[str, ...]
    inventory: tuple[tuple[str, int], ...]
    generator_id: str
    schema_inputs: tuple[str, ...] = ()
    attachment_hashes: tuple[str, ...] = ()
    schedule_digest: str | None = None

    @property
    def source_class(self) -> str:
        """The admission shape represented by this package."""
        return PROVIDER_SOURCE_CLASS

    @classmethod
    def from_files(
        cls,
        provider: str,
        files: tuple[Path, ...],
        *,
        source_paths: tuple[Path, ...] | None = None,
        generator_id: str = "provider-source-package-v1",
        schema_inputs: tuple[str, ...] = (),
        attachment_bytes: tuple[bytes, ...] = (),
        schedule_digest: str | None = None,
    ) -> ProviderSourcePackage:
        """Describe existing provider-shaped files without inventing semantics."""
        if not provider or not files:
            raise ValueError("a provider source package requires a provider and files")
        if any(not path.is_file() for path in files):
            raise ValueError("provider source package files must already exist")
        paths = source_paths or files
        if not paths:
            raise ValueError("provider source package requires a declared source root")
        if any(not path.exists() for path in paths):
            raise ValueError("provider source package source paths must exist")
        wire_hashes = tuple(_sha256_file(path) for path in files)
        suffix_counts: dict[str, int] = {}
        for path in files:
            suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
        inventory = (
            ("files", len(files)),
            ("bytes", sum(path.stat().st_size for path in files)),
            *tuple(sorted((f"suffix:{suffix}", count) for suffix, count in suffix_counts.items())),
        )
        return cls(
            provider=provider,
            source_paths=tuple(paths),
            wire_hashes=wire_hashes,
            inventory=inventory,
            generator_id=generator_id,
            schema_inputs=tuple(schema_inputs),
            attachment_hashes=tuple(hashlib.sha256(item).hexdigest() for item in attachment_bytes),
            schedule_digest=schedule_digest,
        )

    @property
    def identity(self) -> str:
        """Content identity for caching input material, never semantic output."""
        payload = {
            "provider": self.provider,
            "wire_hashes": self.wire_hashes,
            "inventory": self.inventory,
            "generator_id": self.generator_id,
            "schema_inputs": self.schema_inputs,
            "attachment_hashes": self.attachment_hashes,
            "schedule_digest": self.schedule_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"source-package:sha256:{hashlib.sha256(encoded).hexdigest()}"

    def admitted_sources(self) -> tuple[Source, ...]:
        """Return the production ``Source`` inputs for the admission seam."""
        return tuple(Source(name=self.provider, path=path) for path in self.source_paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_source_package(
    provider: str,
    files: tuple[Path, ...],
    *,
    source_paths: tuple[Path, ...] | None = None,
    generator_id: str = "synthetic-corpus-v1",
    schema_inputs: tuple[str, ...] = (),
    attachment_bytes: tuple[bytes, ...] = (),
    schedule_digest: str | None = None,
) -> ProviderSourcePackage:
    """Build one composable provider/source package for a consuming law."""
    return ProviderSourcePackage.from_files(
        provider,
        files,
        source_paths=source_paths,
        generator_id=generator_id,
        schema_inputs=schema_inputs,
        attachment_bytes=attachment_bytes,
        schedule_digest=schedule_digest,
    )


def admit_provider_source_packages(
    archive_root: Path,
    packages: Iterable[ProviderSourcePackage],
) -> object:
    """Admit law-owned provider packages through the production ingest route."""
    selected = tuple(packages)
    if not selected:
        raise ValueError("at least one provider source package is required")
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive

    sources = [source for package in selected for source in package.admitted_sources()]
    return asyncio.run(parse_sources_archive(archive_root, sources, parse_workers=1))


@dataclass
class SyntheticAntigravityLanguageServerClient:
    """Test-only language-server boundary for real synthetic conversation files."""

    root: Path

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        del limit, query
        return [
            AntigravitySessionSummary(
                cascade_id=path.stem,
                title=f"Synthetic Antigravity {path.stem}",
                last_modified_time="2026-01-01T00:00:00Z",
            )
            for path in sorted((self.root / "conversations").glob("*.pb"))
        ]

    def export_markdown(self, cascade_id: str) -> str:
        return (
            "### User Input\n\n"
            f"Synthetic prompt for {cascade_id}\n\n"
            "### Planner Response\n\n"
            f"Synthetic response for {cascade_id}\n"
        )


def make_chatgpt_node(
    msg_id: str,
    role: str,
    content_parts: list[str],
    children: list[str] | None = None,
    timestamp: float | None = None,
    metadata: JsonObject | None = None,
    parent: str | None = None,
) -> JsonObject:
    """Generate a ChatGPT export mapping node for parser tests."""
    message: JsonObject = {
        "id": msg_id,
        "author": {"role": role},
        "content": {"content_type": "text", "parts": content_parts},
    }
    node: JsonObject = {
        "id": msg_id,
        "message": message,
    }
    if children:
        node["children"] = children
    if parent:
        node["parent"] = parent
    if timestamp:
        message["create_time"] = timestamp
    if metadata:
        message["metadata"] = metadata
    return node


def make_claude_chat_message(
    uuid: str,
    sender: str,
    text: str,
    attachments: JsonObjectList | None = None,
    files: JsonObjectList | None = None,
    timestamp: str | None = None,
) -> JsonObject:
    """Generate a Claude AI chat_messages entry for parser tests."""
    msg: JsonObject = {"uuid": uuid, "text": text}
    if sender:
        msg["sender"] = sender
    if attachments:
        msg["attachments"] = attachments
    if files:
        msg["files"] = files
    if timestamp:
        msg["created_at"] = timestamp
    return msg


class ChatGPTExportBuilder:
    """Builder for ChatGPT export payloads with mapping nodes."""

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._nodes: JsonObjectList = []
        self._node_counter = 0
        self._create_time = 1704067200.0
        self._timestamp = self._create_time

    def title(self, title: str) -> ChatGPTExportBuilder:
        self._title = title
        return self

    def add_node(
        self,
        role: str,
        *content_parts: str,
        node_id: str | None = None,
        metadata: JsonObject | None = None,
        model_slug: str | None = None,
    ) -> ChatGPTExportBuilder:
        self._node_counter += 1
        nid = node_id or f"node-{self._node_counter}"

        meta: JsonObject = dict(metadata or {})
        if model_slug:
            meta["model_slug"] = model_slug

        self._nodes.append(
            make_chatgpt_node(
                nid,
                role,
                list(content_parts),
                timestamp=self._timestamp,
                metadata=meta if meta else None,
            )
        )
        self._timestamp += 1.0
        return self

    def add_system_node(self, content: str, node_id: str | None = None) -> ChatGPTExportBuilder:
        return self.add_node("system", content, node_id=node_id)

    def add_tool_node(
        self,
        tool_name: str,
        result: str,
        node_id: str | None = None,
    ) -> ChatGPTExportBuilder:
        return self.add_node("tool", result, node_id=node_id, metadata={"name": tool_name})

    def build(self) -> JsonObject:
        result: JsonObject = {
            "id": self.conv_id,
            "conversation_id": self.conv_id,
            "create_time": self._create_time,
            "current_node": str(self._nodes[-1]["id"]) if self._nodes else "root",
            "mapping": {str(node["id"]): node for node in self._nodes},
        }
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class GenericSessionBuilder:
    """Builder for simple message-list provider payloads."""

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._title: str | None = None
        self._messages: JsonObjectList = []
        self._msg_counter = 0

    def title(self, title: str) -> GenericSessionBuilder:
        self._title = title
        return self

    def add_message(
        self,
        role: str,
        content: str,
        message_id: str | None = None,
        text: str | None = None,
    ) -> GenericSessionBuilder:
        self._msg_counter += 1
        msg_id = message_id or f"m{self._msg_counter}"
        msg: JsonObject = {"id": msg_id, "role": role}
        if text is not None:
            msg["text"] = text
        else:
            msg["content"] = content
        self._messages.append(msg)
        return self

    def add_user(self, content: str, **kwargs: str | None) -> GenericSessionBuilder:
        return self.add_message("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs: str | None) -> GenericSessionBuilder:
        return self.add_message("assistant", content, **kwargs)

    def build(self) -> JsonObject:
        result: JsonObject = {"id": self.conv_id, "messages": self._messages}
        if self._title:
            result["title"] = self._title
        return result

    def write_to(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return path


class InboxBuilder:
    """Builder for inbox directories populated with provider exports."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[Path, str]] = []

    def add_json_file(self, filename: str, data: object) -> InboxBuilder:
        import json

        path = self.base_path / filename
        self.files.append((path, json.dumps(data, indent=2)))
        return self

    def add_jsonl_file(self, filename: str, entries: list[object]) -> InboxBuilder:
        import json

        path = self.base_path / filename
        content = "\n".join(json.dumps(entry) for entry in entries) + "\n"
        self.files.append((path, content))
        return self

    def add_codex_session(
        self,
        conv_id: str,
        title: str | None = None,
        messages: list[tuple[str, str]] | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        builder = GenericSessionBuilder(conv_id)
        if title:
            builder.title(title)
        for role, content in messages or [("user", "Hello"), ("assistant", "Hi there!")]:
            builder.add_message(role, content)
        return self.add_json_file(filename or f"{conv_id}.json", builder.build())

    def add_chatgpt_export(
        self,
        conv_id: str,
        title: str | None = None,
        nodes: JsonObjectList | None = None,
        filename: str | None = None,
    ) -> InboxBuilder:
        payload: JsonObject
        if nodes is None:
            builder = ChatGPTExportBuilder(conv_id)
            if title:
                builder.title(title)
            builder.add_node("user", "Hello").add_node("assistant", "Hi there!")
            payload = builder.build()
        else:
            payload = {
                "id": conv_id,
                "conversation_id": conv_id,
                "create_time": 1704067200.0,
                "current_node": str(nodes[-1]["id"]) if nodes else "root",
                "mapping": {str(node["id"]): node for node in nodes},
            }
            if title:
                payload["title"] = title
        return self.add_json_file(filename or f"chatgpt_{conv_id}.json", payload)

    def add_claude_export(
        self,
        conv_id: str,
        name: str | None = None,
        chat_messages: JsonObjectList | None = None,
        filename: str | None = None,
        wrap_in_sessions: bool = True,
    ) -> InboxBuilder:
        session: JsonObject = {
            "id": conv_id,
            "chat_messages": chat_messages
            or [
                make_claude_chat_message("m1", "human", "Hello"),
                make_claude_chat_message("m2", "assistant", "Hi there!"),
            ],
        }
        if name:
            session["name"] = name
        payload = {"sessions": [session]} if wrap_in_sessions else session
        return self.add_json_file(filename or f"claude_{conv_id}.json", payload)

    def build(self) -> Path:
        for path, content in self.files:
            path.write_text(content, encoding="utf-8")
        return self.base_path

    def get_file_path(self, filename: str) -> Path:
        return self.base_path / filename
