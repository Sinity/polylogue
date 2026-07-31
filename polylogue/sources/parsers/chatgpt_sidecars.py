"""ChatGPT export sidecar resolvers: asset naming and sandbox-file joins.

The 2026-07-29 GDPR/Takeout export ships attachment BYTES for the first time,
as ``.dat`` ZIP members whose basenames are opaque file ids. Two sibling JSON
files supply names/metadata for those ids:

- ``conversation_asset_file_names.json`` — a flat ``{dat_basename: filename}``
  map, 1,656 entries, conversation-asset ids (``file-<b64ish>``).
- ``library_files.json`` — 2,367 richer records (mime, size, sha256, upload/
  processed times, and an EXACT ``origination_message_id``/
  ``origination_thread_id`` join back to the producing assistant message),
  keyed by ``file_id`` (``file_<32hex>`` for Library files, but also seen on
  conversation attachments referencing an existing Library upload).

Together the two sources name all 3,228 known ``.dat`` blobs (bd polylogue-0hwv).
``library_files.json`` also carries the identity join this module uses to
resolve ``sandbox:/mnt/data/<name>`` links the model emits in assistant text
for Code-Interpreter-produced files, which carry no file id of their own
(bd polylogue-dt5s). Resolution is a strict tiered join, not fuzzy matching:

  tier 1  exact ``origination_message_id`` match, name also matches
  tier 2  exact ``origination_message_id`` match, name differs (still an
          identity-grade join — the id is authoritative, the name is a label)
  tier 3  ``origination_thread_id`` + name match (message id unknown/absent)
  tier 4  a globally unique library file_name match (no id evidence at all)
  tier 5  a globally AMBIGUOUS name match (>1 candidate) — evidence, not
          identity; never resolved to a specific file
  tier 6  unresolved — no candidate anywhere

Every resolution records which tier produced it, so an audit can always tell
an id join from a name guess (see ``SandboxResolution.tier``/``method``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_DAT_SUFFIX = ".dat"
_FILE_SERVICE_PREFIX = "file-service://"


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@dataclass(frozen=True, slots=True)
class LibraryFileRecord:
    """One ``library_files.json`` entry (the ChatGPT Library file catalog)."""

    file_id: str
    file_name: str | None
    file_extension: str | None
    mime_type: str | None
    file_size_bytes: int | None
    sha256_digest: str | None
    origination_message_id: str | None
    origination_thread_id: str | None
    library_artifact_type: str | None
    directory_id: str | None
    created_at: str | None
    file_upload_time: str | None
    file_processed_time: str | None


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """A ``.dat`` blob's real identity, resolved from one of the two sidecars."""

    file_id: str
    name: str | None
    mime_type: str | None
    size_bytes: int | None
    sha256_digest: str | None
    #: "library_files" (richer — mime/size/sha) or "conversation_asset_file_names"
    #: (name only). ``library_files`` is preferred whenever both hit.
    source: str


@dataclass(frozen=True, slots=True)
class SandboxResolution:
    """Result of joining one ``sandbox:/mnt/data/<name>`` link to a real file."""

    #: 1-6, see module docstring. 6 means unresolved (metadata-only reference).
    tier: int
    method: str
    file: LibraryFileRecord | None
    #: The library file_name that was actually matched (may differ from the
    #: sandbox link's own filename at tier 2 — the id is authoritative there).
    matched_name: str | None


def parse_library_files(payload: object) -> dict[str, LibraryFileRecord]:
    """Parse ``library_files.json`` into records keyed by ``file_id``."""
    if not isinstance(payload, list):
        return {}
    records: dict[str, LibraryFileRecord] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        file_id = entry.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            continue
        records[file_id] = LibraryFileRecord(
            file_id=file_id,
            file_name=_optional_str(entry.get("file_name")),
            file_extension=_optional_str(entry.get("file_extension")),
            mime_type=_optional_str(entry.get("mime_type")),
            file_size_bytes=_optional_int(entry.get("file_size_bytes")),
            sha256_digest=_optional_str(entry.get("sha256_digest")),
            origination_message_id=_optional_str(entry.get("origination_message_id")),
            origination_thread_id=_optional_str(entry.get("origination_thread_id")),
            library_artifact_type=_optional_str(entry.get("library_artifact_type")),
            directory_id=_optional_str(entry.get("directory_id")),
            created_at=_optional_str(entry.get("created_at")),
            file_upload_time=_optional_str(entry.get("file_upload_time")),
            file_processed_time=_optional_str(entry.get("file_processed_time")),
        )
    return records


def parse_asset_file_names(payload: object) -> dict[str, str]:
    """Parse ``conversation_asset_file_names.json`` (``{dat_basename: name}``).

    Keys are normalized by stripping a trailing ``.dat`` so they line up
    directly with the bare file ids referenced by ``asset_pointer`` and
    ``attachments[].id`` in conversation payloads (neither carries ``.dat``).
    """
    if not isinstance(payload, dict):
        return {}
    names: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            continue
        normalized_key = key[: -len(_DAT_SUFFIX)] if key.endswith(_DAT_SUFFIX) else key
        names[normalized_key] = value
    return names


class ChatGPTAssetIndex:
    """Resolves ``.dat`` asset ids and sandbox-file links against export sidecars.

    Built once per export/source scan (see ``assembly_chatgpt.py``) and reused
    across every conversation the source yields.
    """

    __slots__ = ("_library", "_asset_names", "_by_message", "_by_thread", "_by_name")

    def __init__(
        self,
        library_files: Mapping[str, LibraryFileRecord],
        asset_names: Mapping[str, str],
    ) -> None:
        self._library = dict(library_files)
        self._asset_names = dict(asset_names)
        by_message: dict[str, list[LibraryFileRecord]] = {}
        by_thread: dict[str, list[LibraryFileRecord]] = {}
        by_name: dict[str, list[LibraryFileRecord]] = {}
        for record in self._library.values():
            if record.origination_message_id:
                by_message.setdefault(record.origination_message_id, []).append(record)
            if record.origination_thread_id:
                by_thread.setdefault(record.origination_thread_id, []).append(record)
            if record.file_name:
                by_name.setdefault(record.file_name, []).append(record)
        self._by_message = by_message
        self._by_thread = by_thread
        self._by_name = by_name

    @classmethod
    def build(
        cls,
        *,
        library_files_payload: object = None,
        asset_file_names_payload: object = None,
    ) -> ChatGPTAssetIndex:
        return cls(
            parse_library_files(library_files_payload),
            parse_asset_file_names(asset_file_names_payload),
        )

    @classmethod
    def empty(cls) -> ChatGPTAssetIndex:
        return cls({}, {})

    @property
    def is_empty(self) -> bool:
        return not self._library and not self._asset_names

    def library_record(self, file_id: str) -> LibraryFileRecord | None:
        return self._library.get(_normalize_file_id(file_id))

    def resolve_dat(self, file_id: str) -> ResolvedAsset | None:
        """Resolve a ``.dat`` asset id to its real name/mime/size (bd polylogue-0hwv).

        Prefers ``library_files.json`` (richer: mime/size/sha256) over
        ``conversation_asset_file_names.json`` (name only) when both hit.
        """
        clean = _normalize_file_id(file_id)
        record = self._library.get(clean)
        if record is not None:
            return ResolvedAsset(
                file_id=clean,
                name=record.file_name,
                mime_type=record.mime_type,
                size_bytes=record.file_size_bytes,
                sha256_digest=record.sha256_digest,
                source="library_files",
            )
        name = self._asset_names.get(clean)
        if name is not None:
            return ResolvedAsset(
                file_id=clean,
                name=name,
                mime_type=None,
                size_bytes=None,
                sha256_digest=None,
                source="conversation_asset_file_names",
            )
        return None

    def resolve_sandbox(
        self,
        *,
        message_id: str | None,
        thread_id: str | None,
        file_name: str,
    ) -> SandboxResolution:
        """Resolve a ``sandbox:/mnt/data/<file_name>`` link (bd polylogue-dt5s).

        Layered resolver, not a fuzzy matcher — see the module docstring for
        the six tiers. Tier 5 (ambiguous global name) records evidence but
        never returns a specific ``file``; tier 6 always records that the link
        was seen even with no resolvable bytes/metadata source.
        """
        if message_id:
            candidates = self._by_message.get(message_id)
            if candidates:
                exact = next((c for c in candidates if c.file_name == file_name), None)
                if exact is not None:
                    return SandboxResolution(tier=1, method="message_id+name", file=exact, matched_name=exact.file_name)
                # Identity-grade join on id alone -- the library name is a
                # label here, not part of the key (measured: tier 2, 2026-07-31).
                return SandboxResolution(
                    tier=2,
                    method="message_id",
                    file=candidates[0],
                    matched_name=candidates[0].file_name,
                )
        if thread_id:
            thread_matches = [c for c in self._by_thread.get(thread_id, ()) if c.file_name == file_name]
            if thread_matches:
                return SandboxResolution(
                    tier=3,
                    method="thread_id+name",
                    file=thread_matches[0],
                    matched_name=file_name,
                )
        global_matches = self._by_name.get(file_name)
        if global_matches:
            if len(global_matches) == 1:
                return SandboxResolution(
                    tier=4,
                    method="global_name_unique",
                    file=global_matches[0],
                    matched_name=file_name,
                )
            return SandboxResolution(tier=5, method="global_name_ambiguous", file=None, matched_name=file_name)
        return SandboxResolution(tier=6, method="unresolved", file=None, matched_name=None)


def _normalize_file_id(file_id: str) -> str:
    if file_id.startswith(_FILE_SERVICE_PREFIX):
        return file_id[len(_FILE_SERVICE_PREFIX) :]
    if file_id.endswith(_DAT_SUFFIX):
        return file_id[: -len(_DAT_SUFFIX)]
    return file_id


__all__ = [
    "ChatGPTAssetIndex",
    "LibraryFileRecord",
    "ResolvedAsset",
    "SandboxResolution",
    "parse_asset_file_names",
    "parse_library_files",
]
