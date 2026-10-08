"""Synthetic artifacts lowered by the actual native preparation owner."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from polylogue.browser_capture.models import BrowserCaptureProvenance
from polylogue.browser_capture.native_preparation import envelope_prefix, json_bytes
from polylogue.core.enums import Provider
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.sources.parsers.browser_capture import parse_native_member_streams
from polylogue.sources.prepared_message_sink import ScratchSessionSpill, SqliteMessageStore
from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime


def native_proof_artifact(
    tmp_path: Path, fixture: str, provider: Provider = Provider.CHATGPT
) -> tuple[dict[str, Any], int, int]:
    fixture_path = Path(__file__).parents[1] / "fixtures" / provider.value / fixture
    raw = fixture_path.read_bytes()
    payload = json.loads(raw)
    native_id = (
        payload["conversation"]["conversationId"]
        if provider is Provider.GROK
        else payload["id" if provider is Provider.CHATGPT else "uuid"]
    )
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    store = None
    member_owner = contextlib.ExitStack()
    try:
        with retain_native_sql_lifetimes(directory):
            store = SqliteMessageStore(Path(directory.name) / "prepared.db")
            spill = ScratchSessionSpill(store)
            store.conn.execute(
                "CREATE TABLE capture_preparation_plan (ordinal INTEGER PRIMARY KEY, descriptor_json TEXT NOT NULL)"
            )
            if provider is Provider.GROK:
                members = {}
                for name in ("conversation", "responses", "response_nodes"):
                    member_path = Path(directory.name) / (name + ".json")
                    member_path.write_text(json.dumps(payload[name]), encoding="utf-8")
                    members[name] = member_owner.enter_context(member_path.open("rb"))
            else:
                members = {"conversation": member_owner.enter_context(fixture_path.open("rb"))}
            parsed = parse_native_member_streams(provider, members, native_id, spill)
            prefix = b"".join(
                envelope_prefix(
                    parsed,
                    spill,
                    members,
                    BrowserCaptureProvenance(
                        source_url=(
                            f"https://grok.com/c/{native_id}"
                            if provider is Provider.GROK
                            else f"https://chatgpt.com/c/{native_id}"
                            if provider is Provider.CHATGPT
                            else f"https://claude.ai/chat/{native_id}"
                        ),
                        captured_at="2026-01-01T00:00:00Z",
                        adapter_name=f"{provider.value}-native-v1",
                    ),
                    {"capture_fidelity": "native_full"},
                    lambda: None,
                )
            )
            descriptors = []
            for ordinal, serialized in store.conn.execute(
                "SELECT ordinal, descriptor_json FROM capture_preparation_plan ORDER BY ordinal"
            ):
                descriptor = json.loads(serialized)
                descriptor.pop("original_record_ordinal", None)
                descriptor.pop("original_record_key", None)
                native_content = parsed.attachments[ordinal].inline_bytes
                content = native_content if native_content is not None else f"synthetic attachment {ordinal}".encode()
                descriptor["size_bytes"] = len(content)
                descriptor["content_base64"] = base64.b64encode(content).decode("ascii")
                descriptor["provider_meta"].update(
                    asset_acquisition={"status": "acquired"}, content_sha256=hashlib.sha256(content).hexdigest()
                )
                descriptors.append(json_bytes(descriptor))
            return json.loads(prefix + b",".join(descriptors) + b"]}}"), len(parsed.messages), len(parsed.attachments)
    finally:
        try:
            member_owner.close()
        finally:
            try:
                if store is not None:
                    store.close()
            finally:
                # Original native owners retain this exact directory after any
                # construction/close failure; cleanup requires physical drain.
                if not retained_native_sql_owners_for_lifetime(directory):
                    directory.cleanup()
