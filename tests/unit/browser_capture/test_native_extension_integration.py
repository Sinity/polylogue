"""Actual extension background to canonical receiver parity, outside Vitest.

Run through ``devtools test tests/unit/browser_capture/test_native_extension_integration.py``
in the existing Python development environment with extension npm dependencies.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from threading import Thread

import pytest

from polylogue.browser_capture.server import make_server
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.browser_capture import parse, parse_native_payload
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows


@pytest.mark.uses_real_clock("the extension subprocess owns its browser clock and HTTP progress")
@pytest.mark.parametrize(
    "provider,fixture,native_id",
    [
        ("chatgpt", "chatgpt/native-rich-blocks-v1.json", "native-rich-chatgpt"),
        ("chatgpt", "chatgpt/native-duplicate-attachment-occurrences-v1.json", "native-occurrences"),
        ("claude-ai", "claude-ai/native-rich-blocks-v1.json", "native-rich-blocks"),
        ("grok", "grok/native-bundle.json", "native-conversation"),
    ],
)
def test_extension_background_publishes_canonical_complete_artifact(
    tmp_path: Path,
    provider: str,
    fixture: str,
    native_id: str,
) -> None:
    root = Path(__file__).parents[3]
    source = root / "tests" / "fixtures" / fixture
    token = "synthetic-native-integration-token"
    server = make_server("127.0.0.1", 0, spool_path=tmp_path, auth_token=token)
    server.daemon_threads = False
    server.block_on_close = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                "node",
                str(root / "browser-extension/tests/infra/native-receiver-integration.mjs"),
                f"http://127.0.0.1:{server.server_port}",
                token,
                str(source),
                provider,
                native_id,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    receipt = json.loads(result.stdout)
    assert receipt["summary"]["needs_follow_up"] is (provider == "chatgpt" and "duplicate" in fixture)
    retained = (tmp_path / "capture-jobs" / "artifacts" / f"{receipt['sha256']}.native").read_bytes()
    raw = json.loads(source.read_bytes())
    envelope = json.loads(retained)
    assert envelope["raw_provider_payload"] == raw
    if provider != "grok":
        assert source.read_bytes() in retained
    assert envelope["provenance"]["extension_instance_id"] is None
    assert envelope["provenance"]["acquisition_sequence"] is None
    expected = parse_native_payload(
        Provider.from_string(provider), raw, native_id, prepared_attachment_ownership=provider == "chatgpt"
    )
    captured = parse(envelope, "capture")
    assert captured.messages == expected.messages
    assert captured.attachments == expected.attachments
    assert session_content_hash(captured) == session_content_hash(expected)
    expected_rows = prepare_session_rows(expected)
    captured_rows = prepare_session_rows(captured)
    assert captured_rows.message_rows == expected_rows.message_rows
    assert captured_rows.block_rows == expected_rows.block_rows
