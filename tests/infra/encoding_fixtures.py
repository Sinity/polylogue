"""Build ZIP files with controlled encoding edge cases for acquisition tests.

Each builder creates a ZIP at the given path containing JSON/JSONL with specific
encoding issues. Used to test the ZIP extraction -> parse pipeline end-to-end.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

JsonObject = dict[str, object]


# Minimal valid Codex JSONL records for realistic content
def _valid_codex_record(index: int = 0, role: str = "user") -> JsonObject:
    """Generate a minimal valid Codex message record."""
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": f"Message {index}"}],
        "id": f"msg-{index:04d}",
        "timestamp": f"2025-01-01T00:{index:02d}:00Z",
    }


def _valid_codex_session_header() -> JsonObject:
    """Session header record for Codex JSONL."""
    return {"id": "session-encoding-test", "timestamp": "2025-01-01T00:00:00Z"}


def _valid_chatgpt_conversation() -> JsonObject:
    """Minimal valid ChatGPT conversation JSON."""
    return {
        "title": "Encoding Test",
        "mapping": {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello from encoding test"]},
                    "create_time": 1700000000.0,
                },
                "parent": None,
                "children": ["node-2"],
            },
            "node-2": {
                "id": "node-2",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Response from encoding test"]},
                    "create_time": 1700000060.0,
                },
                "parent": "node-1",
                "children": [],
            },
        },
        "current_node": "node-2",
    }


class EncodingFixtureBuilder:
    """Build ZIP files with controlled encoding edge cases for acquisition tests."""

    @staticmethod
    def bom_utf8_json_zip(dest: Path) -> Path:
        """ZIP containing a JSON file with UTF-8 BOM prefix.

        Tests: json.load() handling of BOM in non-JSONL files inside ZIPs.
        """
        conv = _valid_chatgpt_conversation()
        content = b"\xef\xbb\xbf" + json.dumps(conv).encode("utf-8")

        zip_path = dest / "bom_utf8.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", content)
        return zip_path

    @staticmethod
    def bom_in_jsonl_zip(dest: Path) -> Path:
        """ZIP containing JSONL where individual lines have embedded BOM chars.

        Tests: _decode_json_bytes() BOM stripping on per-line basis.
        """
        header = _valid_codex_session_header()
        records = [_valid_codex_record(i, "user" if i % 2 == 0 else "assistant") for i in range(4)]

        lines = []
        lines.append(json.dumps(header).encode("utf-8"))
        for i, record in enumerate(records):
            line = json.dumps(record).encode("utf-8")
            if i % 2 == 0:
                line = b"\xef\xbb\xbf" + line  # Add BOM to every other line
            lines.append(line)

        zip_path = dest / "bom_jsonl.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("session.jsonl", b"\n".join(lines) + b"\n")
        return zip_path

    @staticmethod
    def mixed_line_endings_zip(dest: Path) -> Path:
        """ZIP with JSONL using mixed CRLF/LF/CR line endings.

        Tests: line iteration handles all common line ending styles.
        """
        header = _valid_codex_session_header()
        records = [_valid_codex_record(i, "user" if i % 2 == 0 else "assistant") for i in range(4)]

        parts = []
        parts.append(json.dumps(header).encode("utf-8"))
        endings = [b"\r\n", b"\n", b"\r", b"\n"]  # mixed
        for record, ending in zip(records, endings, strict=True):
            parts.append(json.dumps(record).encode("utf-8") + ending)

        content = parts[0] + b"\n"
        for part in parts[1:]:
            content += part

        zip_path = dest / "mixed_endings.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("session.jsonl", content)
        return zip_path

    @staticmethod
    def partial_corruption_zip(dest: Path) -> Path:
        """ZIP with mix of valid JSON entries and corrupt entries.

        Tests: valid entries parsed despite corrupt siblings in same ZIP.
        """
        valid_conv = json.dumps(_valid_chatgpt_conversation()).encode("utf-8")
        corrupt = b'{"broken": true, "no_closing_brace'

        zip_path = dest / "partial_corrupt.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("valid.json", valid_conv)
            zf.writestr("corrupt.json", corrupt)
        return zip_path

    @staticmethod
    def utf16_bom_json_zip(dest: Path) -> Path:
        """ZIP containing a JSON file encoded as UTF-16 with BOM.

        Tests: _decode_json_bytes() multi-encoding fallback for non-UTF-8.
        """
        conv = _valid_chatgpt_conversation()
        content = json.dumps(conv).encode("utf-16")  # includes BOM

        zip_path = dest / "utf16_bom.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", content)
        return zip_path
