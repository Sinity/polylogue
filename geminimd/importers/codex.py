from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from ..util import sanitize_filename

_DEFAULT_BASE = Path.home() / ".codex" / "sessions"

_PREVIEW_LINES = 5
_LINE_THRESHOLD = 40
_CHAR_THRESHOLD = 4000


@dataclass
class CodexImportResult:
    markdown_path: Path
    html_path: Optional[Path]
    attachments_dir: Optional[Path]
    document: MarkdownDocument


def _truncate_with_preview(text: str, attachment_name: str) -> str:
    lines = text.splitlines()
    if len(lines) <= _PREVIEW_LINES * 2 + 1:
        return text
    head = lines[:_PREVIEW_LINES]
    tail = lines[-_PREVIEW_LINES:]
    preview = "\n".join(head + ["…", "", f"(Full content saved to {attachment_name})", ""] + tail)
    return preview


def import_codex_session(
    session_id: str,
    *,
    base_dir: Path = _DEFAULT_BASE,
    output_dir: Path,
    collapse_threshold: int = 24,
    html: bool = False,
    html_theme: str = "light",
) -> CodexImportResult:
    session_glob = list(base_dir.rglob(f"*{session_id}.jsonl"))
    if not session_glob:
        raise FileNotFoundError(f"Could not locate session {session_id} under {base_dir}")
    session_path = session_glob[0]

    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict[str, str]] = []
    call_index: Dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_name = sanitize_filename(session_path.stem) + ".md"
    markdown_path = output_dir / markdown_name
    attachments_dir = markdown_path.parent / f"{markdown_path.stem}_attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    with session_path.open(encoding="utf-8") as handle:
        for raw in handle:
            entry = json.loads(raw)
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload", {})
            ptype = payload.get("type")

            if ptype == "message":
                role = payload.get("role", "assistant")
                parts: List[str] = []
                for seg in payload.get("content", []):
                    if not isinstance(seg, dict):
                        continue
                    text = seg.get("text") or ""
                    if text.startswith("<user_instructions>") or text.startswith("<environment_context>"):
                        parts = []
                        break
                    parts.append(text)
                if not parts:
                    continue
                chunks.append({"role": role, "text": "\n".join(parts)})

            elif ptype == "function_call":
                name = payload.get("name", "tool")
                args = payload.get("arguments") or "{}"
                chunk_text = f"Tool call `{name}`\n```json\n{args}\n```"
                chunks.append({"role": "assistant", "text": chunk_text})
                call_index[payload.get("call_id", "")] = len(chunks) - 1

            elif ptype == "function_call_output":
                output_text = payload.get("output", "")
                call_id = payload.get("call_id", "")
                idx = call_index.get(call_id)
                if idx is not None:
                    chunks[idx]["text"] += f"\n\nOutput:\n{output_text}"
                else:
                    chunks.append({"role": "tool", "text": output_text})

    # attachment extraction pass
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        lines = text.count("\n") + 1
        if lines <= _LINE_THRESHOLD and len(text) <= _CHAR_THRESHOLD:
            continue
        attachment_name = f"chunk{idx:03d}.txt"
        attachment_path = attachments_dir / attachment_name
        attachment_path.write_text(text, encoding="utf-8")
        rel = attachment_path.relative_to(markdown_path.parent)
        attachments.append(AttachmentInfo(attachment_name, str(rel), rel, attachment_path.stat().st_size, False))
        per_chunk_links.setdefault(idx, []).append((attachment_name, rel))
        chunk["text"] = _truncate_with_preview(text, attachment_name)

    metadata = {"model": "gpt-5-codex"}

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=markdown_path.stem,
        source_file_id=session_id,
        run_settings=metadata,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        source_mime="application/jsonl",
        source_size=session_path.stat().st_size,
        modified_time=None,
        created_time=None,
        citations=None,
    )

    markdown_path.write_text(document.to_markdown(), encoding="utf-8")

    html_path: Optional[Path] = None
    if html:
        from ..html import write_html

        html_path = markdown_path.with_suffix(".html")
        write_html(document, html_path, html_theme)

    if not attachments:
        try:
            attachments_dir.rmdir()
            attachments_dir = None
        except OSError:
            pass

    return CodexImportResult(
        markdown_path=markdown_path,
        html_path=html_path,
        attachments_dir=attachments_dir,
        document=document,
    )
