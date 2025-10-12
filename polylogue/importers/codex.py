from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError

from ..render import AttachmentInfo, build_markdown_from_chunks
from ..document_store import persist_document
from ..util import assign_conversation_slug
from .base import ImportResult
from .utils import CHAR_THRESHOLD, LINE_THRESHOLD, PREVIEW_LINES, estimate_token_count

_DEFAULT_BASE = Path.home() / ".codex" / "sessions"


class _Payload(BaseModel):
    type: str
    role: Optional[str] = None
    content: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None
    arguments: Optional[Any] = None
    call_id: Optional[str] = None
    output: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class _ResponseItem(BaseModel):
    type: str
    payload: _Payload
    model_config = ConfigDict(extra="allow")

def _truncate_with_preview(text: str, attachment_name: str) -> str:
    lines = text.splitlines()
    if len(lines) <= PREVIEW_LINES * 2 + 1:
        return text
    head = lines[:PREVIEW_LINES]
    tail = lines[-PREVIEW_LINES:]
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
) -> ImportResult:
    candidate = Path(session_id)
    if candidate.exists() and candidate.is_file():
        session_path = candidate
    else:
        session_glob = sorted(base_dir.rglob(f"*{session_id}.jsonl"))
        if not session_glob:
            raise FileNotFoundError(f"Could not locate session {session_id} under {base_dir}")
        session_path = session_glob[0]

    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict[str, str]] = []
    call_index: Dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    title = session_path.stem
    slug = assign_conversation_slug("codex", session_id, title, id_hint=title[:8])
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = markdown_path.parent / f"{slug}_attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    with session_path.open(encoding="utf-8") as handle:
        for raw in handle:
            entry = json.loads(raw)
            try:
                model = _ResponseItem.model_validate(entry)
            except ValidationError:
                continue
            if model.type != "response_item":
                continue
            payload = model.payload
            ptype = payload.type

            if ptype == "message":
                role = payload.role or "assistant"
                canonical_role = "model" if role in {"assistant", "model"} else "user"
                parts: List[str] = []
                for seg in payload.content or []:
                    if not isinstance(seg, dict):
                        continue
                    text = seg.get("text") or ""
                    if text.startswith("<user_instructions>") or text.startswith("<environment_context>"):
                        parts = []
                        break
                    parts.append(text)
                if not parts:
                    continue
                text = "\n".join(parts)
                chunks.append(
                    {
                        "role": canonical_role,
                        "text": text,
                        "tokenCount": estimate_token_count(text, model="gpt-5-codex"),
                    }
                )

            elif ptype == "function_call":
                name = payload.name or "tool"
                args = payload.arguments or "{}"
                if not isinstance(args, str):
                    args = json.dumps(args, indent=2, ensure_ascii=False)
                chunk_text = f"Tool call `{name}`\n```json\n{args}\n```"
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="gpt-5-codex"),
                    }
                )
                call_index[payload.call_id or ""] = len(chunks) - 1

            elif ptype == "function_call_output":
                output_text = payload.output or ""
                call_id = payload.call_id or ""
                idx = call_index.get(call_id)
                if idx is not None:
                    chunks[idx]["text"] += f"\n\nOutput:\n{output_text}"
                    chunks[idx]["tokenCount"] = estimate_token_count(chunks[idx]["text"], model="gpt-5-codex")
                else:
                    chunks.append(
                        {
                            "role": "tool",
                            "text": output_text,
                            "tokenCount": estimate_token_count(output_text, model="gpt-5-codex"),
                        }
                    )

    # attachment extraction pass
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        lines = text.count("\n") + 1
        if lines <= LINE_THRESHOLD and len(text) <= CHAR_THRESHOLD:
            continue
        attachment_name = f"chunk{idx:03d}.txt"
        attachment_path = attachments_dir / attachment_name
        attachment_path.write_text(text, encoding="utf-8")
        rel = attachment_path.relative_to(markdown_path.parent)
        attachments.append(AttachmentInfo(attachment_name, str(rel), rel, attachment_path.stat().st_size, False))
        per_chunk_links.setdefault(idx, []).append((attachment_name, rel))
        chunk["text"] = _truncate_with_preview(text, attachment_name)

    metadata = {"model": "gpt-5-codex"}
    extra_yaml = {
        "sourcePlatform": "codex",
        "sessionPath": str(session_path),
    }

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=title,
        source_file_id=session_id,
        run_settings=metadata,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        source_mime="application/jsonl",
        source_size=session_path.stat().st_size,
        modified_time=None,
        created_time=None,
        citations=None,
        extra_yaml=extra_yaml,
    )

    persist_result = persist_document(
        provider="codex",
        conversation_id=session_id,
        title=title,
        document=document,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        updated_at=None,
        created_at=None,
        html=html,
        html_theme=html_theme,
        attachment_policy={
            "previewLines": PREVIEW_LINES,
            "lineThreshold": LINE_THRESHOLD,
            "charThreshold": CHAR_THRESHOLD,
        },
        extra_state={
            "sessionPath": str(session_path),
        },
        slug_hint=slug,
        id_hint=title[:8],
    )

    if persist_result.skipped:
        return ImportResult(
            markdown_path=persist_result.markdown_path,
            html_path=persist_result.html_path,
            attachments_dir=persist_result.attachments_dir,
            document=None,
            skipped=True,
            skip_reason=persist_result.skip_reason,
            dirty=persist_result.dirty,
            content_hash=persist_result.content_hash,
        )

    return ImportResult(
        markdown_path=persist_result.markdown_path,
        html_path=persist_result.html_path,
        attachments_dir=persist_result.attachments_dir,
        document=persist_result.document or document,
        dirty=persist_result.dirty,
        content_hash=persist_result.content_hash,
    )
