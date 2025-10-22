from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..render import AttachmentInfo, build_markdown_from_chunks
from ..document_store import persist_document
from ..util import assign_conversation_slug
from .base import ImportResult
from .utils import estimate_token_count, store_large_text


DEFAULT_PROJECT_ROOT = Path.home() / ".claude" / "projects"


def import_claude_code_session(
    session_id: str,
    *,
    base_dir: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
) -> ImportResult:
    session_path = _locate_session_file(session_id, base_dir)
    if session_path is None:
        raise FileNotFoundError(f"Claude Code session {session_id} not found under {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    title = session_path.stem
    slug = assign_conversation_slug("claude-code", session_id, title, id_hint=title[:8])
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = markdown_path.parent / f"{slug}_attachments"

    chunks: List[Dict] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    attachments: List[AttachmentInfo] = []
    summaries: List[str] = []
    tool_results: Dict[str, int] = {}

    with session_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                fallback = line.strip()
                if not fallback:
                    continue
                chunk_text = "Unparsed Claude Code log line\n```\n{}```".format(fallback)
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                    }
                )
                continue
            if not isinstance(entry, dict):
                chunk_text = "Unparsed Claude Code entry\n```\n{}```".format(entry)
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                    }
                )
                continue
            etype = entry.get("type")
            if etype == "summary":
                text = entry.get("summary")
                if text:
                    summaries.append(text)
                continue
            if etype == "user":
                text = _extract_text(entry)
                chunks.append(
                    {
                        "role": "user",
                        "text": text or "",
                        "tokenCount": estimate_token_count(text or "", model="claude-code"),
                    }
                )
            elif etype == "assistant":
                text = _extract_text(entry)
                chunks.append(
                    {
                        "role": "model",
                        "text": text or "",
                        "tokenCount": estimate_token_count(text or "", model="claude-code"),
                    }
                )
            elif etype == "tool_use":
                name = entry.get("name") or entry.get("toolName") or "tool"
                input_payload = entry.get("input") or entry.get("args") or {}
                text = f"Tool call `{name}`\n```json\n{json.dumps(input_payload, indent=2, ensure_ascii=False)}\n```"
                chunks.append(
                    {
                        "role": "model",
                        "text": text,
                        "tokenCount": estimate_token_count(text, model="claude-code"),
                    }
                )
                tool_results[entry.get("id") or entry.get("toolUseId") or ""] = len(chunks) - 1
            elif etype == "tool_result":
                result_text = _extract_tool_result(entry)
                tool_id = entry.get("tool_use_id") or entry.get("toolUseId") or entry.get("id")
                idx = tool_results.get(tool_id)
                if idx is not None:
                    original = chunks[idx]["text"]
                    combined = f"{original}\n\nResult:\n{result_text}"
                    chunks[idx]["text"] = combined
                    chunks[idx]["tokenCount"] = estimate_token_count(combined, model="claude-code")
                else:
                    chunks.append(
                        {
                            "role": "tool",
                            "text": result_text,
                            "tokenCount": estimate_token_count(result_text, model="claude-code"),
                        }
                    )

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text") or ""
        preview = store_large_text(
            text,
            chunk_index=idx,
            attachments_dir=attachments_dir,
            markdown_dir=markdown_path.parent,
            attachments=attachments,
            per_chunk_links=per_chunk_links,
            prefix="claudecode",
        )
        if preview != text:
            chunk["text"] = preview

    extra_yaml = {"sourcePlatform": "claude-code", "sessionFile": str(session_path)}
    if summaries:
        extra_yaml["summaries"] = summaries

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=title,
        source_file_id=session_path.name,
        modified_time=None,
        created_time=None,
        run_settings=None,
        citations=None,
        source_mime="application/jsonl",
        source_size=session_path.stat().st_size,
        collapse_threshold=collapse_threshold,
        extra_yaml=extra_yaml,
        attachments=attachments,
    )
    document.metadata["sourceSessionPath"] = str(session_path)
    document.metadata["sourceWorkspace"] = session_path.parent.name

    persist_result = persist_document(
        provider="claude-code",
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
        attachment_policy=None,
        extra_state={
            "sessionFile": str(session_path),
            "workspace": session_path.parent.name,
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


def list_claude_code_sessions(base_dir: Path = DEFAULT_PROJECT_ROOT) -> List[Dict[str, str]]:
    sessions: List[Dict[str, str]] = []
    if not base_dir.exists():
        return sessions
    for path in base_dir.rglob("*.jsonl"):
        sessions.append(
            {
                "path": str(path),
                "name": path.stem,
                "workspace": path.parent.name,
            }
        )
    sessions.sort(key=lambda s: s["path"])
    return sessions


def _locate_session_file(session_id: str, base_dir: Path) -> Optional[Path]:
    candidate = Path(session_id)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if session_id.endswith(".jsonl"):
        path = base_dir / session_id
        if path.exists():
            return path
    if (base_dir / session_id).exists():
        return base_dir / session_id
    pattern = f"*{session_id}*.jsonl"
    matches = list(base_dir.rglob(pattern))
    if matches:
        return matches[0]
    return None


def _extract_text(entry: Dict) -> str:
    message = entry.get("message") or {}
    content = message.get("content") or []
    fragments: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    fragments.append(text)
            elif block.get("type") == "command_output":
                stdout = block.get("text") or block.get("output")
                if stdout:
                    fragments.append(f"````\n{stdout}\n````")
        elif isinstance(block, str):
            fragments.append(block)
    if not fragments and isinstance(message.get("text"), str):
        fragments.append(message["text"])
    return "\n\n".join(fragments)


def _extract_tool_result(entry: Dict) -> str:
    text = entry.get("text") or entry.get("result") or ""
    if isinstance(text, list):
        fragments: List[str] = []
        for block in text:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, dict) and block.get("text"):
                fragments.append(block["text"])
        text = "\n".join(fragments)
    return text
