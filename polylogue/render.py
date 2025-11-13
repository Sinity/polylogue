import json
import math
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import frontmatter
except ImportError:
    from ._vendor import frontmatter  # type: ignore

from .util import sanitize_filename


def _human_size(num: Optional[int]) -> Optional[str]:
    if num is None or num <= 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(math.log(num, 1024)) if num > 0 else 0, len(units) - 1)
    value = num / (1024 ** idx)
    return f"{value:.2f} {units[idx]}"


def _encode_metadata_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value).replace("\n", " ")
    return f'"{text}"'


DRIVE_LINK_RE = re.compile(r"https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)")


@dataclass
class AttachmentInfo:
    name: str
    link: str
    local_path: Optional[Path]
    size_bytes: Optional[int]
    remote: bool


@dataclass
class MarkdownDocument:
    body: str
    metadata: Dict[str, Any]
    attachments: List[AttachmentInfo]
    stats: Dict[str, Any]

    def to_markdown(self) -> str:
        post = frontmatter.Post(self.body, **self.metadata)
        return frontmatter.dumps(post)


def _iter_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_values(v)
    else:
        yield obj


def extract_drive_ids(obj: Any) -> List[str]:
    ids: List[str] = []
    def visit(o: Any):
        if isinstance(o, dict):
            if "driveDocument" in o and isinstance(o["driveDocument"], dict):
                _id = o["driveDocument"].get("id")
                if isinstance(_id, str):
                    ids.append(_id)
            if "driveImage" in o and isinstance(o["driveImage"], dict):
                _id = o["driveImage"].get("id")
                if isinstance(_id, str):
                    ids.append(_id)
            for k, v in o.items():
                if k in ("fileId", "documentId", "driveId") and isinstance(v, str):
                    ids.append(v)
                visit(v)
        elif isinstance(o, list):
            for it in o:
                visit(it)
        elif isinstance(o, str):
            for m in DRIVE_LINK_RE.finditer(o):
                ids.append(m.group(1))
    visit(obj)
    seen = set()
    out = []
    for _id in ids:
        if _id not in seen:
            seen.add(_id)
            out.append(_id)
    return out


def _render_content_parts(cont: List[Dict[str, Any]]) -> str:
    fragments: List[str] = []
    for part in cont:
        if not isinstance(part, dict):
            continue
        p_type = part.get("type")
        if p_type == "text":
            t = part.get("text")
            if isinstance(t, str):
                fragments.append(t)
        elif p_type == "code":
            code_text = part.get("text") or ""
            language = part.get("mimeType") or part.get("language") or ""
            fragments.append(f"```{language}\n{code_text}\n```".strip())
        elif p_type == "list":
            items = part.get("items") or []
            list_lines: List[str] = []
            for item in items:
                if isinstance(item, str):
                    list_lines.append(f"- {item}")
                elif isinstance(item, dict):
                    txt = item.get("text") or json.dumps(item)
                    list_lines.append(f"- {txt}")
            if list_lines:
                fragments.append("\n".join(list_lines))
        elif p_type == "table":
            rows = part.get("rows") or []
            if rows:
                header = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []

                def _normalise(row: Iterable[Any]) -> List[str]:
                    cells: List[str] = []
                    for cell in row:
                        if isinstance(cell, str):
                            cells.append(cell)
                        else:
                            cells.append(json.dumps(cell))
                    return cells

                header_cells = _normalise(header)
                table_lines = ["| " + " | ".join(header_cells) + " |"]
                table_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
                for data in data_rows:
                    table_lines.append("| " + " | ".join(_normalise(data)) + " |")
                fragments.append("\n".join(table_lines))
    return "\n\n".join(fragments)


def _count_words(text: Optional[str]) -> int:
    if not text:
        return 0
    return len([word for word in text.split() if word])


def content_text_from_entry(entry: Dict[str, Any]) -> str:
    msg = entry.get("message")
    if isinstance(msg, dict):
        cont = msg.get("content")
        if isinstance(cont, list):
            rendered = _render_content_parts(cont)
            if rendered:
                return rendered
    if isinstance(entry.get("text"), str):
        return entry["text"]
    texts: List[str] = []
    for v in _iter_values(entry):
        if isinstance(v, str) and len(v) > 0 and len(v) < 20000 and not DRIVE_LINK_RE.search(v):
            texts.append(v)
    return "\n".join(texts[:1]) if texts else ""


def entry_role(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("role"), str):
        return entry["role"]
    if isinstance(entry.get("type"), str):
        t = entry["type"].lower()
        if t in ("user", "assistant", "model", "system"):
            return t
    return "user" if entry.get("isUser", False) else "model"


def per_chunk_remote_links(chunks: List[Dict[str, Any]]) -> Dict[int, List[Tuple[str, str]]]:
    out: Dict[int, List[Tuple[str, str]]] = {}
    for idx, ch in enumerate(chunks):
        ids = extract_drive_ids(ch)
        if not ids:
            continue
        out[idx] = []
        for fid in ids:
            url = f"https://drive.google.com/file/d/{fid}"
            out[idx].append((f"Drive {fid}", url))
    return out


def remote_attachment_info(per_chunk_links: Dict[int, List[Tuple[str, Union[Path, str]]]]) -> List[AttachmentInfo]:
    infos: List[AttachmentInfo] = []
    seen: set[Tuple[str, str]] = set()
    for links in per_chunk_links.values():
        for name, link in links:
            if isinstance(link, Path):
                target = link.as_posix()
                remote = False
            else:
                target = str(link)
                remote = True
            key = (name, target)
            if key in seen:
                continue
            seen.add(key)
            infos.append(
                AttachmentInfo(
                    name=name,
                    link=target,
                    local_path=link if isinstance(link, Path) else None,
                    size_bytes=None,
                    remote=remote,
                )
            )
    return infos


def build_markdown_from_jsonl(
    entries: List[Dict[str, Any]],
    attachment_links: Dict[int, List[Tuple[str, Union[Path, str]]]],
    output_dir: Path,
    title: str,
    source_file_id: Optional[str],
    modified_time: Optional[str],
    created_time: Optional[str],
    run_settings: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Any]] = None,
    source_mime: Optional[str] = None,
    source_size: Optional[int] = None,
) -> str:
    md_parts: List[str] = []
    md_parts.append("---\n")
    md_parts.append(f"title: \"{title}\"\n")
    if source_file_id:
        md_parts.append(f"sourceId: {source_file_id}\n")
    if modified_time:
        md_parts.append(f"sourceModifiedTime: {modified_time}\n")
    if created_time:
        md_parts.append(f"sourceCreatedTime: {created_time}\n")
    if source_mime:
        md_parts.append(f"sourceMimeType: {source_mime}\n")
    if source_size is not None:
        md_parts.append(f"sourceSizeBytes: {source_size}\n")
    if run_settings:
        model = run_settings.get("model")
        if model:
            md_parts.append(f"model: \"{model}\"\n")
        for k in ("temperature", "topP", "topK", "maxOutputTokens"):
            if k in run_settings:
                md_parts.append(f"{k}: {run_settings[k]}\n")
    md_parts.append("---\n\n")

    for idx, entry in enumerate(entries):
        role = entry_role(entry)
        text = content_text_from_entry(entry).strip()
        header = "User" if role.lower() in ("user", "system") else "Model"
        md_parts.append(f"> [!INFO]+ {header}\n")
        if text:
            for line in text.splitlines():
                md_parts.append(f"> {line}\n")
        links = attachment_links.get(idx, [])
        for (name, relpath) in links:
            if hasattr(relpath, "as_posix"):
                enc = urllib.parse.quote(relpath.as_posix())
                md_parts.append(f"> - [{name}]({enc})\n")
            else:
                md_parts.append(f"> - [{name}]({relpath})\n")
        md_parts.append("\n")
    return "".join(md_parts)


def _prepare_chunks_for_render(
    chunks: List[Dict[str, Any]],
    per_chunk_links: Optional[Dict[int, List[Tuple[str, Union[Path, str]]]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[str, Union[Path, str]]]]]:
    """Normalise chunk payloads and drop empty entries prior to rendering."""

    links_map = dict(per_chunk_links or {})
    filtered_chunks: List[Dict[str, Any]] = []
    filtered_links: Dict[int, List[Tuple[str, Union[Path, str]]]] = {}

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        # Coerce ``text`` from structured ``content`` segments when available.
        text = chunk.get("text")
        if (text is None or not str(text).strip()) and isinstance(chunk.get("content"), list):
            rendered = _render_content_parts(chunk["content"])
            if rendered:
                chunk = dict(chunk)
                chunk["text"] = rendered
                text = rendered
        text_str = (text or "").strip()
        links = links_map.get(idx, [])
        has_links = bool(links)
        has_drive_refs = any(
            key in chunk for key in ("driveDocument", "driveImage", "attachments", "files", "driveAttachment")
        )
        has_tool_data = any(
            chunk.get(key)
            for key in (
                "toolCall",
                "toolCalls",
                "functionCall",
                "call",
                "toolResult",
                "toolResults",
                "arguments",
            )
        )
        if not (text_str or has_links or has_drive_refs or has_tool_data):
            continue
        new_idx = len(filtered_chunks)
        filtered_chunks.append(chunk)
        if has_links:
            filtered_links[new_idx] = links

    return filtered_chunks, filtered_links


def build_markdown_from_chunks(
    chunks: List[Dict[str, Any]],
    per_chunk_links: Dict[int, List[Tuple[str, Union[Path, str]]]],
    title: str,
    source_file_id: Optional[str],
    modified_time: Optional[str],
    created_time: Optional[str],
    run_settings: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Any]] = None,
    source_mime: Optional[str] = None,
    source_size: Optional[int] = None,
    collapse_threshold: int = 10,
    extra_yaml: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[AttachmentInfo]] = None,
) -> MarkdownDocument:
    # Drop empty chunks before computing stats or rendering so counts match output.
    chunks, per_chunk_links = _prepare_chunks_for_render(chunks, per_chunk_links)

    metadata: Dict[str, Any] = {"title": title}
    if source_file_id:
        metadata["sourceId"] = source_file_id
    if modified_time:
        metadata["sourceModifiedTime"] = modified_time
    if created_time:
        metadata["sourceCreatedTime"] = created_time
    if source_mime:
        metadata["sourceMimeType"] = source_mime
    if source_size is not None:
        metadata["sourceSizeBytes"] = source_size
    if run_settings:
        model = run_settings.get("model")
        if model:
            metadata["model"] = model
        for k in ("temperature", "topP", "topK", "maxOutputTokens"):
            if k in run_settings:
                metadata[k] = run_settings[k]
    stats: Dict[str, Any] = {}
    try:
        total_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks)
        total_words = sum(_count_words(c.get("text")) for c in chunks)
        model_turns = sum(1 for c in chunks if c.get("role") == "model")
        user_turns = sum(1 for c in chunks if c.get("role") == "user")
        model_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks if c.get("role") == "model")
        user_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks if c.get("role") == "user")
        model_words = sum(_count_words(c.get("text")) for c in chunks if c.get("role") == "model")
        user_words = sum(_count_words(c.get("text")) for c in chunks if c.get("role") == "user")
        thought_blocks = sum(1 for c in chunks if c.get("role") == "model" and c.get("isThought", False))
        att_docs = sum(1 for c in chunks if c.get("role") == "user" and "driveDocument" in c)
        att_imgs = sum(1 for c in chunks if c.get("role") == "user" and "driveImage" in c)
        stats.update(
            {
                "chunkCount": len(chunks),
                "totalTokensApprox": total_tokens,
                "inputTokensApprox": user_tokens,
                "outputTokensApprox": model_tokens,
                "totalWordsApprox": total_words,
                "inputWordsApprox": user_words,
                "outputWordsApprox": model_words,
                "userTurns": user_turns,
                "modelTurns": model_turns,
                "thoughtBlocks": thought_blocks,
                "attachmentDocCount": att_docs,
                "attachmentImageCount": att_imgs,
            }
        )
    except Exception:
        pass
    metadata.update(stats)
    if extra_yaml:
        metadata.update(extra_yaml)
    if citations:
        metadata["citations"] = []
        for cit in citations:
            try:
                uri = cit.get("uri") if isinstance(cit, dict) else str(cit)
                if uri:
                    metadata["citations"].append(uri)
            except Exception:
                continue

    first_ts, last_ts = _conversation_time_bounds(chunks)
    if first_ts and "firstMessageTime" not in metadata:
        metadata["firstMessageTime"] = first_ts
    if last_ts and "lastMessageTime" not in metadata:
        metadata["lastMessageTime"] = last_ts
    if first_ts and last_ts and "conversationDurationSeconds" not in metadata:
        try:
            start_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            duration = max(int((end_dt - start_dt).total_seconds()), 0)
            metadata["conversationDurationSeconds"] = duration
        except Exception:
            pass

    def fmt_text_block(tag: str, text: str, fold: Optional[str] = None, timestamp: Optional[str] = None) -> str:
        if text is None:
            text = ""
        text = text.strip()
        lines = text.splitlines()
        fold_char = fold if fold is not None else "+"
        header = tag
        if timestamp:
            header = f"{tag} · {timestamp}"
        out = [f"> [!INFO]{fold_char} {header}\n"]
        if text:
            for ln in lines:
                out.append(f"> {ln}\n")
        return "".join(out)

    parts: List[str] = []
    chunk_datetimes: List[Optional[datetime]] = []
    origin_time: Optional[datetime] = None
    for chunk in chunks:
        dt = _coerce_datetime(chunk.get("timestamp"))
        chunk_datetimes.append(dt)
        if dt is not None and origin_time is None:
            origin_time = dt

    i = 0
    while i < len(chunks):
        c = chunks[i]
        role = c.get("role", "model")
        is_thought = bool(c.get("isThought", False))
        if role == "user":
            user_text = c.get("text")
            if not user_text and isinstance(c.get("content"), list):
                user_text = _render_content_parts(c.get("content"))
            links = per_chunk_links.get(i, [])
            ts_label = _format_timestamp(chunk_datetimes[i], origin_time)
            if user_text:
                parts.append(fmt_text_block("User", user_text, "+", ts_label))
            elif "driveDocument" in c:
                header = "User (attachment)"
                if ts_label:
                    header += f" · {ts_label}"
                parts.append(f"> [!QUOTE]+ {header}\n")
            elif "driveImage" in c:
                header = "User (image)"
                if ts_label:
                    header += f" · {ts_label}"
                parts.append(f"> [!TIP]+ {header}\n")
            else:
                # Nothing meaningful to render for this chunk; skip it entirely.
                i += 1
                continue
            for name, relpath in links:
                if hasattr(relpath, "as_posix"):
                    enc = urllib.parse.quote(relpath.as_posix())
                    parts.append(f"> - [{name}]({enc})\n")
                else:
                    parts.append(f"> - [{name}]({relpath})\n")
            parts.append("\n")
            i += 1
            continue
        if is_thought and (i + 1) < len(chunks) and chunks[i + 1].get("role") == "model" and not chunks[i + 1].get("isThought", False):
            thought = c.get("text", "") or ""
            if not thought and isinstance(c.get("content"), list):
                thought = _render_content_parts(c.get("content"))
            resp_chunk = chunks[i + 1]
            response = resp_chunk.get("text", "") or ""
            if not response and isinstance(resp_chunk.get("content"), list):
                response = _render_content_parts(resp_chunk.get("content"))
            resp_chunk = chunks[i + 1]
            fold = "-" if (collapse_threshold > 0 and len(response.splitlines()) > collapse_threshold) else "+"
            header = "Model"
            fr = resp_chunk.get("finishReason")
            if fr:
                header += f" (Finish: {fr})"
            resp_ts = _format_timestamp(chunk_datetimes[i + 1], origin_time)
            if "branchParent" in resp_chunk:
                bp = resp_chunk.get("branchParent") or {}
                if isinstance(bp, dict):
                    disp = bp.get("displayName") or bp.get("id")
                else:
                    disp = str(bp) if bp else None
                if disp:
                    header += f" (Parent: {disp})"
            if resp_ts:
                header += f" · {resp_ts}"
            parts.append(f"> [!INFO]{fold} {header}\n")
            if thought.strip():
                parts.append(f"> > [!QUESTION]- Model Thought\n")
                for ln in thought.splitlines():
                    parts.append(f"> > {ln}\n")
                parts.append("> \n")
            if response.strip():
                for ln in response.splitlines():
                    parts.append(f"> {ln}\n")
            links = per_chunk_links.get(i, []) + per_chunk_links.get(i + 1, [])
            for name, relpath in links:
                if hasattr(relpath, "as_posix"):
                    enc = urllib.parse.quote(relpath.as_posix())
                    parts.append(f"> - [{name}]({enc})\n")
                else:
                    parts.append(f"> - [{name}]({relpath})\n")
            parts.append("\n")
            i += 2
            continue
        else:
            txt = c.get("text", "") or ""
            if not txt and isinstance(c.get("content"), list):
                txt = _render_content_parts(c.get("content"))
            fold = "-" if (collapse_threshold > 0 and len(txt.splitlines()) > collapse_threshold) else "+"
            header = "Model"
            fr = c.get("finishReason")
            if fr:
                header += f" (Finish: {fr})"
            if "branchParent" in c:
                bp = c.get("branchParent") or {}
                if isinstance(bp, dict):
                    disp = bp.get("displayName") or bp.get("id")
                else:
                    disp = str(bp) if bp else None
                if disp:
                    header += f" (Parent: {disp})"
            if not txt.strip():
                i += 1
                continue
            parts.append(fmt_text_block(header, txt, fold, _format_timestamp(chunk_datetimes[i], origin_time)))
            links = per_chunk_links.get(i, [])
            for name, relpath in links:
                if hasattr(relpath, "as_posix"):
                    enc = urllib.parse.quote(relpath.as_posix())
                    parts.append(f"> - [{name}]({enc})\n")
                else:
                    parts.append(f"> - [{name}]({relpath})\n")
            parts.append("\n")
            i += 1

    attachments_list = attachments or []
    if attachments_list:
        parts.append("## Attachments\n\n")
        for info in attachments_list:
            size = _human_size(info.size_bytes)
            label = info.name
            target = info.link
            if info.local_path:
                target = urllib.parse.quote(info.local_path.as_posix())
            line = f"- [{label}]({target})"
            if size:
                line += f" — {size}"
            if info.remote:
                line += " (remote)"
            parts.append(line + "\n")
        parts.append("\n")

    body = "".join(parts)
    attachment_count = len(attachments_list)
    attachment_bytes = sum(info.size_bytes or 0 for info in attachments_list)
    stats["attachments"] = attachment_count
    stats["attachmentBytes"] = attachment_bytes
    if attachments_list:
        metadata["attachments"] = [
            {
                "name": info.name,
                "link": info.link,
                "local": info.local_path.as_posix() if info.local_path else None,
                "size": info.size_bytes,
                "remote": info.remote,
            }
            for info in attachments_list
        ]
        metadata["attachmentCount"] = attachment_count
        metadata["attachmentBytes"] = attachment_bytes
    return MarkdownDocument(body=body, metadata=metadata, attachments=attachments_list, stats=stats)


def _conversation_time_bounds(chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    timestamps: List[datetime] = []
    for chunk in chunks:
        dt = _coerce_datetime(chunk.get("timestamp"))
        if dt is not None:
            timestamps.append(dt)
    if not timestamps:
        return None, None
    timestamps.sort()
    first = timestamps[0].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    last = timestamps[-1].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return first, last


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.isdigit():
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _format_timestamp(dt: Optional[datetime], base: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    utc_dt = dt.astimezone(timezone.utc)
    timespec = "seconds" if utc_dt.microsecond == 0 else "milliseconds"
    iso = utc_dt.isoformat(timespec=timespec).replace("+00:00", "Z")
    rel = _format_relative_delta(utc_dt - base) if base is not None else None
    if rel:
        return f"{iso} ({rel})"
    return iso


def _format_relative_delta(delta: Optional[timedelta]) -> Optional[str]:
    if delta is None:
        return None
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return None
    parts: List[str] = []
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return "+" + " ".join(parts)
