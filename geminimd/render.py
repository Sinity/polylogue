import json
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .util import sanitize_filename


DRIVE_LINK_RE = re.compile(r"https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)")


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


def content_text_from_entry(entry: Dict[str, Any]) -> str:
    msg = entry.get("message")
    if isinstance(msg, dict):
        cont = msg.get("content")
        if isinstance(cont, list):
            texts: List[str] = []
            for part in cont:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        texts.append(t)
            if texts:
                return "\n\n".join(texts)
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
        md_parts.append(f"sourceDriveId: {source_file_id}\n")
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
) -> str:
    parts: List[str] = []
    parts.append("---\n")
    parts.append(f"title: \"{title}\"\n")
    if source_file_id:
        parts.append(f"sourceDriveId: {source_file_id}\n")
    if modified_time:
        parts.append(f"sourceModifiedTime: {modified_time}\n")
    if created_time:
        parts.append(f"sourceCreatedTime: {created_time}\n")
    if run_settings:
        model = run_settings.get("model")
        if model:
            parts.append(f"model: \"{model}\"\n")
        for k in ("temperature", "topP", "topK", "maxOutputTokens"):
            if k in run_settings:
                parts.append(f"{k}: {run_settings[k]}\n")
    if source_mime:
        parts.append(f"sourceMimeType: {source_mime}\n")
    if source_size is not None:
        parts.append(f"sourceSizeBytes: {source_size}\n")
    if extra_yaml:
        for k, v in extra_yaml.items():
            try:
                parts.append(f"{k}: {v}\n")
            except Exception:
                continue
    try:
        total_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks)
        model_turns = sum(1 for c in chunks if c.get("role") == "model")
        user_turns = sum(1 for c in chunks if c.get("role") == "user")
        model_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks if c.get("role") == "model")
        user_tokens = sum(int(c.get("tokenCount", 0)) for c in chunks if c.get("role") == "user")
        thought_blocks = sum(1 for c in chunks if c.get("role") == "model" and c.get("isThought", False))
        att_docs = sum(1 for c in chunks if c.get("role") == "user" and "driveDocument" in c)
        att_imgs = sum(1 for c in chunks if c.get("role") == "user" and "driveImage" in c)
        parts.append(f"chunkCount: {len(chunks)}\n")
        parts.append(f"totalTokensApprox: {total_tokens}\n")
        parts.append(f"inputTokensApprox: {user_tokens}\n")
        parts.append(f"outputTokensApprox: {model_tokens}\n")
        parts.append(f"userTurns: {user_turns}\n")
        parts.append(f"modelTurns: {model_turns}\n")
        parts.append(f"thoughtBlocks: {thought_blocks}\n")
        parts.append(f"attachmentDocCount: {att_docs}\n")
        parts.append(f"attachmentImageCount: {att_imgs}\n")
    except Exception:
        pass
    if citations:
        parts.append("citations:\n")
        for cit in citations:
            try:
                uri = cit.get("uri") if isinstance(cit, dict) else str(cit)
                if uri:
                    parts.append(f"  - \"{uri}\"\n")
            except Exception:
                continue
    parts.append("---\n\n")

    def fmt_text_block(tag: str, text: str, fold: Optional[str] = None) -> str:
        if text is None:
            text = ""
        text = text.strip()
        lines = text.splitlines()
        fold_char = fold if fold is not None else "+"
        out = [f"> [!INFO]{fold_char} {tag}\n"]
        if text:
            for ln in lines:
                out.append(f"> {ln}\n")
        return "".join(out)

    i = 0
    while i < len(chunks):
        c = chunks[i]
        role = c.get("role", "model")
        is_thought = bool(c.get("isThought", False))
        if role == "user":
            if "text" in c:
                parts.append(fmt_text_block("User", c.get("text", ""), "+"))
            elif "driveDocument" in c:
                parts.append("> [!QUOTE]+ User (attachment)\n")
            elif "driveImage" in c:
                parts.append("> [!TIP]+ User (image)\n")
            else:
                parts.append(fmt_text_block("User", json.dumps(c), "+"))
            links = per_chunk_links.get(i, [])
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
            response = chunks[i + 1].get("text", "") or ""
            resp_chunk = chunks[i + 1]
            fold = "-" if (collapse_threshold > 0 and len(response.splitlines()) > collapse_threshold) else "+"
            header = "Model"
            fr = resp_chunk.get("finishReason")
            if fr:
                header += f" (Finish: {fr})"
            if "branchParent" in resp_chunk:
                bp = resp_chunk.get("branchParent") or {}
                disp = bp.get("displayName") or bp.get("id")
                if disp:
                    header += f" (Parent: {disp})"
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
            fold = "-" if (collapse_threshold > 0 and len(txt.splitlines()) > collapse_threshold) else "+"
            header = "Model"
            fr = c.get("finishReason")
            if fr:
                header += f" (Finish: {fr})"
            if "branchParent" in c:
                bp = c.get("branchParent") or {}
                disp = bp.get("displayName") or bp.get("id")
                if disp:
                    header += f" (Parent: {disp})"
            parts.append(fmt_text_block(header, txt, fold))
            links = per_chunk_links.get(i, [])
            for name, relpath in links:
                if hasattr(relpath, "as_posix"):
                    enc = urllib.parse.quote(relpath.as_posix())
                    parts.append(f"> - [{name}]({enc})\n")
                else:
                    parts.append(f"> - [{name}]({relpath})\n")
            parts.append("\n")
            i += 1

    return "".join(parts)

