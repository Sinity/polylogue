#!/usr/bin/env python3

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import datetime
import urllib.parse

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    HAS_GOOGLE = True
except Exception:  # ModuleNotFoundError or other import-time failures
    HAS_GOOGLE = False


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"


def colorize(text: str, color: str) -> str:
    colors = {
        "reset": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "grey": "\033[90m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}" if sys.stderr.isatty() else text


def sanitize_filename(filename: str) -> str:
    sanitized = "".join(c for c in filename if ord(c) >= 32)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", sanitized)
    sanitized = sanitized.strip(". ")
    max_len = 200
    encoded = sanitized.encode("utf-8")
    if len(encoded) > max_len:
        sanitized = encoded[:max_len].decode("utf-8", errors="ignore")
    if not sanitized:
        sanitized = "_unnamed_"
    return sanitized


def get_drive_service(credentials_path: Path, verbose: bool = False):
    if not HAS_GOOGLE:
        print(colorize("Google API libraries not available. Install or use --local-dir mode.", "red"), file=sys.stderr)
        return None
    creds = None
    token_path = credentials_path.parent / TOKEN_FILE
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            if verbose:
                print(colorize(f"Loaded token from {token_path}", "magenta"), file=sys.stderr)
        except Exception as e:
            print(colorize(f"Warning: could not load token: {e}", "yellow"), file=sys.stderr)
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                if verbose:
                    print(colorize("Refreshed expired credentials", "magenta"), file=sys.stderr)
            except Exception as e:
                print(colorize(f"Token refresh failed: {e}", "red"), file=sys.stderr)
                try:
                    token_path.unlink(missing_ok=True)
                except Exception:
                    pass
                creds = None
        if not creds:
            if not credentials_path.is_file():
                print(colorize(f"Missing credentials at {credentials_path}", "red"), file=sys.stderr)
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            print(colorize("Authenticate in browser/console as prompted...", "cyan"), file=sys.stderr)
            creds = flow.run_console()
            try:
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
                if verbose:
                    print(colorize(f"Saved token to {token_path}", "magenta"), file=sys.stderr)
            except Exception:
                pass

    try:
        return build("drive", "v3", credentials=creds)
    except HttpError as e:
        print(colorize(f"Drive service error: {e}", "red"), file=sys.stderr)
        return None


def drive_get_file_meta(service, file_id: str, fields: str = "id, name, mimeType, modifiedTime, createdTime, size") -> Optional[Dict[str, Any]]:
    try:
        return service.files().get(fileId=file_id, fields=fields).execute()
    except HttpError as e:
        print(colorize(f"Failed to get file meta {file_id}: {e}", "red"), file=sys.stderr)
        return None


def drive_download_file(service, file_id: str, target_path: Path, force: bool = False, verbose: bool = False) -> bool:
    if target_path.exists() and not force:
        if verbose:
            print(colorize(f"Skip download (exists): {target_path}", "grey"), file=sys.stderr)
        return True
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(fh.getvalue())
        return True
    except HttpError as e:
        print(colorize(f"Download failed {file_id}: {e}", "red"), file=sys.stderr)
        return False


def find_folder_id(service, folder_name: str) -> Optional[str]:
    try:
        resp = service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false",
            fields="files(id, name)",
            pageSize=10,
        ).execute()
        files = resp.get("files", [])
        if not files:
            return None
        return files[0]["id"]
    except HttpError as e:
        print(colorize(f"Folder lookup failed: {e}", "red"), file=sys.stderr)
        return None


def list_children(service, folder_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page_token = None
    while True:
        try:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, size)",
                pageSize=1000,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            print(colorize(f"List children failed: {e}", "red"), file=sys.stderr)
            break
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def detect_jsonl(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    sample = lines[:10]
    ok = 0
    for ln in sample:
        s = ln.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return False
        try:
            json.loads(s)
            ok += 1
        except Exception:
            return False
    return ok >= max(2, len(sample) // 2)


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


def process_local_directory(local_dir: Path, out_dir: Path, verbose: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted([p for p in local_dir.iterdir() if p.is_file()])
    for p in entries:
        # Chats in this corpus: names without extension
        name = p.name
        if "." in name and p.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            if verbose:
                print(colorize(f"Skip unreadable: {p}", "yellow"), file=sys.stderr)
            continue

        # Try JSON -> chunks
        obj = None
        chunks = None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("chunkedPrompt", {}).get("chunks"), list):
                chunks = obj["chunkedPrompt"]["chunks"]
        except Exception:
            pass

        md_path = out_dir / f"{sanitize_filename(p.stem)}.md"
        if chunks is not None:
            # Build remote links only (no downloads in local mode)
            links = per_chunk_remote_links(chunks)
            attach_count = sum(len(v) for v in links.values())
            stat = p.stat()
            local_mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            local_ctime = datetime.datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime)).isoformat()
            # extra YAML
            extra_yaml = {
                "localModifiedTime": local_mtime,
                "localCreatedTime": local_ctime,
                "attachmentCount": attach_count,
                "hasSystemInstruction": bool(obj.get("systemInstruction")) if isinstance(obj, dict) else False,
                "hasPendingInputs": bool(obj.get("chunkedPrompt", {}).get("pendingInputs")) if isinstance(obj, dict) else False,
            }
            md = build_markdown_from_chunks(
                chunks,
                links,
                title=p.stem,
                source_file_id=None,
                modified_time=None,
                created_time=None,
                run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
                source_mime=None,
                source_size=None,
                extra_yaml=extra_yaml,
            )
            md_path.write_text(md, encoding="utf-8")
            try:
                os.utime(md_path, (stat.st_mtime, stat.st_mtime))
            except Exception:
                pass
            if verbose:
                print(colorize(f"Rendered local JSON: {md_path}", "green"), file=sys.stderr)
            continue

        # No jq fallback; proceed to plain text fallback

        # Fallback: plain text as a single block
        lines = text.strip().splitlines()
        md = "---\n" + f"title: \"{p.stem}\"\n---\n\n" + "> [!INFO]+ Content\n" + "\n".join(
            [f"> {ln}" for ln in lines[:10000]]
        ) + "\n"
        md_path.write_text(md, encoding="utf-8")
        if verbose:
            print(colorize(f"Rendered as plain text: {md_path}", "yellow"), file=sys.stderr)



def is_chat_candidate(meta: Dict[str, Any]) -> bool:
    name = meta.get("name", "")
    mime = meta.get("mimeType", "")
    # Rule: chat logs are JSONL files without extension (no dot in name)
    if "." in name:
        return False
    # Exclude folders and Google Docs types
    if mime.startswith("application/vnd.google-apps."):
        return False
    return True


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
    # Common nested forms
    def visit(o: Any):
        if isinstance(o, dict):
            # Known keys
            if "driveDocument" in o and isinstance(o["driveDocument"], dict):
                _id = o["driveDocument"].get("id")
                if isinstance(_id, str):
                    ids.append(_id)
            if "driveImage" in o and isinstance(o["driveImage"], dict):
                _id = o["driveImage"].get("id")
                if isinstance(_id, str):
                    ids.append(_id)
            # Generic keys
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
        else:
            return
    visit(obj)
    # Deduplicate preserving order
    seen = set()
    out = []
    for _id in ids:
        if _id not in seen:
            seen.add(_id)
            out.append(_id)
    return out


def jsonl_lines(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            items.append(json.loads(s))
        except Exception:
            # Non-JSON line; store as a wrapper
            items.append({"type": "text", "text": ln})
    return items


def parse_rfc3339_to_epoch(ts: Optional[str]) -> Optional[float]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Replace Z with +00:00 for fromisoformat
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def parse_input_time_to_epoch(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        # Date only
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            dt = datetime.datetime.fromisoformat(s)
            # Treat as midnight UTC
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        # RFC3339 with optional Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def content_text_from_entry(entry: Dict[str, Any]) -> str:
    # Try multiple shapes to extract user/assistant text
    # 1) Gemini style: {"message": {"content": [{"type": "text", "text": "..."}, ...]}}
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
    # 2) Direct shape: {"text": "..."}
    if isinstance(entry.get("text"), str):
        return entry["text"]
    # 3) Anything string-like inside
    texts: List[str] = []
    for v in _iter_values(entry):
        if isinstance(v, str) and len(v) > 0 and len(v) < 20000:
            # Keep only visibly textual strings
            if not DRIVE_LINK_RE.search(v):
                texts.append(v)
    return "\n".join(texts[:1]) if texts else ""


def entry_role(entry: Dict[str, Any]) -> str:
    # Try common fields
    if isinstance(entry.get("role"), str):
        return entry["role"]
    if isinstance(entry.get("type"), str):
        t = entry["type"].lower()
        if t in ("user", "assistant", "model", "system"): 
            return t
    # Fallback
    return "user" if entry.get("isUser", False) else "model"


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
    # YAML frontmatter
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
        # Attachments linked for this entry
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
    # extra_yaml appended later to group all meta together
    # Token and turn stats (approximate)
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
    if extra_yaml:
        for k, v in (extra_yaml or {}).items():
            try:
                parts.append(f"{k}: {v}\n")
            except Exception:
                continue
    parts.append("---\n\n")

    def fmt_text_block(tag: str, text: str, fold: Optional[str] = None) -> str:
        if text is None:
            text = ""
        text = text.strip()
        lines = text.splitlines()
        if fold is None:
            fold_char = "+"
        else:
            fold_char = fold
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

        # model side, optionally group thought + response
        if is_thought and (i + 1) < len(chunks) and chunks[i + 1].get("role") == "model" and not chunks[i + 1].get("isThought", False):
            thought = c.get("text", "") or ""
            response = chunks[i + 1].get("text", "") or ""
            resp_chunk = chunks[i + 1]
            # Collapse long responses
            fold = "-" if (collapse_threshold > 0 and len(response.splitlines()) > collapse_threshold) else "+"
            # Build header with finish reason and optional parent label
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
            # single model chunk
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
                enc = urllib.parse.quote(relpath.as_posix())
                parts.append(f"> - [{name}]({enc})\n")
            parts.append("\n")
            i += 1

    return "".join(parts)


def sync_folder(
    service,
    folder_id: str,
    out_dir: Path,
    credentials_path: Path,
    force: bool = False,
    prune: bool = False,
    verbose: bool = False,
    remote_links: bool = False,
    since: Optional[str] = None,
    until: Optional[str] = None,
    name_filter: Optional[str] = None,
    collapse_threshold: int = 10,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    children = list_children(service, folder_id)
    chats = [c for c in children if is_chat_candidate(c)]
    # Apply filters
    if name_filter:
        try:
            rx = re.compile(name_filter)
            chats = [c for c in chats if rx.search(c.get("name", "") or "")]
        except re.error:
            print(colorize("Invalid --name-filter regex; ignoring.", "yellow"), file=sys.stderr)
    s_epoch = parse_input_time_to_epoch(since)
    u_epoch = parse_input_time_to_epoch(until)
    if s_epoch or u_epoch:
        fchats = []
        for c in chats:
            mt = parse_rfc3339_to_epoch(c.get("modifiedTime"))
            if mt is None:
                continue
            if s_epoch and mt < s_epoch:
                continue
            if u_epoch and mt > u_epoch:
                continue
            fchats.append(c)
        chats = fchats

    # Handle duplicate names by appending a short id suffix
    name_counts: Dict[str, int] = {}
    safe_name_to_meta: Dict[str, Dict[str, Any]] = {}
    for meta in chats:
        base = sanitize_filename(meta["name"]) or "chat"
        if base in name_counts:
            name_counts[base] += 1
            safe = f"{base}__{meta['id'][:6]}"
        else:
            name_counts[base] = 1
            safe = base
        safe_name_to_meta[safe] = meta

    # Optional pruning: compute expected set
    expected_md = { (out_dir / f"{name}.md") for name in safe_name_to_meta.keys() }

    for safe_name, meta in safe_name_to_meta.items():
        file_id = meta["id"]
        remote_mtime = meta.get("modifiedTime")
        remote_ctime = meta.get("createdTime") if "createdTime" in meta else None
        md_path = out_dir / f"{safe_name}.md"
        attachments_dir = out_dir / f"{safe_name}_attachments"
        meta_path = out_dir / f"{safe_name}.sync.json"

        # Skip if unchanged
        if md_path.exists() and meta_path.exists() and not force:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("sourceDriveId") == file_id and prev.get("modifiedTime") == remote_mtime:
                    if verbose:
                        print(colorize(f"Up-to-date: {md_path.name}", "grey"), file=sys.stderr)
                    continue
            except Exception:
                pass

        # Download chat JSONL
        buf = io.BytesIO()
        try:
            req = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        except HttpError as e:
            print(colorize(f"Failed to download chat {safe_name}: {e}", "red"), file=sys.stderr)
            continue

        content = buf.getvalue().decode("utf-8", errors="replace")

        # Try full JSON parse (Gemini export shape)
        chunks: Optional[List[Dict[str, Any]]] = None
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and isinstance(obj.get("chunkedPrompt", {}).get("chunks"), list):
                chunks = obj["chunkedPrompt"]["chunks"]
        except Exception:
            chunks = None

        per_index_links: Dict[int, List[Tuple[str, Union[Path, str]]]] = {}
        if chunks is not None:
            if remote_links:
                per_index_links = per_chunk_remote_links(chunks)
            else:
                # Extract and download attachments per chunk
                for idx, ch in enumerate(chunks):
                    ids = extract_drive_ids(ch)
                    if not ids:
                        continue
                    per_index_links[idx] = []
                    for att_id in ids:
                        meta_att = drive_get_file_meta(service, att_id)
                        if not meta_att:
                            continue
                        att_name = sanitize_filename(meta_att.get("name", att_id))
                        local_path = attachments_dir / att_name
                        # Decide re-download based on size and existence
                        need = True
                        if local_path.exists() and not force:
                            try:
                                if int(meta_att.get("size", 0)) > 0 and local_path.stat().st_size == int(meta_att["size"]):
                                    need = False
                            except Exception:
                                pass
                        if need:
                            ok = drive_download_file(service, att_id, local_path, force=force, verbose=verbose)
                            if not ok:
                                continue
                        # Set attachment mtime to remote modifiedTime
                        att_mtime = parse_rfc3339_to_epoch(meta_att.get("modifiedTime"))
                        if att_mtime:
                            try:
                                os.utime(local_path, (att_mtime, att_mtime))
                            except Exception:
                                pass
                        try:
                            rel = local_path.relative_to(out_dir)
                        except Exception:
                            rel = local_path
                        per_index_links[idx].append((att_name, rel))

            # Extra YAML derived from source JSON
            rs = obj.get("runSettings") if isinstance(obj, dict) else None
            extra = {
                "hasSystemInstruction": bool(obj.get("systemInstruction")) if isinstance(obj, dict) else False,
                "hasPendingInputs": bool(obj.get("chunkedPrompt", {}).get("pendingInputs")) if isinstance(obj, dict) else False,
            }
            if isinstance(rs, dict):
                if "responseMimeType" in rs:
                    extra["responseMimeType"] = rs.get("responseMimeType")
                if "safetySettings" in rs and isinstance(rs.get("safetySettings"), list):
                    extra["safetySettingsCount"] = len(rs.get("safetySettings"))

            # Build markdown using chunk-aware formatter
            md = build_markdown_from_chunks(
                chunks,
                per_index_links,
                meta.get("name", safe_name),
                file_id,
                remote_mtime,
                remote_ctime,
                run_settings=rs,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
                source_mime=meta.get("mimeType"),
                source_size=int(meta.get("size", 0)) if meta.get("size") is not None else None,
                extra_yaml=extra,
                collapse_threshold=collapse_threshold,
            )
        else:
            # Fallback: treat as JSONL or plain text lines
            entries = jsonl_lines(content)
            per_index_links = {}
            for idx, entry in enumerate(entries):
                ids = extract_drive_ids(entry)
                if not ids:
                    continue
                per_index_links[idx] = []
                for att_id in ids:
                    meta_att = drive_get_file_meta(service, att_id)
                    if not meta_att:
                        continue
                    att_name = sanitize_filename(meta_att.get("name", att_id))
                    local_path = attachments_dir / att_name
                    need = True
                    if local_path.exists() and not force:
                        try:
                            if int(meta_att.get("size", 0)) > 0 and local_path.stat().st_size == int(meta_att["size"]):
                                need = False
                        except Exception:
                            pass
                    if need:
                        ok = drive_download_file(service, att_id, local_path, force=force, verbose=verbose)
                        if not ok:
                            continue
                    att_mtime = parse_rfc3339_to_epoch(meta_att.get("modifiedTime"))
                    if att_mtime:
                        try:
                            os.utime(local_path, (att_mtime, att_mtime))
                        except Exception:
                            pass
                    try:
                        rel = local_path.relative_to(out_dir)
                    except Exception:
                        rel = local_path
                    per_index_links[idx].append((att_name, rel))

            md = build_markdown_from_jsonl(
                entries,
                per_index_links,
                out_dir,
                meta.get("name", safe_name),
                file_id,
                remote_mtime,
                remote_ctime,
                run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
                source_mime=meta.get("mimeType"),
                source_size=int(meta.get("size", 0)) if meta.get("size") is not None else None,
            )

        # Write files
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"sourceDriveId": file_id, "modifiedTime": remote_mtime, "createdTime": remote_ctime}, f, indent=2)

        # Apply remote times to the Markdown file
        mtime_epoch = parse_rfc3339_to_epoch(remote_mtime)
        if mtime_epoch:
            try:
                os.utime(md_path, (mtime_epoch, mtime_epoch))
            except Exception:
                pass

        print(colorize(f"Synced: {md_path}", "green"), file=sys.stderr)

    if prune:
        for p in out_dir.glob("*.md"):
            if p not in expected_md:
                try:
                    p.unlink()
                    att_dir = out_dir / f"{p.stem}_attachments"
                    if att_dir.exists() and att_dir.is_dir():
                        # Only remove if empty
                        try:
                            for child in att_dir.iterdir():
                                child.unlink()
                            att_dir.rmdir()
                        except Exception:
                            pass
                    meta_file = out_dir / f"{p.stem}.sync.json"
                    meta_file.unlink(missing_ok=True)
                    print(colorize(f"Pruned: {p.name}", "yellow"), file=sys.stderr)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Sync Google Drive 'AI Studio' chats or render local JSONs to Markdown (opinionated callout style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--local-dir", type=Path, default=None, help="Process local directory of chats (files without extension) into Markdown. Skips Drive access.")
    parser.add_argument("--folder-id", type=str, default=None, help="Drive folder ID for 'AI Studio'. Overrides --folder-name.")
    parser.add_argument("--folder-name", type=str, default="AI Studio", help="Drive folder name to search if --folder-id not provided.")
    parser.add_argument("--out-dir", type=Path, default=Path("gemini_synced"), help="Local output directory for Markdown + attachments.")
    parser.add_argument("--credentials", type=Path, default=Path("credentials.json"), help="Path to Google OAuth credentials.json")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-render, ignoring cache metadata.")
    parser.add_argument("--prune", action="store_true", help="Delete local Markdown not present in remote folder (safe prune).")
    parser.add_argument("--remote-links", action="store_true", help="Do not download attachments; link to Drive URLs instead.")
    parser.add_argument("--since", type=str, default=None, help="Only process Drive files modified on/after this time (YYYY-MM-DD or RFC3339).")
    parser.add_argument("--until", type=str, default=None, help="Only process Drive files modified on/before this time (YYYY-MM-DD or RFC3339).")
    parser.add_argument("--name-filter", type=str, default=None, help="Regex to include only chats whose name matches.")
    parser.add_argument("--collapse-threshold", type=int, default=10, help="Lines after which model responses are folded by default (0 disables folding).")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files or download attachments; just report actions.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")

    args = parser.parse_args()

    # Local mode branch
    if args.local_dir:
        process_local_directory(
            local_dir=args.local_dir,
            out_dir=args.out_dir,
            verbose=args.verbose,
        )
        return

    service = get_drive_service(args.credentials, verbose=args.verbose)
    if not service:
        sys.exit(2)

    folder_id = args.folder_id
    if not folder_id:
        folder_id = find_folder_id(service, args.folder_name)
        if not folder_id:
            print(colorize(f"Folder not found: {args.folder_name}", "red"), file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(colorize(f"Using folder {args.folder_name} ({folder_id})", "cyan"), file=sys.stderr)

    if args.dry_run:
        print(colorize("Dry-run: listing candidate chats...", "yellow"), file=sys.stderr)
        children = list_children(service, folder_id)
        chats = [c for c in children if is_chat_candidate(c)]
        print(f"Found {len(chats)} chat candidate(s). Output dir would be: {args.out_dir}")
        for c in chats[:20]:
            print(f"- {c['name']} ({c['id']})")
        if len(chats) > 20:
            print("- ...")
    else:
        sync_folder(
            service=service,
            folder_id=folder_id,
            out_dir=args.out_dir,
            credentials_path=args.credentials,
            force=args.force,
            prune=args.prune,
            verbose=args.verbose,
            remote_links=args.remote_links,
            since=args.since,
            until=args.until,
            name_filter=args.name_filter,
            collapse_threshold=args.collapse_threshold,
        )


if __name__ == "__main__":
    main()
