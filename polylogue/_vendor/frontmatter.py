from __future__ import annotations

import json
from typing import Dict, Tuple


class Post:
    def __init__(self, content: str, **metadata: object):
        self.content = content
        self.metadata = dict(metadata)


def _parse(text: str) -> Tuple[Dict[str, object], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta_lines = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            i += 1
            break
        meta_lines.append(lines[i])
        i += 1
    body = "\n".join(lines[i:])
    metadata: Dict[str, object] = {}
    for raw in meta_lines:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip()
        if value.startswith("\"") and value.endswith("\""):
            metadata[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            metadata[key] = value.lower() == "true"
        else:
            try:
                metadata[key] = json.loads(value)
            except Exception:
                metadata[key] = value
    return metadata, body


def _dump(metadata: Dict[str, object], body: str) -> str:
    header_lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            encoded = "true" if value else "false"
        else:
            encoded = json.dumps(value)
        header_lines.append(f"{key}: {encoded}")
    header_lines.append("---\n")
    return "\n".join(header_lines) + body


def loads(text: str) -> Post:
    metadata, body = _parse(text)
    return Post(content=body, **metadata)


def dumps(post: Post) -> str:
    return _dump(post.metadata, post.content)
