from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import frontmatter


_PATH_KEYS = {
    "outputPath",
    "htmlPath",
    "attachmentsDir",
    "sourceExportPath",
    "sessionPath",
    "sessionFile",
    "bundle_path",
    "export_root",
}

_ALWAYS_SCRUB = {"outputPath", "htmlPath", "attachmentsDir"}


def _deep_sort(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_sort(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_deep_sort(item) for item in value]
    return value


def _scrub_paths(value: Any, *, repo_root: Path) -> Any:
    if isinstance(value, dict):
        scrubbed: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _PATH_KEYS and isinstance(item, str):
                if key in _ALWAYS_SCRUB:
                    scrubbed[key] = "<path>"
                else:
                    scrubbed[key] = _scrub_path_string(item, repo_root=repo_root)
            else:
                scrubbed[key] = _scrub_paths(item, repo_root=repo_root)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_paths(item, repo_root=repo_root) for item in value]
    if isinstance(value, str):
        return value
    return value


def _scrub_path_string(value: str, *, repo_root: Path) -> str:
    text = value.strip()
    if not text:
        return value
    try:
        path = Path(text)
    except Exception:
        return value
    if not path.is_absolute():
        return text
    try:
        rel = path.relative_to(repo_root)
        return str(rel)
    except Exception:
        return "<abs>"


def canonicalize_markdown(
    text: str,
    *,
    repo_root: Optional[Path] = None,
    scrub_paths: bool = False,
    sort_keys: bool = True,
) -> str:
    post = frontmatter.loads(text)
    metadata: Dict[str, Any] = dict(post.metadata)
    if scrub_paths and repo_root is not None:
        metadata = _scrub_paths(metadata, repo_root=repo_root)
    if sort_keys:
        metadata = _deep_sort(metadata)
    return frontmatter.dumps(frontmatter.Post(post.content, **metadata))


def canonicalize_file(
    path: Path,
    *,
    repo_root: Optional[Path] = None,
    scrub_paths: bool = False,
    sort_keys: bool = True,
) -> bool:
    text = path.read_text(encoding="utf-8")
    out = canonicalize_markdown(text, repo_root=repo_root, scrub_paths=scrub_paths, sort_keys=sort_keys)
    if out == text:
        return False
    path.write_text(out, encoding="utf-8")
    return True
