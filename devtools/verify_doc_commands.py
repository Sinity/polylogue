"""Verify that doc-file command examples resolve to real commands.

Scans ``README.md`` and every committed ``docs/**/*.md`` file for
inline references to two strict-subcommand surfaces:

- ``polylogued`` -> ``polylogue.daemon.cli:main``
- ``devtools``   -> ``devtools.command_catalog.COMMANDS``

For each occurrence the lint extracts the first non-flag token after
the surface name and verifies it is a real subcommand for that
surface. The ``polylogue`` CLI is intentionally *not* checked as a
strict surface, because the root command is query-first: any bare
token after ``polylogue`` is a valid FTS query, not a typo. Stale
``polylogue ...`` invocations are caught instead by the explicit
denylist below.

The lint only reads tokens that appear inside Markdown code surfaces
(inline ``` `code` ``` spans and fenced ``` ```bash/sh/shell/console``` `` blocks);
plain prose is ignored to avoid false positives from sentences such as
"polylogue and devtools share a workflow".

It exists to keep #1262 / #869 closed: doc drift away from the
daemon-first command surface should fail a CI gate, not survive in
master until a user files a bug.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click

from devtools import repo_root as _get_root
from devtools.command_catalog import COMMANDS, command_name_from_tokens
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.daemon.cli import main as polylogued_root

ROOT = _get_root()


def _doc_files(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    docs_dir = root / "docs"
    if docs_dir.exists():
        paths.extend(sorted(docs_dir.rglob("*.md")))
    return [p for p in paths if p.exists()]


# Match ``surface rest_of_line`` where surface is a strict-subcommand
# command. Only the first token after the surface is inspected.
#
# ``(?![.\w-])`` after the surface name rejects filename/binary
# neighbours such as ``polylogued.service`` (systemd unit) or
# ``polylogue-mcp`` (sibling executable). The preceding ``(?<![\w-])``
# rejects mid-word matches so ``run-polylogued-helper`` doesn't trip.
_SURFACE_RE = re.compile(r"(?<![\w-])(polylogued|devtools)(?![.\w-])([^\n`]*)")
_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# Stale invocation patterns flagged across all three surfaces, including
# ``polylogue``. These are exact substring matches (after stripping
# Markdown link bodies); they exist independently of subcommand lookups
# so that drift across the daemon-first transition cannot creep back in.
STALE_INVOCATIONS: tuple[tuple[str, str], ...] = (
    (
        "polylogued run --api ",
        "polylogued: '--api' is a stale boolean; the HTTP API is on by default — use '--no-api' to disable.",
    ),
    (
        "polylogued run --watch ",
        "polylogued: '--watch' is a stale boolean; the watcher is on by default — use '--no-watch' to disable.",
    ),
    (
        "polylogued run --enable-api",
        "polylogued: '--enable-api' was removed; the HTTP API is on by default — use '--no-api' to disable.",
    ),
    (
        "polylogue run --input",
        "polylogue: 'polylogue run --input' was removed; use 'polylogue ingest PATH' against a running daemon.",
    ),
    (
        "polylogue run site",
        "polylogue: 'polylogue run site' was removed; site rendering lives in 'devtools render pages'.",
    ),
    (
        "polylogue run --source",
        "polylogue: 'polylogue run --source' was removed; ask the running daemon via 'polylogue ingest PATH'.",
    ),
    (
        "polylogue run --watch",
        "polylogue: 'polylogue run --watch' was removed; the daemon owns the watcher (run 'polylogued run').",
    ),
)


@dataclass(frozen=True)
class DocCommandRef:
    """A surface/subcommand pair extracted from a doc file."""

    surface: str
    subcommand: str
    file: Path
    line: int


def _click_subcommands(root: click.Command) -> frozenset[str]:
    """Return top-level subcommand names for a Click root."""
    names: set[str] = set()
    for command_path in iter_command_paths(root, include_root=False):
        if command_path.path:
            names.add(command_path.path[0])
    return frozenset(names)


def _devtools_subcommands() -> frozenset[str]:
    return frozenset(COMMANDS.keys())


def _polylogued_subcommands() -> frozenset[str]:
    return _click_subcommands(polylogued_root)


def _real_tokens(rest: str) -> tuple[str, ...]:
    """Plain command tokens after a surface, ignoring flags and shell glue."""
    stripped = rest.lstrip()
    if not stripped:
        return ()
    for stop in ("&&", "||", "|", ";", "#", "$(", "`"):
        idx = stripped.find(stop)
        if idx >= 0:
            stripped = stripped[:idx]
    parts = stripped.split()
    tokens: list[str] = []
    for part in parts:
        cleaned = part.strip(".,:;\"'`()[]<>")
        if not cleaned:
            continue
        if cleaned.startswith("-"):
            continue
        if "=" in cleaned and not cleaned.startswith("="):
            continue
        if not _TOKEN_RE.match(cleaned):
            continue
        tokens.append(cleaned)
    return tuple(tokens)


def _surface_subcommand(surface: str, rest: str) -> str | None:
    tokens = _real_tokens(rest)
    if not tokens:
        return None
    if surface != "devtools":
        return tokens[0]
    known = command_name_from_tokens(tokens)
    if known is not None:
        return known
    max_len = max((len(spec.command_path) for spec in COMMANDS.values()), default=1)
    return " ".join(tokens[: min(len(tokens), max_len)])


_FENCE_RE = re.compile(r"^\s*```([A-Za-z0-9_+-]*)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_CODE_FENCE_LANGS = frozenset({"", "bash", "sh", "shell", "console", "zsh", "ini"})


def _code_segments(text: str) -> list[tuple[int, str]]:
    """Return (line_no, segment) for every Markdown code segment.

    Segments come from inline backtick spans and fenced ```bash/sh/...
    blocks; prose lines are not returned.
    """
    segments: list[tuple[int, str]] = []
    in_fence = False
    fence_lang = ""
    fence_start_line = 0
    fence_buffer: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            if not in_fence:
                fence_lang = fence_match.group(1).lower()
                if fence_lang in _CODE_FENCE_LANGS:
                    in_fence = True
                    fence_buffer = []
                    fence_start_line = line_no
                continue
            # closing fence
            if fence_lang in _CODE_FENCE_LANGS:
                for offset, buf_line in enumerate(fence_buffer):
                    segments.append((fence_start_line + 1 + offset, buf_line))
            in_fence = False
            fence_lang = ""
            fence_buffer = []
            continue
        if in_fence:
            fence_buffer.append(line)
            continue
        for inline in _INLINE_CODE_RE.findall(line):
            segments.append((line_no, inline))
    return segments


def _scan_file(path: Path, root: Path) -> tuple[list[DocCommandRef], list[str]]:
    rel = path.relative_to(root).as_posix()
    refs: list[DocCommandRef] = []
    stale_hits: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return refs, [f"{rel}: read error: {exc}"]

    # Stale-substring check runs against full lines so it catches the
    # token sequence regardless of code-fence wrapping.
    for line_no, line in enumerate(text.splitlines(), start=1):
        sanitized = re.sub(r"\]\([^)]+\)", "]()", line)
        for needle, hint in STALE_INVOCATIONS:
            if needle in sanitized:
                stale_hits.append(f"{rel}:{line_no}: stale invocation '{needle.rstrip()}' — {hint}")

    # Subcommand validity is checked only inside code segments, and only
    # when the surface name appears in a command-start position. Prose
    # inside ``# comment`` lines of a fenced bash block is skipped so a
    # phrase like ``# polylogued runs ingest...`` does not trip the
    # lint.
    for line_no, segment in _code_segments(text):
        # Strip a leading shell prompt so ``$ polylogued run`` is
        # treated as starting at ``polylogued``.
        head = segment.lstrip()
        if head.startswith(("$ ", "> ")):
            head = head[2:]
        if head.startswith("#"):
            # Bash comment line; not a real invocation.
            continue
        for match in _SURFACE_RE.finditer(segment):
            # Skip if the match is not at command-start. We accept the
            # very first surface match at position 0 of ``head``, plus
            # matches that immediately follow shell-pipeline glue.
            start = match.start(1)
            pre = segment[:start].rstrip()
            if pre and not pre.endswith(("|", "&&", "||", ";", "(", "{", "$", "\\", "=")):
                # Mid-line surface mention (prose-in-code) — skip.
                continue
            surface = match.group(1)
            rest = match.group(2)
            token = _surface_subcommand(surface, rest)
            if token is None:
                continue
            refs.append(DocCommandRef(surface=surface, subcommand=token, file=path, line=line_no))
    return refs, stale_hits


def check_docs(root: Path | None = None) -> tuple[list[str], int]:
    """Return (errors, files_checked)."""
    target_root = root if root is not None else ROOT
    files = _doc_files(target_root)
    surface_names: dict[str, frozenset[str]] = {
        "polylogued": _polylogued_subcommands(),
        "devtools": _devtools_subcommands(),
    }

    errors: list[str] = []
    for path in files:
        refs, stale = _scan_file(path, target_root)
        errors.extend(stale)
        rel = path.relative_to(target_root).as_posix()
        for ref in refs:
            known = surface_names[ref.surface]
            if ref.subcommand in known:
                continue
            errors.append(f"{rel}:{ref.line}: '{ref.surface} {ref.subcommand}' is not a known {ref.surface} subcommand")
    return errors, len(files)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = p.parse_args(list(argv) if argv is not None else None)

    errors, files_checked = check_docs()
    blocking = bool(errors)

    if args.json:
        json.dump(
            {"blocking": blocking, "errors": errors, "files_checked": files_checked},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if errors:
            for e in errors:
                print(f"[BLOCK] {e}")
        else:
            print(f"verify-doc-commands: {files_checked} doc files scanned, no stale commands")
        print()
        print(f"blocking={blocking}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
