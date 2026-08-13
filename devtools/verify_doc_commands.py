"""Verify that doc-file command examples resolve to real commands.

Scans ``README.md`` and every committed ``docs/**/*.md`` file for
inline references to three command surfaces:

- ``polylogued`` -> ``polylogue.daemon.cli:main`` (strict subcommands)
- ``devtools``   -> ``devtools.command_catalog.COMMANDS`` (strict subcommands)
- ``polylogue``  -> the query-first CLI (recognized commands + flags)

For ``polylogued`` and ``devtools`` the lint extracts the first non-flag
token after the surface name and verifies it is a real subcommand.

The ``polylogue`` CLI is query-first: any bare token after ``polylogue`` can
be a valid FTS query. It is therefore validated only when a leading token
resolves to a live command path. A recognized path has its long flags checked
against the live Click tree. Free-text queries remain legal without a registry
of commands that used to exist.

The lint only reads tokens that appear inside Markdown code surfaces
(inline ``` `code` ``` spans and fenced ``` ```bash/sh/shell/console``` `` blocks);
plain prose is ignored to avoid false positives from sentences such as
"polylogue and devtools share a workflow".

The validator derives authority from the current command implementations. It
does not grep for historical spellings or require prose to preserve old names.
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


def _materialized_params(cmd: click.Command) -> list[click.Parameter]:
    """Real parameters of a command, resolving lazy-loaded proxies.

    The root CLI registers many subcommands as lazy proxies whose ``.params``
    attribute is empty until the underlying module is imported. ``get_params``
    triggers that resolution (and includes Click's auto-added ``--help``), so it
    is the only reliable source of a lazy command's true option set.
    """
    try:
        return list(cmd.get_params(click.Context(cmd)))
    except Exception:
        return list(cmd.params)


def _long_opts(cmd: click.Command) -> frozenset[str]:
    """All ``--long`` option strings declared on a Click command."""
    out: set[str] = set()
    for param in _materialized_params(cmd):
        for opt in (*getattr(param, "opts", ()), *getattr(param, "secondary_opts", ())):
            if opt.startswith("--"):
                out.add(opt)
    return frozenset(out)


def _polylogue_cli() -> click.Command:
    from polylogue.cli.click_app import cli

    return cli


def _polylogue_root_value_flags(root: click.Command) -> frozenset[str]:
    """Root long-flags that consume the following token as their value.

    Used so a flag *value* (``--since yesterday``, ``--add-tag export``) is not
    mistaken for a subcommand during command detection.
    """
    out: set[str] = set()
    for param in _materialized_params(root):
        if getattr(param, "is_flag", False) or getattr(param, "count", False):
            continue
        for opt in getattr(param, "opts", ()):
            if opt.startswith("--"):
                out.add(opt)
    return frozenset(out)


def _click_path_flags(root: click.Command) -> dict[tuple[str, ...], frozenset[str]]:
    """Long flags declared on every command path in a Click tree.

    ``iter_command_paths`` descends the full tree, so leaf subcommands such as
    ``analyze insights profiles`` expose their real options here even though the
    top ``analyze`` group does not.
    """
    return {cp.path: _long_opts(cp.command) for cp in iter_command_paths(root, include_root=False) if cp.path}


# Dated point-in-time records under these trees assert the command surface *as
# of their date*, not the current one. Holding them to live-command accuracy
# would force rewriting history, so they are excluded from the drift lint.
_EXCLUDED_DOC_DIRS: tuple[str, ...] = ("docs/audits",)


def _doc_files(root: Path) -> list[Path]:
    paths = [root / "README.md", root / "browser-extension" / "README.md"]
    docs_dir = root / "docs"
    if docs_dir.exists():
        paths.extend(sorted(docs_dir.rglob("*.md")))
    excluded = tuple(root / Path(d) for d in _EXCLUDED_DOC_DIRS)
    return [p for p in paths if p.exists() and not any(p.is_relative_to(d) for d in excluded)]


# Match ``surface rest_of_line`` where surface is a strict-subcommand
# command. Only the first token after the surface is inspected.
#
# ``(?![.\w-])`` after the surface name rejects filename/binary
# neighbours such as ``polylogued.service`` (systemd unit) or
# ``polylogue-mcp`` (sibling executable). The preceding ``(?<![\w-])``
# rejects mid-word matches so ``run-polylogued-helper`` doesn't trip.
_SURFACE_RE = re.compile(r"(?<![\w-])(polylogued|polylogue|devtools)(?![.\w-])([^\n`]*)")
_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


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


def _invocation_tokens(rest: str) -> list[str]:
    """Ordered raw tokens (flags kept) up to a shell/pipeline boundary."""
    stripped = rest.lstrip()
    for stop in ("&&", "||", "|", ";", "#", "$(", "`"):
        idx = stripped.find(stop)
        if idx >= 0:
            stripped = stripped[:idx]
    tokens: list[str] = []
    for part in stripped.split():
        cleaned = part.strip(".,:;\"'`()[]<>")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _click_invocation_errors(
    rel: str,
    line: int,
    rest: str,
    *,
    surface: str,
    ctx: _ClickContext,
) -> list[str]:
    """Validate flags for a command path derived from the live Click tree.

    Validation opts in only after command recognition. This preserves
    query-first free text for ``polylogue`` while still checking strict daemon
    invocations after their command has been identified.
    """
    tokens = _invocation_tokens(rest)
    if not tokens:
        return []

    # Command detection: the first bare token that is a known command.
    #    A token consumed as the value of a root value-flag (``--add-tag export``)
    #    is skipped so a flag value is never read as a subcommand.
    start: int | None = None
    verb: str | None = None
    skip_next = False
    for idx, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            if "=" not in tok and tok in ctx.value_flags:
                skip_next = True
            continue
        if (tok,) in ctx.path_flags:
            start, verb = idx, tok
            break
        # Unknown bare token: a flag value or a free-text query word — keep going.

    if verb is None or start is None or "then" in tokens:
        # Unrecognized leading token (query-first) or a ``then`` chain whose
        # flags attribute to different verbs — leave it alone.
        return []

    # 2. Resolve the full command path by descending on consecutive bare tokens
    #    that are children of the current path. Flags are skipped; the first bare
    #    token that is not a child terminates the path (it is a positional arg).
    path: tuple[str, ...] = (verb,)
    for tok in tokens[start + 1 :]:
        if tok.startswith("-"):
            continue
        if path + (tok,) in ctx.path_flags:
            path = path + (tok,)
            continue
        break

    # 3. Valid flags = root ∪ every command on the resolved path. Lazy commands
    #    are materialized in ``_long_opts`` so ``analyze --count`` and leaf
    #    subcommand flags (``analyze insights profiles --tier``) both resolve.
    valid: set[str] = set(ctx.root_flags)
    for depth in range(1, len(path) + 1):
        valid |= ctx.path_flags.get(path[:depth], frozenset())

    errors: list[str] = []
    label = surface + " " + " ".join(path)
    for tok in tokens:
        if tok == "--":  # end-of-options; remainder is positional
            break
        if not tok.startswith("--"):
            continue
        flag = tok.split("=", 1)[0]
        if flag not in valid:
            errors.append(f"{rel}:{line}: '{flag}' is not a known flag for '{label}'")
    return errors


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


@dataclass(frozen=True)
class _ClickContext:
    root_flags: frozenset[str]
    value_flags: frozenset[str]
    path_flags: dict[tuple[str, ...], frozenset[str]]


def _build_click_context(root: click.Command) -> _ClickContext:
    return _ClickContext(
        root_flags=_long_opts(root),
        value_flags=_polylogue_root_value_flags(root),
        path_flags=_click_path_flags(root),
    )


def _scan_file(
    path: Path,
    root: Path,
    polylogue_ctx: _ClickContext | None = None,
    polylogued_ctx: _ClickContext | None = None,
) -> tuple[list[DocCommandRef], list[str]]:
    rel = path.relative_to(root).as_posix()
    refs: list[DocCommandRef] = []
    command_errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return refs, [f"{rel}: read error: {exc}"]

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
            if surface == "polylogue":
                if polylogue_ctx is not None:
                    command_errors.extend(
                        _click_invocation_errors(rel, line_no, rest, surface=surface, ctx=polylogue_ctx)
                    )
                continue
            if surface == "polylogued" and polylogued_ctx is not None:
                command_errors.extend(_click_invocation_errors(rel, line_no, rest, surface=surface, ctx=polylogued_ctx))
            token = _surface_subcommand(surface, rest)
            if token is None:
                continue
            refs.append(DocCommandRef(surface=surface, subcommand=token, file=path, line=line_no))
    return refs, command_errors


def check_docs(root: Path | None = None) -> tuple[list[str], int]:
    """Return (errors, files_checked)."""
    target_root = root if root is not None else ROOT
    files = _doc_files(target_root)
    surface_names: dict[str, frozenset[str]] = {
        "polylogued": _polylogued_subcommands(),
        "devtools": _devtools_subcommands(),
    }
    polylogue_ctx = _build_click_context(_polylogue_cli())
    polylogued_ctx = _build_click_context(polylogued_root)

    errors: list[str] = []
    for path in files:
        refs, command_errors = _scan_file(path, target_root, polylogue_ctx, polylogued_ctx)
        errors.extend(command_errors)
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
            print(f"verify doc-commands: {files_checked} doc files scanned, no stale commands")
        print()
        print(f"blocking={blocking}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
