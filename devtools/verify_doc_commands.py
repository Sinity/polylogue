"""Verify that doc-file command examples resolve to real commands.

Scans ``README.md`` and every committed ``docs/**/*.md`` file for
inline references to three command surfaces:

- ``polylogued`` -> ``polylogue.daemon.cli:main`` (strict subcommands)
- ``devtools``   -> ``devtools.command_catalog.COMMANDS`` (strict subcommands)
- ``polylogue``  -> the query-first CLI (recognized commands + flags)

For ``polylogued`` and ``devtools`` the lint extracts the first non-flag
token after the surface name and verifies it is a real subcommand.

The ``polylogue`` CLI is query-first — any bare token after ``polylogue``
is normally a valid FTS query, not a typo — so it is validated *by command
recognition* (#2438): a documented invocation is only checked when its
leading token resolves to a known command path or a removed command name.
A recognized command path then has its long-flags validated against the
union of root and full-path options (lazy subcommands are materialized via
``get_params`` so ``analyze insights profiles --tier`` resolves correctly),
while a leading token that resolves to neither a known nor a removed command
is left alone so ``polylogue rate limiting retries`` stays legal.

The lint only reads tokens that appear inside Markdown code surfaces
(inline ``` `code` ``` spans and fenced ``` ```bash/sh/shell/console``` `` blocks);
plain prose is ignored to avoid false positives from sentences such as
"polylogue and devtools share a workflow".

It exists to keep #1262 / #869 / #2438 closed: doc drift away from the
live command surface should fail a CI gate, not survive in master until a
user files a bug.
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


def _polylogue_path_flags(root: click.Command) -> dict[tuple[str, ...], frozenset[str]]:
    """Long-flags declared on every command path in the ``polylogue`` tree.

    ``iter_command_paths`` descends the full tree, so leaf subcommands such as
    ``analyze insights profiles`` expose their real options here even though the
    top ``analyze`` group does not.
    """
    return {cp.path: _long_opts(cp.command) for cp in iter_command_paths(root, include_root=False) if cp.path}


# ``polylogue`` subcommands that were removed/renamed. The root command is
# query-first — any bare token after ``polylogue`` is normally a valid FTS
# query, not a typo — so a removed command name (``polylogue list``) is
# indistinguishable from a search ("find sessions matching 'list'") without
# remembering it once *was* a command. This intentionally small, documented set
# replaces the previous hand-maintained per-flag denylist for the cases where
# command recognition alone cannot fire.
REMOVED_POLYLOGUE_COMMANDS: dict[str, str] = {
    "list": "polylogue: the 'list' verb was removed; use 'read --all' (optionally with --format).",
    "show": "polylogue: the 'show' verb was removed; use 'read --view transcript' for one session.",
}


# Dated point-in-time records under these trees assert the command surface *as
# of their date*, not the current one. Holding them to live-command accuracy
# would force rewriting history, so they are excluded from the drift lint.
_EXCLUDED_DOC_DIRS: tuple[str, ...] = ("docs/audits",)


def _doc_files(root: Path) -> list[Path]:
    paths = [root / "README.md"]
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


def _polylogue_invocation_errors(
    rel: str,
    line: int,
    rest: str,
    *,
    ctx: _PolylogueContext,
) -> list[str]:
    """Validate a single ``polylogue ...`` invocation.

    Opt-in by command recognition: a removed command name fails; a recognized
    command path has its long-flags validated against ``root ∪ path ∪ direct``
    options; a leading token that resolves to neither is left alone so
    query-first FTS examples (``polylogue rate limiting retries``) stay legal.
    """
    tokens = _invocation_tokens(rest)
    if not tokens:
        return []

    # 1. Command detection: the first bare token that is a known/removed command.
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
        if tok in REMOVED_POLYLOGUE_COMMANDS:
            return [f"{rel}:{line}: '{tok}' — {REMOVED_POLYLOGUE_COMMANDS[tok]}"]
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
    label = "polylogue " + " ".join(path)
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
class _PolylogueContext:
    root_flags: frozenset[str]
    value_flags: frozenset[str]
    path_flags: dict[tuple[str, ...], frozenset[str]]


def _build_polylogue_context() -> _PolylogueContext:
    cli = _polylogue_cli()
    return _PolylogueContext(
        root_flags=_long_opts(cli),
        value_flags=_polylogue_root_value_flags(cli),
        path_flags=_polylogue_path_flags(cli),
    )


def _scan_file(
    path: Path, root: Path, polylogue_ctx: _PolylogueContext | None = None
) -> tuple[list[DocCommandRef], list[str]]:
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
            if surface == "polylogue":
                if polylogue_ctx is not None:
                    stale_hits.extend(_polylogue_invocation_errors(rel, line_no, rest, ctx=polylogue_ctx))
                continue
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
    polylogue_ctx = _build_polylogue_context()

    errors: list[str] = []
    for path in files:
        refs, stale = _scan_file(path, target_root, polylogue_ctx)
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
            print(f"verify doc-commands: {files_checked} doc files scanned, no stale commands")
        print()
        print(f"blocking={blocking}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
