#!/usr/bin/env python3
"""Golden-bytes harness for the documented ``polylogue find`` query surface.

Extracts every documented ``polylogue ...`` invocation from
``docs/cli-reference.md`` that exercises the root ``find`` query surface,
runs each one in-process against a deterministic seeded archive with the
daemon disabled, and records exit code + exact stdout/stderr bytes into one
JSON file.

Usage::

    uv run python scripts/golden_find_bytes.py OUTPUT.json

Determinism: the archive is a clone of an immutable content-addressed
seeded artifact (``tests/infra/workload_artifacts``), the clock is frozen
via ``tests.infra.frozen_clock.freeze_clock`` with ``datetime`` patched in
every already-imported ``polylogue`` module, and the environment handed to
the CLI is a fixed allowlist. Running the harness twice must produce byte
identical JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOC = REPO_ROOT / "docs" / "cli-reference.md"

#: Named workload from ``tests/infra/workload_declarations`` used as the
#: corpus. ``cli-mixed`` is the declared read-surface workload (chatgpt +
#: claude-code sessions).
WORKLOAD = "cli-mixed"

#: Frozen wall-clock anchor (``tests.infra.frozen_clock.DEFAULT_FROZEN_EPOCH``).
FROZEN_EPOCH = 1700000000.0


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Invocation:
    """One documented CLI line, with its skip disposition."""

    line: int
    text: str
    argv: tuple[str, ...]
    skip_reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.line:04d}:{self.text}"


def _strip_comment(text: str) -> str:
    """Drop a trailing ``#`` comment that is outside any quoting."""
    out: list[str] = []
    quote: str | None = None
    for char in text:
        if quote is None and char == "#":
            break
        if quote is None and char in "'\"":
            quote = char
        elif quote is not None and char == quote:
            quote = None
        out.append(char)
    return "".join(out).strip()


def _is_find_surface(argv: tuple[str, ...]) -> bool:
    """True when the invocation drives the root ``find`` query surface."""
    return "find" in argv[1:]


def _skip_reason(argv: tuple[str, ...]) -> str | None:
    """Why this documented invocation is not a golden candidate, if so."""
    rest = list(argv)
    if "--to" in rest:
        target = rest[rest.index("--to") + 1] if rest.index("--to") + 1 < len(rest) else ""
        if target in {"browser", "clipboard"}:
            return f"side-effecting output sink: --to {target}"
    if "then" in rest:
        action = rest[rest.index("then") + 1] if rest.index("then") + 1 < len(rest) else ""
        if action == "mark":
            return "mutating action: then mark writes assertions"
        if action == "delete" and "--dry-run" not in rest:
            return "mutating action: then delete without --dry-run"
        if action == "continue" and "--exec" in rest:
            return "side-effecting action: then continue --exec launches an agent"
    if "--help" in rest:
        return "help screen, not a query execution"
    if "<subcommand>" in rest or any("<" in token and ">" in token for token in rest):
        return "documentation placeholder, not a literal invocation"
    return None


def extract_invocations(doc: Path = DOC) -> list[Invocation]:
    """Every documented ``polylogue`` line that uses the root find surface."""
    found: list[Invocation] = []
    for number, raw in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped.startswith("polylogue "):
            continue
        command = _strip_comment(stripped)
        try:
            argv = tuple(shlex.split(command))
        except ValueError:
            continue
        if not argv or argv[0] != "polylogue":
            continue
        if not _is_find_surface(argv):
            continue
        found.append(
            Invocation(
                line=number,
                text=command,
                argv=argv[1:],
                skip_reason=_skip_reason(argv[1:]),
            )
        )
    return found


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------


def build_archive(destination: Path) -> object:
    """Clone the named immutable seeded workload into ``destination``."""
    from tests.infra.workload_artifacts import build_seeded_archive, clone_seeded_archive
    from tests.infra.workload_declarations import named_corpus_specs

    artifact = build_seeded_archive(named_corpus_specs(WORKLOAD))
    return clone_seeded_archive(artifact, destination)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _polylogue_modules() -> tuple[str, ...]:
    """Imported ``polylogue`` modules that bind a ``datetime`` symbol."""
    from datetime import datetime as real_datetime

    names = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("polylogue"):
            continue
        if getattr(module, "datetime", None) is real_datetime:
            names.append(name)
    return tuple(sorted(names))


@dataclass
class Runner:
    """In-process CLI runner over a fixed environment and frozen clock."""

    archive_root: Path
    home: Path
    env: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "TERM": "dumb",
            "COLUMNS": "100",
            "LINES": "40",
            "NO_COLOR": "1",
            "POLYLOGUE_ARCHIVE_ROOT": str(self.archive_root),
            # Daemon OFF: the client-mode knob plus an unroutable URL, so no
            # ambient workstation daemon can answer a read.
            "POLYLOGUE_DAEMON": "off",
            "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
            "XDG_DATA_HOME": str(self.home / ".local/share"),
            "XDG_STATE_HOME": str(self.home / ".local/state"),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "XDG_CACHE_HOME": str(self.home / ".cache"),
            "VOYAGE_API_KEY": "",
        }

    def run(self, argv: tuple[str, ...]) -> dict[str, object]:
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        # ``--no-daemon`` is the declared root flag; keep it in front of the
        # documented argv so the query surface never consults a daemon.
        effective = ("--no-daemon", *argv)
        result = runner.invoke(
            cli,
            list(effective),
            env=self.env,
            catch_exceptions=True,
            color=False,
        )
        stderr = ""
        try:
            stderr = result.stderr
        except ValueError:  # stderr not separately captured on this click
            stderr = ""
        return {
            "argv": list(effective),
            "exit": int(result.exit_code),
            "stdout": result.stdout,
            "stderr": stderr,
        }


@contextmanager
def _frozen(modules: tuple[str, ...]) -> Iterator[None]:
    """Freeze every clock the read path can render into its output.

    ``freeze_clock`` pins ``time.time``/``time.monotonic`` and the
    ``datetime`` symbol in the named modules. The projection-availability
    payload additionally renders a *measured* elapsed budget read from
    ``perf_counter``, which is pure wall-clock jitter, so that is pinned to a
    constant here — both the ``time`` module attribute and every
    ``from time import perf_counter`` binding inside ``polylogue``.
    """
    from tests.infra.frozen_clock import freeze_clock

    def _fixed_perf_counter() -> float:
        return 0.0

    with ExitStack() as stack:
        stack.enter_context(freeze_clock(start=FROZEN_EPOCH, patch_datetime_in_modules=modules))
        stack.enter_context(patch("time.perf_counter", new=_fixed_perf_counter))
        for name, module in list(sys.modules.items()):
            if not name.startswith("polylogue"):
                continue
            if callable(getattr(module, "perf_counter", None)):
                stack.enter_context(patch.object(module, "perf_counter", _fixed_perf_counter))
        yield


def _iter_results(runner: Runner, invocations: list[Invocation]) -> Iterator[tuple[str, dict[str, object]]]:
    import polylogue.cli.click_app  # noqa: F401  (populate sys.modules first)

    modules = _polylogue_modules()
    with _frozen(modules):
        for invocation in invocations:
            if invocation.skip_reason is not None:
                continue
            yield invocation.key, runner.run(invocation.argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path to write the golden JSON")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory for the seeded archive clone and fake HOME",
    )
    args = parser.parse_args(argv)

    workspace = (args.workspace or (REPO_ROOT / ".cache" / "golden-find")).resolve()
    archive_root = workspace / "archive"
    home = workspace / "home"
    for path in (home / ".local/share", home / ".local/state", home / ".config", home / ".cache"):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["POLYLOGUE_DAEMON"] = "off"

    invocations = extract_invocations()
    clone = build_archive(archive_root)

    runner = Runner(archive_root=archive_root, home=home)
    results: dict[str, object] = {}
    try:
        for key, record in _iter_results(runner, invocations):
            results[key] = record
    finally:
        close = getattr(clone, "close", None)
        if callable(close):
            close()

    payload = {
        "workload": WORKLOAD,
        "frozen_epoch": FROZEN_EPOCH,
        "source": "docs/cli-reference.md",
        "skipped": [
            {"line": inv.line, "command": inv.text, "reason": inv.skip_reason}
            for inv in invocations
            if inv.skip_reason is not None
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(results)} goldens ({len(payload['skipped'])} skipped) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
