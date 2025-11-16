from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ui import UI


def add_html_option(parser: ArgumentParser, *, description: str = "HTML preview mode: on/off/auto (default auto)") -> None:
    parser.add_argument(
        "--html",
        dest="html_mode",
        nargs="?",
        const="on",
        default="auto",
        choices=["auto", "on", "off"],
        metavar="MODE",
        help=description,
    )


def add_collapse_option(parser: ArgumentParser, *, help_text: str = "Override collapse threshold") -> None:
    parser.add_argument("--collapse-threshold", type=int, default=None, help=help_text)


def add_out_option(parser: ArgumentParser, *, default_path: Path, help_text: str | None = None) -> None:
    description = help_text or f"Override output directory (default {default_path})"
    parser.add_argument("--out", type=Path, default=None, help=description)


def add_diff_option(parser: ArgumentParser, *, help_text: str = "Write delta diff alongside updated Markdown") -> None:
    parser.add_argument("--diff", action="store_true", help=help_text)


def add_force_option(parser: ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--force", action="store_true", help=help_text)


def add_allow_dirty_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow overwriting files with local edits (requires --force)",
    )


def add_dry_run_option(parser: ArgumentParser, *, help_text: str = "Report actions without writing files") -> None:
    parser.add_argument("--dry-run", action="store_true", help=help_text)


def resolve_base_dir(value: Path | None, default: Path) -> Path:
    """Resolve base directory with expanduser, falling back to default.

    Args:
        value: User-provided path or None
        default: Default path to use if value is None

    Returns:
        Resolved path with ~ expanded
    """
    return Path(value).expanduser() if value else default


def ensure_directory(path: Path, *, create: bool = True) -> Path:
    """Ensure directory exists, optionally creating it.

    Args:
        path: Directory path to ensure
        create: If True, create directory if missing

    Returns:
        The same path (for chaining)
    """
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# Parent Parsers for Shared Flags
# These reduce duplication by grouping common flags that appear across multiple commands


def create_output_parent() -> ArgumentParser:
    """Parent parser for common output flags (--json).

    Use for commands that support machine-readable output.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    return parser


def create_render_parent() -> ArgumentParser:
    """Parent parser for rendering options (--html, --collapse-threshold, --theme).

    Use for commands that generate Markdown/HTML output.
    """
    parser = ArgumentParser(add_help=False)
    add_html_option(parser)
    add_collapse_option(parser)
    parser.add_argument("--theme", type=str, choices=["light", "dark"], default=None, help="HTML theme override")
    return parser


def create_write_parent() -> ArgumentParser:
    """Parent parser for write operation flags (--force, --allow-dirty, --dry-run).

    Use for commands that write/modify files.
    """
    parser = ArgumentParser(add_help=False)
    add_force_option(parser, help_text="Force rewrite even if up-to-date")
    add_allow_dirty_option(parser)
    add_dry_run_option(parser)
    return parser


def create_filter_parent() -> ArgumentParser:
    """Parent parser for common filter flags (--provider, --since, --until).

    Use for commands that filter by provider or time range.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--provider", type=str, default=None, help="Filter by provider name")
    parser.add_argument("--since", type=str, default=None, help="Only include items on/after this timestamp")
    parser.add_argument("--until", type=str, default=None, help="Only include items on/before this timestamp")
    return parser


# Path Handling Standardization
# Provides consistent path validation and creation across all commands


@dataclass
class PathPolicy:
    """Policy for handling missing paths.

    Standardizes behavior when paths don't exist across all commands.
    """

    should_exist: bool = True  # Error if path is missing
    create_if_missing: bool = False  # Auto-create missing paths
    prompt_create: bool = False  # Ask before creating (interactive mode only)

    @staticmethod
    def must_exist() -> "PathPolicy":
        """Path must exist (read operations like stats, inspect)."""
        return PathPolicy(should_exist=True)

    @staticmethod
    def create_ok() -> "PathPolicy":
        """Auto-create if missing (write operations like render, sync)."""
        return PathPolicy(should_exist=False, create_if_missing=True)

    @staticmethod
    def prompt_create() -> "PathPolicy":
        """Ask before creating (interactive operations)."""
        return PathPolicy(should_exist=False, prompt_create=True)


def resolve_path(
    path: Path, policy: PathPolicy, ui: "UI", *, json_mode: bool = False
) -> Optional[Path]:
    """Resolve path according to policy with consistent error handling.

    Args:
        path: Path to resolve
        policy: Policy defining behavior for missing paths
        ui: UI instance for console output and prompts
        json_mode: If True, raise JSONModeError instead of printing to console

    Returns:
        Resolved path if successful, None if path validation failed

    Raises:
        JSONModeError: In JSON mode when path validation fails

    Examples:
        # For read operations (stats, inspect)
        directory = resolve_path(Path(args.dir), PathPolicy.must_exist(), ui)
        if not directory:
            raise SystemExit(1)

        # For write operations (render, sync)
        output_dir = resolve_path(Path(args.out), PathPolicy.create_ok(), ui)

        # For interactive operations
        target_dir = resolve_path(Path(args.dir), PathPolicy.prompt_create(), ui)

        # In JSON mode
        directory = resolve_path(Path(args.dir), PathPolicy.must_exist(), ui, json_mode=True)
    """
    if path.exists():
        return path

    if policy.should_exist:
        if json_mode:
            from .json_output import JSONModeError

            raise JSONModeError(
                "path_not_found",
                f"Path not found: {path}",
                path=str(path),
                hint=f"Create it with: mkdir -p {path}",
            )
        ui.console.print(f"[red]Error: Path not found: {path}")
        ui.console.print(f"[dim]Create it with: mkdir -p {path}")
        return None

    if policy.create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as exc:
            if json_mode:
                from .json_output import JSONModeError

                raise JSONModeError(
                    "path_create_failed",
                    f"Could not create directory: {path}",
                    path=str(path),
                    reason=str(exc),
                )
            ui.console.print(f"[red]Error: Could not create directory: {path}")
            ui.console.print(f"[dim]Reason: {exc}")
            return None

    if policy.prompt_create and not ui.plain:
        if ui.console.confirm(f"Create directory {path}?", default=True):
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError as exc:
                if json_mode:
                    from .json_output import JSONModeError

                    raise JSONModeError(
                        "path_create_failed",
                        f"Could not create directory: {path}",
                        path=str(path),
                        reason=str(exc),
                    )
                ui.console.print(f"[red]Error: Could not create directory: {path}")
                ui.console.print(f"[dim]Reason: {exc}")
                return None
        return None

    return path
