from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ui import UI


@dataclass
class PathPolicy:
    """Policy for handling missing paths."""

    should_exist: bool = True
    create_if_missing: bool = False
    prompt_create: bool = False

    @staticmethod
    def must_exist() -> "PathPolicy":
        return PathPolicy(should_exist=True)

    @staticmethod
    def create_ok() -> "PathPolicy":
        return PathPolicy(should_exist=False, create_if_missing=True)

    @staticmethod
    def prompt_create() -> "PathPolicy":
        return PathPolicy(should_exist=False, prompt_create=True)


def resolve_path(path: Path, policy: PathPolicy, ui: "UI", *, json_mode: bool = False) -> Optional[Path]:
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
        if ui.confirm(f"Create directory {path}?", default=True):
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


__all__ = ["PathPolicy", "resolve_path"]

