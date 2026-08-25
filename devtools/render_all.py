"""Refresh or verify generated repository surfaces.

Each surface declares its input files via ``GeneratedSurface.inputs``.
Before rendering, a typed inventory of those files is compared against a
stored stamp (``.cache/.render-<name>-stamp``). If the complete inventory
matches, the surface is skipped because its last render is still current.

Surfaces render in registry order because generators share process-global state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devtools import system_exit
from devtools.generated_surfaces import GENERATED_SURFACES, GeneratedSurface

CACHE_DIR = Path(".cache")
_STAMP_VERSION = 2
_INPUT_SUFFIXES = {".py", ".yaml", ".md"}


@dataclass(frozen=True, slots=True)
class _InputInventory:
    patterns: tuple[str, ...]
    paths: tuple[str, ...]
    digest: str | None
    read_count: int
    missing_count: int
    unreadable_count: int
    invalid_count: int
    details: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.missing_count == 0 and self.unreadable_count == 0 and self.invalid_count == 0

    def stamp_payload(self) -> dict[str, Any]:
        return {
            "version": _STAMP_VERSION,
            "patterns": list(self.patterns),
            "paths": list(self.paths),
            "digest": self.digest,
        }


def _has_glob(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


def _surface_input_inventory(surface: GeneratedSurface) -> _InputInventory:
    """Read every declared input and retain its status for freshness decisions."""
    patterns = tuple(str(pattern) for pattern in getattr(surface, "inputs", ()))
    matched: dict[str, bytes] = {}
    read_count = missing_count = unreadable_count = invalid_count = 0
    details: list[str] = []

    def read_file(path: Path) -> str:
        nonlocal read_count, missing_count, unreadable_count
        path_text = str(path)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            missing_count += 1
            details.append(f"{path_text}: {exc}")
            return "missing"
        except OSError as exc:
            unreadable_count += 1
            details.append(f"{path_text}: {exc}")
            return "unreadable"
        matched[path_text] = data
        read_count += 1
        return "read"

    def inspect_directory(path: Path) -> None:
        nonlocal read_count, missing_count, unreadable_count
        discovered: list[Path] = []
        walk_errors: list[OSError] = []

        def on_walk_error(exc: OSError) -> None:
            walk_errors.append(exc)

        for current, _directories, filenames in os.walk(path, onerror=on_walk_error):
            discovered.extend(Path(current) / filename for filename in filenames)
        for error in walk_errors:
            if isinstance(error, FileNotFoundError):
                missing_count += 1
            else:
                unreadable_count += 1
            details.append(f"{path}: {error}")
        candidates = sorted((candidate for candidate in discovered if candidate.suffix in _INPUT_SUFFIXES), key=str)
        if not candidates and not walk_errors:
            read_count += 1
        for candidate in candidates:
            read_file(candidate)

    for pattern in patterns:
        path = Path(pattern)
        if _has_glob(pattern):
            try:
                matches = sorted(Path().glob(pattern), key=str)
            except (OSError, ValueError) as exc:
                invalid_count += 1
                details.append(f"{pattern}: invalid input pattern ({exc})")
                continue
            if not matches:
                missing_count += 1
                details.append(f"{pattern}: no declared inputs matched")
                continue
            for match in matches:
                try:
                    match_stat = match.stat()
                except FileNotFoundError as exc:
                    missing_count += 1
                    details.append(f"{match}: {exc}")
                    continue
                except OSError as exc:
                    unreadable_count += 1
                    details.append(f"{match}: {exc}")
                    continue
                if not stat.S_ISREG(match_stat.st_mode):
                    invalid_count += 1
                    details.append(f"{match}: glob matched a non-file input")
                    continue
                read_file(match)
            continue

        try:
            stat_result = path.stat()
        except FileNotFoundError as exc:
            missing_count += 1
            details.append(f"{pattern}: {exc}")
            continue
        except OSError as exc:
            unreadable_count += 1
            details.append(f"{pattern}: {exc}")
            continue

        if stat.S_ISREG(stat_result.st_mode):
            read_file(path)
        elif stat.S_ISDIR(stat_result.st_mode):
            inspect_directory(path)
        else:
            invalid_count += 1
            details.append(f"{pattern}: declared input is neither a file nor a directory")

    paths = tuple(sorted(matched))
    digest: str | None = None
    if missing_count == 0 and unreadable_count == 0 and invalid_count == 0:
        hasher = hashlib.sha256()
        for matched_path in paths:
            hasher.update(matched_path.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(matched[matched_path])
            hasher.update(b"\0")
        digest = hasher.hexdigest()
    return _InputInventory(
        patterns=patterns,
        paths=paths,
        digest=digest,
        read_count=read_count,
        missing_count=missing_count,
        unreadable_count=unreadable_count,
        invalid_count=invalid_count,
        details=tuple(details),
    )


def _stamp_path(name: str) -> Path:
    return CACHE_DIR / f".render-{name}-stamp"


def _is_fresh(surface: GeneratedSurface, inventory: _InputInventory | None = None) -> bool:
    """Return True only when a complete current inventory matches the stamp."""
    if not getattr(surface, "inputs", ()):
        return False
    if inventory is None:
        inventory = _surface_input_inventory(surface)
    if not inventory.complete:
        return False
    try:
        stamp = json.loads(_stamp_path(surface.name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(stamp, dict) and stamp == inventory.stamp_payload()


def _stamp(surface: GeneratedSurface, inventory: _InputInventory | None = None) -> None:
    """Record the complete inventory after a successful render."""
    if not getattr(surface, "inputs", ()):
        return
    inventory = inventory or _surface_input_inventory(surface)
    if not inventory.complete or inventory.digest is None:
        raise OSError("cannot stamp an incomplete declared render input inventory")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _stamp_path(surface.name).write_text(
        json.dumps(inventory.stamp_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inventory_failure(surface: GeneratedSurface, inventory: _InputInventory) -> int:
    if inventory.invalid_count:
        diagnosis = "render_input_invalid"
    elif inventory.unreadable_count:
        diagnosis = "render_input_unreadable"
    else:
        diagnosis = "render_input_missing"
    print(
        f"render all: {surface.name} failed; diagnosis: {diagnosis} "
        f"read={inventory.read_count} missing={inventory.missing_count} "
        f"unreadable={inventory.unreadable_count} invalid={inventory.invalid_count}",
        file=sys.stderr,
    )
    for detail in inventory.details[:10]:
        print(f"  {detail}", file=sys.stderr)
    return 1


def _render_one(surface: GeneratedSurface, check: bool) -> int:
    """Render or check a single surface. Returns exit code."""
    inventory = _surface_input_inventory(surface)
    if not inventory.complete:
        return _inventory_failure(surface, inventory)
    if not check and _is_fresh(surface, inventory):
        print(f"render all: skip {surface.name} (inputs unchanged)", file=sys.stderr)
        return 0

    mode = "check" if check else "render"
    print(f"render all: {mode} {surface.name}", file=sys.stderr)
    if not check:
        try:
            _stamp_path(surface.name).unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"render all: {surface.name} failed; diagnosis: render_stamp_invalidate_failed: {exc}",
                file=sys.stderr,
            )
            return 1
    try:
        result = surface.main(["--check"] if check else [])
    except SystemExit as exc:
        translation = system_exit.translate_system_exit(exc)
        if translation.message is not None:
            print(f"render all: {surface.name}: {translation.message}", file=sys.stderr)
        failure_code = translation.code or 1
        print(
            f"render all: {surface.name} failed; diagnosis: render_surface_system_exit (exit {failure_code})",
            file=sys.stderr,
        )
        return failure_code
    except Exception as exc:
        print(f"render all: {surface.name} failed; diagnosis: render_surface_exception: {exc}", file=sys.stderr)
        return 1
    if type(result) is not int:
        print(
            f"render all: {surface.name} failed; diagnosis: render_surface_invalid_result: "
            f"expected int, received {type(result).__name__}",
            file=sys.stderr,
        )
        return 1
    if result != 0:
        print(f"render all: {surface.name} failed; diagnosis: render_surface_failed (exit {result})", file=sys.stderr)
        return result
    if not check:
        try:
            _stamp(surface, inventory)
        except OSError as exc:
            print(f"render all: {surface.name} failed; diagnosis: render_stamp_write_failed: {exc}", file=sys.stderr)
            return 1
    return 0


def _selected_surfaces(skip: set[str]) -> list[GeneratedSurface]:
    return [surface for surface in GENERATED_SURFACES if surface.name not in skip]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh or verify generated repository surfaces.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any selected generated surface is out of sync.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=sorted(surface.name for surface in GENERATED_SURFACES),
        help="Skip a generated surface by name (repeatable).",
    )
    args = parser.parse_args(argv)

    selected = _selected_surfaces(set(args.skip))
    if not selected:
        print("render all: no surfaces selected", file=sys.stderr)
        return 2

    exit_code = 0
    if args.check:
        for surface in selected:
            result = _render_one(surface, check=True)
            if result != 0:
                exit_code = result if exit_code == 0 else exit_code
        return exit_code

    # Render in registry order. Several generators temporarily change the process
    # working directory or stage shared files, so thread-level parallelism can
    # make unrelated surfaces resolve paths against the wrong directory.
    for surface in selected:
        result = _render_one(surface, check=False)
        if result != 0:
            exit_code = result if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
