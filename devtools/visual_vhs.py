"""Generate VHS tape files for direct visual evidence specs."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VHSTapeSpec:
    """Renderable visual evidence target for one CLI flow."""

    name: str
    description: str
    display_command: tuple[str, ...] = ("polylogue",)
    capture_steps: tuple[str, ...] = ()

    @property
    def display_command_text(self) -> str:
        return " ".join(self.display_command)


@dataclass(frozen=True, slots=True)
class VHSTape:
    """Renderable VHS tape artifact for one visual evidence spec."""

    spec_name: str
    content: str

    @property
    def filename(self) -> str:
        return f"{self.spec_name}.tape"


DEFAULT_TAPE_SPECS: tuple[VHSTapeSpec, ...] = (
    VHSTapeSpec(
        name="help-main",
        description="Main help screen",
        display_command=("polylogue", "--help"),
    ),
    VHSTapeSpec(
        name="stats-default",
        description="Default archive statistics",
        display_command=("polylogue",),
    ),
    VHSTapeSpec(
        name="query-list",
        description="List sessions",
        display_command=("polylogue", "read", "--all", "-n", "1"),
    ),
    VHSTapeSpec(
        name="doctor-readiness",
        description="Archive readiness doctor",
        display_command=("polylogue", "ops", "doctor"),
    ),
    VHSTapeSpec(
        name="query-latest-md",
        description="Latest session as Markdown",
        display_command=("polylogue", "--latest", "-f", "markdown"),
    ),
)


def default_tape_specs() -> tuple[VHSTapeSpec, ...]:
    """Return the committed default visual evidence tape inventory."""
    return DEFAULT_TAPE_SPECS


def generate_tape(
    spec: VHSTapeSpec,
    *,
    output_width: int = 100,
    output_height: int = 30,
    font_size: int = 16,
    padding: int = 20,
) -> str:
    """Generate a VHS tape file content string from a direct tape spec.

    If the spec has ``capture_steps``, those are used verbatim. Otherwise,
    the tape auto-generates Type + Enter + Sleep from the display command.
    """
    output_file = f"{spec.name}.gif"
    pixel_width = output_width * (font_size // 2 + 2)
    pixel_height = output_height * (font_size + 4)

    lines: list[str] = [
        f"# {spec.description}",
        f"# Auto-generated from visual evidence spec: {spec.name}",
        "",
        f"Output {output_file}",
        "",
        f"Set FontSize {font_size}",
        f"Set Width {pixel_width}",
        f"Set Height {pixel_height}",
        f"Set Padding {padding}",
        "",
    ]

    if spec.capture_steps:
        for step in spec.capture_steps:
            lines.append(step)
    else:
        lines.append(f'Type "{spec.display_command_text}"')
        lines.append("Enter")
        lines.append("Sleep 2s")

    lines.append("")
    lines.append("# Let output settle")
    lines.append("Sleep 1s")
    lines.append("")

    return VHSTape(spec_name=spec.name, content="\n".join(lines)).content


def generate_all_tapes(
    specs: Sequence[VHSTapeSpec] | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate tape files for default or explicitly supplied tape specs.

    Returns a dict mapping spec name to tape content.
    If ``output_dir`` is provided, also writes ``.tape`` files to disk.
    """
    targets = specs if specs is not None else default_tape_specs()

    tapes: dict[str, str] = {}
    for spec in targets:
        tape = VHSTape(spec_name=spec.name, content=generate_tape(spec))
        tapes[spec.name] = tape.content

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / tape.filename).write_text(tape.content)

    return tapes


def run_vhs_capture(
    tape_path: Path,
    output_path: Path,
    *,
    timeout: float = 60.0,
) -> bool:
    """Run ``vhs`` binary against a tape file.

    Returns True if the capture succeeded, False otherwise.
    """
    try:
        subprocess.run(
            ["vhs", str(tape_path)],
            cwd=str(output_path.parent),
            timeout=timeout,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return True


def check_vhs_available() -> bool:
    """Check if the ``vhs`` binary is available in PATH."""
    return shutil.which("vhs") is not None


__all__ = [
    "DEFAULT_TAPE_SPECS",
    "VHSTape",
    "VHSTapeSpec",
    "check_vhs_available",
    "default_tape_specs",
    "generate_all_tapes",
    "generate_tape",
    "run_vhs_capture",
]
