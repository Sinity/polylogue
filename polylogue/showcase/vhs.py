"""Generate VHS tape files from showcase exercise metadata.

Converts Exercise instances with vhs_capture=True into VHS .tape files
that can be recorded with ``vhs`` to produce GIF/video captures.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from polylogue.showcase.exercises import Exercise, vhs_exercises


@dataclass(frozen=True, slots=True)
class VHSTape:
    """Renderable VHS tape artifact for one showcase exercise."""

    exercise_name: str
    content: str

    @property
    def filename(self) -> str:
        return f"{self.exercise_name}.tape"


def generate_tape(
    exercise: Exercise,
    *,
    output_width: int = 100,
    output_height: int = 30,
    font_size: int = 16,
    padding: int = 20,
) -> str:
    """Generate a VHS tape file content string from an Exercise.

    If the exercise has ``capture_steps``, those are used verbatim as the
    interaction body.  Otherwise, the tape auto-generates Type + Enter + Sleep
    from the exercise's CLI args.
    """
    output_file = f"{exercise.name}.gif"
    pixel_width = output_width * (font_size // 2 + 2)
    pixel_height = output_height * (font_size + 4)

    lines: list[str] = [
        f"# {exercise.description}",
        f"# Auto-generated from showcase exercise: {exercise.name}",
        "",
        f"Output {output_file}",
        "",
        f"Set FontSize {font_size}",
        f"Set Width {pixel_width}",
        f"Set Height {pixel_height}",
        f"Set Padding {padding}",
        "",
    ]

    if exercise.capture_steps:
        for step in exercise.capture_steps:
            lines.append(step)
    else:
        # Auto-generate from CLI args
        lines.append(f'Type "{exercise.display_command_text}"')
        lines.append("Enter")
        lines.append("Sleep 2s")

    lines.append("")
    lines.append("# Let output settle")
    lines.append("Sleep 1s")
    lines.append("")

    return VHSTape(exercise_name=exercise.name, content="\n".join(lines)).content


def generate_all_tapes(
    exercises: list[Exercise] | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate tape files for all vhs_capture exercises.

    Returns a dict mapping exercise name to tape content.
    If ``output_dir`` is provided, also writes ``.tape`` files to disk.
    """
    targets = exercises if exercises is not None else vhs_exercises()

    tapes: dict[str, str] = {}
    for ex in targets:
        if not ex.vhs_capture:
            continue
        tape = VHSTape(exercise_name=ex.name, content=generate_tape(ex))
        tapes[ex.name] = tape.content

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
