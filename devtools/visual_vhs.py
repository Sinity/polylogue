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
    output_width: int = 120
    output_height: int = 32
    font_size: int = 16
    padding: int = 20

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
        name="demo-tour",
        description="One-command deterministic public demo tour",
        display_command=("polylogue", "demo", "tour"),
        capture_steps=(
            'Type "polylogue demo tour --out-dir demo-tour --force"',
            "Enter",
            "Sleep 12s",
            'Type "cat demo-tour/report.md"',
            "Enter",
            "Sleep 2s",
        ),
        output_width=116,
        output_height=36,
        font_size=18,
        padding=18,
    ),
    VHSTapeSpec(
        name="query-tour",
        description="Query/read drilldown against the deterministic demo archive",
        display_command=("polylogue", "find", "pytest", "then", "read", "--view", "messages"),
        capture_steps=(
            'Type "polylogue demo seed --root query-tour/archive --force --with-overlays --format json"',
            "Enter",
            "Sleep 4s",
            'Type "POLYLOGUE_ARCHIVE_ROOT=query-tour/archive polylogue find pytest then read --view messages --limit 3"',
            "Enter",
            "Sleep 3s",
            'Type "POLYLOGUE_ARCHIVE_ROOT=query-tour/archive polylogue find pytest then analyze --facets"',
            "Enter",
            "Sleep 2s",
        ),
        output_width=132,
        output_height=34,
    ),
    VHSTapeSpec(
        name="reader-evidence-tour",
        description="Browserless local reader evidence lane against synthetic fixtures",
        display_command=("devtools", "lab", "smoke", "run", "reader-visual-smoke"),
        capture_steps=(
            'Type "devtools lab smoke run reader-visual-smoke --json --report-dir reader-evidence-tour"',
            "Enter",
            "Sleep 5s",
            "Type \"python -m json.tool reader-evidence-tour/reader-visual-smoke.json | sed -n '1,80p'\"",
            "Enter",
            "Sleep 2s",
        ),
        output_width=132,
        output_height=34,
    ),
    VHSTapeSpec(
        name="browser-capture-tour",
        description="Browser-backed deterministic ChatGPT/Claude capture proof",
        display_command=("devtools", "workspace", "dev-loop", "--browser-provider-smoke"),
        capture_steps=(
            'Type "devtools workspace dev-loop --browser-provider-smoke --json > browser-capture-tour.json"',
            "Enter",
            "Sleep 18s",
            "Type \"python - <<'PY'\"",
            "Enter",
            'Type "import json"',
            "Enter",
            "Type \"p=json.load(open('browser-capture-tour.json'))['browser_provider_smoke']\"",
            "Enter",
            "Type \"print('ok', p['ok'])\"",
            "Enter",
            "Type \"print('providers', p['provider_statuses'])\"",
            "Enter",
            "Type \"print('popup', p['popup_status'])\"",
            "Enter",
            "Type \"print('screenshot', p['artifacts'].get('popup_screenshot'))\"",
            "Enter",
            "Type \"print('artifact_refs', p['artifact_refs'])\"",
            "Enter",
            'Type "PY"',
            "Enter",
            "Sleep 2s",
        ),
        output_width=132,
        output_height=34,
    ),
)


def default_tape_specs() -> tuple[VHSTapeSpec, ...]:
    """Return the committed default visual evidence tape inventory."""
    return DEFAULT_TAPE_SPECS


def generate_tape(
    spec: VHSTapeSpec,
    *,
    output_width: int | None = None,
    output_height: int | None = None,
    font_size: int | None = None,
    padding: int | None = None,
) -> str:
    """Generate a VHS tape file content string from a direct tape spec.

    If the spec has ``capture_steps``, those are used verbatim. Otherwise,
    the tape auto-generates Type + Enter + Sleep from the display command.
    """
    output_file = f"{spec.name}.gif"
    resolved_output_width = output_width if output_width is not None else spec.output_width
    resolved_output_height = output_height if output_height is not None else spec.output_height
    resolved_font_size = font_size if font_size is not None else spec.font_size
    resolved_padding = padding if padding is not None else spec.padding
    pixel_width = resolved_output_width * (resolved_font_size // 2 + 2)
    pixel_height = resolved_output_height * (resolved_font_size + 4)

    lines: list[str] = [
        f"# {spec.description}",
        f"# Auto-generated from visual evidence spec: {spec.name}",
        "",
        f"Output {output_file}",
        "",
        f"Set FontSize {resolved_font_size}",
        f"Set Width {pixel_width}",
        f"Set Height {pixel_height}",
        f"Set Padding {resolved_padding}",
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
