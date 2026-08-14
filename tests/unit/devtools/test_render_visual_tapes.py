from pathlib import Path

from devtools import render_visual_tapes
from devtools.visual_vhs import default_tape_specs


def test_visual_tape_command_writes_every_default_spec(tmp_path: Path) -> None:
    assert render_visual_tapes.main(["--output-dir", str(tmp_path)]) == 0

    expected = {f"{spec.name}.tape" for spec in default_tape_specs()}
    written = {path.name for path in tmp_path.glob("*.tape")}
    assert written == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
