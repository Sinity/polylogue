import json
from pathlib import Path

from polylogue.config import persist_config, IndexConfig


def test_persist_config_includes_roots(tmp_path: Path) -> None:
    input_root = tmp_path / "inbox"
    output_root = tmp_path / "archive"
    roots = {"work": tmp_path / "work-archive"}

    config_path = persist_config(
        input_root=input_root,
        output_root=output_root,
        collapse_threshold=10,
        html_previews=True,
        html_theme="dark",
        index=IndexConfig(),
        path=tmp_path / "config.json",
        roots=roots,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["paths"]["roots"]["work"]["render"].endswith("/work-archive/render")
