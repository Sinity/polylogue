from __future__ import annotations

from pathlib import Path

from polylogue.cli_common import resolve_inputs
def test_resolve_inputs_plain_requires_explicit_selection(monkeypatch, tmp_path):
    # Non-TTY should refuse to auto-select
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    files = [data_dir / f"file-{idx}.json" for idx in range(3)]
    for f in files:
        f.write_text("{}", encoding="utf-8")

    resolved = resolve_inputs(data_dir, plain=True)

    assert resolved is None
