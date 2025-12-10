from pathlib import Path

from polylogue.cli.app import _find_anchor_line


def test_find_anchor_line(tmp_path: Path) -> None:
    target = tmp_path / "conversation.md"
    target.write_text(
        """
line-one
<a id=\"msg-2\"></a>
body
""".strip()
    )

    assert _find_anchor_line(target, "msg-2") == 2
    assert _find_anchor_line(target, "#msg-2") == 2
    assert _find_anchor_line(target, "msg-99") is None
