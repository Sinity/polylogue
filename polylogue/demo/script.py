"""Shell script rendering for the deterministic demo archive."""

from __future__ import annotations

from pathlib import Path


def render_demo_script(root: Path, *, shell: str = "bash") -> str:
    """Return a copy-pastable demo script for local docs and recordings."""

    if shell != "bash":
        raise ValueError("only bash demo scripts are supported")
    root_text = str(root)
    return (
        "\n".join(
            [
                "set -euo pipefail",
                f"export POLYLOGUE_ARCHIVE_ROOT={root_text!r}",
                "export POLYLOGUE_FORCE_PLAIN=1",
                "polylogue demo tour --out-dir polylogue-demo-tour --force",
                'polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json',
                'polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json',
                "polylogue find pytest then read --view messages --limit 3",
                "polylogue find pytest then analyze --facets",
            ]
        )
        + "\n"
    )


__all__ = ["render_demo_script"]
