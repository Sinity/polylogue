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
                f"export POLYLOGUE_DEMO_ROOT={root_text!r}",
                "export POLYLOGUE_FORCE_PLAIN=1",
                'polylogue demo tour --root "$POLYLOGUE_DEMO_ROOT" --out-dir polylogue-demo-tour --force',
                'polylogue demo seed --root "$POLYLOGUE_DEMO_ROOT" --force --with-overlays --format json',
                'polylogue demo verify --root "$POLYLOGUE_DEMO_ROOT" --require-overlays --format json',
                'POLYLOGUE_ARCHIVE_ROOT="$POLYLOGUE_DEMO_ROOT" polylogue find pytest then read --view messages --limit 3',
                'POLYLOGUE_ARCHIVE_ROOT="$POLYLOGUE_DEMO_ROOT" polylogue find pytest then analyze --facets',
            ]
        )
        + "\n"
    )


__all__ = ["render_demo_script"]
