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
                'polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays',
                'polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays',
                "polylogue find pytest then read --view summary --limit 1",
                "polylogue find assertions where kind:decision then read --format json --limit 5",
            ]
        )
        + "\n"
    )


__all__ = ["render_demo_script"]
