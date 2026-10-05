"""Path authority for fixture cursor writes.

Every cursor write carries the canonical coordinate and profile of the file it
describes. A fixture seeding a cursor directly names its synthetic source by
its resolved path and claims no profile namespace.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.live.cursor import CursorPathAuthority


def fixture_cursor_authority(path: Path | str) -> CursorPathAuthority:
    """The authority a fixture claims for the synthetic source at ``path``."""
    return CursorPathAuthority(str(Path(path).resolve()), None)
