"""Seam B — the CLI's one rendering layer over declared operation results.

Everything under this package consumes an operation result (``result.value``,
already validated against the operation's declared result model) and turns it
into bytes on a terminal plus one process exit code. Nothing here opens an
archive, compiles a query spec, or decides a terminal outcome: the outcome is
decided once at the operation boundary (``polylogue.surfaces.outcome``) and is
only *read* here, and the exit code is computed only by
:func:`polylogue.surfaces.outcome.outcome_exit_code`.
"""

from __future__ import annotations

__all__: list[str] = []
