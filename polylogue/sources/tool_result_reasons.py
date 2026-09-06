"""The one mapping from structural tool-result evidence to an unknown reason.

Every parser that builds a ``tool_result`` block states three structural facts
about the record it just read -- whether it resolved a verdict, whether the
record carried an outcome-bearing field at all, and whether the payload that
would carry one survived acquisition -- and this module turns them into the
normalized :class:`ToolResultUnknownReason`. Provider-wire decoding stays in
the parser; the vocabulary lives here once.

``DISTRUSTED`` has no derivation: it is the parser positively refusing a
verdict the provider did state, so the refusing site names it directly.
"""

from __future__ import annotations

from polylogue.core.enums import ToolResultUnknownReason

__all__ = ["unknown_reason"]


def unknown_reason(
    *,
    is_error: bool | None,
    exit_code: int | None = None,
    outcome_field_present: bool = False,
    source_intact: bool = True,
) -> str | None:
    """Return the reason this result's outcome is unknown, or ``None`` if known.

    ``outcome_field_present`` means the record carried an outcome-bearing
    field whose value the parser's declared mapping does not cover -- a
    structure with a verdict in it that this parser did not read. ``source_intact``
    is false when the source declared such a payload but did not retain it.
    """
    if is_error is not None or exit_code is not None:
        return None
    if not source_intact:
        return ToolResultUnknownReason.SOURCE_TRUNCATED.value
    if outcome_field_present:
        return ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT.value
    return ToolResultUnknownReason.NOT_REPORTED.value
