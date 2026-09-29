"""Delete a session through the facade's prepare-then-present route."""

from __future__ import annotations

from polylogue.api import Polylogue


async def delete_session_with_preview(poly: Polylogue, session_id: str) -> bool:
    """Prepare one session delete, present its preview, and report the effect.

    Returns ``False`` for an unknown session, which has no preview to present.
    """
    preview = await poly.prepare_delete_session(session_id)
    if preview.preview_ref is None:
        return False
    return await poly.delete_session(session_id, preview_ref=preview.preview_ref)
