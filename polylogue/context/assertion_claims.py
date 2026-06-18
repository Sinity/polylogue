"""Assertion-claim helpers for context composition surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_claims


def context_claim_text(*, kind: str, body_text: str | None, target_ref: str) -> str:
    text = body_text or "(empty assertion)"
    return f"{kind}: {text} [{target_ref}]"


def user_db_injectable_claim_texts(user_db_path: Path, *, session_id: str) -> list[str]:
    """Read active, explicitly injectable assertion claims for a session."""
    if not user_db_path.exists():
        return []
    uri = f"file:{user_db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        claims = list_assertion_claims(
            conn,
            target_ref=f"session:{session_id}",
            statuses=("active",),
            context_inject=True,
            limit=20,
        )
    return [
        context_claim_text(kind=claim.kind, body_text=claim.body_text, target_ref=claim.target_ref) for claim in claims
    ]


__all__ = ["context_claim_text", "user_db_injectable_claim_texts"]
