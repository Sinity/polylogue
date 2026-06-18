"""Assertion-claim helpers for context composition surfaces."""

from __future__ import annotations


def context_claim_text(*, kind: str, body_text: str | None, target_ref: str) -> str:
    text = body_text or "(empty assertion)"
    return f"{kind}: {text} [{target_ref}]"


__all__ = ["context_claim_text"]
