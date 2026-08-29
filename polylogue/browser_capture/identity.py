"""Provider-native browser identity contract.

This module is deliberately small and provider-neutral.  A DOM position or
rendered text is useful diagnostic evidence, but is never an identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from polylogue.core.enums import Origin, Provider

_UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def legacy_browser_capture_native_id(provider: Provider | str | None, provider_session_id: str | None) -> str | None:
    """Recover provider-native IDs from old browser-extension synthetic IDs."""
    if not provider_session_id:
        return None
    provider_value = provider.value if isinstance(provider, Provider) else provider
    if not provider_value:
        return provider_session_id
    synthetic_prefix = f"{provider_value}:"
    if provider_session_id.startswith(synthetic_prefix):
        parts = provider_session_id.split(":")
        if len(parts) == 3 and parts[1] and "/" not in parts[1]:
            return parts[1]
        if provider_value == Provider.CHATGPT.value and len(parts) == 4 and parts[1] == "WEB" and parts[2]:
            return f"WEB:{parts[2]}"
    hyphen_prefix = f"{provider_value}-"
    if provider_session_id.startswith(hyphen_prefix):
        match = _UUID_PATTERN.search(provider_session_id)
        if match:
            if provider_value == Provider.CHATGPT.value and provider_session_id.startswith("chatgpt-WEB-"):
                return f"WEB:{match.group(0)}"
            return match.group(0)
    return provider_session_id


IdentityFidelity = Literal["native", "dom_degraded", "unknown"]
IdentityDegradedReason = Literal[
    "missing_conversation_id",
    "missing_message_id",
    "adapter_drift",
    "ambiguous",
    "receiver_disagreement",
    "unsupported_provider",
]


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    origin: str
    provider_conversation_id: str | None
    provider_message_id: str | None
    parent_provider_message_id: str | None = None
    branch_context: str | None = None
    variant_index: int | None = None
    content_fingerprint: str | None = None
    dom_ordinal: int | None = None
    adapter_name: str = ""
    adapter_version: str | None = None
    observed_at: str | None = None
    fidelity: IdentityFidelity = "unknown"
    degraded_reason: IdentityDegradedReason | None = None


@dataclass(frozen=True, slots=True)
class CanonicalIdentity:
    session_ref: str | None
    message_ref: str | None = None
    evidence_ref: str | None = None
    fidelity: IdentityFidelity = "unknown"
    degraded_reason: IdentityDegradedReason | None = None
    adapter_version: str | None = None


def content_fingerprint(text: str | None) -> str | None:
    if not text:
        return None
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_origin(provider: Provider | str) -> str | None:
    value = provider.value if isinstance(provider, Provider) else str(provider)
    return {
        "chatgpt": Origin.CHATGPT_EXPORT.value,
        "claude": Origin.CLAUDE_AI_EXPORT.value,
        "claude-ai": Origin.CLAUDE_AI_EXPORT.value,
    }.get(value)


def canonical_session_ref(origin: str, conversation_id: str) -> str:
    return f"{origin}:{conversation_id}"


def canonical_message_ref(session_ref: str, message_id: str) -> str:
    return f"{session_ref}:n:{message_id}"


def resolve_identity(
    observation: IdentityObservation,
    accepted: CanonicalIdentity,
) -> CanonicalIdentity:
    """Bind an observation only to the exact identity accepted by the receiver."""
    if not observation.provider_conversation_id:
        return CanonicalIdentity(None, None, fidelity="unknown", degraded_reason="missing_conversation_id")
    if not accepted.session_ref or not accepted.session_ref.endswith(f":{observation.provider_conversation_id}"):
        return CanonicalIdentity(None, None, fidelity="unknown", degraded_reason="receiver_disagreement")
    if not observation.provider_message_id:
        return CanonicalIdentity(
            accepted.session_ref, None, fidelity="dom_degraded", degraded_reason="missing_message_id"
        )
    if accepted.adapter_version and observation.adapter_version != accepted.adapter_version:
        return CanonicalIdentity(accepted.session_ref, None, fidelity="unknown", degraded_reason="adapter_drift")
    expected = canonical_message_ref(accepted.session_ref, observation.provider_message_id)
    if accepted.message_ref != expected or observation.fidelity != "native":
        return CanonicalIdentity(
            accepted.session_ref,
            None,
            fidelity="unknown",
            degraded_reason=("receiver_disagreement" if accepted.message_ref != expected else "ambiguous"),
        )
    return CanonicalIdentity(accepted.session_ref, expected, accepted.evidence_ref, "native")


__all__ = [
    "CanonicalIdentity",
    "IdentityObservation",
    "canonical_message_ref",
    "canonical_origin",
    "canonical_session_ref",
    "content_fingerprint",
    "resolve_identity",
    "legacy_browser_capture_native_id",
]
