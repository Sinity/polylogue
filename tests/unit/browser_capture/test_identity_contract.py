from polylogue.browser_capture.identity import (
    CanonicalIdentity,
    IdentityObservation,
    canonical_message_ref,
    resolve_identity,
)


def native(conversation: str = "conv", message: str = "message") -> IdentityObservation:
    return IdentityObservation(
        origin="chatgpt-export",
        provider_conversation_id=conversation,
        provider_message_id=message,
        adapter_name="fixture-native-v1",
        fidelity="native",
    )


def test_exact_native_identity_resolves_to_receiver_ack() -> None:
    session = "chatgpt-export:conv"
    result = resolve_identity(
        native(),
        CanonicalIdentity(session, canonical_message_ref(session, "message"), "artifact#message:message", "native"),
    )
    assert result.message_ref == "chatgpt-export:conv:n:message"
    assert result.evidence_ref == "artifact#message:message"


def test_reordering_duplicate_text_and_wrong_ack_cannot_cross_bind() -> None:
    session = "chatgpt-export:conv"
    result = resolve_identity(
        native(message="message-a"),
        CanonicalIdentity(session, canonical_message_ref(session, "message-b"), fidelity="native"),
    )
    assert result.message_ref is None
    assert result.degraded_reason == "receiver_disagreement"


def test_missing_id_or_dom_only_is_degraded_and_not_authoritative() -> None:
    observation = IdentityObservation(
        origin="claude-ai-export",
        provider_conversation_id="conv",
        provider_message_id=None,
        adapter_name="claude-dom-v1",
        dom_ordinal=0,
        fidelity="dom_degraded",
        degraded_reason="missing_message_id",
    )
    result = resolve_identity(observation, CanonicalIdentity("claude-ai-export:conv", fidelity="unknown"))
    assert result.session_ref == "claude-ai-export:conv"
    assert result.message_ref is None
    assert result.fidelity == "dom_degraded"


def test_missing_conversation_id_is_unknown_even_with_a_dom_hint() -> None:
    result = resolve_identity(
        IdentityObservation(
            origin="chatgpt-export",
            provider_conversation_id=None,
            provider_message_id="m",
            adapter_name="chatgpt-dom-v1",
            dom_ordinal=2,
            fidelity="dom_degraded",
        ),
        CanonicalIdentity("chatgpt-export:conv", fidelity="unknown"),
    )
    assert result.session_ref is None
    assert result.degraded_reason == "missing_conversation_id"


def test_adapter_version_drift_is_typed_unknown() -> None:
    result = resolve_identity(
        IdentityObservation(
            "chatgpt-export", "conv", "m", adapter_name="fixture", adapter_version="1", fidelity="native"
        ),
        CanonicalIdentity("chatgpt-export:conv", "chatgpt-export:conv:n:m", fidelity="native", adapter_version="2"),
    )
    assert result.message_ref is None
    assert result.degraded_reason == "adapter_drift"
