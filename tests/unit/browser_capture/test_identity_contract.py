import pytest

from polylogue.browser_capture.identity import (
    CanonicalIdentity,
    IdentityObservation,
    canonical_message_ref,
    resolve_identity,
)
from polylogue.core.enums import Provider


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


@pytest.mark.parametrize(
    ("provider_session_id", "expected"),
    [
        ("chatgpt:conv-1:tail", "conv-1"),
        ("chatgpt:WEB:conv-1:tail", "WEB:conv-1"),
        ("chatgpt:with/slash:tail", "chatgpt:with/slash:tail"),
        ("chatgpt::tail", "chatgpt::tail"),
        ("chatgpt:a:b:c:d", "chatgpt:a:b:c:d"),
        ("chatgpt:" + ":" * 50_000, "chatgpt:" + ":" * 50_000),
    ],
)
def test_legacy_native_id_recovery_is_unchanged_by_its_split_bound(provider_session_id: str, expected: str) -> None:
    """Bounding the split must not move any recognised or rejected shape.

    Anti-vacuity: change the bound to 3 and the ``chatgpt:WEB:conv-1:tail`` row
    regresses from ``WEB:conv-1`` to the unrecovered input; remove the bound
    and this stays green, which is why the companion test below asserts the
    bound itself.
    """
    from polylogue.browser_capture.identity import legacy_browser_capture_native_id

    assert legacy_browser_capture_native_id(Provider.CHATGPT, provider_session_id) == expected


def test_legacy_native_id_recovery_splits_under_a_bound() -> None:
    """Untrusted IDs are never split into one object per character.

    ``provider_session_id`` is untrusted browser-capture input with no length
    bound; an unbounded ``split(":")`` on a colon-dense value allocates one
    ``str`` object per character, amplifying a crafted payload far past its own
    byte size. Only three- and four-part shapes are recognised, so five parts
    is sufficient.

    Anti-vacuity: drop the ``maxsplit`` argument and this goes red on the
    missing bound.
    """
    import ast
    import inspect

    from polylogue.browser_capture import identity

    tree = ast.parse(inspect.getsource(identity.legacy_browser_capture_native_id))
    bounds = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
    ]
    assert bounds == [4], bounds
