"""The human_authored-implies-user-role law at the parser boundary (polylogue-n6gnd).

All three session-count paths in the index writer count
``material_origin='human_authored'`` messages into ``authored_user_*`` columns
with no role filter, while the ``user_*`` columns filter ``role='user'``. Those
columns only mean "authored USER words" if no parser can emit
``human_authored`` on a non-user role. No CHECK constraint expresses that
cross-column implication, so it is pinned here over the real fixture corpus.
"""

from __future__ import annotations

import pytest

from polylogue.core.enums import MaterialOrigin, Role
from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence
from tests.infra.origin_capability_matrix import CapabilityWitness, load_manifest, load_witness_fixture


def _supported_witnesses() -> list[tuple[str, CapabilityWitness]]:
    manifest = load_manifest()
    witnesses: list[tuple[str, CapabilityWitness]] = []
    for entry in manifest.entries:
        if entry.unsupported is not None:
            continue
        for witness in entry.witnesses:
            if witness.route == "vendor":
                # Vendor-client routes need a language-server shim; they are
                # covered by tests/unit/sources/test_origin_capability_matrix.py.
                continue
            witnesses.append((str(entry.origin), witness))
    return witnesses


_WITNESSES = _supported_witnesses()


@pytest.mark.parametrize(("origin", "witness"), _WITNESSES, ids=[name for name, _ in _WITNESSES])
def test_human_authored_messages_always_carry_the_user_role(origin: str, witness: CapabilityWitness) -> None:
    """Anti-vacuity: a parser emitting HUMAN_AUTHORED on a non-user role turns this red.

    The denominator assertion below also fails if the witness stops producing
    messages at all, so a parser that silently drops its corpus cannot pass
    this law vacuously.
    """
    claim = witness.parser_claims[0]
    payload = load_witness_fixture(witness)
    sessions = parse_payload(
        claim.provider,
        payload,
        witness.fallback_id,
        source_path=witness.fixture_path,
    )
    accepted = require_positive_conversational_evidence(
        sessions,
        provider=claim.provider,
        source_path=witness.fixture_path,
    )

    messages = [message for session in accepted for message in session.messages]
    assert messages, f"{origin}: witness produced no messages"

    offenders = [
        (message.provider_message_id, str(message.role))
        for message in messages
        if message.material_origin is MaterialOrigin.HUMAN_AUTHORED and message.role is not Role.USER
    ]
    assert offenders == [], f"{origin}: human_authored on non-user roles: {offenders}"


def test_the_corpus_actually_contains_human_authored_messages() -> None:
    """Denominator guard: the law above is only meaningful over a non-empty population."""
    human_authored = 0
    for _origin, witness in _WITNESSES:
        claim = witness.parser_claims[0]
        payload = load_witness_fixture(witness)
        sessions = parse_payload(
            claim.provider,
            payload,
            witness.fallback_id,
            source_path=witness.fixture_path,
        )
        for session in sessions:
            for message in session.messages:
                if message.material_origin is MaterialOrigin.HUMAN_AUTHORED:
                    human_authored += 1
    assert human_authored > 0
