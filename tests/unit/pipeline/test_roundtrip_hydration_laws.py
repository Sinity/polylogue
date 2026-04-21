"""Roundtrip hydration laws: payload → parse → transform → save → hydrate → verify.

Proves that the durable archive path preserves semantic facts through
every stage of the pipeline. These are the most valuable property tests
in the suite — a failure here means the archive silently loses data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TypeAlias

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.schemas.synthetic.core import SyntheticCorpus
from tests.infra.pipeline_roundtrip import parse_and_transform_payload, save_transform_and_hydrate
from tests.infra.storage_records import db_setup

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

PROVIDERS_WITH_SYNTHETIC = ("chatgpt", "claude-code", "claude-ai", "codex", "gemini")
SyntheticPayload: TypeAlias = tuple[str, bytes, str]

_corpus_cache: dict[str, SyntheticCorpus] = {}


def _get_corpus(provider: str) -> SyntheticCorpus:
    if provider not in _corpus_cache:
        _corpus_cache[provider] = SyntheticCorpus.for_provider(provider)
    return _corpus_cache[provider]


@st.composite
def synthetic_payload(
    draw: st.DrawFn,
    providers: tuple[str, ...] = PROVIDERS_WITH_SYNTHETIC,
) -> SyntheticPayload:
    """Generate a (provider_name, raw_bytes, unique_id) tuple from synthetic corpus."""
    provider = draw(st.sampled_from(providers))
    seed = draw(st.integers(min_value=0, max_value=2**16))
    corpus = _get_corpus(provider)
    raw_items = corpus.generate(count=1, seed=seed)
    unique_id = f"{provider}-{seed}"
    return provider, raw_items[0], unique_id


# ---------------------------------------------------------------------------
# Law 1: Message count preserved through the full pipeline
# ---------------------------------------------------------------------------


class TestMessageCountPreservation:
    @given(data=synthetic_payload())
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_message_count_preserved(self: object, data: SyntheticPayload, workspace_env: dict[str, Path]) -> None:
        provider_name, raw_bytes, unique_id = data
        roundtrip = parse_and_transform_payload(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)

        assert len(roundtrip.transform.bundle.messages) == len(roundtrip.parsed.messages), (
            f"Transform changed message count: {len(roundtrip.parsed.messages)} → "
            f"{len(roundtrip.transform.bundle.messages)}"
        )

    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_message_count_survives_save_hydrate(
        self: object, data: SyntheticPayload, workspace_env: dict[str, Path]
    ) -> None:
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_and_transform_payload(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)

            assert len(list(hydrated.messages)) == len(roundtrip.parsed.messages), (
                f"Hydration changed message count: {len(roundtrip.parsed.messages)} → {len(list(hydrated.messages))}"
            )


# ---------------------------------------------------------------------------
# Law 2: Role multiset preserved
# ---------------------------------------------------------------------------


class TestRolePreservation:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_role_multiset_preserved(self: object, data: SyntheticPayload, workspace_env: dict[str, Path]) -> None:
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_and_transform_payload(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)

            parsed_roles = Counter(str(m.role) for m in roundtrip.parsed.messages)
            hydrated_roles = Counter(str(m.role) for m in hydrated.messages)
            assert parsed_roles == hydrated_roles, f"Role multiset changed: {parsed_roles} → {hydrated_roles}"


# ---------------------------------------------------------------------------
# Law 3: Title stability
# ---------------------------------------------------------------------------


class TestTitleStability:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_title_preserved(self: object, data: SyntheticPayload, workspace_env: dict[str, Path]) -> None:
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_and_transform_payload(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)

            assert hydrated.title == roundtrip.parsed.title, (
                f"Title changed: {roundtrip.parsed.title!r} → {hydrated.title!r}"
            )


# ---------------------------------------------------------------------------
# Law 4: Content hash determinism
# ---------------------------------------------------------------------------


class TestContentHashDeterminism:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_same_payload_same_hash(self: object, data: SyntheticPayload, tmp_path: Path) -> None:
        provider_name, raw_bytes, unique_id = data
        result1 = parse_and_transform_payload(provider_name, raw_bytes, tmp_path, unique_id).transform
        result2 = parse_and_transform_payload(provider_name, raw_bytes, tmp_path, unique_id).transform

        assert result1.content_hash == result2.content_hash, "Same payload produced different content hashes"


# ---------------------------------------------------------------------------
# Law 5: Idempotent re-import
# ---------------------------------------------------------------------------


class TestIdempotentReimport:
    @given(data=synthetic_payload())
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_second_import_is_noop(self: object, data: SyntheticPayload, workspace_env: dict[str, Path]) -> None:
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection
        from tests.infra.storage_records import store_records

        with open_connection(db_path) as conn:
            result = parse_and_transform_payload(
                provider_name, raw_bytes, workspace_env["archive_root"], unique_id
            ).transform
            bundle = result.bundle

            store_records(
                conversation=bundle.conversation,
                messages=bundle.messages,
                attachments=bundle.attachments,
                conn=conn,
            )

            counts2 = store_records(
                conversation=bundle.conversation,
                messages=bundle.messages,
                attachments=bundle.attachments,
                conn=conn,
            )

            assert counts2["conversations"] == 0, "Re-import should not re-insert conversation"
            assert counts2["messages"] == 0, "Re-import should not re-insert messages"


# ---------------------------------------------------------------------------
# Law 6: Provider identity preserved
# ---------------------------------------------------------------------------


class TestProviderIdentity:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_provider_preserved(self: object, data: SyntheticPayload, workspace_env: dict[str, Path]) -> None:
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_and_transform_payload(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = save_transform_and_hydrate(roundtrip.transform, conn)

            assert str(hydrated.provider) == str(roundtrip.parsed.provider_name), (
                f"Provider changed: {roundtrip.parsed.provider_name!r} → {hydrated.provider!r}"
            )


# ---------------------------------------------------------------------------
# Law 7: Conversation ID determinism
# ---------------------------------------------------------------------------


class TestConversationIdDeterminism:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_same_payload_same_cid(self: object, data: SyntheticPayload, tmp_path: Path) -> None:
        provider_name, raw_bytes, unique_id = data
        result1 = parse_and_transform_payload(provider_name, raw_bytes, tmp_path, unique_id).transform
        result2 = parse_and_transform_payload(provider_name, raw_bytes, tmp_path, unique_id).transform

        assert result1.candidate_cid == result2.candidate_cid, "Same payload produced different conversation IDs"


# ---------------------------------------------------------------------------
# Parametrized per-provider smoke: each provider can complete the full path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_name", PROVIDERS_WITH_SYNTHETIC)
def test_provider_completes_full_roundtrip(provider_name: str, workspace_env: dict[str, Path]) -> None:
    """Each provider can complete generate → parse → transform → save → hydrate."""
    corpus = _get_corpus(provider_name)
    raw_items = corpus.generate(count=1, seed=42)
    raw_bytes = raw_items[0]

    db_path = db_setup(workspace_env)

    from polylogue.storage.backends.connection import open_connection

    with open_connection(db_path) as conn:
        roundtrip = parse_and_transform_payload(
            provider_name, raw_bytes, workspace_env["archive_root"], f"{provider_name}-42"
        )
        hydrated = save_transform_and_hydrate(roundtrip.transform, conn)

        assert hydrated is not None
        assert len(list(hydrated.messages)) > 0
        assert hydrated.title is not None or roundtrip.parsed.title is None
