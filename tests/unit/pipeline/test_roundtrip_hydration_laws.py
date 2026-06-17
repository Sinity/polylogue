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

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.schemas.synthetic.core import SyntheticCorpus
from tests.infra.pipeline_roundtrip import parse_payload_roundtrip, write_and_hydrate
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
    """Generate a (source_name, raw_bytes, unique_id) tuple from synthetic corpus."""
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
    def test_message_count_survives_save_hydrate(
        self: object, data: SyntheticPayload, workspace_env: dict[str, Path]
    ) -> None:
        source_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_payload_roundtrip(source_name, raw_bytes, unique_id)
            hydrated = write_and_hydrate(roundtrip, conn)

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
        source_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_payload_roundtrip(source_name, raw_bytes, unique_id)
            hydrated = write_and_hydrate(roundtrip, conn)

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
        source_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_payload_roundtrip(source_name, raw_bytes, unique_id)
            hydrated = write_and_hydrate(roundtrip, conn)

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
    def test_same_payload_same_hash(self: object, data: SyntheticPayload) -> None:
        source_name, raw_bytes, unique_id = data
        result1 = parse_payload_roundtrip(source_name, raw_bytes, unique_id)
        result2 = parse_payload_roundtrip(source_name, raw_bytes, unique_id)

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
        source_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            result = parse_payload_roundtrip(source_name, raw_bytes, unique_id)

            session_id = write_parsed_session_to_archive(conn, result.parsed, content_hash=result.content_hash)
            first_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            first_messages = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

            write_parsed_session_to_archive(conn, result.parsed, content_hash=result.content_hash)
            second_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            second_messages = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

            assert second_sessions == first_sessions, "Re-import should not re-insert session"
            assert second_messages == first_messages, "Re-import should not re-insert messages"


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
        source_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            roundtrip = parse_payload_roundtrip(source_name, raw_bytes, unique_id)
            hydrated = write_and_hydrate(roundtrip, conn)

            expected_origin = origin_from_provider(Provider.from_string(str(roundtrip.parsed.source_name))).value
            assert str(hydrated.origin) == expected_origin, (
                f"Origin changed: {roundtrip.parsed.source_name!r} → {hydrated.origin!r}"
            )


# ---------------------------------------------------------------------------
# Law 7: Session ID determinism
# ---------------------------------------------------------------------------


class TestSessionIdDeterminism:
    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_same_payload_same_cid(self: object, data: SyntheticPayload) -> None:
        from polylogue.core.identity_law import session_id as archive_session_id

        source_name, raw_bytes, unique_id = data
        p1 = parse_payload_roundtrip(source_name, raw_bytes, unique_id).parsed
        p2 = parse_payload_roundtrip(source_name, raw_bytes, unique_id).parsed

        cid1 = archive_session_id(origin_from_provider(p1.source_name).value, p1.provider_session_id)
        cid2 = archive_session_id(origin_from_provider(p2.source_name).value, p2.provider_session_id)
        assert cid1 == cid2, "Same payload produced different session IDs"


# ---------------------------------------------------------------------------
# Parametrized per-provider smoke: each provider can complete the full path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_name", PROVIDERS_WITH_SYNTHETIC)
def test_provider_completes_full_roundtrip(source_name: str, workspace_env: dict[str, Path]) -> None:
    """Each provider can complete generate → parse → transform → save → hydrate."""
    corpus = _get_corpus(source_name)
    raw_items = corpus.generate(count=1, seed=42)
    raw_bytes = raw_items[0]

    db_path = db_setup(workspace_env)

    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        roundtrip = parse_payload_roundtrip(source_name, raw_bytes, f"{source_name}-42")
        hydrated = write_and_hydrate(roundtrip, conn)

        assert hydrated is not None
        assert len(list(hydrated.messages)) > 0
        assert hydrated.title is not None or roundtrip.parsed.title is None
