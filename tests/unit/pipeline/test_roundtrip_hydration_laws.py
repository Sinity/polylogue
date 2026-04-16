"""Roundtrip hydration laws: payload → parse → transform → save → hydrate → verify.

Proves that the durable archive path preserves semantic facts through
every stage of the pipeline. These are the most valuable property tests
in the suite — a failure here means the archive silently loses data.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.pipeline.prepare_transform import transform_to_records
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.store import AttachmentRecord
from tests.infra.storage_records import db_setup

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

PROVIDERS_WITH_SYNTHETIC = ("chatgpt", "claude-code", "claude-ai", "codex", "gemini")

_corpus_cache: dict[str, object] = {}


def _get_corpus(provider: str):
    if provider not in _corpus_cache:
        from polylogue.schemas.synthetic.core import SyntheticCorpus

        _corpus_cache[provider] = SyntheticCorpus.for_provider(provider)
    return _corpus_cache[provider]


@st.composite
def synthetic_payload(draw, providers=PROVIDERS_WITH_SYNTHETIC):
    """Generate a (provider_name, raw_bytes, unique_id) tuple from synthetic corpus."""
    provider = draw(st.sampled_from(providers))
    seed = draw(st.integers(min_value=0, max_value=2**16))
    corpus = _get_corpus(provider)
    raw_items = corpus.generate(count=1, seed=seed)
    unique_id = f"{provider}-{seed}"
    return provider, raw_items[0], unique_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_payload(raw_bytes: bytes):
    """Decode raw bytes, handling both JSON and JSONL (Codex/Claude Code)."""
    text = raw_bytes.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [json.loads(line) for line in text.strip().splitlines() if line.strip()]
        return lines


def _parse_and_transform(provider_name: str, raw_bytes: bytes, archive_root: Path, unique_id: str = "default"):
    """Run the full parse → transform path, returning (parsed, transform_result)."""
    payload = _decode_payload(raw_bytes)
    detected = detect_provider(payload)
    assert detected is not None, f"Provider detection failed for {provider_name}"

    parsed_list = parse_payload(detected, payload, f"rt-{unique_id}")
    assert len(parsed_list) >= 1, "Parser returned no conversations"
    parsed = parsed_list[0]

    result = transform_to_records(parsed, f"test-{provider_name}", archive_root=archive_root)
    return parsed, result


def _save_and_hydrate(result, db_conn):
    """Save records to DB and hydrate back to domain model."""
    from polylogue.storage.backends.queries.mappers_archive import (
        _row_to_conversation,
        _row_to_message,
    )
    from tests.infra.storage_records import store_records

    bundle = result.bundle
    store_records(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
        conn=db_conn,
    )

    cid = bundle.conversation.conversation_id
    conv_row = db_conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", (cid,)).fetchone()
    assert conv_row is not None, f"Conversation {cid} not found in DB"
    conv_record = _row_to_conversation(conv_row)

    msg_rows = db_conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY sort_key", (cid,)).fetchall()
    msg_records = [_row_to_message(r) for r in msg_rows]

    att_records: list[AttachmentRecord] = []

    hydrated = conversation_from_records(conv_record, msg_records, att_records)
    return hydrated


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
    def test_message_count_preserved(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        parsed, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)

        assert len(result.bundle.messages) == len(parsed.messages), (
            f"Transform changed message count: {len(parsed.messages)} → {len(result.bundle.messages)}"
        )

    @given(data=synthetic_payload())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_message_count_survives_save_hydrate(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            parsed, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = _save_and_hydrate(result, conn)

            assert len(list(hydrated.messages)) == len(parsed.messages), (
                f"Hydration changed message count: {len(parsed.messages)} → {len(list(hydrated.messages))}"
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
    def test_role_multiset_preserved(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            parsed, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = _save_and_hydrate(result, conn)

            parsed_roles = Counter(str(m.role) for m in parsed.messages)
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
    def test_title_preserved(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            parsed, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = _save_and_hydrate(result, conn)

            assert hydrated.title == parsed.title, f"Title changed: {parsed.title!r} → {hydrated.title!r}"


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
    def test_same_payload_same_hash(self, data, tmp_path):
        provider_name, raw_bytes, unique_id = data
        _, result1 = _parse_and_transform(provider_name, raw_bytes, tmp_path, unique_id)
        _, result2 = _parse_and_transform(provider_name, raw_bytes, tmp_path, unique_id)

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
    def test_second_import_is_noop(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection
        from tests.infra.storage_records import store_records

        with open_connection(db_path) as conn:
            _, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
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
    def test_provider_preserved(self, data, workspace_env):
        provider_name, raw_bytes, unique_id = data
        db_path = db_setup(workspace_env)

        from polylogue.storage.backends.connection import open_connection

        with open_connection(db_path) as conn:
            parsed, result = _parse_and_transform(provider_name, raw_bytes, workspace_env["archive_root"], unique_id)
            hydrated = _save_and_hydrate(result, conn)

            assert str(hydrated.provider) == str(parsed.provider_name), (
                f"Provider changed: {parsed.provider_name!r} → {hydrated.provider!r}"
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
    def test_same_payload_same_cid(self, data, tmp_path):
        provider_name, raw_bytes, unique_id = data
        _, result1 = _parse_and_transform(provider_name, raw_bytes, tmp_path, unique_id)
        _, result2 = _parse_and_transform(provider_name, raw_bytes, tmp_path, unique_id)

        assert result1.candidate_cid == result2.candidate_cid, "Same payload produced different conversation IDs"


# ---------------------------------------------------------------------------
# Parametrized per-provider smoke: each provider can complete the full path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_name", PROVIDERS_WITH_SYNTHETIC)
def test_provider_completes_full_roundtrip(provider_name, workspace_env):
    """Each provider can complete generate → parse → transform → save → hydrate."""
    corpus = _get_corpus(provider_name)
    raw_items = corpus.generate(count=1, seed=42)
    raw_bytes = raw_items[0]

    db_path = db_setup(workspace_env)

    from polylogue.storage.backends.connection import open_connection

    with open_connection(db_path) as conn:
        parsed, result = _parse_and_transform(
            provider_name, raw_bytes, workspace_env["archive_root"], f"{provider_name}-42"
        )
        hydrated = _save_and_hydrate(result, conn)

        assert hydrated is not None
        assert len(list(hydrated.messages)) > 0
        assert hydrated.title is not None or parsed.title is None
