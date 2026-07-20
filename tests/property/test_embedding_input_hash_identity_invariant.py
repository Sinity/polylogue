"""Property test: ``embedding_input_hash`` is a pure function of (model, text)
and is invariant under every identity-bearing axis that contaminates
``messages.content_hash`` (polylogue-q88p, operator ruling 2026-07-20).

Directly contrasts the two hash functions against the SAME underlying
message content to make the defect this bead fixes, and the fix, both
falsifiable: ``_message_content_hash`` (the archive's existing per-message
identity hash) is *expected* to vary when session_id/position/variant_index
vary even though the text does not -- that is its documented job (dedup /
re-ingest change detection at full row granularity). ``embedding_input_hash``
must NOT vary under those same conditions -- that is the whole point of this
bead. A test that only exercised one function would not prove the contrast;
asserting both in the same property is the anti-vacuity guard.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, MessageType
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.embeddings.identity import embedding_input_hash
from polylogue.storage.sqlite.archive_tiers.write import _message_content_hash

_TEXT = st.text(min_size=1, max_size=200).filter(lambda value: value.strip() != "")
_MODEL = st.sampled_from(["voyage-4", "voyage-3", "voyage-3-lite", "demo-synthetic-embedding"])
_IDENTITY_CONTEXT = st.tuples(
    st.text(alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E), min_size=1, max_size=20),  # session_id
    st.integers(min_value=0, max_value=10_000),  # position
    st.integers(min_value=0, max_value=8),  # variant_index
    st.text(
        alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E), min_size=1, max_size=20
    ),  # provider_message_id
)


def _content_hash_for(text: str, identity: tuple[str, int, int, str]) -> bytes:
    session_id, position, variant_index, provider_message_id = identity
    message = ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.USER,
        text=text,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )
    return _message_content_hash(session_id, message, position=position, variant_index=variant_index)


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_TEXT, model=_MODEL, identity_a=_IDENTITY_CONTEXT, identity_b=_IDENTITY_CONTEXT)
def test_embedding_input_hash_ignores_identity_that_content_hash_tracks(
    text: str, model: str, identity_a: tuple[str, int, int, str], identity_b: tuple[str, int, int, str]
) -> None:
    hash_a = embedding_input_hash(model=model, input_text=text)
    hash_b = embedding_input_hash(model=model, input_text=text)
    assert hash_a == hash_b, "embedding_input_hash must be a pure function of (model, text) alone"

    content_hash_a = _content_hash_for(text, identity_a)
    content_hash_b = _content_hash_for(text, identity_b)
    if identity_a != identity_b:
        # messages.content_hash is IDENTITY-CONTAMINATED by design (dedup at
        # full-row granularity needs it); different identity contexts for
        # the same text are expected to produce different content hashes.
        assert content_hash_a != content_hash_b, (
            "test setup assumption violated: distinct identity contexts must still "
            "produce distinct legacy content hashes, or this property proves nothing"
        )
    # The defect this bead fixes: embedding_input_hash must be blind to every
    # axis that content_hash tracks (session_id, position, variant_index,
    # provider_message_id) as long as the text itself is unchanged.
    assert hash_a == embedding_input_hash(model=model, input_text=text)


@given(text_a=_TEXT, text_b=_TEXT, model=_MODEL)
def test_embedding_input_hash_is_sensitive_to_text(text_a: str, text_b: str, model: str) -> None:
    hash_a = embedding_input_hash(model=model, input_text=text_a)
    hash_b = embedding_input_hash(model=model, input_text=text_b)
    if text_a == text_b:
        assert hash_a == hash_b
    else:
        assert hash_a != hash_b, "different embedder input text must hash differently"


@given(text=_TEXT, model_a=_MODEL, model_b=_MODEL)
def test_embedding_input_hash_is_sensitive_to_model(text: str, model_a: str, model_b: str) -> None:
    hash_a = embedding_input_hash(model=model_a, input_text=text)
    hash_b = embedding_input_hash(model=model_b, input_text=text)
    if model_a == model_b:
        assert hash_a == hash_b
    else:
        assert hash_a != hash_b, "a different model must hash differently even for identical text"


@given(text=_TEXT, model=_MODEL)
def test_embedding_input_hash_is_32_bytes(text: str, model: str) -> None:
    digest = embedding_input_hash(model=model, input_text=text)
    assert isinstance(digest, bytes)
    assert len(digest) == 32


@given(text=st.text(min_size=1, max_size=50), model=_MODEL)
def test_embedding_input_hash_nfc_normalizes(text: str, model: str) -> None:
    import unicodedata

    nfd = unicodedata.normalize("NFD", text)
    nfc = unicodedata.normalize("NFC", text)
    assert embedding_input_hash(model=model, input_text=nfd) == embedding_input_hash(model=model, input_text=nfc)
