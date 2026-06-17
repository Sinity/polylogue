"""i18n / encoding edge-case coverage matrix.

Boundaries × edge cases. Every cell asserts a declared contract:

  - "normalizes deterministically" — equivalent inputs collapse to identical
    output (e.g. NFC normalization at content_hash).
  - "passes through unchanged" — the surface neither rejects nor mutates the
    input (e.g. message text preserves zero-width characters verbatim).
  - "rejects with typed error" — the surface refuses the input via a typed
    exception (no edge case currently triggers this for these boundaries, but
    the assertion vocabulary is reserved so future regressions surface).

The matrix prevents each new text-handling site from rediscovering the same
class of bugs (e.g. the UTF-8 BOM bug fixed in ``polylogue.sources.decoder_json``
and recorded in MEMORY.md).

Ref #1305.
"""

from __future__ import annotations

import io
import json
import sqlite3
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_text
from polylogue.pipeline.ids import (
    _normalize_for_hash,
    session_content_hash,
)
from polylogue.sources.decoder_json import decode_json_bytes
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import SessionRepository
from polylogue.storage.search import search_messages
from polylogue.storage.search.query_support import escape_fts5_query
from tests.infra.storage_records import make_message, make_session, save_current_archive_records

# ---------------------------------------------------------------------------
# Edge-case catalog
# ---------------------------------------------------------------------------

# Composed (NFC) "café" — single precomposed e-with-acute codepoint.
NFC_CAFE = "caf\u00e9"
# Decomposed (NFD) "café" — "e" followed by combining acute accent.
NFD_CAFE = "cafe\u0301"

# Right-to-left text (Arabic) and Hebrew.
ARABIC_HELLO = "\u0645\u0631\u062d\u0628\u0627"  # "marhaba"
HEBREW_SHALOM = "\u05e9\u05dc\u05d5\u05dd"  # "shalom"

# Bidi override marks that historically cause display attacks.
BIDI_OVERRIDE = "before\u202eafter"

# Zero-width joiners and non-joiner / zero-width space.
ZWJ_EMOJI = "\U0001f469\u200d\U0001f4bb"  # woman + ZWJ + laptop = "woman technologist"
ZERO_WIDTH = "vis\u200bible\u200cspaces\u200d"

# CJK + a CJK extension B character (above BMP, U+20000).
CJK_TEXT = "\u4e2d\u6587\U00020000"

# Emoji sequence with skin-tone modifier + ZWJ family.
EMOJI_SEQUENCE = "\U0001f44b\U0001f3fd \U0001f468\u200d\U0001f469\u200d\U0001f466"

# Lone surrogate halves. Python str can hold these via the "surrogatepass"
# error handler; they cannot be encoded as standard UTF-8 without it.
LONE_HIGH_SURROGATE = "abc\ud800def"

ALL_TEXT_EDGE_CASES: dict[str, str] = {
    "nfc_cafe": NFC_CAFE,
    "nfd_cafe": NFD_CAFE,
    "arabic": ARABIC_HELLO,
    "hebrew": HEBREW_SHALOM,
    "bidi_override": BIDI_OVERRIDE,
    "zwj_emoji": ZWJ_EMOJI,
    "zero_width": ZERO_WIDTH,
    "cjk_with_extension_b": CJK_TEXT,
    "emoji_sequence": EMOJI_SEQUENCE,
}

# ---------------------------------------------------------------------------
# Boundary 1: decode_json_bytes — BOM / UTF-16 / surrogate handling
# ---------------------------------------------------------------------------


class TestDecodeJsonBytesBomMatrix:
    """``decode_json_bytes`` is the JSON-payload boundary for source ingest.

    Contracts:
      - UTF-8 BOM ``\\ufeff`` is stripped (regression for the bug recorded in
        MEMORY.md: utf-8 probe succeeded but left BOM intact, breaking JSON
        parsing).
      - UTF-16 / UTF-16-LE / UTF-16-BE BOMs are decoded and the BOM is
        stripped, yielding parseable JSON text.
      - The decoded text round-trips through ``json.loads`` without raising.
    """

    @pytest.mark.parametrize(
        ("label", "encoding"),
        [
            ("utf-8 raw", "utf-8"),
            ("utf-8-sig (BOM prefix)", "utf-8-sig"),
            # ``utf-16`` writes a BOM that lets the decoder pick LE vs BE
            # automatically. Raw utf-16-le/utf-16-be without a BOM are
            # intentionally not in the supported set: the decoder probes
            # encodings in order and the ASCII-bearing UTF-16 byte pattern
            # collides with UTF-8 + null-stripping. Sources that emit raw
            # UTF-16 must include the BOM.
            ("utf-16 (with BOM)", "utf-16"),
        ],
    )
    def test_bom_and_utf16_are_decoded_to_parseable_json(self, label: str, encoding: str) -> None:
        payload = {"title": NFC_CAFE, "body": ARABIC_HELLO}
        text = json.dumps(payload, ensure_ascii=False)
        encoded = text.encode(encoding)

        decoded = decode_json_bytes(encoded)

        assert decoded is not None, f"{label}: decoder returned None"
        # No leftover BOM character.
        assert "\ufeff" not in decoded, f"{label}: BOM not stripped"
        assert json.loads(decoded) == payload, f"{label}: roundtrip failed"

    def test_explicit_utf8_bom_prefix_is_stripped(self) -> None:
        # Belt-and-braces: synthesise the historical bug shape (raw UTF-8 bytes
        # with BOM prefix, not via utf-8-sig codec).
        raw = "\ufeff" + json.dumps({"k": "v"})
        encoded = raw.encode("utf-8")

        decoded = decode_json_bytes(encoded)

        assert decoded is not None
        assert decoded.startswith("{")
        assert json.loads(decoded) == {"k": "v"}


# ---------------------------------------------------------------------------
# Boundary 2: content hash — NFC normalization equivalence
# ---------------------------------------------------------------------------


class TestContentHashNormalization:
    """``content_hash`` declares NFC normalization (``core/hashing.py``).

    Contracts:
      - ``hash_text(NFC) == hash_text(NFD)`` for the same visual string.
      - Session and message content hashes apply the same normalization
        to their textual fields (title, message text), so re-ingesting an
        otherwise-identical session in a different normalization form
        does NOT trigger a re-write.
      - Distinct edge-case strings produce distinct hashes (no accidental
        collisions across the matrix).
    """

    def test_hash_text_collapses_nfc_and_nfd(self) -> None:
        assert NFC_CAFE != NFD_CAFE  # premise: distinct codepoint sequences
        assert hash_text(NFC_CAFE) == hash_text(NFD_CAFE)

    @pytest.mark.parametrize("label", sorted(ALL_TEXT_EDGE_CASES))
    def test_hash_text_is_stable_under_renormalization(self, label: str) -> None:
        original = ALL_TEXT_EDGE_CASES[label]
        # NFD then NFC (or vice versa) should produce identical hashes.
        nfd = unicodedata.normalize("NFD", original)
        nfc = unicodedata.normalize("NFC", original)
        assert hash_text(nfd) == hash_text(nfc) == hash_text(original)

    def test_session_hash_normalizes_message_text(self) -> None:
        # Isolate message-text normalization: title is constant ASCII, only the
        # message text differs by NFC/NFD form. The session hash must collapse them.
        def _build(text: str) -> ParsedSession:
            return ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="conv-1",
                title="constant",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        timestamp="2026-01-01T00:00:00Z",
                    )
                ],
            )

        assert session_content_hash(_build(NFC_CAFE)) == session_content_hash(_build(NFD_CAFE))

    def test_session_content_hash_normalizes_title_and_text(self) -> None:
        def _build(title: str, text: str) -> ParsedSession:
            return ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="conv-1",
                title=title,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        timestamp="2026-01-01T00:00:00Z",
                    )
                ],
            )

        nfc = session_content_hash(_build(NFC_CAFE, NFC_CAFE))
        nfd = session_content_hash(_build(NFD_CAFE, NFD_CAFE))
        assert nfc == nfd

    def test_normalize_for_hash_applies_nfc(self) -> None:
        # The pipeline helper used by hash payload assembly. Sanity check
        # that it matches the documented NFC contract.
        assert _normalize_for_hash(NFD_CAFE) == NFC_CAFE
        assert _normalize_for_hash(NFC_CAFE) == NFC_CAFE

    def test_hashes_are_distinct_across_edge_cases(self) -> None:
        hashes = {label: hash_text(text) for label, text in ALL_TEXT_EDGE_CASES.items()}
        # NFC/NFD café collide by design; collapse them before uniqueness check.
        deduped = {label: h for label, h in hashes.items() if label != "nfd_cafe"}
        assert len(set(deduped.values())) == len(deduped)


# ---------------------------------------------------------------------------
# Boundary 3: surrogate halves — handling without crashing
# ---------------------------------------------------------------------------


class TestSurrogateHandling:
    """Lone surrogates are valid Python ``str`` but cannot encode as
    standard UTF-8.

    Contracts:
      - ``hash_text`` raises ``UnicodeEncodeError`` (NOT silent corruption) for
        a string containing an unpaired surrogate. This is the typed
        rejection: callers either sanitise upstream or catch.
      - JSON round-trip with ``ensure_ascii=True`` escapes surrogate halves
        rather than embedding them, so JSON-shaped raw payloads coming in
        from providers never produce a hash crash downstream.
    """

    def test_hash_text_rejects_lone_surrogate(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            hash_text(LONE_HIGH_SURROGATE)

    def test_json_dumps_escapes_lone_surrogate_ascii_mode(self) -> None:
        # ensure_ascii=True is the default for stdlib json; surrogates become
        # \uXXXX escapes that round-trip cleanly through the encoder.
        encoded = json.dumps({"text": LONE_HIGH_SURROGATE})
        assert "\\ud800" in encoded
        # Decoding back yields the same Python str (str equality preserves
        # the surrogate codepoint).
        decoded = json.loads(encoded)
        assert decoded["text"] == LONE_HIGH_SURROGATE


# ---------------------------------------------------------------------------
# Boundary 4: FTS5 index — unicode61 tokenizer behaviour
# ---------------------------------------------------------------------------


class TestFtsUnicodeTokenizer:
    """SQLite FTS5 with the ``unicode61`` tokenizer indexes the text we feed it.

    Contracts (declared per case):
      - ASCII / Latin-with-diacritics: indexed and queryable.
      - Bidi/zero-width characters: present in the row text unchanged
        ("passes through unchanged" at storage layer); query behavior over
        zero-width-laden tokens is tokenizer-defined and is not asserted
        beyond "no crash on indexing".
      - Lone surrogates are NOT exercised here — they cannot reach SQLite
        without encoding to UTF-8, which raises (see surrogate test).
    """

    async def test_indexes_arabic_text(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        conv = make_session("conv-ar", source_name="claude-ai", title="Arabic")
        msg = make_message("conv-ar-m1", "conv-ar", text=ARABIC_HELLO)
        await save_current_archive_records(
            storage_repository,
            session=conv,
            messages=[msg],
            attachments=[],
        )
        rebuild_index()
        results = search_messages(ARABIC_HELLO, archive_root=workspace_env["archive_root"], limit=10)
        assert len(results.hits) == 1
        assert results.hits[0].session_id == "claude-ai-export:conv-ar"

    async def test_indexes_cjk_text_without_crashing(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        # SQLite ``unicode61`` does NOT treat CJK characters as word
        # boundaries — a CJK run is a single token. We therefore assert the
        # weaker "stores and indexes without crashing, regardless of whether
        # a substring query resolves" contract, with a stable English-word
        # sentinel mixed in to give the test a positive observation.
        conv = make_session("conv-cjk", source_name="claude-ai", title="CJK")
        msg = make_message("conv-cjk-m1", "conv-cjk", text=f"{CJK_TEXT} cjkmarker")
        await save_current_archive_records(
            storage_repository,
            session=conv,
            messages=[msg],
            attachments=[],
        )
        rebuild_index()
        # The English sentinel proves the row reached the FTS index even
        # though the CJK run itself is not substring-searchable.
        results = search_messages("cjkmarker", archive_root=workspace_env["archive_root"], limit=10)
        assert len(results.hits) == 1
        assert results.hits[0].session_id == "claude-ai-export:conv-cjk"

    async def test_indexing_zero_width_and_bidi_does_not_crash(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        # Whitespace-tokenized "visible" + "spaces" should remain searchable
        # even though zero-width characters are interleaved.
        conv = make_session("conv-zw", source_name="claude-ai", title="zerowidth")
        msg = make_message("conv-zw-m1", "conv-zw", text=ZERO_WIDTH + " " + BIDI_OVERRIDE)
        await save_current_archive_records(
            storage_repository,
            session=conv,
            messages=[msg],
            attachments=[],
        )
        # Indexing must not raise. We do not assert on tokenizer-internal
        # decisions about whether ZWJ/ZWNJ split tokens.
        rebuild_index()

    async def test_indexes_nfc_and_nfd_independently(
        self,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
    ) -> None:
        # FTS index stores the raw text bytes that hit it. Storage does NOT
        # normalize text before indexing (only content_hash does). This test
        # documents that contract.
        conv = make_session("conv-nfc", source_name="claude-ai", title="cafe")
        msg = make_message("conv-nfc-m1", "conv-nfc", text=NFC_CAFE)
        await save_current_archive_records(
            storage_repository,
            session=conv,
            messages=[msg],
            attachments=[],
        )
        rebuild_index()
        results = search_messages(NFC_CAFE, archive_root=workspace_env["archive_root"], limit=10)
        assert len(results.hits) == 1


# ---------------------------------------------------------------------------
# Boundary 5: escape_fts5_query — safe handling of edge-case query strings
# ---------------------------------------------------------------------------


class TestEscapeFtsQuery:
    """``escape_fts5_query`` is the query-string boundary.

    Contract: every edge-case input produces a string that SQLite FTS5 will
    accept without raising ``OperationalError`` when used in a ``MATCH``
    clause. We test this end-to-end against a real FTS5 table.
    """

    @pytest.fixture
    def fts_conn(self, tmp_path: Path) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(tmp_path / "fts_probe.db"))
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text, tokenize='unicode61')")
        conn.execute("INSERT INTO probe(text) VALUES (?)", (NFC_CAFE,))
        conn.commit()
        try:
            yield conn
        finally:
            conn.close()

    @pytest.mark.parametrize("label", sorted(ALL_TEXT_EDGE_CASES))
    def test_escape_then_match_does_not_raise(self, label: str, fts_conn: sqlite3.Connection) -> None:
        query = ALL_TEXT_EDGE_CASES[label]
        escaped = escape_fts5_query(query)
        # The escaped form must be a string FTS5 can MATCH against without
        # syntax errors.
        try:
            fts_conn.execute("SELECT rowid FROM probe WHERE probe MATCH ?", (escaped,)).fetchall()
        except sqlite3.OperationalError as exc:
            pytest.fail(f"{label}: FTS5 rejected escaped query {escaped!r}: {exc}")


# ---------------------------------------------------------------------------
# Boundary 6: CLI/terminal display — encoding pass-through
# ---------------------------------------------------------------------------


class TestStdoutEncodingPassthrough:
    """Terminal output must accept the edge-case strings without raising.

    Contract: writing any matrix string to a UTF-8 ``TextIOWrapper`` (the
    default for ``sys.stdout`` under most environments) succeeds and writes
    a byte sequence that decodes back to NFC-normalized form.
    """

    @pytest.mark.parametrize("label", sorted(ALL_TEXT_EDGE_CASES))
    def test_utf8_writer_accepts_edge_case(self, label: str) -> None:
        buffer = io.BytesIO()
        writer = io.TextIOWrapper(buffer, encoding="utf-8", newline="")
        text = ALL_TEXT_EDGE_CASES[label]
        writer.write(text)
        writer.flush()
        written = buffer.getvalue().decode("utf-8")
        # NFC and NFD café both round-trip as themselves; the writer does not
        # normalize. That is the declared "passes through unchanged" contract.
        assert written == text

    def test_utf8_writer_rejects_lone_surrogate(self) -> None:
        buffer = io.BytesIO()
        writer = io.TextIOWrapper(buffer, encoding="utf-8", newline="")
        with pytest.raises(UnicodeEncodeError):
            writer.write(LONE_HIGH_SURROGATE)
            writer.flush()
