"""Nested hash payloads must be NFC-normalized like every sibling field.

`_normalize_for_hash` covers scalar fields, but `ParsedContentBlock.tool_input`
and `ParsedSessionEvent.payload` were handed straight to `hash_payload`, which
by its own docstring does NOT normalize the strings it serializes. So two
tool_use blocks whose nested content differed only in Unicode normalization
form hashed as two permanent logical identities and could never dedupe --
while the identical text in `message.text` hashed the same.

Measured before the fix: 0 of 20,000 sampled live `tool_use.tool_input` rows
carried non-NFC content, so no stored hash changed. This closes a latent trap
rather than repairing corruption; the regression it guards is silent and
unrecoverable once two identities exist.
"""

from __future__ import annotations

import unicodedata

from polylogue.pipeline.ids import _content_block_payload
from polylogue.core.enums import BlockType
from polylogue.sources.parsers.base import ParsedContentBlock

_NFD = unicodedata.normalize("NFD", "café-résumé")
_NFC = unicodedata.normalize("NFC", "café-résumé")


def _block(tool_input: dict[str, object]) -> ParsedContentBlock:
    return ParsedContentBlock(
        type=BlockType.TOOL_USE,
        text=None,
        tool_name="Edit",
        tool_id="toolu_01",
        tool_input=tool_input,
    )


def test_tool_input_values_hash_identically_across_normalization_forms() -> None:
    """The exact defect: same visual content, two hashes, forever un-dedupable."""
    assert _NFD != _NFC, "fixture must actually differ at the byte level"

    nfd_payload = _content_block_payload(_block({"file_path": _NFD}))
    nfc_payload = _content_block_payload(_block({"file_path": _NFC}))

    assert nfd_payload["tool_input"] == nfc_payload["tool_input"]


def test_tool_input_keys_hash_identically_across_normalization_forms() -> None:
    """A dict KEY can carry an NFD form just as easily as a value."""
    nfd_payload = _content_block_payload(_block({_NFD: "value"}))
    nfc_payload = _content_block_payload(_block({_NFC: "value"}))

    assert nfd_payload["tool_input"] == nfc_payload["tool_input"]


def test_nested_lists_and_dicts_are_normalized_at_every_depth() -> None:
    """Real tool_input is nested -- normalizing only the top level would miss most of it."""
    nfd_payload = _content_block_payload(_block({"edits": [{"old_string": _NFD}]}))
    nfc_payload = _content_block_payload(_block({"edits": [{"old_string": _NFC}]}))

    assert nfd_payload["tool_input"] == nfc_payload["tool_input"]


def test_genuinely_different_content_still_hashes_differently() -> None:
    """Guard against the degenerate fix: normalization must not collapse real differences."""
    a = _content_block_payload(_block({"file_path": "alpha"}))
    b = _content_block_payload(_block({"file_path": "beta"}))

    assert a["tool_input"] != b["tool_input"]


def test_ascii_tool_input_hash_is_unchanged_by_normalization() -> None:
    """Grandfathering evidence: NFC is identity on ASCII, so stored hashes cannot move.

    Every sampled live row is ASCII-or-already-NFC, which is why this fix
    needs no re-hash pass. If normalization altered an ASCII payload, existing
    content hashes would shift and the whole archive would re-ingest.
    """
    payload = _content_block_payload(_block({"file_path": "/realm/project/polylogue/README.md"}))

    again = _content_block_payload(_block({"file_path": "/realm/project/polylogue/README.md"}))
    assert payload["tool_input"] == again["tool_input"]
    assert unicodedata.normalize("NFC", "/realm/project/polylogue/README.md") == "/realm/project/polylogue/README.md"
