"""Unicode normalization on the content-identity path.

Content identity is computed over NFC text, so every hash, digest and
canonical encoding in the archive normalizes before it encodes. :func:`nfc`
is that one call site.
"""

from __future__ import annotations

import unicodedata


def nfc(text: str) -> str:
    """Return *text* in NFC form.

    ASCII is NFC by construction, so an ASCII string is returned as itself.
    ``str.isascii`` reads the interpreter's compact-ASCII flag, which is why
    this is cheaper than entering ``unicodedata`` to reach the same answer.
    """
    return text if text.isascii() else unicodedata.normalize("NFC", text)


__all__ = ["nfc"]
