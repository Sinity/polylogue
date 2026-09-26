"""Session and content format adapters.

These renderers serialize archive data into output formats. The HTML adapter
uses ``polylogue.ui.theme`` for shared palette and syntax-theme tokens;
terminal interaction and Rich layout remain owned by ``polylogue.ui``.
"""

from __future__ import annotations
