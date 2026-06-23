"""Browser-capture identity normalization helpers."""

from __future__ import annotations

import re

from polylogue.core.enums import Provider

_UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def legacy_browser_capture_native_id(provider: Provider | str | None, provider_session_id: str | None) -> str | None:
    """Recover provider-native ids from old browser-extension synthetic ids."""
    if not provider_session_id:
        return None
    provider_value = provider.value if isinstance(provider, Provider) else provider
    if not provider_value:
        return provider_session_id
    synthetic_prefix = f"{provider_value}:"
    if provider_session_id.startswith(synthetic_prefix):
        parts = provider_session_id.split(":")
        if len(parts) == 3 and parts[1] and "/" not in parts[1]:
            return parts[1]
        if provider_value == Provider.CHATGPT.value and len(parts) == 4 and parts[1] == "WEB" and parts[2]:
            return f"WEB:{parts[2]}"
    hyphen_prefix = f"{provider_value}-"
    if provider_session_id.startswith(hyphen_prefix):
        match = _UUID_PATTERN.search(provider_session_id)
        if match:
            if provider_value == Provider.CHATGPT.value and provider_session_id.startswith("chatgpt-WEB-"):
                return f"WEB:{match.group(0)}"
            return match.group(0)
    return provider_session_id
