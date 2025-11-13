from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol


class ProviderSession(Protocol):
    """Lightweight protocol describing a provider SDK session."""

    name: str
    title: str


@dataclass
class ProviderDescriptor:
    name: str
    session: ProviderSession


class ProviderRegistry:
    """Registry tracking instantiated provider sessions."""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderDescriptor] = {}

    def register(self, session: ProviderSession) -> None:
        self._providers[session.name] = ProviderDescriptor(name=session.name, session=session)

    def get(self, name: str) -> Optional[ProviderSession]:
        descriptor = self._providers.get(name)
        return descriptor.session if descriptor else None

    def names(self) -> Dict[str, ProviderDescriptor]:
        return dict(self._providers)
