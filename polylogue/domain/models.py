from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class Attachment:
    """Represents a file attachment associated with a message."""

    attachment_id: str
    name: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    source_url: Optional[str] = None
    local_path: Optional[Path] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    """Normalized representation of a chat message."""

    message_id: str
    role: str
    text: str
    position: Optional[int] = None
    timestamp: Optional[str] = None
    model: Optional[str] = None
    attachments: List[Attachment] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def has_attachments(self) -> bool:
        return bool(self.attachments)


@dataclass(slots=True)
class Branch:
    """A branch within a conversation (canonical or divergent)."""

    branch_id: str
    messages: List[Message]
    is_canonical: bool = False
    divergence_index: Optional[int] = None
    divergence_role: Optional[str] = None
    divergence_snippet: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def iter_messages(self) -> Iterable[Message]:
        return iter(self.messages)


@dataclass(slots=True)
class Conversation:
    """Provider-independent conversation abstraction."""

    provider: str
    conversation_id: str
    slug: str
    branches: List[Branch]
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)

    def canonical_branch(self) -> Optional[Branch]:
        for branch in self.branches:
            if branch.is_canonical:
                return branch
        return self.branches[0] if self.branches else None

    def branch_by_id(self, branch_id: str) -> Optional[Branch]:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        return None

    def total_messages(self) -> int:
        return sum(len(branch.messages) for branch in self.branches)
