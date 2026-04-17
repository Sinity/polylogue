"""Batch orchestration helpers for synthetic corpora."""

from __future__ import annotations

import random
from typing import Any

from polylogue.schemas.synthetic.models import (
    SyntheticArtifact,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES, ConversationTheme


def generate_batch(
    self: Any,
    count: int = 5,
    messages_per_conversation: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
) -> SyntheticGenerationBatch:
    if style not in {"default", "showcase"}:
        raise ValueError(f"Unknown synthetic style: {style}")

    rng = random.Random(seed)
    artifacts: list[SyntheticArtifact] = []
    for _ in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        n_messages = rng.choice(messages_per_conversation)
        theme = rng.choice(_SHOWCASE_THEMES) if style == "showcase" else None
        data = self._generate_conversation(n_messages, rng, theme=theme)
        artifacts.append(
            SyntheticArtifact(
                raw_bytes=self._serialize(data),
                message_count=n_messages,
                style=style,
            )
        )

    report = SyntheticGenerationReport(
        provider=self.provider,
        package_version=self.package_version,
        element_kind=self.element_kind,
        wire_encoding=self.wire_format.encoding,
        requested_count=count,
        generated_count=len(artifacts),
        style=style,
        seed=seed,
    )
    return SyntheticGenerationBatch(artifacts=artifacts, report=report)


def _generate_conversation(
    self: Any,
    n_messages: int,
    rng: random.Random,
    *,
    theme: ConversationTheme | None = None,
) -> Any:
    wf = self.wire_format
    if wf.encoding == "jsonl":
        return self._generate_jsonl_records(n_messages, rng, theme=theme)
    if wf.tree and wf.tree.container_path:
        return self._generate_tree_json(n_messages, rng, theme=theme)
    if wf.messages_path:
        return self._generate_linear_json(n_messages, rng, theme=theme)
    return self._generate_from_schema(self.schema, rng)


def _role_cycle(self: Any) -> list[str]:
    match self.provider:
        case "chatgpt":
            return ["user", "assistant"]
        case "claude-ai":
            return ["human", "assistant"]
        case "claude-code":
            return ["user", "assistant"]
        case "codex":
            return ["user", "assistant"]
        case "gemini":
            return ["user", "model"]
        case _:
            return ["user", "assistant"]


__all__ = [
    "_generate_conversation",
    "_role_cycle",
    "generate_batch",
]
