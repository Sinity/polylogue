"""Conversation-scoped scoring heuristics for semantic schema inference."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.semantic_inference_models import (
    ANTI_TITLE_NAME_TOKENS,
    SemanticCandidate,
)


def score_title(path: str, fs: FieldStats, all_stats: dict[str, FieldStats]) -> SemanticCandidate | None:
    """Score as conversation_title: short, high-cardinality, non-multiline."""
    del all_stats
    if fs.dominant_format in {"uuid4", "uuid", "hex-id", "url", "email", "base64"}:
        return None

    length_stats = fs.string_length_stats
    if not length_stats:
        return None

    evidence: dict[str, object] = {}
    score = 0.0
    avg_len = length_stats["avg"]
    if 3 <= avg_len <= 100:
        score += 0.2
        evidence["avg_length"] = round(avg_len, 1)
    elif avg_len > 200:
        return None

    n_distinct = len(fs.observed_values)
    if n_distinct >= 5:
        score += 0.15
        evidence["distinct_values"] = n_distinct

    newline_rate = fs.newline_rate
    if newline_rate <= 0.05:
        score += 0.15
    elif newline_rate > 0.3:
        score *= 0.3
    evidence["newline_rate"] = round(newline_rate, 3)

    depth = path.count(".")
    if depth <= 2:
        score += 0.1
    elif depth > 4:
        score *= 0.5
    evidence["depth"] = depth

    if "[*]" in path:
        score *= 0.3

    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    title_name_signals = {"title", "name", "subject", "topic", "heading"}
    if terminal in title_name_signals:
        score += 0.2
        evidence["name_signal"] = terminal

    terminal_lower = terminal.replace("_", " ").replace("-", " ")
    for anti_token in ANTI_TITLE_NAME_TOKENS:
        if anti_token in terminal_lower:
            score *= 0.1
            evidence["anti_name_signal"] = anti_token
            break
    if terminal.endswith(("id", "Id", "ID", "_id", "-id")):
        score *= 0.1
        evidence["anti_name_signal"] = "id_suffix"

    if fs.newline_rate > 0.05:
        score *= 0.4
    if fs.approximate_entropy is not None and fs.approximate_entropy < 1.0:
        score *= 0.6
        evidence["low_entropy"] = round(fs.approximate_entropy, 2)

    if fs.observed_values:
        slash_count = sum(1 for value in fs.observed_values if isinstance(value, str) and "/" in value)
        slash_ratio = slash_count / len(fs.observed_values)
        if slash_ratio > 0.3:
            score *= 0.3
            evidence["slash_ratio"] = round(slash_ratio, 3)

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="conversation_title", confidence=min(1.0, score), evidence=evidence)


__all__ = ["score_title"]
