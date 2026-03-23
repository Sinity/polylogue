"""Per-role scoring heuristics for semantic schema inference."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.semantic_inference_models import (
    ANTI_TITLE_NAME_TOKENS,
    KNOWN_ROLE_VALUES,
    SemanticCandidate,
)


def score_container(path: str, fs: FieldStats, all_stats: dict[str, FieldStats]) -> SemanticCandidate | None:
    """Score as message_container: repeated arrays/maps of objects."""
    if not fs.array_lengths and not fs.object_key_counts:
        return None

    evidence: dict[str, object] = {}
    score = 0.0
    if fs.array_lengths:
        avg_len = fs.avg_array_length
        if avg_len is not None and avg_len >= 2:
            score += min(0.3, avg_len / 30.0)
            evidence["avg_array_length"] = round(avg_len, 1)
    if fs.object_key_counts:
        avg_fanout = fs.avg_object_fanout
        if avg_fanout is not None and avg_fanout >= 3:
            score += min(0.3, avg_fanout / 30.0)
            evidence["avg_object_fanout"] = round(avg_fanout, 1)

    child_prefix = f"{path}[*]" if fs.array_lengths else f"{path}.*"
    child_paths = [item for item in all_stats if item.startswith(child_prefix + ".")]
    if len(child_paths) >= 3:
        score += 0.2
        evidence["child_field_count"] = len(child_paths)

    depth = path.count(".")
    if 0 <= depth <= 3:
        score += 0.1
    evidence["depth"] = depth

    if fs.frequency >= 0.9:
        score += 0.1
        evidence["frequency"] = round(fs.frequency, 3)

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="message_container", confidence=min(1.0, score), evidence=evidence)


def score_role(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_role: low-cardinality string with role-like values."""
    if not fs.observed_values:
        return None

    evidence: dict[str, object] = {}
    score = 0.0
    n_distinct = len(fs.observed_values)
    if n_distinct > 15:
        return None
    evidence["distinct_values"] = n_distinct
    if 2 <= n_distinct <= 6:
        score += 0.25
    elif n_distinct <= 10:
        score += 0.1

    observed_lower = {value.lower() for value in fs.observed_values}
    role_overlap = observed_lower & KNOWN_ROLE_VALUES
    if role_overlap:
        score += min(0.5, len(role_overlap) * 0.15)
        evidence["known_roles"] = sorted(role_overlap)

    if fs.frequency >= 0.95:
        score += 0.1
    evidence["frequency"] = round(fs.frequency, 3)

    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    role_name_signals = {"role", "sender", "author", "type", "from"}
    if terminal in role_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    if fs.newline_rate > 0.05:
        score *= 0.3

    length_stats = fs.string_length_stats
    if length_stats and length_stats["avg"] <= 20:
        score += 0.05
        evidence["avg_length"] = round(length_stats["avg"], 1)

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="message_role", confidence=min(1.0, score), evidence=evidence)


def score_body(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_body: long, multiline, medium-high entropy text."""
    length_stats = fs.string_length_stats
    if not length_stats:
        return None

    evidence: dict[str, object] = {}
    score = 0.0
    avg_len = length_stats["avg"]
    if avg_len >= 50:
        score += min(0.35, avg_len / 500.0)
        evidence["avg_length"] = round(avg_len, 1)
    elif avg_len < 10:
        return None

    nl_rate = fs.newline_rate
    if nl_rate >= 0.3:
        score += 0.2
    elif nl_rate >= 0.1:
        score += 0.1
    evidence["newline_rate"] = round(nl_rate, 3)

    entropy = fs.approximate_entropy
    if entropy is not None:
        if entropy >= 3.0:
            score += 0.15
        elif entropy >= 2.0:
            score += 0.05
        evidence["entropy"] = round(entropy, 2)

    n_distinct = len(fs.observed_values)
    if n_distinct >= 20:
        score += 0.1
        evidence["distinct_values"] = n_distinct

    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    body_name_signals = {
        "text", "content", "body", "message", "parts",
        "input", "output", "prompt", "response",
    }
    if terminal in body_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    if length_stats["stddev"] >= 50:
        score += 0.05
        evidence["stddev"] = round(length_stats["stddev"], 1)

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="message_body", confidence=min(1.0, score), evidence=evidence)


def score_timestamp(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_timestamp: epoch/ISO format, monotonic within arrays."""
    evidence: dict[str, object] = {}
    score = 0.0

    fmt = fs.dominant_format
    is_timestamp_fmt = fmt in {"unix-epoch", "unix-epoch-str", "iso8601"}
    if is_timestamp_fmt:
        score += 0.35
        evidence["format"] = fmt
    elif fs.detected_formats:
        ts_formats = {"unix-epoch", "unix-epoch-str", "iso8601"}
        ts_count = sum(fs.detected_formats.get(item, 0) for item in ts_formats)
        if ts_count > 0 and fs.value_count > 0:
            ts_ratio = ts_count / fs.value_count
            if ts_ratio >= 0.5:
                score += 0.2
                evidence["timestamp_ratio"] = round(ts_ratio, 3)
    elif fs.num_min is not None and fs.num_max is not None:
        if fs.num_min >= 946684800.0 and fs.num_max <= 2208988800.0:
            score += 0.2
            evidence["range"] = [fs.num_min, fs.num_max]

    if score == 0:
        return None

    mono = fs.monotonicity_score
    if mono is not None and mono >= 0.8:
        score += 0.2
        evidence["monotonicity"] = round(mono, 3)

    if fs.frequency >= 0.9:
        score += 0.1
    evidence["frequency"] = round(fs.frequency, 3)

    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    ts_name_signals = {
        "timestamp", "created_at", "create_time", "updated_at",
        "update_time", "time", "date", "datetime", "ts",
        "created", "modified",
    }
    if terminal in ts_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    if fs.is_multiline > 0:
        score *= 0.5
    if fs.is_enum_like and len(fs.observed_values) < 5:
        score *= 0.3

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="message_timestamp", confidence=min(1.0, score), evidence=evidence)


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

    nl_rate = fs.newline_rate
    if nl_rate <= 0.05:
        score += 0.15
    elif nl_rate > 0.3:
        score *= 0.3
    evidence["newline_rate"] = round(nl_rate, 3)

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

    if fs.observed_values:
        slash_count = sum(1 for value in fs.observed_values if isinstance(value, str) and "/" in value)
        slash_ratio = slash_count / len(fs.observed_values)
        if slash_ratio > 0.3:
            score *= 0.3
            evidence["slash_ratio"] = round(slash_ratio, 3)

    if score < 0.15:
        return None
    return SemanticCandidate(path=path, role="conversation_title", confidence=min(1.0, score), evidence=evidence)


__all__ = ["score_body", "score_container", "score_role", "score_timestamp", "score_title"]
