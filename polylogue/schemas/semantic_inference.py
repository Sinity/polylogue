"""Semantic candidate scoring for schema fields.

Assigns semantic roles to JSON paths based on statistical evidence from
FieldStats.  Each role has heuristic scoring criteria that produce a
confidence value (0-1) and an evidence dict for debuggability.

Semantic roles:
    - message_container: arrays/maps holding the conversation messages
    - message_role: low-cardinality field indicating speaker identity
    - message_body: high-length, multiline text content field
    - message_timestamp: monotonic epoch/ISO values within message order
    - conversation_title: short, high-cardinality string outside the container
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polylogue.schemas.field_stats import FieldStats

# Known role value tokens across providers
_KNOWN_ROLE_VALUES = frozenset({
    "user", "assistant", "system", "model", "tool", "human",
    "developer", "function", "admin",
})


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """A scored candidate for a semantic role assignment."""

    path: str
    role: str  # one of the SEMANTIC_ROLES
    confidence: float  # 0.0 - 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


SEMANTIC_ROLES = (
    "message_container",
    "message_role",
    "message_body",
    "message_timestamp",
    "conversation_title",
)


def infer_semantic_roles(
    stats: dict[str, FieldStats],
) -> list[SemanticCandidate]:
    """Score all field paths for all semantic roles.

    Returns candidates sorted by confidence (highest first), with at most
    one candidate per (path, role) pair.
    """
    candidates: list[SemanticCandidate] = []

    for path, fs in stats.items():
        for role in SEMANTIC_ROLES:
            candidate = _score(path, fs, role, stats)
            if candidate is not None and candidate.confidence > 0.1:
                candidates.append(candidate)

    candidates.sort(key=lambda c: -c.confidence)
    return candidates


def select_best_roles(
    candidates: list[SemanticCandidate],
) -> dict[str, SemanticCandidate]:
    """Select the single best candidate for each semantic role.

    Returns a mapping of role → best candidate.  Ties broken by confidence.
    """
    best: dict[str, SemanticCandidate] = {}
    for c in candidates:
        if c.role not in best or c.confidence > best[c.role].confidence:
            best[c.role] = c
    return best


# ---------------------------------------------------------------------------
# Per-role scoring
# ---------------------------------------------------------------------------

def _score(
    path: str,
    fs: FieldStats,
    role: str,
    all_stats: dict[str, FieldStats],
) -> SemanticCandidate | None:
    match role:
        case "message_container":
            return _score_container(path, fs, all_stats)
        case "message_role":
            return _score_role(path, fs)
        case "message_body":
            return _score_body(path, fs)
        case "message_timestamp":
            return _score_timestamp(path, fs)
        case "conversation_title":
            return _score_title(path, fs, all_stats)
    return None


def _score_container(
    path: str,
    fs: FieldStats,
    all_stats: dict[str, FieldStats],
) -> SemanticCandidate | None:
    """Score as message_container: repeated arrays/maps of objects."""
    # Must be an array field (has array_lengths) or a dict with dynamic keys
    if not fs.array_lengths and not fs.object_key_counts:
        return None

    evidence: dict[str, Any] = {}
    score = 0.0

    # Array with multiple elements
    if fs.array_lengths:
        avg_len = fs.avg_array_length
        if avg_len is not None and avg_len >= 2:
            score += min(0.3, avg_len / 30.0)
            evidence["avg_array_length"] = round(avg_len, 1)

    # Dynamic-key container (like ChatGPT mapping)
    if fs.object_key_counts:
        avg_fanout = fs.avg_object_fanout
        if avg_fanout is not None and avg_fanout >= 3:
            score += min(0.3, avg_fanout / 30.0)
            evidence["avg_object_fanout"] = round(avg_fanout, 1)

    # Children have diverse subfields (structural similarity)
    child_prefix = f"{path}[*]" if fs.array_lengths else f"{path}.*"
    child_paths = [p for p in all_stats if p.startswith(child_prefix + ".")]
    if len(child_paths) >= 3:
        score += 0.2
        evidence["child_field_count"] = len(child_paths)

    # Depth: containers are typically at depth 1-3
    depth = path.count(".")
    if 0 <= depth <= 3:
        score += 0.1
    evidence["depth"] = depth

    # High frequency (container should be present in most samples)
    if fs.frequency >= 0.9:
        score += 0.1
        evidence["frequency"] = round(fs.frequency, 3)

    if score < 0.15:
        return None

    return SemanticCandidate(
        path=path,
        role="message_container",
        confidence=min(1.0, score),
        evidence=evidence,
    )


def _score_role(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_role: low-cardinality string with role-like values."""
    if not fs.observed_values:
        return None

    evidence: dict[str, Any] = {}
    score = 0.0

    # Must be low cardinality
    n_distinct = len(fs.observed_values)
    if n_distinct > 15:
        return None
    evidence["distinct_values"] = n_distinct

    # Bonus for very low cardinality (2-6 values typical for roles)
    if 2 <= n_distinct <= 6:
        score += 0.25
    elif n_distinct <= 10:
        score += 0.1

    # Check for known role value overlap
    observed_lower = {v.lower() for v in fs.observed_values}
    role_overlap = observed_lower & _KNOWN_ROLE_VALUES
    if role_overlap:
        score += min(0.5, len(role_overlap) * 0.15)
        evidence["known_roles"] = sorted(role_overlap)

    # High frequency (roles are typically always present)
    if fs.frequency >= 0.95:
        score += 0.1
    evidence["frequency"] = round(fs.frequency, 3)

    # Field name heuristic
    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    role_name_signals = {"role", "sender", "author", "type", "from"}
    if terminal in role_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    # Should NOT be multiline
    if fs.newline_rate > 0.05:
        score *= 0.3

    # Should be short strings
    length_stats = fs.string_length_stats
    if length_stats and length_stats["avg"] <= 20:
        score += 0.05
        evidence["avg_length"] = round(length_stats["avg"], 1)

    if score < 0.15:
        return None

    return SemanticCandidate(
        path=path,
        role="message_role",
        confidence=min(1.0, score),
        evidence=evidence,
    )


def _score_body(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_body: long, multiline, medium-high entropy text."""
    length_stats = fs.string_length_stats
    if not length_stats:
        return None

    evidence: dict[str, Any] = {}
    score = 0.0

    # High average length
    avg_len = length_stats["avg"]
    if avg_len >= 50:
        score += min(0.35, avg_len / 500.0)
        evidence["avg_length"] = round(avg_len, 1)
    elif avg_len < 10:
        return None  # too short to be message body

    # Multiline incidence
    nl_rate = fs.newline_rate
    if nl_rate >= 0.3:
        score += 0.2
    elif nl_rate >= 0.1:
        score += 0.1
    evidence["newline_rate"] = round(nl_rate, 3)

    # Medium-high entropy (not repetitive like role fields)
    entropy = fs.approximate_entropy
    if entropy is not None:
        if entropy >= 3.0:
            score += 0.15
        elif entropy >= 2.0:
            score += 0.05
        evidence["entropy"] = round(entropy, 2)

    # High cardinality (each message body is unique)
    n_distinct = len(fs.observed_values)
    if n_distinct >= 20:
        score += 0.1
        evidence["distinct_values"] = n_distinct

    # Field name heuristic
    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    body_name_signals = {
        "text", "content", "body", "message", "parts",
        "input", "output", "prompt", "response",
    }
    if terminal in body_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    # Length variance (bodies vary widely)
    if length_stats["stddev"] >= 50:
        score += 0.05
        evidence["stddev"] = round(length_stats["stddev"], 1)

    if score < 0.15:
        return None

    return SemanticCandidate(
        path=path,
        role="message_body",
        confidence=min(1.0, score),
        evidence=evidence,
    )


def _score_timestamp(path: str, fs: FieldStats) -> SemanticCandidate | None:
    """Score as message_timestamp: epoch/ISO format, monotonic within arrays."""
    evidence: dict[str, Any] = {}
    score = 0.0

    # Check for timestamp format
    fmt = fs.dominant_format
    is_timestamp_fmt = fmt in {"unix-epoch", "unix-epoch-str", "iso8601"}
    if is_timestamp_fmt:
        score += 0.35
        evidence["format"] = fmt
    elif fs.detected_formats:
        ts_formats = {"unix-epoch", "unix-epoch-str", "iso8601"}
        ts_count = sum(fs.detected_formats.get(f, 0) for f in ts_formats)
        if ts_count > 0 and fs.value_count > 0:
            ts_ratio = ts_count / fs.value_count
            if ts_ratio >= 0.5:
                score += 0.2
                evidence["timestamp_ratio"] = round(ts_ratio, 3)
    else:
        # Numeric range in epoch territory
        if fs.num_min is not None and fs.num_max is not None:
            if 946684800.0 <= fs.num_min and fs.num_max <= 2208988800.0:
                score += 0.2
                evidence["range"] = [fs.num_min, fs.num_max]

    if score == 0:
        return None

    # Monotonicity bonus
    mono = fs.monotonicity_score
    if mono is not None and mono >= 0.8:
        score += 0.2
        evidence["monotonicity"] = round(mono, 3)

    # High frequency (timestamps usually always present)
    if fs.frequency >= 0.9:
        score += 0.1
    evidence["frequency"] = round(fs.frequency, 3)

    # Field name heuristic
    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    ts_name_signals = {
        "timestamp", "created_at", "create_time", "updated_at",
        "update_time", "time", "date", "datetime", "ts",
        "created", "modified",
    }
    if terminal in ts_name_signals:
        score += 0.15
        evidence["name_signal"] = terminal

    # Should NOT be multiline or enum-like
    if fs.is_multiline > 0:
        score *= 0.5
    if fs.is_enum_like and len(fs.observed_values) < 5:
        score *= 0.3

    if score < 0.15:
        return None

    return SemanticCandidate(
        path=path,
        role="message_timestamp",
        confidence=min(1.0, score),
        evidence=evidence,
    )


_ANTI_TITLE_NAME_TOKENS = frozenset({
    "uuid", "model", "version", "hash", "key", "token",
    "ref", "parent", "slug", "path", "config", "setting",
})


def _score_title(
    path: str,
    fs: FieldStats,
    all_stats: dict[str, FieldStats],
) -> SemanticCandidate | None:
    """Score as conversation_title: short, high-cardinality, non-multiline."""
    # Format-aware suppression: ID/UUID-format fields cannot be titles
    if fs.dominant_format in {"uuid4", "uuid", "hex-id", "url", "email", "base64"}:
        return None

    length_stats = fs.string_length_stats
    if not length_stats:
        return None

    evidence: dict[str, Any] = {}
    score = 0.0

    # Short strings (titles are typically 5-100 chars)
    avg_len = length_stats["avg"]
    if 3 <= avg_len <= 100:
        score += 0.2
        evidence["avg_length"] = round(avg_len, 1)
    elif avg_len > 200:
        return None  # too long for a title

    # High cardinality (each conversation has a unique title)
    n_distinct = len(fs.observed_values)
    if n_distinct >= 5:
        score += 0.15
        evidence["distinct_values"] = n_distinct

    # Low multiline rate (titles are single-line)
    nl_rate = fs.newline_rate
    if nl_rate <= 0.05:
        score += 0.15
    elif nl_rate > 0.3:
        score *= 0.3
    evidence["newline_rate"] = round(nl_rate, 3)

    # Should NOT be inside the message container
    # (titles are at conversation level, not message level)
    depth = path.count(".")
    if depth <= 2:
        score += 0.1
    elif depth > 4:
        score *= 0.5
    evidence["depth"] = depth

    # Not inside array items (titles are per-conversation, not per-message)
    if "[*]" in path:
        score *= 0.3

    # Field name heuristic
    terminal = path.rsplit(".", 1)[-1].lower().replace("[*]", "")
    title_name_signals = {"title", "name", "subject", "topic", "heading"}
    if terminal in title_name_signals:
        score += 0.2
        evidence["name_signal"] = terminal

    # Anti-name signals: field names containing ID/model/hash/etc. tokens
    # penalize heavily — these are structural fields, not titles.
    terminal_lower = terminal.replace("_", " ").replace("-", " ")
    for anti_token in _ANTI_TITLE_NAME_TOKENS:
        if anti_token in terminal_lower:
            score *= 0.1
            evidence["anti_name_signal"] = anti_token
            break
    # Also check for "id" as a suffix/word boundary (e.g. "parentId", "user_id")
    if terminal.endswith(("id", "Id", "ID", "_id", "-id")):
        score *= 0.1
        evidence["anti_name_signal"] = "id_suffix"

    # Value structure penalty: if >30% of values contain "/" → likely paths/model identifiers
    if fs.observed_values:
        slash_count = sum(1 for v in fs.observed_values if isinstance(v, str) and "/" in v)
        slash_ratio = slash_count / len(fs.observed_values)
        if slash_ratio > 0.3:
            score *= 0.3
            evidence["slash_ratio"] = round(slash_ratio, 3)

    if score < 0.15:
        return None

    return SemanticCandidate(
        path=path,
        role="conversation_title",
        confidence=min(1.0, score),
        evidence=evidence,
    )


__all__ = [
    "SemanticCandidate",
    "SEMANTIC_ROLES",
    "infer_semantic_roles",
    "select_best_roles",
]
