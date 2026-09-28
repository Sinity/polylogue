"""Distribution-driven synthetic workloads for fixtures and benchmarks.

The schema-driven generator in this package proves *shape*: every schema
construct has a witness that parses. It does not reproduce the archive's
*workload*: session sizes, record-kind mix, tool call/result pairing, text
length tails, subagent lineage and sidecar files. This module does, from a
committed, aggregate-only workload profile per origin
(``polylogue/schemas/providers/<origin>/workload-corpus.json``) measured over
the operator's real sources by ``devtools schema workload-profile``.

A profile holds only histograms and rates: record-kind transition counts,
log2-bucketed record counts, text lengths and time gaps, and a few shares.
Generation is deterministic from a seed, streams one session at a time, and
keeps every observed tail (no size caps). Text is gibberish with the observed
non-ASCII share; everything a parser or the pipeline reads structurally
(ids, parent chains, tool call/result ids, usage, lineage, sidecars) is
real-shaped.

Usage::

    corpus = generate_workload_corpus(seed=7, target_bytes=50_000_000)
    for item in corpus.iter_files():   # in memory, one session at a time
        ...
    corpus.write(tmp_path)             # the source trees the daemon watches
"""

from __future__ import annotations

import bisect
import json
import random
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path

WORKLOAD_PROFILE_FILE = "workload-corpus.json"
WORKLOAD_PROFILE_KIND = "polylogue.synthetic-workload-profile"
WORKLOAD_PROFILE_VERSION = 1

#: Origins with a workload renderer.
WORKLOAD_ORIGINS: tuple[str, ...] = ("claude-code", "codex")

_PROVIDERS_ROOT = Path(__file__).resolve().parents[1] / "providers"
_BASE_EPOCH = datetime(2025, 6, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Record-kind classification (shared by the extractor and the renderers)
# ---------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,47}$")

#: Record kinds each renderer builds by hand because they carry relations
#: (ids, parent chains, tool call/result pairing, cumulative usage). Every
#: other kind is a *template kind* ("record:<type>[:<subtype>]") rendered from
#: its measured key skeleton.
CLAUDE_CODE_RELATIONAL_KINDS = frozenset(
    {"user_text", "user_tool_result", "assistant_text", "assistant_thinking", "assistant_tool_use"}
)
CODEX_RELATIONAL_KINDS = frozenset(
    {
        "session_meta",
        "turn_context",
        "user_message",
        "assistant_message",
        "developer_message",
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "event_token_count",
    }
)


def _safe_token(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.match(value) else None


_SOURCES_ROOT = Path(__file__).resolve().parents[2] / "sources"


@cache
def published_kind_tokens() -> frozenset[str]:
    """Record-type vocabulary already public in Polylogue's own source adapters.

    A record ``type``/subtype value is published into a profile only when it
    appears as a string literal in ``polylogue/sources``; any other value is
    source data, not vocabulary, and is folded away.
    """
    import ast

    tokens: set[str] = set()
    for path in sorted(_SOURCES_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _IDENTIFIER.match(node.value):
                tokens.add(node.value)
    return frozenset(tokens)


def _kind(base: object, subtype: object) -> str:
    vocabulary = published_kind_tokens()
    base_token = _safe_token(base)
    if base_token is None or base_token not in vocabulary:
        return "record:other"
    subtype_token = _safe_token(subtype)
    return f"record:{base_token}:{subtype_token}" if subtype_token in vocabulary else f"record:{base_token}"


def classify_claude_code_record(record: Mapping[str, object]) -> str:
    """Return the workload kind of one Claude Code transcript record."""
    record_type = record.get("type")
    message = record.get("message")
    if record_type in {"user", "assistant"} and isinstance(message, Mapping):
        content = message.get("content")
        block_types = (
            {block.get("type") for block in content if isinstance(block, Mapping)}
            if isinstance(content, list)
            else set()
        )
        if record_type == "user":
            return "user_tool_result" if "tool_result" in block_types else "user_text"
        # A record carrying several blocks is classified by its most
        # consequential one: a tool call outranks thinking outranks text.
        if "tool_use" in block_types:
            return "assistant_tool_use"
        if "thinking" in block_types:
            return "assistant_thinking"
        return "assistant_text"
    data = record.get("data")
    attachment = record.get("attachment")
    subtype = (
        (data.get("type") if isinstance(data, Mapping) else None)
        or (attachment.get("type") if isinstance(attachment, Mapping) else None)
        or record.get("subtype")
    )
    return _kind(record_type, subtype)


def classify_codex_record(record: Mapping[str, object]) -> str:
    """Return the workload kind of one Codex rollout record.

    Records without a ``payload`` envelope belong to the legacy flat format,
    which a rollout never mixes with envelopes; they classify as
    ``legacy`` and are not rendered.
    """
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return "legacy"
    payload_type = payload.get("type")
    if record_type in {"session_meta", "turn_context"}:
        return str(record_type)
    if record_type == "response_item":
        if payload_type == "message":
            role = payload.get("role")
            return f"{role}_message" if role in {"user", "assistant", "developer"} else "assistant_message"
        if payload_type in {
            "reasoning",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        }:
            return str(payload_type)
    if record_type == "event_msg" and payload_type == "token_count":
        return "event_token_count"
    return _kind(record_type, payload_type)


#: Claude Code's persisted-output wrapper, which names the full size of the
#: result it moved into a tool-results sidecar.
PERSISTED_OUTPUT = "<persisted-output>"
_PERSISTED_SIZE = re.compile(r"Output too large \(([0-9.]+)\s*(B|KB|MB|GB)\)")
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def persisted_output_size(body: str) -> int | None:
    """Full size, in characters, of a result moved into a sidecar, or None."""
    if not body.startswith(PERSISTED_OUTPUT):
        return None
    match = _PERSISTED_SIZE.search(body[:400])
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2)]) if match else None


def _compact(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def measured_text(origin: str, kind: str, record: Mapping[str, object]) -> str | None:
    """The dominant variable-size text of a relational record, or None.

    This is the field whose length and character class the profile measures
    and the renderer reproduces: the classified block's text, thinking, tool
    input or result body (Claude Code); the message text, call arguments or
    input, output, or reasoning content (Codex).
    """
    if kind.startswith("record:") or kind == "legacy":
        return None
    if origin == "claude-code":
        message = record.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            return content
        wanted = {"assistant_tool_use": "tool_use", "user_tool_result": "tool_result", "assistant_thinking": "thinking"}
        blocks = [block for block in content if isinstance(block, Mapping)] if isinstance(content, list) else []
        block = next((item for item in blocks if item.get("type") == wanted.get(kind)), blocks[0] if blocks else None)
        if not isinstance(block, Mapping):
            return None
        if kind == "assistant_tool_use":
            return _compact(block.get("input"))
        if kind == "user_tool_result":
            return _compact(block.get("content") or "")
        if kind == "assistant_thinking":
            return str(block.get("thinking") or "")
        return str(block.get("text") or "")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    if kind.endswith("_message"):
        content = payload.get("content")
        if isinstance(content, list):
            return "".join(str(item.get("text") or "") for item in content if isinstance(item, Mapping))
        return None
    if kind == "function_call":
        return str(payload.get("arguments") or "")
    if kind == "custom_tool_call":
        return str(payload.get("input") or "")
    if kind in {"function_call_output", "custom_tool_call_output"}:
        return _compact(payload.get("output") or "")
    if kind == "reasoning":
        return str(payload.get("encrypted_content") or "")
    return None


def text_measure(origin: str, kind: str, record: Mapping[str, object]) -> int | None:
    """Length of a relational record's dominant text, or None.

    A Claude Code result moved into a sidecar counts at its full size, so the
    large-result tail and the sidecar share are measured on one population.
    """
    text = measured_text(origin, kind, record)
    if text is None:
        return None
    if origin == "claude-code" and kind == "user_tool_result":
        persisted = persisted_output_size(text)
        if persisted is not None:
            return persisted
    return len(text)


# ---------------------------------------------------------------------------
# Key skeletons (template kinds)
# ---------------------------------------------------------------------------

_SKELETON_DEPTH = 6
#: Containers whose keys are data (JSON-schema properties, model ids, file
#: paths, user-defined structured output), so their inner keys are not kept.
_OPAQUE_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "modelUsage", "trackedFileBackups"}
)


def record_skeleton(value: object, depth: int = 0, *, allowed: frozenset[str] | None = None) -> object:
    """Key/type skeleton of a record: field names and JSON types, no values.

    Keys that are not identifier-like (paths, hashes, free text used as keys)
    are dropped: they are data, not structure. With ``allowed``, only field
    names already published in the origin's committed schema package are
    kept, so a skeleton never introduces a name the reviewed schema lacks.
    """
    if isinstance(value, Mapping):
        if depth >= _SKELETON_DEPTH:
            return "obj"
        opaque = _OPAQUE_KEYS | ({"data"} if value.get("type") == "structured_output" else set())
        return {
            key: "obj" if key in opaque else record_skeleton(item, depth + 1, allowed=allowed)
            for key, item in sorted(value.items())
            if isinstance(key, str) and _IDENTIFIER.match(key) and (allowed is None or key in allowed)
        }
    if isinstance(value, list):
        if depth >= _SKELETON_DEPTH or not value:
            return []
        return [record_skeleton(value[0], depth + 1, allowed=allowed)]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "null"


@cache
def published_field_names(origin: str) -> frozenset[str]:
    """Every property name in the origin's committed schema packages."""
    import gzip

    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            properties = node.get("properties")
            if isinstance(properties, Mapping):
                names.update(key for key in properties if isinstance(key, str))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    for path in sorted((_PROVIDERS_ROOT / origin / "versions").glob("*/elements/*.schema.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            walk(json.load(handle))
    return frozenset(names)


def string_lengths(value: object, depth: int = 0) -> Iterator[int]:
    """Lengths of the string leaves of a record (template kinds)."""
    if depth > _SKELETON_DEPTH:
        return
    if isinstance(value, str):
        yield len(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from string_lengths(item, depth + 1)
    elif isinstance(value, list):
        for item in value[:4]:
            yield from string_lengths(item, depth + 1)


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


def log2_bucket(value: float) -> int:
    """Bucket index for a non-negative value: 0 → 0, [2^(b-1), 2^b) → b."""
    if value < 1:
        return 0
    return int(value).bit_length()


def bucket_bounds(bucket: int) -> tuple[int, int]:
    if bucket <= 0:
        return (0, 0)
    return (1 << (bucket - 1), (1 << bucket) - 1)


@dataclass(frozen=True)
class Histogram:
    """Log2-bucketed histogram; samples uniformly inside the chosen bucket."""

    buckets: tuple[int, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> Histogram:
        if not payload:
            return cls((0,), (1.0,))
        items = sorted((int(key), _number(value)) for key, value in payload.items() if _number(value) > 0)
        return cls(tuple(item[0] for item in items), tuple(item[1] for item in items))

    @property
    def cumulative(self) -> tuple[float, ...]:
        return _cumulative(self.weights)

    def sample(self, rng: random.Random) -> int:
        index = bisect.bisect_right(self.cumulative, rng.random() * self.cumulative[-1])
        low, high = bucket_bounds(self.buckets[min(index, len(self.buckets) - 1)])
        return rng.randint(low, high) if high > low else low


@cache
def _cumulative(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = 0.0
    out = []
    for weight in weights:
        total += weight
        out.append(total)
    return tuple(out)


@dataclass(frozen=True)
class StreamProfile:
    """One stream family (a main session transcript, or a subagent transcript)."""

    records: Histogram
    start: Mapping[str, float]
    transitions: Mapping[str, Mapping[str, float]]
    lengths: Mapping[str, Histogram]
    gap_ms: Histogram

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> StreamProfile:
        raw_lengths = payload.get("lengths")
        lengths = raw_lengths if isinstance(raw_lengths, Mapping) else {}
        raw_transitions = payload.get("transitions")
        transitions = raw_transitions if isinstance(raw_transitions, Mapping) else {}
        raw_start = payload.get("start")
        start = raw_start if isinstance(raw_start, Mapping) else {}
        return cls(
            records=Histogram.from_payload(_mapping(payload.get("records"))),
            start={str(k): _number(v) for k, v in start.items()},
            transitions={
                str(k): {str(k2): _number(v2) for k2, v2 in _mapping(v).items()} for k, v in transitions.items()
            },
            lengths={str(k): Histogram.from_payload(_mapping(v)) for k, v in lengths.items()},
            gap_ms=Histogram.from_payload(_mapping(payload.get("gap_ms"))),
        )

    def kind_sequence(self, rng: random.Random, count: int) -> list[str]:
        kinds: list[str] = []
        current = _weighted(rng, self.start) if self.start else "other"
        for _ in range(count):
            kinds.append(current)
            row = self.transitions.get(current)
            current = _weighted(rng, row) if row else (_weighted(rng, self.start) if self.start else current)
        return kinds

    def length(self, rng: random.Random, kind: str) -> int:
        histogram = self.lengths.get(kind)
        return histogram.sample(rng) if histogram is not None else rng.randint(8, 64)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float:
    """A profile number; anything else (a malformed entry) weighs nothing."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    keys = tuple(weights)
    values = tuple(weights[key] for key in keys)
    cumulative = _cumulative(values)
    return keys[min(bisect.bisect_right(cumulative, rng.random() * cumulative[-1]), len(keys) - 1)]


@dataclass(frozen=True)
class WorkloadProfile:
    origin: str
    streams: Mapping[str, StreamProfile]
    #: Subagent transcripts per main session.
    subagents_per_session: Histogram
    #: Named rates and conditional shares (see ``devtools schema workload-profile``).
    shares: Mapping[str, float]
    #: Byte share of this origin in the measured source set.
    source_bytes: int
    #: Main (non-subagent) sessions in the measured source set.
    main_sessions: int = 0
    #: Share of texts carrying non-ASCII characters, per record kind.
    non_ascii_by_kind: Mapping[str, float] = field(default_factory=dict)
    #: Measured key skeletons per template kind, with weights.
    templates: Mapping[str, tuple[tuple[object, float], ...]] = field(default_factory=dict)
    #: String-leaf lengths per template kind.
    template_strings: Mapping[str, Histogram] = field(default_factory=dict)
    #: Tool-call names (modelled tools only; others fold into ``other``).
    tool_names: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> WorkloadProfile:
        streams = _mapping(payload.get("streams"))
        shares = _mapping(payload.get("shares"))
        return cls(
            origin=str(payload.get("origin")),
            streams={str(name): StreamProfile.from_payload(_mapping(value)) for name, value in streams.items()},
            subagents_per_session=Histogram.from_payload(_mapping(payload.get("subagents_per_session"))),
            shares={str(k): float(v) for k, v in shares.items() if isinstance(v, int | float)},
            source_bytes=int(_number(payload.get("source_bytes"))),
            main_sessions=int(_number(payload.get("main_sessions"))),
            tool_names={str(k): _number(v) for k, v in _mapping(payload.get("tool_names")).items() if _number(v) > 0},
            non_ascii_by_kind={
                str(k): float(v)
                for k, v in _mapping(payload.get("non_ascii_by_kind")).items()
                if isinstance(v, int | float)
            },
            templates={
                str(kind): tuple(
                    (entry["skeleton"], _number(entry.get("weight")))
                    for entry in entries
                    if isinstance(entry, Mapping) and "skeleton" in entry
                )
                for kind, entries in _mapping(payload.get("templates")).items()
                if isinstance(entries, list)
            },
            template_strings={
                str(kind): Histogram.from_payload(_mapping(value))
                for kind, value in _mapping(payload.get("template_strings")).items()
            },
        )

    def share(self, name: str, default: float = 0.0) -> float:
        return self.shares.get(name, default)

    def non_ascii(self, kind: str) -> float:
        return self.non_ascii_by_kind.get(kind, self.share("non_ascii_text_share", 0.0))

    def template_record(self, rng: random.Random, kind: str, fill: Mapping[str, object]) -> dict[str, object]:
        """Instantiate a template kind from one of its measured skeletons."""
        entries = self.templates.get(kind)
        if not entries:
            skeleton: object = {}
        else:
            weights = {str(index): weight for index, (_, weight) in enumerate(entries)}
            skeleton = entries[int(_weighted(rng, weights))][0]
        lengths = self.template_strings.get(kind, _DEFAULT_STRINGS)
        record = _instantiate(skeleton, rng, fill, lengths)
        return record if isinstance(record, dict) else {}

    def orphan_subagents(self, rng: random.Random) -> int:
        """Subagent transcripts whose parent is not a retained session."""
        return _rate(rng, self.share("orphan_subagents_per_session"))

    def nested_subagents(self, rng: random.Random) -> int:
        """Subagents spawned by one subagent (a nested spawn)."""
        return _rate(rng, self.share("nested_subagents_per_subagent"))


def _rate(rng: random.Random, mean: float) -> int:
    whole, fraction = divmod(mean, 1.0)
    return int(whole) + (1 if rng.random() < fraction else 0)


_DEFAULT_STRINGS = Histogram((3, 4, 5), (1.0, 1.0, 1.0))


def _instantiate(skeleton: object, rng: random.Random, fill: Mapping[str, object], lengths: Histogram) -> object:
    if isinstance(skeleton, Mapping):
        out: dict[str, object] = {}
        for key, sub in skeleton.items():
            if key in fill:
                value = fill[key]
                if value is not _SKIP:
                    out[key] = value
            else:
                out[key] = _instantiate(sub, rng, fill, lengths)
        return out
    if isinstance(skeleton, list):
        return [_instantiate(skeleton[0], rng, fill, lengths) for _ in range(rng.randint(1, 3))] if skeleton else []
    if skeleton == "str":
        return synthetic_text(rng, lengths.sample(rng), non_ascii=False)
    if skeleton == "int":
        return rng.randint(0, 5000)
    if skeleton == "float":
        return round(rng.random() * 100, 3)
    if skeleton == "bool":
        return rng.random() < 0.5
    if skeleton == "obj":
        return {}
    return None


#: Marker for a fill entry that removes the key instead of setting it.
_SKIP = object()


def workload_profile_path(origin: str) -> Path:
    return _PROVIDERS_ROOT / origin / WORKLOAD_PROFILE_FILE


@cache
def load_workload_profile(origin: str) -> WorkloadProfile:
    path = workload_profile_path(origin)
    if not path.exists():
        raise FileNotFoundError(f"no committed workload profile for {origin!r}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != WORKLOAD_PROFILE_KIND or payload.get("version") != WORKLOAD_PROFILE_VERSION:
        raise ValueError(f"{path} is not a v{WORKLOAD_PROFILE_VERSION} {WORKLOAD_PROFILE_KIND}")
    return WorkloadProfile.from_payload(payload)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_SYLLABLES = (
    "ka", "lo", "mi", "ne", "ru", "ta", "vo", "si", "pe", "da", "gu", "ho", "ze", "xi", "fa", "bo",
    "en", "ar", "ul", "is", "om", "et", "an", "or",
)  # fmt: skip
#: Single-code-unit characters, so a substitution keeps the exact length.
_NON_ASCII_BMP = ("ą", "ż", "ó", "é", "ü", "ß", "ñ", "ł", "ś", "ć", "—", "…", "→", "λ", "Ж", "日", "本")
_NON_ASCII = ("ą", "ż", "ó", "é", "ü", "ß", "ñ", "ł", "ś", "ć", "—", "…", "→", "λ", "Ж", "日", "本", "🙂")


@cache
def _text_pool(non_ascii_per_mille: int, seed: int = 0x5EED) -> str:
    """A fixed pseudo-text buffer; slices of it are the synthetic text."""
    rng = random.Random(seed * 1009 + non_ascii_per_mille)
    words: list[str] = []
    size = 0
    while size < 1 << 21:
        word = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(1, 4)))
        if non_ascii_per_mille and rng.randrange(1000) < non_ascii_per_mille:
            position = rng.randint(0, len(word))
            word = word[:position] + rng.choice(_NON_ASCII) + word[position:]
        roll = rng.random()
        separator = "\n" if roll < 0.04 else (". " if roll < 0.12 else " ")
        words.append(word + separator)
        size += len(word) + len(separator)
    return "".join(words)


def synthetic_text(rng: random.Random, length: int, *, non_ascii: bool) -> str:
    """Gibberish of exactly ``length`` characters (tails included, no cap)."""
    if length <= 0:
        return ""
    pool = _text_pool(60 if non_ascii else 0)
    if length <= len(pool):
        start = rng.randrange(0, len(pool) - length + 1)
        text = pool[start : start + length]
    else:
        repeats, remainder = divmod(length, len(pool))
        text = pool * repeats + pool[:remainder]
    if non_ascii and text.isascii():
        # A text sampled into the non-ASCII class must carry one.
        position = rng.randrange(length)
        text = text[:position] + rng.choice(_NON_ASCII_BMP) + text[position + 1 :]
    return text


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadFile:
    origin: str
    #: Path relative to the corpus root, e.g. ``claude-code/projects/p/<sid>.jsonl``.
    relpath: str
    data: bytes
    #: ``transcript`` for a session stream, ``subagent`` or ``sidecar`` otherwise.
    role: str
    session_id: str
    parent_session_id: str | None = None


@dataclass
class WorkloadStats:
    files: int = 0
    bytes: int = 0
    sessions: int = 0
    subagent_sessions: int = 0
    records: int = 0
    tool_calls: int = 0
    sidecars: int = 0
    per_origin_bytes: dict[str, int] = field(default_factory=dict)

    def add(self, item: WorkloadFile, records: int = 0) -> None:
        self.files += 1
        self.bytes += len(item.data)
        self.records += records
        self.per_origin_bytes[item.origin] = self.per_origin_bytes.get(item.origin, 0) + len(item.data)
        if item.role == "transcript":
            self.sessions += 1
        elif item.role == "subagent":
            self.subagent_sessions += 1
        elif item.role == "sidecar":
            self.sidecars += 1


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _token(rng: random.Random, prefix: str, length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "".join(rng.choice(alphabet) for _ in range(length))


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _dumps(record: object) -> bytes:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _Clock:
    def __init__(self, rng: random.Random, start: datetime, gaps: Histogram) -> None:
        self._rng = rng
        self.now = start
        self._gaps = gaps

    def tick(self) -> str:
        self.now += timedelta(milliseconds=self._gaps.sample(self._rng))
        return _iso(self.now)


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

#: Claude Code tools whose input shape is modelled; any other measured tool
#: name folds into ``other`` and renders with a generic input.
CLAUDE_CODE_TOOLS = ("Bash", "Read", "Edit", "Grep", "Glob", "Write", "TodoWrite", "WebFetch", "Agent", "Task")
_CC_OTHER_TOOLS = ("WebSearch", "NotebookEdit", "Skill")
_TODO_STATUSES = ("pending", "in_progress", "completed")


def _claude_code_tool_input(rng: random.Random, name: str, cwd: str, body: str) -> dict[str, object]:
    """The input object a tool of this name takes, carrying ``body`` as its variable text."""
    path = f"{cwd}/src/{synthetic_text(rng, rng.randint(4, 12), non_ascii=False).replace(' ', '_').strip('._') or 'mod'}.py"
    if name == "Bash":
        return {"command": body, "description": synthetic_text(rng, rng.randint(12, 40), non_ascii=False)}
    if name == "Read":
        return {"file_path": path}
    if name == "Edit":
        cut = len(body) // 2
        return {"file_path": path, "old_string": body[:cut], "new_string": body[cut:]}
    if name == "Write":
        return {"file_path": path, "content": body}
    if name == "Grep":
        return {"pattern": body[:80] or "x", "path": cwd}
    if name == "Glob":
        return {"pattern": "**/*.py", "path": cwd}
    if name == "TodoWrite":
        todos = [
            {"content": part, "status": rng.choice(_TODO_STATUSES), "activeForm": part}
            for part in (body[i : i + 120] for i in range(0, max(1, len(body)), 120))
            if part
        ] or [{"content": "x", "status": "pending", "activeForm": "x"}]
        return {"todos": todos}
    if name == "WebFetch":
        return {"url": f"https://example.invalid/{rng.getrandbits(32):08x}", "prompt": body}
    if name in {"Agent", "Task"}:
        return {"description": body[:60] or "x", "prompt": body, "subagent_type": "general-purpose"}
    return {"query": body}


_CC_MODEL = "claude-synthetic-1"
_CC_SIDECAR_THRESHOLD = 30_000


def _claude_code_stream(
    rng: random.Random,
    profile: WorkloadProfile,
    stream: StreamProfile,
    *,
    session_id: str,
    project_dir: str,
    agent_id: str | None,
    clock: _Clock,
    sidecars: list[tuple[str, str, str]],
) -> tuple[bytes, int, int]:
    count = max(1, stream.records.sample(rng))
    kinds = stream.kind_sequence(rng, count)
    cwd = f"/workspace/{project_dir}"
    common: dict[str, object] = {
        "isSidechain": agent_id is not None,
        "userType": "external",
        "cwd": cwd,
        "sessionId": session_id,
        "version": "2.1.0",
        "gitBranch": "main",
    }
    if agent_id is not None:
        common["agentId"] = agent_id
    lines: list[bytes] = []
    open_calls: list[tuple[str, str]] = []
    parent: str | None = None
    tool_calls = 0
    error_share = profile.share("tool_error_share", 0.03)
    usage_share = profile.share("assistant_usage_share", 1.0)

    def text(kind: str) -> str:
        return synthetic_text(rng, stream.length(rng, kind), non_ascii=rng.random() < profile.non_ascii(kind))

    for kind in kinds:
        if kind == "user_tool_result" and not open_calls:
            kind = "assistant_tool_use"
        record_uuid = _uuid(rng)
        timestamp = clock.tick()
        base = {"parentUuid": parent, **common, "uuid": record_uuid, "timestamp": timestamp}
        if kind.startswith("assistant_"):
            if kind == "assistant_tool_use":
                call_id = _token(rng, "toolu_", 24)
                name = _weighted(rng, profile.tool_names) if profile.tool_names else rng.choice(CLAUDE_CODE_TOOLS)
                if name == "other":
                    name = rng.choice(_CC_OTHER_TOOLS)
                block: dict[str, object] = {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": _claude_code_tool_input(rng, name, cwd, text(kind)),
                }
                open_calls.append((call_id, record_uuid))
                tool_calls += 1
            elif kind == "assistant_thinking":
                block = {"type": "thinking", "thinking": text(kind), "signature": _token(rng, "", 180)}
            else:
                block = {"type": "text", "text": text(kind)}
            message: dict[str, object] = {
                "model": _CC_MODEL,
                "id": _token(rng, "msg_", 24),
                "type": "message",
                "role": "assistant",
                "content": [block],
                "stop_reason": None,
                "stop_sequence": None,
            }
            if rng.random() < usage_share:
                message["usage"] = {
                    "input_tokens": rng.randint(1, 40),
                    "cache_creation_input_tokens": rng.randint(0, 4000),
                    "cache_read_input_tokens": rng.randint(0, 200_000),
                    "output_tokens": rng.randint(1, 2000),
                    "service_tier": "standard",
                }
            record = {**base, "type": "assistant", "message": message, "requestId": _token(rng, "req_", 24)}
        elif kind == "user_tool_result":
            call_id, call_uuid = open_calls.pop(0)
            body = text(kind)
            if len(body) > _CC_SIDECAR_THRESHOLD and rng.random() < profile.share("sidecar_share_of_large", 0.5):
                name = f"{call_id}.txt"
                sidecar_path = f"{cwd_sidecar_root(project_dir, session_id)}/{name}"
                sidecars.append((session_id, name, body))
                body = (
                    f"<persisted-output>\nOutput too large ({len(body) / 1024:.1f}KB). "
                    f"Full output saved to: {sidecar_path}\n\nPreview (first 2KB):\n{body[:2048]}\n</persisted-output>"
                )
            is_error = rng.random() < error_share
            record = {
                **base,
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"tool_use_id": call_id, "type": "tool_result", "content": body, "is_error": is_error}],
                },
                "toolUseResult": {"stdout": "", "stderr": "", "interrupted": False, "isImage": False},
                "sourceToolAssistantUUID": call_uuid,
            }
        elif kind == "user_text":
            record = {**base, "type": "user", "message": {"role": "user", "content": text(kind)}}
        else:
            record = _claude_code_template(profile, rng, kind, base, open_calls, parent)
        if record.get("uuid") == record_uuid:
            parent = record_uuid
        lines.append(_dumps(record))
    # Calls the stream ended without answering stay unanswered, as in real
    # interrupted sessions.
    return b"\n".join(lines) + b"\n", len(lines), tool_calls


_CC_TYPE_OWNERS = ("data", "attachment")


def _claude_code_template(
    profile: WorkloadProfile,
    rng: random.Random,
    kind: str,
    base: Mapping[str, object],
    open_calls: Sequence[tuple[str, str]],
    parent: str | None,
) -> dict[str, object]:
    """A non-relational record (progress, attachment, system, snapshot, ...)."""
    parts = kind.split(":")
    record_type = parts[1] if len(parts) > 1 and parts[1] != "other" else "system"
    subtype = parts[2] if len(parts) > 2 else None
    fill: dict[str, object] = {
        "type": record_type,
        "uuid": base["uuid"],
        "parentUuid": parent,
        "sessionId": base["sessionId"],
        "timestamp": base["timestamp"],
        "isSidechain": base["isSidechain"],
        "cwd": base["cwd"],
        "leafUuid": parent,
        "messageId": base["uuid"],
        "toolUseID": open_calls[-1][0] if open_calls else _token(rng, "toolu_", 24),
    }
    if "agentId" in base:
        fill["agentId"] = base["agentId"]
    record = profile.template_record(rng, kind, fill)
    record["type"] = record_type
    if subtype is not None:
        owner = next((record[name] for name in _CC_TYPE_OWNERS if isinstance(record.get(name), dict)), None)
        if isinstance(owner, dict):
            owner["type"] = subtype
        else:
            record["subtype"] = subtype
    return record


def cwd_sidecar_root(project_dir: str, session_id: str) -> str:
    return f"{{projects_root}}/{project_dir}/{session_id}/tool-results"


def _claude_code_session(
    rng: random.Random, profile: WorkloadProfile, *, index: int
) -> tuple[list[WorkloadFile], WorkloadStats]:
    stats = WorkloadStats()
    project_dir = f"-workspace-synthetic-{index % 23:02d}"
    session_id = _uuid(rng)
    start = _BASE_EPOCH + timedelta(seconds=rng.randint(0, 400 * 86400))
    main = profile.streams["main"]
    files: list[WorkloadFile] = []
    sidecars: list[tuple[str, str, str]] = []
    clock = _Clock(rng, start, main.gap_ms)
    data, records, calls = _claude_code_stream(
        rng, profile, main, session_id=session_id, project_dir=project_dir, agent_id=None,
        clock=clock, sidecars=sidecars,
    )  # fmt: skip
    transcript = WorkloadFile("claude-code", f"claude-code/projects/{project_dir}/{session_id}.jsonl", data,
                              "transcript", session_id)  # fmt: skip
    files.append(transcript)
    stats.add(transcript, records)
    stats.tool_calls += calls
    sub_stream = profile.streams.get("subagent")
    subagents = profile.subagents_per_session.sample(rng) if sub_stream is not None else 0
    orphans = profile.orphan_subagents(rng) if sub_stream is not None else 0
    for number in range(subagents + orphans):
        assert sub_stream is not None
        owner = session_id if number < subagents else _uuid(rng)
        agent_id = "a" + _token(rng, "", 16).lower()
        sub_clock = _Clock(rng, clock.now - timedelta(seconds=rng.randint(0, 600)), sub_stream.gap_ms)
        data, records, calls = _claude_code_stream(
            rng, profile, sub_stream, session_id=owner, project_dir=project_dir, agent_id=agent_id,
            clock=sub_clock, sidecars=sidecars,
        )  # fmt: skip
        item = WorkloadFile("claude-code", f"claude-code/projects/{project_dir}/{owner}/subagents/agent-{agent_id}.jsonl",
                            data, "subagent", f"{owner}:agent-{agent_id}", owner)  # fmt: skip
        files.append(item)
        stats.add(item, records)
        stats.tool_calls += calls
        meta = WorkloadFile("claude-code", f"claude-code/projects/{project_dir}/{owner}/subagents/agent-{agent_id}.meta.json",
                            _dumps({"agentType": "general-purpose", "description": synthetic_text(rng, 40, non_ascii=False)}),
                            "sidecar", owner, owner)  # fmt: skip
        files.append(meta)
        stats.add(meta)
    for owner, name, body in sidecars:
        item = WorkloadFile("claude-code", f"claude-code/projects/{project_dir}/{owner}/tool-results/{name}",
                            body.encode("utf-8"), "sidecar", owner, owner)  # fmt: skip
        files.append(item)
        stats.add(item)
    return files, stats


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

_CODEX_TOOLS = ("shell", "apply_patch", "exec_command", "write_stdin", "update_plan")


def _codex_stream(
    rng: random.Random,
    profile: WorkloadProfile,
    stream: StreamProfile,
    *,
    thread_id: str,
    parent_thread_id: str | None,
    clock: _Clock,
) -> tuple[bytes, int, int]:
    count = max(2, stream.records.sample(rng))
    kinds = stream.kind_sequence(rng, count)
    turn_id = _uuid(rng)
    lines: list[bytes] = []
    open_calls: list[tuple[str, str]] = []
    tool_calls = 0
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}

    def text(kind: str) -> str:
        return synthetic_text(rng, stream.length(rng, kind), non_ascii=rng.random() < profile.non_ascii(kind))

    source: object = "cli"
    meta_payload: dict[str, object] = {
        "id": thread_id,
        "session_id": thread_id,
        "timestamp": clock.tick(),
        "cwd": "/workspace/synthetic",
        "originator": "codex_cli_rs",
        "cli_version": "0.99.0",
        "model_provider": "openai",
        "base_instructions": {"text": synthetic_text(rng, 400, non_ascii=False)},
    }
    if parent_thread_id is not None:
        source = {"subagent": {"thread_spawn": {"parent_thread_id": parent_thread_id, "depth": 1,
                                                "agent_nickname": "helper", "agent_role": "worker"}}}  # fmt: skip
        meta_payload["parent_thread_id"] = parent_thread_id
    meta_payload["source"] = source
    lines.append(_dumps({"timestamp": meta_payload["timestamp"], "type": "session_meta", "payload": meta_payload}))
    passthrough = {"turn_id": turn_id}
    for kind in kinds:
        if kind in {"session_meta", "legacy"}:
            continue
        if kind in {"function_call_output", "custom_tool_call_output"} and not open_calls:
            kind = "function_call" if kind == "function_call_output" else "custom_tool_call"
        timestamp = clock.tick()
        payload: dict[str, object]
        record_type = "response_item"
        if kind == "turn_context":
            turn_id = _uuid(rng)
            passthrough = {"turn_id": turn_id}
            record_type = "turn_context"
            payload = {"turn_id": turn_id, "cwd": "/workspace/synthetic", "model": "gpt-synthetic",
                       "approval_policy": "never", "sandbox_policy": {"type": "danger-full-access"}, "effort": "high"}  # fmt: skip
        elif kind.endswith("_message"):
            role = kind.removesuffix("_message")
            part_type = "output_text" if role == "assistant" else "input_text"
            payload = {"type": "message", "id": _token(rng, "msg_", 48), "role": role,
                       "content": [{"type": part_type, "text": text(kind)}],
                       "internal_chat_message_metadata_passthrough": passthrough}  # fmt: skip
        elif kind == "reasoning":
            payload = {"type": "reasoning", "id": _token(rng, "rs_", 48), "summary": [], "content": None,
                       "encrypted_content": _token(rng, "", max(16, stream.length(rng, kind)))}  # fmt: skip
        elif kind in {"function_call", "custom_tool_call"}:
            call_id = _token(rng, "call_", 24)
            open_calls.append((call_id, kind))
            tool_calls += 1
            if kind == "function_call":
                payload = {"type": "function_call", "name": rng.choice(_CODEX_TOOLS),
                           "arguments": json.dumps({"cmd": text(kind)}, ensure_ascii=False), "call_id": call_id}  # fmt: skip
            else:
                payload = {"type": "custom_tool_call", "id": _token(rng, "ctc_", 48), "status": "completed",
                           "call_id": call_id, "name": "apply_patch", "input": text(kind)}  # fmt: skip
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            wanted = "function_call" if kind == "function_call_output" else "custom_tool_call"
            position = next((i for i, (_, k) in enumerate(open_calls) if k == wanted), 0)
            call_id, call_kind = open_calls.pop(position)
            output_kind = f"{call_kind}_output"
            payload = {
                "type": output_kind,
                "call_id": call_id,
                "output": _codex_output(rng, profile, output_kind, text(kind)),
            }
        elif kind == "event_token_count":
            record_type = "event_msg"
            last = {"input_tokens": rng.randint(100, 60_000), "cached_input_tokens": rng.randint(0, 50_000),
                    "output_tokens": rng.randint(1, 3000), "reasoning_output_tokens": rng.randint(0, 2000)}  # fmt: skip
            for key, value in last.items():
                totals[key] += value
            payload = {"type": "token_count", "info": {
                "total_token_usage": {**totals, "total_tokens": totals["input_tokens"] + totals["output_tokens"]},
                "last_token_usage": {**last, "total_tokens": last["input_tokens"] + last["output_tokens"]},
                "model_context_window": 272_000}}  # fmt: skip
        else:
            lines.append(_dumps(_codex_template(profile, rng, kind, timestamp, turn_id)))
            continue
        lines.append(_dumps({"timestamp": timestamp, "type": record_type, "payload": payload}))
    return b"\n".join(lines) + b"\n", len(lines), tool_calls


def _codex_output(rng: random.Random, profile: WorkloadProfile, output_kind: str, body: str) -> str:
    """A tool output in the producer's structural form, with its exit code.

    Exec-style calls answer with the unified-exec envelope and custom tools
    with a JSON object carrying ``metadata.exit_code``, at their measured
    shares; the rest are bare text, which the parser records as an unknown
    outcome, as it does for real bare outputs.
    """
    if output_kind == "function_call_output":
        if rng.random() >= profile.share("codex_exec_envelope_share", 0.0):
            return body
        code = 1 if rng.random() < profile.share("codex_exec_error_share", 0.0) else 0
        return (
            f"Chunk ID: {rng.getrandbits(24):06x}\nWall time: {rng.random() * 30:.4f} seconds\n"
            f"Process exited with code {code}\nOriginal token count: {max(1, len(body) // 4)}\nOutput:\n{body}"
        )
    if rng.random() >= profile.share("codex_custom_json_share", 0.0):
        return body
    code = 1 if rng.random() < profile.share("codex_custom_error_share", 0.0) else 0
    metadata = {"exit_code": code, "duration_seconds": round(rng.random() * 5, 1)}
    return json.dumps({"output": body, "metadata": metadata}, ensure_ascii=False, separators=(",", ":"))


def _codex_template(
    profile: WorkloadProfile, rng: random.Random, kind: str, timestamp: str, turn_id: str
) -> dict[str, object]:
    """A non-relational rollout record (events, compaction, other items)."""
    parts = kind.split(":")
    record_type = parts[1] if len(parts) > 1 and parts[1] != "other" else "event_msg"
    subtype = parts[2] if len(parts) > 2 else None
    record = profile.template_record(rng, kind, {"timestamp": timestamp, "turn_id": turn_id})
    record["timestamp"] = timestamp
    record["type"] = record_type
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}
        record["payload"] = payload
    if subtype is not None:
        payload["type"] = subtype
    return record


def _codex_session(
    rng: random.Random, profile: WorkloadProfile, *, index: int
) -> tuple[list[WorkloadFile], WorkloadStats]:
    del index
    stats = WorkloadStats()
    files: list[WorkloadFile] = []

    def rollout(stream: StreamProfile, parent: str | None, start: datetime) -> tuple[WorkloadFile, datetime]:
        thread_id = _uuid(rng)
        clock = _Clock(rng, start, stream.gap_ms)
        data, records, calls = _codex_stream(rng, profile, stream, thread_id=thread_id, parent_thread_id=parent,
                                             clock=clock)  # fmt: skip
        stamp = start.strftime("%Y-%m-%dT%H-%M-%S")
        relpath = f"codex/sessions/{start:%Y/%m/%d}/rollout-{stamp}-{thread_id}.jsonl"
        item = WorkloadFile("codex", relpath, data, "subagent" if parent else "transcript", thread_id, parent)
        stats.add(item, records)
        stats.tool_calls += calls
        return item, clock.now

    start = _BASE_EPOCH + timedelta(seconds=rng.randint(0, 400 * 86400))
    main, end = rollout(profile.streams["main"], None, start)
    files.append(main)
    sub_stream = profile.streams.get("subagent")
    if sub_stream is not None:
        # Spawn edges: main → subagents, subagent → nested subagents (at the
        # measured rate), and orphans whose parent was never retained. A child
        # starts inside its direct parent's lifetime, never before it.
        pending = [(main.session_id, start, end)] * profile.subagents_per_session.sample(rng)
        pending += [(_uuid(rng), start, end) for _ in range(profile.orphan_subagents(rng))]
        while pending:
            parent, parent_start, parent_end = pending.pop()
            child_start = parent_start + (parent_end - parent_start) * rng.random()
            item, child_end = rollout(sub_stream, parent, child_start)
            files.append(item)
            pending += [(item.session_id, child_start, child_end)] * profile.nested_subagents(rng)
    return files, stats


_SESSION_BUILDERS = {"claude-code": _claude_code_session, "codex": _codex_session}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadCorpus:
    """A deterministic, lazily generated workload.

    Iterate ``iter_files()`` to generate in memory; ``write(root)`` lays the
    files out as the source trees the daemon watches, resolving each Claude
    Code tool-result sidecar reference against ``root``.
    """

    seed: int
    origins: tuple[tuple[str, float], ...]
    target_bytes: int | None = None
    target_sessions: int | None = None

    def iter_sessions(self) -> Iterator[tuple[list[WorkloadFile], WorkloadStats]]:
        if self.target_bytes is None and self.target_sessions is None:
            raise ValueError("a workload needs target_bytes or target_sessions")
        rng = random.Random(self.seed)
        profiles = {origin: load_workload_profile(origin) for origin, _ in self.origins}
        names = tuple(origin for origin, _ in self.origins)
        weights = tuple(weight for _, weight in self.origins)
        produced_bytes = 0
        index = 0
        while True:
            if self.target_sessions is not None and index >= self.target_sessions:
                return
            if self.target_bytes is not None and produced_bytes >= self.target_bytes:
                return
            origin = rng.choices(names, weights=weights, k=1)[0]
            session_rng = random.Random(f"{self.seed}\x1f{origin}\x1f{index}")
            files, stats = _SESSION_BUILDERS[origin](session_rng, profiles[origin], index=index)
            produced_bytes += stats.bytes
            index += 1
            yield files, stats

    def iter_files(self, *, projects_root: str = "/synthetic/claude-code/projects") -> Iterator[WorkloadFile]:
        for files, _ in self.iter_sessions():
            for item in files:
                yield _resolve_sidecar_refs(item, projects_root)

    def write(self, root: Path) -> WorkloadStats:
        """Write the corpus under ``root``, which must hold no earlier workload.

        A daemon watching ``root`` ingests whatever is there, so writing over a
        previous generation would silently mix two corpora.
        """
        for tree in (root / "claude-code", root / "codex"):
            if tree.exists() and any(tree.iterdir()):
                raise FileExistsError(f"{tree} already holds a workload; write into an empty root")
        total = WorkloadStats()
        projects_root = str((root / "claude-code" / "projects").resolve())
        for files, stats in self.iter_sessions():
            for item in files:
                item = _resolve_sidecar_refs(item, projects_root)
                path = root / item.relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(item.data)
                # Sizes are counted on the written bytes: resolving a sidecar
                # reference changes a transcript's length with the root.
                total.files += 1
                total.bytes += len(item.data)
                total.per_origin_bytes[item.origin] = total.per_origin_bytes.get(item.origin, 0) + len(item.data)
            total.sessions += stats.sessions
            total.subagent_sessions += stats.subagent_sessions
            total.records += stats.records
            total.tool_calls += stats.tool_calls
            total.sidecars += stats.sidecars
        return total


def _resolve_sidecar_refs(item: WorkloadFile, projects_root: str) -> WorkloadFile:
    if item.origin != "claude-code" or item.role not in {"transcript", "subagent"}:
        return item
    marker = b"{projects_root}"
    if marker not in item.data:
        return item
    return WorkloadFile(
        item.origin,
        item.relpath,
        item.data.replace(marker, json.dumps(projects_root)[1:-1].encode("utf-8")),
        item.role,
        item.session_id,
        item.parent_session_id,
    )


def default_origin_weights(origins: Sequence[str] = WORKLOAD_ORIGINS) -> tuple[tuple[str, float], ...]:
    """Weight origins by their measured main-session populations.

    Sessions are drawn one at a time, so session counts are the right weight;
    each origin's own size distribution then yields the byte mix.
    """
    return tuple((origin, float(max(1, load_workload_profile(origin).main_sessions))) for origin in origins)


def generate_workload_corpus(
    *,
    seed: int,
    target_bytes: int | None = None,
    target_sessions: int | None = None,
    origins: Sequence[str] | Mapping[str, float] = WORKLOAD_ORIGINS,
) -> WorkloadCorpus:
    """Return a deterministic workload sized by bytes or session count.

    ``origins`` is a list (weighted by measured source bytes) or an explicit
    ``{origin: weight}`` mapping.
    """
    weighted = (
        tuple((str(origin), float(weight)) for origin, weight in origins.items())
        if isinstance(origins, Mapping)
        else default_origin_weights(tuple(origins))
    )
    unknown = [origin for origin, _ in weighted if origin not in _SESSION_BUILDERS]
    if unknown:
        raise ValueError(f"no workload renderer for {unknown}; supported: {sorted(_SESSION_BUILDERS)}")
    return WorkloadCorpus(seed=seed, origins=weighted, target_bytes=target_bytes, target_sessions=target_sessions)


__all__ = [
    "WORKLOAD_ORIGINS",
    "WorkloadCorpus",
    "WorkloadFile",
    "WorkloadProfile",
    "WorkloadStats",
    "classify_claude_code_record",
    "classify_codex_record",
    "generate_workload_corpus",
    "load_workload_profile",
    "synthetic_text",
    "text_measure",
]
