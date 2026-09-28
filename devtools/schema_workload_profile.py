"""Measure a committed workload profile from real provider sources.

The profile is what ``polylogue.schemas.synthetic.workload`` generates from:
record-kind transitions, log2-bucketed record counts, text lengths and time
gaps, subagent fan-out, and a few shares. It is aggregate-only: no strings,
identifiers, paths or timestamps from the sources are retained; record-type
values and field names are published only when they are already public
(Polylogue's source adapters and committed schema packages); and every count
is rounded to two significant figures.

Sampling is stratified: the largest files of each stream family are always
measured (so tails survive), and a uniform sample of the rest is weighted up
to the full population.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path

from polylogue.core.sources import source_for_family
from polylogue.schemas.synthetic.workload import (
    CLAUDE_CODE_TOOLS,
    PERSISTED_OUTPUT,
    WORKLOAD_PROFILE_KIND,
    WORKLOAD_PROFILE_VERSION,
    classify_claude_code_record,
    classify_codex_record,
    log2_bucket,
    measured_text,
    published_field_names,
    record_skeleton,
    string_lengths,
    text_measure,
    workload_profile_path,
)

#: Skeletons kept per template kind.
_SKELETONS_PER_KIND = 4
#: Source family whose runtime root each origin is measured from by default.
_SOURCE_FAMILIES = {"claude-code": "claude-code-session", "codex": "codex-session"}

Weights = defaultdict[str, float]


def _weights() -> Weights:
    return defaultdict(float)


def _buckets() -> defaultdict[int, float]:
    return defaultdict(float)


class _Templates:
    """Key skeletons and string-leaf lengths of template kinds, across streams.

    Kinds are already restricted to public record-type vocabulary by the
    classifiers, and skeleton keys to field names in the committed schema
    package, so every kind observed gets a template.
    """

    def __init__(self, allowed: frozenset[str]) -> None:
        self.allowed = allowed
        self.skeletons: defaultdict[str, Weights] = defaultdict(_weights)
        self.strings: defaultdict[str, defaultdict[int, float]] = defaultdict(_buckets)

    def add(self, kind: str, record: Mapping[str, object], weight: float) -> None:
        key = json.dumps(record_skeleton(record, allowed=self.allowed), sort_keys=True, separators=(",", ":"))
        self.skeletons[kind][key] += weight
        for length in string_lengths(record):
            self.strings[kind][log2_bucket(length)] += weight

    def payload(self) -> tuple[dict[str, object], dict[str, object]]:
        templates: dict[str, object] = {}
        strings: dict[str, object] = {}
        for kind, entries in sorted(self.skeletons.items()):
            ranked = sorted(entries.items(), key=lambda item: (-item[1], item[0]))[:_SKELETONS_PER_KIND]
            templates[kind] = [{"skeleton": json.loads(key), "weight": _round2(weight)} for key, weight in ranked]
            strings[kind] = _histogram(self.strings[kind])
        return templates, strings


def default_source_root(origin: str) -> Path:
    """The origin's canonical runtime root, as Polylogue itself resolves it."""
    source = source_for_family(_SOURCE_FAMILIES[origin])
    if source is None or source.runtime_root is None:
        raise ValueError(f"{origin} has no runtime root")
    return Path(os.path.expanduser(source.runtime_root))


_EXEC_EXIT = re.compile(r"^Process exited with code (-?\d+)$", re.MULTILINE)


def _round2(value: float) -> float:
    if value <= 0:
        return 0.0
    digits = 1 - int(math.floor(math.log10(value)))
    return round(value, digits)


def _histogram(counter: Mapping[int, float]) -> dict[str, float]:
    return {str(bucket): _round2(weight) for bucket, weight in sorted(counter.items()) if _round2(weight) > 0}


def _parse_ms(value: object) -> float | None:
    if not isinstance(value, str) or len(value) < 19:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000
    except ValueError:
        return None


def _records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                yield value


class _Stream:
    def __init__(self) -> None:
        self.records = _buckets()
        self.start = _weights()
        self.transitions: defaultdict[str, Weights] = defaultdict(_weights)
        self.lengths: defaultdict[str, defaultdict[int, float]] = defaultdict(_buckets)
        self.gaps = _buckets()

    def measure(self, origin: str, path: Path, weight: float, shares: Weights, templates: _Templates) -> int:
        classify = classify_claude_code_record if origin == "claude-code" else classify_codex_record
        previous_kind: str | None = None
        previous_ms: float | None = None
        count = 0
        for record in _records(path):
            kind = classify(record)
            count += 1
            if previous_kind is None:
                self.start[kind] += weight
            else:
                self.transitions[previous_kind][kind] += weight
            previous_kind = kind
            length = text_measure(origin, kind, record)
            if length is not None:
                self.lengths[kind][log2_bucket(length)] += weight
            if kind.startswith("record:"):
                templates.add(kind, record, weight)
            moment = _parse_ms(record.get("timestamp"))
            if moment is not None:
                if previous_ms is not None and moment >= previous_ms:
                    self.gaps[log2_bucket(moment - previous_ms)] += weight
                previous_ms = moment
            _count_shares(origin, kind, record, shares, weight)
        if count:
            self.records[log2_bucket(count)] += weight
        return count

    def payload(self) -> dict[str, object]:
        return {
            "records": _histogram(self.records),
            "start": {kind: _round2(weight) for kind, weight in _ranked(self.start)},
            "transitions": {
                kind: {target: _round2(weight) for target, weight in _ranked(row) if _round2(weight) > 0}
                for kind, row in sorted(self.transitions.items())
            },
            "lengths": {kind: _histogram(counter) for kind, counter in sorted(self.lengths.items())},
            "gap_ms": _histogram(self.gaps),
        }


def _ranked(weights: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))


def _count_shares(origin: str, kind: str, record: Mapping[str, object], shares: Weights, weight: float) -> None:
    # Character class is measured on the same field the lengths are.
    text = measured_text(origin, kind, record)
    if text is not None:
        shares["texts"] += weight
        shares[f"texts:{kind}"] += weight
        if not text[:2000].isascii():
            shares["non_ascii_texts"] += weight
            shares[f"non_ascii_texts:{kind}"] += weight
    if origin == "claude-code":
        message = record.get("message")
        if kind == "assistant_tool_use" and isinstance(message, Mapping):
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    name = block.get("name")
                    shares[f"tool:{name if name in CLAUDE_CODE_TOOLS else 'other'}"] += weight
        if kind.startswith("assistant_") and isinstance(message, Mapping):
            shares["assistant"] += weight
            if isinstance(message.get("usage"), Mapping):
                shares["assistant_usage"] += weight
        if kind == "user_tool_result" and isinstance(message, Mapping):
            content = message.get("content")
            block = (
                next(
                    (item for item in content if isinstance(item, Mapping) and item.get("type") == "tool_result"),
                    None,
                )
                if isinstance(content, list)
                else None
            )
            if isinstance(block, Mapping):
                shares["tool_results"] += weight
                if block.get("is_error") is True:
                    shares["tool_errors"] += weight
                body = block.get("content")
                if isinstance(body, str):
                    if body.startswith(PERSISTED_OUTPUT):
                        shares["sidecar_refs"] += weight
                    elif len(body) > 30_000:
                        shares["large_inline"] += weight
    elif kind in {"function_call_output", "custom_tool_call_output"}:
        payload = record.get("payload")
        output = payload.get("output") if isinstance(payload, Mapping) else None
        if kind == "function_call_output":
            shares["exec_outputs"] += weight
            match = _EXEC_EXIT.search(output) if isinstance(output, str) else None
            if match is not None:
                shares["exec_envelopes"] += weight
                if int(match.group(1)) != 0:
                    shares["exec_errors"] += weight
        else:
            shares["custom_outputs"] += weight
            metadata = None
            if isinstance(output, str) and output.startswith('{"output"'):
                try:
                    decoded = json.loads(output)
                except json.JSONDecodeError:
                    decoded = None
                metadata = decoded.get("metadata") if isinstance(decoded, dict) else None
            if isinstance(metadata, dict) and isinstance(metadata.get("exit_code"), int):
                shares["custom_json"] += weight
                if metadata["exit_code"] != 0:
                    shares["custom_errors"] += weight


def _stream_families(origin: str, root: Path) -> dict[str, list[Path]]:
    files = sorted(root.rglob("*.jsonl"))
    if origin == "claude-code":
        # Subagent transcripts sit directly in ``<session>/subagents/`` as
        # ``agent-*.jsonl``; workflow journals and runs below it are
        # orchestration facts, not sessions.
        return {
            "main": [path for path in files if path.parent.parent == root],
            "subagent": [path for path in files if path.parent.name == "subagents" and path.name.startswith("agent-")],
        }
    families: dict[str, list[Path]] = {"main": [], "subagent": []}
    for path in files:
        if _codex_is_legacy(path):
            continue
        families["subagent" if _codex_parent(path) else "main"].append(path)
    return families


def _codex_is_legacy(path: Path) -> bool:
    """A rollout in the pre-envelope flat format (records without ``payload``)."""
    try:
        with path.open("rb") as handle:
            first = json.loads(handle.readline() or b"{}")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return True
    return not (isinstance(first, dict) and isinstance(first.get("payload"), dict))


def _codex_parent(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            first = json.loads(handle.readline() or b"{}")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    payload = first.get("payload") if isinstance(first, dict) else None
    if not isinstance(payload, dict):
        return None
    parent = payload.get("parent_thread_id")
    source = payload.get("source")
    if not parent and isinstance(source, dict):
        spawn = source.get("subagent")
        if isinstance(spawn, dict) and isinstance(spawn.get("thread_spawn"), dict):
            parent = spawn["thread_spawn"].get("parent_thread_id")
    return parent if isinstance(parent, str) and parent else None


def _fanout(
    origin: str, root: Path, families: Mapping[str, list[Path]]
) -> tuple[defaultdict[int, float], float, float]:
    """Subagent fan-out per main session, nested spawns per subagent, orphans per main session.

    A nested spawn is a subagent whose parent is itself a subagent; an orphan
    is one whose parent is not among the measured sessions at all.
    """
    fanout = _buckets()
    if origin == "claude-code":
        parents = {path: path.parent.parent.name for path in families["subagent"] if path.parent.name == "subagents"}
        main_ids = {main.stem for main in families["main"]}
        subagent_ids: set[str] = set()
    else:
        parents = {path: parent for path in families["subagent"] if (parent := _codex_parent(path))}
        main_ids = {path.stem[-36:] for path in families["main"]}
        subagent_ids = {path.stem[-36:] for path in families["subagent"]}
    per_parent: defaultdict[str, int] = defaultdict(int)
    for parent in parents.values():
        per_parent[parent] += 1
    for session_id in main_ids:
        fanout[log2_bucket(per_parent.get(session_id, 0))] += 1
    nested = sum(count for parent, count in per_parent.items() if parent in subagent_ids and parent not in main_ids)
    orphans = sum(
        count for parent, count in per_parent.items() if parent not in main_ids and parent not in subagent_ids
    )
    return fanout, nested / max(1, len(parents)), orphans / max(1, len(main_ids))


def measure(origin: str, root: Path, *, sample: int, tail: int, seed: int) -> dict[str, object]:
    families = _stream_families(origin, root)
    shares = _weights()
    templates = _Templates(published_field_names(origin))
    streams: dict[str, object] = {}
    source_bytes = 0
    rng = random.Random(seed)
    for name, paths in families.items():
        if not paths:
            continue
        sizes = {path: path.stat().st_size for path in paths}
        source_bytes += sum(sizes.values())
        by_size = sorted(paths, key=sizes.__getitem__, reverse=True)
        tail_paths = by_size[:tail]
        rest = by_size[tail:]
        drawn = rng.sample(rest, min(sample, len(rest)))
        weight = len(rest) / len(drawn) if drawn else 0.0
        stream = _Stream()
        for index, path in enumerate(tail_paths + drawn):
            stream.measure(origin, path, 1.0 if path in tail_paths else weight, shares, templates)
            if index % 200 == 0:
                print(f"  {origin}/{name}: {index + 1}/{len(tail_paths) + len(drawn)}", file=sys.stderr, flush=True)
        streams[name] = stream.payload()

    def ratio(numerator: str, denominator: str) -> float:
        return round(shares[numerator] / shares[denominator], 4) if shares[denominator] else 0.0

    fanout, nested_per_subagent, orphans_per_session = _fanout(origin, root, families)
    kinds = sorted({key.split(":", 1)[1] for key in shares if key.startswith("texts:")})
    template_payload, template_strings = templates.payload()
    large = shares["sidecar_refs"] + shares["large_inline"]
    outcome_shares = (
        {
            "codex_exec_envelope_share": ratio("exec_envelopes", "exec_outputs"),
            "codex_exec_error_share": ratio("exec_errors", "exec_envelopes"),
            "codex_custom_json_share": ratio("custom_json", "custom_outputs"),
            "codex_custom_error_share": ratio("custom_errors", "custom_json"),
        }
        if origin == "codex"
        else {}
    )
    return {
        "kind": WORKLOAD_PROFILE_KIND,
        "version": WORKLOAD_PROFILE_VERSION,
        "origin": origin,
        "source_bytes": int(_round2(source_bytes)),
        "main_sessions": int(_round2(len(families.get("main", ())))),
        "streams": streams,
        "subagents_per_session": _histogram(fanout),
        "shares": {
            "non_ascii_text_share": ratio("non_ascii_texts", "texts"),
            "assistant_usage_share": ratio("assistant_usage", "assistant") if origin == "claude-code" else 1.0,
            "tool_error_share": ratio("tool_errors", "tool_results") if origin == "claude-code" else 0.03,
            "sidecar_share_of_large": round(shares["sidecar_refs"] / large, 4) if large else 0.0,
            "orphan_subagents_per_session": round(orphans_per_session, 4),
            "nested_subagents_per_subagent": round(nested_per_subagent, 4),
            **outcome_shares,
        },
        "non_ascii_by_kind": {kind: ratio(f"non_ascii_texts:{kind}", f"texts:{kind}") for kind in kinds},
        "tool_names": {
            key.split(":", 1)[1]: _round2(weight) for key, weight in _ranked(shares) if key.startswith("tool:")
        },
        "templates": template_payload,
        "template_strings": template_strings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure an aggregate-only synthetic workload profile.")
    parser.add_argument("--origin", required=True, choices=sorted(_SOURCE_FAMILIES))
    parser.add_argument("--source", type=Path, help="Source root (defaults to the origin's usual location).")
    parser.add_argument("--sample", type=int, default=1500, help="Uniformly sampled files per stream family.")
    parser.add_argument("--tail", type=int, default=5, help="Largest files per family always measured.")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--write", action="store_true", help="Write the committed profile instead of printing it.")
    args = parser.parse_args(argv)
    root = (args.source or default_source_root(args.origin)).resolve()
    profile = measure(args.origin, root, sample=args.sample, tail=args.tail, seed=args.seed)
    streams = profile.get("streams")
    if not isinstance(streams, dict) or "main" not in streams:
        print(f"no {args.origin} session streams found under {root}; nothing measured", file=sys.stderr)
        return 1
    text = json.dumps(profile, indent=1, sort_keys=True) + "\n"
    if args.write:
        path = workload_profile_path(args.origin)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
