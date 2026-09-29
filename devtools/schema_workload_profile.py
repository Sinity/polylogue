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
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path

from polylogue.core.sources import source_for_family
from polylogue.schemas.synthetic.workload import (
    CODEX_TURN_INSTRUCTION_FIELDS,
    PERSISTED_OUTPUT,
    WORKLOAD_PROFILE_KIND,
    WORKLOAD_PROFILE_VERSION,
    classify_claude_code_record,
    classify_codex_record,
    log2_bucket,
    measured_text,
    published_field_names,
    published_kind_tokens,
    record_skeleton,
    result_length,
    template_measures,
    text_measure,
    tool_calls_of,
    tool_result_texts,
    workload_profile_path,
)
from polylogue.sources.parsers.codex import _codex_exec_envelope_outcome, _decoded_json_value, _structural_outcome

#: Source family whose runtime root each origin is measured from by default.
_SOURCE_FAMILIES = {"claude-code": "claude-code-session", "codex": "codex-session"}

Weights = defaultdict[str, float]


def _weights() -> Weights:
    return defaultdict(float)


def _buckets() -> defaultdict[int, float]:
    return defaultdict(float)


class _Templates:
    """Key skeletons, and per-field string and list lengths, of template kinds.

    Kinds are already restricted to public record-type vocabulary by the
    classifiers, and skeleton keys to field names in the committed schema
    package, so every kind observed gets a template.
    """

    def __init__(self, allowed: frozenset[str], values: frozenset[str]) -> None:
        self.allowed = allowed
        self.values = values
        self.skeletons: defaultdict[str, Weights] = defaultdict(_weights)
        #: ``measure -> kind -> field path -> bucket -> weight``.
        self.measures: dict[str, defaultdict[str, defaultdict[str, defaultdict[int, float]]]] = {
            "str": defaultdict(lambda: defaultdict(_buckets)),
            "list": defaultdict(lambda: defaultdict(_buckets)),
            "int": defaultdict(lambda: defaultdict(_buckets)),
            "bool": defaultdict(lambda: defaultdict(_buckets)),
            "float": defaultdict(lambda: defaultdict(_buckets)),
        }

    def add(self, kind: str, record: Mapping[str, object], weight: float) -> None:
        key = json.dumps(
            record_skeleton(record, allowed=self.allowed, values=self.values), sort_keys=True, separators=(",", ":")
        )
        self.skeletons[kind][key] += weight
        for measure, path, length in template_measures(record, allowed=self.allowed):
            self.measures[measure][kind][path][log2_bucket(length)] += weight

    def payload(self) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
        """The skeletons, and each measure's per-kind, per-path histograms."""
        templates: dict[str, object] = {}
        per_path: dict[str, dict[str, object]] = {measure: {} for measure in self.measures}
        for kind, entries in sorted(self.skeletons.items()):
            # Every observed variant: a rare skeleton is the one structure
            # some parser path sees, and it passed the public allowlists.
            ranked = sorted(entries.items(), key=lambda item: (-item[1], item[0]))
            templates[kind] = [{"skeleton": json.loads(key), "weight": _round2(weight)} for key, weight in ranked]
            for measure, by_kind in self.measures.items():
                per_path[measure][kind] = {
                    path: _histogram(buckets) for path, buckets in sorted(by_kind[kind].items()) if _histogram(buckets)
                }
        return templates, per_path


def default_source_root(origin: str) -> Path:
    """The origin's canonical runtime root, as Polylogue itself resolves it."""
    source = source_for_family(_SOURCE_FAMILIES[origin])
    if source is None or source.runtime_root is None:
        raise ValueError(f"{origin} has no runtime root")
    return Path(os.path.expanduser(source.runtime_root))


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


class MalformedSourceError(ValueError):
    """A sampled transcript holds a record that is not one JSON object line."""


def _records(path: Path) -> Iterator[dict[str, object]]:
    """Every record of a JSONL transcript; a malformed line refuses the file.

    A truncated line (a live file read mid-write) or invalid UTF-8 would
    otherwise vanish from the counts, transitions and tails while the profile
    still reports success. Only blank lines are skipped.
    """
    with path.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise MalformedSourceError(
                    f"{path.name} line {number} is not a JSON record ({type(exc).__name__}); profile a quiescent source"
                ) from exc
            if not isinstance(value, dict):
                raise MalformedSourceError(f"{path.name} line {number} is not a JSON object")
            yield value


class _Stream:
    def __init__(self) -> None:
        self.records = _buckets()
        self.start = _weights()
        self.transitions: defaultdict[str, Weights] = defaultdict(_weights)
        self.lengths: defaultdict[str, defaultdict[int, float]] = defaultdict(_buckets)
        self.gaps = _buckets()
        #: This family's own share counters (see :func:`_family_payload`).
        self.shares = _weights()

    def measure(self, origin: str, path: Path, weight: float, templates: _Templates) -> int:
        classify = classify_claude_code_record if origin == "claude-code" else classify_codex_record
        previous_kind: str | None = None
        previous_ms: float | None = None
        count = 0
        #: Open Codex function calls by call id, so each output is counted
        #: against the tool that produced it.
        called: dict[str, str] = {}
        for record in _records(path):
            kind = classify(record)
            count += 1
            if previous_kind is None:
                self.start[kind] += weight
            else:
                self.transitions[previous_kind][kind] += weight
            previous_kind = kind
            calls = tool_calls_of(origin, kind, record)
            if calls:
                # Each call its own length, also per tool: a Read path and a
                # Write body differ by orders of magnitude within one kind.
                for tool, text in calls:
                    self.lengths[kind][log2_bucket(len(text))] += weight
                    self.lengths[f"{kind}:{tool}"][log2_bucket(len(text))] += weight
                if origin == "claude-code":
                    self.lengths[f"{kind}:blocks"][log2_bucket(len(calls))] += weight
                    # Thinking and text beside the calls in the same message:
                    # how many of each (zero included), and each one's length.
                    companions = _companion_blocks(record)
                    for companion in ("thinking", "text"):
                        companion_count = sum(1 for name, _text in companions if name == companion)
                        self.lengths[f"{kind}:{companion}_blocks"][log2_bucket(companion_count)] += weight
                    for companion, companion_text in companions:
                        self.lengths[f"{kind}:{companion}"][log2_bucket(len(companion_text))] += weight
            elif origin == "claude-code" and kind == "user_tool_result":
                # Every result block of the message, not only the first: a
                # short success beside a multi-megabyte persisted output.
                for result_text in tool_result_texts(record):
                    self.lengths[kind][log2_bucket(result_length(result_text))] += weight
            else:
                length = text_measure(origin, kind, record)
                if length is not None:
                    self.lengths[kind][log2_bucket(length)] += weight
            payload = record.get("payload")
            called_tool: str | None = None
            if origin == "codex" and isinstance(payload, Mapping):
                call_id = payload.get("call_id")
                if kind == "function_call" and isinstance(call_id, str) and calls:
                    called[call_id] = calls[0][0]
                elif kind == "function_call_output" and isinstance(call_id, str):
                    called_tool = called.pop(call_id, None)
                if kind == "reasoning":
                    summary = _reasoning_summary(payload)
                    if summary:
                        self.lengths["reasoning:summary"][log2_bucket(len(summary))] += weight
            if origin == "claude-code" and kind == "user_tool_result":
                results = _tool_result_blocks(record)
                if results:
                    self.lengths[f"{kind}:blocks"][log2_bucket(len(results))] += weight
            if kind == "turn_context":
                for field_name in CODEX_TURN_INSTRUCTION_FIELDS:
                    value = payload.get(field_name) if isinstance(payload, Mapping) else None
                    if isinstance(value, str):
                        self.lengths[f"turn_context:{field_name}"][log2_bucket(len(value))] += weight
            if kind.startswith("record:"):
                templates.add(kind, record, weight)
            moment = _parse_ms(record.get("timestamp"))
            if moment is not None:
                if previous_ms is not None and moment >= previous_ms:
                    self.gaps[log2_bucket(moment - previous_ms)] += weight
                previous_ms = moment
            _count_shares(origin, kind, record, self.shares, weight, called_tool=called_tool)
        if count:
            self.records[log2_bucket(count)] += weight
        return count

    def payload(self, origin: str) -> dict[str, object]:
        return {
            **_family_payload(origin, self.shares),
            "records": _histogram(self.records),
            "start": {kind: _round2(weight) for kind, weight in _ranked(self.start)},
            "transitions": {
                kind: {target: _round2(weight) for target, weight in _ranked(row) if _round2(weight) > 0}
                for kind, row in sorted(self.transitions.items())
            },
            "lengths": {kind: _histogram(counter) for kind, counter in sorted(self.lengths.items())},
            "gap_ms": _histogram(self.gaps),
        }


def _companion_blocks(record: Mapping[str, object]) -> list[tuple[str, str]]:
    """``(kind, text)`` of the thinking and text blocks of a Claude tool-call message."""
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    found: list[tuple[str, str]] = []
    for block in content if isinstance(content, list) else ():
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
            found.append(("thinking", str(block["thinking"])))
        elif block.get("type") == "text" and isinstance(block.get("text"), str):
            found.append(("text", str(block["text"])))
    return found


def _ranked(weights: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))


def _reasoning_summary(payload: Mapping[str, object]) -> str:
    """The human-readable summary text of a Codex reasoning item (what production materializes)."""
    summary = payload.get("summary")
    return "".join(
        str(item.get("text") or "")
        for item in (summary if isinstance(summary, list) else ())
        if isinstance(item, Mapping)
    )


def _tool_result_blocks(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return [
        block
        for block in (content if isinstance(content, list) else ())
        if isinstance(block, Mapping) and block.get("type") == "tool_result"
    ]


def _count_shares(
    origin: str,
    kind: str,
    record: Mapping[str, object],
    shares: Weights,
    weight: float,
    *,
    called_tool: str | None = None,
) -> None:
    # Character class is measured on the same fields the lengths are.
    texts = (
        tool_result_texts(record)
        if origin == "claude-code" and kind == "user_tool_result"
        else [text]
        if (text := measured_text(origin, kind, record)) is not None
        else []
    )
    for measured in texts:
        shares["texts"] += weight
        shares[f"texts:{kind}"] += weight
        if not measured.isascii():
            shares["non_ascii_texts"] += weight
            shares[f"non_ascii_texts:{kind}"] += weight
    for tool, _text in tool_calls_of(origin, kind, record):
        shares[f"tool:{tool}" if origin == "claude-code" else f"tool:{kind}:{tool}"] += weight
    if kind == "turn_context":
        payload = record.get("payload")
        shares["turn_contexts"] += weight
        for field_name in CODEX_TURN_INSTRUCTION_FIELDS:
            if isinstance(payload, Mapping) and isinstance(payload.get(field_name), str):
                shares[f"turn_context:{field_name}"] += weight
    if origin == "claude-code":
        message = record.get("message")
        if kind.startswith("assistant_") and isinstance(message, Mapping):
            shares["assistant"] += weight
            if isinstance(message.get("usage"), Mapping):
                shares["assistant_usage"] += weight
        if kind == "user_tool_result":
            # Every result block: its outcome and sidecar evidence each count.
            for block in _tool_result_blocks(record):
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
            # Counted overall and per called tool: an exec envelope belongs
            # to exec tools, not to ``update_plan``.
            suffixes = ("", f":{called_tool}") if called_tool is not None else ("",)
            # The parser's own anchored recognizer, so the shares are what
            # production parsing observes.
            _is_error, exit_code = _codex_exec_envelope_outcome(output)
            for suffix in suffixes:
                shares[f"exec_outputs{suffix}"] += weight
                if exit_code is not None:
                    shares[f"exec_envelopes{suffix}"] += weight
                    if exit_code != 0:
                        shares[f"exec_errors{suffix}"] += weight
        else:
            shares["custom_outputs"] += weight
            # Production's own structural recognizer: key order and leading
            # whitespace do not matter.
            decoded = _decoded_json_value(output) if isinstance(output, str) else output
            is_error, exit_code = _structural_outcome(decoded)
            if exit_code is not None or is_error is not None:
                shares["custom_json"] += weight
                if is_error or (exit_code is not None and exit_code != 0):
                    shares["custom_errors"] += weight
    elif kind == "reasoning":
        payload = record.get("payload")
        shares["reasoning"] += weight
        if isinstance(payload, Mapping) and _reasoning_summary(payload):
            shares["reasoning_summary"] += weight


def _family_payload(origin: str, shares: Mapping[str, float]) -> dict[str, object]:
    """One stream family's shares, character classes and tool mix."""

    def ratio(numerator: str, denominator: str) -> float:
        return round(shares.get(numerator, 0.0) / shares[denominator], 4) if shares.get(denominator) else 0.0

    large = shares.get("sidecar_refs", 0.0) + shares.get("large_inline", 0.0)
    origin_shares = (
        {
            "codex_exec_envelope_share": ratio("exec_envelopes", "exec_outputs"),
            "codex_exec_error_share": ratio("exec_errors", "exec_envelopes"),
            "codex_custom_json_share": ratio("custom_json", "custom_outputs"),
            "codex_custom_error_share": ratio("custom_errors", "custom_json"),
            "codex_reasoning_summary_share": ratio("reasoning_summary", "reasoning"),
            **{
                f"codex_exec_envelope_share:{key.split(':', 1)[1]}": ratio(key.replace("outputs", "envelopes"), key)
                for key in shares
                if key.startswith("exec_outputs:")
            },
            **{
                f"codex_exec_error_share:{key.split(':', 1)[1]}": ratio(
                    key.replace("outputs", "errors"), key.replace("outputs", "envelopes")
                )
                for key in shares
                if key.startswith("exec_outputs:")
            },
            **{
                f"codex_turn_context_{field_name}_share": ratio(f"turn_context:{field_name}", "turn_contexts")
                for field_name in CODEX_TURN_INSTRUCTION_FIELDS
            },
        }
        if origin == "codex"
        else {
            "assistant_usage_share": ratio("assistant_usage", "assistant"),
            "tool_error_share": ratio("tool_errors", "tool_results"),
            "sidecar_share_of_large": round(shares.get("sidecar_refs", 0.0) / large, 4) if large else 0.0,
        }
    )
    kinds = sorted({key.split(":", 1)[1] for key in shares if key.startswith("texts:")})
    return {
        "shares": {"non_ascii_text_share": ratio("non_ascii_texts", "texts"), **origin_shares},
        "non_ascii_by_kind": {kind: ratio(f"non_ascii_texts:{kind}", f"texts:{kind}") for kind in kinds},
        "tool_names": {
            key.split(":", 1)[1]: _round2(weight) for key, weight in _ranked(shares) if key.startswith("tool:")
        },
    }


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


def _first_record(path: Path) -> dict[str, object] | None:
    """The first record :func:`_records` would measure (blank lines skipped)."""
    return next(_records(path), None)


def _codex_is_legacy(path: Path) -> bool:
    """A rollout in the pre-envelope flat format (records without ``payload``).

    Decided from the first record the profile would measure, so a leading
    blank line does not classify an envelope rollout as legacy and drop it.
    An unreadable file raises rather than being skipped as legacy.
    """
    first = _first_record(path)
    return not (first is not None and isinstance(first.get("payload"), dict))


def _codex_parent(path: Path) -> str | None:
    first = _first_record(path)
    payload = first.get("payload") if first is not None else None
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
) -> tuple[defaultdict[int, float], defaultdict[int, float], defaultdict[int, float], float]:
    """Subagent fan-out per main session, the nested-spawn topology, orphans per main session.

    A nested spawn is a subagent whose parent is itself a subagent; an orphan
    is one whose parent is not among the measured sessions at all. Nesting
    is measured as two finite distributions: the nested descendants below
    each first-level subagent, and the spawns of each subagent in those
    nested trees. A mean nesting rate cannot describe a long finite chain:
    rounded, it reads as every subagent spawning one more, forever.
    """
    fanout = _buckets()
    if origin == "claude-code":
        parents = {path: path.parent.parent.name for path in families["subagent"] if path.parent.name == "subagents"}
        main_ids = {main.stem for main in families["main"]}
        subagent_of = {path: path.stem for path in parents}
    else:
        parents = {path: parent for path in families["subagent"] if (parent := _codex_parent(path))}
        main_ids = {path.stem[-36:] for path in families["main"]}
        subagent_of = {path: path.stem[-36:] for path in families["subagent"]}
    subagent_ids = set(subagent_of.values()) if origin != "claude-code" else set()
    per_parent: defaultdict[str, int] = defaultdict(int)
    children: defaultdict[str, list[str]] = defaultdict(list)
    for path, parent in parents.items():
        per_parent[parent] += 1
        if parent in subagent_ids and parent not in main_ids:
            children[parent].append(subagent_of[path])
    for session_id in main_ids:
        fanout[log2_bucket(per_parent.get(session_id, 0))] += 1
    orphans = sum(
        count for parent, count in per_parent.items() if parent not in main_ids and parent not in subagent_ids
    )
    descendants = _buckets()
    spawns = _buckets()
    for path, parent in parents.items():
        if parent in subagent_ids and parent not in main_ids:
            continue
        # A first-level subagent: walk its nested tree without recursion (a
        # measured chain can be tens of thousands deep).
        count = 0
        queue = [subagent_of[path]]
        seen = set(queue)
        while queue:
            node = queue.pop()
            below = [child for child in children.get(node, ()) if child not in seen]
            spawns[log2_bucket(len(below))] += 1
            count += len(below)
            seen.update(below)
            queue.extend(below)
        descendants[log2_bucket(count)] += 1
    return fanout, descendants, spawns, orphans / max(1, len(main_ids))


def measure(origin: str, root: Path, *, sample: int, tail: int, seed: int) -> dict[str, object]:
    families = _stream_families(origin, root)
    templates = _Templates(published_field_names(origin), published_kind_tokens())
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
            stream.measure(origin, path, 1.0 if path in tail_paths else weight, templates)
            if index % 200 == 0:
                print(f"  {origin}/{name}: {index + 1}/{len(tail_paths) + len(drawn)}", file=sys.stderr, flush=True)
        streams[name] = stream.payload(origin)

    fanout, nested_descendants, nested_spawns, orphans_per_session = _fanout(origin, root, families)
    template_payload, per_path = templates.payload()
    return {
        "kind": WORKLOAD_PROFILE_KIND,
        "version": WORKLOAD_PROFILE_VERSION,
        "origin": origin,
        "source_bytes": int(_round2(source_bytes)),
        "main_sessions": int(_round2(len(families.get("main", ())))),
        "streams": streams,
        "subagents_per_session": _histogram(fanout),
        "nested_descendants_per_subagent": _histogram(nested_descendants),
        "nested_spawns_per_subagent": _histogram(nested_spawns),
        "shares": {
            "orphan_subagents_per_session": round(orphans_per_session, 4),
        },
        "templates": template_payload,
        "template_strings": per_path["str"],
        "template_lists": per_path["list"],
        "template_ints": per_path["int"],
        "template_bools": per_path["bool"],
        "template_floats": per_path["float"],
    }


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure an aggregate-only synthetic workload profile.")
    parser.add_argument("--origin", required=True, choices=sorted(_SOURCE_FAMILIES))
    parser.add_argument("--source", type=Path, help="Source root (defaults to the origin's usual location).")
    parser.add_argument("--sample", type=_positive, default=1500, help="Uniformly sampled files per stream family.")
    parser.add_argument("--tail", type=_non_negative, default=5, help="Largest files per family always measured.")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--write", action="store_true", help="Write the committed profile instead of printing it.")
    args = parser.parse_args(argv)
    root = (args.source or default_source_root(args.origin)).resolve()
    profile = measure(args.origin, root, sample=args.sample, tail=args.tail, seed=args.seed)
    streams = profile.get("streams")
    main_stream = streams.get("main") if isinstance(streams, dict) else None
    if not isinstance(main_stream, dict) or not main_stream.get("records"):
        print(f"no {args.origin} session records measured under {root}; refusing to write", file=sys.stderr)
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
