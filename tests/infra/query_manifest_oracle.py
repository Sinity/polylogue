"""Independent provider-wire facts for the query cardinality survivor.

The manifest in this module is deliberately test-owned.  It renders Codex
JSONL inputs and computes the expected action relation directly from those
planted facts; it never asks Polylogue's parser, query compiler, SQL views, or
repository for an expected value.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

QUERY_CARDINALITY_TOKEN = "testdiet-cardinality-law"

ActionIdentity: TypeAlias = tuple[str, str, str | None, int | None, int | None]


@dataclass(frozen=True)
class PlantedCodexCall:
    """One provider-native function call planted in a Codex JSONL source."""

    call_id: str
    command: str
    timestamp: str


@dataclass(frozen=True)
class PlantedCodexResult:
    """One provider-native function result, including orphan-result cases."""

    call_id: str
    output: str
    exit_code: int
    timestamp: str

    @property
    def wire_output(self) -> str:
        return json.dumps(
            {"metadata": {"exit_code": self.exit_code}, "output": self.output},
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class PlantedCodexSession:
    """Provider facts for one source file and its explicit canonical identity."""

    native_session_id: str
    canonical_session_id: str
    timestamp: str
    calls: tuple[PlantedCodexCall, ...]
    results: tuple[PlantedCodexResult, ...]

    def wire_records(self) -> tuple[dict[str, object], ...]:
        records: list[dict[str, object]] = [
            {
                "type": "session_meta",
                "payload": {
                    "id": self.native_session_id,
                    "timestamp": self.timestamp,
                },
            }
        ]
        for index, call in enumerate(self.calls):
            records.append(
                {
                    "type": "response_item",
                    "timestamp": call.timestamp,
                    "payload": {
                        "type": "function_call",
                        "id": f"fc-{self.native_session_id}-{index}",
                        "call_id": call.call_id,
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": call.command}, sort_keys=True, separators=(",", ":")),
                    },
                }
            )
        for index, result in enumerate(self.results):
            records.append(
                {
                    "type": "response_item",
                    "timestamp": result.timestamp,
                    "payload": {
                        "type": "function_call_output",
                        "id": f"fco-{self.native_session_id}-{index}",
                        "call_id": result.call_id,
                        "output": result.wire_output,
                    },
                }
            )
        return tuple(records)


@dataclass(frozen=True)
class ExpectedActionFact:
    """The intended ordinal pairing of one planted use with its result."""

    session_id: str
    command: str
    occurred_at_ms: int
    output_text: str | None
    is_error: int | None
    exit_code: int | None

    @property
    def identity(self) -> ActionIdentity:
        return (
            self.session_id,
            self.command,
            self.output_text,
            self.is_error,
            self.exit_code,
        )


@dataclass(frozen=True)
class QueryCardinalityManifest:
    """Fixed known-answer corpus for membership and cardinality laws."""

    sessions: tuple[PlantedCodexSession, ...]

    def write_sources(self, root: Path) -> tuple[Path, ...]:
        root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for session in self.sessions:
            path = root / f"{session.native_session_id}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in session.wire_records()
                ),
                encoding="utf-8",
            )
            paths.append(path)
        return tuple(paths)

    def all_actions(self) -> tuple[ExpectedActionFact, ...]:
        """Pair the nth call/result sharing an ID, directly from planted facts."""
        actions: list[ExpectedActionFact] = []
        for session in self.sessions:
            result_groups: dict[str, list[PlantedCodexResult]] = defaultdict(list)
            for planted_result in session.results:
                result_groups[planted_result.call_id].append(planted_result)
            call_ranks: Counter[str] = Counter()
            for call in session.calls:
                call_rank = call_ranks[call.call_id]
                call_ranks[call.call_id] += 1
                matching_results = result_groups.get(call.call_id, [])
                paired_result = matching_results[call_rank] if call_rank < len(matching_results) else None
                actions.append(
                    ExpectedActionFact(
                        session_id=session.canonical_session_id,
                        command=call.command,
                        occurred_at_ms=_epoch_ms(call.timestamp),
                        output_text=paired_result.wire_output if paired_result is not None else None,
                        is_error=(int(paired_result.exit_code != 0) if paired_result is not None else None),
                        exit_code=paired_result.exit_code if paired_result is not None else None,
                    )
                )
        return tuple(actions)

    def matching_actions(self) -> tuple[ExpectedActionFact, ...]:
        """Evaluate the survivor's fixed public predicate without production code."""
        return tuple(
            sorted(
                (action for action in self.all_actions() if QUERY_CARDINALITY_TOKEN in action.command),
                key=lambda action: (action.occurred_at_ms, action.identity),
            )
        )

    def matching_action_identities(self) -> tuple[ActionIdentity, ...]:
        return tuple(action.identity for action in self.matching_actions())

    def matching_session_ids(self) -> tuple[str, ...]:
        selected = {action.session_id for action in self.matching_actions()}
        return tuple(
            session.canonical_session_id for session in self.sessions if session.canonical_session_id in selected
        )

    def is_error_partition(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for action in self.matching_actions():
            key = "unknown" if action.is_error is None else str(action.is_error)
            counts[key] += 1
        return dict(counts)

    @property
    def decoy_session_id(self) -> str:
        return "codex-session:testdiet-query-decoy"


def query_cardinality_manifest() -> QueryCardinalityManifest:
    """Return the stable micro-corpus layered onto the realized C-03 archive."""
    return QueryCardinalityManifest(
        sessions=(
            PlantedCodexSession(
                native_session_id="testdiet-query-alpha",
                canonical_session_id="codex-session:testdiet-query-alpha",
                timestamp="2026-07-01T00:00:00Z",
                calls=(
                    PlantedCodexCall(
                        call_id="shared-duplicate-id",
                        command=f"printf {QUERY_CARDINALITY_TOKEN}-alpha-one",
                        timestamp="2026-07-01T00:00:01Z",
                    ),
                    PlantedCodexCall(
                        call_id="shared-duplicate-id",
                        command=f"printf {QUERY_CARDINALITY_TOKEN}-alpha-two",
                        timestamp="2026-07-01T00:00:02Z",
                    ),
                    PlantedCodexCall(
                        call_id="missing-result",
                        command=f"printf {QUERY_CARDINALITY_TOKEN}-alpha-missing",
                        timestamp="2026-07-01T00:00:03Z",
                    ),
                    PlantedCodexCall(
                        call_id="alpha-decoy",
                        command="printf alpha-command-outside-law",
                        timestamp="2026-07-01T00:00:04Z",
                    ),
                ),
                results=(
                    PlantedCodexResult(
                        call_id="shared-duplicate-id",
                        output="alpha-one",
                        exit_code=0,
                        timestamp="2026-07-01T00:00:10Z",
                    ),
                    PlantedCodexResult(
                        call_id="shared-duplicate-id",
                        output="alpha-two",
                        exit_code=2,
                        timestamp="2026-07-01T00:00:11Z",
                    ),
                    PlantedCodexResult(
                        call_id="alpha-decoy",
                        output="alpha-decoy",
                        exit_code=0,
                        timestamp="2026-07-01T00:00:12Z",
                    ),
                    PlantedCodexResult(
                        call_id="orphan-result",
                        output=f"{QUERY_CARDINALITY_TOKEN}-orphan-must-not-be-an-action",
                        exit_code=7,
                        timestamp="2026-07-01T00:00:13Z",
                    ),
                ),
            ),
            PlantedCodexSession(
                native_session_id="testdiet-query-beta",
                canonical_session_id="codex-session:testdiet-query-beta",
                timestamp="2026-07-01T00:01:00Z",
                calls=(
                    PlantedCodexCall(
                        call_id="beta-success",
                        command=f"printf {QUERY_CARDINALITY_TOKEN}-beta-success",
                        timestamp="2026-07-01T00:00:05Z",
                    ),
                    PlantedCodexCall(
                        call_id="beta-error",
                        command=f"printf {QUERY_CARDINALITY_TOKEN}-beta-error",
                        timestamp="2026-07-01T00:00:06Z",
                    ),
                ),
                results=(
                    PlantedCodexResult(
                        call_id="beta-success",
                        output="beta-success",
                        exit_code=0,
                        timestamp="2026-07-01T00:01:10Z",
                    ),
                    PlantedCodexResult(
                        call_id="beta-error",
                        output="beta-error",
                        exit_code=3,
                        timestamp="2026-07-01T00:01:11Z",
                    ),
                ),
            ),
            PlantedCodexSession(
                native_session_id="testdiet-query-decoy",
                canonical_session_id="codex-session:testdiet-query-decoy",
                timestamp="2026-07-01T00:02:00Z",
                calls=(
                    PlantedCodexCall(
                        call_id="decoy-output-only",
                        command="printf output-only-lookalike",
                        timestamp="2026-07-01T00:00:07Z",
                    ),
                ),
                results=(
                    PlantedCodexResult(
                        call_id="decoy-output-only",
                        output=f"{QUERY_CARDINALITY_TOKEN}-appears-only-in-output",
                        exit_code=0,
                        timestamp="2026-07-01T00:02:10Z",
                    ),
                ),
            ),
        )
    )


def _epoch_ms(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


__all__ = [
    "ActionIdentity",
    "ExpectedActionFact",
    "PlantedCodexCall",
    "PlantedCodexResult",
    "PlantedCodexSession",
    "QUERY_CARDINALITY_TOKEN",
    "QueryCardinalityManifest",
    "query_cardinality_manifest",
]
