# Session owner operations

Polylogue owns indexed session listing, lexical search, transcript reads, orchestration evidence, resume context, event timelines, and explicit reads of original local session JSONL files. These operations share typed requests and result contracts in `polylogue/operations/session_contracts.py`. The archive operations run through controlled reads; raw fallback never ingests a file or assigns an archive session identity.

Generate exact per-operation JSON Schemas with:

```sh
python -m polylogue.cli.session_operations contracts
```

The manifest maps each operation to its request, result, error, authority, effects, and MCP invocation. It comes from the same Pydantic models that validate execution. Consumers should generate their inputs from these contracts rather than reflecting the whole MCP dispatcher signature.

The existing MCP `query` tool accepts a discriminated `session_operation` request:

```json
{"projection":"session-operations","session_operation":{"operation":"sessions.list","origin":"codex-session","limit":20}}
```

The same owner operation is available over the machine CLI. It reads one bounded JSON request from stdin and writes one JSON result:

```sh
printf '%s' '{"operation":"sessions.list","limit":20}' | python -m polylogue.cli.session_operations execute
```

| Operation | Scope and result |
| --- | --- |
| `sessions.list` | Filtered indexed session summaries |
| `sessions.search` | Lexical search over indexed session content |
| `sessions.read` | A page of lineage-composed messages for a session ref |
| `sessions.orchestration` | Retained launch, model, usage, and topology evidence |
| `context.resume` | Ranked resume candidates, lineage, project state, and assertion guidance |
| `sessions.timeline` | Cross-session messages and semantic session events ordered by recorded event time |
| `sessions.raw.list` | Original session files ordered by filesystem modification time |
| `sessions.raw.search` | Bounded literal search of original JSONL bytes |
| `sessions.raw.read` | Bounded byte read of one original reference |
| `sessions.raw.timeline` | Bounded merge of original files by filesystem modification time |
| `memory.raw.search` | Bounded literal-search fanout across original session sources |
| `memory.raw.get` | The same original byte read as `sessions.raw.read` |

Indexed pages return `continuation`, bound to the archive frame, operation, filters, and ordering. Resume with the operation and continuation, plus `ref` for transcript reads. Conflicting arguments or a changed archive frame are rejected. The existing `query(projection="sessions")` surface also accepts these continuations. `query(projection="timeline")` exposes the event timeline.

Timeline time bounds are timezone-bearing ISO timestamps. Indexed events lacking timestamps are excluded from placement and reported as a coverage gap. A file's modification time is never substituted for a missing event timestamp. Message rows carry their stable message reference. Semantic event rows carry their session reference and exact event ID, resolvable through the session events projection. Both carry a recorded timestamp and bounded text.

Raw fallback is explicit. Its only public origins are `claude-code-session` and `codex-session`; original references retain their acquisition spelling, such as `claude-code:project/example.jsonl` or `codex:2026/example.jsonl`. Raw results always report `indexed_session_id: null`. That means the operation has not established an indexed correspondence. `mtime_ns` is filesystem evidence, and only raw list/timeline operations use it for time ordering. Missing roots remain unavailable sources with coverage gaps.

Raw search has a per-source byte budget and bounded result count. A returned continuation advances the same source observation, including scans that found no matches before exhausting the byte budget. Source changes invalidate raw continuations. Raw reads return `next_offset` measured in consumed bytes and preserve UTF-8 boundaries. Caller input cannot change owner source roots.

Resume context uses the existing context compiler and can record a disposable scheduler receipt in the ops tier. It does not grant archive mutation capability. Contract generation requires no archive access.
