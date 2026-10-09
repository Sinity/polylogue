# Session owner operations

Polylogue owns indexed session listing, lexical search, transcript reads, orchestration evidence, resume context, event timelines, and explicit reads of original local session JSONL files. These operations share typed requests and result contracts in `polylogue/operations/session_contracts.py`. The archive operations run through controlled reads; raw fallback never ingests a file or assigns an archive session identity.

## A second declared read executor, on purpose

Indexed session reads have two declared executors, and this family is the second one. `polylogue/operations/daemon_reads.py:execute_read_operation` answers the generic, dict-keyed read operations (`cli.query`, `session.read`, `facets`, …) reached by the CLI, the daemon transport and the HTTP client. `polylogue/operations/session_reads.py:execute_session_operation` answers this typed family, and is reached by the MCP `query` tool and by `python -m polylogue.cli.session_operations execute`. Nothing under `polylogue/mcp/` calls the generic executor.

The split is deliberate, not drift. This family declares per-operation request, result, error, authority and effects contracts generated from Pydantic models; the generic path carries untyped payload dictionaries keyed by an operation name. Half of the operations below — every `sessions.raw.*` and `memory.raw.*` read — answer from original local JSONL bytes with no archive authority at all, which the generic executor cannot express: it requires a pinned `ArchiveStore` snapshot and keys its result cache on the index generation. Collapsing this family into the generic read path would discard the contracts this document exists to declare.

What must not diverge is the *mechanics* a read shares regardless of executor. `polylogue/operations/transcript_window.py` already owns window arithmetic, snapshot binding, epoch validation and continuation vocabulary for the transcript window, and `sessions.read` projects onto it rather than deciding a window itself. Where the two executors answer the same question — which sessions a filter selects, in what order, how many exist, where the next page starts — they are required to answer it identically, and that requirement is discharged by comparing the two executors against each other on one seeded archive (`tests/unit/operations/test_session_owner_route_equivalence.py`), not by driving two surfaces through one function.

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
| `sessions.list` | Filtered indexed session summaries (default limit matches `DEFAULT_SESSION_LIST_LIMIT`, 20) |
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

Indexed pages retain the facade’s explicitly selected Index and return `continuation`, bound to the opened physical archive and Index generation, relation currency, operation, filters, and ordering. Compiled expressions decide the initial window and ordering; `latest` selects at most one session and has no continuation. Owner pages require deterministic ordering; random and sampled selections use the sampled query operation. Missing session dates sort last in either direction. Resume with the operation and continuation, plus `ref` for transcript reads. Conflicting arguments or a changed archive frame are rejected. The existing `query(projection="sessions")` surface also accepts these continuations. When `limit` is omitted, both owner and generic routes use `DEFAULT_SESSION_LIST_LIMIT` (20), preserving the same first-page boundary. A resumed page keeps its continuation's page size when `limit` is omitted; an explicit smaller `limit` narrows that page, and a larger one is refused as widening the bound window. A continuation minted with transcript filters resumes under those filters without restating them. `query(projection="timeline")` exposes the event timeline.

The machine session adapter exits with the shared terminal codes: `ok` 0, `empty` 2, and `degraded` or `error` 1.

Timeline time bounds are timezone-bearing ISO timestamps. Indexed events lacking timestamps are excluded from placement and reported as a coverage gap. A file's modification time is never substituted for a missing event timestamp. Message rows carry their stable message reference. Semantic event rows carry their session reference and exact event ID, resolvable through the session events projection. Both carry a recorded timestamp and bounded text.

Raw fallback is explicit. Its only public origins are `claude-code-session` and `codex-session`; original references retain their acquisition spelling, such as `claude-code:project/example.jsonl` or `codex:2026/example.jsonl`. Raw results always report `indexed_session_id: null`. That means the operation has not established an indexed correspondence. `mtime_ns` is filesystem evidence, and only raw list/timeline operations use it for time ordering. Missing roots remain unavailable sources with coverage gaps.

Raw search has a per-source byte budget and bounded result count. A returned continuation advances the same source observation, including scans that found no matches before exhausting the byte budget. A raw search continuation names a retained snapshot of its selected file population (relative paths and stat identity only, never content) kept as a private per-user state file for one hour after its last use, at most 64 in total with the least recently used evicted first; the token itself carries only that handle and a scan position. Files created after the first page are excluded, and an already-scanned file that later changes is irrelevant. A selected file that vanishes, changes before it is searched, or races with its read is skipped and named in `coverage.gaps` (`degraded`), and a final page still reports files skipped on earlier pages. An expired or evicted handle returns a `degraded` page with no continuation; a pre-snapshot token or a continuation replayed under a different reference filter is refused as `stale_continuation`. List and timeline continuations bind the observation of their whole original enumeration: a change to any enumerated file, including one already emitted, refuses the next page as `stale_continuation`, except that a head already held in the continuation is emitted with the observation it was enumerated under. A search continuation whose retained snapshot was taken for a different scope (for example a changed source root) is also refused as `stale_continuation`. A snapshot that cannot be read because of a transient system error (for example EMFILE or EIO) is refused as `retryable`, and the same continuation can be retried. Raw reads return `next_offset` measured in consumed bytes and preserve UTF-8 boundaries.

Literal search is greedy and non-overlapping over UTF-8 bytes. Its drained match sequence is independent of `scan_bytes` and result-page size. Each hit exposes the half-open byte span `match_offset`/`match_end`; `offset` remains the start of its evidence snippet and `line` the one-based line containing the match start. Snippet context is selected relative to that match, not to a scan block. Search tokens created before the non-overlap position was retained are refused as `stale_continuation` rather than resumed under different matching semantics.

Raw byte reads default to **live** semantics (`consistency: "live"`): each call observes the source anew, so tailers can read newly appended bytes. Every successful page also returns an opaque `observation` witness. To assemble coherent evidence, pass that first page's witness as `expected_observation` on every following request; successful responses then report `consistency: "bound"`. A changed observation produces the typed `source_changed` error, never a successful mixed-version page. The witness binds the configured root, canonical reference and opened file's device, inode, size, mtime and ctime. It is a stat witness, not a content digest or historical snapshot: rewrites, replacements, truncations and appends invalidate a bound read. Continue append-only tailing without `expected_observation`; do not treat such live pages as one fixed observation. The owner validates the descriptor before and after each read, and both `sessions.raw.read` and `memory.raw.get` preserve the same fields and refusal. Caller input cannot change owner source roots.

Resume context uses the existing context compiler and can record a disposable scheduler receipt in the ops tier. It does not grant archive mutation capability. Contract generation requires no archive access.
