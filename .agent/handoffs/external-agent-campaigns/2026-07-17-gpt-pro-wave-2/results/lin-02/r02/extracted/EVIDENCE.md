# EVIDENCE — continuity replay revision r02

## Snapshot authority

`polylogue-overview.json` states:

```text
branch: master
commit: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
dirty: true
generated_at: 2026-07-17T180950Z
```

The bundled Git graph resolves that commit to:

```text
2026-07-17 18:55:47 +0200
fix(repair): harden raw authority convergence (#3046)
```

The supplied branch-delta artifacts report the same commit as the merge base against `origin/master` and contain zero bytes of patch, file-list, and commit-list data. The captured tracked worktree comparison found no source delta. The implementation therefore targets the exact commit and records the dirty marker as unrecoverable metadata state.

## Beads authority

### `polylogue-t8t`

Current title: **Declare continuity replay scenarios and independent known-answer oracles**.

The complete current record requires:

1. seven executable declarations plus the parallel-Claude incident;
2. sparse wording, fact/coverage inventory, independently computed answer, allowed discovery, expected refs, plan equivalence, result semantics, page/cancel/resource bounds, and stop conditions;
3. receipts/transcripts that distinguish source/coverage, discovery/formulation, plan/pushdown, execution/cancellation, projection/rendering, and reasoning failures;
4. explicit incident-call grading;
5. six named mutation families;
6. consumption by `z9gh.7` without duplicating the final live gate.

Later notes materially supersede older shorthand:

- Candidate list: reasonable but physically oversized.
- Exact operator phrase: wrong-corpus assumption.
- Sonnet text query: weak lexical proxy induced by absent structured model/material discovery.
- Sessions-only `query_units` call: product-induced and reasonable under shipped instructions, because the parser requires a terminal unit while installed guidance advertised the nonterminal shape.
- Later correct topology and delegation calls: execution failures.
- Pagination success requires every logical row/node/edge exactly once; a metadata envelope or identical retry is not recovery.
- The canonical declaration location is `polylogue/product/continuity_scenarios.py`, using existing scenario/workflow registries.
- The deterministic oracle must be fixture-derived and not computed through production queries.
- A July 16 external package was rejected because it covered only two of seven jobs and introduced a competing scenario seam.
- The July 17 implementation-readiness note identifies the current scenario, workflow, MCP, and fixture seams as authority.

The record also notes that PR #3018 supplied an initial continuity fixture/known-answer receipt, while the real cold-model/live terminal walk remains with `z9gh.7`.

### `polylogue-z9gh.7`

Current title: **Prove mandate recovery through real agent continuity replays**.

Material acceptance requirements include:

- all seven `t8t` flows as real MCP walks;
- sparse recovery of coordinator `cf0c6474-da22-44be-af3e-666037aa5ea4` and run `wf_54d4fb2e-841`;
- exactly 129 coordinator children partitioned into 91 target-run attempts and 38 other children;
- four Workflow invocations, 50 call keys, 91 attempts, 65 result records, 49 completed keys, one unresolved key, and one final structured result;
- membership proof from run state/journal/metadata/invocations/source refs, not `parent_session_id` alone;
- model/material/call/attempt/effect scope distinctions;
- git/PR/Beads effects with uncertainty;
- lossless paging, cancellation, latency/memory SLOs;
- cold-model success;
- mechanism-removal mutations;
- disposition of every mandate bead.

The synthetic corpus mirrors the corrected counts and separates target/non-target membership. It does not substitute synthetic markers for the live gate's stronger membership/effect evidence.

## Relevant history

The continuity paths first appeared in:

```text
9163d0134 feat(query): bound agent-facing archive reads (#3018)
```

That commit added a declaration prototype, a small `devtools/continuity_replay.py`, one incident fixture, and bounded query infrastructure. At the target snapshot, the continuity runner was not an end-to-end MCP known-answer replay: it classified supplied fixture/observation data rather than constructing a complete all-job archive and driving the public route.

Revision `r01` correctly replaced that prerecorded-observation path with eight synthetic scenarios, independent facts, archive construction, and registered production handlers. Its principal remaining weaknesses were:

- no JSON-RPC/stdio framing;
- no discovery receipt;
- pagination receipts could not independently prove the selected population;
- no exact duplicate-identity invariant;
- no executable six-case original-call curriculum;
- only two anti-vacuity mutations;
- no shipped prompt/parser parity check.

Revision `r02` preserves the sound seam/corpus/oracle work and replaces those weak points.

## Existing scenario seam

The relevant inheritance/composition chain is:

```text
ContinuityScenarioSpec
  -> NamedScenarioSource
    -> ScenarioSpec
      -> ScenarioProjectionSource
      -> ScenarioMetadata
```

`NamedScenarioSource` already supplies stable authored names/descriptions. `ScenarioSpec` already supplies scenario payload/projection composition. Existing query-action workflow IDs are resolved through `polylogue/product/workflows.py` and rejected when unknown. This is why no `polylogue/continuity/scenarios` framework or duplicate workflow registry was introduced.

## Existing fixture seam

`tests/infra/archive_scenarios.py` provides `ArchiveScenario`, `ScenarioMessage`, `ScenarioContentBlock`, native session ID construction, and archive seeding. `tests/infra/storage_records.py` provides `SessionBuilder`, provider/native identity, lineage, role/branch type, and transcript construction.

The production `actions` unit derives facts from paired tool-use and tool-result blocks. The only fixture-seam change adds optional tool-result ID, error flag, and exit code so existing builders can express those production facts. Assertions and model usage are seeded with their established storage primitives.

## Production route evidence

The primary runner starts the real read-role server through `polylogue.mcp.cli` and uses the official MCP SDK's stdio client/session. Discovery and calls therefore exercise framing, initialization, FastMCP registration, production handlers, query parsing/lowering, archive APIs, and SQLite reads.

The public tools used are:

- `query_units`
- `provider_usage`
- `explain_query_expression`
- `list_read_view_profiles`

The route discovery receipt from the fresh replay records:

```text
transport: mcp-stdio-json-rpc
protocol_version: 2025-11-25
server_name: polylogue
server_version: 1.28.1
tool_count: 66
```

## Independent population evidence

Before any MCP call, direct SQLite reads establish:

```text
coordinator children: 129
target-run members: 91
other children: 38
Workflow invocations: 4
final structured results: 1
curriculum cases: 6
input tokens: 1200
output tokens: 300
cached-input tokens: 400
cache-write tokens: 50
total tokens: 1950
```

The target 91 rows encode 50 call keys, 65 result records, 49 completed keys, and one unresolved key. The production route independently rediscovers these counts from archive rows.

## Pagination evidence

For aggregate-capable terminal units, `r02` executes a second production query with `| count`. This does not make the expected answer dependent on the route; the expected answer remains fixture/oracle-owned. It does give the transport harness an independent selected-row total against which to test continuation completeness.

The incident member route produced:

```text
exact selected count: 91
pages: 6
page-local totals: 17, 17, 17, 17, 17, 6
enumerated rows: 91
unique identities: 91
stable query ref: yes
stable result ref: yes
continuation exhausted: yes
population count verified: yes
exact enumeration verified: yes
```

The capped-pseudo-total mutation removes the first continuation but leaves the exact count at 91. The runner sees only 17 enumerated rows and fails with `pagination_count_mismatch`. The identical-call mutation reuses first-page identities under valid second-page metadata and fails with `duplicate_pagination_identity`.

## Incident-grade independence

Raw corpus messages expose only the grading inputs. For example, the sessions-only case states that its query shape was sessions-only, physical size was bounded, corpus match was true, structural discovery was absent, and shipped instruction was advertised. No `grade` or `expected` field is planted in the source message.

The oracle separately expects `product_induced_hidden_grammar`. The runner's classifier derives that grade from the source features. Removing only the `shipped_instruction:advertised` feature changes the derived grade and produces `attempt_grade_mismatch`, proving the expected label is not copied from the route payload.

## Prompt/parser parity finding

Inspection of `polylogue/mcp/server_prompts.py` against `parse_unit_source_expression` found two invalid advertised recipes:

1. `unacknowledged_failures` placed `since` inside an action expression although `since` is a top-level tool argument.
2. `sessions_touching_file` used an unsupported file predicate `repo:` instead of accepted repository/session scope.

Both are corrected. The new parity test executes all shipped prompt functions with `query_units` recipes, extracts the recipes, parses their expressions with the production parser, and checks advertised arguments against the production tool schema. This directly implements the later Beads query-grade correction rather than merely documenting it.

## Contradictions and resolutions

### Dirty metadata versus empty delta

Contradiction: snapshot overview says `dirty=true`; recoverable branch delta is empty and tracked files match the commit.

Resolution: patch the exact named commit and record unknown dirty state as unavailable. Do not invent source.

### Earlier standalone continuity type versus current scenario seam

Contradiction: the historical continuity prototype used a standalone type; current Beads explicitly bind the work to existing scenario protocols and reject a competing seam.

Resolution: subclass `NamedScenarioSource`, validate workflow IDs, and use existing archive fixture builders.

### Parent-child count versus Workflow membership

Contradiction: earlier shorthand conflated coordinator children with target Workflow members; corrected authority states 129 total, 91 target, 38 other.

Resolution: plant, query, and validate all three populations separately. The dropped-filter mutation proves why parent lineage alone is insufficient.

### Advertised sessions-only query versus executable grammar

Contradiction: shipped guidance taught a nonterminal shape while the parser requires a terminal unit.

Resolution: preserve the historical exposure fact in the independent curriculum so the original call is graded product-induced, correct current prompts, and add an executable parity test.

### Cost wording versus billing evidence

Constraint: `provider_usage` reports usage lanes and pricing grain, not a guaranteed currency bill.

Resolution: validate exact reported token counters and pricing grain, not a fabricated monetary amount.

### Reusable catalog versus terminal gate

Constraint: `t8t` owns reusable scenarios/oracles; `z9gh.7` owns the authorized all-green live run.

Resolution: parameterize archive and oracle inputs and document the live invocation, while leaving cold-model/effect/SLO/live claims explicitly unverified.

## Evidence boundary

Supported by this package:

- exact patch application to the named commit;
- eight structured declarations on the existing scenario seam;
- synthetic archive construction through existing fixture/storage primitives;
- independent direct-SQLite facts and separate oracle values;
- official MCP stdio JSON-RPC discovery and calls;
- parser/lowering/archive-backed route execution;
- exact continuation, identity, count-probe, response-hash, and budget receipts;
- six original-call grades and six named mutation diagnoses;
- prompt/parser/schema parity;
- focused and surrounding regression/static checks.

Not supported by this package:

- current operator worktree state beyond the supplied snapshot;
- authorized live archive access;
- live incident effect/provenance proof;
- cold external-model reasoning;
- runner-issued MCP cancellation;
- SSE/HTTP transport;
- live latency/memory SLOs;
- Nix/deployment/full-repository health;
- terminal completion of `polylogue-z9gh.7`.
