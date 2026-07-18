# Evidence ledger

## Scope and authority

This ledger grounds the README draft and positioning kit in the supplied Polylogue snapshot.

Authority order used:

1. current source and generated artifacts at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`;
2. repository instructions and architecture contracts;
3. complete relevant Beads records from `polylogue-beads-export.jsonl`;
4. older plans and prose only where they do not conflict with current source or executable evidence.

The working-tree archive excluded `.agent/`, but the snapshot included a dedicated `.agent` export. Demo-shelf claims use that dedicated export. The analysis did not access a live browser, deployment, hosted release channel, provider account, provider invoice, or the operator’s live archive. Private-archive numbers below come only from committed aggregate artifacts.

Classification:

- **Observed:** read directly from source, generated artifact, or a command executed against the snapshot.
- **Source-supported inference:** follows from multiple observed contracts but was not itself executed end to end.
- **Unresolved:** evidence conflicts or a parallel lane is absent.
- **Recommendation:** proposed copy or integration action.

## Snapshot identity and command validation

| Item | Result | Classification | Provenance |
|---|---|---|---|
| Git revision | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` | Observed | `git rev-parse HEAD` in reconstructed snapshot |
| Package version | `0.2.0+536a53ef-dirty` | Observed | `polylogue --version` in editable snapshot environment |
| Console scripts | `polylogue`, `polylogued`, `polylogue-mcp` | Observed | `pyproject.toml:[project.scripts]`; each `--help` executed |
| Python floor | Python `>=3.11` | Observed | `pyproject.toml`; current README badge |
| Direct demo seed | 15 sessions, 62 messages, 37 declared construct checks passing | Observed | `polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json`; `polylogue/demo/seed.py`; `polylogue/demo/constructs.py` |
| Direct demo verification | `ok=true`; no absolute path leaks | Observed | `polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json` |
| Demo lexical find | 13 hits for `pytest` in direct seeded archive | Observed | `polylogue --plain find pytest` |
| Exact receipt read | Failed test, claim, edit, later pass rendered from one exact ref | Observed | `polylogue --plain find id:codex-session:demo-receipts then read --view messages` |
| Structured failure aggregate | `Bash=4`, `exec_command=2`, `Edit=1` in direct seeded archive | Observed | <code>polylogue --plain 'actions where is_error:true &#124; group by tool &#124; count'</code> |
| Receipt command | Verdict `contradicted_at_claim_time_then_repaired`; failed exit `1`; later exit `0`; prose “error” hits `2`; structurally failed controls `0` | Observed | `polylogue demo receipts --compact`; `polylogue/demo/receipts.py` |
| Claim-vs-evidence seeded reproduction | 7 structured failures classified in direct demo; artifact family generated | Observed | `python -m devtools workspace claim-vs-evidence --archive-root "$POLYLOGUE_ARCHIVE_ROOT" --limit 5000 --out-dir "$POLYLOGUE_CVE_OUT" --json` |
| Provider completeness registry | Eight accepted provider/capture packages plus one proposed browser-capture mode | Observed | `python -m devtools lab provider completeness --json`; `polylogue/sources/provider_completeness.py:54-205` |
| Documentation command gate | Passes: 87 documentation files scanned, no stale commands | Observed | `python -m devtools verify doc-commands` |
| Provider completeness gate | Passes: 9 rows, 8 accepted/complete, 1 proposed, 0 accepted blocked | Observed | `python -m devtools lab provider completeness --check` |
| Generated-surface gate | Passes for CLI schemas, OpenAPI, devtools, demo datasheet, workflow/docs surfaces, MCP equivalence/index, topology, and pages | Observed | `python -m devtools render all --check` |
| Demo shelf index gate | Fails: manifest, summary index, shelf README, and curated catalog drift | Observed | `python -m devtools workspace demo-shelf --check`, executed from the dedicated `.agent/demos` export with the source checkout on `PYTHONPATH` |
| Demo packet registry | Fails in the dedicated `.agent` export: three packets lack `run.log` and have stale receipt hashes | Observed | `python -m devtools.verify_demo_packet_registry`, executed from the dedicated `.agent/demos` export with the source checkout on `PYTHONPATH` |
| Daemon demo import | Timed out semantic verification | Observed | Fresh temp archive, live `polylogued run`, then `polylogue import --demo --wait` |
| Daemon-ingested state | 15 sessions, 60 messages; verifier false | Observed | `polylogue demo verify` against daemon-ingested archive |

## README feature-claim ledger

| ID | README claim | Classification | Source provenance | Demonstrating command or executable contract |
|---|---|---|---|---|
| P-01 | Polylogue is a local archive for AI sessions. | Observed | `docs/architecture.md:12-17`; current `README.md:12-22` | `polylogue config paths --format json`; `polylogue ops status --format json` |
| P-02 | It is a forensics layer over normalized messages, actions, outcomes, lineage, and usage. | Observed | `docs/schema.md:57-70`; `docs/search.md`; `docs/cost-model.md`; actions view in `polylogue/storage/sqlite/archive_tiers/index.py` | `polylogue --plain find id:codex-session:demo-receipts then read --view messages`; `polylogue analyze usage --detail headline --format json --limit 0` |
| P-03 | Imports are daemon-scheduled and do not have a silent success path. | Observed | `polylogue/cli/commands/import_command.py:1-22, 227-234` | `polylogue import --demo --wait --timeout 30` prints scheduling acknowledgment and then fails honestly when semantic verification does not converge |
| P-04 | One daemon process owns writes. | Observed | `CLAUDE.md` and `AGENTS.md` daemon/write doctrine; daemon storage architecture | `polylogued run`; `polylogue ops status --full --format json` |
| P-05 | Source evidence and reviewed user state are durable; index, embeddings, and ops state are rebuildable/disposable by tier contract. | Observed | `docs/architecture.md:53-56`; `docs/schema.md:38-113` | `polylogue config paths --format json`; `polylogue ops reset --help` |
| P-06 | Ingest identity uses SHA-256 over NFC-normalized payloads and idempotent upsert-if-changed semantics. | Observed in source; end-to-end repeat import remains a merge gate | `docs/architecture.md:120-138`; `CLAUDE.md` hashing doctrine; `polylogue/core/hashing.py` | README fallback `polylogue import "${TMPDIR:-/tmp}/polylogue-fallback-demo.json" --explain --format json`; QA-01 requires duplicate-free repeat import |
| Q-01 | `pipx install polylogue` is a documented install route and the package exposes three scripts. | Observed in source, not release-smoked here | `docs/installation.md:9-23`; `pyproject.toml:111-117` | `pipx install polylogue`; `polylogue --version`; `polylogued --help`; `polylogue-mcp --help` |
| Q-02 | `polylogue import --demo --wait` is a registered CLI path that stages, schedules, waits, and verifies. | Observed contract, failing current execution | `polylogue/cli/commands/import_command.py:89-131, 170-235, 322-333` | `polylogue import --demo --wait --timeout 30` |
| Q-03 | Exact-ref read is supported. | Observed | root CLI help; generated `docs/cli-reference.md`; exact-ref cardinality tests | `polylogue find id:codex-session:demo-receipts then read --view messages` |
| Q-04 | Action queries can group structurally failed actions by tool. | Observed | `docs/search.md` action-unit grammar; action query implementation | <code>polylogue 'actions where is_error:true &#124; group by tool &#124; count'</code> |
| S-01 | Default text retrieval is lexical dialogue FTS. | Observed | `docs/search.md:727-741` | `polylogue --lexical find pytest` or ordinary `polylogue find pytest` |
| S-02 | Semantic search is explicit and requires embeddings. | Observed | `docs/search.md:592-605, 692-725, 727-748`; `docs/schema.md:72-78` | `polylogue ops embed status`; `polylogue --semantic find 'prompt'` |
| S-03 | Hybrid search is explicit. | Observed | `docs/search.md:733-738`; root CLI `--retrieval-lane` help | `polylogue --retrieval-lane hybrid find pytest` |
| S-04 | FTS does not index `Write.content` or `Edit.old_string/new_string`. | Observed | `docs/search.md:927-982`; `polylogue/storage/sqlite/archive_tiers/index.py`; drift tests | `polylogue 'actions where tool:edit AND text:"SharedClock"'` exercises the unindexed action-evidence workaround against the demo corpus |
| A-01 | Architecture has four rings. | Observed | `docs/architecture.md:12-56` | Diagram in architecture doc; `python -m devtools render all --check` guards generated surfaces |
| A-02 | Local storage has five SQLite tiers. | Observed | `docs/schema.md:5-31, 38-113` | `polylogue config paths --format json` |
| A-03 | Derived index rebuild is explicit through the authority-safe maintenance command. | Observed; docs conflict | `polylogue/cli/commands/reset.py:476-544`; `polylogue/cli/commands/maintenance/_rebuild_index.py`; executed reset and rebuild | `polylogue ops maintenance rebuild-index --plan --output-format json`; after reset, `polylogue ops maintenance rebuild-index --output-format json` |
| A-04 | Large source content uses a content-addressed blob store. | Observed | `docs/schema.md:40-55`; current `README.md:138` | `polylogue config paths --format json` exposes blob-store root |
| M-01 | `polylogue-mcp` is standalone, not a `polylogue` subcommand. | Observed | `docs/mcp-reference.md:44-61`; `pyproject.toml`; runtime help | `polylogue-mcp --help` |
| M-02 | Runtime roles are `read`, `write`, `review`, `admin`. | Observed | `polylogue/mcp/cli.py:15`; runtime help; role-gated registration in `polylogue/mcp/server_tools.py:1429-1433` | `polylogue-mcp --help` |
| M-03 | Generated reference currently enumerates 104 tool names. | Observed | `docs/mcp-reference.md:66-71`; generated registry | `python -m devtools render all --check` validates generated surface |
| M-04 | Six-tool replacement exists. | Unresolved, absent from snapshot | User-supplied parallel-lane context; Beads `polylogue-t46.1` and `polylogue-t46.8` support simplification direction but do not evidence six tools | Fill only after landed registry and tests |
| L-01 | Trusted single-user host; not multi-user isolation. | Observed | `SECURITY.md:3-10`; `docs/security.md` | `polylogue ops status --full --format json` shows local endpoint/state, not isolation proof |
| L-02 | OS is responsible for encryption at rest. | Observed | `SECURITY.md:7-10` | Host configuration, outside Polylogue command surface |
| L-03 | Pre-1.0 with no LTS branch. | Observed | `SECURITY.md:18-21`; package version | `polylogue --version` |
| C-01 | Cost output is not billing. | Observed | `docs/cost-model.md:8-11, 374-384`; agent-forensics caveats | `polylogue analyze usage --detail full --format json --limit 0` |
| C-02 | Missing usage is unknown, not zero. | Observed | `docs/cost-model.md:57-68`; usage payload tests | `polylogue analyze usage --detail full --format json --limit 0` |
| C-03 | Pricing lanes remain separate. | Observed | `docs/cost-model.md:97-132`; committed agent-forensics packet | `polylogue analyze usage --detail headline --format json --limit 0` |
| R-01 | The product does not supply an ambient desktop timeline in the current archive schema. | Observed | `.agent/demos/anti-demo-multi-source-reconstruction/packet.json`; archive-tier DDL | <code>grep -h "CREATE TABLE" polylogue/storage/sqlite/archive_tiers/*.py &#124; grep -iE "window&#124;focus&#124;shell_history&#124;browser_tab&#124;activitywatch" &#124; wc -l</code> returns `0` |
| R-02 | `grok-export` has no parser. | Observed | `docs/architecture.md:166-172`; `polylogue/core/enums.py:42-55` | `python -m devtools lab provider completeness --json` has no Grok package row; parser registry inspection confirms no wired parser |
| R-03 | Browser capture is a capture mode, maps to provider origins, and is proposed in completeness registry. | Observed | `docs/provider-origin-identity.md`; `polylogue/sources/provider_completeness.py:133-151` | `python -m devtools lab provider completeness --json` |

## Provider and capture provenance

| Origin or mode | Package evidence | Fidelity evidence | Usage evidence | Public copy boundary |
|---|---|---|---|---|
| `claude-code-session` / `export-jsonl` | Accepted/complete row in `polylogue/sources/provider_completeness.py:54-72` | `docs/providers/claude-code.md:14-50` | `docs/cost-model.md` marks exact where `message.usage` exists | Safe to claim structured actions, thinking, errors, subagents, compaction, and captured file/git activity; do not claim workspace reconstruction |
| `codex-session` / `session-jsonl` | Accepted/complete row at `provider_completeness.py:73-87` | `docs/providers/openai-codex.md:5-31`; current parser/tests and deterministic receipt contradict its last limitation | Exact where `token_count` exists | Say supported records produce normalized actions; do not claim generation-wide losslessness until docs/tests are reconciled |
| `chatgpt-export` / `takeout-json` | Accepted/complete row at `provider_completeness.py:88-102` | `docs/providers/chatgpt.md:5-22` | Estimate-only | Attachment metadata yes, binaries no |
| `claude-ai-export` / `export-json` | Accepted/complete row at `provider_completeness.py:103-117` | `docs/providers/claude-ai.md:5-28` | Estimate-only | Typed message/block fidelity; attachment metadata only |
| `aistudio-drive` / `drive-like-export` | Accepted/complete row at `provider_completeness.py:118-132` | `docs/providers/gemini.md:5-29` | Partial | Attachment references and Drive acquisition; OAuth required for Drive bytes |
| Browser live receiver | Proposed row at `provider_completeness.py:133-151` | `docs/browser-capture.md`; provider-origin identity doctrine | Depends on parsed provider payload | Do not call package complete; do not expose browser capture as an origin |
| `antigravity-session` / language-server export | Accepted/complete row at `provider_completeness.py:152-169` | parser/tests named by row; architecture detector table | Unsupported | Claim accepted normalized import only |
| `gemini-cli-session` / local-agent document | Accepted/complete row at `provider_completeness.py:170-184` | parser/tests named by row | Partial | Claim accepted local-agent import, not full Gemini parity |
| `hermes-session` / `state-db` | Accepted/complete row at `provider_completeness.py:185+` | `docs/architecture.md:156-164`; `polylogue/sources/parsers/hermes_spans.py:1-67, 332-417` for ATIF boundary | Exact where state counters exist | Keep state-db and ATIF paths distinct; label exact, inferred, and absent fields |
| `unknown-export` | Generic storage/parser fallback | Best-effort only | Unsupported | README fallback smoke writes a one-message generic JSON file and runs `polylogue import "${TMPDIR:-/tmp}/polylogue-fallback-demo.json" --explain --format json`, which reports `detected_origin=unknown-export`; the current CLI does not accept this token through `--origin` |
| `grok-export` | Reserved enum token only | No wired parser | Unsupported | State explicitly as unsupported |

## Demo number ledger

| Number used in draft | Meaning | Exact provenance | Reproduction or check |
|---|---|---|---|
| `1` | Failed `pytest` exit status in receipt demo | `polylogue demo receipts --compact`; deterministic session `codex-session:demo-receipts`; `polylogue/demo/receipts.py` | `polylogue demo receipts --compact` |
| `5,000` | Structured failed outcomes inspected by current bounded claim-vs-evidence field packet | `.agent/demos/claim-vs-evidence/README.md:13-32`; `PUBLIC_REPRODUCTION.md:15-29`; `claim-vs-evidence.report.json` | Private count is aggregate-only; the complete seeded reproduction block is in `REPORT.md` |
| `16,816` | Physical sessions in committed v24 agent-forensics packet | `.agent/demos/agent-forensics/current/summary.json` at `archive_cardinality.physical_sessions`; generated `2026-07-05T07:36:52Z` | Committed `command_proofs` record the operator-archive run; the seeded README command reproduces report lanes, not this private count |
| `0` | Required ambient-source tables in anti-demo | `.agent/demos/anti-demo-multi-source-reconstruction/packet.json`, `results.measurements[required_ambient_source_tables]`; archive-tier DDL | Narrow schema grep in R-01 returns `0`; global packet-registry verification currently fails for unrelated shelf drift |
| `104` | Current generated MCP tool names | `docs/mcp-reference.md:66-71` | `python -m devtools render all --check` plus registry tests |
| `15 / 62 / 37` | Direct demo sessions, messages, and passing construct checks | Executed `polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json`; `polylogue/demo/seed.py`; `polylogue/demo/constructs.py` | Deliberately excluded from public README pending daemon parity |
| `15 / 60` | Daemon-ingested sessions/messages | Executed live daemon path and verifier | Blocker evidence only |
| `3 / 19` | Stale success banner and integration-test contract | `polylogue/cli/commands/import_command.py:330-333`; `tests/integration/test_demo_daemon_convergence.py:26-45, 114-141, 203-210` | Must be removed or regenerated |
| `13 / 55` | Older published demo-tour state | `docs/proof-artifacts.md:28+`; `docs/examples/demo-tour/uvx-proof.md:16` | Historical only; superseded by current direct/daemon executions |
| `90 seconds` | Requested quickstart label | User mission; no snapshot timing artifact | Remains `[90-second timing proof]` until QA-03 captures a released-package run |
| `30 seconds` | Quickstart convergence timeout | `polylogue import --help` default and registered option | Syntax executed; current semantic verification times out |
| `3` | Codex format generations | `docs/providers/openai-codex.md:5-18`; `CodexRecord.format_type` in `polylogue/sources/providers/codex.py` | Format-specific fixtures/tests; provider docs still need pairing reconciliation |
| `4` | Architecture rings | `docs/architecture.md:12-56` | Adapted Mermaid diagram; A-01 |
| `10` | Bounded embedding preflight sample in limitations | Root CLI help for `ops embed preflight`; command executed with `--max-sessions 10` | Example bound, not a performance claim |
| `105` | GitHub about-line character count | Direct Unicode code-point count of the proposed line | Static validation requires `<=120` |
| `n=5` and `n>=12` | Current uplift pilot and minimum next-stage target used in the gap report | `.agent/demos/uplift-two-arm/README.md`; `current/report.md` states `n=5` and protocol tier `n=12-20` | Exclusion evidence only; no uplift claim is published |
| `3` arms | Reader-comprehension comparison requested in gap report | Bead `polylogue-3tl.19` | Future experiment, not observed result |

## Demo packet claim boundaries

### Deterministic receipts

Observed claim: structural tool evidence can contradict assistant prose at claim time and later show repair.

Observed non-claim: no field prevalence. The command seeds a private-data-free archive when no root is supplied.

Sources: `polylogue/demo/receipts.py`, deterministic corpus records in `polylogue/demo/seed.py`, command output executed during this analysis.

### Claim-vs-evidence

Observed claim: failure predicate is `is_error=1` or non-zero `exit_code`; immediate follow-up classifier is explicit and marker-based; sample frame and calibration are published.

Observed field artifact: 42,033 structured failures in frame, 101 unpaired, 5,000 inspected, generated against index schema v24. Only 5,000 is used as the card headline. Other rates remain out of the top-level draft to avoid turning a bounded private finding into a product promise.

Observed non-claim: deterministic reproduction validates method and artifact shape, not private rates; ambiguous rows remain in the denominator; private archive is unavailable to this analysis.

Sources: `.agent/demos/claim-vs-evidence/README.md`, `PUBLIC_REPRODUCTION.md`, `COLD_READER_GATE.md`, `claim-vs-evidence.report.json`, `public-summary.json`, `summary.json`.

### Agent forensics

Observed claim: product commands regenerate physical and logical token grains plus separate pricing provenance lanes.

Observed artifact: 16,816 physical sessions, 4,364,655 messages, 399,898,650,781 physical-session tokens accounted, 292,924,997,563 logical-session high-water tokens accounted. Stored priced cost and catalog API-equivalent lanes are distinct. Only the session count is used as the card headline.

Observed non-claim: not billing truth. The `cost-rollups` drilldown timed out at 120 seconds in the committed packet, so the README uses the bounded `analyze usage --detail headline` path.

Source: `.agent/demos/agent-forensics/current/summary.json` and its command proofs.

### Multi-source reconstruction refusal

Observed claim: current archive-tier DDL has no tables for the required window-focus, independent shell-history, and browser-tab domains, so the product should return `not_supported` rather than synthesize a minute-level timeline.

Observed non-claim: no statement about a future federated architecture or external activity system.

Source: `.agent/demos/anti-demo-multi-source-reconstruction/packet.json`, `report.md`, `evidence.ndjson`, archive-tier DDL.

## Contradiction and drift ledger

| ID | Conflict | Adjudication | README effect |
|---|---|---|---|
| X-01 | Demo counts appear as 3/19 in import banner/test, 13/55 in older proof prose, 15/60 through daemon, and 15/62 through direct seed. | Current executable direct seeder and verifier are strongest for intended corpus; daemon path is the required public path and currently fails parity. | Publish no corpus count. Merge-gate the quickstart. |
| X-02 | Direct seeder injects provider usage, rebuilds insights, seeds repo/embeddings/overlays; daemon path materializes source and relies on normal convergence. | These paths are not equivalent. Missing constructs are expected from current orchestration, not a transient wait. | QA-01 must define one canonical semantic contract. |
| X-03 | Canonical AI Studio ID is `aistudio-drive:demo-00`; daemon parser produced `aistudio-drive:demo-00-0`. | Verifier and scenario registry define intended identity, but parser behavior is current runtime fact. | Treat as blocker, not a README typo. |
| X-04 | `docs/providers/openai-codex.md` says tool pairing is unimplemented; current deterministic Codex receipt renders paired actions/results. | Current source and executable artifact supersede stale prose, but format-wide coverage remains unproven. | Use conservative generation-sensitive copy and schedule doc repair. |
| X-05 | MCP generated reference says 104 tools and omits runtime `review` role in prose; runtime help exposes four roles. | Runtime source/help is authority for role set; generated count is authority for current tool registry. | Current fallback names four roles and tells integrator to regenerate docs. |
| X-06 | `.agent/demos/claim-vs-evidence/` exists and has a cold-reader packet, but `CURATED_CATALOG.md` omits it; the shelf checker also reports drift in the manifest, summary index, shelf README, and catalog. | Dedicated packet is current evidence; global shelf indexes are stale. | Include the demo, regenerate every shelf index, and require `workspace demo-shelf --check` to pass before merge. |
| X-07 | Bead `polylogue-uhl` records deterministic corpus completeness as closed, but current daemon path fails. | Current source and execution supersede closure history. | Reopen or create a regression blocker; do not cite closure as proof. |
| X-08 | Six-tool MCP replacement is requested but absent; tracker simplification work discusses a smaller verb set, not six exact tools. | Parallel-lane requirement is credible integration context but not snapshot evidence. | Preserve `[six-tool table]` slot and current fallback. |
| X-09 | The dedicated demo export contains `registry.json`, but the global verifier fails for `_packet-contract-stub`, `d4-behavioral-archaeology`, and `anti-demo-multi-source-reconstruction`: missing `run.log` plus receipt hash mismatches. | Current command execution supersedes packet metadata that declares deterministic registry success. The anti-demo's archive-schema oracle still reproduces independently. | Use the narrow oracle in public copy; regenerate packet receipts and require a clean global verifier before merge. |
| X-10 | `CLAUDE.md`, `AGENTS.md`, `docs/schema.md`, and `docs/internals.md` prescribe `polylogue ops reset --index && polylogued run`; the current reset command prints `polylogue ops maintenance rebuild-index`, and daemon startup alone left search stale in an executed temp archive. | Executable CLI behavior is current authority. The older doctrine is stale or the runtime behavior is a regression; either way, public docs must not prescribe daemon startup as sufficient replay. | README uses the maintenance plan/execution path. Add a doctrine reconciliation and reset-to-ready test. |

## Tracker records that constrain positioning

| Bead | State and relevance |
|---|---|
| `polylogue-3tl.1` | Closed artifact-first README task. Its skim-ladder principle is preserved, but the prior “system of record” category is narrowed to archive and forensics because the trust boundary and evidence are clearer. |
| `polylogue-3tl.12` | Open de-meta/de-persuasion pass. Acceptance requires runnable commands per capability and at least two cold-reader-reproducible claims. This draft supplies command mapping but does not claim the cold-reader gate is complete. |
| `polylogue-3tl.19` | Open reader-comprehension harness. No real three-arm result exists, so this rewrite is evidence-led but not reader-tested. |
| `polylogue-t46.1`, `polylogue-t46.8` | MCP simplification direction. They do not evidence the user-specified six-tool final table. |
| `polylogue-212.2` | Real merged-PR receipts demo remains open. Deterministic receipts prove the contract, not a field PR history. |
| `polylogue-r3o3` | Demo shelf hygiene. Relevant because the curated catalog omits claim-vs-evidence. |
| `polylogue-cfk` and follow-ups `57bg`, `e5b5`, `x35k`, `wnse` | Uplift evidence is explicitly non-publishable at current sample/method. Memory-benefit claims stay out. |
| `polylogue-013x` | Closed FTS coverage decision. Option B documented the exclusion/workaround; indexed Write/Edit-body coverage remains future work requiring a size probe. |
| `polylogue-sru` | Closed claim-vs-evidence campaign. Current v24 packet, not older Bead numbers, is used. |
| `polylogue-tf2` | Closed agent-forensics campaign. Current v24 packet supersedes older v23 closure numbers. |

## Unsupported certainty deliberately removed

The draft does not claim:

- universal lossless provider ingestion;
- current six-tool MCP availability;
- a working daemon-backed demo in this snapshot;
- a canonical demo corpus count;
- a measured 90-second completion time;
- billing reconciliation;
- production browser-capture readiness;
- downstream memory or continuation uplift;
- full-text coverage of every tool-input field;
- ambient desktop reconstruction;
- Grok parser support;
- performance at a named archive size without a current benchmark packet.
