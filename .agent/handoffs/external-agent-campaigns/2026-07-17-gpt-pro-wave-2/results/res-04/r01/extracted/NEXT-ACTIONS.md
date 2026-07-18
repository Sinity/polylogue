# Integration and re-verification checklist

## Merge rule

Do not merge the quickstart or fixed demo metrics until QA-01 through QA-04 pass on the live integration branch. The positioning copy, provider-fidelity framing, architecture, limitations, and “what it is not” list can be integrated independently if their command references remain current.

## P0: Repair daemon-backed demo parity

### QA-01: Make `polylogue import --demo --wait` converge to the canonical semantic contract

**Owning areas**

- `polylogue/cli/commands/import_command.py`
- `polylogue/demo/seed.py`
- `polylogue/demo/verify.py`
- `polylogue/demo/constructs.py`
- `polylogue/scenarios/corpus.py`
- daemon ingest and convergence stages under `polylogue/daemon/`, `polylogue/pipeline/`, and archive insight materialization
- `tests/integration/test_demo_daemon_convergence.py`

**Observed failure to resolve**

- Direct seed: 15 sessions, 62 messages, all 37 declared constructs pass.
- Daemon path: 15 sessions, 60 messages.
- AI Studio identity differs: expected `aistudio-drive:demo-00`, observed `aistudio-drive:demo-00-0`.
- Daemon path lacks provider usage messages, capture-gap events, three browser-capture raw variants, source-outage interval events, synthetic message embeddings, and embedding status rows.
- Success banner and integration test still require 3 sessions and 19 messages.

**Implementation decision required**

Choose one owner for post-parse deterministic demo augmentation:

1. move every intended construct into source-shaped fixtures so normal daemon convergence produces it; or
2. make daemon demo scheduling call an explicit, idempotent post-ingest demo augmentation stage that is also used by direct seed.

Do not leave direct seed with a private sequence of insight rebuilds, usage injection, repo seeding, embedding seeding, and overlays that the public daemon path cannot execute.

**Acceptance criteria**

- Fresh temp `POLYLOGUE_ARCHIVE_ROOT` and `XDG_CONFIG_HOME`.
- Live daemon on an isolated loopback port.
- `polylogue import --demo --wait --timeout 30 --daemon-url "$POLYLOGUE_DAEMON_URL"` exits zero.
- `polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --format json` returns `ok=true` without suppressing semantic checks.
- Exact session identity set matches the canonical scenario registry.
- Every declared construct satisfies its minimum.
- Direct seed and daemon import produce the same semantic verifier result. Raw acquisition bookkeeping may differ only if explicitly declared and tested.
- Repeating the import is idempotent: no duplicate sessions, messages, actions, links, usage rows, embeddings, or overlays.
- The CLI success line derives counts and overlay state from the verifier result; no hardcoded `3` or `19` remains.
- Integration tests consume canonical generated expectations rather than a second hand-maintained session tuple.
- Public verification exposes no operator-specific absolute source paths.

**Falsification evidence**

Any timeout, session-ID mismatch, missing construct, count mismatch between banner and verifier, duplicate on second import, or hidden direct-seed-only augmentation fails QA-01.

### QA-02: Establish one generated demo metrics source

**Owning areas**

- `polylogue/scenarios/corpus.py`
- demo report/datasheet renderer
- `docs/plans/demo-corpus-construct-audit.md`
- `docs/generate.md`
- `docs/proof-artifacts.md`
- README `[demo numbers]`
- demo-shelf packet metadata
- integration and unit tests

**Acceptance criteria**

- One generated JSON payload contains commit, schema versions, session count, message count, origin count, construct minima/observations, exact session IDs, overlay state, and generation command.
- README, CLI banner, tests, docs, and demo packets consume that payload or generated derivatives.
- `python -m devtools render all --check` fails when any consumer drifts.
- Older 3/19 and 13/55 prose is removed or explicitly archived as historical.

### QA-03: Re-run the exact quickstart from a released install

**Channels**

At minimum, use the channel the README presents first. If pipx remains first, test an actual published wheel in a clean environment. Test Homebrew/Nix separately before claiming parity across channels.

**Capture**

- host OS, CPU, Python version, package version, and artifact digest;
- install wall time;
- daemon readiness wall time;
- `import --demo --wait` wall time;
- first `find`, exact `read`, and action aggregate wall times;
- stdout/stderr and exit status for each command.

**Acceptance criteria**

- Every command in the quickstart is copy-paste runnable without a source checkout.
- No provider account, API key, or existing personal archive is read.
- The throwaway root can be removed after stopping the daemon.
- Only after a measured run fits the bound may `[90-second timing proof]` be replaced with a factual timing statement.

### QA-04: Recheck the exact user-facing observations

Run after QA-01:

```bash
polylogue --plain find pytest
polylogue --plain find id:codex-session:demo-receipts then read --view messages
polylogue --plain 'actions where is_error:true | group by tool | count'
polylogue demo receipts --compact
```

Acceptance:

- `find pytest` returns the documented fixture refs.
- Exact read contains the failed receipt, assistant claim, edit, and later successful receipt.
- Aggregate counts only structurally failed actions.
- Anti-grep control still has prose hits but no structurally failed action.

## P0: Fill the MCP integration slot

### MCP-01: Replace fallback with the landed six-tool registry

**Owning areas**

- landed parallel MCP registry and declarations
- `polylogue/mcp/`
- generated `docs/mcp-reference.md`
- MCP contract tests and expected-name registry
- README `[six-tool table]`

**Required table columns**

- tool name;
- user/agent intent;
- minimum role;
- key input;
- result envelope and ref semantics;
- one copy-paste client call or configuration example.

**Acceptance criteria**

- Registry has exactly six public tools by the new contract.
- Generated reference and runtime `list_tools` agree.
- Role matrix is generated and includes any `review` capability if it survives the replacement.
- Each tool has a bounded result contract, exact ref semantics, and error/non-support behavior.
- Current 104-tool fallback is removed from README, not left beside the new table.
- Migration note maps removed public tools to the six new intents or states that no equivalent exists.
- `polylogue-mcp --help`, generated docs, and client smoke agree.

**Falsification evidence**

A seventh public tool, hidden legacy aliases, docs/runtime role mismatch, unbounded payload, or missing migration path fails MCP-01.

## P1: Refresh demo shelf and proof packets

### DEMO-01: Regenerate the four README demo cards on the merge commit

**Commands**

```bash
polylogue demo receipts --compact
python -m devtools workspace claim-vs-evidence \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" \
  --limit 5000 \
  --out-dir .agent/demos/claim-vs-evidence \
  --json
polylogue --plain analyze usage \
  --detail headline \
  --format json \
  --limit 0
# Regenerate stale packet receipts and required run logs first.
python -m devtools.verify_demo_packet_registry
python -m devtools workspace demo-shelf
```

**Acceptance criteria**

- Every packet records merge commit, generation date, schema versions, command, claim, non-claim, proof fields, and caveat fields.
- `CURATED_CATALOG.md`, manifest, and summary index include `claim-vs-evidence`.
- No private transcript preview enters a public packet.
- README headline numbers match generated artifacts exactly.
- Agent-forensics continues to separate physical/logical grains and pricing lanes.
- Anti-demo still returns `not_supported` from an independent schema oracle.
- The global packet registry exits zero; no packet has a missing `run.log` or receipt hash mismatch.

### DEMO-02: Reconcile Codex provider documentation

**Owning areas**

- `docs/providers/openai-codex.md`
- Codex parser and action-pairing tests
- provider completeness fixtures

**Acceptance criteria**

- Each of the three detected format generations has a fixture.
- Generated report says whether calls/results are paired, unpaired, or absent per generation.
- Provider doc no longer says pairing is globally unimplemented if executable tests prove otherwise.
- README table is regenerated from the reconciled evidence.

### DEMO-03: Keep field findings bounded

For claim-vs-evidence, preserve:

- failure predicate;
- total and inspected frame;
- unpaired count;
- origin-stratified selection;
- explicit marker classifier;
- ambiguous denominator treatment;
- calibration sample and labels;
- seeded public reproduction;
- cold-reader gate.

Do not update only the headline rate.

## P1: Documentation and source drift gates

### DOC-01: Run the registered documentation gates

From the integration checkout:

```bash
python -m devtools verify doc-commands
python -m devtools lab provider completeness --check
python -m devtools workspace demo-shelf --check
python -m devtools render all --check
```

Acceptance:

- No stale command or subcommand in README/docs.
- All accepted provider rows have required evidence.
- Demo indexes are current.
- CLI/MCP/generated docs match runtime source.

### DOC-02: Audit every README feature sentence

For each sentence that claims a capability:

- name a user command or checkout verification command;
- identify its evidence path in `EVIDENCE.md`;
- distinguish deterministic contract, private field finding, source-supported architecture, and unresolved integration state;
- remove any adjective that cannot be tied to a metric or artifact.

Automated checks should reject:

- unmeasured speed claims;
- fixed demo counts outside the generated metrics source;
- “all,” “every,” “lossless,” or “complete” without an explicit scope;
- browser capture listed as an origin;
- costs described as bills or savings;
- memory uplift language before the outcome study passes.

### DOC-03: Verify style mechanically

- No em dash used as a sentence connector in the README.
- No stacked three-part slogan cadence.
- No unsupported superlatives.
- GitHub about line remains at or below 120 characters.
- Every code block is executable in its declared environment.
- Placeholder strings `[demo numbers]`, `[six-tool table]`, and `[90-second timing proof]` are either filled from evidence or intentionally retained in the integration branch, never accidentally published.


### DOC-04: Reconcile derived-index rebuild doctrine with the executable path

**Observed conflict**

- `CLAUDE.md`, `AGENTS.md`, `docs/schema.md`, and multiple `docs/internals.md` entries say `polylogue ops reset --index && polylogued run`.
- `polylogue/cli/commands/reset.py` prints `polylogue ops maintenance rebuild-index` as the required next action.
- In a throwaway seeded archive, reset followed by daemon startup left search stale and emitted an uninitialized-tier raw-authority error; the explicit maintenance rebuild completed with ready surfaces.

**Owning areas**

- `polylogue/cli/commands/reset.py`
- `polylogue/cli/commands/maintenance/_rebuild_index.py`
- daemon startup/readiness behavior
- `CLAUDE.md`, `AGENTS.md`, `docs/schema.md`, and `docs/internals.md`
- reset/rebuild integration tests

**Acceptance criteria**

- One command sequence is canonical in runtime help, operator docs, repository doctrine, diagnostics, and tests.
- Reset followed by that sequence replays all eligible source authority into a promoted index generation.
- Search, actions, raw links, profiles, threads, phases, and work events report ready.
- Quarantined or adoption-deferred rows are named with reasons; no source evidence is silently lost.
- `python -m devtools verify doc-commands` and `python -m devtools render all --check` fail on future sequence drift.

**Falsification evidence**

A stale search surface, daemon-only sequence that does not replay source, mismatch between reset output and docs, or unexplained cardinality loss fails DOC-04.

## P2: Evidence needed for claims intentionally excluded

### PERF-01: Large-archive performance packet

Measure cold and warm latency for:

- lexical `find`;
- exact-ref message read;
- structured action aggregation;
- usage headline;
- one bounded MCP read tool;
- optional semantic and hybrid lanes after preflight.

Record hardware, schema versions, archive cardinality, query, result cardinality, repetitions, percentile method, and cache state. This would support a precise performance sentence without a superlative.

### SEARCH-01: Decide indexed Write/Edit-body coverage

Run the live FTS size and latency probe required by `polylogue-013x`. Compare:

- current FTS plus unindexed action-input predicate;
- expanded generated `search_text`;
- dedicated indexed action-input lane.

Acceptance must include derived-tier rebuild instructions, disk delta, ingest amplification, query latency, and false-positive behavior.

### BROWSER-01: Promote browser capture only with release evidence

Required evidence:

- extension artifact and daemon version parity;
- loopback/auth/origin checks;
- readiness and capture status;
- provider-origin mapping;
- coalescing and raw-variant retention;
- source-outage/capture-gap behavior;
- privacy caveats;
- clean-host release smoke.

### OUTCOME-01: Continue the handoff-pack experiment

Use generated production packs, independent subjects, clean blinding, preregistered scoring, and the protocol’s publishable sample tier. Keep the false-fact counterexample and verify packs against live state. Until this closes, do not position Polylogue as a proven memory-uplift or retrieval-benchmark product.

### BILLING-01: Cost reconciliation

A billing claim requires a matched provider invoice period, model mapping, usage coverage ratio, currency/tax treatment, subscription effects, and unexplained residual. Without that packet, keep “cost evidence” and “catalog API-equivalent estimate” wording.

### READER-01: Run the comprehension harness

Execute `polylogue-3tl.19` with at least:

- current README;
- this evidence-first candidate;
- a materially different alternative.

Report arm assignment, exposure time, category comprehension, first successful action, proof recall, boundary recall, sample size, and selection bias. Use the result to tune ordering, not to weaken factual caveats.

## Final pre-merge command set

```bash
polylogue --version
polylogued run --help
polylogue import --help
polylogue find --help
polylogue read --help
polylogue analyze usage --help
polylogue-mcp --help

python -m devtools verify doc-commands
python -m devtools lab provider completeness --json
python -m devtools lab provider completeness --check
python -m devtools workspace demo-shelf --check
python -m devtools render all --check
```

Then run the complete throwaway quickstart exactly as printed in `REPORT.md`, from the selected release artifact, without relying on the source checkout.

## Expected value of the next iteration

Another pass is high value after QA-01 and MCP-01. It will replace the two largest placeholders with executable evidence, establish whether the timing label is supportable, reconcile generated docs with runtime roles, and let the final README publish one canonical demo state. Before those land, another prose-only pass has low value and risks creating new drift.
