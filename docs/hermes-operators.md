[← Back to Docs](README.md)

# Hermes Operator Guide

You run [Hermes](https://github.com/HexLab98/hermes-agent-fork) and are deciding
whether to point Polylogue at its local runtime. This page answers that
decision with source citations, not marketing copy: exactly what Polylogue
watches under your Hermes root, what fidelity each artifact class actually
gets, and — the part that matters most for trust — what it **cannot** claim
from your runtime today.

Everything below is verified against the parser/watcher source in this repo
and against real (redacted) Hermes exports checked into
`tests/fixtures/hermes/`, not against documentation-only assumptions. Where a
command output is quoted, it was run against this fixture set.

## TL;DR

- Polylogue watches your Hermes root (`~/.hermes` by default) for four
  independent artifact classes and imports each with an explicit,
  machine-readable fidelity declaration — never a silent guess.
- Two of those classes (state.db, verification_evidence.db) are your live
  SQLite databases; Polylogue snapshots the bytes read-only before parsing
  and never writes to them.
- Conversational content (`state.db` messages) and runtime-evidence content
  (ATIF/ATOF spans) are imported honestly but are **not physically merged**
  into one session today — see [What Polylogue cannot claim](#what-polylogue-cannot-claim-from-your-hermes-runtime).
- Everything is local: no Hermes bytes leave the machine, and the acquisition
  path for every class below is a local file read, never a network call.

## Configured runtime roots

Polylogue resolves one Hermes root through the same five-layer config chain
every other runtime path uses (`polylogue/config.py`):

| Layer (highest wins) | Value |
| --- | --- |
| Environment variable | `POLYLOGUE_HERMES_ROOT` |
| Config file | `sources.hermes.root` in `polylogue.toml` |
| Default | `~/.hermes` |

There is no dedicated `polylogued run` flag for the Hermes root specifically
(unlike, say, `--debounce-s`); use the environment variable or config file.
`polylogued run --root <path>` is a different, more drastic override — it
replaces the daemon's *entire* watched-source set with the roots you list,
not just the Hermes one (see [Try it yourself](#try-it-yourself) for where
that is actually useful: isolating a throwaway daemon for testing).

Source: `ConfigInventoryEntry("hermes_root", toml_path="sources.hermes.root",
env_var="POLYLOGUE_HERMES_ROOT", ...)` in `polylogue/config.py:780-786`, resolved
into `ResolvedRuntimeConfig.source_paths.hermes` at `polylogue/config.py:1718-1721`
(fallback `bootstrap.home / ".hermes"`). The same default appears in
`polylogue/paths/_roots.py:hermes_sessions_path()` and in `polylogue init`'s
source-detection table (`polylogue/cli/commands/init.py:62`: *"Hermes agent
state.db and fallback session exports"*).

Run `polylogue init` once to detect whether `~/.hermes` (or your configured
override) exists and record it in `polylogue.toml`; both the daemon and the
CLI read that file.

## Watched source classes

`polylogued run` includes a single `WatchSource` scoped to the resolved
Hermes root, admitting four file suffixes — `.json`, `.jsonl`, `.db`,
`.sqlite`, `.sqlite3` (`polylogue/sources/live/watcher.py:1204-1214`,
`default_sources()`). Admission is filename/suffix-based; **routing** to the
right parser is shape-based, checked in this tightness order
(`polylogue/sources/dispatch.py:160-171`):

| Artifact | Shape check | Parser | Acquisition method |
| --- | --- | --- | --- |
| `state.db` | SQLite with `sessions`/`messages` tables (`hermes_state.looks_like_state_db_payload`) | `polylogue/sources/parsers/hermes_state.py` | `sqlite_backup` — read-only snapshot before parse |
| `verification_evidence.db` | SQLite with `verification_events`/`verification_state` tables | `polylogue/sources/parsers/hermes_verification.py` | `sqlite_backup` |
| ATIF document (`*.json`) | Top-level `schema_version` prefixed `"ATIF"`, plus `session_id`/`steps` (`hermes_spans.looks_like_atif_payload`) | `polylogue/sources/parsers/hermes_spans.py` | `json_fallback`-shaped, but a real file read (see fidelity below) |
| ATOF stream (`*.jsonl`) | Each line an `atof_version`/`kind`(scope\|mark)/`category` event (`hermes_spans.looks_like_atof_payload`) | `polylogue/sources/parsers/hermes_spans.py:parse_atof_stream` | `jsonl_stream` |
| Legacy JSON snapshot fallback | Loose dict shape, no `state.db`/ATIF/ATOF match | `polylogue/sources/parsers/local_agent.py` | `json_fallback` — much lower fidelity, see below |

`Origin.HERMES_SESSION = "hermes-session"` (`polylogue/core/enums.py:48`) is
the one public origin token all five of these normalize to; the underlying
`OriginSpec` (`polylogue/sources/origin_specs.py:448-469`) declares detection
`tightness=20` — tighter than the loose dict-key checks used for ChatGPT/
Claude web/Gemini, so a Hermes artifact cannot be misclassified as one of
those looser shapes.

**A separate, always-on channel is not scoped to this root at all**: the
durable hook-event spool (`polylogue/sources/hooks.py`, drained by the daemon
component `watcher.hook_spool.drain`) accepts lifecycle events from any
configured origin, including Hermes, through `$XDG_DATA_HOME/polylogue/hooks/pending`
by default. It only carries events if you additionally install a Hermes-side
hook emitter (the versioned export contract in
[`docs/design/hermes-archival-export-contract.md`](design/hermes-archival-export-contract.md)
describes the wire format); pointing `sources.hermes.root` at your install
does not by itself enable it.

## Fidelity model

Every Hermes import returns a machine-readable `HermesImportFidelity`
declaration (`polylogue/sources/parsers/hermes_state.py:131-156`), never a
bare "imported OK." Each capability is scored on a closed vocabulary:

| Status | Meaning |
| --- | --- |
| `exact` | Every expected instance was observed and structurally normalized — not inferred, not guessed. |
| `degraded` | Some but not all expected instances were observed (partial coverage). |
| `absent` | The source artifact carries no evidence for this capability; never represented as an empty-but-present value. |
| `inferred` | The value is derived by the normalizer (e.g. from message role), not read from an explicit source field. |
| `redacted` | Reserved for capabilities whose evidence exists but is deliberately not retained. |

`import_fidelity_declaration()` is deliberately conservative per its own
docstring: *"values derived by the normalizer are marked inferred, and
evidence that no source artifact carries is absent rather than represented
as an empty successful value"* (`hermes_state.py:239-244`).

### `state.db` (your live conversational session store)

Acquired as a `sqlite_backup` — the whole file is snapshotted read-only
before any row is parsed, so `retained_blob_reproducibility` is `exact`
(`hermes_state.py:351-357`). Per-session-revision capabilities, keyed off
which SQLite columns your live schema actually has
(`_SESSION_CAPABILITIES`/`_MESSAGE_CAPABILITIES`, `hermes_state.py:49-101`):

| Capability | Typical status | Why |
| --- | --- | --- |
| `material_origin` | `inferred` (always) | Hermes has no explicit human/assistant addressing field; normalized from role + the source `observed` flag. |
| `message_state` | `exact`/`degraded` | Active/observed/rewound/compacted per-message state, when the schema carries those columns. |
| `cost_provenance` | `exact`/`absent` | Billing provider/mode/status/pricing-version columns — present from schema v18+ gateway metadata onward, absent on older installs. |
| `lifecycle`, `relationship`, `repository` | `exact`/`absent` | End reason, rewind count, parent-session id, cwd/git branch — present when the corresponding columns exist. |
| `gateway_identity` | `exact`/`absent` | Schema v18+ only (Hermes #9006 chat-platform bridging columns: `session_key`/`chat_id`/`chat_type`/`thread_id`). Absent on plain-CLI installs with no gateway bridging. |
| `compression_recovery` | `exact`/`absent` | Schema v19+ only (`compression_failure_cooldown_until`/`compression_failure_error`). |
| `runtime_spans` | `absent` (always, from this artifact alone) | `state.db` carries no runtime-event stream — that evidence only exists in ATIF/ATOF. |
| `span_snapshot_merge` | `absent` (always, from this artifact alone) | No runtime spans are merged into a `state.db`-only import. |

The parser's live-verified schema baseline is v16
(`polylogue/schemas/providers/hermes/state_db_v16.contract.json`); a
schema-drift canary test pins the v19 column additions above against a real
schema sample (`tests/unit/sources/parsers/test_hermes_state_schema_canary.py`).
A version this parser does not recognize is read structurally by column
presence, not by a hardcoded version gate, so newer columns degrade
gracefully to `absent` rather than failing import.

**If your install predates the SQLite backend** (`json_fallback`
acquisition, e.g. a legacy exported document via `local_agent.py`), fidelity
drops sharply: `retained_blob_reproducibility` itself is `absent` (the
fallback proves nothing about a retained SQLite snapshot), and every
capability except `material_origin`/`lifecycle` (both `inferred`) is `absent`
(`hermes_state.py:363-447`). Treat a `json_fallback`-acquired Hermes session
as a much thinner record than a live `state.db` import.

### ATIF trajectory documents (NeMo Relay `schema_version: "ATIF-*"`)

Verified live against the checked-in real fixture
(`tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json`) via
`polylogue import <file> --explain`:

```
"capabilities": {
  "llm_request_spans":    {"status": "exact",  "observed": 5, "expected": 9},
  "tool_execution_spans": {"status": "exact",  "observed": 4, "expected": 9},
  "subagent_delegation":  {"status": "absent",  "observed": 0, "expected": 9},
  "decision_points":      {"status": "absent",  "observed": 0, "expected": 9},
  "error_taxonomy":       {"status": "absent",  "observed": 0, "expected": 9}
}
```

ATIF's documented step schema (`source`/`tool_calls`/`observation`/`message`)
carries no approval-hook or error-hook vocabulary at all, so
`decision_points`/`error_taxonomy` are honestly `absent` for ATIF-only
imports — that evidence exists only in the raw ATOF stream (see below), never
backfilled by assumption. `subagent_delegation` is recorded from
`subagent_trajectories` entries when present, but **not** materialized into
`topology_edges`/`session_links` — no real trajectory sampled so far has a
non-empty `subagent_trajectories` list, so this capability stays `absent` on
every fixture observed to date (`hermes_spans.py:35-38`).

Payload hygiene: unlike the verification ledger below, ATIF/ATOF never copy
prompt or tool-argument text. A tool-call step records only ids, argument
*presence*, and structural outcome — never the arguments or the response
text itself (`hermes_spans.py:44-52`).

### ATOF event stream (append-only `events.jsonl`)

Verified the same way against
`tests/fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl`:

```
"capabilities": {
  "llm_request_spans":    {"status": "exact", "observed": 3, "expected": 13},
  "tool_execution_spans": {"status": "exact", "observed": 2, "expected": 13},
  "decision_points":      {"status": "exact", "observed": 2, "expected": 13},
  "error_taxonomy":       {"status": "exact", "observed": 1, "expected": 13},
  "subagent_delegation":  {"status": "exact", "observed": 1, "expected": 13},
  "context_events":       {"status": "exact", "observed": 3, "expected": 13},
  "topology_edges":       {"status": "absent", "observed": 0, "expected": 1},
  "unpaired_scope_debt":  {"status": "degraded", "observed": 1, "expected": 13}
}
```

ATOF is the richer of the two runtime-evidence channels: it is the only
source that carries `hermes.approval.*` decision outcomes and structural
error evidence at all — ATIF cannot supply either. `unpaired_scope_debt`
being `degraded` (not `absent`) on this fixture reflects a real,
intentionally-surfaced fact: a scope UUID that never saw both its start and
end phase (crashed request, or a file rotation cutting a pending scope in
half) is recorded as explicit acquisition debt rather than silently treated
as equivalent to a completed pair (`hermes_spans.py`, `import_explain.py`).

One real live-ingestion detail worth knowing if you watch a growing Hermes
install: the real ATOF producer writes **one shared `events.jsonl` file
across every session on the install** — a single growth batch can span a
session boundary. The live watcher never attempts incremental append for
this source class; it always routes through the full/bundle path, which
groups by session id correctly (`hermes_spans.py:64-82`).

### `verification_evidence.db` (Hermes's own claim-vs-evidence ledger)

This is the newest and, for verification-coverage claims, the most
important artifact class — Hermes's own record of commands it ran to verify
its own work (lint/typecheck/build/format/check/test, plus ad-hoc scripts),
each with a literal command, canonical command name, exit code, pass/fail
status, and a bounded output summary
(`polylogue/sources/parsers/hermes_verification.py:1-48`).

```
"capabilities": {
  "command_evidence":        {"status": "exact"},
  "outcome_evidence":        {"status": "exact"},
  "output_evidence":         {"status": "exact" if any output_summary present else "absent"},
  "changed_paths":           {"status": "exact" if verification_state rows present else "absent"},
  "correlation":             {"status": "degraded" if any session_id == 'default'},
  "retention_completeness":  {"status": "degraded", "always"}
}
```

Two structural facts worth calling out explicitly:

- `exit_code` is `NOT NULL` in Hermes's own schema — every recorded event has
  a definite outcome, unlike the NULL-means-unknown
  `tool_result_exit_code` convention used elsewhere in this archive
  (`hermes_verification.py:23-24`).
- `retention_completeness` is **always** `degraded`, never `exact`, because
  the producer itself prunes events older than 30 days and caps at 100
  events per `(session_id, root)` / 10,000 unreferenced total. This import
  reflects only what your live Hermes install currently retains, never a
  complete historical ledger (`hermes_verification.py:376-384`).
- A `session_id` of `"default"` is Hermes's own fallback for "session id
  unknown" and is surfaced as an explicit `ambiguous_correlation` caveat —
  never silently trusted as a real session reference
  (`hermes_verification.py:36-40`).

Payload hygiene here is the **deliberate exception** to the ATOF/ATIF rule
above: `command`/`canonical_command`/`output_summary` round-trip verbatim,
because this is Hermes's own recorded evidence, not conversational content —
the producer already bounds `output_summary` to 2000 characters
(`hermes_verification.py:30-35`).

## Session identity: what does and does not get correlated

Every artifact class above is keyed by the same raw Hermes session id, but
Polylogue deliberately does **not** physically merge them into one queryable
session yet:

| Artifact | Archive session id | Prefix |
| --- | --- | --- |
| `state.db` (conversational transcript) | `hermes-session:<hermes_session_id>` | none |
| ATIF (trajectory export, runtime-evidence observer session) | `hermes-session:observer:atif:<hermes_session_id>` | `observer:atif:` |
| ATOF (event stream, runtime-evidence observer session) | `hermes-session:observer:atof:<hermes_session_id>` | `observer:atof:` |
| `verification_evidence.db` (verification ledger) | `hermes-session:verification:<hermes_session_id>` | `verification:` |

ATIF and ATOF get **distinct** artifact-qualified prefixes (`fs1.14`) — see
[What Polylogue cannot claim](#what-polylogue-cannot-claim-from-your-hermes-runtime)
item 1 for the collision this fixed. When the acquiring directory (the
`profile_root` Polylogue derives from the artifact's own source path) is
known, ATIF's, ATOF's, and the verification ledger's identity all
additionally carry the same `@profile-<key>` qualifier the `state.db`
conversational session uses — two separate Hermes installs (profiles) that
happen to reuse the same raw session id get fully independent archive rows
on every artifact class, not just independent artifact families (`fs1.14`
for ATIF/ATOF, `polylogue-y9zx` for the verification ledger; see item 1
below). Read-side helpers cross-reference all four without merging them:
`hermes_spans.hermes_atif_session_id_for()`,
`hermes_spans.hermes_atof_session_id_for()`, and
`hermes_verification.hermes_verification_session_id_for()` each take a
`state.db`-ingested conversational session id and return the matching
observer/verification session id, *preserving* whatever `@profile-<key>`
qualifier it carries — a reader holding the qualified conversational id
resolves that artifact's evidence for the same install, not merely the same
raw session id.
`polylogue/insights/hermes_topology_projection.py` composes all four artifact
classes for one raw Hermes session id into one typed, read-only projection
(availability, per-artifact fidelity, unpaired-trace debt, and explicit
producer-conflict detection for disagreeing subagent-session evidence) — but
like `hermes_verification_coverage`, it is not yet wired into any CLI/MCP
surface (see item 3 below). None of this materializes a
`topology_edges`/`session_links` relationship between them — correlation is a
read-time lookup, not a stored graph edge.

## What Polylogue cannot claim from your Hermes runtime

This is the section to read before you trust anything above it.

1. **[Fixed, `fs1.14`] ATIF and ATOF used to collide when they shared a
   session id, on two independent axes.** Before `fs1.14`, both artifact
   families mapped to the identical `hermes-session:observer:<hermes_session_id>`
   archive session with no profile qualifier at all, and importing both for
   the same underlying Hermes session was **not additive**: the second full
   parse's differing content hash replaced rather than unioned the first's
   parsed rows, silently discarding whichever artifact was ingested first
   (verified empirically at the time — importing the checked-in ATIF fixture
   then the checked-in ATOF fixture into one scratch archive left exactly one
   `sessions` row carrying only ATIF's evidence). The same unqualified
   `observer:<id>` identity separately let two different Hermes installs
   (profiles) reusing the same raw session id collapse onto that one archive
   row too. Both axes are fixed together: ATIF and ATOF now get
   artifact-qualified session ids (`observer:atif:<id>` / `observer:atof:<id>`,
   see table above), and when the acquiring directory is known each of those
   ids is *additionally* profile-qualified (`observer:atif:<id>@profile-<key>`
   / `observer:atof:<id>@profile-<key>`) using the exact qualifier the
   `state.db` conversational session computes for the same install. Importing
   both artifacts for the same Hermes session, from the same or different
   installs, now retains fully independent archive rows — nothing replaces
   anything else. This still does **not** mean they are unioned into one
   queryable session: read the relevant ids (or use
   `hermes_topology_projection.project_hermes_topology`) to see the combined
   evidence. The verification ledger (`hermes_verification.py`) had the
   identical unqualified-id collapse risk on the profile axis (it was never
   part of the ATIF/ATOF artifact-family collision, since it already used
   its own `verification:` prefix) — fixed separately as `polylogue-y9zx`,
   merged as PR #3227, using the same shared `hermes_identity.py` qualifier
   scheme; see the "Session identity" section above.
2. **No physical merge across `state.db` transcript ↔ observer evidence ↔
   verification ledger.** Three logically-related session ids can exist for
   one real Hermes session (conversational, observer, verification). Reading
   the conversational session id does not automatically surface runtime
   spans or verification outcomes — you must look up the paired id via the
   helpers above, or use the `hermes_verification_coverage` primitive
   (next point).
3. **Verification-coverage correlation has no CLI or MCP surface yet.**
   `polylogue/insights/hermes_verification_coverage.py` (`fs1.4`) is a real,
   tested, pure-aggregation function that summarizes one Hermes session's
   `verification_evidence.db` coverage — but it is not registered in the
   insight registry, has no `polylogue read --view` entry, and is not
   exposed through MCP. It is reachable only from Python by calling
   `correlate_verification_coverage()` yourself. The bead tracking this
   (`polylogue-fs1.4`) records this as a deliberate scoping decision, not an
   oversight: composing the primitive into a named CLI/MCP surface is
   separate, tracked follow-up work.
4. **The named Hermes forensics report does not exist yet.** There is no
   `polylogue forensics hermes` command or `read --view forensics`. What
   exists today composes from generic, origin-agnostic primitives already
   documented elsewhere: session topology (`get_session_topology`), the
   postmortem bundle (`polylogue/insights/postmortem.py`,
   `docs/agent-forensics.md`), and git-commit correlation
   (`polylogue/insights/session_commit.py`) all work on Hermes sessions the
   same as any other origin, because none of them branch on origin. But
   there is no single command that packages all five forensic sections for
   a Hermes session yet.
5. **`subagent_delegation` never reaches `topology_edges`/`session_links`.**
   Even where ATIF/ATOF record subagent evidence, it stays observer-only
   evidence; Polylogue's session-lineage graph does not learn about it.
   `hermes_topology_projection.project_hermes_topology` surfaces it read-side
   as `subagent_evidence` (tagged by which artifact reported it) and flags a
   self-referential or ATIF/ATOF-disagreeing subagent-session id as an
   explicit `HermesTopologyConflict` rather than silently trusting one side —
   but this is still evidence composition, not a physical session link.
6. **The verification ledger is a bounded retention window, not history.**
   `retention_completeness` is structurally `degraded` on every import —
   events older than 30 days, or beyond the producer's 100-per-scope/10,000-
   total caps, were never retained by Hermes in the first place. Absence of
   an old verification event in Polylogue's archive can mean "never ran" or
   "already pruned by Hermes" — the import cannot distinguish the two.
7. **`material_origin` for `state.db` messages is always inferred, never
   read from an explicit field** — Hermes has no addressing field
   distinguishing human-authored from other message classes; Polylogue
   derives it from role plus the source's `observed` flag. Cost/user-word
   accounting built on `material_origin` inherits that inference.
8. **A `json_fallback`-acquired session (pre-SQLite installs, or a
   hand-exported document) has almost no fidelity beyond message text and
   role.** Cost, lifecycle detail, repository context, and even the
   retained-bytes reproducibility guarantee are all `absent` on that path —
   see the `state.db` fidelity table above.
9. **No cost/usage rollup guarantee.** `cost_provenance` is `exact` only
   when your live `state.db` schema carries billing columns (schema v18+
   gateway-metadata era or later, `estimated_cost_usd`/`actual_cost_usd`/
   `pricing_version`, etc.); older schemas report it `absent`, not a zero
   cost.

## Query and forensics entry points

All commands below are real, run against this repo's checked-in fixtures —
copy-pasteable, not illustrative pseudocode.

### Preview fidelity for one file, no daemon required

`--explain` parses the file in-process and prints its fidelity declaration
without staging or scheduling anything — the safest way to preview what an
artifact will yield before you point a running daemon at it:

```bash
polylogue import tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json --explain
polylogue import tests/fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl --explain
```

Each prints `detector`/`parser`/`produced` counts, any `caveats`, and the
full `fidelity` block shown in the tables above.

### Search and read Hermes sessions once ingested

```bash
polylogue --origin hermes-session find "your search terms" then read --all --format json
polylogue find 'sessions where origin:hermes-session' then read --all --format json
polylogue find id:hermes-session:observer:atif:<hermes-session-id> then read --view transcript
polylogue find id:hermes-session:observer:atof:<hermes-session-id> then read --view raw --format json
```

`--origin hermes-session` and `origin:hermes-session` are equivalent (root
filter vs. DSL field). `read --view raw --format json` returns the raw
artifact's content-hash `raw_id`, `source_path`, and `blob_size` — the same
identity used for the durable-retention guarantee above.

### MCP (agent-facing)

The current standing MCP surface is a small set of unified verb tools —
`query`, `read`, `get`, `explain`, `context`, `status` — the live contract
enforced by `tests/infra/mcp.py:EXPECTED_TOOL_NAMES`
(see `docs/agent-manual.md` for the generated, currently-accurate reference;
treat `docs/mcp-reference.md`'s larger per-category tool list as describing
an earlier surface generation, not this one). The `query` tool's typed
request accepts an `origin` field
(`polylogue/mcp/query_contracts.py:88`) exactly like the CLI's `--origin`
flag, so `query(origin="hermes-session", ...)` scopes a search or aggregate
to Hermes sessions the same way.

### Composable, origin-agnostic forensic primitives

These already work on Hermes sessions without any Hermes-specific code,
because none of them branch on origin — see
[Agent Forensics](agent-forensics.md) and
[Insights Rigor Matrix](insights-rigor-matrix.md) for the general contract:

- Session topology / logical-session composition (parents, resumes,
  compactions, branches).
- The postmortem bundle (`polylogue/insights/postmortem.py`) — cost/token
  lanes, tool-category profiles, pathology-detector failure modes.
- Git-commit correlation (`polylogue/insights/session_commit.py`).

## Privacy boundaries

- **Local-only, single-writer.** The daemon binds `127.0.0.1` by default; the
  primary threat model is a hostile browser tab reaching the loopback API,
  not remote access — see [Security](security.md) for the full trust-boundary
  table. Remote binding requires an explicit
  `--insecure-allow-remote` plus a token; refused by default.
- **No network calls for Hermes ingestion.** Every acquisition method above
  (`sqlite_backup`, `json_fallback`, `jsonl_stream`) is a local file read.
  Nothing about pointing `sources.hermes.root` at your install causes an
  outbound request.
- **Payload hygiene is enforced per artifact class, not assumed.** ATIF/ATOF
  never copy prompt or tool-argument text — only ids, presence, and
  structural outcome. The verification ledger is the one deliberate
  exception (it retains Hermes's own command/output text verbatim, because
  that text *is* the evidence, not conversational content) — see the
  per-artifact sections above for exactly which fields round-trip.
- **You control the root.** Nothing is watched outside the resolved
  `sources.hermes.root` (or the general, separately-configured hook spool,
  which requires its own opt-in Hermes-side hook install).

## Try it yourself

Run these against a throwaway archive, not your live one, so this exercise
never mixes fixture data into your real history:

```bash
export POLYLOGUE_ARCHIVE_ROOT=$(mktemp -d)/polylogue-hermes-smoke

# 1. Fidelity preview, no daemon, no network, no writes outside the temp dir:
polylogue import tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json --explain
polylogue import tests/fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl --explain

# 2. Full ingest + query, watching only the throwaway archive's own inbox
#    (never the real ~/.claude, ~/.codex, or ~/.hermes; the hook spool is
#    scoped to $POLYLOGUE_ARCHIVE_ROOT/hooks automatically):
polylogued run --no-browser-capture --no-source-catchup \
  --root "$POLYLOGUE_ARCHIVE_ROOT/inbox" &
polylogue import tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json
polylogue --origin hermes-session find "hermes" then read --all --format json
```

One safety note learned the hard way while writing this guide:

- Never run `polylogued run` with source catch-up enabled (the default)
  against a throwaway archive on a machine with a real `~/.claude`,
  `~/.codex`, or `~/.hermes` — catch-up scans and ingests every configured
  default root, real data included. Always pass `--no-source-catchup` for a
  fixture-only smoke test, and prefer `--root` scoped to the throwaway
  inbox as shown above.

## Related docs

- [Configuration](configuration.md) — full five-layer config reference.
- [Daemon](daemon.md) — convergence stages, watcher behavior, HTTP API.
- [Security](security.md) — threat model and trust boundaries.
- [Agent Forensics](agent-forensics.md) — origin-agnostic forensic methods.
- [Hermes Archival Export Contract](design/hermes-archival-export-contract.md)
  — the proposed (Polylogue-side-only) versioned session-export + lifecycle-
  hook design for a deeper, opt-in integration.
- [Provider, Origin, and Source Identity](provider-origin-identity.md) —
  vocabulary this page assumes.
