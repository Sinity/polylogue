# Polylogue Rust foundation

This repository contains the standalone Rust foundation. It implements foundation identity/value contracts, durable SQLite schema checks, shared content files, one archive writer, read-only handles and a real daemon/status client. Explicit immutable Codex and context-declared Claude regular-file imports and provider-state progress are registered. Selected uncontested Messages have a streaming terminal reference read, and known selected Session catalogs have a degraded normalized JSON read. Full canonical finalization, export, search, MCP and HTTP remain unfinished. Status reports `product_path: "partial"`; an initialized empty schema is not a completed ingestion or M1 demonstration.

## Build and run

```sh
nix flake check
nix build .#polylogue
./result/bin/polylogue --help
./result/bin/polylogue --version
./result/bin/polylogue config schema
./result/bin/polylogue --archive /path/to/isolated-archive daemon
./result/bin/polylogue --archive /path/to/isolated-archive status
./result/bin/polylogue --archive /path/to/isolated-archive import /path/to/rollout.jsonl --format codex-rollout
./result/bin/polylogue --archive /path/to/isolated-archive progress INPUT
```

Use fresh isolated state. `XDG_RUNTIME_DIR` must be absolute. The archive defaults to `$XDG_DATA_HOME/polylogue`, or `$HOME/.local/share/polylogue`; an explicit `--archive` takes precedence. No product-specific environment variable selects an archive. Static help/version/schema do not open or create state. Unsupported commands and bad arguments return a non-success outcome instead of an empty answer.

The foreground daemon owns the archive lock, writable schema and its derived socket. Ctrl-C stops admission, closes clients and joins handlers and its capture/compute workers before releasing the writer. A second archive can run independently; competing archive aliases and occupied endpoints are refused. Startup never adopts a foreign listener. Missing/empty database files with retained recovery sidecars require explicit recovery, and newer durable schemas are refused before mutation.

## Development

The declared AgentCTL operations run the Nix build/check and development-shell commands. `cargo run -p xtask -- check` checks target formatting, actual Cargo dependency boundaries and workspace tests. A synthetic Cargo-edge mutation and compile-fail examples demonstrate forbidden access. `.githooks` contains successor hooks; in linked worktrees select them through per-worktree Git configuration, never by changing another checkout's shared hook configuration.

`crates/credential-output` owns the shared pure streaming credential presentation engine; terminal and Session rendering use it without changing their response limits. `crates/model` is pure. Only `crates/store` owns SQLite connections; public handles expose typed operations. `crates/hashed-blobs` is the reviewed shared file layer plus an explicit read-only capability, documented in [source provenance](docs/shared-crate-source.json). The owner publishes the reviewed file layer as `hashed-blobs-v0.1.0`; consumer pins require their own verification. [Store limits](crates/store/README.md) distinguish these primitives from complete input acquisition and recovery.

## Delivery boundary

[Replacement disposition](docs/REPLACEMENT.md) records predecessor removal and preserved legal/history material. [Contract snapshot](docs/rewrite/README.md) links current semantic owners; canonical DDL provenance lives beside `crates/store/schema`.

The foundation source and standalone `hashed-blobs-v0.1.0` release are published. Consumer pins, runtime activation and archive migration require their own qualification. Checksum failures on main-database page reads report the actual file and page. Cold durable WAL startup uses a cancellable bounded-memory integrity guard under archive ownership; ambiguous, partial and stale-generation tails require controlled recovery without changing retained bytes. This can refuse legitimate crashes and does not prove an external expected history head. Full WAL recovery remains open. Index rebuild/swap, cross-file read views and the first canonical source-to-query product path require further work. Canonical M1 and deployment acceptance remain open.

An import reply certifies durable raw retention, independently of provider parse
progress. The client prints its UUIDv7 request ID and acquisition time before
sending; reuse both with `--request-id ID --seen-at MICROS` to retry the same
action, including after the original path disappears. A reused ID with changed
metadata is refused. Provider work runs outside the writer actor, which admits
one provider installation transaction between peer requests. `progress` reports
raw offsets, accounting, failure and pending finalization; EOF does not make
`canonical_complete` true. `retry INPUT` explicitly retries a failed disposable
parse. `--wait`, `--dry-run`, archives/directories, symlinks and other source
formats remain unavailable. Claude context is explicitly supplied with a version 1
`--context FILE` declaration and persisted with the retained input; automatic
format recognition remains unavailable.
The declaration names interpretation context, not the collector machine or proof
of provider freshness.

The actor uses fixed record/cell/request admission allowances. Oversized records
fail explicitly with owed work retained; configurable larger allowances and full
resource qualification remain unfinished. Its current source scheduling follows
one imported input to provider EOF; cross-input fair parse rounds and canonical
finalization are still owed. Public search refuses even after provider EOF.

The canonical worker prepares uncontested Codex Text/Thinking and supported
Claude Messages through the same durable private roots and field-origin checks,
then selects a whole certified source body. Claude signatures and derived tool
reasons are retained without increasing source-presence rank. Declared legacy
call identity uses the proved source Session and supplied call ID; global
pairing and cross-input operation selection remain unfinished. Terminal Message-reference and Session JSON reads are exposed; search and export remain unavailable. Competing inputs, unsupported operations and unresolved
mapping remain pending; `canonical_complete` remains false. Hash validation is
cooperative off actor but can still process a whole Message/value per worker
call, and normalized-leaf windowing can repeat serialization. These costs have
not earned bounded worker-turn or append-local resource acceptance.

Read one selected Message with `polylogue --archive DIR message:<opaque-id>
read --format terminal`. This uses the stored Session relation, preserves one
whole selected body and its interleaved reasoning/text order, and streams UTF-8
payload ranges without collecting the full Message. Control characters are
escaped for terminal safety. The command explicitly exits 5 (degraded) because
history and complete source-locator coverage remain incomplete. The terminal renderer supports Text and Thinking blocks. Pending references, truncated streams and unsupported formats fail explicitly. The explicit `message:` route does not accept JSON format.


Read a known Session with `polylogue --archive DIR SID read --format json`. This streams a normalized `polylogue-session` version 1 document from one pinned Session/Message snapshot, using the same document writer intended for export. Supported Message fields include typed UTC millisecond times, usage/service tier, retained signatures, tool payloads/channels, exact source citations and field-origin descriptors. Text streams in bounded ranges; an individual nontext JSON leaf above 64 KiB is explicitly refused. Credential-bearing metadata strings are masked; structural keys, identities or interpretation selectors requiring a change are refused. The result exits 5 with a coverage warning. Known pending members and ambiguous source order refuse rather than silently return a selected subset. Source catalog completeness, display-title reductions, non-Message populations, conflict-alternative export and partial-excision verdicts remain unavailable. Omitted populations are not fabricated as empty arrays. This path does not make `import --wait` or `canonical_complete: true` available.

Import accepts paired `--machine ID --host LABEL` acquisition-observation metadata. These identify the collector, not the original session machine or the immutable interpretation context. Without the pair, the client reads `/etc/machine-id` and the local hostname. Explicit retries must reuse the original observation metadata, request ID and timestamp.

Selected terminal body reads and Session JSON string values apply the fixed shared `credential-output-v1` catalog through the daemon’s streaming masker before socket encoding. All 22 families are implemented, including contextual secret-only extents, Unicode boundaries, non-HTTP URL userinfo and matching-label PEM bodies. Markers show the credential kind without a preview. Undecided text above 512 UTF-8 bytes or more than 2,048 simultaneous regex candidates returns an explicit error; recognized token runs and PEM bodies stream without accumulating their contents. The pinned 132-vector module qualification and precise limits are documented in [credential masking](docs/rewrite/credential-masking.md). Retained bytes and canonical values are unchanged. Derived `has:secret`/status counts and reveal remain unsupported. Module conformance does not qualify every response encoder or complete product privacy behavior.

## Morrow client qualification

`morrow_demo` runs `scripts/check-morrow.py` through the declared development
environment. Supply `--spec-repo`, an explicit `--spec-revision`, and a new
absolute `--output` directory. It builds this checkout's Nix package, retains the
selected committed synthetic fixtures/oracle, and launches only that package
with isolated archive, home and runtime paths. Intended checks cover repeated
import, append at the same provider pathname followed by immutable snapshot
import, restart, empty/malformed inputs and read/export agreement. The append
changes only the suffix and records before/after sizes and digests. These
imports prove snapshot union, not continuation of one observed input/chain.
Each loose Claude file uses explicit `--format claude-code-jsonl` and the fixture's
v1 main-transcript context through `--context FILE`. The retained context file
declares the native session independently of its opaque public reference and
filename, including empty and malformed inputs. Automatic format recognition
is outside this run's qualification.
Observed-source append, separator-only progress and restart continuation remain
separate unfinished acceptance. Every command retains argv, exits and raw output; failures leave
later steps explicitly unrun and stop the exact owned daemon.

The driver currently fails on unavailable canonical clients; it does not
qualify those clients. `--control launch-failure` deliberately collides the
archive with a file, while `--control corrupt-answer` corrupts a captured real
read before comparison. Each control must return nonzero and record
`control_exercised: true`, plus `control_result.state: launch_refused` or
`oracle_rejected` respectively. A nonzero exit alone cannot qualify a control.
The corruption control first validates and preserves the original real answer;
the planted answer is a separate artifact. Oracle acceptance and unrelated
failures remain distinguishable. `scripts/test-check-morrow.py --checker PATH`
tests that reporting boundary with synthetic replies and broken oracles; optional
`--binary /nix/store/.../bin/polylogue` adds the real launch refusal. These are
controller checks, not canonical-product acceptance.
The base/append cases require `import --wait` to affirm `canonical_complete: true`
before comparing answers. The malformed case instead requires degraded exit5,
`canonical_complete: false`, and the retained interpretation failure in result
diagnostics/unknown coverage through wait, listing, read and export. Its known
rows still face the unchanged oracle; freshness may be true despite degradation.
Standalone documents and manifests must retain their own result qualifiers. A missing, null or mistyped field cannot stand in
for completion; this assertion alone does not qualify source accounting.
Source accounting, provenance, observed durability boundaries, provider loss
and real MCP acceptance remain separate unfinished checks in the report.

Every run writes `result.json` and a local `report.html`, including failures.
The HTML shows scenario outcomes, independent expectations, captured-answer
links, exact commands/exits, owned processes and remaining acceptance. It is a
view over retained evidence; rendering cannot turn a failed comparison into a
pass. New runs record start/end times and invocation. Older JSON can be viewed
with `python3 scripts/morrow_report.py PATH/result.json`; missing historical
timestamps remain unknown. Artifact links are relative and refuse files outside
the report tree, so copy the whole run directory when relocating it.

