# Polylogue Rust foundation

This repository contains the standalone Rust foundation. It implements foundation identity/value contracts, durable SQLite schema checks, shared content files, one archive writer, read-only handles and a real daemon/status client. Explicit immutable Codex and context-declared Claude regular-file imports and provider-state progress are registered. Selected source Messages have a streaming terminal reference read, and exact native Claude Session reads can prove a supported normalized JSON/terminal shape. Session listing and human/model prose search share one typed daemon query; general queries retain explicit degraded coverage. A thin MCP stdio adapter exposes retained search, bounded reads, status and attributed note/correction writes. Full canonical finalization, general export, full MCP capabilities and HTTP remain unfinished. Status reports `product_path: "partial"`; an initialized empty schema is not a completed ingestion or M1 demonstration.

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

## Controlled copy recovery

`polylogue --archive SOURCE recover --to NEW_DEST` is a one-shot daemon mode. The source must be stopped, with its existing archive lock. It copies and hash-verifies independent files, admits one complete previous-generation durable WAL tail after a checksum-valid committed current prefix, and recovers only the copy through pinned SQLite and the checksum VFS. Malformed/current-generation damage, partial or mixed tails, busy sources, overlapping paths and existing destinations refuse. The first path supports the present plain file blobs written by this foundation; other blob layouts refuse explicitly.

The command validates database integrity, immutable identities, blob and retained-revision bytes, then reopens through the production reader. It retains exact source-manifest bytes and a checks receipt under `NEW_DEST/recovery-receipts/`. Publication is an atomic non-replacing rename after syncing. Failed owned staging is named in the error; a killed operation may leave `.polylogue-recovery-*` staging beside the destination. Neither is the requested destination. A completed command returns **degraded exit 5**, with the usable destination and typed `recovery` qualification. It starts no ingestion and adopts no publication lineage. Independent historical acknowledgements must be compared separately; internal integrity cannot establish the latest acknowledged head.

Archive format **2** persists this qualification in `archive.db`, and older binaries refuse it. Status, retained reads and matching or empty queries retain the qualification after ordinary writes, restart and index recreation. There is no clearing command. Healthy format-1 archives migrate under the ordinary daemon owner, after preserving the exact database main/WAL/SHM set under `migrations/v1-to-v2-*`; the migration changes format and advances the local write counter, preserving retained identities and bytes. A cold standalone reader requires the daemon to perform that migration. No general corruption recovery, index rebuild/swap, portable export/import or service cutover is implied.

## Development

The declared AgentCTL operations run the Nix build/check and development-shell commands. `cargo run -p xtask -- check` checks target formatting, actual Cargo dependency boundaries and workspace tests. A synthetic Cargo-edge mutation and compile-fail examples demonstrate forbidden access. `.githooks` contains successor hooks; in linked worktrees select them through per-worktree Git configuration, never by changing another checkout's shared hook configuration.

`crates/credential-output` owns the shared pure streaming credential presentation engine; terminal and Session rendering use it without changing their response limits. `crates/model` is pure. Only `crates/store` owns SQLite connections; public handles expose typed operations. `crates/hashed-blobs` is the reviewed shared file layer plus an explicit read-only capability, documented in [source provenance](docs/shared-crate-source.json). The owner publishes the reviewed file layer as `hashed-blobs-v0.1.0`; consumer pins require their own verification. [Store limits](crates/store/README.md) distinguish these primitives from complete input acquisition and recovery.

## Observed Claude source subset

One explicit regular Claude main location can now be retained at startup with
`daemon --config FILE [--until-idle] [--json]`. Verified growth preserves its
input/chain and parses retained pieces through the existing source owner. The
[observed-source guide](docs/rewrite/observed-source.md) documents configuration,
source-row qualification, separator and unfinished-tail behavior, retained
`input cat ID` presentation, and the independent client demonstration. Default
input presentation is credential-masked; source offsets and hashes remain raw
coordinates. Directory discovery, watchers, rewrites and full M1 remain open.

## Delivery boundary

[Replacement disposition](docs/REPLACEMENT.md) records predecessor removal and preserved legal/history material. [Contract snapshot](docs/rewrite/README.md) links current semantic owners; canonical DDL provenance lives beside `crates/store/schema`.

The foundation source and standalone `hashed-blobs-v0.1.0` release are published. Consumer pins, runtime activation and archive migration require their own qualification. Checksum failures on main-database page reads report the actual file and page. Cold durable WAL startup uses a cancellable bounded-memory integrity guard under archive ownership; ambiguous, partial and stale-generation tails require controlled recovery without changing retained bytes. This can refuse legitimate crashes and does not prove an external expected history head. The explicit copy path below handles one measured reused-tail shape; general WAL recovery remains open. Index rebuild/swap, cross-file read views and the first canonical source-to-query product path require further work. Canonical M1 and deployment acceptance remain open.

An import reply certifies durable raw retention, independently of provider parse
progress. The client prints its UUIDv7 request ID and acquisition time before
sending; reuse both with `--request-id ID --seen-at MICROS` to retry the same
action, including after the original path disappears. A reused ID with changed
metadata is refused. Provider work runs outside the writer actor, which admits
one provider installation transaction between peer requests. `progress` reports
raw offsets, accounting, failure and pending finalization; EOF alone does not
establish source-row settlement. `retry INPUT` explicitly retries a failed disposable
parse. `--dry-run`, archives/directories, symlinks and other source
formats remain unavailable. Claude context is explicitly supplied with a version 1
`--context FILE` declaration and persisted with the retained input; automatic
format recognition remains unavailable.
The declaration names interpretation context, not the collector machine or proof
of provider freshness.

`import --wait` waits for the admitted immutable source basis and its required
canonical row effects. Its typed `source_rows` result is independent of Session
completeness, global freshness and search readiness. Existing step dispositions
and current source certificates are audited before a durable source receipt is
published; unsupported effects or retained malformed records remain named
unavailable. Usable partial rows give degraded exit 5; otherwise a typed error.
`--wait-timeout SECONDS` controls only the caller's post-admission wait (default 30,
maximum 86400). One absolute deadline covers progress connect, writes and the
complete response frame; trickled bytes do not extend it. An already-settled
admission passes a zero-second check without a progress RPC. Timeout/disconnect
leaves retained owner work runnable. Retry the
same receipt after provider deletion/restart. Import/progress no longer expose a
hardcoded `canonical_complete` placeholder; Session read coverage remains explicit.

Exact native Claude Session qualification traverses the pinned retained admission catalog, including contributors with no selected rows. It checks source-row receipts, retained Session/response/contribution cells and the expected selected Message membership/order. Supported complete reads require a provider title, agreeing metadata, known authors and a linear main conversation without unresolved lineage, reference-link, event or companion products. Times reduce the selected own Message times, and `models` counts selected named model Messages. Expected/active Message membership shares one admission across all contributing inputs. Missing fields stay absent; competing metadata remains qualified. Complete overlapping immutable snapshots with equal normalized bodies and matching raw evidence/context use conservative compatibility election. Nonidentical candidates and general field fill remain unavailable. A proven read emits empty non-Message arrays only after proving their absence from the relevant retained effects and existing products.

Selection and normalized-document proofs have different dependency scopes. Exact selection can exclude another main file's owned Messages by its retained context. A document must also cover incoming links: an unfinished foreign main remains a possible contributor; a completed one is excluded only after its retained effects/dispositions prove no relevant link or unknown record. Subagent/unknown shapes remain unresolved. Malformed records count once per retained input, including malformed-only inputs. `result.settlement` states the operation, fixed target basis and serving view separately. Later admission creates a new basis; it cannot upgrade an old proof or hide pending newer rows. Global `up_to_date` reports caught-up source interpretation independently. Deadline/disconnect stops read-side work without cancelling retained owner work.

The actor uses fixed record/cell/request admission allowances. Oversized records
fail explicitly with owed work retained; configurable larger allowances and full
resource qualification remain unfinished. Its current source scheduling follows
one imported input to provider EOF; cross-input fair parse rounds and complete Session
finalization are still owed. Known selected prose is searchable with explicit population qualifications; provider EOF does not remove them.

The canonical worker prepares Codex Text/Thinking and supported
Claude Messages through the same durable private roots and field-origin checks,
then selects a whole certified source body. Claude signatures and derived tool
reasons are retained without increasing source-presence rank. Declared legacy
call identity uses the proved source Session and supplied call ID; global
pairing and cross-input operation selection remain unfinished. Terminal Message-reference, Session JSON and known Session query reads are exposed; exact Session JSON directory export consumes the same pinned read. Complete immutable snapshots with identical source evidence, interpretation context, rank, normalized Message bodies and source order share one selected content carrier while retaining each input's exact citations and origins. The carrier uses the validated content-derived InputId as a provenance tie, independent of import arrival. Unequal bodies, ambiguous within-input versions, unsupported operations and unresolved mapping remain pending; complete qualification is limited to the exact shape above. Hash validation is
cooperative off actor but can still process a whole Message/value per worker
call, and normalized-leaf windowing can repeat serialization. These costs have
not earned bounded worker-turn or append-local resource acceptance.

Read one selected Message with `polylogue --archive DIR message:<opaque-id>
read --format terminal`. This uses the stored Session relation, preserves one
whole selected body and its interleaved reasoning/text order, and streams UTF-8
payload ranges without collecting the full Message. Control characters are
escaped for terminal safety. The command explicitly exits 5 (degraded) because
history and complete source-locator coverage remain incomplete. The terminal renderer supports Text and Thinking blocks. Pending references, truncated streams and unsupported formats fail explicitly. The explicit `message:` route does not accept JSON format.


Read one Session with `polylogue --archive DIR SID read --format json`, or use `SID read` / `SID read --format terminal` for terminal output. JSON streams a normalized `polylogue-session` version 1 document from one pinned Session/Message snapshot. Terminal output consumes that same qualified document, with a 16 MiB client admission, and displays text, thinking, tool calls and results with control characters escaped. The supported exact shape exits 0; other usable shapes retain degraded exit 5 and named diagnostics. Known pending selected members and ambiguous order refuse before document bytes. A nontext JSON leaf above 64 KiB is refused. Credential-bearing strings are masked; identities or structural selectors requiring a change are refused. Exact source citations, field origins and source-backed payloads retain their existing writer. Exact Session JSON directory export uses this same document and qualification; productive external-result companion acquisition remains unavailable.

Export one exact native Session with `polylogue --archive DIR SID export --format json --to DIR`. The existing Session read streams the detached document unchanged into a private staging directory. `manifest.json` retains its actual result and recovery qualification, selected index generation/counter, actual freshness, exact Session reference and UTC run time. Each Session goes to `<origin>/<percent-encoded native ID>.json`; opaque IDs are never split into path components. Healthy export exits 0; usable degraded export exits 5. An unknown Session fails without publishing an empty export. `with_notes: true` records the default inclusion policy. This supported shape emits empty authored `notes[]`; it does not qualify general authored-note preservation or roundtrip.

The destination parent must already exist. A new directory is published with a no-replace rename. An existing directory is replaced only when it has a recognized version 1 export manifest; a symlink or unrelated directory refuses. Files and directories are synced before Linux atomic directory exchange. Errors before publication preserve the prior export. Ctrl-C uses ordinary SIGINT termination, closing the client socket even while IO is blocked. It can leave private `.polylogue-export-*` staging beside the destination, without automatic cleanup or reuse. If the atomic rename already occurred, the completed replacement may be visible and the prior tree may occupy that private name; inspect both before cleanup. A detected rename race or post-rename sync failure retains the exchanged directory at the reported private staging path and reports uncertain publication; inspect both paths before retrying. Concurrent external mutation and general power-loss behavior are not qualified. Query selection, JSONL, Markdown, profile flags, blob bundles, note/inheritance options and normalized reimport remain unsupported.

Import accepts paired `--machine ID --host LABEL` acquisition-observation metadata. These identify the collector, not the original session machine or the immutable interpretation context. Without the pair, the client reads the configured machine-id file (default `/etc/machine-id`) and the local hostname. Explicit retries must reuse the original observation metadata, request ID and timestamp.

Selected terminal body reads and Session JSON string values apply the fixed shared `credential-output-v1` catalog through the daemon’s streaming masker before socket encoding. All 22 families are implemented, including contextual secret-only extents, Unicode boundaries, non-HTTP URL userinfo and matching-label PEM bodies. Markers show the credential kind without a preview. Undecided text above 512 UTF-8 bytes or more than 2,048 simultaneous regex candidates returns an explicit error; recognized token runs and PEM bodies stream without accumulating their contents. The pinned 132-vector module qualification and precise limits are documented in [credential masking](docs/rewrite/credential-masking.md). Retained bytes and canonical values are unchanged. Derived `has:secret`/status counts and reveal remain unsupported. Module conformance does not qualify every response encoder or complete product privacy behavior.

## Morrow client qualification

`morrow_demo` runs `scripts/check-morrow.py` through the declared development
environment. Supply `--spec-repo`, an explicit `--spec-revision`, and a new
absolute `--output` directory. It builds this checkout's Nix package, retains the
selected committed synthetic fixtures/oracle, and launches only that package
with isolated archive, home and runtime paths. Socket files use an owned short
`/tmp` directory, removed after the daemon stops; archives, captures and reports
stay in the selected output directory. Long result paths do not lengthen socket
names, and `result.json` records runtime cleanup. Intended checks cover repeated
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

The full driver retains its ordinary listing/read/export assertions and independent content oracle. Exact Session directory export is implemented; byte-identical complete overlapping snapshots use conservative compatible-candidate selection; nonidentical ranking/fill remains unavailable. `--control launch-failure` deliberately collides the
archive with a file, while `--control corrupt-answer` corrupts a captured real
read before comparison. Each control must return nonzero and record
`control_exercised: true`, plus `control_result.state: launch_refused` or
`oracle_rejected` respectively. A nonzero exit alone cannot qualify a control.
The corruption control first validates and preserves the original real answer;
the planted answer is a separate artifact. Oracle acceptance and unrelated
failures remain distinguishable. `scripts/test-check-morrow.py --checker PATH`
tests that reporting boundary with synthetic replies and broken oracles; optional
`--binary /nix/store/.../bin/polylogue` adds real launch refusal and daemon
startup/cleanup under a long result path. These are
controller checks, not canonical-product acceptance.
The driver requires `import --wait` to return typed `source_rows` for the
admitted input: settled for base/append/empty, unavailable/degraded5 with usable
rows for malformed. This source-only proof does not satisfy its later ordinary
listing/read/export assertions, which still require complete results and the
unchanged independent oracle. Missing or mismatched scope/basis cannot stand
in for that proof.
The controller checks alone do not qualify native source accounting.
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

## Known Session listing and prose search

`polylogue [QUERY] [list] [--json]` and `polylogue find QUERY [--json]` use the same daemon query capability. Bare `polylogue` lists known Sessions. The registered subset accepts `sessions [where]`, conjunctive `origin:`, `kind:main|chat|subagent|*`, exact `id:` or full origin-prefixed Session IDs, alphanumeric words, double-quoted prose phrases, word prefixes such as `refact*`, `| sort by last_active desc` and a final `| limit N`. Unknown fields, units, flags, Boolean shapes and stages refuse before acquiring the read view. The full slice-1 query language and relevance ranking are not registered. See [the query contract](crates/query/README.md).

The daemon traverses the entire pinned retained admission and selected Session/Message catalog. Text search uses SQLite FTS5 membership over a query-local corpus built only from selected human/model Text blocks, with case/diacritic and stroke-letter folding. Thinking, tool output, harness and unknown authors do not enter prose matching. Every term must match the same Message, and a phrase must stay within one Text block. Canonical distinct-text relevance statistics remain unavailable; the response explicitly uses known Message activity order. Preparation/staged bodies cannot affect matching. There is no newest-row search window. The query-local corpus is an intermediate serving implementation, not a published archive FTS index or a performance qualification.

Rows expose stable Session IDs, selected catalog metadata, `known_messages` and `known_last_message_at` (provider milliseconds). Listing order uses this known Message time, with unknown times last and stable ID ties. These are partial selected-row facts, not completed Session reductions. Search hits carry retained Message IDs; snippets remain unavailable because cutting before credential presentation can remove required masking context. Open the Session with `SID read --format json` or a hit with `message:MID read --format terminal` through the existing readers. Structured payloads and origins retain their owning representation.

`count` is the number of known candidates before an explicit selection limit; `next_cursor: null` ends this admitted enumeration. No continuation is registered. General nonempty retained populations and unmatched prose searches over them remain degraded (exit 5), with named failures, pending inputs, unresolved dependency shapes and unknown authors. Malformed-only input participates even when it published no matching row. Only an accessible empty retained population with no outstanding relevant interpretation obligation yields empty (exit 1). Global `up_to_date` reports whether every admitted input has a validated terminal source-row receipt; it does not certify complete searched-population settlement. An exact supported selection can be settled while unrelated source work keeps global freshness false. Deadline/disconnect stops read-side work without cancelling retained interpretation.

Admission currently permits 20,000 catalog Sessions and prose Messages, 4 MiB of prose per Message, 32 MiB of selected prose per query, and 4 MiB of accumulated result metadata/encoded response. Exceeding an allowance fails the whole query instead of truncating its searched population. SQLite scratch work and whole admitted Messages remain coarse work units; these are logical admission limits, not whole-process CPU/RSS guarantees. General fields, linked subagent promotion, inherited/active-path completeness, snippets, paging, persistent search publication, general export, general MCP search and full Morrow/M1 acceptance remain open.

`scripts/check-session-query.py --binary /nix/store/.../bin/polylogue --output /absolute/new/private/directory` exercises an installed package against independent synthetic fixtures, including an oldest-only match behind 110 newer nonmatches, failed-population qualifications and retained navigation after provider removal and graceful restart, plus SIGKILL/restart in a separate small cohort. Large-cohort SIGKILL can reach the existing ambiguous WAL-generation refusal; that recovery limitation remains open. It records exact argv, exits, outputs and owned process cleanup without changing the Morrow oracle.

`scripts/check-exact-session.py --binary /nix/store/.../bin/polylogue --spec-repo /realm/inbox/spec --spec-revision COMMIT --output /absolute/new/private/directory` runs the exact selection/read stage separately. It retains committed Morrow fixtures and the unchanged independent oracle, attempts repeat/append/restart/provider removal, JSON and terminal contents, metadata and malformed/empty results, and exercises pending/unknown dependency controls. The current conservative compatibility path admits identical overlapping bodies and their append/restart controls; nonidentical competing candidates remain unavailable. Historical stopped-stage reports retain their original unrun steps. It records the frozen candidate, immutable binary digest, commands, source receipts and process cleanup. This stage does not claim the full export demonstration or M1 acceptance.


## Bounded model reading

`polylogue REF read --budget TOKENS [--continuation TOKEN] [--view dialogue]` returns a compact structured model page on stdout. Either flag selects this operation; a continuation supplies its last budget when no new budget is given. `next.ref` and `next.continuation` are arguments for `polylogue REF read --continuation TOKEN`. Explicit capacity increases/decreases refit the end at the same starting position. Whole `read --format json` documents and directory exports retain their existing byte semantics. Combining model flags with `--format`, destinations or other rendering flags refuses.

The first supported selection is a native Session's active Text/Thinking blocks, a `message:` reference, or an exact `block:` ordinary/reasoning reference with supported preceding logical-body blocks. Thinking is intentionally omitted except for an exact block. This default remains the named native subset, not a full transcript. Explicit `--view dialogue` supports the human/model Text portion of dialogue and reports omitted block counts by author, body kind and reason, including tools and harness content. Human attachment/reference rendering remains intended dialogue behavior; this native subset refuses unsupported included kinds rather than reporting them as intentional omissions. An exact supported block explicitly selects its full Text/Thinking body even with dialogue. The default subset still refuses included tools. Explicit `--view transcript --tools full --thinking full --harness full` includes Text, Thinking, ToolCall and ToolResult blocks, with typed Message metadata and tool fields. Optional boolean `--branches` on that full selection enumerates known retained Session-catalog members in catalog order, including abandoned or uncertain-path members. It preserves degraded record errors, notes and unavailable settlement; `read_as` states the narrower scope. It asserts no active-path membership or complete topology and provides no invented branch indentation. Default active-path reads still refuse unavailable membership. Existing current row-audit authority, pending work and visibility/incarnation fences remain required; equal error counts are never authority. A selected zero-block Message refuses explicitly because this block-oriented representation cannot preserve its metadata. Other detail/collapse options, included attachments, unknown body kinds, ranges, historical views and linked history remain unavailable. Structured fields and textual tool output use the existing 65,536-byte normalized JSON admission; required oversized metadata refuses. Present empty output stays distinct from absent output. `shown` carries each fragment, its actual Message/block identity, author, exact retained source citations, field-origin descriptors and half-open Unicode scalar display span. `next: null` finishes this selected presentation, independently of result completeness. Every usable page retains current qualification, recovery, freshness, UTC zone and the archive-text warning. Pending/unavailable relevant source effects refuse instead of continuing an uncertain current body.

The shared operation is `polylogue::model_read::execute`, called by the daemon adapter. It uses typed immutable store views and the engine's existing Session qualification. Certificates frame sealed normalized value hashes, actual attribution and exact source-origin descriptor hashes. Equal text with changed preferred origins stales; implementation-only body row replacement is insufficient to stale. A different unrelated retained source can advance the serving view without invalidating an exact block. Tokens contain hashes and numeric positions only, authenticated by a daemon-instance key. Restart/key loss refuses an old token; a fresh retained read remains available after provider removal. No old view or payload is retained for continuation service.

Credential-output-v1 replays preceding Text, Thinking and textual ToolResult output in the same Message, even when hidden. Ordinary recognizers end at each block boundary; recognized PEM suppression and END progress survive until the Message ends. ToolCall fields and structured result channels are independent strings and cannot change this body carry. Fully excluded Messages require no replay. Necessary ToolResult output uses the same typed normalized-field decoder as JSON export, admitted at 65,536 encoded bytes; larger/conflicting fields and unknown necessary body kinds refuse before usable data. Arbitrary huge tool-body support remains unavailable. Scalar/escape/replacement boundaries stay indivisible. The reader retains a finite display window and fits it by measured compact UTF-8 JSON size using `ceil(bytes/4)`; socket frames are transport only. CLI has no duplicated text companion. MCP requests select the shared `mcp` presentation, which fits its ready MCP follow-up call and fixed short text companion in this same operation. Budgets below 128 receive the explicit minimum diagnostic envelope; insufficient supported budgets return a sufficient capacity for retrying the same reference and cursor. Supported capacity is at most 1,048,576 estimated tokens; larger required metadata or existing typed metadata admissions refuse explicitly. Body scans replay prefixes, so CPU cost is not append-local; existing source-effect cells and qualification membership remain admitted allocation units. Whole-process/corpus resource acceptance is unqualified.

The request uses the existing daemon worker and absolute 30-second read lifetime, with cooperative disconnect checks. Live removal epochs/generation and the Session's current excision authority are rechecked before handoff, and current selected bodies are re-resolved. Native authored privacy/excision products and their broader acceptance remain unavailable. The MCP adapter below reuses this operation; the broader agent-read/M1 capability remains incomplete.

`cargo test --locked -p polylogue model_` exercises actual native CLI/daemon Unicode/repeated-identity reconstruction, credential expansion, budgets, late native metadata growth and same-cursor retry, unrelated-source append, preferred-origin changes, default mixed-tool refusal, explicit dialogue on unchanged Morrow, mixed-block masking boundaries, admitted tool output and provider-loss/restart. Store certificate tests cover actual normalized changes and removal-epoch invalidation. `scripts/check-bounded-read.py --binary /absolute/bin/polylogue --output /absolute/new/private/directory` is the independent binary oracle; it retains all calls/results and checks required native metadata larger than a transport frame. Run it through the declared `bounded_read_demo` operation. Broad privacy/removal products and full Morrow/M1 remain open.


Machine attribution for authored CLI/MCP writes and the import default reads `/etc/machine-id`. Processes without that host path can configure `POLYLOGUE_MACHINE_ID_FILE` to an absolute file containing the actual or explicitly assigned machine identity. It must contain a nonzero 32-character lowercase hexadecimal machine-id, with at most 128 file bytes including surrounding whitespace. Missing or malformed identity refuses before sending; Polylogue never invents a hostname-based identity. This process configuration controls machine attribution only; it cannot change the harness-derived author or grant operator authority. Preserve the same identity configuration when retrying a write. Explicit import `--machine/--host` observation metadata retains its existing meaning.

## Attributed correction tracer

Use `polylogue --archive DIR REF note "Final text" --kind correction --request-id UUIDv7`, where REF is one exact `message:` or `block:` reference. The default kind is `note`. `--corrects REF` records a descriptive relation, including an uncollected exact target. `--target-basis FILE` accepts an array of `{ "ref": "...", "seen_hash": "<64 lowercase hex>" }`; creation requires the same selected target. Omit `seen_hash` to record unknown basis. The CLI prints a generated request ID before sending; retain and reuse it for a lost-reply retry. An unchanged action returns the original immutable effect even after later revisions. Changed intent with that ID conflicts. A lost transport reply carries `request_id`, `commit_state:"unknown"` and same-ID retry advice. If an already committed outcome cannot preserve inherited identities or fit the metadata reply, it returns degraded `presentation:"retained_but_unpresentable"` with `commit_state:"committed"` and its safe action/note identity. It never reports that presentation failure as no effect. Both clients check returned action/note/revision identities against the submitted request; a mismatched success becomes commit uncertainty with the original request ID. Omitted creation time is resolved only by the original commit.

`note:<root>@<full-revision-UUIDv7> revise "Replacement text" --request-id UUIDv7` refuses a stale revision. Omitting `--target-basis` preserves the earlier targets and supplied hashes; an explicit replacement file changes them. `note:<root> withdraw --reason "Reason" --request-id UUIDv7` removes the family from its effective state. Withdrawing a newer pinned revision restores its earlier available head; withdrawing that withdrawal restores the newer revision. Control parents must already be retained, and combined revision/withdrawal controls must be acyclic. Timestamps never elect a head. Generic targets and `corrects` may await acquisition.

Read `note:<root> read --budget 12000` for current state or pin a full revision ID for history. Results preserve immutable `revision.targets`, author, machine, revision links, history and withdrawal trail. Live `target_status` and aggregate `stale` are separate: a differing comparable basis is true, all comparable matching bases are false, and insufficient evidence is null. A supplied basis prepared before target collection stays exactly supplied when a different target arrives. An omitted basis stays unknown. CLI attribution is the known harness session, otherwise `agent` in a harness, `operator` for an interactive terminal outside a harness, or `script` for redirected use.

Exact Text/Thinking block reads and provably single Text/Thinking-block Message reads expose `authored_basis` with `projection:"decoded-utf8-text-v1"`, `covers:"whole_target"` and an ordinary BLAKE3 `seen_hash` of the selected decoded UTF-8 body before credential masking. This hash can cover body text beyond the displayed page. It is attributed evidence supplied to authoring, not proof someone read the text. It differs from source hashes and selected-body continuation certificates. Hashing uses admitted incremental decoded ranges; the witness and backlinks count toward the normal presentation budget. Composite, structured or encrypted targets retain supplied basis but comparison remains unknown. Exact target reads expose historical matching and currently matching revision references separately, so retargeted revisions cannot masquerade as current attachments.

The existing archive writer, `client_request` identity and note tables own this path; index reconstruction and provider removal do not erase authored history. No archive schema or pinned SQL comments change. Input admission remains 8 targets, 16 KiB final text and 64 KiB request metadata. Effective-state evaluation reads topology without historical value bodies, using indexed point/keyset steps and request-local disk scratch in the process-owned `TMPDIR`. Its SQLite page-cache target is 64 KiB, scratch is capped at 16 MiB, and each evaluation has a million-step allowance plus the existing absolute deadline/cancellation checks. Scratch is removed on success, error and cooperative cancellation. Prospective evaluation covers all reached control families and new target-membership populations inside the writer transaction, after committed-request replay; capacity refusal rolls back the new note, request identity and counter. These are request resource bounds, not lifetime limits on a revision family. Exact body hydration has a separate 256 KiB admission; current-reference responses admit 128 heads. Encoded authored presentation must fit 64 KiB and the selected estimated-token budget, otherwise it refuses explicitly. History and related annotations have bounded continuation, described below. Other kinds, ordinal revision pins, general notes queries, tags, inline notes, judgments, pages, anchors, checks, expiry, bulk writes, authored-return and portable authored export/reimport remain unavailable. Native import/union of authored control graphs is unimplemented. This is a selected durable interpretation path, not full slice 7, M0 or M1 acceptance.

Authored reads use the same public reader: `polylogue note:ROOT read --budget 12000`, or MCP `read {"ref":"note:ROOT","budget_tokens":12000}`. A default family page shows selected exact content, current references, up to 32 historical references and a bounded withdrawal trail. Follow `family.history_next`, `family.withdrawals_next` and `family.current_next` as named read calls. Explicit `view:"history"`, `"withdrawals"` and `"current"` select those products; a null completion flag means the product was not selected. The CLI equivalent is `--view history|withdrawals|current` with `--continuation TOKEN` from the emitted call. Current pages require an unpinned family reference. A pinned `view:"value"` (`--view value`) retrieves its immutable value independently of effective-state evaluation; it always returns `status_evaluated:false`, `current:null`, `withdrawn:null` and a degraded qualification. Retrieval does not make historical content current.

Page tokens bind archive identity, exact reference, product, original archive counter and a unique physical row key. They are process authenticated and expire across daemon restart. All relevant revisions, targets and control edges are evaluated at the retained `authored_archive_counter` / `status_at_archive_counter`; later writes cannot alter only the tail of a traversal. Later status is explicitly not current-now. A fresh read selects the new basis. The native append-only note tables retain stable row keys during this process; arbitrary external archive rewriting, authored import/union and authored visibility/excision are unsupported. Exact membership remains checked on every read. Target attachment and stale-basis comparison use the live retained source selection, separately labelled from effective status at the authored page basis.

Primary source content and its annotations have separate completion. `primary_content_complete` describes the source-body continuation. `authored_associations_complete:false` degrades the contextual claim while leaving usable source text and citations available. The annotation page exposes a unique retained family population, including superseded and withdrawn attachments, with `current_target_revisions` separate from historical matching references. Association embedding has a two-second cooperative allowance within the existing read deadline. Oversized or unavailable association metadata is deferred with `authored_notes.next`; call it through MCP or CLI `REF read --budget TOKENS --view annotations [--continuation TOKEN]`. Each explicit association page returns one family and a ready next call. Its family history can itself continue; `matching_target_revisions_complete:false` requires following that history and exact revision targets, rather than treating a prefix as all attachments. Source-body continuations bind the authored archive counter independently of presentation budget; a new authored context makes an older source continuation stale. This preserves the existing source-body visibility, identity, masking and citation checks.

## MCP stdio

`polylogue --archive DIR mcp` serves the official `rmcp` 3.5.1 SDK over stdio. It supports MCP 2026-07-28 `server/discover` and older initialized clients through the SDK. `status`, `search`, `read` and `note` are advertised in that order with input/output schemas. The three readers have read-only annotations; `note` has `readOnlyHint:false`. Discovery works without a running daemon. Tool calls use the same daemon, credentials fence and typed outcomes as the CLI; the adapter does not create an archive or start its daemon.

- `status {}` returns current partial daemon/source status. It does not report live agent sessions or certify M1 readiness.
- `search {"query":"words origin:claude-code | limit 20","budget_tokens":4000}` uses the ordinary supported Session query. It preserves the CLI's selection, retained-population, settlement, recovery and freshness qualifiers. The budget includes structured content and its fixed short text companion. An oversized result fails whole with an explicit suggestion to narrow the query or select `| limit N`. A compact refusal uses the same 128-token minimum diagnostic allowance. There is no search page, snippet, hits budget or cursor implementation.
- `read {"ref":"claude-code:exact-native-id","budget_tokens":4000}` uses the bounded retained reader. Exact `message:`, `block:` and `note:` references are also supported. Follow its `next` object as the next tool call. Use explicit `view:"dialogue"` for its human/model Text subset, or `view:"transcript", tools:"full", thinking:"full", harness:"full"` for the supported Text/Thinking/ToolCall/ToolResult transcript and its typed metadata. Add boolean `branches:true` to read known catalog members when active-path membership is unavailable, preserving degraded scope and current selected-row authority. The ready call carries the same reference, effective explicit selection when requested, opaque continuation and budget; capacity can be changed explicitly at the same position. Text and exact Thinking-block support, omissions, admission limits, credential masking, source witnesses and stale-continuation rules match the shared reader. Both structured content and the short text companion count toward the requested estimated-token budget. Below 128 tokens, the explicit minimum diagnostic envelope is allowed, as on the CLI.

- `note {"targets":["block:claude-code:neutral:answer:0"],"kind":"correction","value":"Final attributed correction","request_id":"01900000-0000-7000-8000-000000000001"}` uses the same writer as the CLI. Targets may instead be `{ "ref": "...", "seen_hash": "<64 lowercase hex>" }`. Revise with one `revises` relation to the exact current pinned revision and omit `targets` to retain its supplied basis. Withdraw with `kind:"withdrawal"`, no targets and one `withdraws` relation to an earlier retained root/revision. MCP supplies harness-session or `agent` attribution and never claims `operator`. Supply and retain `request_id` before sending when a retry must survive a lost reply; an automatically generated ID is returned only with the reply. Repeating the same arguments without an explicit ID creates another action, so the tool advertises `idempotentHint:false`. Caller retries must reuse the original explicit ID; the three reader tools retain `idempotentHint:true`.

Unknown or unsupported arguments fail explicitly. General notes queries, `self`, live sessions, ranges, other views, included attachments, other tool/detail modes, briefings, installation, HTTP and full query syntax remain unavailable. Results retain their normal `empty`, `degraded` or typed error meaning; MCP never manufactures an empty success for a missing daemon. Cancellation notifications and stdio input EOF close only that request's downstream socket, including blocked reads. An accepted note may commit after cancellation or a lost reply; retry unchanged intent with the same request ID. Source retention work has a separate lifetime. Connection admission remains under the same absolute 30-second deadline, with cancellable nonblocking connect retries bounded at 10 ms.

`scripts/check-mcp-read.py --binary /absolute/bin/polylogue --output /absolute/new/private/directory` runs the independent native fixture oracle through the actual SDK process, including modern discovery, legacy initialization, Unicode/repeated-body reconstruction, credentials, ready follow-up calls, required metadata beyond a frame, same-position capacity retry, empty/unavailable distinctions and downstream EOF after cancellation/disconnect. Run it with the declared `mcp_read_demo` operation. Packaged source, current visibility/removal fences and this selected agent-read path do not certify broad privacy/removal products, live corpus acceptance, activation or full Morrow/M1.

`scripts/check-dialogue-read.py --binary /absolute/bin/polylogue --output /absolute/new/private/directory --spec-repo /absolute/spec --spec-revision COMMIT` follows emitted CLI and MCP dialogue calls against unchanged pinned Morrow inputs and independently authored mixed native controls. It preserves the whole-document Morrow comparator and explicitly expects Messages 1, 6, 7, 10, 11 and 12 for dialogue. Run it through `dialogue_read_demo`. This selection does not answer omitted tool-output questions or certify full Morrow/M1.


The Morrow driver also has an MCP fact route:
`scripts/check-morrow.py --route mcp --sdk-python /absolute/pinned/python --spec-repo /absolute/spec --spec-revision COMMIT --output /absolute/new/private/results`.
It prepares unchanged full Morrow inputs, uses the ordinary official SDK helper
and checks every registered identity/text/tool/result/metadata fact against actual
returned public calls. It covers repeat, append, daemon restart, available-empty,
malformed input and retained reads after provider removal/restart. CLI is used
only for setup and source settlement; answer facts come solely from MCP. This
route preserves the existing CLI/export comparator. `--control wrong-tool`,
`missing-result` and `mcp-launch-failure` are damaging controls; `launch-failure`
retains its existing daemon-start check. An optional `--binary` records an
explicit immutable package and hash instead of building; its source binding must
come from the owning package receipt. Run through `morrow_demo`. These commands
are acceptance tools, not a claim that a particular candidate or full M1 passed.
