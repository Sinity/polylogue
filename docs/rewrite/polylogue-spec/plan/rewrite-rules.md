# Rules for the rewrite

Every bug class found in the 3,589 closed Polylogue Beads (July to October
2026) and in the whole-codebase audit of the Python version, merged by root
cause. Each rule is something every part of the rewrite must do, with how it
is enforced. A rule enforced only by "agents should remember" is not allowed
here; lessons without a concrete enforcement are in `plan/lessons.md`.

Enforcement, strongest first: **type** (cannot compile otherwise), **design**
(the spec makes the bug impossible; file and section named), **lint**
(clippy, `deny.toml`, nextest config; `plan/workspace.md`), **law**
(`plan/testing/laws.md`, by number), **drift** (`plan/testing.md`, "The spec
and the code cannot drift apart"), **example** (an input under
`examples/sources/` with its expected output), **CI** or **xtask**
(`plan/workspace.md`).

Numbers are stable. A rule merged into another keeps a one-line entry saying
where it went. Slice prompts cite rule numbers; `AGENTS.md` points here. The
"From" column lists the Beads (without the `polylogue-` prefix; "+N" is how
many more share the cause) and audit findings each rule came from; the
evidence is in `evidence/beads/closed-1.md` to `closed-6.md` and `audit/`.

## 1. Unknown, failure and absence

| # | Rule | Enforced by | From |
|---|---|---|---|
| R1 | An unknown value stays unknown from parse to output: counts, measurements, costs, times and ratios are `Option`, shown as `null` or "—", never `0`, `0.0`, `100%`, `ok` or `ready`. A ratio with an empty or inexact denominator is unmeasured, and every figure names its numerator, denominator and population. A missing quality or provenance tag reads as the weakest value, and a value made up to fill a gap is never labelled exact. | type (`Option<T>`; `UnknownReason`; `Ratio::new` returns `Option`; provenance enums default only to `Unknown`; `spec/model.md`, Value types); design (`query.md`, "Unknown is not zero"; `analysis.md` §11); law 45 | 9e5.29, b2r9, iuyr, jfeaz, 9onfa, 0glm0, njcms, kc1x0, 1dk1, 4r2r, cpf.5, gqkae, 2gyb7 +160; audit P05, P17-B1 |
| R2 | A read that fails returns outcome `error` or `degraded`, never `empty` or `ok`; a degraded side of a search is named in the result. | type (query `Outcome`: `Empty` is built only from a successful read); law 45 | s7zzs, roax, f9kk, co8b, azf7, w4hfb, 20d.17.2 +40 |
| R3 | Errors are values up to the binary's top level: no discarded `Result`, no catch-all handler except there and in the parse pool's per-input capture. | lint (`unused_must_use`, `clippy::let_underscore_must_use` denied) | 1xc.11, cpf.4, hu24g, 4riby, feqr, oitx +20 |
| R4 | Merged into R1. | | |
| R5 | Every `status` figure is measured, measured-absent or unmeasured, and unmeasured is never shown as healthy; an unreadable location or missing root is an error, never zero work. | type (`plan/workspace.md`, rules held by construction); design (`behavior.md`, Observability); law 46 | ebjar, g827i, oil1q, 1fu1a, 4b4am, lu1dk +40 |
| R6 | A check or state file that cannot be read fails closed: a safety check refuses, and a lock, token, port, spool-index or cursor file that is corrupt, partial or of an unknown version is unreadable, never "nothing recorded". | type (each loader returns `Result<T, Unreadable>`); drift (no `unwrap_or_default` on such a result); design (`storage/blobs.md`, crash safety); law 45 | 9jelj, 0ora2, 113nu |
| R7 | A shown count is the count of all matches, never the page length; when a list is shortened, the total and omitted count come from the full list, and a full-detail route returns every item. | type (`render::Shortened<T> { shown, total }`, built only from the complete iterator); design (`query.md`, Results); law 15 | 1vv, 4pm, 4ts.6, cuxz.13, i6p4x +8 |
| R8 | A missing time stays missing: a time filter excludes untimed rows and counts them in `notes`, and grouping by time has an `unknown` bucket. | type (`Option<Timestamp>`); drift (SQL inventory rejects a literal `0` fallback on time columns); law 47 | z29t, rvtu, cpf.6, srjq, s5mm, 2seq, q30k; audit P10-r2 |
| R9 | A provider time far beyond collection time is kept, flagged and never used as last activity; a stand-in value is marked and never outranks a real one in a merge. | design (`sources/README.md`, several routes; `model.md`, Time); example (`claude-code/future-timestamp/`) | tc3pp, dgr8c, exwho, m5djs; audit P07-B2 |

## 2. Nothing lost, nothing invented

| # | Rule | Enforced by | From |
|---|---|---|---|
| R10 | Every field the providers write, and every field a parser produces, is mapped to a column, kept in `attrs`, or listed in its format's ignore table with a reason. A field scan of the operator's data reports unmapped fields above 0.1% presence and degenerate fields before a slice ships. | drift (wire structs and parser output against `sources/README.md` mapping tables); type (`serde_ignored` counts); law 48; `xtask field-scan` (a report, not a pass/fail check) | cgfy, 3b0eu, vabba, sz6ed, xnk51, tn3lu, fmc8u, qrfq9, 3szyi.3, i152l, aoszy, 4b4wi, 0vpos, 1wtm, vhjs +160; audit P23-B3 |
| R11 | Every input value reaches the archive: each string leaf longer than 16 bytes reaches the export (text, `attrs` or the raw anchor) or the ignore table. | type (`plan/workspace.md`, rules held by construction); law 48 | pbuh, ah21, t5lg, 7pjy3, pirai, ksgg.4 +20 |
| R12 | Parsing is total: every record ends parsed, counted as unrecognized with its type, or refused with a reason; one bad record or ZIP member loses only itself; a file that vanishes mid-read keeps every complete record read. | type (`plan/workspace.md`, rules held by construction); law 12, law 37; design (`sources/README.md`, How sources work); example (`claude-code/vanished-mid-read/`) | b5959, vslhb, 5gdj9, b2elq, payqy, 1k9l, x1uh +15 |
| R13 | Merged into R10. | | |
| R14 | Each wire generation is a serde alias with its own example input; when a record holds a truncated and a full form, the full one is stored. | example (one per generation); design (`sources/README.md`) | 7yji2, i415, mctu, 22ldr, 3vsoj +10 |
| R15 | Numbers are parsed losslessly: integers as `i64`, others kept as the JSON number text when not exactly representable; non-finite values are kept. | design (`sources/README.md`); example (`claude-code/large-numbers/`) | 7pjy3, pirai |
| R16 | A session is written only with positive conversational evidence; a file's role comes from its location and declared shape; decoy files are examples whose expected outcome is "not a session". | design (`sources/README.md`; `FileRole`, `plan/slice-01.md`); law 73 | 31r1, gvgi, 9ykn, 21qj, 6bebe, ioz7, eo81, hbtj2 +10 |
| R17 | A session whose provider records include messages never parses to zero messages. | law 49 | i415, z1rdw, 58jjk |
| R18 | Meaning comes only from structure: `written_by`, tool outcomes and the `paste` event come from structural fields, never from content text, except fixed-format lines the harness itself writes into tool-result text, each listed in its provider's section (`sources/README.md`, classification); a prefix may choose the parser but never drop records; no keyword classifiers. | design (`principles.md`; `sources/README.md`); law 72 | uc0j2, cmgk8, sfu70, 670mf, 6r7oq, b0b.1, h01l, ve9z; audit P03-A2 |
| R19 | Every format variant is an example taken from real producer bytes (scrubbed); a parser bug is fixed by strengthening the law or generator that should have caught it, and by adding the input when no example shows its variant. | example (`plan/testing.md` layer 2); `plan/testing.md`, Rules | segf, 1frn, j2zz, zoc3, 1zex, dpc2, erf3 |
| R20 | Merged into R10. | | |

## 3. Identity and merging

| # | Rule | Enforced by | From |
|---|---|---|---|
| R21 | IDs come only from the provider's raw records, carry their full scope, and are built by one function per ID kind; never from positions, paths, sort order or wall clock. Canonical JSON keeps escapes, surrogates, number text and the difference between absent, `null`, empty and literal; occurrence numbers never reuse a gap; a lone surrogate in a native ID is kept by a reversible encoding. | type (ID constructors in `model`); law 10; drift (reference-stability snapshot); design (`model.md`, Identity); example (`claude-code/lone-surrogate-id/`) | 85z0, hith, qkuq, 7zp4, gysk3, slshy, aki9t, sf7ii, vp5qk, 31zag +40; audit P08 |
| R22 | Comparing two versions of a record looks only at content fields, never at acquisition state, position, volatile metadata or thinking signatures. | law 50; design (`sources/README.md`, several routes); example (`claude-code/resigned-thinking-fork/`) | nuec, bu1i, flxh, hith, qkuq |
| R23 | Two sessions sharing a provider ID stay distinct; the same bytes at two paths stay one. | law 51, law 40 | foi1h, irtix, uxtke |
| R24 | A key collision on a derived row is an error or goes through the merge rule; `INSERT OR REPLACE` and `INSERT OR IGNORE` are not used. | drift (SQL inventory) | 85z0, 1xc.5, vwia, fmob |
| R25 | Merging is a union by identity: arrival order never decides a value, a later acquisition is never assumed to contain an earlier one, and the active branch is the provider's own pointer; ordering by a provider's recorded time only where the spec states it (titles). | law 4; design (`decisions/merging.md`; `sources/README.md` rule 6) | oycw, geop, c429, 5q2u, 866e, 9qq7 |
| R26 | Paths are normalized lexically, never by resolving the filesystem, before they enter a key. | design (`model.md`, Identity); example (`claude-code/symlinked-project-dir/`) | m706z, d9t8u; audit P24-A1 |

## 4. One owner per fact, one behavior per surface

| # | Rule | Enforced by | From |
|---|---|---|---|
| R27 | One function per effect and per decision: each table has one writer function in `store`, each derived value one computing function, and ID resolution, default filters, the `status` verdict, setting defaults and derived fields are each computed by one function every path calls. A durable state is read through its owner; every filter runs in the one compiled statement; no raw SQLite connection leaves `store`. | drift (each fixed write statement appears in one `store` function); type (`engine::derive::Product`, one variant per derived value); lint (`disallowed-types`: `rusqlite::Connection` outside `store`); law 76 | sjf6, t0dy, vwia, layg, lyv4, exxly, 7k3n0, r9x56, se08w +30; audit P16-r2:A1, P04-r3:B1 |
| R28 | Every surface takes the one path through the daemon's socket, so CLI `--json`, MCP and HTTP return identical rows, counts and outcomes for the same query; no client opens a database. | law 52, law 83, law 84; design (`principles.md`; `surfaces/README.md`, processes) | vclez, 1kavq, 29naz, 391tg, 9layg, g9j6 +35 |
| R29 | Every declared parameter is honoured or refused on every surface; nothing is silently ignored or coerced. | type (`plan/workspace.md`, rules held by construction); law 53 | b5pok, 3ahdg, yu1l5, 45pkf, 2qolr, 9layg +17 |
| R30 | A mutation over a query acts on every match, not one page; `excise` sends exactly the scopes it showed, one `excision` row each, and reaches every table holding a session, message, text or input key, on every input route. | law 54, law 56 (generated from the DDL), law 36; design (`surfaces/cli.md`, `excise`) | w3s0q, 3jr6e, 5hcbg, 8j9rh, 0zhhc, 14ucm, 9lrqs, layg, re4a, bgnxh +10 |
| R31 | The shipped skill and MCP tool descriptions name only tools and fields that exist. | drift (against `surfaces/mcp-and-hooks.md` and the field tables) | hg97, 6olqi, xi3bs |
| R32 | One timestamp type: an all-digit time is read in the unit its wire field declares, a time without offset is unknown unless its format declares a zone (`spec/model.md`, Time), and only display converts to the local zone. | type (`Timestamp`; wire types `IsoTime`, `EpochSecs`, `EpochMillis`); lint (`disallowed-methods`: system time zone and `Zoned::now` outside `render`); examples (`codex/epoch-units/`, `hooks/naive-timestamp/`) | z3sv, 8ytj, mzp8, xj7ru |
| R33 | Each provider maps raw model strings to canonical names in one table beside its parser; models without a price are listed in `status` and their cost is unknown; credits and USD are different types. | type (`Usd`, `Credits`); design (`analysis.md` §1); law 70 | 6j9c, shnc, 3b607, o1qhm, pd63j, oyiux |
| R34 | Usage is split at parse time into disjoint kinds; one function prices a message and every other cost is its sum; provider totals are checks, never inputs; a value copied onto several records of one reply is stored once. | type (`plan/workspace.md`, rules held by construction); law 14; design (`analysis.md` §1, one cost function); example (`chatgpt/repeated-reasoning-time/`) | 4ts.2, d3xj, ivsc, 5uutw, trmhh, r7p6 +8; audit P23-B4, P36-A6 |

## 5. Writes, durability and storage

| # | Rule | Enforced by | From |
|---|---|---|---|
| R35 | One writer per archive, as a type, plus an OS lock on the archive directory's resolved path. | type (`store::Writer`); design (`storage/README.md`); law 61 | v7tuy, v2hbj, zxb7t, 9zscn, siet, 41ow +20 |
| R36 | No network, model call or non-SQLite IO inside a write transaction: the writer receives finished batches. | lint (`deny.toml`: HTTP clients only in `engine` and the binary); design (`convergence.md`, Parse) | 1ifhp, 8ik6g, awicu, jih4s, qoa75 |
| R37 | Record before effect: write the intent, do the effect, mark done; files go through a temporary name, fsync, rename and directory fsync; a collect cursor moves only after the bytes are stored; the hook command and native host answer success only after the spool file is synced. | design (`storage/blobs.md`; `surfaces/mcp-and-hooks.md`; `surfaces/extension-and-clients.md`); law 7, law 21 | piyqz, dv05i, fhikb, l66gd, 7j50a, 4hlxy, 83v8i, b8of0, aw070 +15 |
| R38 | Readers open read-only and never create files; `immutable=1` is never used on a live database; no process opens a second descriptor on a SQLite file it has open. | type (`plan/workspace.md`, rules held by construction); design (`storage/README.md`); law 62 | p6tgn, ryp2j, lcakh |
| R39 | SQLite `BUSY`, `LOCKED` and `PROTOCOL` are retried with backoff; a deterministic failure is retried only when the input or parser version changes; a panic or timeout in one parse is recorded for that input. | design (`convergence.md`; `behavior.md`, Failures); law 63 | 7c5k6, wyi9p, 6r7wv, slc55, 627m3, 3lfvy |
| R40 | A blob is deleted only by excision or when a newer revision contains its bytes; every blob is reachable from a revision (attachments are inputs). | design (`storage/blobs.md`); law 55 | 9jelj, w06b, 5gjre, tfzw0, c5mb, 1eei6, co2iz +15 |
| R41 | A rebuilt `index.db` replaces the old file in one step, and no bookkeeping table grows with each pass. A rebuild adds no `archive.db` rows. | design (`convergence.md`, rebuilds); law 64, law 81 | wmft, wkc6, bo9n |
| R42 | `index.db` is never migrated; `archive.db` has one ordered migration list; a parser output change raises its format's parser version, and the re-recorded outputs land as their own commit by `xtask golden`. | design (`storage/archive-db.md`); drift (Parser versions) | gucv, lzh8, t3gk, p155, te59g, o7mim +60 |
| R43 | No SQL `CHECK` lists of allowed values; enums keep unknown values; origins are an open set. | design (`storage/index-db.md`; `model.md`); law 65 | sd9s, 052vs, 20c8, h57ic |
| R44 | Merged into R30. | | |
| R45 | No `IN (...)` list built from data; sets of IDs go through `json_each` or a temporary table. | type (`plan/workspace.md`, rules held by construction); design (`plan/workspace.md`); law 35 | h2ejm, 1tyq5.187 |
| R46 | Each provider declares how its files change; a change confined to Claude Code's mutable first record is a header revision plus an append. | design (`storage/blobs.md`; `sources/README.md`); law 21 | ps055, jbcln, ic9h5, rkdej, 0cm7m |

## 6. Bounded work

| # | Rule | Enforced by | From |
|---|---|---|---|
| R47 | Work is proportional to what is new: appending k bytes costs about k in every stage, including `seq`, facts and tree cost; doubling the archive at most doubles each stage's work; fixed statements avoid full scans of `message`, `block` and `text`; no SQL triggers. | law 18, law 57; drift (query-plan check; no triggers in the schema) | 35d, xy95, dlmv, rgbj, z9gh.2, 07g6, meoz, l3tk, nh44, 1gxyu, 2fguj +40; audit P13-A2 |
| R48 | No caps that change results: stream or page, refuse only at a named physical limit, apply per-session bounds per session, and honour or refuse a user-given limit, never clamp it. | design (`principles.md`; `query.md`); law 15; drift (every `*MAX*`, `*LIMIT*`, `*CAP*` constant in product crates carries a `// physical:` comment) | t93b, o21.5, cpisn, 39faa, byghv, lxx9i +8; audit P04-A1, P19-A3..A5, P10-r3 |
| R49 | A large file parses within the memory target by streaming, not by admission estimates. | law 58 | xb4i, ng9m, p60ik, 33y5z, 0v5b |
| R50 | Every cross-boundary call has a deadline and a timeout cancels the work it reports failed; a write timed out after `COMMIT` may have run reports "indeterminate"; accepted connections have deadlines; thread pools size to the CPUs the process may use. | type (`store::WriteOutcome::{Committed, RolledBack, Indeterminate}`); design (`behavior.md`, Concurrency); law 66 | wcw9f, osiia, 8hjsv, 4az8g; audit P11-A6, P11-A7 |
| R51 | The loop takes at most one bounded step from each input with pending work per round, with no credit or debt carried; only committed work is progress; `status` names a stalled loop and a stuck input after 10 minutes; maintenance never delays a live append past 2 seconds. | design (`convergence.md`, fair stepping; `behavior.md`); law 59 | qlae, 11cg9, mhx95, 5xxmc, tboca, hat0; audit P10-A6, P11-A1 |
| R52 | A cursor carries the full order with an ID tie-break; continuation is by key with no transaction held between pages; a request with a cursor takes the query from it, refuses a different one, and may change only the page size. | type (`PageRequest::{First, Next}`); design (`query.md`, Continuation); law 15, law 53 | kkonq, os51i, 89jcg, tw0jr, a9sei, itg5z, vlrqi +4; audit P16-r1:K3 |

## 7. Locations, isolation and untrusted input

| # | Rule | Enforced by | From |
|---|---|---|---|
| R53 | Configuration is one declared `Config`, read only by the binary: the archive is the XDG default or `--archive DIR`; no environment variable is read outside the binary's `main`, and each one read is a declared `Config` field; every runtime path derives from the archive's resolved path except the hook spool, which stays under `$XDG_DATA_HOME/polylogue/hooks` so recording works before the daemon runs and small synced appends use the root SSD; every path built from content stays inside its root. | type (`RuntimeDir` built only from an `ArchiveDir`); lint (`disallowed-methods`: `std::env::var` and `var_os` outside the binary crate); drift (`Config`'s fields against the documented settings and variables, `plan/workspace.md`); law 67 | l1qg, kadx3, o3a1t, dl6af, hq9g3, xzndh, nrlsl, o7hx +10; Python: 12 of 14 `POLYLOGUE_*` variables were never set |
| R54 | Tests never reach the operator's archive or home: the harness points `HOME` at a temporary directory and fails on any open under the real home. | law fixture (`plan/testing.md`); `plan/testing.md`, Rules | 9dtr, 010x, jsxj, o7hx, 1eei6, 251y8 +12 |
| R55 | Downloads go only to the provider's own hosts, never to private or loopback addresses, and never follow a redirect away. | design (`behavior.md`, Untrusted input); law 68 | 56jc6, h5l6i |
| R56 | HTTP and MCP errors name the error kind, never paths or traces; HTML is escaped by default with no inline handlers. | type (`plan/workspace.md`, rules held by construction); design (`surfaces/web.md`; `rendering.md`); surface snapshot (`plan/testing.md` §6) | khkup, l16iz, 2n39, gnie |
| R57 | Committed inferred schemas pass the privacy filter; fixtures are synthetic; `.gitignore` re-admits no agent-output subtree. | CI (privacy filter; `git check-ignore`) | b4cs, 0bgr, 32pwq, ut3r, x2y9 +10; audit P10 |

## 8. Lineage and tool calls

| # | Rule | Enforced by | From |
|---|---|---|---|
| R58 | Links come only from evidence and record it; sessions keep full message lists with copies marked. | design (`decisions/lineage.md`); laws 4, 11, 25, 29–34, 39 | 4ts.2, 4ts.3, 9p0y, kqnw8, x97o9, gy2yu, uoq3x, e0xan +25 |
| R59 | A tool result pairs with exactly one earlier call in the same session with the same ID; a re-emitted identical call is the same call; an empty output is `ok`; a provider's error text is kept. | law 60; example (`claude-code/reemitted-tool-call/`); design (`sources/README.md`; `model.md`) | xnkf, uv2qe, zbtr9, 0f8kb, q759m |

## 9. Tests that can fail and tooling that stays small

| # | Rule | Enforced by | From |
|---|---|---|---|
| R60 | A run that selects zero tests fails; a skip fails unless it names an absent optional tool; a hung test fails with its name. | lint (nextest `--no-tests=fail`, `slow-timeout` with `terminate-after`) | u3g08, h9xx0, hohj7, 27rb, y6tb, lxyt, 7ey6, 1suq6 +10 |
| R61 | Laws whose failure is easy to miss, and every drift check, ship a known-bad case that must make them fail. | drift (Known-bad cases, over the list in `plan/testing/laws.md`) | gt1z, 7qw4, 3a61 + class O of `closed-5.md` |
| R62 | Completeness is checked against ground truth independent of the producer: stored inputs against indexed sessions, texts against search rows. | law 12, law 37, law 55 | in24n, r4jiu, fb61x, gibn1 |
| R63 | Everything produced has a consumer: every written column a reader (the schema is created whole from slice 1, `plan/slice-01.md`, storage subset, and a table is first written in the slice that reads it), every parser or loop count a place in `status` or the export, every function argument an effect. | drift (SQL inventory: every DDL column read somewhere; parse-report structs serialized whole into `status --json`); lint (`unused_variables`; `_`-prefixed parameters outside trait impls fail a name check) | at44, w379, v2mg, kktg, ei0d, e5hiz +40; audit P16-r1, P10-r4 |
| R64 | A test run tests a fixed commit; checks never write tracked files; no caches or generated output are committed. | xtask and CI (`xtask test` records the commit and refuses a tree changed mid-run; CI rejects files over 1 MB outside `examples/`) | nhs3u, 5coki, 72cbt, 0tiaa, eanhw |
| R65 | The query generator covers alternatives, negation, `OR` under a session scope, Unicode case and paths with spaces; SQL and the reference evaluator share one collation. | law 17 | zypl8, 8h6td, i9o1o, pwjbe, g8lih, dwbf7 |
| R66 | Calendar windows use calendar arithmetic in the configured time zone; time ranges are half-open. | type (`plan/workspace.md`, rules held by construction); design (`analysis.md` §1, §3; `query.md`); law 69 | qyvz3 |
| R67 | The vector key is the full recipe (provider, model, dimensions, input type, normalization); a permanently rejected text is recorded once and retried only when the model or text changes. | design (`storage/embeddings.md`; `analysis.md` §8) | w4koi, liwst, n846, egm8 |
| R68 | `status` (while a daemon runs) and `--version` show the binary's version and the schema versions; a daemon refuses to start on a newer archive. | design (`behavior.md`; `storage/archive-db.md`); law 71 | mhx95, 9qnzy |
| R69 | `export --to` replaces a file only by rename after the new one is complete; a URL percent-encodes a whole ID once and decodes it once. | type (`plan/workspace.md`, rules held by construction); design (`surfaces/cli.md`; `surfaces/web.md`); law 74 | n8ssw, 9n3ng |
| R70 | Destructive commands show what they will change and ask; agents cannot excise; agent claims are labelled; no MCP or hook path records the operator as author. | type (the MCP note function takes an `AgentAuthor`); design (`surfaces/cli.md`; `user-data.md`); law 75 | jnj.5, jn40, 37t.15, ldau; audit P24-r4:B2 |

## 10. From the audit of the Python code

| # | Rule | Enforced by | From |
|---|---|---|---|
| R71 | Merged into R76. | | |
| R72 | The access token goes only to a listener the operator owns, and the daemon serves only the operator's uid; an undeterminable owner is refused. | design (`surfaces/web.md`, Access); law 77 | audit P01-A3, P10-A5 |
| R73 | A row states its owner once: two references that must agree are tied by a composite foreign key. | design (`storage/index-db.md`: composite keys); drift (DDL check) | asp4b, dba5k; audit P33-A1 |
| R74 | No compatibility code: an identifier with `legacy`, `compat` or `deprecated` in a product crate must name a provider-format vintage listed in `sources/README.md`. | drift (name check against `sources/README.md`) | audit P16-K1, P28, P19, P15-C1 |
| R75 | The WAL stays bounded: the daemon checkpoints in bounded steps, no reader holds a transaction between pages, and checkpoints name `main`. | design (`storage/blobs.md`, WAL); law 78 | audit P31-A1, P32-K2 |
| R76 | Ambiguity is final: a lookup meant to return one row returns one, several (an error listing them) or none; a fallback runs only after none and records which stage matched; nothing picks among candidates by position or SQL row order; a one-row statement binds a unique key or ends its `ORDER BY` with one; no `LIKE` pattern is built by joining a value into it. | type (`Resolution<T> { One, Many, None }`, `#[must_use]`, no conversion to `Option`); drift (SQL inventory: `query_row` binds a `UNIQUE` index or a unique final sort column; `LIKE` only with a literal or the one escaping function); law 76, law 79; examples (`chatgpt/sandbox-link-two-files/`, `codex/code-mode-unmatched-item/`) | audit P36-B3, P36-A1, P16-r2, P24-r2, P24-r3, P07-r2:A5 |
| R77 | A budget is charged from what is delivered: every size counted against a budget is measured from the rendered output, and the recorded delivery size equals the size charged. | type (`render::Rendered { text, tokens }`, `tokens` computed in its only constructor); law 80 | audit P07-r2:F1 |

## 11. Code shape

From the Python version's architecture: its archive package mixed domain
types with the query engine, which made one import cycle across 39 of its 45
packages; it had about 2,000 record types, one per layer, and 53 modules that
only re-exported others.

| # | Rule | Enforced by | From |
|---|---|---|---|
| R78 | Domain types live only in the `model` crate; other crates use them and add none of their own for the same concept. | lint (`deny.toml` and the crate graph in `plan/workspace.md`: `model` depends on no workspace crate); xtask (`xtask surface` lists public types per crate, and a type whose name matches one in `model` fails) | python-architecture-deep |
| R79 | One type per concept and one error enum per crate: no `XRecord`, `XRow`, `XView` copies of one concept across layers; conversions are `From` impls on the one type. | xtask (`xtask surface` fails on two public types in different crates whose field sets match, and on more than one public `enum *Error` per crate) | python-architecture-deep |
| R80 | No re-export-only modules inside a crate: a module that only re-exports items is not allowed (the crate root may re-export its public API). | xtask (`xtask surface` fails on a non-root module whose items are all `pub use`) | python-architecture-deep |
| R81 | No function longer than 150 lines. | lint (`clippy::too_many_lines` with `too-many-lines-threshold = 150` in `clippy.toml`, denied) | python-architecture-deep |
| R82 | The size of the tests is visible: `xtask surface` reports test lines per crate beside product lines, so a crate whose tests outgrow its code, or have none, is seen in review. | xtask (`xtask surface` output, pasted into each slice's PR description) | python-architecture-deep |

## 12. Schema and data flow

From the schema and data-flow audits of the Python version
(`evidence/deep-schema.md`, `evidence/deep-dataflow.md`, `evidence/deep-writes.md`):
391 persistent tables, 209 of them never read outside the file that writes
them, about 200 parser spill tables and 40 tables that recorded checks; five
functions entering one parser and two ways to write its result, which gave
different digests and titles for the same bytes.

| # | Rule | Enforced by | From |
|---|---|---|---|
| R83 | No table without a reader in another module: every table has a `SELECT` outside the module that writes it (extends R63 from columns to tables). | drift (SQL inventory: tables per writing and reading module) | deep-schema (209 tables) |
| R84 | No table records a check: verification results are test output or log lines, never rows. | drift (DDL shape check in `plan/testing.md`: table names against its process-vocabulary list) | deep-schema (~40 frontier, census, seal and receipt tables) |
| R85 | No on-disk staging in parsers: a parser's working state lives in memory, bounded by the record being parsed, and an append resumes from `parse_state.carry`. | lint (`deny.toml`: `sources` does not depend on `rusqlite`); law 58 | deep-schema (~200 spill tables) |
| R86 | No pending-work tables: work is a query over state, so the schema has no "pending", "debt", "queue" or "obligation" table; staleness is a comparison of a product's computed-from columns with the current versions and `session.changed`. | drift (DDL check on table names and on tables whose rows exist only while work is pending); design (`spec/convergence.md`) | deep-schema (`convergence_debt` and five more) |
| R87 | A new concern adds columns or rows to existing tables first; a new table family needs a stated reason in the schema test's allow list. | drift (DDL check: the table list is compared with an allow list that carries a reason per table) | deep-schema (five excision tables, four Sinex publication tables, nine retained-query tables) |
| R88 | One parse entry per provider: each provider implements exactly one per-record interface, `record(&mut self, &Record) -> RecordOutcome` and `finish(self) -> Parsed`, called by the shared framework, which owns framing, decoding, byte ranges and counting (`spec/sources/README.md`, the framework owns iteration); nothing else constructs a parsed session. | type (the parsed session's constructor is `pub(crate)` in `sources`); drift (functions returning a parsed session, other than `Provider::parse` implementations, fail) | deep-dataflow (five entries, two write paths) |
| R89 | Derived values are computed once, at write, from the parsed result: rendering and surfaces read stored fields (`tool_call.outcome`, `session_facts.display_title`, lineage marks) and never pair tool calls, choose titles or compute outcomes. A value that depends on the clock is computed at read time by the one function that owns it, never stored. | lint (`deny.toml`: `render` and the binary do not depend on `engine::derive`); laws 5, 102 | deep-dataflow (tool outcome in 5 places, title in 8) |
| R90 | Reads never parse or write: nothing proves a read by replaying inputs into scratch; stored rows are the answer and their provenance (`input_pk`, byte range) is how a read cites its source. | type (the read pool's connections are read-only, R38); lint (the query crate does not depend on `sources`) | deep-dataflow (read-time replay route) |
| R91 | Bounded memory is a property of the one parse route (streaming records, commits in steps); there is no separate low-memory route. | design (`spec/convergence.md`, Parse); law 58; R88 | deep-dataflow (prepared streaming branch) |
| R92 | Fixed SQL holds no `ON DELETE CASCADE`, `INSERT OR REPLACE`, `INSERT OR IGNORE` or `REPLACE INTO` (makes R24 and R73 checkable). | drift (DDL and SQL check in `plan/testing.md`) | deep-writes A3 |
