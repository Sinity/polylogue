# One observed Claude location

This native subset retains and interprets one explicitly configured regular
Claude main JSONL file. Normal configured daemon mode checks the main and its
explicit required tool-result bodies once per second. `--until-idle` admits one
startup snapshot and waits for its existing parser work; it does not wait for
future polling timers. Directory discovery remains unsupported.

Each acquisition compares the entire retained predecessor prefix before
accepting new bytes. A worker retains the checked descriptor, parser-reference
scan and CAS proofs while the existing owner grants reusable acquisition
quanta. The owner installs typed proofs; provider IO and full-prefix verification
run outside its writer. The initial descriptor size bounds the read. Changed
metadata, a short read, changed prefix, shrink or replacement refuses that
attempt and preserves previously acknowledged bytes. This checked read is not
an atomic filesystem snapshot or capture of unexamined revisions.

Create a source configuration outside the repository:

```json
{"claude_main":{"path":"/absolute/path/SESSION.jsonl","machine":"collector-id","host":"collector-label"}}
```

Run `polylogue --archive DIR daemon --config FILE --until-idle --json`.
The output identifies the acquisition route, input, chain, revision, retained
length and actual parser/source-row outcome. This idle check concerns the one
configured input, not every archive dependency. A waiting tail or unavailable
interpretation returns degraded exit 5 with its reason. General watcher
convergence and whole-archive idle acceptance remain open. Interrupting the
pending idle wait returns an interruption refusal (exit 3), never success; actual deadline expiry remains `Timeout` (exit 7).

Without `--until-idle`, the daemon stays available for ordinary clients and continues the configured acquisition. Adding
`--json` prints and flushes a `scope: retention_only` admission before waiting;
that admission does not certify normalized rows. Plain `daemon` also recovers
already retained observed inputs without reopening their provider paths.

The location keeps its UUID input and chain across verified appends. The first
retained pathname supplies Claude context. Every acquired piece, including blank
separators and unfinished records, retains its original byte position. A
separator-only append advances byte progress without inventing a Session or
record. An unfinished nonblank record waits at its complete-record checkpoint;
restart rereads the retained tail and a later piece may finish that same record.
Complete prefix Messages can finish their existing canonical publication while
the tail waits; the source as a whole remains pending. Bounded exact Message
and Block reads require that prefix's selected body and current source evidence
in the same pinned view. They preserve pending Session context. A whole-Session
transcript still refuses pending work, as do rows without that current proof.
Changed acquisition invalidates the old row-settlement receipt. Observed
receipts use `scope: observed_source_rows`, identify the admitted revision and
length, and omit the immutable-only `original_hash` field. Their interpretation
binding is not a full-content hash.

`polylogue --archive DIR input cat ID` reads retained pieces without provider
fallback and verifies backing piece hashes before output. Default output passes
through the shared daemon-side credential masker. Hashes, citations and source
byte offsets continue to identify the original retained bytes, independently of
changed display length. This subset refuses invalid UTF-8, binary/control bytes
and JSON escape spellings before payload output; encoded strings need a decoded
presentation that is not implemented here. There is no `--reveal` support.
Credential-free supported inputs can still reproduce their exact retained bytes
through this presentation. Unknown inputs return `NotFound`; unsupported safe
presentation returns `NotYetAvailable`, excision requires a refusal, and expiry
returns `Timeout`. The client preserves these typed failures. Verification and
presentation check peer disappearance at bounded read boundaries; disconnecting
a read does not cancel the retained source owner.

The independent `observed_source_demo` operation accepts `--binary ABSOLUTE_PATH
--output NEW_DIRECTORY`. It exercises actual daemon and CLI processes, captures
all attempted client commands and their streams, and keeps `report.json` even
when a check fails. It distinguishes an owned process kill after retention while
row work is pending from exact internal install-boundary interruption. Store
unit controls separately cover registration failure after archive commit,
missing/overlapping extents, replacement, prefix mutation and short capture.
The shipped checker validates complete Session, Message and block identities,
logical text, source spans and decoded origin selectors, and rejects four planted
identity/binding/span/duplicate mutants. It also witnesses pending target work
before SIGINT and requires the interrupted idle command to fail.

Source scanning, sampled verification, renames, rewrites, revision coalescing,
piece joining, general companion acquisition, marker/attachment equivalence, corpus
resource guarantees, deployment and full M1 are not qualified by this subset.

## Required tool-result bodies

Configured main acquisition also reads tool-result paths explicitly named by
complete retained records. The supported location is
`<main directory>/<main stem>/tool-results/*.txt`. The source adapter owns
`persistedOutputPath` and the existing saved-output text spellings. Acquisition
does not enumerate the directory. Each attempt scans the full checked main in 64 KiB windows under cooperative grants; this does not establish append-local scan cost. Required bodies are checked even when the main metadata and bytes are unchanged. It admits at most 32 referenced paths, a
1 MiB reference record and 1 MiB per regular body. The configured main parent resolves normally. Held directory handles then open
the appended session directory, `tool-results` directory and body leaf with
`NOFOLLOW`; symlinks in any of those components and nonregular descriptors refuse. UTF-8 is required for normalized text.
Full structured output keeps the public reader's existing 65536-byte normalized
JSON admission; larger retained bodies can still refuse that presentation.

Each acquired body is an immutable `claude-code-tool-result` input. Its context
contains the owning main input ID, exact path and `required` or
`inline-duplicate`. The last local acquisition observation for each path selects
its snapshot. This reuses Archive input, revision, extent, blob and observation
ownership. It adds no durable table or mutable parser authority. A checked exact
copy of complete inline text stays redundant. Its bytes do not join the Message
version, and losing that retained replica cannot invalidate the inline result.

The sorted retained manifest participates in the main parser basis. A changed
required snapshot causes normal re-evaluation from retained bytes. The parser
gets body cells on that same basis, and the result Message version includes its
raw record and exact required-body digest/length under the owning compound
evidence recipe. The assistant call's version stays independent. Missing bodies
leave required work unresolved. Denied reads and retained corruption are errors;
neither becomes an empty result. Missing provider copies preserve already
acquired snapshots, and ordinary daemon restart reads only retained bytes.

Session JSON and bounded CLI/MCP rows expose the selected Message's
`version_hash`. Full tool-result text has a `claude-code-tool-result-body` field
origin with the companion input, chain, snapshot revision, exact path, byte
range and raw blob hash. The origin verifier reads and hashes that retained
snapshot instead of treating the inline preview as the full output. Public
Session qualification also verifies required retained snapshots. A normalized
copy cannot make a lost or changed required witness readable. Output and
`input cat` retain their existing credential masking.

`claude_body_demo` takes `--binary`, `--binary-candidate`, `--output`, `--sdk-python`, `--sdk-client`
and `--b3sum`. The SDK client is the official-SDK adapter from the pinned joint
demo. The driver constructs synthetic inputs and independently assembles the
expected compound Message-version bytes. It checks CLI and SDK output, native
identity, exact retained origins, separator-only continuation, later content,
body-only revision, restart, provider removal, unavailable/corrupt retained
bytes, missing/denied provider reads, symlinks in each appended path component,
redundant inline copies and private file
modes. Every attempted CLI command and failed-run report remains inspectable.
This is one configured-main body shape, not general attachments, watchers,
directory discovery, subagent acquisition or full M1 acceptance.

An exact Message or block read verifies the required body witnesses pinned by
that selected Message's field origins. A different Message's lost required body
does not prevent an independently backed Text member from being read: its
Session context remains degraded and unavailable. A full Session read and the
owning tool-result read still refuse the missing witness.

## Configured continuation controls

The optional `claude_main.acquisition_quantum_bytes` sets a reusable charged-work
quantum from 65536 through 8388608 bytes (default 8388608). It limits a granted
turn, including retained verification, rather than the source's lifetime size.
The same parked acquisition slot and ordinary-import arbitration serve physical
history and Claude. Ordinary read workers continue using acknowledged retained
state before a larger capture is admitted. A yielded proof is never a selected
input or settled candidate. Cancellation discards uninstalled acquisition
proofs; a later process verifies the retained predecessor afresh.

Status exposes `claude_continuation`: active acquisition, attempts, acknowledged
length, unchanged passes and bounded diagnostics. Missing and denied main paths
remain distinct from an accessible empty main. Missing required provider bodies
preserve any retained snapshot; a never-acquired body leaves its Message pending.
Unchanged polls add neither input revisions nor body observations and do not
restart unchanged parser work. A changed required manifest restarts the existing
interpretation even when the main revision is unchanged. Pending incomplete
records remain owed until a later append completes them.

`configured_claude_continuation_demo` accepts the same explicit binary, candidate,
SDK client and digest-tool arguments as `claude_body_demo`. It starts one
synthetic configured daemon and discovers a later operator correction through
ordinary CLI and official SDK MCP search/read, then checks an incomplete suffix,
late body arrival, body-only revision, exact versions/origins, stable polls,
missing/denied main paths and provider-free interrupted-tail restart. Controlled
engine tests park a small acquisition quantum, complete a real ordinary capture
and retained read before the larger append is selected, and cancel/retry a
parked capture. This qualifies one configured source shape. General watcher
semantics, rewrites, directory discovery, installed routing, corpus resource
bounds and full M1 remain open.

## Explicit Claude projects directory

A daemon can select one provider directory instead of one main:

```json
{"claude_projects":{"root":"/absolute/synthetic/projects","machine":"collector-id","host":"collector-label","acquisition_quantum_bytes":65536}}
```

Choose exactly one of `claude_projects`, `claude_main` and `codex_home`. No home
source is selected implicitly. Projects collection currently runs continuously;
`--until-idle` refuses before archive creation for this configuration.

The anchored predecessor layout admits `-project/SESSION.jsonl` or `.ndjson`,
`-project/OWNER/subagents/agent-AGENT.jsonl`, and the corresponding
`subagents/workflows/RUN/agent-AGENT.jsonl` path. Main stems supply main context;
separate-agent records supply membership, while the containing owner supplies
lineage. The durable observed-context relation preserves that declaration across
provider removal and parser rebuild. A main declaration cannot silently replace
an established agent declaration.

Traversal uses at most 128 entries per turn and a fixed-depth directory stack.
Runnable scan/capture backlog continues without waiting for another filesystem
event; completed cycles pause before rescanning. Each transcript uses the same
parked acquisition slot, charged-byte grants and required-body resolver as the
single-main route. Reads continue using acknowledged state. Full-prefix
verification remains source work on every capture attempt; no metadata-only
unchanged-file shortcut or corpus throughput guarantee is claimed.

Retained interpretation finishes its current input before choosing another.
A recent acquisition or explicit retry can run between background catalog inputs;
these choices alternate so ongoing acquisition cannot suppress the background
scan. One recent-input hint coalesces intermediate changes without replacing the
durable catalog. Restart recovers work through that catalog. This does not
preempt a large current input or promise a fixed interpretation latency.

Held directory descriptors and descriptor-relative, beneath-root, no-symlink
opens fence traversal and capture. An archive cannot be used as the configured
provider root. Root replacement starts a fresh walk; root absence or denied
access is unavailable, distinct from an accessible empty scope. Neither an
interrupted scan nor a partial JSONL tail certifies complete conversation
coverage.

Status reports scan-cycle progress, deferred/excluded entries, attempted source
acquisitions and a bounded sample of retained input IDs. The sample never drives
acquisition. Selected-source byte totals sum acknowledged lengths, not unique
bytes or actual IO. Memory documents, session indexes, workflow journals and
agent metadata are not interpreted by this route. Only explicitly referenced
required bodies are acquired; unclaimed or hook files are not conversations.

The projects status names the currently acquiring path separately from the last
acknowledged input. This marker does not publish its prepared prefix. Missing
initial subagent ownership stays unresolved: the containing folder never
substitutes for `sessionId`. A later actual owner does not retroactively assign
those earlier records, and a conflicting later owner does not change an
established Session. Parser version 10 records this interpretation change.

Requested Session membership is qualified independently of population-wide
possible relationships. An unrelated main's incomplete suffix leaves global
relationship coverage pending without declaring the requested, independently
proved Message membership pending. Unknown relevant dependency shapes and
current selected-source guards continue to constrain reads.

Qualification streams the pinned source catalog. `target_basis.inputs` is a
bounded display sample plus required exact-source witnesses; `inputs_total`,
`inputs_enumerated` and `inputs_digest` identify whether it enumerates every
represented basis and commit to all of them. The digest is ordinary BLAKE3 over
`polylogue.session.input-bases.v1\0` followed, in catalog order, by each basis's
JSON byte length as big-endian u64 and its serialized bytes. It is an operation
witness, not independent storage authority or complete-population evidence.
The observed-member exception counts every proved membership-capable source,
never the displayed sample length. Incoming relationship witnesses remain in
the full context count and digest without becoming a source of the requested
Session’s Message membership. Unresolved relevant inputs still constrain reads. Relevant witness and
Message-membership metadata retain their separate admission bounds.

`configured_claude_projects_demo` accepts an explicit packaged binary/candidate,
SDK interpreter/client, digest tool, output directory, and pinned format-3 binary
and source. Its synthetic journey includes multiple projects, separate agents,
shared-parent required bodies, late files/appends, thousands of small sources,
read service, provider-free restart and the preserving old-preparation route.
The driver checks default SDK budgets after the larger population exists.
`--population-size 0` runs only focused continuation/restart/migration controls
and reports population acceptance as unrun; the default remains 5,000. Its
receipts report actual dimensions and elapsed work; they are not a production
throughput, RSS, whole-corpus or activation qualification.

Companion lookup uses an archive-owned partial expression index on the exact
main context value. Unrelated inputs do not contribute to that lookup's row
scan; NULL or nonmatching context values retain the query's existing semantics.
This does not change capture, snapshot selection or publication authority.
The earlier private format-4 candidate at `14cc1c702` predates this schema
index. Its retained acceptance fixtures belong to that exact binary and source;
current exact-schema validation may refuse them. Formats 1 through 3 retain
the preserving upgrade route described above. No installed format-4 archive
compatibility is claimed for those unpublished candidates.

Claude parser version 11 admits supplied boolean `toolUseResult.isImage` and
`toolUseResult.noOutputExpected` as named execution metadata. Each present value
has its exact retained source path and ordinary JSON leaf selection. Absence
adds no default; neither flag infers image content, outcome, or missing output.
Wrong types and unknown metadata still leave mapping obligations pending.

The parser-version basis changes normal retained-source interpretation, so a
retained contribution parsed by an older version is re-evaluated before it can
be current. This also re-applies the existing later-fragment parent resolution
repair. Selection codec 1 and adapter recipe 2 keep their existing meanings;
no archive reset, provider edit, or new recovery operation is required by this
mapping change. This does not establish complete source restoration.

Claude parser version 12 additionally selects supplied `message.stop_details`
as the named opaque response metadata key `stop_details`. Supplied null is a
value; absence creates no default. The last actual supplier wins in the existing
response update order. A same-UUID overlay that omits the field preserves its
previous donor and cannot become a later supplier. Selected values retain their
exact raw selectors; response correction evidence distinguishes a later update
to an earlier contribution from contribution-order reduction. This metadata
never infers finish state or stop reason. Unknown nested extras and collisions
with a literal top-level selected key still refuse canonical certification.
The normal parser-version basis re-evaluates retained interpretation; existing
selection codec and recipe meanings are unchanged. No private replay or complete
source restoration is established by this mapping's synthetic checks.

Claude parser version 13 admits immediate provider block members as opaque,
source-backed block extras. Each leaf keeps its exact raw path and original
value. A block member that aliases a selected structured ToolResult channel
still refuses, as do ambiguous nested paths and existing unmapped effects.
No status or image meaning is inferred from an extra.

Normalized JSON and export use `data` for the extra bag on ToolCall and
ToolResult blocks. Other block kinds use `attrs`. These bags hold named opaque
provider members; a Thinking block's typed ciphertext `data` remains separate
from `attrs`, including an extra whose own key is `data`. Bounded model metadata
uses the same container names, with canonical-JSON envelopes after privacy
assessment, and field origins use the same qualified paths. Existing leaf,
metadata, deadline and cancellation admissions apply. Unknown active-path
membership still refuses default bounded reads; when current known rows are
readable with no pending inputs, the refusal points to explicit `--branches`
(MCP `branches:true`) for known retained catalog members.
