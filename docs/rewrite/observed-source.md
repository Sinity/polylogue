# One observed Claude location

This native subset retains and interprets one explicitly configured regular
Claude main JSONL file at daemon startup. It does not discover directories or
watch a running provider. Each subsequent startup compares the entire retained
predecessor prefix before accepting new bytes. The initial opened descriptor's
size bounds the read; changed descriptor metadata, a short read, changed prefix,
shrink or replacement refuses that acquisition. This is a checked descriptor
read, not an atomic filesystem snapshot or capture of unexamined revisions.

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

Without `--until-idle`, the daemon stays available for ordinary clients. Adding
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
the tail waits; the source as a whole remains pending.
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
piece joining, companion acquisition, marker/attachment equivalence, corpus
resource guarantees, deployment and full M1 are not qualified by this subset.
