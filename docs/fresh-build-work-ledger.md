# Offline fresh-build work ledger

The ledger reduces an existing JSONL capture. It never reruns a daemon, queries
an archive, publishes derived output, or grants reuse authority. Raw captures
remain with their owner. `export_work_ledger(input_path, output_directory)`
creates a new directory outside the checkout and outside the raw input's
ancestors. It refuses overwrite. `manifest.json` is the last completion marker;
a failed export may leave an incomplete directory with no complete manifest.

The manifest binds input SHA-256, both reducer source hashes, the host monotonic
clock origin, and hashes of `ledger.json` and `trace.json`. Runtime and allocation
provider evidence remain unavailable when the capture has no independently
bound receipt. This reducer does not infer instrumentation from installed tools,
and Python allocation coverage would not imply native or SQLite allocation
coverage.

`reduce_event_ledger` preserves the existing `work.body.started` and terminal
records (`ok`, `empty`, `error`, `unmeasured`), artifact consumer decisions, span
endpoints, compute queue/service spans, and logging loss counters. Queue/admission,
writer, readiness, retry, body and artifact decisions are separate categories.
Writer release hold/wait durations lack endpoints: they remain duration facts,
without invented chronology. Counts describe observed records, not complete
work populations or current readiness. Artifact products and consumer effects
are separate from body executions; equal keys do not establish avoidable work.

Exclusive elapsed and CPU values subtract the union of observed same-thread
work children. They are parent-local residuals, not additive global totals.
Cross-thread child elapsed union records asynchronous overlap without treating
it as CPU. Measured CPU union avoids inclusive parent/child double counting;
PID/TID-only span clocks cannot identify native thread reuse. The separate body
union uses process and worker generations. Known sequence gaps, malformed input,
field loss or sink loss withdraw qualified body CPU and amplification ratios,
while retaining observed endpoints. Missing producers, missing capture boundaries
and wholly absent events remain unknown.

Exports allow only named numeric measurements, categorical tokens and hashed
identity operands. They omit free text, paths, SQL, payloads and arbitrary context.
The Trace Event JSON uses microsecond timestamps and durations, complete `X`
events on separate logical tracks, and parent-cause `s`/`f` flows. Separate tracks
avoid ambiguous non-nested asynchronous overlap; native thread operands remain
in the ledger. This follows [Perfetto's supported Trace Event format](https://perfetto.dev/docs/getting-started/other-formats).
