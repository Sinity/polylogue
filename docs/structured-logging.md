# Structured logging

Polylogue emits events with stable dotted names and allowlisted fields. Legacy
prose is bridged into the same configured sink with its message quarantined in
`error_detail`; the rendered line is a view of the event record.

The design target is one specific reader: someone scrolling a completed
fresh-start rebuild log — a long unattended convergence over the whole archive
— asking *"did this actually ingest everything, and if not, where did it stop
and why?"*. Every decision below follows from that question.

## Why the previous arrangement could not answer it

`polylogue/logging.py` already wrapped structlog, and ~140 modules already
imported `get_logger`. But `configure_logging()` is only called on the
`--verbose` / `--json-logs` paths. Everywhere else `get_logger` returns
`_StdlibBoundLogger`, whose `bind()` is a no-op and whose `_stdlib_log_kwargs`
discards every keyword except `exc_info`/`stack_info`/`stacklevel`/`extra`.

So the structured fields those call sites passed were silently thrown away in
the common case, and the sink was a bare `StreamHandler(sys.stderr)` with no
formatter — no timestamp, no level, no logger name. The call sites had also
drifted to interpolated prose:

```python
logger.warning("converger: execute failed for %s stage=%s: %s", path, stage_name, exc)
```

That line cannot be filtered, counted, or correlated. It is also indistinguishable,
to a reader, from a line that meant nothing.

## The shape

```python
from polylogue.logging import emit, span, bind

with bind(run_id=run_id, component="daemon"):  # correlation scope
    with span("daemon.converge.file", path=path) as pass_span:
        emit("daemon.stage.executed", stage=name, outcome="ok", duration_ms=ms)
        pass_span.ok(files=1)
```

Three primitives, each one line at the call site:

| Primitive | Purpose |
| --- | --- |
| `emit(event, level=INFO, **fields)` | one point-in-time event |
| `span(name, **fields)` | a unit of work: `.start` plus exactly one terminal event with `duration_ms` |
| `bind(**fields)` | attach correlation fields to everything emitted inside the block |

### Correlation

`bind` and `span` write to a `contextvars.ContextVar`. That choice is what makes
one unit of work followable across the daemon's concurrency:

- **`await` boundaries** — contextvars are per-task; propagation is automatic.
- **`asyncio.to_thread`** — copies the context; automatic. The daemon uses this
  heavily.
- **The writer lease** — `write_coordinator._run_in_daemon_thread` already calls
  `contextvars.copy_context()` before the thread hop, so correlation crosses the
  lease for free.
- **New `threading.Thread`** — inheritance depends on the interpreter's
  `thread_inherit_context` setting. Explicitly wrap a new target when its
  correlation must survive both GIL-enabled and free-threaded builds.
- **Pooled threads** — a reused worker does not acquire the submitter's current
  context automatically. `propagate(fn)` copies the full context, including
  any writer authority, so it belongs only at an intentional full-context
  handoff.

`span` also issues `trace_id` / `span_id` / `parent_span_id`, so nested work
forms a tree within one `run_id`.

### Honesty

The campaign's recurring defect is a refusal or an unmeasured state rendered as
success. Three mechanisms push back:

- A span's terminal event is emitted from `__exit__`, which runs **before** any
  enclosing `except`. An exception swallowed by a broad handler upstream is
  still recorded.
- A span that exits without declaring an outcome emits `<name>.unmeasured` at
  WARNING. Silence is never promoted to success.
- `outcome` is a closed vocabulary — `ok`, `empty`, `degraded`, `error`,
  `refused`, `unmeasured`, `skipped` — deliberately mirroring the surface
  outcome vocabulary in `surfaces/outcome.py`, so "a probe timed out" and "a
  probe returned nothing" cannot collapse into the same reading.

Reserved keys (`ts`, `level`, `event`) are assigned after caller fields, so a
call site cannot rename its own event or restate its own level. Terminal span
facts also win over caller metadata without masking an operation exception.

### Cost

`emit` compares one integer before doing anything else, so a suppressed event
costs the caller's kwargs dict and nothing more. The configured sink uses a
bounded queue and a daemon worker. A blocked device therefore costs the caller
an enqueue, and `diagnostic_snapshot()` reports queue depth, drops, delivery
failures, and delivered records. Warning and error records can displace queued
routine records. Flush happens on idle and shutdown; shutdown waits briefly,
then reports queued drops and any in-flight record it could not prove drained.

Guard genuinely expensive field computation with `is_enabled(level)`.

## The PII boundary

Tracked content, commits and CI logs are public; session transcripts must never
reach them. The boundary is therefore an **allowlist in code**, not reviewer
discipline: `polylogue/logging_fields.py` registers every emittable field name,
and `emit`/`bind`/`Span.set` drop anything unregistered — recording the drop as
its own `log.field_rejected` event, so a leak attempt is visible rather than
silent.

Consequences:

- `emit("x", text=transcript)` cannot leak. `text` is unregistered, so the value
  never reaches a record. `FORBIDDEN_FIELDS` additionally names the obvious
  content words so the rejection says `reason=content_field` rather than
  `unregistered_field`.
- Non-scalar values are reduced to `<TypeName>` rather than `repr`'d, so an
  object cannot smuggle content through its representation.
- `bind` validates too — content bound once would otherwise ride along on every
  downstream event, the worst possible leak shape.
- Exactly **one** free-text field exists, `error_detail`. It is truncated to 300
  characters and it is the only field a redacting renderer must strip.
  `POLYLOGUE_LOG_REDACT=1` strips it from **both** rendered forms -- the JSON
  storage form and the console view -- and `render_json(record, redact=True)` /
  `render_console(record, redact=True)` do the same directly. A test asserts the
  quarantine set stays at one entry, so this property cannot drift.
- Counts require non-negative integers, durations require finite non-negative
  numbers, and outcomes use the declared vocabulary. A `stage_timings_ms` map
  admits at most 12 short phase names with finite millisecond values. Scalar
  strings and whole records are bounded before dispatch. `reason` must be a
  short token; diagnostic prose belongs in quarantined `error_detail`.

Filesystem paths *are* allowed — the rebuild reader needs "where did it stop" —
but are marked `LOCAL_ONLY_FIELDS` for any future export path.

## Configuration

`configure_events()` installs the default sink and threshold.

| Variable | Values | Default |
| --- | --- | --- |
| `POLYLOGUE_LOG_FORMAT` | `json`, `console` | `console` |
| `POLYLOGUE_LOG_LEVEL` | `trace`…`error` | `info` |
| `POLYLOGUE_LOG_FILE` | path; appends | stderr |
| `POLYLOGUE_LOG_REDACT` | `1`/`true`/`yes`; strips quarantined fields from both the JSON and console forms | off |

`json` is the storage form (one object per line, sorted keys — diffable and
`jq`-able). `console` is the operator view. For an unattended rebuild, run with
`POLYLOGUE_LOG_FORMAT=json POLYLOGUE_LOG_FILE=<path>` and render afterwards.

Additional sinks: `add_sink` / `remove_sink`, and `capture()` for tests. Those
explicit sinks run on the caller thread; the configured stream is queued.
The configured stream receives `emit`, pre-configuration stdlib loggers,
post-configuration structlog loggers, and third-party stdlib records. Audit
receipts remain in their durable tier, separate from disposable diagnostics.
Delivery is best effort: a failed device can lose events, and the loss counters
must be read before treating a file log as a complete diagnostic record.

The source call-site audit kept `value`, `context`, and `stage_payload` rejected:
they can carry raw provider or request data. `source_path` remains rejected in
favor of the bounded registered `path` field. Acquisition gained typed `size`,
`mtime`, `source_name`, `evidence`, and `stage_timings_ms` fields because those
are the measured decision and timing units used by its real producer. Other
unregistered legacy call-site fields still produce `log.field_rejected`; a
warning about such a field is diagnostic loss, not a failed input file.

### Relationship to `devtools` receipts

They answer different questions and must not be merged. A `devtools` receipt
(`.cache/verify/runs/...`) is *evidence about a verification run* — which
selection ran, what its outcome was, keyed on declared inputs. The event log is
*the production runtime's own record of what it did to the archive*. A receipt
is written once per run by the harness; events are written continuously by the
daemon. The log is not a receipt store, and `devtools` should not grow a second
copy of runtime state — where the two need to meet, a receipt can cite a
`run_id` and the log answers the rest.

## What the daemon emits

One convergence pass over two files, the second stage failing, rendered in
console view (real output, `component`/`run_id` bound once at daemon start):

```text
INFO  daemon.run.start component=daemon pid=1234 run_id=rebuild01
DEBUG daemon.converge.file.start path=claude-code/a.jsonl run_id=rebuild01 span_id=94e3…
INFO  daemon.stage.executed duration_ms=0.0 outcome=ok path=claude-code/a.jsonl stage=acquire span_id=94e3…
ERROR daemon.stage.execute_failed error_detail=unparseable payload: zero sessions
      error_type=RuntimeError outcome=error path=claude-code/a.jsonl stage=parse span_id=94e3…
INFO  daemon.stage.executed duration_ms=0.0 outcome=ok path=claude-code/a.jsonl stage=index span_id=94e3…
WARN  daemon.converge.file.degraded duration_ms=0.115 errors=1 files=1 outcome=degraded
      path=claude-code/a.jsonl reason=stage_errors span_id=94e3…
INFO  daemon.run.stop files=2 run_id=rebuild01
```

The reader can now answer the rebuild question directly:

```sh
jq -r 'select(.outcome=="error" or .outcome=="unmeasured") | "\(.stage) \(.path) \(.reason // .error_type)"' rebuild.jsonl
jq -r 'select(.event=="daemon.converge.file.ok") | .path' rebuild.jsonl | wc -l
```

Note what the excerpt makes visible that prose hid: `index` reported `ok` on a
file whose `parse` had already failed. That is a real observation about stage
ordering, and it is the kind of thing the old log could not have surfaced.

## Converting the rest

The remaining work is grind, not judgement.

**Find the legacy form:**

```sh
grep -rn "logging.getLogger" --include="*.py" polylogue        # 17 modules
grep -rn "logger\.\(debug\|info\|warning\|error\|exception\)(" --include="*.py" polylogue
```

**Convert one site:**

1. Replace the module's `logger = logging.getLogger(__name__)` /
   `get_logger(__name__)` with `from polylogue.logging import emit, span`.
2. Turn the prose into a dotted event name: subject first, then what happened —
   `daemon.stage.execute_failed`, not `"converger: execute failed for %s"`.
3. Move every interpolated value into a keyword. If the field is not registered,
   add it to `logging_fields.py` — that addition is the review point.
4. `except Exception:` → `except Exception as exc:` and pass
   `error_type=type(exc).__name__, error_detail=str(exc)`.
5. Where the code has a natural unit of work with a duration or an outcome, use
   `span` instead of paired `emit` calls, and declare the outcome explicitly.
6. Delete the module-level `logger`.

`convergence.py` is the worked example: 14 legacy sites, zero remaining.
The rest of `daemon/` followed it — every module except `cli.py` and `http.py`
now emits events, which is why the baseline below has shrunk.

**Nothing goes dark during the migration.** `configure_events()` installs a
`logging.Handler` that bridges surviving stdlib records into the event stream as
`stdlib.record`, with the prose in `error_detail`. An event carrying prose is
precisely an event that has not been converted yet — which is also how you
measure remaining work on a live run.

### The gate (landed, as a ratchet)

The legacy form is now a `devtools gate patterns` rule rather than a new gate:
`devtools/patterns/legacy-stdlib-logger.yml` matches `logging.getLogger($$$)`,
registered `enforcing` against
`devtools/patterns/baselines/legacy-stdlib-logger.txt`.

That registry already provides exactly the semantics this needs — a
grandfathered baseline that may shrink and never grow, with stale entries
reported as prunable debt — so adding a parallel gate would have duplicated it.

**It could not land at zero baseline**, so it did not pretend to. A new
occurrence anywhere in the tree fails the gate today (verified by introducing
one and observing `status: failed`), while converting a module shrinks the
baseline. The end state is a baseline containing only `polylogue/logging.py`,
which legitimately owns the sanctioned `logging.getLogger` calls — the stdlib
bridge's own backing logger and handler wiring.

The baseline now stands at its end state: **4 matches, all in
`polylogue/logging.py`** — the stdlib bridge's own backing logger and handler
wiring, which legitimately use `logging.getLogger`. Every other module in the
tree is converted. The last three to go were `event_bus.py`, `intake.py` and
`supervisor.py`, the only daemon modules that acquired a stdlib logger
directly.

This is the end state *for this rule*, which is narrower than "no legacy
logging remains". See the scope note below.

Two properties of the ratchet are worth knowing before you read a verdict:

- The rule matches the AST, so a `logging.getLogger` mentioned in a docstring
  or comment is not a match. A `grep -c` will therefore read higher than the
  gate's match count; reconcile against the gate, not against grep.
- A baseline entry is `path:sha1(stripped source line)`. Editing a *baselined*
  line — even cosmetically — retires the old anchor and presents a new one,
  which the ratchet reads as a new violation. Convert such a line rather than
  reformatting it.

Note the rule deliberately matches only the *acquisition* of a legacy logger,
not each `logger.warning(...)` call. Acquisition is the reviewable choke point;
once a module has no legacy logger, its call sites cannot survive.

### Scope note: what the ratchet does not catch

`legacy-stdlib-logger` matches `logging.getLogger(...)` — the *acquisition* of
a stdlib logger. That is not the only legacy form, and today it is no longer
the dominant one.

Roughly 140 modules acquire their logger through
`polylogue.logging.get_logger`, which the rule does not match. Those call sites
still emit prose through `_StdlibBoundLogger`, whose `bind()` is a no-op and
which discards every structured keyword it is given. Converting the two largest
daemon modules (59 and 31 prose sites) eliminated **zero** baseline entries for
exactly this reason: neither had ever used `logging.getLogger`.

So a green `legacy-stdlib-logger` means "no module acquires a stdlib logger
directly", not "no module logs prose". Do not read the former as the latter.

That gap is now closed by a **second** rule, `legacy-prose-logging`, which
matches prose calls on a `logger`/`_logger` name rather than the acquisition.
It is enforcing against a baseline recorded at the true remaining count —
**374 matches across 282 anchors** when it landed — and verified to block by
introducing a new `logger.warning` and observing exit 1, then restoring.

Widening the older rule was not available: an enforcing baseline may only
shrink, and widening would have grown it by every wrapper module at once. A
new ratchet, by contrast, may start wherever reality is, and shrink from
there.

Read the two together. `legacy-stdlib-logger` at 4 means no module acquires a
stdlib logger directly. `legacy-prose-logging` at 374 is the honest size of
what remains: prose that still discards its structured fields. Neither number
alone answers "is the tree converted".
