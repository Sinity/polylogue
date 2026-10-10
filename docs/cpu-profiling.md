# Owned CPU diagnostics

The development shell includes Linux `perf`; the product package has no profiling dependency. The provider records `cpu-clock:u` samples for an owned command, its threads and inherited descendants. It offers no attach operation or privilege changes. Kernel CPU and off-CPU waits are outside this measurement.

Use the ordinary runner's cheap `/proc` CPU accounting on every run. Use the perf provider separately when diagnosing native CPU costs. Recording and decoding overhead are diagnostic costs and cannot qualify a product performance budget. Compare with an unprofiled run using the same interpreter, inputs and effective compute policy.

## Launch and reduce

`PerfCpuProfiler.prepare(command, new_output_dir)` returns `record_command` and `environment_delta`. Launch that command under the existing managed job and child custody rules; merge the environment delta into the child environment. After the owned recording has settled, call `finish(record_exit_code, runtime_after_imports=child_runtime)`. The output directory must be new. Existing evidence and completed reduction stages are refused as overwrite targets.

The standalone wrapper has the same operation:

```sh
agentctl job start polylogue scratch --workspace /realm/worktree/example -- \
  -m devtools.fresh_build_bench.profiler_provider \
  --output-dir /realm/tmp/work/cpu-diagnostic/new-recording \
  --runtime-receipt /realm/tmp/work/cpu-diagnostic/child-runtime.json \
  -- /realm/worktree/example/.venv/bin/python /realm/tmp/work/cpu-diagnostic/child.py
```

Use an owned synthetic or explicitly selected diagnostic child. The managed operation retains its cgroup and descendant reaper. The wrapper forwards cancellation to its recording process group and waits for the recorder. It does not replace the ordinary daemon driver's readiness, reservation, native custody or output qualification checks.

The runtime receipt path must also be new. The child writes `devtools.fresh_build_bench.sampler.runtime_identity()` after importing its product and diagnostic dependencies, either directly or in a `runtime_after_imports` field. The existing sampler also writes that field when it dumps its accounting. A missing effective GIL observation and a GIL-enabled runtime remain explicit limitations. A configured environment variable alone does not establish effective runtime compatibility. A release launcher may clear `PYTHONPATH`; explicitly insert a candidate source path in an owned diagnostic child when needed, and bind that source separately.

## Evidence and coverage

The wrapper writes `profile.json`, raw recordings, decoded native leaf rows and stage stdout/stderr. The receipt binds the tool executable and wrapped implementation, build options, launcher user/capabilities, `perf_event_paranoid`, exact command, runtime observation and artifact digests. Raw DWARF captures contain process stack memory; keep the directory in private diagnostic storage. Python's JIT metadata uses its own `/tmp` files, so retained decoded reports are the evidence for later reading; the receipt does not promise portable reconstruction of JIT callchains after those files disappear.

Sampling uses Python's perf JIT support, monotonic timestamps, a user-space software CPU event and an 8192-byte DWARF stack snapshot. This route can identify SQLite, OpenSSL and interpreter native CPU even when Python callchains cannot be unwound. Interpreter support and a successful JIT injection do not prove Python function coverage. The reducer records total samples, losses, unknown leaf symbols, empty callchains and observed Python symbol samples independently. Partial symbol observations remain partial coverage. Inherited descendants are not independently qualified by the parent trial.

Render a receipt with:

```sh
python -m devtools.fresh_build_bench.profile_report \
  /realm/tmp/work/cpu-diagnostic/new-recording/profile.json
```

The native table is labelled as leaf samples. The renderer refuses collapsed stacks for this document, since a leaf table cannot reconstruct callchains. Missing tools, failed recording/decoding, no samples and wholly unresolved symbols produce unavailable outcomes; useful native rows with coverage gaps produce a degraded outcome. The original in-process sampler continues to refuse cross-thread Python frame traversal on free-threaded builds while retaining CPU accounting.

For Python's documented JIT unwind requirements, see [Python 3.14 perf profiling](https://docs.python.org/3.14/howto/perf_profiling.html). Qualification must use the actual interpreter and tool build: ABI support advertised by [py-spy](https://github.com/benfred/py-spy/releases) or [Austin](https://github.com/P403n1x87/austin) is insufficient without usable samples from the owned child.
