# Component diagnostic captures

`devtools.fresh_build_bench.diagnostics.capture_diagnostic` surrounds the component owner's existing production callable. It starts no process, selects no input, and performs no parser, validation, convergence, or publication itself. The owner keeps preparation, consumption, and resource retirement outside the capture when those are outside the intended stage.

```python
from devtools.fresh_build_bench.diagnostics import BoundModule, capture_diagnostic

with capture_diagnostic(
    "cprofile",  # or "tracemalloc", in a separate diagnostic invocation
    output_directory=new_private_directory,
    bindings=component_owner_bindings,
    module_authority=loaded_candidate_modules,
) as diagnostic:
    result = exact_production_callable()

# result is the callable's original object; the helper never serializes it.
receipt_reference = {
    "path": str(diagnostic.manifest_path),
    "sha256": diagnostic.manifest_sha256,
}
```

The supplied bindings contain `candidate`, `product_tree`, `runtime`, `input_sha256`, `recipe_sha256`, `schema_inventory_digest`, `component`, and `runner_sha256`, with optional owner launch/arm references. Input, recipe, schemas, candidate, and runner are the component owner's authorities; the helper does not invent or weaken them. Use runtime fields `path` (resolved `sys.executable`), `sha256` (that binary), `version` (exact `sys.version`), and `gil_enabled` (effective state after product imports). The helper checks actual runtime before tool imports, after imports, after start, and after stop. A mismatch refuses diagnostic qualification. The module map is absolute source filename to `BoundModule(actual_module_name, source_sha256)` for modules actually loaded at entry. Those hashes are checked before and after capture. Later product imports belong in the owner's final receipt; unknown capture locations remain explicitly unbound.

Output must be a new directory outside every checkout, with an existing parent. The helper creates it with mode 0700 and files with mode 0600. It never overwrites a prior capture. `manifest.json` binds full owner/runtime/tool identities, native TID, independent wall/thread CPU/process CPU boundaries, two process RSS observations, raw artifacts, and the compact summary through exact file SHA256 and byte counts. Those references identify captured bytes while retained by their owner; they are not new product-state authority or a perpetual-retention guarantee. Raw captures and full manifests stay private and external to the checkout. They include private source filenames. Reduced summaries expose bound module names/hashes/lines and function names; unbound filenames/functions become opaque hashes. No arguments, returned payloads, local variables, SQL, or parameter values are captured.

`cprofile` retains original `calls.pstats`, reported self/inclusive/call-count rankings, and reported module self totals. The qualified Python 3.14 implementation records interpreter-scoped monitoring events: an actual worker-thread discriminator confirms worker calls enter the same pstats. There is no native TID attribution or independent per-thread call stack in this data. Concurrent interleaving makes nesting and reported elapsed attribution unqualified; module self sums do not certify an exclusive partition. The default timer is elapsed, including waits and exposed C calls, with C-call internals unresolved. Inclusive rows overlap and must not be summed as CPU. Independent caller-thread CPU boundaries remain separate from merged profile statistics. An owner-established isolated single-thread retained-validation call remains a useful constituent diagnostic. For concurrent preparation/page work, use allocation diagnostics or owner CPU counters; do not claim a cProfile timing gain. All instrumented diagnostics remain excluded from unprofiled matched comparisons. [Python profiler documentation](https://docs.python.org/3.14/library/profile.html), [CPython 3.14.4 profiler implementation](https://github.com/python/cpython/blob/v3.14.4/Modules/_lsprof.c).

`tracemalloc` retains original before/after snapshots, signed net live byte/block differences, and traced current/peak bytes since its own reset. It covers traced Python allocations across the process, including other threads, with no thread attribution. It excludes allocations made before start and untraced native memory; net change is not allocation churn. Tracer/snapshot/helper costs are present. Process RSS observations remain separate and do not establish an interval peak. `traceback_frames` defaults to 1 and explicitly limits recorded allocation ancestry. [Python tracemalloc documentation](https://docs.python.org/3.14/library/tracemalloc.html).

`summary_rows` defaults to 20 per ranking and records omitted counts; this only bounds the reduced display. Original pstats/snapshots remain available. Serialization is outside the boundary interval; entry/exit bookkeeping remains in scope. Capture refuses an existing profiler/tracer, overlapping captures, and unsupported modes. `stack` specifically refuses unsupported stack sampling without reading live frames. There is no daemon attach, native/off-CPU attribution, or claim of complete multi-thread call coverage.

A callable exception propagates unchanged and records only its class. If diagnostic saving also fails, the original exception gains a class-only note and remains the failure. A successful callable with a diagnostic save/binding failure raises that diagnostic failure visibly. Killing the process can leave an incomplete directory without a final manifest; do not treat raw files alone as a completed diagnostic. Reuse the existing unprofiled component route for speed comparisons, independently of these captures.
