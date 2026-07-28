import { defineConfig } from "vitest/config";

// Worker concurrency cap (polylogue-0v5b): 2026-07-13 evidence showed an
// uncapped `npm test` spawning a swarm of `forks`-pool child processes
// (Vitest's default pool since 2.0 is `forks`, i.e. one `child_process`-forked
// node per test file, NOT worker_threads) inside an 8G sinnix-background
// agent scope; systemd-oomd killed the whole scope mid-iteration. Each fork
// carries its own jsdom global + module graph, so per-fork overhead compounds
// with test-file count and host CPU count (24 on the dev workstation, which
// is what the default fork count scales toward).
//
// DEFAULT_MAX_WORKERS mirrors the Python-side house convention
// (`devtools/verify.py: DEFAULT_TESTMON_WORKERS = "4"`) rather than
// hard-coding a single-worker policy — 4 keeps peak RSS well under an 8G
// scope while still running multiple test files concurrently. Override with
// POLYLOGUE_EXTENSION_TEST_WORKERS for a one-off wider/narrower run (e.g. CI
// runners with fewer CPUs, or a deliberately generous local run) without
// editing this file. Do not remove or raise this cap without new measured
// evidence — see tests/vitest_config.test.js.
const DEFAULT_MAX_WORKERS = 4;

function resolveMaxWorkers() {
  const raw = process.env.POLYLOGUE_EXTENSION_TEST_WORKERS;
  if (!raw) return DEFAULT_MAX_WORKERS;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return DEFAULT_MAX_WORKERS;
  return parsed;
}

const maxWorkers = resolveMaxWorkers();

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: false,
    include: ["tests/**/*.test.js"],
    // Applies uniformly to `vitest run` (local/CI) and `vitest` watch mode
    // (dev loop) — there is no separate CI test command to keep in sync.
    // Watch mode still runs affected files across up to `maxWorkers`
    // workers, and a single focused test file only ever needs one worker
    // regardless of this cap, so interactive parallelism is unaffected.
    //
    // `maxWorkers`/`minWorkers` is the pool-agnostic fallback Vitest resolves
    // for whichever pool is active; `poolOptions.forks.*` additionally pins
    // the cap for the current default `forks` pool explicitly, so the cap
    // survives even if the default pool changes again upstream.
    maxWorkers,
    minWorkers: 1,
    poolOptions: {
      forks: {
        maxForks: maxWorkers,
        minForks: 1,
      },
      threads: {
        maxThreads: maxWorkers,
        minThreads: 1,
      },
    },
  },
});
