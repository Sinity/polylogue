// Regression guard for polylogue-0v5b: caps `npm test` worker concurrency so
// a full-suite run stays well under an 8G sinnix-background agent scope.
// 2026-07-13 evidence: an uncapped run let Vitest's default `forks` pool
// (one child_process per test file, NOT worker_threads) scale toward the
// host's CPU count; systemd-oomd killed the whole scope mid-iteration.
//
// This deliberately does NOT dynamically `import()` vitest.config.js: that
// module transitively pulls in `vitest/config` -> vite -> esbuild, and
// esbuild's startup invariant check breaks when run a second time inside
// this suite's jsdom environment (`TextEncoder().encode("") instanceof
// Uint8Array` comes back false under jsdom's polyfilled globals). Parsing
// the source text is also more honest here: it pins the literal config
// shape a future edit could regress, without needing a real Vitest runtime
// to evaluate it.
//
// This test does not re-run the suite or measure RSS (see the PR/bead for
// the measured before/after numbers) — it only pins the *shape* of the
// config so a future edit can't silently drop the cap or raise it back to
// an unbounded value without a deliberate, reviewed change.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, "..", "vitest.config.js");
const CONFIG_SOURCE = readFileSync(CONFIG_PATH, "utf8");

// A generous ceiling, not the house default (4) — this only guards against
// the cap being removed or raised to something effectively unbounded
// (e.g. left to default to host CPU count, which was the original bug).
const MAX_ALLOWED_WORKERS = 8;

function extractDefaultMaxWorkers(source) {
  const match = source.match(/const\s+DEFAULT_MAX_WORKERS\s*=\s*(\d+)\s*;/);
  if (!match) return null;
  return Number.parseInt(match[1], 10);
}

describe("vitest.config.js worker concurrency cap", () => {
  it("declares a numeric, bounded default worker cap", () => {
    const defaultMaxWorkers = extractDefaultMaxWorkers(CONFIG_SOURCE);
    expect(defaultMaxWorkers, "DEFAULT_MAX_WORKERS constant must exist").not.toBeNull();
    expect(defaultMaxWorkers).toBeGreaterThanOrEqual(1);
    expect(defaultMaxWorkers).toBeLessThanOrEqual(MAX_ALLOWED_WORKERS);
  });

  it("wires the cap into the active (forks) pool", () => {
    expect(CONFIG_SOURCE).toMatch(/poolOptions\s*:\s*{[\s\S]*forks\s*:\s*{[\s\S]*maxForks\s*:\s*maxWorkers/);
  });

  it("wires the cap into the pool-agnostic maxWorkers fallback", () => {
    expect(CONFIG_SOURCE).toMatch(/test\s*:\s*{[\s\S]*maxWorkers\s*,/);
  });

  it("keeps the threads pool capped too, in case the default pool changes again", () => {
    expect(CONFIG_SOURCE).toMatch(/poolOptions\s*:\s*{[\s\S]*threads\s*:\s*{[\s\S]*maxThreads\s*:\s*maxWorkers/);
  });

  it("supports a bounded env override without unbounding the default", () => {
    // The override must be validated (falls back to a safe default on
    // garbage input) rather than passed through raw.
    expect(CONFIG_SOURCE).toMatch(/POLYLOGUE_EXTENSION_TEST_WORKERS/);
    expect(CONFIG_SOURCE).toMatch(/Number\.isFinite\(parsed\)/);
    expect(CONFIG_SOURCE).toMatch(/parsed\s*<\s*1/);
  });
});
