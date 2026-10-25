import { defineConfig } from "vitest/config";

// The service worker has no DOM, so tests run in Node; the page-script tests
// build their own jsdom page. At most four test files run at once, which
// keeps memory small on a shared machine; POLYLOGUE_EXTENSION_TEST_WORKERS
// changes that for one run.
const workers = Number.parseInt(process.env.POLYLOGUE_EXTENSION_TEST_WORKERS || "", 10);

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.js"],
    pool: "forks",
    maxWorkers: Number.isFinite(workers) && workers > 0 ? workers : 4,
  },
});
