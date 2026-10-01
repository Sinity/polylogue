// Execute the screenshot entrypoint and the popup's real HTML/script route.
// Restoring the second popup.js injection makes this fail on its page error.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const script = join(dirname(fileURLToPath(import.meta.url)), "screenshots.mjs");

test("screenshots initialize the real popup once and emit all declared PNG sizes", async () => {
  const out = await mkdtemp(join(tmpdir(), "polylogue-popup-screenshots-"));
  try {
    const args = [script, "--out", out];
    if (process.env.POLYLOGUE_SCREENSHOT_TEST_BROWSER) {
      args.push("--browser-executable", process.env.POLYLOGUE_SCREENSHOT_TEST_BROWSER);
    }
    const child = spawn(process.execPath, args);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    const status = await new Promise((resolve, reject) => {
      child.on("error", reject);
      child.on("close", resolve);
    });
    assert.equal(status, 0, stderr);
    const names = [];
    for (const state of ["online-captured", "online-unsupported", "offline"]) {
      for (const [width, height] of [[1280, 800], [640, 400], [750, 1334]]) {
        const name = `popup-${state}-${width}x${height}.png`;
        names.push(name);
        const png = await readFile(join(out, name));
        assert.deepEqual(png.subarray(0, 8), Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
        assert.equal(png.readUInt32BE(16), width * 2);
        assert.equal(png.readUInt32BE(20), height * 2);
      }
    }
    assert.deepEqual((await readdir(out)).sort(), names.sort());
    assert.equal(stdout.split("\n").filter((line) => line.startsWith("captured ")).length, 9);
  } finally {
    await rm(out, { recursive: true, force: true });
  }
});
