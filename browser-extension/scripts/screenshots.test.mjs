// Execute the screenshot entrypoint and the popup's real HTML/script route.
// Restoring the second popup.js injection makes this fail on its page error.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const script = join(dirname(fileURLToPath(import.meta.url)), "screenshots.mjs");

async function runHelper(helper, out) {
  const args = [helper, "--out", out];
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
  return { status, stdout, stderr };
}

test("screenshots initialize the real popup once and emit all declared PNG sizes", async () => {
  const out = await mkdtemp(join(tmpdir(), "polylogue-popup-screenshots-"));
  try {
    const { status, stdout, stderr } = await runHelper(script, out);
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

for (const transport of ["file", "http"]) {
  test(`screenshots fail when a popup script cannot load over ${transport}`, async () => {
    const fixture = await mkdtemp(join(tmpdir(), "polylogue-popup-broken-load-"));
    const server = transport === "http" ? createServer((_, response) => {
      response.writeHead(404, { "Content-Type": "text/javascript" });
      response.end();
    }) : null;
    try {
      await mkdir(join(fixture, "scripts"));
      await cp(script, join(fixture, "scripts", "screenshots.mjs"));
      await cp(join(dirname(script), "..", "src"), join(fixture, "src"), { recursive: true });
      // Import resolution must use the already installed test dependency.
      await cp(join(dirname(script), "..", "node_modules", "playwright"), join(fixture, "node_modules", "playwright"), { recursive: true });
      await cp(join(dirname(script), "..", "node_modules", "playwright-core"), join(fixture, "node_modules", "playwright-core"), { recursive: true });
      if (server) {
        await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
        const url = `http://127.0.0.1:${server.address().port}/missing-popup.js`;
        const htmlPath = join(fixture, "src", "popup.html");
        const html = await readFile(htmlPath, "utf8");
        await writeFile(htmlPath, html.replace('src="popup.js"', `src="${url}"`));
      } else {
        await rm(join(fixture, "src", "popup.js"));
      }
      const { status, stdout, stderr } = await runHelper(join(fixture, "scripts", "screenshots.mjs"), join(fixture, "screenshots"));
      assert.equal(status, 1, stderr);
      assert.match(stderr, /popup_script_load_failed/);
      assert.equal(stdout, "");
    } finally {
      if (server) await new Promise((resolve) => server.close(resolve));
      await rm(fixture, { recursive: true, force: true });
    }
  });
}
