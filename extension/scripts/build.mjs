#!/usr/bin/env node
// Build Chrome ZIP and Firefox XPI artifacts from explicit runtime inputs.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { browserManifest, browserVersion, FIREFOX_GECKO_ID, runtimeFiles } from "./package.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const args = { out: join(root, "dist"), version: null };
for (let at = 2; at < process.argv.length; at += 1) {
  const flag = process.argv[at];
  if (flag === "--help" || flag === "-h") {
    process.stdout.write("Usage: build.mjs [--version VERSION] [--out DIR]\n");
    process.exit(0);
  }
  const value = process.argv[at + 1];
  if (!["--version", "--out"].includes(flag) || !value || value.startsWith("--")) {
    throw new Error(`invalid argument: ${flag}`);
  }
  args[flag.slice(2)] = value;
  at += 1;
}
const out = resolve(args.out);
const rawVersion = args.version ?? JSON.parse(readFileSync(join(root, "package.json"), "utf8")).version;
const version = browserVersion(rawVersion);
const source = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
for (const file of runtimeFiles) {
  if (!existsSync(join(root, file))) throw new Error(`missing runtime file: ${file}`);
}
mkdirSync(out, { recursive: true });
const stageRoot = mkdtempSync(join(tmpdir(), "polylogue-extension-build-"));
try {
  const artifacts = {};
  for (const [browser, suffix] of [["chrome", "zip"], ["firefox", "xpi"]]) {
    const stage = join(stageRoot, browser);
    for (const file of runtimeFiles) {
      mkdirSync(dirname(join(stage, file)), { recursive: true });
      if (file === "manifest.json") {
        writeFileSync(join(stage, file), JSON.stringify(browserManifest(source, version, browser), null, 2) + "\n");
      } else copyFileSync(join(root, file), join(stage, file));
    }
    const filename = `polylogue-browser-capture-${version}-${browser}.${suffix}`;
    const archive = join(out, filename);
    // zip updates an existing archive, so removed inputs must not survive it.
    rmSync(archive, { force: true });
    try {
      execFileSync("zip", ["-q", archive, ...runtimeFiles], { cwd: stage, stdio: "pipe" });
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      execFileSync("python3", ["-c", `import json, sys, zipfile
from pathlib import Path
root, archive = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for name in json.loads(sys.argv[3]):
        output.write(root / name, name)
`, stage, archive, JSON.stringify(runtimeFiles)], { stdio: "pipe" });
    }
    artifacts[browser] = filename;
  }
  writeFileSync(join(out, "build-manifest.json"), JSON.stringify({
    version, raw_version: rawVersion, artifacts, firefox_gecko_id: FIREFOX_GECKO_ID,
  }, null, 2) + "\n");
} finally {
  rmSync(stageRoot, { recursive: true, force: true });
}
