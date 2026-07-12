// Tests for scripts/build.mjs + scripts/validate-manifest.mjs.
//
// These exercise the version sync + Firefox manifest transform + archive
// emission so a future change to the build pipeline cannot silently break
// the release artifact shape.

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");
const BUILD = join(EXT_ROOT, "scripts", "build.mjs");
const VALIDATE = join(EXT_ROOT, "scripts", "validate-manifest.mjs");
const MANIFEST_PATH = join(EXT_ROOT, "manifest.json");
const PACKAGE_PATH = join(EXT_ROOT, "package.json");

const ORIGINAL_MANIFEST = readFileSync(MANIFEST_PATH, "utf8");
const ORIGINAL_PACKAGE = readFileSync(PACKAGE_PATH, "utf8");

function restore() {
  writeFileSync(MANIFEST_PATH, ORIGINAL_MANIFEST);
  writeFileSync(PACKAGE_PATH, ORIGINAL_PACKAGE);
}

describe("validate-manifest.mjs", () => {
  it("accepts the committed manifest", () => {
    expect(() => execFileSync("node", [VALIDATE], { stdio: "pipe" })).not.toThrow();
  });

  it("rejects a manifest with an overly broad host permission", async () => {
    const dir = await mkdtemp(join(tmpdir(), "polylogue-ext-validate-"));
    try {
      const broken = JSON.parse(ORIGINAL_MANIFEST);
      broken.host_permissions = ["<all_urls>"];
      const path = join(dir, "manifest.json");
      writeFileSync(path, JSON.stringify(broken, null, 2));
      let threw = false;
      try {
        execFileSync("node", [VALIDATE, path], { stdio: "pipe" });
      } catch {
        threw = true;
      }
      expect(threw).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("build.mjs", () => {
  afterEach(restore);

  it("rewrites manifest + package.json to the requested version", () => {
    execFileSync("node", [BUILD, "--version", "9.8.7", "--sync-only"], { stdio: "pipe" });
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    const pkg = JSON.parse(readFileSync(PACKAGE_PATH, "utf8"));
    expect(manifest.version).toBe("9.8.7");
    expect(pkg.version).toBe("9.8.7");
  });

  it("strips dev/pre-release suffixes before writing the Chrome version", () => {
    execFileSync("node", [BUILD, "--version", "1.2.3.dev4+gabc", "--sync-only"], { stdio: "pipe" });
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    expect(manifest.version).toBe("1.2.3");
  });
});

describe("build.mjs full archive emission", () => {
  let outDir;
  beforeEach(async () => {
    outDir = await mkdtemp(join(tmpdir(), "polylogue-ext-out-"));
  });
  afterEach(() => {
    rmSync(outDir, { recursive: true, force: true });
    restore();
  });

  it("emits chrome zip + firefox xpi with build-manifest.json", () => {
    execFileSync("node", [BUILD, "--version", "0.9.0", "--out", outDir], { stdio: "pipe" });
    expect(existsSync(join(outDir, "build-manifest.json"))).toBe(true);
    expect(existsSync(join(outDir, "polylogue-browser-capture-0.9.0-chrome.zip"))).toBe(true);
    expect(existsSync(join(outDir, "polylogue-browser-capture-0.9.0-firefox.xpi"))).toBe(true);
    const summary = JSON.parse(readFileSync(join(outDir, "build-manifest.json"), "utf8"));
    expect(summary.version).toBe("0.9.0");
    expect(summary.firefox_gecko_id).toMatch(/@/);
    const listing = execFileSync(
      "python3",
      ["-c", "import sys,zipfile; print('\\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))", join(outDir, "polylogue-browser-capture-0.9.0-chrome.zip")],
      { encoding: "utf8" },
    );
    expect(listing).toContain("src/backfill/coordinator.js");
    expect(listing).toContain("src/backfill/providers.js");
    expect(listing).toContain("src/backfill/storage.js");
  });
});
