import { expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { browserVersion } from "../scripts/package.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const sourceManifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
const sourceVersion = JSON.parse(readFileSync(join(root, "package.json"), "utf8")).version;
// Independent release oracle, separate from the production input declaration.
const members = [
  "manifest.json", "src/actions/chatgpt.js", "src/background.js",
  "src/background/backfill.js", "src/background/capture.js", "src/background/chat_actions.js",
  "src/background/host.js", "src/background/main.js", "src/background/notes.js",
  "src/background/outbox.js", "src/background/provider.js", "src/background/tabs.js",
  "src/content/chatgpt.js", "src/content/claude.js", "src/content/gemini.js",
  "src/content/grok.js", "src/content/page.js", "src/content/panel.js",
  "src/popup.html", "src/popup.js", "src/wording.js",
].sort();
const hosts = ["https://chatgpt.com/*", "https://claude.ai/*", "https://grok.com/*", "https://gemini.google.com/*"];
const tools = JSON.parse(execFileSync("python3", ["-c", "import json, shutil; print(json.dumps({name: shutil.which(name) for name in ['zip', 'python3']}))"], { encoding: "utf8" }));

function sourceHashes() {
  const hashes = {};
  function visit(directory) {
    for (const entry of readdirSync(join(root, directory), { withFileTypes: true })) {
      if (["node_modules", ".npm-cache", ".cache", "dist"].includes(entry.name)) continue;
      const file = join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) hashes[file] = createHash("sha256").update(readFileSync(join(root, file))).digest("hex");
    }
  }
  visit("");
  return hashes;
}

function inspect(archive) {
  return JSON.parse(execFileSync(tools.python3, ["-c", `import hashlib, json, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    print(json.dumps({"members": sorted(archive.namelist()), "manifest": json.loads(archive.read("manifest.json")), "hashes": {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name != "manifest.json"}}))
`, archive], { encoding: "utf8" }));
}

function verify(out, version, rawVersion, before) {
  const summary = JSON.parse(readFileSync(join(out, "build-manifest.json"), "utf8"));
  expect(summary).toEqual({
    version, raw_version: rawVersion, firefox_gecko_id: "polylogue-browser-capture@sinity.dev",
    artifacts: { chrome: `polylogue-browser-capture-${version}-chrome.zip`, firefox: `polylogue-browser-capture-${version}-firefox.xpi` },
  });
  for (const browser of ["chrome", "firefox"]) {
    const actual = inspect(join(out, summary.artifacts[browser]));
    expect(actual.members).toEqual(members);
    expect(actual.manifest.host_permissions).toEqual(hosts);
    expect(actual.manifest.permissions).toEqual(["alarms", "nativeMessaging", "scripting", "storage", "tabs"]);
    const expected = structuredClone(sourceManifest);
    expected.version = version;
    if (browser === "firefox") {
      expected.background = { service_worker: "src/background.js", type: "module", scripts: ["src/background.js"] };
      expected.browser_specific_settings = { gecko: { id: "polylogue-browser-capture@sinity.dev", strict_min_version: "121.0" } };
    }
    expect(actual.manifest).toEqual(expected);
    for (const [file, hash] of Object.entries(actual.hashes)) expect(hash, file).toBe(before[file]);
  }
  expect(sourceHashes()).toEqual(before);
  return summary;
}

it.each(["zip", "python"])("builds exact artifacts with real %s without rewriting source", (backend) => {
  expect(tools.python3, "python3 is required to inspect real archives").toBeTruthy();
  expect(tools.zip, "zip is required to qualify both archive backends").toBeTruthy();
  const directory = mkdtempSync(join(tmpdir(), "polylogue-package-test-"));
  try {
    const bin = join(directory, "bin");
    const out = join(directory, "out");
    mkdirSync(bin);
    symlinkSync(tools.python3, join(bin, "python3"));
    if (backend === "zip") symlinkSync(tools.zip, join(bin, "zip"));
    // Node is invoked by absolute path. This PATH contains actual tools only;
    // omitting zip exercises the production ENOENT fallback, not a fake archiver.
    const env = { ...process.env, PATH: bin };
    const before = sourceHashes();
    const build = (...args) => execFileSync(process.execPath, [join(root, "scripts/build.mjs"), "--out", out, ...args], { env, stdio: "pipe" });
    build();
    const summary = verify(out, sourceVersion, sourceVersion, before);
    for (const archive of Object.values(summary.artifacts)) {
      execFileSync(tools.python3, ["-c", "import sys, zipfile; archive=zipfile.ZipFile(sys.argv[1], 'a'); archive.writestr('stale.txt', 'old release'); archive.close()", join(out, archive)]);
    }
    build();
    verify(out, sourceVersion, sourceVersion, before);
    build("--version", "0.5.0.dev2+fixture");
    verify(out, "0.5.0", "0.5.0.dev2+fixture", before);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

it("rejects malformed versions before creating output or modifying source", () => {
  expect(browserVersion("0.5.0.dev2+fixture")).toBe("0.5.0");
  expect(browserVersion("1.2.3.65535")).toBe("1.2.3.65535");
  const directory = mkdtempSync(join(tmpdir(), "polylogue-package-invalid-"));
  try {
    const out = join(directory, "out");
    const before = sourceHashes();
    for (const version of ["1.bad.2", "01.2", "1.65536", "1.2.3.4.5", "1..2", ""]) {
      expect(() => browserVersion(version)).toThrow("invalid browser version");
      expect(() => execFileSync(process.execPath, [join(root, "scripts/build.mjs"), "--out", out, "--version", version], { stdio: "pipe" })).toThrow();
      expect(existsSync(out)).toBe(false);
    }
    for (const flag of ["--out", "--version"]) {
      expect(() => execFileSync(process.execPath, [join(root, "scripts/build.mjs"), flag], { stdio: "pipe" })).toThrow();
    }
    expect(sourceHashes()).toEqual(before);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
