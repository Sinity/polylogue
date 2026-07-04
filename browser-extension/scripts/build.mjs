#!/usr/bin/env node
// Build packed browser extension artifacts (Chrome .zip + Firefox .xpi).
//
// Usage:
//   node scripts/build.mjs                  # auto-detect version from pyproject.toml
//   node scripts/build.mjs --version 0.2.0  # override version explicitly
//   node scripts/build.mjs --out ./dist     # override output directory
//
// Produces:
//   <out>/polylogue-browser-capture-<version>-chrome.zip
//   <out>/polylogue-browser-capture-<version>-firefox.xpi
//
// Side effect: rewrites browser-extension/manifest.json + package.json
// "version" fields to match the resolved version so the committed source
// stays in sync with the published artifact. The Firefox artifact uses a
// dedicated manifest with browser_specific_settings.gecko populated.

import { execFileSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");
const REPO_ROOT = resolve(EXT_ROOT, "..");
const FIREFOX_GECKO_ID = "polylogue-browser-capture@sinity.dev";

function parseArgs(argv) {
  const args = { version: null, out: join(EXT_ROOT, "dist"), syncOnly: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--version") {
      args.version = argv[++i];
    } else if (a === "--out") {
      args.out = resolve(argv[++i]);
    } else if (a === "--sync-only") {
      args.syncOnly = true;
    } else if (a === "--help" || a === "-h") {
      process.stdout.write(
        "Usage: build.mjs [--version X.Y.Z] [--out DIR] [--sync-only]\n",
      );
      process.exit(0);
    } else {
      process.stderr.write(`unknown argument: ${a}\n`);
      process.exit(2);
    }
  }
  return args;
}

function readPyprojectVersion() {
  const path = join(REPO_ROOT, "pyproject.toml");
  const text = readFileSync(path, "utf8");
  // Restrict to the [project] table to avoid matching [tool.*] version pins.
  const projectSection = text.split(/^\[/m).find((s) => s.startsWith("project]"));
  if (!projectSection) throw new Error("no [project] table in pyproject.toml");
  const match = projectSection.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("no version = \"...\" in [project] table");
  return match[1];
}

function normalizeChromeVersion(v) {
  // Chrome Web Store requires manifest version of form 1-4 dot-separated
  // integers, each <= 65535, no leading zeros. Strip any pre-release/dev
  // suffix (e.g. "0.2.0.dev1+gabc" → "0.2.0").
  const cleaned = v.replace(/[+-].*$/, "").replace(/\.dev\d*$/, "");
  const parts = cleaned.split(".").filter((p) => /^\d+$/.test(p));
  if (parts.length === 0) throw new Error(`cannot derive chrome version from "${v}"`);
  return parts.slice(0, 4).join(".");
}

function syncJsonVersion(path, version) {
  const original = readFileSync(path, "utf8");
  const data = JSON.parse(original);
  if (data.version === version) return false;
  data.version = version;
  // Preserve trailing newline if present.
  const serialized = JSON.stringify(data, null, 2) + (original.endsWith("\n") ? "\n" : "");
  writeFileSync(path, serialized);
  return true;
}

function buildFirefoxManifest(manifest) {
  const fxManifest = JSON.parse(JSON.stringify(manifest));
  // Firefox does not yet support "service_worker" in MV3; declare a
  // scripts-array background fallback in addition.
  if (fxManifest.background && fxManifest.background.service_worker) {
    fxManifest.background = {
      ...fxManifest.background,
      scripts: [fxManifest.background.service_worker],
    };
  }
  fxManifest.browser_specific_settings = {
    gecko: {
      id: FIREFOX_GECKO_ID,
      strict_min_version: "121.0",
    },
  };
  return fxManifest;
}

function copyTree(src, dst) {
  // Use cp -a to preserve permissions; portable across CI runners that
  // ship Node + coreutils (Linux/macOS). Windows is not supported as a
  // build host for this artifact.
  execFileSync("cp", ["-a", `${src}/.`, dst], { stdio: "inherit" });
}

const ARCHIVE_EXCLUDE_DIRS = new Set([
  ".cache",
  ".local",
  "node_modules",
  "tests",
  "scripts",
  "dist",
  "screenshots",
]);
const ARCHIVE_EXCLUDE_FILES = new Set([
  "eslint.config.js",
  "vitest.config.js",
  "package-lock.json",
  ".DS_Store",
]);

function makeArchive(srcDir, archivePath, kind) {
  // Try zip(1) first (CI runners ship it). Fall back to Python's
  // zipfile so the build still runs inside the Nix devshell, which
  // ships Python but not unzip/zip.
  const which = (bin) => {
    try {
      execFileSync("sh", ["-c", `command -v ${bin}`], { stdio: "pipe" });
      return true;
    } catch {
      return false;
    }
  };
  if (which("zip")) {
    const args = ["-r", "-q", archivePath, "."];
    for (const dir of ARCHIVE_EXCLUDE_DIRS) args.push("-x", `${dir}/*`);
    for (const file of ARCHIVE_EXCLUDE_FILES) args.push("-x", file);
    execFileSync("zip", args, { cwd: srcDir, stdio: "inherit" });
  } else {
    const py = `
import os, sys, zipfile, pathlib
root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
exclude_dirs = set(${JSON.stringify(Array.from(ARCHIVE_EXCLUDE_DIRS))})
exclude_files = set(${JSON.stringify(Array.from(ARCHIVE_EXCLUDE_FILES))})
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for name in filenames:
            if name in exclude_files:
                continue
            full = pathlib.Path(dirpath) / name
            zf.write(full, full.relative_to(root).as_posix())
`;
    execFileSync("python3", ["-c", py, srcDir, archivePath], { stdio: "inherit" });
  }
  process.stdout.write(`built ${kind}: ${relative(EXT_ROOT, archivePath)}\n`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rawVersion = args.version ?? readPyprojectVersion();
  const version = normalizeChromeVersion(rawVersion);
  process.stdout.write(`polylogue browser-extension build: version ${version} (raw ${rawVersion})\n`);

  const manifestPath = join(EXT_ROOT, "manifest.json");
  const packagePath = join(EXT_ROOT, "package.json");
  syncJsonVersion(manifestPath, version);
  syncJsonVersion(packagePath, version);

  if (args.syncOnly) {
    process.stdout.write("sync-only mode — skipping archive build\n");
    return;
  }

  if (!existsSync(args.out)) mkdirSync(args.out, { recursive: true });

  const stageRoot = mkdtempSync(join(tmpdir(), "polylogue-ext-build-"));
  try {
    // --- Chrome artifact ---
    const chromeDir = join(stageRoot, "chrome");
    mkdirSync(chromeDir);
    copyTree(EXT_ROOT, chromeDir);
    const chromeArchive = join(args.out, `polylogue-browser-capture-${version}-chrome.zip`);
    if (existsSync(chromeArchive)) rmSync(chromeArchive);
    makeArchive(chromeDir, chromeArchive, "chrome zip");

    // --- Firefox artifact ---
    const firefoxDir = join(stageRoot, "firefox");
    mkdirSync(firefoxDir);
    copyTree(EXT_ROOT, firefoxDir);
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const fxManifest = buildFirefoxManifest(manifest);
    writeFileSync(join(firefoxDir, "manifest.json"), JSON.stringify(fxManifest, null, 2) + "\n");
    const firefoxArchive = join(args.out, `polylogue-browser-capture-${version}-firefox.xpi`);
    if (existsSync(firefoxArchive)) rmSync(firefoxArchive);
    makeArchive(firefoxDir, firefoxArchive, "firefox xpi");

    // --- Manifest summary for downstream upload jobs ---
    const summary = {
      version,
      raw_version: rawVersion,
      artifacts: {
        chrome: relative(args.out, chromeArchive) || chromeArchive,
        firefox: relative(args.out, firefoxArchive) || firefoxArchive,
      },
      firefox_gecko_id: FIREFOX_GECKO_ID,
    };
    writeFileSync(join(args.out, "build-manifest.json"), JSON.stringify(summary, null, 2) + "\n");
    process.stdout.write("wrote build-manifest.json\n");
  } finally {
    rmSync(stageRoot, { recursive: true, force: true });
  }
}

main();
