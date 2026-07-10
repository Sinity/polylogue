#!/usr/bin/env node
// Lightweight in-tree manifest validator.
//
// Catches the failure modes we have actually hit during local builds before
// the heavier web-ext lint is invoked from CI. Specifically:
//   - manifest_version is 3
//   - version is 1-4 dot-separated integers, each <= 65535
//   - all content_scripts[*].js paths exist on disk
//   - background.service_worker exists on disk
//   - host_permissions do not include "<all_urls>" or "*://*/*"
//   - permissions list is a subset of an allowlist (no surprising additions)
//
// Exit non-zero with a summary on failure. Intended for `npm run validate`
// and as a pre-archive gate in scripts/build.mjs callers.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");

const ALLOWED_PERMISSIONS = new Set([
  "activeTab",
  "alarms",
  "scripting",
  "storage",
  "tabs",
]);
const ALLOWED_HOST_GLOBS = [
  /^https:\/\/chatgpt\.com\/\*$/,
  /^https:\/\/claude\.ai\/\*$/,
  /^https:\/\/grok\.com\/\*$/,
  /^https:\/\/x\.com\/\*$/,
  /^https:\/\/twitter\.com\/\*$/,
  /^http:\/\/127\.0\.0\.1\/\*$/,
];

function fail(errors) {
  process.stderr.write(`manifest validation failed (${errors.length} issues):\n`);
  for (const err of errors) process.stderr.write(`  - ${err}\n`);
  process.exit(1);
}

function validate(manifestPath) {
  const errors = [];
  if (!existsSync(manifestPath)) {
    fail([`manifest not found at ${manifestPath}`]);
  }
  const raw = readFileSync(manifestPath, "utf8");
  let manifest;
  try {
    manifest = JSON.parse(raw);
  } catch (err) {
    fail([`manifest is not valid JSON: ${err.message}`]);
  }

  if (manifest.manifest_version !== 3) {
    errors.push(`manifest_version must be 3, got ${manifest.manifest_version}`);
  }
  if (typeof manifest.version !== "string" || !/^\d+(\.\d+){0,3}$/.test(manifest.version)) {
    errors.push(`version must be 1-4 dot-separated integers, got ${JSON.stringify(manifest.version)}`);
  } else {
    for (const part of manifest.version.split(".")) {
      if (Number.parseInt(part, 10) > 65535) {
        errors.push(`version component ${part} exceeds Chrome maximum 65535`);
      }
    }
  }
  if (!manifest.name) errors.push("name is required");
  if (!manifest.description) errors.push("description is required");

  for (const perm of manifest.permissions ?? []) {
    if (!ALLOWED_PERMISSIONS.has(perm)) {
      errors.push(`permission "${perm}" not in allowlist (declared narrowest set)`);
    }
  }
  for (const host of manifest.host_permissions ?? []) {
    if (host === "<all_urls>" || host === "*://*/*") {
      errors.push(`host permission "${host}" is too broad`);
    }
    const ok = ALLOWED_HOST_GLOBS.some((re) => re.test(host));
    if (!ok) errors.push(`host_permission "${host}" not in declared allowlist`);
  }

  const sw = manifest.background?.service_worker;
  if (sw) {
    const swPath = join(EXT_ROOT, sw);
    if (!existsSync(swPath)) errors.push(`background.service_worker missing on disk: ${sw}`);
  } else if (!manifest.background?.scripts) {
    errors.push("background must declare service_worker or scripts");
  }
  for (const entry of manifest.content_scripts ?? []) {
    for (const js of entry.js ?? []) {
      if (!existsSync(join(EXT_ROOT, js))) {
        errors.push(`content_script js missing on disk: ${js}`);
      }
    }
  }
  if (manifest.action?.default_popup) {
    const popup = join(EXT_ROOT, manifest.action.default_popup);
    if (!existsSync(popup)) errors.push(`action.default_popup missing on disk: ${manifest.action.default_popup}`);
  }

  if (errors.length > 0) fail(errors);
  process.stdout.write(`manifest ok: ${manifestPath} (v${manifest.version})\n`);
}

const manifestPath = process.argv[2]
  ? resolve(process.argv[2])
  : join(EXT_ROOT, "manifest.json");
validate(manifestPath);
