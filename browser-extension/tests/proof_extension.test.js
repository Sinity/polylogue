// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, expect, it } from "vitest";
import { EventEmitter } from "node:events";
import { createProofExtension, verifyProofExtension } from "../scripts/proof_extension.mjs";
import { createOwnedTargetCleanup } from "../scripts/shared_chrome_proof_cleanup.mjs";
import { execFileSync } from "node:child_process";
import { firstControlJson } from "../scripts/shared_chrome_control.mjs";
import { runSharedChromeControlWorkflow } from "../scripts/dev_loop_shared_chrome_proof.mjs";

const owned = [];
afterEach(() => { for (const directory of owned.splice(0)) rmSync(directory, { recursive: true, force: true }); });

it("reads actual pretty control JSON while preserving diagnostic-prefixed single-line results", () => {
  const loaded = { id: "p".repeat(32), path: "/neutral/extension" };
  expect(firstControlJson(JSON.stringify(loaded, null, 2))).toEqual(loaded);
  expect(firstControlJson(`diagnostic\n${JSON.stringify(loaded)}\n`)).toEqual(loaded);
});

it("uninstalls the returned owned artifact even when its key identity differs", async () => {
  const calls = [];
  const root = path.resolve(".");
  await expect(runSharedChromeControlWorkflow({ extensionRoot: root,
    verifyBinding: () => ({ extension_id: "p".repeat(32) }), browserVersion: async () => ({}),
    control: async args => { calls.push(args); return args[0] === "load-extension" ? { id: "o".repeat(32), path: root } : {}; },
    connect: async () => ({ call: async (method, args) => calls.push([method, args]), close: () => calls.push(["browser.close"]) }),
  })).rejects.toThrow("proof_extension_identity_mismatch");
  expect(calls).toEqual([["status"], ["load-extension", "--path", root],
    ["Extensions.uninstall", { id: "o".repeat(32) }], ["browser.close"]]);
});

it("refuses the provider proof before any Chrome or native-manifest mutation", () => {
  let report;
  try { execFileSync(globalThis.process.execPath, [path.resolve("scripts/live_provider_proof.mjs")], { env: {}, stdio: "pipe" }); }
  catch (error) { report = JSON.parse(error.stdout.toString("utf8")); }
  expect(report).toMatchObject({ ok: false, error: { phase: "provider_preflight", category: "provider_isolation_refused" } });
});

it("binds a distinct page-only identity and exact candidate resources without capture permissions", () => {
  const root = mkdtempSync(path.join(tmpdir(), "polylogue-proof-extension-")); owned.push(root);
  const hostName = `com.polylogue.browser_capture.proof_${"a".repeat(32)}`;
  const first = path.join(root, "first"); const second = path.join(root, "second");
  const a = createProofExtension({ destination: first, hostName });
  const b = createProofExtension({ destination: second, hostName });
  expect(a.extension_id).not.toBe(b.extension_id);
  expect(verifyProofExtension(first)).toEqual(a);
  expect(JSON.parse(readFileSync(path.join(first, "manifest.json"), "utf8"))).toEqual({
    manifest_version: 3, name: "Polylogue isolated native transport proof", version: a.version,
    key: JSON.parse(readFileSync(path.join(first, "manifest.json"), "utf8")).key, permissions: ["nativeMessaging"],
  });
  expect(() => createProofExtension({ destination: first, hostName })).toThrow();
  const native = path.join(first, "src/background/native_fetch.js");
  writeFileSync(native, `${readFileSync(native, "utf8")}\n// unbound change\n`);
  expect(() => verifyProofExtension(first)).toThrow("proof_extension_resource_mismatch");
});

it("refuses a production host name and an added capture worker", () => {
  const root = mkdtempSync(path.join(tmpdir(), "polylogue-proof-extension-")); owned.push(root);
  expect(() => createProofExtension({ destination: path.join(root, "bad"), hostName: "com.polylogue.browser_capture" })).toThrow("proof_native_host_invalid");
  const destination = path.join(root, "good");
  createProofExtension({ destination, hostName: `com.polylogue.browser_capture.proof_${"b".repeat(32)}` });
  const manifestPath = path.join(destination, "manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  manifest.background = { service_worker: "src/background.js", type: "module" };
  writeFileSync(manifestPath, JSON.stringify(manifest));
  expect(() => verifyProofExtension(destination)).toThrow("proof_extension_binding_invalid");
});

it("settles the owned extension removal before propagating a signal, sharing normal cleanup", async () => {
  const processLike = new EventEmitter();
  const events = [];
  processLike.pid = 123; processLike.kill = () => events.push("signal");
  let close; let uninstall;
  const closing = new Promise(resolve => { close = resolve; });
  const removing = new Promise(resolve => { uninstall = resolve; });
  const cleanup = createOwnedTargetCleanup({ targetId: "A".repeat(32), processLike,
    control: async () => { events.push("close"); await closing; },
    afterClose: async () => { events.push("uninstall"); await removing; } });
  processLike.emit("SIGTERM");
  const finished = cleanup.finish();
  await Promise.resolve(); expect(events).toEqual(["close"]);
  close(); await new Promise(resolve => globalThis.setTimeout(resolve, 0));
  expect(events).toEqual(["close", "uninstall"]);
  uninstall(); await finished; await new Promise(resolve => globalThis.setTimeout(resolve, 0));
  expect(events).toEqual(["close", "uninstall", "signal"]);
});
