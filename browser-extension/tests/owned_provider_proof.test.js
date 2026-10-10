// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, expect, it } from "vitest";
import { createOwnedProviderProofExtension, verifyOwnedProviderProofExtension } from "../scripts/owned_provider_extension.mjs";
import { captureProvider, currentProofFailure, evaluateJson, proofFailureReport } from "../scripts/live_provider_proof.mjs";
import { ownedCaptureDiagnostic, runOwnedProviderProof } from "../scripts/owned_provider_proof.mjs";
const owned = [];
afterEach(() => { for (const root of owned.splice(0)) rmSync(root, { recursive: true, force: true }); });
function fixture({ mismatch = false, alias = false, extra = false } = {}) {
  const root = mkdtempSync(path.join(tmpdir(), "polylogue-owned-provider-workflow-")); owned.push(root);
  const extensionRoot = path.join(root, "extension");
  const targets = [{ name: "chatgpt", url: "https://chatgpt.com/c/neutral-first", nativeId: "neutral-first" },
    { name: "claude", url: "https://claude.ai/chat/neutral-second", nativeId: "neutral-second" }];
  const binding = createOwnedProviderProofExtension({ destination: extensionRoot, hostName: `com.polylogue.browser_capture.proof_${"c".repeat(32)}`,
    receiverUrl: "http://127.0.0.1:18765", receiverId: "neutral-receiver", targets });
  const trace = [];
  const deps = { extensionRoot,
    control: async args => {
      trace.push(["control", ...args]);
      if (args[0] !== "load-extension") return {};
      const sealed = verifyOwnedProviderProofExtension(extensionRoot);
      expect(sealed.owned_targets_bound).toBe(true);
      expect(JSON.parse(readFileSync(path.join(extensionRoot, "owned-scope.json"), "utf8")).targets.map(row => row.windowId)).toEqual([21, 22]);
      return { id: mismatch ? "o".repeat(32) : binding.extension_id, path: extensionRoot };
    },
    browserVersion: async () => ({}),
    connect: async () => ({
      call: async (method, args) => {
        trace.push([method, args]);
        if (method === "Browser.getWindowForTarget") return { windowId: alias ? 21 : args.targetId === targets[0].url ? 21 : 22 };
        return {};
      }, close: () => trace.push(["browser.close"]),
    }),
    ownBrowser: () => {}, installCleanup: () => {}, requireHidden: async () => trace.push(["hidden"]),
    openWindow: async url => { trace.push(["open", url]); return url; },
    settleCleanup: async unload => { trace.push(["close.owned.windows"]); await unload(); },
    connectWorker: async () => ({ close: () => trace.push(["worker.close"]) }),
    connectPage: async () => ({ close: () => trace.push(["page.close"]) }),
    verifyInstalled: async () => ({ id: binding.extension_id, version: "0.3.0" }),
    evaluate: async (_client, expression) => {
      trace.push(["evaluate", expression]);
      if (expression.includes("ProofReady")) return true;
      if (expression.includes("startCapture")) return { ok: true };
      if (expression.includes("ownedTabs")) return [...targets.map((target, index) => ({ id: 11 + index, windowId: 21 + index, url: target.url })),
        ...(extra ? [{ id: 99, windowId: 90, url: targets[0].url }] : [])];
      const index = expression.includes('11, "neutral-first"') ? 0 : 1;
      const target = targets[index]; const provider = index === 0 ? "chatgpt" : "claude-ai";
      return { result: { ok: true, envelope: { session: { provider, provider_session_id: target.nativeId },
        receiver_native: { sha256: "a".repeat(64) }, capture_summary: { captureMode: "native_full", turnCount: 1, attachmentCount: 0 } },
        captureResult: { artifact_ref: "neutral-artifact", receiver_request_id: "neutral-request" } } };
    },
  };
  deps.capture = async (_worker, provider, tabId) => {
    const captured = await deps.evaluate(_worker, `(async () => ({result:await globalThis.__polylogueOwnedProviderProof.consumeCapture(${JSON.stringify(tabId)}, ${JSON.stringify(provider.nativeId)})}))()`);
    return captured;
  };
  return { deps, trace, binding, targets };
}

it("opens only declared targets, seals authority before load, and captures through the guarded production worker", async () => {
  const { deps, trace, binding, targets } = fixture();
  const result = await runOwnedProviderProof(deps);
  expect(result.ok).toBe(true); expect(result.automatic_capture_enabled).toBe(true);
  expect(result.isolation).toMatchObject({ admitted_tab_count: 2, static_content_scripts: false, document_bound_effects: true });
  const loadIndex = trace.findIndex(row => row[0] === "control" && row[1] === "load-extension");
  expect(trace.slice(0, loadIndex).filter(row => row[0] === "open").map(row => row[1])).toEqual(targets.map(row => row.url));
  expect(trace.filter(row => row[0] === "open").map(row => row[1])).toEqual([...targets.map(row => row.url), `chrome-extension://${binding.extension_id}/proof.html`]);
  expect(trace.filter(row => row[0] === "evaluate").every(row => !row[1].includes("chrome.tabs"))).toBe(true);
  expect(trace.findIndex(row => row[0] === "close.owned.windows")).toBeLessThan(trace.findIndex(row => row[0] === "Extensions.uninstall"));
  expect(trace).toContainEqual(["Extensions.uninstall", { id: binding.extension_id }]);
});

it("closes owned windows and uninstalls correlated load even if key binding fails", async () => {
  const { deps, trace } = fixture({ mismatch: true });
  await expect(runOwnedProviderProof(deps)).rejects.toThrow("proof_owned_provider_binding_invalid");
  expect(trace).toContainEqual(["close.owned.windows"]);
  expect(trace).toContainEqual(["Extensions.uninstall", { id: "o".repeat(32) }]);
  expect(trace.some(row => row[0] === "evaluate")).toBe(false);
});

it("refuses physical window aliasing before extension registration", async () => {
  const { deps, trace } = fixture({ alias: true });
  await expect(runOwnedProviderProof(deps)).rejects.toThrow("proof_owned_provider_binding_invalid");
  expect(trace.some(row => row[0] === "control" && row[1] === "load-extension")).toBe(false);
  expect(trace).toContainEqual(["close.owned.windows"]);
});

it("refuses an extra reported tab before consuming capture and still settles owned custody", async () => {
  const { deps, trace, binding } = fixture({ extra: true });
  await expect(runOwnedProviderProof(deps)).rejects.toThrow("proof_owned_tab_refused");
  expect(trace.filter(row => row[0] === "evaluate" && row[1].includes(".consumeCapture(")).length).toBe(0);
  expect(trace).toContainEqual(["Extensions.uninstall", { id: binding.extension_id }]);
});


it("reports failed automatic admission before consuming results", async () => {
  const { deps, trace } = fixture();
  const evaluate = deps.evaluate;
  deps.evaluate = async (client, expression) => expression.includes("startCapture") ? { ok: false, error: "private detail" } : evaluate(client, expression);
  let failure;
  try { await runOwnedProviderProof(deps); } catch (error) { failure = error; }
  expect(currentProofFailure(failure).error).toEqual({ phase: "capture_start", category: "automatic_capture_start_failed" });
  expect(trace.some(row => row[0] === "evaluate" && row[1].includes("consumeCapture"))).toBe(false);
});

it("retains fixed capture guard codes through CDP and the public phase boundary", async () => {
  for (const [code, category] of [["proof_automatic_capture_missing", "automatic_capture_missing"],
    ["proof_automatic_capture_pending", "automatic_capture_pending"],
    ["proof_capture_listener_invalid", "capture_listener_invalid"]]) {
    const client = { call: async () => ({ exceptionDetails: { exception: { description: `Error: ${code}\nprivate target details` } } }) };
    let failure;
    try { await evaluateJson(client, "neutral expression"); } catch (error) { failure = error; }
    const report = proofFailureReport("capture_result", failure);
    expect(report.error).toEqual({ phase: "capture_result", category });
    expect(JSON.stringify(report)).not.toContain("private target details");
  }
  expect(proofFailureReport("capture_membership", new Error("proof_owned_tab_refused")).error)
    .toEqual({ phase: "capture_membership", category: "provider_isolation_refused" });
});


it("retains missing-capture worker evidence before cleanup and preserves the closed refusal", async () => {
  const { deps, trace } = fixture();
  const evaluate = deps.evaluate;
  const details = { boundary_failures: [{ stage: "script_injection", error: "private selected-runtime exception" }], debug_log: [], state: null };
  deps.evaluate = async (client, expression) => expression.includes("captureFailureDetails") ? details : evaluate(client, expression);
  deps.capture = async () => { throw new Error("proof_automatic_capture_missing"); };
  deps.retainCaptureFailure = snapshot => { expect(snapshot).toBe(details); trace.push(["private.capture.diagnostic"]); };
  let failure;
  try { await runOwnedProviderProof(deps); } catch (error) { failure = error; }
  expect(currentProofFailure(failure).error).toEqual({ phase: "capture_result", category: "automatic_capture_missing" });
  expect(JSON.stringify(currentProofFailure(failure))).not.toContain("private selected-runtime exception");
  expect(trace.findIndex(row => row[0] === "private.capture.diagnostic")).toBeLessThan(trace.findIndex(row => row[0] === "close.owned.windows"));
});


it("retains every non-ok reply privately before sanitization and owned cleanup", async () => {
  const { deps, trace } = fixture();
  const destination = path.join(deps.extensionRoot, "private-capture-diagnostic.json");
  const diagnostic = ownedCaptureDiagnostic(destination);
  const evaluate = deps.evaluate;
  deps.evaluate = async (client, expression) => {
    if (expression.includes("captureFailureDetails")) return { boundary_failures: [], debug_log: [{ stage: "native_assets" }], state: { error: "private runtime detail" } };
    if (expression.includes("consumeCapture")) return { result: { ok: false, error: "private returned failure", outcome: "failed" } };
    return evaluate(client, expression);
  };
  deps.connectWorker = async () => ({ call: async (_method, args) => ({ result: { value: await deps.evaluate(null, args.expression) } }), close() {} });
  deps.capture = captureProvider;
  deps.retainCaptureFailure = details => { diagnostic.retain(details); trace.push(["private.capture.diagnostic"]); };
  const cleanup = deps.settleCleanup;
  deps.settleCleanup = async unload => {
    const snapshot = JSON.parse(readFileSync(destination, "utf8"));
    expect(snapshot.snapshots.map(row => row.capture_reply)).toEqual(["chatgpt", "claude-ai"].map(provider => ({
      provider, returned_ok: false, returned_error: "private returned failure", returned_outcome: "failed" })));
    expect(snapshot.snapshots.every(row => row.state.error === "private runtime detail")).toBe(true);
    expect(statSync(destination).mode & 0o777).toBe(0o600);
    await cleanup(unload);
  };
  let failure;
  try { await runOwnedProviderProof(deps); } catch (error) { failure = error; } finally { diagnostic.close(); }
  expect(currentProofFailure(failure).error).toEqual({ phase: "capture_result", category: "capture_incomplete" });
  expect(JSON.stringify(currentProofFailure(failure))).not.toContain("private returned failure");
  expect(trace.filter(row => row[0] === "private.capture.diagnostic")).toHaveLength(2);
  expect(trace.findIndex(row => row[0] === "private.capture.diagnostic")).toBeLessThan(trace.findIndex(row => row[0] === "close.owned.windows"));
  const duplicate = ownedCaptureDiagnostic(destination);
  try { expect(() => duplicate.retain({ replacement: true })).toThrow(); } finally { duplicate.close(); }
  expect(JSON.parse(readFileSync(destination, "utf8")).snapshots).toHaveLength(2);
});
