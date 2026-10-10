#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { bindOwnedProviderTargets, verifyOwnedProviderProofExtension } from "./owned_provider_extension.mjs";
import { connectCdp, publishProofFailure, evaluateJson, inProofPhase,
  installShutdownCleanup, openProofWindow, ownProofBrowser, pageClient, providerSummary,
  requireExpectedServiceContext, requireProofRunning, captureProvider, requireHiddenProofWorkspace, runChromeControl,
  verifyInstalledExtension, settleProofCleanup } from "./live_provider_proof.mjs";

const sleep = ms => new Promise(resolve => globalThis.setTimeout(resolve, ms));
async function workerFor(extensionId) {
  while (true) {
    requireProofRunning();
    const targets = await (await globalThis.fetch("http://127.0.0.1:9222/json/list")).json();
    const target = targets.find(row => row.type === "service_worker" && row.url === `chrome-extension://${extensionId}/proof_bootstrap.mjs`);
    if (target) return connectCdp(target.webSocketDebuggerUrl);
    await sleep(250);
  }
}

export async function runOwnedProviderProof({ extensionRoot, control = runChromeControl,
  connect = connectCdp, connectPage = pageClient, connectWorker = workerFor, evaluate = evaluateJson,
  verifyInstalled = verifyInstalledExtension, openWindow = openProofWindow,
  ownBrowser = ownProofBrowser, settleCleanup = settleProofCleanup,
  installCleanup = installShutdownCleanup, requireHidden = requireHiddenProofWorkspace, capture = captureProvider,
  retainCaptureFailure = null,
  browserVersion = async () => (await globalThis.fetch("http://127.0.0.1:9222/json/version")).json() }) {
  const initial = verifyOwnedProviderProofExtension(extensionRoot, { bound: false });
  const scope = JSON.parse(readFileSync(path.join(extensionRoot, "owned-scope.json"), "utf8"));
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  await inProofPhase("chrome_status", () => control(["status"]));
  const version = await browserVersion();
  const browser = await connect(version.webSocketDebuggerUrl);
  ownBrowser(browser);
  let worker = null; let page = null; let loadedId = null;
  let loading = null; let uninstalling = null;
  const unload = () => {
    if (uninstalling === null) uninstalling = (async () => {
      if (loading !== null) { try { await loading; } catch { /* A correlated id still owns removal. */ } }
      if (loadedId !== null) await browser.call("Extensions.uninstall", { id: loadedId });
    })();
    return uninstalling;
  };
  installCleanup({ afterTargets: unload });
  let failure = null; let result = null;
  const cleanupFailures = [];
  try {
    const targets = [];
    for (const selected of scope.targets) {
      await inProofPhase("provider_preflight", () => requireHidden());
      const targetId = await inProofPhase("provider_open", () => openWindow(selected.url, 10_000, control));
      const window = await browser.call("Browser.getWindowForTarget", { targetId });
      if (!Number.isInteger(window.windowId)) throw new Error("proof_owned_tab_refused");
      targets.push({ url: selected.url, windowId: window.windowId });
    }
    const binding = bindOwnedProviderTargets(extensionRoot, targets);
    requireProofRunning();
    loading = inProofPhase("extension_load", async () => {
      const loaded = await control(["load-extension", "--path", extensionRoot]);
      if (!/^[a-p]{32}$/.test(loaded.id || "") || typeof loaded.path !== "string"
          || path.resolve(loaded.path) !== path.resolve(extensionRoot)) throw new Error("proof_owned_provider_binding_invalid");
      loadedId = loaded.id;
      if (loadedId !== initial.extension_id) throw new Error("proof_owned_provider_binding_invalid");
    });
    await loading;
    worker = await inProofPhase("extension_startup", () => connectWorker(binding.extension_id));
    while (await evaluate(worker, "Boolean(globalThis.__polylogueOwnedProviderProofReady)") !== true) {
      requireProofRunning(); await sleep(100);
    }
    if (await evaluate(worker, "globalThis.__polylogueOwnedProviderProofReady") !== true) throw new Error("proof_owned_tab_refused");
    // Only this extension-owned page is loaded. The production popup's global
    // tab lookup is deliberately absent from the proof artifact.
    const pageTarget = await inProofPhase("popup_open", () => openWindow(`chrome-extension://${binding.extension_id}/proof.html`, 10_000, control));
    page = await connectPage(pageTarget, 30_000);
    const installed = await inProofPhase("revision", () => verifyInstalled(page, extensionRoot, binding.extension_id, manifest,
      ["proof.html", "owned_provider_browser.mjs", "proof_bootstrap.mjs", "owned-scope.json"]));
    await inProofPhase("capture_start", async () => {
      const started = await evaluate(worker, "globalThis.__polylogueOwnedProviderProof.startCapture()");
      if (started?.ok !== true) throw new Error("proof_automatic_capture_start_failed");
    });
    const admitted = await inProofPhase("capture_membership", async () => {
      const rows = await evaluate(worker, "globalThis.__polylogueOwnedProviderProof.ownedTabs()");
      if (!Array.isArray(rows) || rows.length !== scope.targets.length || new Set(rows.map(row => row.id)).size !== rows.length
          || rows.some(row => !Number.isInteger(row.id) || !targets.some(target => row.url === target.url
            && row.windowId === target.windowId && row.pinned !== true))) throw new Error("proof_owned_tab_refused");
      return rows;
    });
    const providers = {};
    for (const selected of scope.targets) {
      const tab = admitted.find(row => row.url === selected.url);
      if (!tab) throw new Error("proof_owned_tab_refused");
      const provider = { ...selected, host: new URL(selected.url).hostname, provider: selected.name === "chatgpt" ? "chatgpt" : "claude-ai" };
      const captured = await inProofPhase("capture_result", () => capture(worker, provider, tab.id));
      providers[provider.host] = providerSummary(provider, captured);
    }
    if (!Object.values(providers).every(row => row.ok === true)) throw new Error("proof_capture_incomplete");
    result = { ok: true, extension: installed, proof_binding: binding, providers, automatic_capture_enabled: true,
      isolation: { declared_window_count: targets.length, admitted_tab_count: admitted.length, static_content_scripts: false,
        current_window: "first_declared_owned_window", document_bound_effects: true } };
  } catch (error) {
    failure = error;
    if (error?.message === "proof_automatic_capture_missing" && worker !== null && retainCaptureFailure !== null) {
      try {
        const details = await evaluate(worker, "globalThis.__polylogueOwnedProviderProof.captureFailureDetails()");
        await retainCaptureFailure(details);
      } catch (diagnosticError) {
        failure = new AggregateError([error, diagnosticError], "proof_capture_diagnostic_failed", { cause: error });
      }
    }
  }
  finally {
    try { await settleCleanup(unload, failure); } catch (error) { cleanupFailures.push(error); }
    for (const client of [page, worker, browser]) {
      try { client?.close(); } catch (error) { cleanupFailures.push(error); }
    }
  }
  if (cleanupFailures.length) throw new AggregateError([...(failure ? [failure] : []), ...cleanupFailures], "proof_owned_provider_cleanup_failed", { cause: failure });
  if (failure) throw failure;
  return result;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  Promise.resolve().then(() => {
    requireExpectedServiceContext();
    const extensionRoot = process.env.POLYLOGUE_LIVE_PROVIDER_EXTENSION_ROOT;
    if (!extensionRoot) throw new Error("proof_owned_provider_binding_invalid");
    const diagnosticPath = process.env.POLYLOGUE_LIVE_PROVIDER_DIAGNOSTIC_PATH;
    return runOwnedProviderProof({ extensionRoot: path.resolve(extensionRoot),
      retainCaptureFailure: diagnosticPath ? details => writeFileSync(diagnosticPath, `${JSON.stringify(details)}\n`, { flag: "wx", mode: 0o600 }) : null });
  }).then(result => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch(error => { publishProofFailure(error); process.exitCode = 1; });
}
