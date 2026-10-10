#!/usr/bin/env node
// Shared-Chrome control proof for the deterministic dev-loop operation. It
// never launches a browser, allocates a debugging port, or creates a profile.

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { assertAgentWindow, firstControlJson, runChromeControlBytes } from "./shared_chrome_control.mjs";
import { createOwnedTargetCleanup } from "./shared_chrome_proof_cleanup.mjs";
import { connectCdp, evaluateJson, pageClient, verifyInstalledExtension } from "./live_provider_proof.mjs";
import { verifyProofExtension } from "./proof_extension.mjs";

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be supplied by the declared dev-loop service`);
  return value;
}

function requireExpectedServiceContext() {
  // The runtime exports AGENTCTL_*; older hosts export the same values as SINNIXD_*.
  const prefix = process.env.AGENTCTL_JOB_ID ? "AGENTCTL_" : "SINNIXD_";
  if (!process.env[`${prefix}JOB_ID`]) throw new Error("shared-Chrome dev-loop proof requires a runtime job id");
  if (process.env[`${prefix}PROJECT_ID`] !== "polylogue" || process.env[`${prefix}OPERATION`] !== "dev_loop_proof") {
    throw new Error("shared-Chrome dev-loop proof rejects execution outside the fixed dev-loop service context");
  }
  const cgroup = readFileSync("/proc/self/cgroup", "utf8").split("\n").find((line) => line.includes("::"))?.split("::", 2)[1] || "";
  if (!cgroup.split("/").includes("agentctl-interactive.slice") && !cgroup.split("/").includes("sinnixd-pueue-interactive.slice")) {
    throw new Error("shared-Chrome dev-loop proof is not inside the interactive pool");
  }
}

export { assertAgentWindow } from "./shared_chrome_control.mjs";

// Only the isolated neutral proof page may retain an unknown CDP description.
// This private artifact never crosses the ordinary provider error boundary.
export function retainNeutralEvaluationDiagnostic(destination, exceptionDetails) {
  try {
    writeFileSync(destination, `${JSON.stringify({ exceptionDetails })}\n`, { flag: "wx", mode: 0o600 });
  } catch { throw new Error("proof_evaluation_diagnostic_failed"); }
}

export async function runChromeControl(args, timeoutMs, spawnCommand) {
  return firstControlJson(await runChromeControlBytes(args, timeoutMs, spawnCommand)) || {};
}

export async function runSharedChromeControlWorkflow({ extensionRoot, transportInputs, control = runChromeControl,
  verifyBinding = verifyProofExtension, connect = connectCdp, connectPage = pageClient,
  verifyInstalled = verifyInstalledExtension, evaluate = evaluateJson, retainUnknownException = null,
  browserVersion = async () => (await globalThis.fetch("http://127.0.0.1:9222/json/version")).json() }) {
  const binding = verifyBinding(extensionRoot);
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  await control(["status"]);
  const version = await browserVersion();
  const browser = await connect(version.webSocketDebuggerUrl);
  let createdTargetId = null;
  let cleanup = null;
  let page = null;
  let loadedId = null;
  let unloadPromise = null;
  const unload = () => {
    if (unloadPromise === null) unloadPromise = Promise.resolve().then(() => {
      page?.close();
      return loadedId === null ? undefined : browser.call("Extensions.uninstall", { id: loadedId });
    });
    return unloadPromise;
  };
  try {
    const loaded = await control(["load-extension", "--path", extensionRoot]);
    if (!/^[a-p]{32}$/.test(loaded.id || "") || typeof loaded.path !== "string"
        || path.resolve(loaded.path) !== path.resolve(extensionRoot)) throw new Error("proof_extension_load_custody_invalid");
    // The correlated load result binds this id to our exact owned path. Take
    // cleanup custody before comparing the expected key-derived identity.
    loadedId = loaded.id;
    if (loadedId !== binding.extension_id) throw new Error("proof_extension_identity_mismatch");
    const url = `chrome-extension://${binding.extension_id}/proof.html`;
    const target = await control(["agent-window", "--url", url]);
    if (typeof target?.id === "string" && /^[A-F0-9]{32}$/i.test(target.id)) createdTargetId = target.id;
    if (createdTargetId !== null) cleanup = createOwnedTargetCleanup({ control, targetId: createdTargetId, afterClose: unload });
    assertAgentWindow(target, url);
    page = await connectPage(createdTargetId, 30_000);
    const installed = await verifyInstalled(page, extensionRoot, binding.extension_id, manifest, ["proof.html", "proof_transport.mjs"]);
    const transport = await evaluate(page, `(async () => {
      const { proveNativeTransport } = await import(chrome.runtime.getURL("proof_transport.mjs"));
      return proveNativeTransport(${JSON.stringify(transportInputs)});
    })()`, { retainUnknownException });
    return { ok: true, shared_chrome: { extension_loaded: true, target_closed: true, extension_unloaded: true },
      installed_extension: installed, proof_binding: binding, native_transport: transport };
  } finally {
    try { if (cleanup !== null) await cleanup.finish(); }
    finally {
      try { await unload(); }
      finally { browser.close(); }
    }
  }
}

export async function runDevLoopSharedChromeProof() {
  requireExpectedServiceContext();
  const diagnosticPath = process.env.POLYLOGUE_DEV_LOOP_DIAGNOSTIC_PATH;
  return runSharedChromeControlWorkflow({ extensionRoot: path.resolve(requiredEnvironment("POLYLOGUE_DEV_LOOP_EXTENSION_ROOT")),
    retainUnknownException: diagnosticPath ? details => retainNeutralEvaluationDiagnostic(diagnosticPath, details) : null,
    transportInputs: { receiverUrl: requiredEnvironment("POLYLOGUE_DEV_LOOP_RECEIVER_URL"),
      receiverId: requiredEnvironment("POLYLOGUE_DEV_LOOP_RECEIVER_ID"),
      attachmentUrl: requiredEnvironment("POLYLOGUE_DEV_LOOP_ATTACHMENT_URL"),
      attachmentSha256: requiredEnvironment("POLYLOGUE_DEV_LOOP_ATTACHMENT_SHA256") } });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runDevLoopSharedChromeProof()
    .then((result) => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch((error) => {
      process.stderr.write(`${error.stack || error.message || error}\n`);
      process.exitCode = 1;
    });
}
