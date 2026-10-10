// Shared-Chrome provider proof for the declared AgentCTL operation. It never
// launches a browser or reads a browser profile. Sinnix owns the one existing
// Chrome process and parks each proof window on its hidden agent workspace.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { firstControlJson } from "./shared_chrome_control.mjs";

const _CONTROL_COMMAND = "/home/sinity/.local/bin/sinnix-chrome-control";
const _DESKTOP_COMMAND = "/run/current-system/sw/bin/hyprctl";
const _CDP_PORT = 9222;
const _AGENT_WORKSPACE = "agentbrowser";
const _CONTROL_TIMEOUT_MS = 10_000;

const PROOF_PHASES = new Set([
  "service_context", "inputs", "chrome_status", "extension_load", "chrome_connect",
  "extension_startup", "popup_open", "popup_connect", "receiver_snapshot", "pause",
  "revision", "permission_grant", "desktop_snapshot", "popup_bind", "provider_preflight",
  "receiver_pairing", "provider_open", "provider_window", "provider_wait", "capture", "summary",
  "capture_start", "capture_membership", "capture_result",
]);
const ERROR_CATEGORIES = new Map([
  ["shared Chrome control command failed", "control_failed"],
  ["proof_window_precondition_failed", "window_precondition_failed"],
  ["proof_window_rules_failed", "window_rules_failed"],
  ["proof_window_creation_failed", "window_creation_failed"],
  ["proof_window_navigation_failed", "window_navigation_failed"],
  ["proof_window_compositor_changed", "window_compositor_changed"],

  ["proof_agent_workspace_visible", "window_visibility_refused"],
  ["proof_agent_window_unverified", "window_refused"],
  ["proof_desktop_unavailable", "control_failed"],
  ["proof_popup_binding_failed", "window_unknown"],
  ["shared Chrome control command timed out", "control_timeout"],
  ["shared Chrome proof window was not verified hidden on agentbrowser", "window_refused"],
  ["shared Chrome control returned no agent-window result", "window_unknown"],
  ["shared Chrome control returned an invalid proof target", "window_unknown"],
  ["proof_shutdown_requested", "shutdown"],
  ["proof_installed_revision_mismatch", "revision_mismatch"],
  ["proof_installed_resource_missing", "revision_missing"],
  ["proof_receiver_configuration_failed", "receiver_configuration_failed"],
  ["proof_receiver_permission_refused", "receiver_permission_refused"],
  ["proof_receiver_handshake_failed", "receiver_handshake_failed"],
  ["proof_receiver_configuration_changed", "configuration_changed"],
  ["proof_pause_failed", "pause_failed"],
  ["proof_pause_restore_failed", "pause_failed"],
  ["proof_host_permission_failed", "control_failed"],
  ["proof_capture_incomplete", "capture_incomplete"],
  ["proof_owned_provider_isolation_unavailable", "provider_isolation_refused"],
  ["proof_owned_tab_refused", "provider_isolation_refused"],
  ["proof_automatic_capture_missing", "automatic_capture_missing"],
  ["proof_automatic_capture_pending", "automatic_capture_pending"],
  ["proof_automatic_capture_start_failed", "automatic_capture_start_failed"],
  ["proof_capture_listener_invalid", "capture_listener_invalid"],
  ["proof_owned_provider_binding_invalid", "provider_isolation_refused"],
  ["loopback_endpoint_required", "loopback_endpoint_required"],
  ["receiver_identity_mismatch", "receiver_identity_mismatch"],
  ["receiver_authentication_failed", "receiver_authentication_failed"],
  ["receiver_unreachable", "receiver_unreachable"],
  ["receiver_observation_storage_failed", "receiver_observation_storage_failed"],
  ["receiver_transport_namespace_unavailable", "receiver_transport_namespace_unavailable"],
  ["native_request_invalid", "native_request_invalid"],
  ["native_input_incomplete", "native_input_incomplete"],
  ["native_operation_cancelled", "native_operation_cancelled"],
  ["native_request_body_invalid", "native_request_body_invalid"],
  ["native_route_forbidden", "native_route_forbidden"],
  ["native_request_header_forbidden", "native_request_header_forbidden"],
  ["native_messaging_unavailable", "native_messaging_unavailable"],
  ["native_transport_disconnected", "native_transport_disconnected"],
  ["native_response_frame_invalid", "native_response_frame_invalid"],
  ["native_response_cancelled", "native_response_cancelled"],
  ["proof_native_inputs_invalid", "proof_native_inputs_invalid"],
  ["proof_native_upload_mismatch", "proof_native_upload_mismatch"],
  ["proof_native_refusal_missing", "proof_native_refusal_missing"],
  ["proof_native_cancel_empty", "proof_native_cancel_empty"],
  ["proof_native_cancel_missing", "proof_native_cancel_missing"],
  ["proof_native_download_refused", "proof_native_download_refused"],
  ["proof_native_download_mismatch", "proof_native_download_mismatch"],
]);
let proofPhase = "service_context";
let shutdownPhase = null;
let nativeProgress = [];
let captureEvidence = [];
const CAPTURE_CATEGORIES = new Set(["native_capture_unavailable", "provider_throttle_authority_unavailable", "rate_limited", "capture_cancelled", "capture_rejected", "capture_timed_out"]);

const BRIDGE_FAILURE_CATEGORIES = new Set([...CAPTURE_CATEGORIES, "conversation_api_url_not_found", "asset_body_stream_unavailable", "receiver_unpaired", "capture_staging_unavailable", "capture_staging_sender_invalid", "capture_staging_request_invalid", "capture_staging_owner_mismatch", "capture_staging_invalid_ref", "capture_staging_sequence_mismatch", "capture_staging_records_unavailable", "native_acquisition_sequence_invalid", "capture_response_metadata_invalid", "capture_response_metadata_conflict", "capture_staging_incomplete", "capture_staging_interrupted", "capture_staging_missing_bytes"]);

function summaryChecks(provider, payload) {
  const result = payload?.result;
  const envelope = result?.envelope;
  const summary = envelope?.capture_summary;
  return {
    response_ok: result?.ok === true,
    identity_matches: envelope?.session?.provider_session_id === provider.nativeId && envelope?.session?.provider === provider.provider,
    native_digest_valid: typeof envelope?.receiver_native?.sha256 === "string" && /^[a-f0-9]{64}$/.test(envelope.receiver_native.sha256),
    native_full: summary?.captureMode === "native_full",
    turn_count_valid: Number.isInteger(summary?.turnCount) && summary.turnCount > 0,
    attachment_count_valid: Number.isInteger(summary?.attachmentCount) && summary.attachmentCount >= 0,
    artifact_present: Boolean(result?.captureResult?.artifact_ref),
    receiver_request_present: Boolean(result?.captureResult?.receiver_request_id),
  };
}

function captureCategory(value) {
  return CAPTURE_CATEGORIES.has(value) ? value : "unknown";
}

export function retainCaptureEvidence(provider, payload) {
  if (!["chatgpt", "claude-ai"].includes(provider.provider)) throw new Error("proof_capture_incomplete");
  const result = payload?.result;
  const object = result !== null && typeof result === "object" && !Array.isArray(result);
  const attempts = object && Array.isArray(result.native_attempts) ? result.native_attempts : [];
  const bridges = attempts.filter(row => row?.stage === "page_bridge_fetch");
  // Multiple bridge rows cannot be attributed to this returned acquisition.
  const bridge = bridges.length === 1 ? bridges[0] : null;
  const terminal = captureCategory(result?.error);
  const outcome = result?.outcome === "cancelled" ? "capture_cancelled" : captureCategory(result?.outcome);
  const category = result == null ? "response_empty" : !object || typeof result.ok !== "boolean" ? "response_invalid"
    : result.deferred === true ? "deferred" : result.ok === true ? "accepted" : captureCategory(result.error);
  const evidence = {
    provider: provider.provider, category: category === "unknown" && terminal === "unknown" ? outcome : category,
    bridge: {
      observed: Boolean(bridge),
      failure_stage: bridge?.accepted === false && bridge.error != null && ["admission", "provider_fetch", "staging"].includes(bridge.failure_stage) ? bridge.failure_stage : "unknown",
      accepted: typeof bridge?.accepted === "boolean" ? bridge.accepted : null,
      status: Number.isInteger(bridge?.status) && bridge.status >= 100 && bridge.status <= 599 ? bridge.status : null,
      category: bridge?.error == null ? "none" : BRIDGE_FAILURE_CATEGORIES.has(bridge.error) ? bridge.error : "unknown",
    },
    summary: summaryChecks(provider, payload),
  };
  captureEvidence = [...captureEvidence.filter(row => row.provider !== provider.provider), evidence];
}
const NATIVE_PROGRESS_STAGES = new Set(["throttle", "pending_header", "restore", "staging", "provider_auth", "provider_response", "body", "header", "canonical", "publication"]);
const BACKGROUND_PROGRESS_STAGES = new Set(["normalize_admission", "native_prepare", "native_assets", "native_finalize"]);
export function retainNativeProgress(progress) {
  if (!Array.isArray(progress) || progress.length > 6 || progress.some(entry =>
    !entry || typeof entry !== "object" || !["BEGIN", "END"].includes(entry.state) ||
    !((Object.keys(entry).length === 2 && NATIVE_PROGRESS_STAGES.has(entry.stage)) ||
      (Object.keys(entry).length === 3 && entry.source === "background_debug_log" && BACKGROUND_PROGRESS_STAGES.has(entry.stage))))) {
    throw new Error("proof_capture_incomplete");
  }
  nativeProgress = progress.map(entry => ({ stage: entry.stage, state: entry.state, ...(entry.source ? { source: entry.source } : {}) }));
}
let failurePublished = false;
const cleanupEvidence = { receiver: "not_required", permission: "not_required", mutations: "not_required", targets: "not_required" };

// Only these fixed categories cross the process boundary. Error text and stacks
// may contain a selected conversation, receiver credential, or captured content.
export function proofFailureReport(phase, error, cleanup = cleanupEvidence) {
  const states = new Set(["not_required", "settled", "failed", "unknown"]);
  return { ok: false, error: {
    phase: PROOF_PHASES.has(phase) ? phase : "unknown",
    category: ERROR_CATEGORIES.get((error instanceof AggregateError && error.cause ? error.cause : error)?.message) || (error instanceof AggregateError && !error.cause ? "cleanup_failed" : "operation_failed"),
  }, native_progress: nativeProgress.map(entry => ({ ...entry })), capture_evidence: captureEvidence.map(entry => ({ ...entry, bridge: { ...entry.bridge }, summary: { ...entry.summary } })), cleanup: Object.fromEntries(Object.keys(cleanupEvidence).map(key => [key, states.has(cleanup[key]) ? cleanup[key] : "unknown"])) };
}

export function currentProofFailure(error) {
  return proofFailureReport(shutdownPhase ?? proofPhase, error);
}

export function publishProofFailure(error) {
  if (failurePublished) return;
  failurePublished = true;
  globalThis.process.stdout.write(`${JSON.stringify(currentProofFailure(error))}\n`);
}

export async function inProofPhase(phase, operation) {
  requireProofRunning();
  proofPhase = PROOF_PHASES.has(phase) ? phase : "unknown";
  return await operation();
}

export function requireExpectedServiceContext() {
  // The runtime exports AGENTCTL_*; older hosts export the same values as SINNIXD_*.
  const prefix = globalThis.process.env.AGENTCTL_JOB_ID ? "AGENTCTL_" : "SINNIXD_";
  if (!globalThis.process.env[`${prefix}JOB_ID`]) {
    throw new Error("live provider proof requires a runtime job id");
  }
  if (globalThis.process.env[`${prefix}PROJECT_ID`] !== "polylogue" || globalThis.process.env[`${prefix}OPERATION`] !== "live_provider_proof") {
    throw new Error("live provider proof rejects execution outside its fixed service context");
  }
  // The runtime places the declared operation's job in the pool slice of its
  // declaration (.agentctl/project.toml: pool = "interactive"); a shell cannot.
  const cgroup = readFileSync("/proc/self/cgroup", "utf8").split("\n").find((line) => line.includes("::"))?.split("::", 2)[1] || "";
  const parts = cgroup.split("/");
  if (!["agentctl-interactive.slice", "sinnixd-pueue-interactive.slice"].some((slice) => parts.includes(slice))) {
    throw new Error("live provider proof is not inside the interactive pool");
  }
}

function sleep(ms) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
}

// Consume diagnostic bytes only to recognize the installed helper's fixed
// placement refusal. No original stderr line survives this classifier.
export function controlStderrClassifier() {
  const beforeId = "window ";
  const afterId = " opened but was not verified on agentbrowser; last compositor state: ";
  const visibility = ["true", "false", "unknown"].map(value => `; visible=${value}; stable_checks=`);
  let prefixOffset = 0;
  let prefixValid = true;
  let visibilityOffset = 0;
  let visibilityCandidates = visibility;
  let unavailableOffset = 0;
  let objectDepth = 0;
  let inString = false;
  let escaped = false;
  let bodyEnded = false;
  let found = false;
  // Match only fixed installed-helper line prefixes. Payload suffixes are
  // discarded byte by byte and never retained as diagnostic evidence.
  const fixedPrefixes = [
    ["agent-window requires flock", "proof_window_precondition_failed"],
    ["browser websocket unavailable: ", "proof_window_precondition_failed"],
    ["agent-window requires a live Hyprland compositor with a focused operator client; ", "proof_window_precondition_failed"],
    ["timed out waiting to create an agent browser window", "proof_window_precondition_failed"],
    ["failed to clear stale agent-window compositor rules", "proof_window_rules_failed"],
    ["failed to install temporary agent-window compositor rules", "proof_window_rules_failed"],
    ["failed to disable temporary agent-window compositor rules", "proof_window_rules_failed"],
    ["failed to create agent browser target (status ", "proof_window_creation_failed"],
    ["CDP created no target ID for agent window", "proof_window_creation_failed"],
    ["parked agent target disappeared before navigation", "proof_window_navigation_failed"],
    ["failed to navigate parked agent target ", "proof_window_navigation_failed"],
    ...["before CDP target creation", "after CDP target creation", "while identifying target", "while verifying target placement", "after navigating agent window", "after agent-window transaction"].map(phase => [`compositor state changed ${phase}: `, "proof_window_compositor_changed"]),
    ...["before CDP target creation", "after CDP target creation", "while identifying target", "while verifying target placement", "after navigating agent window", "after agent-window transaction"].map(phase => [`focused compositor client disappeared ${phase}: `, "proof_window_compositor_changed"]),
    ["focused operator client disappeared while clearing stale agent-window rules", "proof_window_compositor_changed"],
  ];
  let fixedCandidates = fixedPrefixes;
  let fixedOffset = 0;
  let fixedFailure = null;
  let refused = false;
  let visibleRefused = false;
  const prefixLength = beforeId.length + 32 + afterId.length;
  const finishLine = () => {
    fixedCandidates = fixedPrefixes; fixedOffset = 0;
    const matched = prefixValid && prefixOffset === prefixLength && found;
    refused ||= matched;
    visibleRefused ||= matched && visibilityCandidates[0] === visibility[0];
    prefixOffset = 0; prefixValid = true; visibilityOffset = 0; visibilityCandidates = visibility; unavailableOffset = 0; objectDepth = 0; inString = false; escaped = false; bodyEnded = false; found = false;
  };
  return {
    consume(bytes) {
      for (const byte of globalThis.Buffer.from(bytes)) {
        if (byte === 10) { finishLine(); continue; }
        fixedCandidates = fixedCandidates.filter(([prefix]) => byte === prefix.charCodeAt(fixedOffset));
        fixedOffset += 1;
        const fixed = fixedCandidates.find(([prefix]) => prefix.length === fixedOffset);
        if (fixed) fixedFailure ??= fixed[1];
        if (prefixValid && prefixOffset < prefixLength) {
          if (prefixOffset < beforeId.length) prefixValid = byte === beforeId.charCodeAt(prefixOffset);
          else if (prefixOffset < beforeId.length + 32) prefixValid = (byte >= 48 && byte <= 57) || (byte >= 65 && byte <= 70) || (byte >= 97 && byte <= 102);
          else prefixValid = byte === afterId.charCodeAt(prefixOffset - beforeId.length - 32);
          if (prefixValid) prefixOffset += 1;
          continue;
        }
        if (!prefixValid || prefixOffset < prefixLength) continue;
        if (!bodyEnded) {
          // The helper's JSON client state can contain the same text as the
          // later visibility field. Skip it structurally without storing it.
          if (objectDepth === 0 && (unavailableOffset > 0 || byte === 117)) {
            if (byte !== "unavailable".charCodeAt(unavailableOffset)) prefixValid = false;
            else if (++unavailableOffset === "unavailable".length) bodyEnded = true;
          } else if (inString) {
            if (escaped) escaped = false;
            else if (byte === 92) escaped = true;
            else if (byte === 34) inString = false;
          } else if (byte === 34 && objectDepth > 0) inString = true;
          else if (byte === 123) objectDepth += 1;
          else if (byte === 125 && objectDepth > 0) {
            objectDepth -= 1;
            if (objectDepth === 0) bodyEnded = true;
          } else if (objectDepth === 0 && byte !== 32) prefixValid = false;
          continue;
        }
        if (!found) {
          visibilityCandidates = visibilityCandidates.filter(value => byte === value.charCodeAt(visibilityOffset));
          if (visibilityCandidates.length === 0) { prefixValid = false; continue; }
          visibilityOffset += 1;
          if (visibilityOffset === visibilityCandidates[0].length) found = true;
        }
      }
    },
    failureCode() {
      finishLine();
      return visibleRefused ? "proof_agent_workspace_visible" : refused ? "proof_agent_window_unverified" : fixedFailure || "shared Chrome control command failed";
    },
  };
}

function runControlCommand(command, args, timeoutMs, spawnChild = spawn) {
  return new Promise((resolve, reject) => {
    const environment = { ...globalThis.process.env };
    if (command === _DESKTOP_COMMAND) delete environment.LD_LIBRARY_PATH;
    const child = spawnChild(command, args, { stdio: ["ignore", "pipe", "pipe"], env: environment });
    const diagnostic = controlStderrClassifier();
    let stdout = "";
    let settled = false;
    const finish = (callback) => (value) => {
      if (settled) return;
      settled = true;
      globalThis.clearTimeout(timeout);
      callback(value);
    };
    const timeout = globalThis.setTimeout(() => {
      child.kill("SIGTERM");
      finish(reject)(new Error("shared Chrome control command timed out"));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", chunk => diagnostic.consume(chunk));
    child.once("error", finish(reject));
    child.once("close", (code) => {
      if (code !== 0) finish(reject)(new Error(diagnostic.failureCode()));
      else if (command === _DESKTOP_COMMAND && args[0] === "eval") {
        if (stdout.trim() === "ok") finish(resolve)({ ok: true });
        else finish(reject)(new Error("proof_desktop_unavailable"));
      } else if (command === _DESKTOP_COMMAND) {
        try { finish(resolve)(JSON.parse(stdout)); }
        catch { finish(reject)(new Error("proof_desktop_unavailable")); }
      } else finish(resolve)(firstControlJson(stdout));
    });
  });
}

export function runChromeControl(args, timeoutMs = _CONTROL_TIMEOUT_MS, spawnChild = spawn) {
  return runControlCommand(_CONTROL_COMMAND, args, timeoutMs, spawnChild);
}

export function runProofDesktop(args, spawnChild = spawn) {
  return runControlCommand(_DESKTOP_COMMAND, args, _CONTROL_TIMEOUT_MS, spawnChild);
}

function checkedProofMonitors(monitors) {
  if (!Array.isArray(monitors) || monitors.length === 0 || monitors.some(m =>
    !Number.isSafeInteger(m.id) || m.id < 0 || !Number.isSafeInteger(m.activeWorkspace?.id) || typeof m.activeWorkspace?.name !== "string")) throw new Error("proof_desktop_unavailable");
  if (new Set(monitors.map(m => m.id)).size !== monitors.length) throw new Error("proof_desktop_unavailable");
}

export async function requireHiddenProofWorkspace(control = runProofDesktop) {
  const monitors = await control(["monitors", "-j"]);
  checkedProofMonitors(monitors);
  if (monitors.some(m => m.activeWorkspace.name === _AGENT_WORKSPACE)) throw new Error("proof_agent_workspace_visible");
}

export function assertAgentWindow(candidate, expectedUrl) {
  if (!candidate || typeof candidate !== "object") throw new Error("shared Chrome control returned no agent-window result");
  if (!/^[A-F0-9]{32}$/i.test(candidate.id || "")) throw new Error("shared Chrome control returned an invalid proof target");
  if (candidate.url !== expectedUrl || candidate.parked !== true || candidate.workspace !== _AGENT_WORKSPACE || candidate.show_with !== "F7") {
    throw new Error("shared Chrome proof window was not verified hidden on agentbrowser");
  }
  return candidate.id;
}

export async function openAgentWindow(url, timeoutMs, onCreated = () => {}, control = runChromeControl) {
  const response = await control(["agent-window", "--url", url], timeoutMs);
  try {
    if (response && typeof response.id === "string" && /^[A-F0-9]{32}$/i.test(response.id)) onCreated(response.id);
    return assertAgentWindow(response, url);
  } catch (error) {
    if (error instanceof SyntaxError) throw new Error("shared Chrome control returned invalid agent-window JSON");
    throw error;
  }
}

export function requireProofRunning() {
  if (shutdownRequested || targetSettlement) throw new Error("proof_shutdown_requested");
}

export function ownProofBrowser(client) {
  requireProofRunning();
  activeBrowserClient = client;
}

// Keep the original remote transaction through its response and registration.
// Shutdown cannot conclude target cleanup while this owned creation is pending.
export function openProofWindow(url, timeoutMs, control = runChromeControl) {
  requireProofRunning();
  cleanupEvidence.targets = "unknown";
  if (pendingWindowCreation) throw new Error("proof_window_creation_pending");
  const creation = openAgentWindow(url, timeoutMs, id => createdTargetIds.push(id), control);
  const tracked = creation.finally(() => {
    if (pendingWindowCreation === tracked) pendingWindowCreation = null;
  });
  pendingWindowCreation = tracked;
  return tracked;
}

export function closeOwnedProofWindows() {
  if (targetSettlement) return targetSettlement;
  targetSettlement = (async () => {
    const failures = [];
    const creation = pendingWindowCreation;
    if (creation || createdTargetIds.length) cleanupEvidence.targets = "unknown";
    if (creation) {
      try { await creation; } catch (error) { failures.push(error); }
    }
    try {
      if (activeBrowserClient) await closeProofTargets(activeBrowserClient, createdTargetIds);
    } catch (error) { failures.push(error); }
    if (failures.length) { cleanupEvidence.targets = "failed"; throw new AggregateError(failures, "proof_owned_window_cleanup_failed"); }
    if (creation || createdTargetIds.length) cleanupEvidence.targets = "settled";
  })();
  return targetSettlement;
}

export function connectCdp(webSocketDebuggerUrl) {
  const socket = new globalThis.WebSocket(webSocketDebuggerUrl);
  const pending = new Map();
  let sequence = 0;
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    const deferred = pending.get(message.id);
    if (!deferred) return;
    pending.delete(message.id);
    if (message.error) deferred.reject(new Error(JSON.stringify(message.error)));
    else deferred.resolve(message.result);
  };
  return new Promise((resolve, reject) => {
    socket.onerror = reject;
    socket.onopen = () => resolve({
      call(method, params = {}) {
        const id = ++sequence;
        socket.send(JSON.stringify({ id, method, params }));
        return new Promise((resolveCall, rejectCall) => pending.set(id, { resolve: resolveCall, reject: rejectCall }));
      },
      close() { socket.close(); },
    });
  });
}

async function waitJson(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "unavailable";
  while (Date.now() < deadline) {
    try {
      const response = await globalThis.fetch(url);
      if (response.ok) return await response.json();
      lastError = `${response.status}`;
    } catch (error) {
      lastError = String(error?.message || error);
    }
    await sleep(250);
  }
  throw new Error(`timed out waiting for shared Chrome CDP: ${lastError}`);
}

export async function evaluateJson(client, expression, { userGesture = false, retainUnknownException = null } = {}) {
  const result = await client.call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, userGesture });
  if (result.exceptionDetails) {
    // CDP's text is usually generic. Only an exact known first line of the
    // original exception may cross this boundary; descriptions contain stacks.
    const description = result.exceptionDetails.exception?.description;
    const firstLine = typeof description === "string" ? description.split("\n", 1)[0] : null;
    const known = [...ERROR_CATEGORIES.keys()].find(code => firstLine === `Error: ${code}`);
    if (!known && retainUnknownException !== null) await retainUnknownException(result.exceptionDetails);
    throw new Error(known || "proof_evaluation_failed");
  }
  return result.result?.value;
}

export async function pageClient(targetId, timeoutMs) {
  const targets = await waitJson(`http://127.0.0.1:${_CDP_PORT}/json/list`, timeoutMs);
  const target = targets.find(item => item.id === targetId && item.type === "page");
  if (!target) throw new Error("owned proof page disappeared");
  return connectCdp(target.webSocketDebuggerUrl);
}

export async function verifyInstalledExtension(client, extensionRoot, extensionId, manifest, extraFiles = []) {
  const files = ["manifest.json", ...extraFiles, ...readdirSync(path.join(extensionRoot, "src"), { recursive: true, withFileTypes: true })
    .filter(entry => entry.isFile()).map(entry => path.relative(extensionRoot, path.join(entry.parentPath, entry.name)))].sort();
  const expected = files.map(file => [file, createHash("sha256").update(readFileSync(path.join(extensionRoot, file))).digest("hex")]);
  const observed = await evaluateJson(client, `(async () => {
    const rows = [];
    for (const file of ${JSON.stringify(files)}) {
      const response = await fetch(chrome.runtime.getURL(file));
      if (!response.ok) throw new Error("proof_installed_resource_missing");
      const digest = await crypto.subtle.digest("SHA-256", await response.arrayBuffer());
      rows.push([file, Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("")]);
    }
    return { id: chrome.runtime.id, version: chrome.runtime.getManifest().version, rows };
  })()`);
  if (observed.id !== extensionId || observed.version !== manifest.version || JSON.stringify(observed.rows) !== JSON.stringify(expected)) throw new Error("proof_installed_revision_mismatch");
  return { id: extensionId, version: observed.version, bundle_sha256: sha256(JSON.stringify(observed.rows)), verified_file_count: files.length };
}

export async function captureProvider(workerClient, provider, tabId, retainFailure = null) {
  const captured = await evaluateJson(workerClient, `(async () => ({
    result: await globalThis.__polylogueOwnedProviderProof.consumeCapture(${JSON.stringify(tabId)}, ${JSON.stringify(provider.nativeId)})
  }))()`);
  if (captured?.result?.ok !== true && retainFailure !== null) await retainFailure({
    provider: provider.provider, returned_ok: captured?.result?.ok ?? null,
    returned_error: captured?.result?.error ?? null, returned_outcome: captured?.result?.outcome ?? null,
  });
  retainCaptureEvidence(provider, captured);
  if (provider.provider === "chatgpt") retainNativeProgress(captured?.result?.native_progress ?? []);
  return captured;
}

function sha256(value) {
  return createHash("sha256").update(String(value || "")).digest("hex");
}


export function providerSummary(provider, payload) {
  const envelope = payload?.result?.envelope || {};
  const session = envelope.session || {};
  const summary = envelope.capture_summary || {};
  const capture = payload?.result?.captureResult || {};
  const native = envelope.receiver_native;
  const turnCount = summary.turnCount;
  const attachmentCount = summary.attachmentCount;
  return {
    ok: Object.values(summaryChecks(provider, payload)).every(Boolean),
    host: provider.host,
    source_url_sha256: sha256(envelope.provenance?.source_url),
    provider: session.provider || null,
    provider_session_id_sha256: sha256(session.provider_session_id),
    turn_count: turnCount, attachment_count: attachmentCount,
    artifact_ref: capture.artifact_ref || null,
    artifact_sha256: native?.sha256 || null,
    receiver_request_id: capture.receiver_request_id || null,
  };
}

export async function closeProofTargets(browserClient, targetIds) {
  const results = await Promise.allSettled(targetIds.map((targetId) => browserClient.call("Target.closeTarget", { targetId })));
  const failures = results.flatMap((result, index) => result.status === "rejected"
    ? [`${targetIds[index]}: ${result.reason?.message || result.reason}`]
    : result.value?.success === false ? [`${targetIds[index]}: closeTarget returned success=false`] : []);
  if (failures.length) throw new Error(`failed to close proof targets: ${failures.join("; ")}`);
}

let activeBrowserClient = null;
let createdTargetIds = [];
let shutdownRequested = false;
let pendingWindowCreation = null;
let targetSettlement = null;

export function installShutdownCleanup({ afterTargets = null } = {}) {
  for (const signal of ["SIGINT", "SIGTERM"]) {
    globalThis.process.once(signal, () => {
      if (shutdownRequested) return;
      // Preserve the interrupted main phase before receiver/target cleanup can
      // resolve an in-flight capture. Cleanup has its separate custody states.
      shutdownPhase = proofPhase;
      shutdownRequested = true;
      settleProofCleanup(afterTargets, null)
        .catch(() => globalThis.process.stderr.write("proof_signal_owned_cleanup_failed\n"))
        .finally(() => {
          publishProofFailure(new Error("proof_shutdown_requested"));
          globalThis.process.exit(signal === "SIGINT" ? 130 : 143);
        });
    });
  }
}

export async function settleProofCleanup(unloadExtension, failure) {
  const cleanupErrors = [];
  try { await closeOwnedProofWindows(); } catch (error) { cleanupErrors.push(error); }
  if (unloadExtension !== null) {
    cleanupEvidence.mutations = "unknown";
    try { await unloadExtension(); cleanupEvidence.mutations = "settled"; }
    catch (error) { cleanupEvidence.mutations = "failed"; cleanupErrors.push(error); }
  }
  if (cleanupErrors.length) throw new AggregateError([...(failure ? [failure] : []), ...cleanupErrors], "live proof cleanup failed", { cause: failure });
}
