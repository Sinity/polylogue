// Shared-Chrome provider proof for the declared AgentCTL operation. It never
// launches a browser or reads a browser profile. Sinnix owns the one existing
// Chrome process and parks each proof window on its hidden agent workspace.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { firstControlJson } from "./shared_chrome_control.mjs";

const PROVIDERS = {
  chatgpt: { host: "chatgpt.com", provider: "chatgpt" },
  claude: { host: "claude.ai", provider: "claude-ai" },
};
const _CONTROL_COMMAND = "/home/sinity/.local/bin/sinnix-chrome-control";
const _DESKTOP_COMMAND = "/run/current-system/sw/bin/hyprctl";
const _CDP_PORT = 9222;
const _AGENT_WORKSPACE = "agentbrowser";
const _WORKFLOW_TIMEOUT_MS = 90_000;
const _STARTUP_TIMEOUT_MS = 30_000;
const _INTERACTIVE_WAIT_MS = 15_000;
const _CONTROL_TIMEOUT_MS = 10_000;

const PROOF_PHASES = new Set([
  "service_context", "inputs", "chrome_status", "extension_load", "chrome_connect",
  "extension_startup", "popup_open", "popup_connect", "receiver_snapshot", "pause",
  "revision", "permission_grant", "desktop_snapshot", "popup_bind", "provider_preflight",
  "receiver_pairing", "provider_open", "provider_window", "provider_wait", "capture", "summary",
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

function retainCaptureEvidence(provider, payload) {
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

function requiredEnvironment(name) {
  const value = globalThis.process.env[name];
  if (!value) throw new Error(`${name} must be supplied by the declared live-provider service`);
  return value;
}

function receiverPortFromEnvironment(name) {
  const port = Number(requiredEnvironment(name));
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${name} is not a loopback port number`);
  }
  return port;
}

function requireExpectedServiceContext() {
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

function fixedInputs() {
  const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
  const receiverPort = receiverPortFromEnvironment("POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT");
  return {
    extensionRoot: path.resolve(scriptDirectory, ".."),
    receiverBaseUrl: `http://127.0.0.1:${receiverPort}`,
    conversations: JSON.parse(requiredEnvironment("POLYLOGUE_LIVE_PROVIDER_CONVERSATIONS")),
    timeoutMs: _WORKFLOW_TIMEOUT_MS,
    startupTimeoutMs: _STARTUP_TIMEOUT_MS,
    interactiveWaitMs: _INTERACTIVE_WAIT_MS,
  };
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

function checkedDesktopWindow(value) {
  if (!value || !/^0x[0-9a-f]+$/i.test(value.address || "") ||
      !Number.isSafeInteger(value.stable_id) || value.stable_id < 0 ||
      !Number.isSafeInteger(value.workspace_id) || !Number.isSafeInteger(value.monitor_id) || value.monitor_id < 0) throw new Error("proof_desktop_unavailable");
  return value;
}

function desktopWindow(value) {
  const stable = value?.stableId;
  // Hyprland JSON writes stableId as unprefixed hexadecimal, even when all
  // digits happen to be decimal. Lua exposes the same identity as a number.
  if (typeof stable !== "number" && !(typeof stable === "string" && /^[0-9a-f]+$/i.test(stable))) throw new Error("proof_desktop_unavailable");
  const identity = typeof stable === "string" ? Number.parseInt(stable, 16) : stable;
  return checkedDesktopWindow({ address: value?.address, stable_id: identity, workspace_id: value?.workspace?.id, monitor_id: value?.monitor });
}

export async function captureProofOperator(control = runProofDesktop) {
  const value = await control(["activewindow", "-j"]);
  if (value?.workspace?.name === _AGENT_WORKSPACE) throw new Error("proof_agent_workspace_visible");
  const original = desktopWindow(value);
  const monitors = await control(["monitors", "-j"]);
  checkedProofMonitors(monitors);
  if (monitors.some(m => m.activeWorkspace.name === _AGENT_WORKSPACE)) throw new Error("proof_agent_workspace_visible");
  original.monitor_workspaces = monitors.map(m => ({ monitor_id: m.id, workspace_id: m.activeWorkspace.id }));
  return original;
}

export async function bindProofPopup(client, targetId, remaining, control = runProofDesktop, wait = sleep) {
  if (!/^[A-F0-9]{32}$/i.test(targetId)) throw new Error("proof_popup_binding_failed");
  // The original target id identifies only this disposable popup. Bind it to
  // its compositor window before a native permission prompt can take focus.
  const title = `Polylogue proof ${targetId}`;
  await evaluateJson(client, `(() => { document.title = ${JSON.stringify(title)}; return true; })()`);
  while (true) {
    requireProofRunning();
    remaining();
    const rows = await control(["clients", "-j"]);
    if (!Array.isArray(rows)) throw new Error("proof_popup_binding_failed");
    const matches = rows.filter(w => w.class === "google-chrome" && w.title === `${title} - Google Chrome` && w.workspace?.name === _AGENT_WORKSPACE);
    if (matches.length === 0) {
      await wait(Math.min(250, remaining()));
      continue;
    }
    if (matches.length !== 1) throw new Error("proof_popup_binding_failed");
    try { return desktopWindow(matches[0]); }
    catch { throw new Error("proof_popup_binding_failed"); }
  }
}

export async function restoreProofOperator(original, popup, control = runProofDesktop) {
  checkedDesktopWindow(original); checkedDesktopWindow(popup);
  const before = original.monitor_workspaces?.find(m => m.monitor_id === popup.monitor_id);
  if (!before || !Number.isSafeInteger(before.workspace_id)) throw new Error("proof_desktop_unavailable");
  // This check and focus dispatch execute in one compositor transaction. A
  // newer independent focus is preserved, including another Chrome window.
  const response = await control(["eval", `local current = hl.get_active_window(); local original = hl.get_window("address:${original.address}"); if current and current.address == "${popup.address}" and current.stable_id == ${popup.stable_id} and current.workspace and current.workspace.name == "agentbrowser" and current.monitor and current.monitor.id == ${popup.monitor_id} then local workspace = hl.get_workspace(${before.workspace_id}); if original and original.stable_id == ${original.stable_id} and original.workspace and original.workspace.id == ${original.workspace_id} and workspace and workspace.monitor and workspace.monitor.id == ${popup.monitor_id} then hl.dispatch(hl.dsp.focus({workspace = workspace})); hl.dispatch(hl.dsp.focus({window = original})) else error("proof_desktop_unavailable") end end`]);
  if (response?.ok !== true) throw new Error("proof_desktop_unavailable");
  const current = desktopWindow(await control(["activewindow", "-j"]));
  return current.address === original.address && current.stable_id === original.stable_id && current.workspace_id === original.workspace_id;
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

export async function evaluateJson(client, expression, { userGesture = false } = {}) {
  const result = await client.call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, userGesture });
  if (result.exceptionDetails) {
    // CDP's text is usually generic. Only an exact known first line of the
    // original exception may cross this boundary; descriptions contain stacks.
    const description = result.exceptionDetails.exception?.description;
    const firstLine = typeof description === "string" ? description.split("\n", 1)[0] : null;
    const known = [...ERROR_CATEGORIES.keys()].find(code => firstLine === `Error: ${code}`);
    throw new Error(known || "proof_evaluation_failed");
  }
  return result.result?.value;
}

// Chrome derives an unpacked extension's id from its absolute path (or from the
// manifest `key` when one is declared): the first 16 bytes of the SHA-256,
// hex digits mapped onto a-p. Matching on that id, not on the manifest name,
// binds the proof to the extension just loaded rather than to a same-named
// extension loaded earlier from another checkout.
function unpackedExtensionId(extensionRoot, manifest) {
  const material = manifest.key ? globalThis.Buffer.from(manifest.key, "base64") : globalThis.Buffer.from(path.resolve(extensionRoot), "utf8");
  const digest = createHash("sha256").update(material).digest("hex").slice(0, 32);
  return [...digest].map((digit) => String.fromCharCode("a".charCodeAt(0) + Number.parseInt(digit, 16))).join("");
}

async function waitForExtensionWorker(extensionId, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const prefix = `chrome-extension://${extensionId}/`;
  while (Date.now() < deadline) {
    const targets = await waitJson(`http://127.0.0.1:${_CDP_PORT}/json/list`, Math.min(timeoutMs, 2000));
    const target = targets.find((item) => item.type === "service_worker" && item.url?.startsWith(prefix));
    if (target) return connectCdp(target.webSocketDebuggerUrl);
    await sleep(250);
  }
  throw new Error(`the extension just loaded (${extensionId}) has no service worker in shared Chrome`);
}

export async function receiverConfiguration(client) {
  return evaluateJson(client, "chrome.storage.local.get(['receiverBaseUrl', 'polylogueReceiverPairing', 'polylogueAmbientSettings'])");
}

export async function configureReceiver(client, receiverBaseUrl) {
  const outcome = await evaluateJson(client, `(async () => {
    const pause = await chrome.runtime.sendMessage({ type: "polylogue.ambient.configure", automatic_capture_enabled: false });
    if (!pause?.ok) throw new Error("proof_pause_failed");
    const configured = await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", receiverBaseUrl: ${JSON.stringify(receiverBaseUrl)} });
    if (!configured?.ok || !Number.isSafeInteger(configured.configurationRevision)) throw new Error(configured?.error === "receiver_origin_not_permitted" ? "proof_receiver_permission_refused" : "proof_receiver_configuration_failed");
    let revision = configured.configurationRevision;
    try {
      const handshake = await chrome.runtime.sendMessage({ type: "polylogue.receiverPairing.reset", expectedConfigurationRevision: revision });
      if (Number.isSafeInteger(handshake?.configurationRevision)) revision = handshake.configurationRevision;
      if (!handshake?.ok || handshake?.health?.status !== "ok" || !handshake?.pairing?.receiver_id || handshake.pairing.api_schema !== "polylogue-browser-capture/v1") return { ok: false, revision };
      return { ok: true, revision, receiver_id: handshake.pairing.receiver_id, api_schema: handshake.pairing.api_schema };
    } catch {
      return { ok: false, revision };
    }
  })()`);
  if (!outcome?.ok) {
    const error = new Error("proof_receiver_handshake_failed");
    error.receiverConfigurationRevision = Number.isSafeInteger(outcome?.revision) ? outcome.revision : null;
    throw error;
  }
  return outcome;
}

export async function restoreReceiverConfiguration(client, previous, owned) {
  return evaluateJson(client, `(async () => {
    const restored = await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", restore: {
      previous: ${JSON.stringify(previous)}, owned: ${JSON.stringify(owned)}
    } });
    if (!restored?.ok) throw new Error("proof_receiver_configuration_changed");
    // Incident policy remains paused. Existing queues and capture custody are
    // untouched; this configuration restore grants no automatic resumption.
    const paused = await chrome.runtime.sendMessage({ type: "polylogue.ambient.configure", automatic_capture_enabled: false });
    if (!paused?.ok) throw new Error("proof_pause_restore_failed");
    return true;
  })()`);
}

export async function pageClient(targetId, timeoutMs) {
  const targets = await waitJson(`http://127.0.0.1:${_CDP_PORT}/json/list`, timeoutMs);
  const target = targets.find(item => item.id === targetId && item.type === "page");
  if (!target) throw new Error("owned proof page disappeared");
  return connectCdp(target.webSocketDebuggerUrl);
}

// One owner is registered before the first receiver or permission mutation.
// Signal and normal cleanup borrow the same settlement promise; they cannot
// remove a permission until its original grant has finished.
export function proofReceiverCustody(client, previous, owned, origin) {
  const owner = { client, previous, owned: { ...owned, revision: null }, origin,
    permissionAdded: false, permissionGrant: null, configuration: null,
    cleaning: false, settlement: null,
    cleanup: { receiver: "unknown", permission: "not_required", mutations: "not_required" } };
  Object.assign(cleanupEvidence, owner.cleanup);
  pendingReceiverRestore = () => cleanupProofReceiver(owner);
  return owner;
}

export function requestProofHostPermission(owner, afterSettlement = async () => {}) {
  if (owner.cleaning) throw new Error("proof_shutdown_requested");
  owner.cleanup.permission = cleanupEvidence.permission = "unknown";
  owner.permissionGrant = (async () => {
    let primary;
    try {
      const permission = `{ origins: [${JSON.stringify(owner.origin)}] }`;
      const existing = await evaluateJson(owner.client, `chrome.permissions.contains(${permission})`);
      if (existing === true) {
        owner.cleanup.permission = cleanupEvidence.permission = "not_required";
      } else {
        if (existing !== false) throw new Error("proof_evaluation_failed");
        if (owner.cleaning) throw new Error("proof_shutdown_requested");
        // Request declared optional access in the owned popup with Chrome's user
        // gesture contract, matching its Save action. Settings-site grants do not
        // activate optional permissions. Keep this promise owned through settlement.
        const granted = await evaluateJson(owner.client, `chrome.permissions.request(${permission})`, { userGesture: true });
        if (granted === false) {
          owner.cleanup.permission = cleanupEvidence.permission = "not_required";
          throw new Error("proof_receiver_permission_refused");
        }
        if (granted !== true) throw new Error("proof_evaluation_failed");
        owner.permissionAdded = true;
        const active = await evaluateJson(owner.client, `chrome.permissions.contains(${permission})`);
        if (active === false) throw new Error("proof_receiver_permission_refused");
        if (active !== true) throw new Error("proof_evaluation_failed");
      }
    } catch (error) { primary = error; }
    // Settle the original prompt guard after either grant result. Preserve both
    // errors without replacing the primary permission/setup cause.
    try { await afterSettlement(); }
    catch (error) {
      if (primary) throw new AggregateError([primary, error], "proof_prompt_restore_failed", { cause: primary });
      throw error;
    }
    if (primary) throw primary;
  })();
  return owner.permissionGrant;
}

export function configureProofReceiver(owner) {
  if (owner.cleaning) throw new Error("proof_shutdown_requested");
  owner.configuration = configureReceiver(owner.client, owner.owned.baseUrl)
    .then(handshake => { owner.owned.receiverId = handshake.receiver_id; owner.owned.revision = handshake.revision; return handshake; }, error => {
      if (Number.isSafeInteger(error.receiverConfigurationRevision)) owner.owned.revision = error.receiverConfigurationRevision;
      throw error;
    });
  return owner.configuration;
}

export function cleanupProofReceiver(owner) {
  if (owner.settlement) return owner.settlement;
  owner.cleaning = true;
  owner.settlement = (async () => {
    const failures = [];
    const mutations = [owner.permissionGrant, owner.configuration].filter(Boolean);
    owner.cleanup.mutations = mutations.length ? "settled" : "not_required";
    for (const mutation of mutations) {
      if (mutation) {
        try { await mutation; } catch (error) { owner.cleanup.mutations = "failed"; failures.push(error); }
      }
    }
    try { await restoreReceiverConfiguration(owner.client, owner.previous, owner.owned); owner.cleanup.receiver = "settled"; }
    catch (error) { owner.cleanup.receiver = "failed"; failures.push(error); }
    try {
      if (owner.permissionAdded) {
        const removed = await evaluateJson(owner.client, `chrome.permissions.remove({ origins: [${JSON.stringify(owner.origin)}] })`);
        if (removed !== true) throw new Error("proof_host_permission_failed");
        owner.permissionAdded = false;
        owner.cleanup.permission = "settled";
      }
    } catch (error) { owner.cleanup.permission = "failed"; failures.push(error); }
    Object.assign(cleanupEvidence, owner.cleanup);
    if (failures.length) throw new AggregateError(failures, "proof_receiver_cleanup_failed");
  })();
  return owner.settlement;
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

async function proofWindowId(browserClient, targetId) {
  const result = await browserClient.call("Browser.getWindowForTarget", { targetId });
  if (!Number.isInteger(result.windowId)) throw new Error("shared Chrome proof target has no browser window");
  return result.windowId;
}

export async function captureProvider(workerClient, provider, windowId, timeoutMs) {
  const captured = await evaluateJson(workerClient, `(async () => {
    const deadline = Date.now() + ${JSON.stringify(timeoutMs)};
    while (Date.now() < deadline) {
      const tabs = await chrome.tabs.query({ windowId: ${JSON.stringify(windowId)} });
      if (tabs.length === 1) {
        const tab = tabs[0];
        try {
          if (tab.url === ${JSON.stringify(provider.url)} && tab.pinned !== true) {
            const result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage", providerSessionId: ${JSON.stringify(provider.nativeId)} });
            return { result };
          }
        } catch { /* The content script is still loading. */ }
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return { result: { ok: false, error: "capture_timed_out" } };
  })()`);
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
// Set while the operator's receiver settings are replaced by the proof's, so a
// signal restores them before exiting instead of leaving the shared extension
// pointed at a receiver that is about to shut down.
let pendingReceiverRestore = null;

export function installShutdownCleanup() {
  for (const signal of ["SIGINT", "SIGTERM"]) {
    globalThis.process.once(signal, () => {
      if (shutdownRequested) return;
      // Preserve the interrupted main phase before receiver/target cleanup can
      // resolve an in-flight capture. Cleanup has its separate custody states.
      shutdownPhase = proofPhase;
      shutdownRequested = true;
      Promise.resolve().then(() => pendingReceiverRestore && pendingReceiverRestore())
        .catch(() => globalThis.process.stderr.write("proof_signal_receiver_cleanup_failed\n"))
        .then(() => closeOwnedProofWindows())
        .catch(() => globalThis.process.stderr.write("proof_signal_target_cleanup_failed\n"))
        .finally(() => {
          publishProofFailure(new Error("proof_shutdown_requested"));
          globalThis.process.exit(signal === "SIGINT" ? 130 : 143);
        });
    });
  }
}

export async function settleProofCleanup(receiverOwner, failure) {
  const cleanupErrors = [];
  try { if (receiverOwner) await cleanupProofReceiver(receiverOwner); }
  catch (error) { cleanupErrors.push(error); }
  pendingReceiverRestore = null;
  try { await closeOwnedProofWindows(); }
  catch (error) { cleanupErrors.push(error); }
  if (cleanupErrors.length) throw new AggregateError([...(failure ? [failure] : []), ...cleanupErrors], "live proof cleanup failed", { cause: failure });
}

async function runLiveProviderProof() {
  // Pausing background capture does not stop static MAIN-world interception
  // or tab-status readers. No provider artifact may load until registration
  // is restricted to proof-owned tabs before those effects can begin.
  await inProofPhase("provider_preflight", () => { throw new Error("proof_owned_provider_isolation_unavailable"); });
  await inProofPhase("service_context", () => requireExpectedServiceContext());
  installShutdownCleanup();
  const { extensionRoot, receiverBaseUrl, conversations, timeoutMs, startupTimeoutMs, interactiveWaitMs } = await inProofPhase("inputs", () => fixedInputs());
  const deadline = Date.now() + timeoutMs;
  const remaining = (phase) => {
    requireProofRunning();
    const budget = deadline - Date.now();
    if (budget <= 0) throw new Error(`live provider proof timed out during ${phase}`);
    return budget;
  };
  const selected = conversations.map(({ name, url, nativeId }) => {
    const provider = PROVIDERS[name];
    if (!provider || new globalThis.URL(url).hostname !== provider.host || !nativeId) throw new Error("invalid exact proof conversation");
    return { ...provider, url, nativeId };
  });
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  let workerClient;
  let previousReceiverConfiguration;
  let popupClient;
  let receiverOwner;
  let failure;
  let result;
  const origin = `${receiverBaseUrl}/*`;
  try {
    await inProofPhase("chrome_status", () => runChromeControl(["status"], Math.min(_CONTROL_TIMEOUT_MS, remaining("shared Chrome status"))));
    await inProofPhase("extension_load", () => runChromeControl(["load-extension", "--path", extensionRoot], Math.min(_CONTROL_TIMEOUT_MS, remaining("extension load"))));
    const version = await inProofPhase("chrome_connect", () => waitJson(`http://127.0.0.1:${_CDP_PORT}/json/version`, Math.min(startupTimeoutMs, remaining("shared Chrome CDP"))));
    ownProofBrowser(await inProofPhase("chrome_connect", () => connectCdp(version.webSocketDebuggerUrl)));
    workerClient = await inProofPhase("extension_startup", () => waitForExtensionWorker(unpackedExtensionId(extensionRoot, manifest), Math.min(startupTimeoutMs, remaining("extension startup"))));
    const originalOperator = await inProofPhase("desktop_snapshot", () => captureProofOperator());
    const popupTarget = await inProofPhase("popup_open", () => openProofWindow(`chrome-extension://${unpackedExtensionId(extensionRoot, manifest)}/src/popup.html`, remaining("proof popup")));
    popupClient = await inProofPhase("popup_connect", () => pageClient(popupTarget, remaining("proof popup")));
    const ownedPopup = await inProofPhase("popup_bind", () => bindProofPopup(popupClient, popupTarget, () => remaining("popup binding")));
    previousReceiverConfiguration = await inProofPhase("receiver_snapshot", () => receiverConfiguration(popupClient));
    receiverOwner = proofReceiverCustody(popupClient, previousReceiverConfiguration,
      { baseUrl: receiverBaseUrl, receiverId: null },
      origin);
    if (shutdownRequested) throw new Error("proof_shutdown_requested");
    const paused = await inProofPhase("pause", () => evaluateJson(popupClient, 'chrome.runtime.sendMessage({ type: "polylogue.ambient.configure", automatic_capture_enabled: false })'));
    if (!paused?.ok) throw new Error("proof_pause_failed");
    const installedExtension = await inProofPhase("revision", () => verifyInstalledExtension(popupClient, extensionRoot, unpackedExtensionId(extensionRoot, manifest), manifest));
    await inProofPhase("permission_grant", () => requestProofHostPermission(receiverOwner, () => restoreProofOperator(originalOperator, ownedPopup)));
    if (shutdownRequested) throw new Error("proof_shutdown_requested");
    await inProofPhase("receiver_pairing", () => configureProofReceiver(receiverOwner));
    if (shutdownRequested) throw new Error("proof_shutdown_requested");
    const proofTargets = [];
    for (const provider of selected) {
      await inProofPhase("provider_preflight", () => requireHiddenProofWorkspace());
      const targetId = await inProofPhase("provider_open", () => openProofWindow(provider.url, Math.min(_CONTROL_TIMEOUT_MS, remaining(`open ${provider.host}`))));
      proofTargets.push({ provider, windowId: await inProofPhase("provider_window", () => proofWindowId(activeBrowserClient, targetId)) });
    }
    if (interactiveWaitMs > 0) await inProofPhase("provider_wait", () => sleep(Math.min(interactiveWaitMs, remaining("interactive wait"))));
    const captures = await inProofPhase("capture", () => Promise.all(proofTargets.map(async ({ provider, windowId }) =>
      [provider, await captureProvider(popupClient, provider, windowId, remaining("provider capture"))])));
    const summary = await inProofPhase("summary", () => Object.fromEntries(captures.map(([provider, captured]) => [provider.host, providerSummary(provider, captured)])));
    if (!Object.values(summary).every(item => item.ok === true)) throw new Error("proof_capture_incomplete");
    result = { extension: installedExtension, ok: Object.values(summary).every((item) => item.ok === true), providers: summary, automatic_capture_enabled: false, privacy_posture: "shared-Chrome output hashes session ids and omits transcript text" };
  } catch (error) {
    failure = error;
  } finally {
    try { await settleProofCleanup(receiverOwner, failure); }
    finally {
      // A signal's handler owns target closure and process exit. Keep its
      // clients alive until that same receiver settlement has completed.
      if (!shutdownRequested) {
        if (workerClient) workerClient.close();
        if (popupClient) popupClient.close();
        if (activeBrowserClient) activeBrowserClient.close();
        activeBrowserClient = null;
        createdTargetIds = [];
      }
    }
  }
  if (failure) throw failure;
  return result;
}

if (globalThis.process.argv[1] && import.meta.url === pathToFileURL(globalThis.process.argv[1]).href) {
  runLiveProviderProof()
    .then((result) => { if (!shutdownRequested) globalThis.process.stdout.write(`${JSON.stringify(result)}\n`); })
    .catch((error) => {
      publishProofFailure(error);
      globalThis.process.exitCode = 1;
    });
}
