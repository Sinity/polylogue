// Shared-Chrome provider proof for the declared AgentCTL operation. It never
// launches a browser or reads a browser profile. Sinnix owns the one existing
// Chrome process and parks each proof window on its hidden agent workspace.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { firstControlJson } from "./shared_chrome_control.mjs";

const PROVIDERS = {
  chatgpt: { host: "chatgpt.com", url: "https://chatgpt.com/", provider: "chatgpt", adapters: ["chatgpt-native-v1", "chatgpt-dom-v1"] },
  claude: { host: "claude.ai", url: "https://claude.ai/", provider: "claude-ai", adapters: ["claude-ai-native-v1", "claude-ai-dom-v1"] },
};
const _CONTROL_COMMAND = "/home/sinity/.local/bin/sinnix-chrome-control";
const _CDP_PORT = 9222;
const _AGENT_WORKSPACE = "agentbrowser";
const _WORKFLOW_TIMEOUT_MS = 90_000;
const _STARTUP_TIMEOUT_MS = 30_000;
const _INTERACTIVE_WAIT_MS = 15_000;
const _CONTROL_TIMEOUT_MS = 10_000;

function requiredEnvironment(name) {
  const value = process.env[name];
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
  const prefix = process.env.AGENTCTL_JOB_ID ? "AGENTCTL_" : "SINNIXD_";
  if (!process.env[`${prefix}JOB_ID`]) {
    throw new Error("live provider proof requires a runtime job id");
  }
  if (process.env[`${prefix}PROJECT_ID`] !== "polylogue" || process.env[`${prefix}OPERATION`] !== "live_provider_proof") {
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
    receiverToken: requiredEnvironment("POLYLOGUE_LIVE_PROVIDER_RECEIVER_TOKEN"),
    providers: ["chatgpt", "claude"],
    timeoutMs: _WORKFLOW_TIMEOUT_MS,
    startupTimeoutMs: _STARTUP_TIMEOUT_MS,
    interactiveWaitMs: _INTERACTIVE_WAIT_MS,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runChromeControl(args, timeoutMs = _CONTROL_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const child = spawn(_CONTROL_COMMAND, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let settled = false;
    const finish = (callback) => (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      callback(value);
    };
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      finish(reject)(new Error("shared Chrome control command timed out"));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.once("error", finish(reject));
    child.once("close", (code) => {
      if (code !== 0) finish(reject)(new Error("shared Chrome control command failed"));
      else finish(resolve)(firstControlJson(stdout));
    });
  });
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

function connectCdp(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
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
      const response = await fetch(url);
      if (response.ok) return await response.json();
      lastError = `${response.status}`;
    } catch (error) {
      lastError = String(error?.message || error);
    }
    await sleep(250);
  }
  throw new Error(`timed out waiting for shared Chrome CDP: ${lastError}`);
}

async function evaluateJson(client, expression) {
  const result = await client.call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "CDP evaluation failed");
  return result.result?.value;
}

// Chrome derives an unpacked extension's id from its absolute path (or from the
// manifest `key` when one is declared): the first 16 bytes of the SHA-256,
// hex digits mapped onto a-p. Matching on that id, not on the manifest name,
// binds the proof to the extension just loaded rather than to a same-named
// extension loaded earlier from another checkout.
function unpackedExtensionId(extensionRoot, manifest) {
  const material = manifest.key ? Buffer.from(manifest.key, "base64") : Buffer.from(path.resolve(extensionRoot), "utf8");
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

export async function receiverConfiguration(workerClient) {
  return evaluateJson(workerClient, "chrome.storage.local.get(['receiverBaseUrl', 'receiverAuthToken', 'polylogueReceiverPairing'])");
}

export async function configureReceiver(workerClient, receiverBaseUrl, receiverToken) {
  return evaluateJson(workerClient, `(async () => {
    const response = await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver",
      receiverBaseUrl: ${JSON.stringify(receiverBaseUrl)}, receiverAuthToken: ${JSON.stringify(receiverToken)} });
    if (!response?.ok) throw new Error("proof_receiver_configuration_failed");
    return true;
  })()`);
}

export async function restoreReceiverConfiguration(workerClient, previous, owned) {
  return evaluateJson(workerClient, `(async () => {
    const response = await chrome.runtime.sendMessage({ type: "polylogue.configureReceiver", restore: {
      previous: ${JSON.stringify(previous)}, owned: ${JSON.stringify(owned)}
    } });
    if (!response?.ok) throw new Error("proof_receiver_configuration_changed");
    return true;
  })()`);
}

export async function restoreProofReceiverAfterConfiguration(workerClient, previous, owned, configuration) {
  // A signal may arrive while the original mutation is still awaiting storage.
  // Settle that mutation before returning the exact original settings.
  await configuration.catch(() => undefined);
  return restoreReceiverConfiguration(workerClient, previous, owned);
}

async function proofWindowId(browserClient, targetId) {
  const result = await browserClient.call("Browser.getWindowForTarget", { targetId });
  if (!Number.isInteger(result.windowId)) throw new Error("shared Chrome proof target has no browser window");
  return result.windowId;
}

async function captureProvider(workerClient, provider, windowId, timeoutMs) {
  return evaluateJson(workerClient, `(async () => {
    const deadline = Date.now() + ${JSON.stringify(timeoutMs)};
    while (Date.now() < deadline) {
      const tabs = await chrome.tabs.query({ windowId: ${JSON.stringify(windowId)} });
      if (tabs.length === 1) {
        const tab = tabs[0];
        try {
          if (new URL(tab.url || "about:blank").hostname === ${JSON.stringify(provider.host)} && tab.pinned !== true) {
            const result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" });
            return { result };
          }
        } catch { /* The content script is still loading. */ }
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return { result: { ok: false, error: "capture_timed_out" } };
  })()`);
}

function sha256(value) {
  return createHash("sha256").update(String(value || "")).digest("hex");
}

function redactUrl(value) {
  try {
    const url = new URL(value);
    const pathParts = url.pathname.split("/").filter(Boolean).map((part, index) => {
      if (index === 0 && ["c", "chat"].includes(part)) return part;
      return /^[A-Za-z0-9_-]{10,}$/.test(part) ? `<sha256:${sha256(part).slice(0, 12)}>` : part;
    });
    return `${url.origin}/${pathParts.join("/")}`.replace(/\/$/, url.pathname === "/" ? "/" : "");
  } catch {
    return "unparseable-url";
  }
}

function providerSummary(provider, payload) {
  const session = payload?.result?.envelope?.session || {};
  const provenance = payload?.result?.envelope?.provenance || {};
  const capture = payload?.result?.captureResult || {};
  const sourceUrl = provenance.source_url || provider.url;
  const adapter = provenance.adapter_name || null;
  const turnCount = Array.isArray(session.turns) ? session.turns.length : 0;
  return {
    ok: payload?.result?.ok === true && session.provider === provider.provider && provider.adapters.includes(adapter) && turnCount > 0 && Boolean(capture.artifact_ref),
    host: provider.host,
    opened_url_redacted: redactUrl(provider.url),
    source_url_redacted: redactUrl(sourceUrl),
    source_url_sha256: sha256(sourceUrl),
    provider: session.provider || null,
    provider_session_id_sha256: session.provider_session_id ? sha256(session.provider_session_id) : null,
    adapter_name: adapter,
    turn_count: turnCount,
    artifact_ref: capture.artifact_ref || null,
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
// Set while the operator's receiver settings are replaced by the proof's, so a
// signal restores them before exiting instead of leaving the shared extension
// pointed at a receiver that is about to shut down.
let pendingReceiverRestore = null;

function installShutdownCleanup() {
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      if (shutdownRequested) return;
      shutdownRequested = true;
      Promise.resolve(pendingReceiverRestore && pendingReceiverRestore())
        .catch(() => undefined)
        .then(() => activeBrowserClient && closeProofTargets(activeBrowserClient, createdTargetIds))
        .catch(() => undefined)
        .finally(() => process.exit(signal === "SIGINT" ? 130 : 143));
    });
  }
}

async function runLiveProviderProof() {
  requireExpectedServiceContext();
  installShutdownCleanup();
  const { extensionRoot, receiverBaseUrl, receiverToken, providers, timeoutMs, startupTimeoutMs, interactiveWaitMs } = fixedInputs();
  const deadline = Date.now() + timeoutMs;
  const remaining = (phase) => {
    const budget = deadline - Date.now();
    if (budget <= 0) throw new Error(`live provider proof timed out during ${phase}`);
    return budget;
  };
  const selected = providers.map((name) => {
    const provider = PROVIDERS[name];
    if (!provider) throw new Error(`unsupported live provider: ${name}`);
    return provider;
  });
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  let workerClient;
  let previousReceiverConfiguration;
  let primaryFailure;
  try {
    await runChromeControl(["status"], Math.min(_CONTROL_TIMEOUT_MS, remaining("shared Chrome status")));
    await runChromeControl(["load-extension", "--path", extensionRoot], Math.min(_CONTROL_TIMEOUT_MS, remaining("extension load")));
    const version = await waitJson(`http://127.0.0.1:${_CDP_PORT}/json/version`, Math.min(startupTimeoutMs, remaining("shared Chrome CDP")));
    activeBrowserClient = await connectCdp(version.webSocketDebuggerUrl);
    workerClient = await waitForExtensionWorker(unpackedExtensionId(extensionRoot, manifest), Math.min(startupTimeoutMs, remaining("extension startup")));
    previousReceiverConfiguration = await receiverConfiguration(workerClient);
    const restoreClient = workerClient;
    const saved = previousReceiverConfiguration;
    const owned = { baseUrl: receiverBaseUrl.replace(/\/+$/, ""), token: receiverToken, receiverId: null };
    const configuration = configureReceiver(workerClient, owned.baseUrl, owned.token);
    pendingReceiverRestore = () => restoreProofReceiverAfterConfiguration(restoreClient, saved, owned, configuration);
    await configuration;
    const proofTargets = [];
    for (const provider of selected) {
      const targetId = await openAgentWindow(provider.url, Math.min(_CONTROL_TIMEOUT_MS, remaining(`open ${provider.host}`)), (id) => createdTargetIds.push(id));
      proofTargets.push({ provider, windowId: await proofWindowId(activeBrowserClient, targetId) });
    }
    if (interactiveWaitMs > 0) await sleep(Math.min(interactiveWaitMs, remaining("interactive wait")));
    const summary = Object.fromEntries(await Promise.all(proofTargets.map(async ({ provider, windowId }) => [provider.host, providerSummary(provider, await captureProvider(workerClient, provider, windowId, remaining("provider capture")))])));
    return { ok: Object.values(summary).every((item) => item.ok === true), providers: summary, privacy_posture: "shared-Chrome output redacts URLs and session ids and omits transcript text" };
  } catch (error) {
    primaryFailure = error;
    throw error;
  } finally {
    const cleanupFailures = [];
    try {
      if (pendingReceiverRestore) await pendingReceiverRestore();
    } catch (error) {
      cleanupFailures.push(error);
    }
    pendingReceiverRestore = null;
    try {
      if (activeBrowserClient) await closeProofTargets(activeBrowserClient, createdTargetIds);
    } catch (error) {
      cleanupFailures.push(error);
    } finally {
      if (workerClient) workerClient.close();
      if (activeBrowserClient) activeBrowserClient.close();
      activeBrowserClient = null;
      createdTargetIds = [];
    }
    if (cleanupFailures.length) {
      throw new AggregateError(primaryFailure ? [primaryFailure, ...cleanupFailures] : cleanupFailures, "proof_receiver_cleanup_failed");
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runLiveProviderProof()
    .then((result) => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch((error) => {
      process.stderr.write(`${error.stack || error.message || error}\n`);
      process.exitCode = 1;
    });
}
