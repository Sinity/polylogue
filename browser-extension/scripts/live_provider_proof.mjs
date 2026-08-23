// Fixed copied-profile provider proof semantics for the declared AgentCTL
// operation. Imports are non-launching; direct execution checks its transient
// Sinnixd unit before it can spawn Chrome.

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const PROVIDERS = {
  chatgpt: { host: "chatgpt.com", url: "https://chatgpt.com/", provider: "chatgpt", adapters: ["chatgpt-native-v1", "chatgpt-dom-v1"] },
  claude: { host: "claude.ai", url: "https://claude.ai/", provider: "claude-ai", adapters: ["claude-ai-native-v1", "claude-ai-dom-v1"] },
};
const _CDP_PORT_RANGE = [49056, 49119];
const _RECEIVER_PORT_RANGE = [49120, 49183];
const _PROFILE_DIR = "/realm/state/polylogue/live-provider-proof-profile";
const _WORKFLOW_TIMEOUT_MS = 90_000;
const _STARTUP_TIMEOUT_MS = 30_000;
const _INTERACTIVE_WAIT_MS = 15_000;

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be supplied by the declared live-provider service`);
  return value;
}

function leasedPort(name, range) {
  const port = Number(requiredEnvironment(name));
  if (!Number.isInteger(port) || port < range[0] || port > range[1]) {
    throw new Error(`${name} is outside its declared AgentCTL lease range`);
  }
  return port;
}

function requireExpectedServiceContext() {
  const jobId = process.env.SINNIXD_JOB_ID || "";
  const unit = `sinnixd-job-${jobId}.service`;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(jobId)) {
    throw new Error("live provider proof requires a Sinnixd job UUID");
  }
  if (process.env.SINNIXD_PROJECT_ID !== "polylogue" || process.env.SINNIXD_OPERATION !== "live_provider_proof") {
    throw new Error("live provider proof rejects execution outside its fixed service context");
  }
  const cgroup = readFileSync("/proc/self/cgroup", "utf8").split("\n").find((line) => line.includes("::"))?.split("::", 2)[1] || "";
  if (!cgroup.includes(`/agent.slice/${unit}`)) {
    throw new Error("live provider proof is not inside its matching Sinnixd transient unit");
  }
  const unitExecStart = spawnSync(
    "systemctl",
    ["--user", "show", unit, "--property=ExecStart", "--value"],
    { encoding: "utf8", timeout: 2000 },
  );
  const childCommand = unitExecStart.stdout.slice(unitExecStart.stdout.indexOf("/env -i") + "/env -i".length);
  if (unitExecStart.status !== 0 || !unitExecStart.stdout.includes("/env -i") || !["SINNIXD_JOB_ID", "SINNIXD_PROJECT_ID", "SINNIXD_OPERATION"].every((name) => childCommand.includes(`${name}=${process.env[name]}`))) {
    throw new Error("live provider proof transient unit does not match the declared operation");
  }
}

function resolveChromeBinary() {
  for (const candidate of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome-for-testing"]) {
    const resolved = spawnSync("sh", ["-c", `command -v ${candidate}`], { encoding: "utf8" });
    if (resolved.status === 0 && resolved.stdout.trim()) return resolved.stdout.trim();
  }
  throw new Error("no fixed Chrome/Chromium executable is available on the declared service PATH");
}

function fixedInputs() {
  const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
  const cdpPort = leasedPort("POLYLOGUE_LIVE_PROVIDER_CDP_PORT", _CDP_PORT_RANGE);
  const receiverPort = leasedPort("POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT", _RECEIVER_PORT_RANGE);
  return {
    chromeBinary: resolveChromeBinary(),
    cdpPort,
    extensionRoot: path.resolve(scriptDirectory, ".."),
    outputPath: path.join(requiredEnvironment("TMPDIR"), "polylogue-live-provider-proof.json"),
    profileDir: _PROFILE_DIR,
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

async function terminateProcessGroup(child) {
  if (!child || child.pid === undefined) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
    return;
  }
  await Promise.race([child.exitCode === null ? once(child, "close") : sleep(200), sleep(2000)]);
  try {
    process.kill(-child.pid, "SIGKILL");
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
  }
  if (child.exitCode === null) await once(child, "close");
}

let activeChrome = null;
let shutdownRequested = false;

function installShutdownCleanup() {
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      if (shutdownRequested) return;
      shutdownRequested = true;
      terminateProcessGroup(activeChrome)
        .catch((error) => process.stderr.write(`${error.stack || error.message || error}\n`))
        .finally(() => process.exit(signal === "SIGINT" ? 130 : 143));
    });
  }
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

function copiedProfileRoots() {
  const home = process.env.HOME ? path.resolve(process.env.HOME) : "";
  if (!home) return [];
  return [path.join(home, ".config", "google-chrome"), path.join(home, ".config", "chromium")];
}

function assertCopiedProfile(profileDir) {
  const resolved = path.resolve(profileDir);
  if (!existsSync(resolved) || !statSync(resolved).isDirectory()) throw new Error("copied profile directory is required");
  if (copiedProfileRoots().some((root) => resolved === root || resolved.startsWith(`${root}${path.sep}`))) {
    throw new Error("live provider proof refuses a live browser profile root");
  }
  if (["SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"].some((name) => existsSync(path.join(resolved, name)))) {
    throw new Error("copied profile must exclude Chrome singleton lock files");
  }
  return resolved;
}

function launchFixedChrome(chromeBinary, chromeArgs) {
  // The spawn boundary carries its own guard so no internal caller can turn
  // forged environment variables into a Chrome launch.
  requireExpectedServiceContext();
  return spawn(chromeBinary, chromeArgs, { detached: true, stdio: "ignore" });
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
  throw new Error(`timed out waiting for CDP: ${lastError}`);
}

async function evaluateJson(client, expression) {
  const result = await client.call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "CDP evaluation failed");
  return result.result?.value;
}

async function waitForExtensionWorker(port, expectedName, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const targets = await waitJson(`http://127.0.0.1:${port}/json/list`, Math.min(timeoutMs, 2000));
    for (const target of targets.filter((item) => item.type === "service_worker" && item.url?.startsWith("chrome-extension://"))) {
      const client = await connectCdp(target.webSocketDebuggerUrl);
      const name = await evaluateJson(client, "chrome.runtime.getManifest().name").catch(() => null);
      if (name === expectedName) return { target, client };
      client.close();
    }
    await sleep(250);
  }
  throw new Error("Polylogue extension service worker was not found");
}

async function configureReceiver(workerClient, receiverBaseUrl, receiverToken) {
  return evaluateJson(workerClient, `(async () => { await chrome.storage.local.set({ receiverBaseUrl: ${JSON.stringify(receiverBaseUrl)}, receiverAuthToken: ${JSON.stringify(receiverToken)} }); return true; })()`);
}

async function captureProvider(workerClient, provider, timeoutMs) {
  return evaluateJson(workerClient, `(async () => {
    const deadline = Date.now() + ${JSON.stringify(timeoutMs)};
    while (Date.now() < deadline) {
      const tab = (await chrome.tabs.query({})).find((candidate) => { try { return new URL(candidate.url || "about:blank").hostname === ${JSON.stringify(provider.host)}; } catch { return false; } });
      if (tab && typeof tab.id === "number") {
        try { const result = await chrome.tabs.sendMessage(tab.id, { type: "polylogue.capturePage" }); if (result?.ok) return { tab, result }; } catch { /* page is still loading */ }
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return { result: { ok: false, error: "capture_timed_out" } };
  })()`);
}

function providerSummary(provider, payload) {
  const session = payload?.result?.envelope?.session || {};
  const provenance = payload?.result?.envelope?.provenance || {};
  const capture = payload?.result?.captureResult || {};
  const sourceUrl = provenance.source_url || payload?.tab?.url || provider.url;
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

async function runLiveProviderProof() {
  requireExpectedServiceContext();
  installShutdownCleanup();
  const { chromeBinary, cdpPort, extensionRoot, outputPath, profileDir, receiverBaseUrl, receiverToken, providers, timeoutMs, startupTimeoutMs, interactiveWaitMs } = fixedInputs();
  const deadline = Date.now() + timeoutMs;
  const remaining = (phase) => {
    const budget = deadline - Date.now();
    if (budget <= 0) throw new Error(`live provider proof timed out during ${phase}`);
    return budget;
  };
  const profile = assertCopiedProfile(profileDir);
  const selected = providers.map((name) => {
    const provider = PROVIDERS[name];
    if (!provider) throw new Error(`unsupported live provider: ${name}`);
    return provider;
  });
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const chrome = launchFixedChrome(chromeBinary, [
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${cdpPort}`,
    "--disable-gpu", "--no-first-run", "--disable-default-apps", "--no-default-browser-check", "--enable-unsafe-extension-debugging",
    `--unsafely-treat-insecure-origin-as-secure=${receiverBaseUrl}`,
    `--disable-extensions-except=${extensionRoot}`, `--load-extension=${extensionRoot}`, "--new-window", "about:blank",
  ], { detached: true, stdio: "ignore" });
  activeChrome = chrome;
  let browserClient;
  let workerClient;
  try {
    const version = await waitJson(`http://127.0.0.1:${cdpPort}/json/version`, Math.min(startupTimeoutMs, remaining("Chrome startup")));
    browserClient = await connectCdp(version.webSocketDebuggerUrl);
    const worker = await waitForExtensionWorker(cdpPort, manifest.name, Math.min(startupTimeoutMs, remaining("extension startup")));
    workerClient = worker.client;
    await configureReceiver(workerClient, receiverBaseUrl.replace(/\/+$/, ""), receiverToken);
    for (const provider of selected) await browserClient.call("Target.createTarget", { url: provider.url });
    if (interactiveWaitMs > 0) await sleep(Math.min(interactiveWaitMs, remaining("interactive wait")));
    const captureTimeoutMs = remaining("provider capture");
    const summary = Object.fromEntries(await Promise.all(selected.map(async (provider) => [provider.host, providerSummary(provider, await captureProvider(workerClient, provider, captureTimeoutMs))])));
    const result = { ok: Object.values(summary).every((item) => item.ok === true), providers: summary, privacy_posture: "copied-profile output redacts URLs and session ids and omits transcript text" };
    writeFileSync(outputPath, `${JSON.stringify(result)}\n`, "utf8");
    return result;
  } finally {
    if (workerClient) workerClient.close();
    if (browserClient) { await browserClient.call("Browser.close").catch(() => undefined); browserClient.close(); }
    await terminateProcessGroup(chrome);
    activeChrome = null;
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
