// Private copied-profile provider proof semantics. This module deliberately has
// no executable entrypoint, environment launcher, free-port allocator, or
// process-tree cleanup policy. A future typed AgentCTL operation must provide
// the copied profile, receiver, output location, and descriptor-leased CDP port.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const PROVIDERS = {
  chatgpt: { host: "chatgpt.com", url: "https://chatgpt.com/", provider: "chatgpt", adapters: ["chatgpt-native-v1", "chatgpt-dom-v1"] },
  claude: { host: "claude.ai", url: "https://claude.ai/", provider: "claude-ai", adapters: ["claude-ai-native-v1", "claude-ai-dom-v1"] },
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

export async function runLiveProviderProof({ chromeBinary, cdpPort, extensionRoot, outputPath, profileDir, receiverBaseUrl, receiverToken, providers = ["chatgpt", "claude"], timeoutMs = 120000, interactiveWaitMs = 45000 }) {
  if (!Number.isInteger(cdpPort) || cdpPort < 1024 || cdpPort > 65535) throw new Error("a descriptor-leased CDP port is required");
  if (!chromeBinary || !extensionRoot || !outputPath || !receiverBaseUrl || !receiverToken) throw new Error("typed live-provider service inputs are required");
  const profile = assertCopiedProfile(profileDir);
  const selected = providers.map((name) => {
    const provider = PROVIDERS[name];
    if (!provider) throw new Error(`unsupported live provider: ${name}`);
    return provider;
  });
  const manifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const chrome = spawn(chromeBinary, [
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${cdpPort}`,
    "--no-first-run", "--disable-default-apps", "--no-default-browser-check", "--enable-unsafe-extension-debugging",
    `--unsafely-treat-insecure-origin-as-secure=${receiverBaseUrl}`,
    `--disable-extensions-except=${extensionRoot}`, `--load-extension=${extensionRoot}`, "--new-window", "about:blank",
  ], { detached: false, stdio: "ignore" });
  let browserClient;
  let workerClient;
  try {
    const version = await waitJson(`http://127.0.0.1:${cdpPort}/json/version`, timeoutMs);
    browserClient = await connectCdp(version.webSocketDebuggerUrl);
    const worker = await waitForExtensionWorker(cdpPort, manifest.name, timeoutMs);
    workerClient = worker.client;
    await configureReceiver(workerClient, receiverBaseUrl.replace(/\/+$/, ""), receiverToken);
    for (const provider of selected) await browserClient.call("Target.createTarget", { url: provider.url });
    if (interactiveWaitMs > 0) await sleep(interactiveWaitMs);
    const summary = Object.fromEntries(await Promise.all(selected.map(async (provider) => [provider.host, providerSummary(provider, await captureProvider(workerClient, provider, timeoutMs))])));
    const result = { ok: Object.values(summary).every((item) => item.ok === true), providers: summary, privacy_posture: "copied-profile output redacts URLs and session ids and omits transcript text" };
    writeFileSync(outputPath, `${JSON.stringify(result)}\n`, "utf8");
    return result;
  } finally {
    if (workerClient) workerClient.close();
    if (browserClient) { await browserClient.call("Browser.close").catch(() => undefined); browserClient.close(); }
    chrome.unref();
  }
}
