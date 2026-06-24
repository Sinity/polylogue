#!/usr/bin/env node
// Launch a local Chrome/Chromium with an operator-approved copied profile, load
// the unpacked extension, open live ChatGPT/Claude pages, and write a redacted
// proof summary. This script is intentionally local-only: it refuses CI by
// default, refuses common live profile roots, and never prints raw conversation
// text. Receiver spool artifacts are local ignored state.

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import net from "node:net";
import path from "node:path";

const receiverBaseUrl = (process.env.POLYLOGUE_LIVE_PROOF_RECEIVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "");
const receiverAuthToken = process.env.POLYLOGUE_LIVE_PROOF_RECEIVER_TOKEN || "";
const extensionRoot = path.resolve(process.env.POLYLOGUE_LIVE_PROOF_EXTENSION_ROOT || ".");
const outputPath = process.env.POLYLOGUE_LIVE_PROOF_OUT || "";
const profileDir = process.env.POLYLOGUE_LIVE_PROOF_PROFILE_DIR
  ? path.resolve(process.env.POLYLOGUE_LIVE_PROOF_PROFILE_DIR)
  : "";
const timeoutMs = Number(process.env.POLYLOGUE_LIVE_PROOF_TIMEOUT_MS || "120000");
const interactiveWaitMs = Number(process.env.POLYLOGUE_LIVE_PROOF_WAIT_MS || "45000");
const spoolDir = process.env.POLYLOGUE_LIVE_PROOF_SPOOL_DIR || "";
const requestedProviders = parseProviders(process.env.POLYLOGUE_LIVE_PROOF_PROVIDERS || "chatgpt,claude");
const headless = envFlag("POLYLOGUE_LIVE_PROOF_HEADLESS", false);
const noSandbox = envFlag("POLYLOGUE_LIVE_PROOF_NO_SANDBOX", headless);

const providerCatalog = {
  chatgpt: {
    key: "chatgpt",
    host: "chatgpt.com",
    url: process.env.POLYLOGUE_LIVE_PROOF_CHATGPT_URL || "https://chatgpt.com/",
    expectedProvider: "chatgpt",
    expectedAdapters: ["chatgpt-native-v1", "chatgpt-dom-v1"],
  },
  claude: {
    key: "claude",
    host: "claude.ai",
    url: process.env.POLYLOGUE_LIVE_PROOF_CLAUDE_URL || "https://claude.ai/",
    expectedProvider: "claude-ai",
    expectedAdapters: ["claude-ai-native-v1", "claude-ai-dom-v1"],
  },
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function envFlag(name, defaultValue = false) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return defaultValue;
  return ["1", "true", "yes", "on"].includes(String(raw).trim().toLowerCase());
}

function sha256(text) {
  return createHash("sha256").update(String(text || "")).digest("hex");
}

function redactUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const redactedParts = parts.map((part, index) => {
      if (index === 0 && ["c", "chat"].includes(part)) return part;
      if (/^[A-Za-z0-9_-]{10,}$/.test(part)) return `<sha256:${sha256(part).slice(0, 12)}>`;
      return part;
    });
    return `${parsed.origin}/${redactedParts.join("/")}`.replace(/\/$/, parsed.pathname === "/" ? "/" : "");
  } catch {
    return "unparseable-url";
  }
}

function parseProviders(value) {
  const providers = String(value || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  const unique = [...new Set(providers)];
  const invalid = unique.filter((provider) => !["chatgpt", "claude"].includes(provider));
  if (invalid.length) throw new Error(`unsupported live proof provider(s): ${invalid.join(", ")}`);
  if (!unique.length) throw new Error("at least one live proof provider is required");
  return unique;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

async function waitJson(url, timeout = timeoutMs) {
  const deadline = Date.now() + timeout;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
      lastError = `${response.status} ${await response.text()}`;
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`timed out waiting for ${url}: ${lastError}`);
}

let nextCdpId = 0;

function connectCdp(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
  const pending = new Map();
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const callbacks = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) callbacks.reject(new Error(JSON.stringify(message.error)));
    else callbacks.resolve(message.result);
  };
  return new Promise((resolve, reject) => {
    socket.onerror = reject;
    socket.onopen = () => {
      resolve({
        call(method, params = {}) {
          const id = ++nextCdpId;
          socket.send(JSON.stringify({ id, method, params }));
          return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
        },
        close() {
          socket.close();
        },
      });
    };
  });
}

async function evaluateJson(client, expression) {
  const result = await client.call("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text || "CDP evaluation failed");
  }
  return result.result?.value;
}

function serviceWorkerSuffix(manifest) {
  const worker = manifest?.background?.service_worker;
  if (typeof worker !== "string" || !worker) {
    throw new Error("manifest background.service_worker is missing");
  }
  return `/${worker.replace(/^\/+/, "")}`;
}

function resolveChromeBinary() {
  if (process.env.POLYLOGUE_LIVE_PROOF_CHROME) return process.env.POLYLOGUE_LIVE_PROOF_CHROME;
  for (const candidate of ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"]) {
    const found = spawnSync("sh", ["-c", `command -v ${candidate}`], { encoding: "utf8" });
    if (found.status === 0 && found.stdout.trim()) return candidate;
  }
  return "google-chrome-stable";
}

function commonLiveProfileRoots() {
  const home = process.env.HOME ? path.resolve(process.env.HOME) : "";
  if (!home) return [];
  return [
    path.join(home, ".config", "google-chrome"),
    path.join(home, ".config", "chromium"),
    path.join(home, "Library", "Application Support", "Google", "Chrome"),
    path.join(home, "Library", "Application Support", "Chromium"),
  ];
}

function hasProfileLock(dir) {
  return ["SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"].some((name) => existsSync(path.join(dir, name)));
}

function assertLocalProfileCopy(dir) {
  if (!dir) throw new Error("POLYLOGUE_LIVE_PROOF_PROFILE_DIR is required");
  if (!existsSync(dir)) throw new Error(`live proof profile directory does not exist: ${dir}`);
  if (!statSync(dir).isDirectory()) throw new Error(`live proof profile path is not a directory: ${dir}`);
  const allowLiveProfile = process.env.POLYLOGUE_LIVE_PROOF_ALLOW_LIVE_PROFILE === "1";
  const resolved = path.resolve(dir);
  const liveRoots = commonLiveProfileRoots();
  if (!allowLiveProfile && liveRoots.some((root) => resolved === root || resolved.startsWith(`${root}${path.sep}`))) {
    throw new Error(
      `refusing common live browser profile root ${resolved}; copy it into .local/browser-profiles/ first or set POLYLOGUE_LIVE_PROOF_ALLOW_LIVE_PROFILE=1 locally`,
    );
  }
  if (!allowLiveProfile && hasProfileLock(resolved)) {
    throw new Error(
      `profile copy contains Chrome singleton lock files; recopy with Singleton* excluded or set POLYLOGUE_LIVE_PROOF_ALLOW_LIVE_PROFILE=1 locally`,
    );
  }
}

function signalChromeTree(child, signal) {
  try {
    process.kill(-child.pid, signal);
  } catch (_error) {
    try {
      child.kill(signal);
    } catch (_childError) {
      // Chrome may already be gone; shutdown should stay best-effort.
    }
  }
}

function terminateProcess(child) {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(termTimer);
      clearTimeout(killTimer);
      child.off("exit", finish);
      child.off("error", finish);
      resolve();
    };
    const termTimer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) signalChromeTree(child, "SIGTERM");
    }, 0);
    const killTimer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) signalChromeTree(child, "SIGKILL");
      setTimeout(finish, 500);
    }, 3000);
    child.once("exit", finish);
    child.once("error", finish);
  });
}

async function waitForExtensionWorker(debuggingPort, expectedWorkerSuffix, expectedManifestName) {
  const deadline = Date.now() + timeoutMs;
  let targets = [];
  while (Date.now() < deadline) {
    targets = await waitJson(`http://127.0.0.1:${debuggingPort}/json/list`, Math.min(2000, timeoutMs));
    const candidates = targets.filter(
      (target) => target.type === "service_worker" && target.url.startsWith("chrome-extension://"),
    );
    for (const candidate of candidates) {
      const client = await connectCdp(candidate.webSocketDebuggerUrl);
      await client.call("Runtime.enable");
      const manifestName = await evaluateJson(client, "chrome.runtime.getManifest().name").catch(() => null);
      if (manifestName === expectedManifestName) return { worker: candidate, client };
      if (candidates.length === 1) return { worker: candidate, client };
      client.close();
    }
    await sleep(250);
  }
  throw new Error(
    `Polylogue extension service worker not found; expectedSuffix=${expectedWorkerSuffix}; targets=${JSON.stringify(targets)}`
  );
}

async function openProviderTarget(browserClient, debuggingPort, url) {
  const created = await browserClient.call("Target.createTarget", { url });
  const targetId = created.targetId;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const targets = await waitJson(`http://127.0.0.1:${debuggingPort}/json/list`, Math.min(2000, timeoutMs));
    const target = targets.find((candidate) => candidate.id === targetId || candidate.url === url);
    if (target?.webSocketDebuggerUrl) {
      const client = await connectCdp(target.webSocketDebuggerUrl);
      await client.call("Runtime.enable");
      return { target, client };
    }
    await sleep(250);
  }
  throw new Error(`live provider page target not found for ${redactUrl(url)}`);
}

async function waitForReadyPage(client) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const readyState = await evaluateJson(client, "document.readyState");
    if (readyState === "complete" || readyState === "interactive") return readyState;
    await sleep(250);
  }
  throw new Error("live provider page did not become ready");
}

async function configureExtension(workerClient) {
  return evaluateJson(
    workerClient,
    `(async () => {
      if (!globalThis.chrome?.storage?.local) {
        if (${JSON.stringify(receiverBaseUrl)} === "http://127.0.0.1:8765" && !${JSON.stringify(receiverAuthToken)}) {
          return {
            receiverBaseUrl: "http://127.0.0.1:8765",
            receiverAuthToken: "",
            storageConfigured: false,
            caveat: "chrome.storage.local unavailable in service-worker CDP context; using extension defaults"
          };
        }
        throw new Error("extension service-worker CDP target does not expose chrome.storage.local");
      }
      await chrome.storage.local.set({
        receiverBaseUrl: ${JSON.stringify(receiverBaseUrl)},
        receiverAuthToken: ${JSON.stringify(receiverAuthToken)}
      });
      return { ...(await chrome.storage.local.get(["receiverBaseUrl", "receiverAuthToken"])), storageConfigured: true };
    })()`,
  );
}

async function captureProvider(workerClient, providerConfig) {
  return evaluateJson(
    workerClient,
    `(async () => {
      if (!globalThis.chrome?.tabs?.query) {
        throw new Error("extension service-worker CDP target does not expose chrome.tabs");
      }
      const deadline = Date.now() + ${JSON.stringify(timeoutMs)};
      let last = null;
      while (Date.now() < deadline) {
        const tabs = await chrome.tabs.query({});
        const tab = tabs.find((candidate) => {
          try {
            return new URL(candidate.url || "about:blank").hostname === ${JSON.stringify(providerConfig.host)};
          } catch (_error) {
            return false;
          }
        });
        if (tab && typeof tab.id === "number") {
          try {
            const result = await chrome.tabs.sendMessage(tab.id, {type: "polylogue.capturePage"});
            last = {tab: {id: tab.id, url: tab.url, title: tab.title}, result};
            if (result && result.ok) return {ok: true, tab: {id: tab.id, url: tab.url, title: tab.title}, result};
          } catch (error) {
            last = {tab: {id: tab.id, url: tab.url, title: tab.title}, error: String(error && error.message ? error.message : error)};
          }
        } else {
          last = {error: "tab_not_found"};
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      return {ok: false, ...last};
    })()`,
  );
}

function safeProviderSummary(providerConfig, capturePayload) {
  const result = capturePayload?.result || {};
  const envelope = result.envelope || {};
  const session = envelope.session || {};
  const provenance = envelope.provenance || {};
  const captureResult = result.captureResult || {};
  const archiveState = result.archiveState || {};
  const turns = Array.isArray(session.turns) ? session.turns : [];
  const roles = turns.map((turn) => turn.role).filter(Boolean);
  const artifactRef = typeof captureResult.artifact_ref === "string" ? captureResult.artifact_ref : null;
  const artifactPath = artifactRef && spoolDir ? path.join(spoolDir, artifactRef) : null;
  const adapterName = typeof provenance.adapter_name === "string" ? provenance.adapter_name : null;
  const provider = typeof session.provider === "string" ? session.provider : null;
  const sourceUrl = typeof provenance.source_url === "string" ? provenance.source_url : capturePayload?.tab?.url || providerConfig.url;
  const providerSessionId = typeof session.provider_session_id === "string" ? session.provider_session_id : "";
  const ok = Boolean(
    capturePayload?.ok &&
      result.ok === true &&
      provider === providerConfig.expectedProvider &&
      providerConfig.expectedAdapters.includes(adapterName) &&
      turns.length > 0 &&
      artifactRef &&
      captureResult.receiver_request_id &&
      archiveState.receiver_request_id &&
      (!artifactPath || existsSync(artifactPath)),
  );
  return {
    ok,
    host: providerConfig.host,
    opened_url_redacted: redactUrl(providerConfig.url),
    source_url_redacted: redactUrl(sourceUrl),
    source_url_sha256: sha256(sourceUrl),
    tab: capturePayload?.tab
      ? {
          id: capturePayload.tab.id,
          title: capturePayload.tab.title || null,
          url_redacted: redactUrl(capturePayload.tab.url || providerConfig.url),
          url_sha256: sha256(capturePayload.tab.url || providerConfig.url),
        }
      : null,
    expected_provider: providerConfig.expectedProvider,
    provider,
    provider_session_id_sha256: providerSessionId ? sha256(providerSessionId) : null,
    expected_adapters: providerConfig.expectedAdapters,
    adapter_name: adapterName,
    turn_count: turns.length,
    roles,
    capture_result: {
      artifact_ref: artifactRef,
      receiver_request_id: captureResult.receiver_request_id || null,
      bytes_written: captureResult.bytes_written || null,
    },
    archive_state: {
      captured: archiveState.captured ?? null,
      receiver_request_id: archiveState.receiver_request_id || null,
    },
    artifact_path: artifactPath,
    artifact_exists: artifactPath ? existsSync(artifactPath) : null,
    error: capturePayload?.error || result.error || null,
  };
}

function withoutUrlArgs(args) {
  return args.map((arg) => {
    if (String(arg).startsWith("--user-data-dir=")) return "--user-data-dir=<copied-profile>";
    return arg;
  });
}

async function main() {
  if (process.env.CI && process.env.POLYLOGUE_LIVE_PROOF_ALLOW_CI !== "1") {
    throw new Error("live provider proof refuses to run in CI; use deterministic smokes in CI and run this locally");
  }
  assertLocalProfileCopy(profileDir);
  mkdirSync(profileDir, { recursive: true });
  const providerConfigs = requestedProviders.map((provider) => providerCatalog[provider]);
  const localManifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const expectedWorkerSuffix = serviceWorkerSuffix(localManifest);
  const chromeBinary = resolveChromeBinary();
  const debuggingPort = await freePort();
  const chromeArgs = [
    `--user-data-dir=${profileDir}`,
    `--remote-debugging-port=${debuggingPort}`,
    "--no-first-run",
    "--disable-default-apps",
    "--no-default-browser-check",
    `--unsafely-treat-insecure-origin-as-secure=${receiverBaseUrl}`,
    `--disable-extensions-except=${extensionRoot}`,
    `--load-extension=${extensionRoot}`,
    "about:blank",
  ];
  if (headless) chromeArgs.splice(2, 0, "--headless=new");
  if (noSandbox) chromeArgs.splice(3, 0, "--no-sandbox");
  if (!headless) chromeArgs.splice(chromeArgs.length - 1, 0, "--new-window");
  const chrome = spawn(chromeBinary, chromeArgs, { detached: true, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  chrome.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  chrome.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  let workerClient = null;
  let browserClient = null;
  const pageClients = [];
  try {
    const browserVersion = await waitJson(`http://127.0.0.1:${debuggingPort}/json/version`);
    browserClient = await connectCdp(browserVersion.webSocketDebuggerUrl);
    const { worker, client } = await waitForExtensionWorker(debuggingPort, expectedWorkerSuffix, localManifest.name);
    workerClient = client;
    const extensionId = new URL(worker.url).host;
    const configuredReceiver = await configureExtension(workerClient);

    for (const config of providerConfigs) {
      const { client: pageClient } = await openProviderTarget(browserClient, debuggingPort, config.url);
      pageClients.push(pageClient);
      await waitForReadyPage(pageClient);
    }

    if (interactiveWaitMs > 0) await sleep(interactiveWaitMs);

    const providers = {};
    for (const config of providerConfigs) {
      const capturePayload = await captureProvider(workerClient, config);
      providers[config.key] = safeProviderSummary(config, capturePayload);
    }

    const summary = {
      ok: Object.values(providers).every((provider) => provider.ok === true),
      chrome_binary: chromeBinary,
      chrome_args: withoutUrlArgs(chromeArgs),
      debugging_port: debuggingPort,
      extension_id: extensionId,
      extension_root: extensionRoot,
      manifest: {
        name: localManifest.name,
        version: localManifest.version,
        manifest_version: localManifest.manifest_version,
      },
      opened_providers: providerConfigs.map((provider) => provider.key),
      privacy_posture:
        "operator-local copied-profile proof; summary redacts URLs/session ids and omits raw turn text; receiver spool may contain raw captured content under ignored local artifacts",
      profile_dir: profileDir,
      profile_policy: "copied user-data-dir only; never CI/cloud; never commit profile, cookies, screenshots, or receiver spool",
      receiver_base_url: receiverBaseUrl,
      receiver_configured: {
        receiverBaseUrl: configuredReceiver.receiverBaseUrl,
        authConfigured: Boolean(configuredReceiver.receiverAuthToken),
      },
      service_worker_url: worker.url,
      spool_dir: spoolDir || null,
      interactive_wait_ms: interactiveWaitMs,
      providers,
    };
    if (outputPath) writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify(summary)}\n`);
    if (!summary.ok) process.exitCode = 1;
  } finally {
    for (const client of pageClients) client.close();
    if (workerClient) workerClient.close();
    if (browserClient) browserClient.close();
    await terminateProcess(chrome);
    if (stderr && process.exitCode) process.stderr.write(stderr);
    if (stdout && process.env.POLYLOGUE_LIVE_PROOF_DEBUG_STDOUT === "1") process.stderr.write(stdout);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message || error}\n`);
  process.exit(1);
});
