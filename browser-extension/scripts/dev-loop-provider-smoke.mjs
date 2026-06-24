#!/usr/bin/env node
// Launch Chrome/Chromium with the unpacked extension and deterministic provider
// fixture pages served on the real supported origins. This proves content-script
// injection and service-worker receiver delivery without live ChatGPT/Claude.ai
// cookies or cloud browser state.

import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";

const receiverBaseUrl = (process.env.POLYLOGUE_PROVIDER_SMOKE_RECEIVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "");
const receiverAuthToken = process.env.POLYLOGUE_PROVIDER_SMOKE_RECEIVER_TOKEN || "";
const extensionRoot = path.resolve(process.env.POLYLOGUE_PROVIDER_SMOKE_EXTENSION_ROOT || ".");
const outputPath = process.env.POLYLOGUE_PROVIDER_SMOKE_OUT || "";
const profileDir =
  process.env.POLYLOGUE_PROVIDER_SMOKE_PROFILE_DIR || mkdtempSync(path.join(tmpdir(), "polylogue-provider-smoke-"));
const keepProfile = process.env.POLYLOGUE_PROVIDER_SMOKE_KEEP_PROFILE === "1";
const timeoutMs = Number(process.env.POLYLOGUE_PROVIDER_SMOKE_TIMEOUT_MS || "10000");
const spoolDir = process.env.POLYLOGUE_PROVIDER_SMOKE_SPOOL_DIR || "";
const headless = process.env.POLYLOGUE_PROVIDER_SMOKE_HEADLESS !== "0";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function nixStoreChromiumCandidates() {
  const storeRoot = "/nix/store";
  if (!existsSync(storeRoot)) return [];
  return readdirSync(storeRoot)
    .map((entry) => {
      const match = entry.match(/-chromium-(\d+(?:\.\d+)+)$/);
      if (!match) return null;
      const binary = path.join(storeRoot, entry, "bin", "chromium");
      if (!existsSync(binary)) return null;
      return {
        binary,
        version: match[1].split(".").map((part) => Number(part)),
      };
    })
    .filter(Boolean)
    .sort((left, right) => {
      const width = Math.max(left.version.length, right.version.length);
      for (let index = 0; index < width; index += 1) {
        const delta = (right.version[index] || 0) - (left.version[index] || 0);
        if (delta !== 0) return delta;
      }
      return left.binary.localeCompare(right.binary);
    })
    .map((candidate) => candidate.binary);
}

function resolveChromeBinary() {
  if (process.env.POLYLOGUE_PROVIDER_SMOKE_CHROME) return process.env.POLYLOGUE_PROVIDER_SMOKE_CHROME;
  const pathCandidates = ["chromium", "chromium-browser", "chrome-for-testing"];
  for (const candidate of pathCandidates) {
    const found = spawnSync("sh", ["-c", `command -v ${candidate}`], { encoding: "utf8" });
    if (found.status === 0 && found.stdout.trim()) return candidate;
  }
  for (const candidate of nixStoreChromiumCandidates()) return candidate;
  for (const candidate of ["google-chrome-stable", "google-chrome"]) {
    const found = spawnSync("sh", ["-c", `command -v ${candidate}`], { encoding: "utf8" });
    if (found.status === 0 && found.stdout.trim()) return candidate;
  }
  return "google-chrome-stable";
}

function fixtureHtml(provider) {
  if (provider === "chatgpt") {
    return `<!doctype html>
<html>
  <head><title>Polylogue ChatGPT provider smoke</title></head>
  <body>
    <main>
      <article data-testid="conversation-turn-user-1">ChatGPT fixture user turn</article>
      <article data-testid="conversation-turn-assistant-1">ChatGPT fixture assistant turn</article>
    </main>
  </body>
</html>`;
  }
  return `<!doctype html>
<html>
  <head><title>Polylogue Claude provider smoke</title></head>
  <body>
    <main>
      <article data-message-author-role="human">Claude fixture user turn</article>
      <article data-message-author-role="assistant">Claude fixture assistant turn</article>
    </main>
  </body>
</html>`;
}

function generateCertificate(certDir) {
  const keyPath = path.join(certDir, "provider-smoke.key.pem");
  const certPath = path.join(certDir, "provider-smoke.cert.pem");
  const result = spawnSync(
    "openssl",
    [
      "req",
      "-x509",
      "-newkey",
      "rsa:2048",
      "-nodes",
      "-keyout",
      keyPath,
      "-out",
      certPath,
      "-days",
      "1",
      "-subj",
      "/CN=Polylogue Dev Loop Provider Smoke",
      "-addext",
      "subjectAltName=DNS:chatgpt.com,DNS:claude.ai",
    ],
    { encoding: "utf8" },
  );
  if (result.error) {
    throw new Error(`openssl certificate generation failed: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`openssl certificate generation failed: ${result.stderr || result.stdout || result.status}`);
  }
  return { key: readFileSync(keyPath), cert: readFileSync(certPath) };
}

async function startProviderFixtureServer(certDir) {
  const server = https.createServer(generateCertificate(certDir), (request, response) => {
    const host = String(request.headers.host || "").split(":")[0];
    const provider = host.includes("claude.ai") ? "claude" : "chatgpt";
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
    });
    response.end(fixtureHtml(provider));
  });
  await new Promise((resolve, reject) => {
    server.on("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  return {
    port,
    server,
    urls: {
      chatgpt: "https://chatgpt.com/c/polylogue-dev-loop-provider-smoke",
      claude: "https://claude.ai/chat/polylogue-dev-loop-provider-smoke",
    },
  };
}

async function startProviderProxyServer(fixturePort) {
  const server = http.createServer((_request, response) => {
    response.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
    response.end("Polylogue provider smoke proxy only supports CONNECT\n");
  });
  server.on("connect", (request, clientSocket, head) => {
    const [host, portText] = String(request.url || "").split(":");
    const port = Number(portText || "443");
    if (!["chatgpt.com", "claude.ai"].includes(host) || port !== 443) {
      clientSocket.write("HTTP/1.1 403 Forbidden\r\n\r\n");
      clientSocket.destroy();
      return;
    }
    const upstream = net.connect(fixturePort, "127.0.0.1", () => {
      clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      if (head.length) upstream.write(head);
      upstream.pipe(clientSocket);
      clientSocket.pipe(upstream);
    });
    upstream.on("error", () => {
      clientSocket.destroy();
    });
    clientSocket.on("error", () => {
      upstream.destroy();
    });
  });
  await new Promise((resolve, reject) => {
    server.on("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  return { port, server };
}

function closeServer(server) {
  return new Promise((resolve) => server.close(resolve));
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
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve();
  }
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

async function openExtensionController(workerClient, debuggingPort, extensionId) {
  const controllerUrl = await evaluateJson(
    workerClient,
    `chrome.runtime.getURL("src/popup.html")`,
  );
  const created = await evaluateJson(
    workerClient,
    `(async () => {
      if (!globalThis.chrome?.tabs?.create) {
        throw new Error("extension service-worker CDP target does not expose chrome.tabs.create");
      }
      let tab;
      try {
        tab = await chrome.tabs.create({url: ${JSON.stringify(controllerUrl)}, active: false});
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        if (!message.includes("No current window") || !globalThis.chrome?.windows?.create) {
          throw error;
        }
        const createdWindow = await chrome.windows.create({url: ${JSON.stringify(controllerUrl)}, focused: false, type: "normal"});
        tab = Array.isArray(createdWindow.tabs) && createdWindow.tabs.length ? createdWindow.tabs[0] : null;
      }
      if (!tab) throw new Error("extension API did not return a controller tab");
      return {id: tab.id ?? null, url: tab.url ?? null, pendingUrl: tab.pendingUrl ?? null, title: tab.title ?? null};
    })()`,
  );
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const targets = await waitJson(`http://127.0.0.1:${debuggingPort}/json/list`, Math.min(2000, timeoutMs));
    const target = targets.find((candidate) => candidate.url === controllerUrl);
    if (target?.webSocketDebuggerUrl) {
      const client = await connectCdp(target.webSocketDebuggerUrl);
      await client.call("Runtime.enable");
      await client.call("Page.enable");
      await waitForReadyPage(client);
      await waitForExtensionControllerApis(client);
      return { target, client, tab: created };
    }
    await sleep(250);
  }
  throw new Error(`Polylogue extension controller page not found for ${controllerUrl}; tab=${JSON.stringify(created)}`);
}

async function waitForExtensionControllerApis(client) {
  const deadline = Date.now() + timeoutMs;
  let last = null;
  while (Date.now() < deadline) {
    last = await evaluateJson(
      client,
      `(() => ({
        href: location.href,
        readyState: document.readyState,
        hasChrome: Boolean(globalThis.chrome),
        hasRuntime: Boolean(globalThis.chrome?.runtime),
        hasTabsCreate: Boolean(globalThis.chrome?.tabs?.create),
        hasTabsQuery: Boolean(globalThis.chrome?.tabs?.query),
        hasStorageLocal: Boolean(globalThis.chrome?.storage?.local),
        hasScripting: Boolean(globalThis.chrome?.scripting?.executeScript)
      }))()`,
    ).catch((error) => ({ error: String(error.message || error) }));
    if (last?.hasTabsCreate && last?.hasTabsQuery && last?.hasStorageLocal && last?.hasScripting) return last;
    await sleep(100);
  }
  throw new Error(`extension controller missing required APIs: ${JSON.stringify(last)}`);
}

async function openProviderTargetFromExtension(extensionClient, debuggingPort, url) {
  const createdTab = await evaluateJson(
    extensionClient,
    `(async () => {
      if (!globalThis.chrome?.tabs?.create) {
        throw new Error("extension controller CDP target does not expose chrome.tabs.create");
      }
      let tab;
      try {
        tab = await chrome.tabs.create({url: ${JSON.stringify(url)}, active: false});
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        if (!message.includes("No current window") || !globalThis.chrome?.windows?.create) {
          throw error;
        }
        const createdWindow = await chrome.windows.create({url: ${JSON.stringify(url)}, focused: false, type: "normal"});
        tab = Array.isArray(createdWindow.tabs) && createdWindow.tabs.length ? createdWindow.tabs[0] : null;
      }
      if (!tab) throw new Error("extension API did not return a provider tab");
      return {id: tab.id ?? null, url: tab.url ?? null, pendingUrl: tab.pendingUrl ?? null, title: tab.title ?? null};
    })()`,
  );
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const targets = await waitJson(`http://127.0.0.1:${debuggingPort}/json/list`, Math.min(2000, timeoutMs));
    const target = targets.find((candidate) => candidate.url === url);
    if (target?.webSocketDebuggerUrl) {
      const client = await connectCdp(target.webSocketDebuggerUrl);
      await client.call("Runtime.enable");
      await client.call("Page.enable");
      await client.call("Page.navigate", { url });
      return { target, client, tab: createdTab };
    }
    await sleep(250);
  }
  throw new Error(`provider page target not found for ${url}`);
}

async function waitForReadyPage(client) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const readyState = await evaluateJson(client, "document.readyState");
    if (readyState === "complete" || readyState === "interactive") return readyState;
    await sleep(100);
  }
  throw new Error("provider fixture page did not become ready");
}

async function providerPageDiagnostics(client) {
  return evaluateJson(
    client,
    `(() => ({
      href: location.href,
      title: document.title || null,
      readyState: document.readyState,
      articleCount: document.querySelectorAll("article").length,
      hasPolylogueCapture: Boolean(window.polylogueCapture),
      bodyTextSample: (document.body?.innerText || document.body?.textContent || "").slice(0, 240)
    }))()`,
  );
}

async function configureExtension(extensionClient) {
  return evaluateJson(
    extensionClient,
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
        throw new Error("extension controller CDP target does not expose chrome.storage.local");
      }
      await chrome.storage.local.set({
        receiverBaseUrl: ${JSON.stringify(receiverBaseUrl)},
        receiverAuthToken: ${JSON.stringify(receiverAuthToken)}
      });
      return { ...(await chrome.storage.local.get(["receiverBaseUrl", "receiverAuthToken"])), storageConfigured: true };
    })()`,
  );
}

async function captureProvider(extensionClient, providerConfig) {
  return evaluateJson(
    extensionClient,
    `(async () => {
      if (!globalThis.chrome?.tabs?.query) {
        throw new Error("extension controller CDP target does not expose chrome.tabs");
      }
      async function sendCapture(tabId) {
        try {
          const result = await chrome.tabs.sendMessage(tabId, {type: "polylogue.capturePage"});
          return {ok: Boolean(result && result.ok), result, injection_mode: "manifest"};
        } catch (error) {
          const message = String(error && error.message ? error.message : error);
          if (!message.includes("Receiving end does not exist") || !globalThis.chrome?.scripting?.executeScript) {
            return {ok: false, error: message, injection_mode: "none"};
          }
          await chrome.scripting.executeScript({target: {tabId}, files: ["src/common.js"]});
          await chrome.scripting.executeScript({target: {tabId}, files: [${JSON.stringify(providerConfig.contentScriptFile)}]});
          const result = await chrome.tabs.sendMessage(tabId, {type: "polylogue.capturePage"});
          return {ok: Boolean(result && result.ok), result, injection_mode: "scripted_retry"};
        }
      }
      const summarizeTabs = (tabs) => tabs.map((tab) => ({
        id: typeof tab.id === "number" ? tab.id : null,
        url: tab.url || tab.pendingUrl || null,
        title: tab.title || null,
        active: Boolean(tab.active)
      }));
      const deadline = Date.now() + ${JSON.stringify(timeoutMs)};
      let last = null;
      while (Date.now() < deadline) {
        const tabs = await chrome.tabs.query({});
        const tabInventory = summarizeTabs(tabs);
        const expectedTabId = ${JSON.stringify(providerConfig.expectedTabId ?? null)};
        const openedUrl = ${JSON.stringify(providerConfig.openedTab?.url || providerConfig.openedTab?.pendingUrl || null)};
        const tab = tabs.find((candidate) => typeof expectedTabId === "number" && candidate.id === expectedTabId)
          || tabs.find((candidate) => {
            try {
              return new URL(candidate.url || candidate.pendingUrl || openedUrl || "about:blank").hostname === ${JSON.stringify(providerConfig.host)};
            } catch (_error) {
              return false;
            }
          });
        if (tab && typeof tab.id === "number") {
          const capture = await sendCapture(tab.id);
          last = {
            tab: {id: tab.id, url: tab.url, title: tab.title},
            tab_inventory: tabInventory,
            injection_mode: capture.injection_mode,
            result: capture.result,
            error: capture.error
          };
          if (capture.ok) return {ok: true, ...last};
        } else {
          last = {error: "tab_not_found", expected_host: ${JSON.stringify(providerConfig.host)}, tab_inventory: tabInventory};
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
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
  const artifactRef = typeof captureResult.artifact_ref === "string" ? captureResult.artifact_ref : null;
  const artifactPath = artifactRef && spoolDir ? path.join(spoolDir, artifactRef) : null;
  const adapterName = typeof provenance.adapter_name === "string" ? provenance.adapter_name : null;
  const roles = turns.map((turn) => turn.role).filter(Boolean);
  const ok = Boolean(
    capturePayload?.ok &&
      result.ok === true &&
      session.provider === providerConfig.expectedProvider &&
      providerConfig.expectedAdapters.includes(adapterName) &&
      turns.length >= 2 &&
      roles.includes("user") &&
      roles.includes("assistant") &&
      artifactRef &&
      captureResult.receiver_request_id &&
      archiveState.receiver_request_id &&
      (!artifactPath || existsSync(artifactPath)),
  );
  return {
    ok,
    url: providerConfig.url,
    expected_host: providerConfig.host,
    page_target: providerConfig.pageTarget || null,
    page_diagnostics: providerConfig.pageDiagnostics || null,
    opened_tab: providerConfig.openedTab || null,
    tab: capturePayload?.tab || null,
    tab_inventory: capturePayload?.tab_inventory || null,
    injection_mode: capturePayload?.injection_mode || null,
    expected_provider: providerConfig.expectedProvider,
    provider: session.provider || null,
    provider_session_id: session.provider_session_id || null,
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

async function main() {
  const localManifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const expectedWorkerSuffix = serviceWorkerSuffix(localManifest);
  mkdirSync(profileDir, { recursive: true });
  const chromeBinary = resolveChromeBinary();
  const debuggingPort = await freePort();
  const fixtureServer = await startProviderFixtureServer(profileDir);
  const proxyServer = await startProviderProxyServer(fixtureServer.port);
  const chromeArgs = [
    `--user-data-dir=${profileDir}`,
    `--remote-debugging-port=${debuggingPort}`,
    "--no-sandbox",
    "--disable-features=ExtensionsMenuAccessControl",
    "--enable-unsafe-extension-debugging",
    "--ignore-certificate-errors",
    "--allow-insecure-localhost",
    `--proxy-server=http://127.0.0.1:${proxyServer.port}`,
    "--proxy-bypass-list=127.0.0.1;localhost",
    `--unsafely-treat-insecure-origin-as-secure=${receiverBaseUrl}`,
    `--disable-extensions-except=${extensionRoot}`,
    `--load-extension=${extensionRoot}`,
    "about:blank",
  ];
  if (headless) chromeArgs.splice(2, 0, "--headless=new");
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
  let extensionControllerClient = null;
  const pageClients = [];
  try {
    const browserVersion = await waitJson(`http://127.0.0.1:${debuggingPort}/json/version`);
    browserClient = await connectCdp(browserVersion.webSocketDebuggerUrl);
    const providerConfigs = [
      {
        key: "chatgpt",
        host: "chatgpt.com",
        url: fixtureServer.urls.chatgpt,
        expectedProvider: "chatgpt",
        expectedAdapters: ["chatgpt-native-v1", "chatgpt-dom-v1"],
        contentScriptFile: "src/content/chatgpt.js",
      },
      {
        key: "claude",
        host: "claude.ai",
        url: fixtureServer.urls.claude,
        expectedProvider: "claude-ai",
        expectedAdapters: ["claude-ai-native-v1", "claude-ai-dom-v1"],
        contentScriptFile: "src/content/claude.js",
      },
    ];
    const { worker, client } = await waitForExtensionWorker(debuggingPort, expectedWorkerSuffix, localManifest.name);
    workerClient = client;
    const extensionId = new URL(worker.url).host;
    const { target: extensionController, client: controllerClient, tab: extensionControllerTab } = await openExtensionController(
      workerClient,
      debuggingPort,
      extensionId,
    );
    extensionControllerClient = controllerClient;
    const configuredReceiver = await configureExtension(extensionControllerClient);

    for (const config of providerConfigs) {
      const { client, target, tab } = await openProviderTargetFromExtension(
        extensionControllerClient,
        debuggingPort,
        config.url,
      );
      config.pageTarget = {
        id: target.id || null,
        type: target.type || null,
        url: target.url || null,
        title: target.title || null,
      };
      config.openedTab = tab;
      config.expectedTabId = typeof tab?.id === "number" ? tab.id : null;
      pageClients.push(client);
      await waitForReadyPage(client);
      config.pageDiagnostics = await providerPageDiagnostics(client);
    }

    const providers = {};
    for (const config of providerConfigs) {
      const capturePayload = await captureProvider(extensionControllerClient, config);
      providers[config.key] = safeProviderSummary(config, capturePayload);
    }

    const summary = {
      ok: Object.values(providers).every((provider) => provider.ok === true),
      chrome_binary: chromeBinary,
      chrome_args: chromeArgs,
      headless,
      debugging_port: debuggingPort,
      extension_id: extensionId,
      extension_root: extensionRoot,
      fixture_server: {
        port: fixtureServer.port,
        urls: fixtureServer.urls,
        proxy_port: proxyServer.port,
        proxy_mode: "CONNECT chatgpt.com:443 and claude.ai:443 to fixture server",
      },
      manifest: {
        name: localManifest.name,
        version: localManifest.version,
        manifest_version: localManifest.manifest_version,
      },
      privacy_posture: "deterministic fixture pages only; summary omits raw turn text and copied-profile data",
      profile_dir: profileDir,
      receiver_base_url: receiverBaseUrl,
      receiver_configured: {
        receiverBaseUrl: configuredReceiver.receiverBaseUrl,
        authConfigured: Boolean(configuredReceiver.receiverAuthToken),
      },
      spool_dir: spoolDir || null,
      service_worker_url: worker.url,
      extension_controller: {
        id: extensionController.id || null,
        type: extensionController.type || null,
        url: extensionController.url || null,
        title: extensionController.title || null,
        tab: extensionControllerTab || null,
      },
      providers,
    };
    if (outputPath) writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify(summary)}\n`);
    if (!summary.ok) process.exitCode = 1;
  } finally {
    for (const client of pageClients) client.close();
    if (extensionControllerClient) extensionControllerClient.close();
    if (workerClient) workerClient.close();
    if (browserClient) browserClient.close();
    await terminateProcess(chrome);
    await closeServer(proxyServer.server);
    await closeServer(fixtureServer.server);
    if (!keepProfile && !process.env.POLYLOGUE_PROVIDER_SMOKE_PROFILE_DIR) {
      rmSync(profileDir, { recursive: true, force: true });
    }
    if (stderr && process.exitCode) process.stderr.write(stderr);
    if (stdout && process.env.POLYLOGUE_PROVIDER_SMOKE_DEBUG_STDOUT === "1") process.stderr.write(stdout);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message || error}\n`);
  process.exit(1);
});
