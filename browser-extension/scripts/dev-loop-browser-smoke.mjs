#!/usr/bin/env node
// Launch Chrome with the unpacked extension, verify the MV3 service worker is
// the Polylogue extension, then exercise the local receiver from the extension
// service-worker origin. This intentionally has no npm dependencies.

import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";

const receiverBaseUrl = (process.env.POLYLOGUE_BROWSER_SMOKE_RECEIVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "");
const receiverAuthToken = process.env.POLYLOGUE_BROWSER_SMOKE_RECEIVER_TOKEN || "";
const extensionRoot = path.resolve(process.env.POLYLOGUE_BROWSER_SMOKE_EXTENSION_ROOT || ".");
const chromeBinary = resolveChromeBinary();
const outputPath = process.env.POLYLOGUE_BROWSER_SMOKE_OUT || "";
const profileDir =
  process.env.POLYLOGUE_BROWSER_SMOKE_PROFILE_DIR || mkdtempSync(path.join(tmpdir(), "polylogue-browser-smoke-"));
const keepProfile = process.env.POLYLOGUE_BROWSER_SMOKE_KEEP_PROFILE === "1";
const timeoutMs = Number(process.env.POLYLOGUE_BROWSER_SMOKE_TIMEOUT_MS || "20000");

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

function fixtureEnvelope() {
  return {
    polylogue_capture_kind: "browser_llm_session",
    schema_version: 1,
    capture_id: "browser-smoke:dev-loop-browser-smoke",
    source: "browser-extension",
    provenance: {
      source_url: "https://chatgpt.com/c/dev-loop-browser-smoke",
      page_title: "Polylogue real browser smoke",
      captured_at: "2026-06-21T00:00:00+00:00",
      adapter_name: "dev-loop-real-browser-smoke",
      capture_mode: "snapshot",
    },
    session: {
      provider: "chatgpt",
      provider_session_id: "dev-loop-browser-smoke",
      title: "Polylogue real browser smoke",
      turns: [{ provider_turn_id: "turn-1", role: "user", text: "real browser smoke" }],
    },
  };
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
  if (process.env.POLYLOGUE_BROWSER_SMOKE_CHROME) return process.env.POLYLOGUE_BROWSER_SMOKE_CHROME;
  for (const candidate of ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"]) {
    const found = spawnSync("sh", ["-c", `command -v ${candidate}`], { encoding: "utf8" });
    if (found.status === 0 && found.stdout.trim()) return candidate;
  }
  return "google-chrome-stable";
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

async function main() {
  const localManifest = JSON.parse(readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
  const expectedWorkerSuffix = serviceWorkerSuffix(localManifest);
  mkdirSync(profileDir, { recursive: true });
  const debuggingPort = await freePort();
  const chromeArgs = [
    `--user-data-dir=${profileDir}`,
    `--remote-debugging-port=${debuggingPort}`,
    "--headless=new",
    "--no-sandbox",
    `--disable-extensions-except=${extensionRoot}`,
    `--load-extension=${extensionRoot}`,
    "about:blank",
  ];
  const chrome = spawn(chromeBinary, chromeArgs, { detached: true, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  chrome.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });
  chrome.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  try {
    const deadline = Date.now() + timeoutMs;
    let worker = null;
    let workerClient = null;
    let targets = [];
    while (Date.now() < deadline && !worker) {
      targets = await waitJson(`http://127.0.0.1:${debuggingPort}/json/list`, Math.min(2000, timeoutMs));
      const candidates = targets.filter(
        (target) => target.type === "service_worker" && target.url.endsWith(expectedWorkerSuffix)
      );
      for (const candidate of candidates) {
        const client = await connectCdp(candidate.webSocketDebuggerUrl);
        await client.call("Runtime.enable");
        worker = candidate;
        workerClient = client;
        break;
      }
      if (!worker) await sleep(250);
    }
    if (!worker || !workerClient) {
      throw new Error(`Polylogue extension service worker not found; targets=${JSON.stringify(targets)}`);
    }
    const extensionId = new URL(worker.url).host;
    const authHeader = JSON.stringify(`Bearer ${receiverAuthToken}`);
    const receiverUrl = JSON.stringify(receiverBaseUrl);
    const envelope = JSON.stringify(fixtureEnvelope());
    const fetchProbe = await evaluateJson(
      workerClient,
      `(async () => {
        async function request(label, url, init) {
          try {
            const response = await fetch(url, init);
            return {
              label,
              status: response.status,
              request_id: response.headers.get("X-Request-ID"),
              body: await response.json().catch(() => ({}))
            };
          } catch (error) {
            return {label, threw: String(error && error.message ? error.message : error)};
          }
        }
        return Promise.all([
          request("unauthenticated", ${receiverUrl} + "/v1/browser-captures", {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-Request-ID": "browser-smoke-reject"},
            body: JSON.stringify(${envelope})
          }),
          request("status", ${receiverUrl} + "/v1/status", {
            headers: {"Authorization": ${authHeader}, "X-Request-ID": "browser-smoke-status"}
          }),
          request("capture", ${receiverUrl} + "/v1/browser-captures", {
            method: "POST",
            headers: {"Content-Type": "application/json", "Authorization": ${authHeader}, "X-Request-ID": "browser-smoke-capture"},
            body: JSON.stringify(${envelope})
          })
        ]);
      })()`
    );
    workerClient.close();
    const [unauthenticated, status, capture] = fetchProbe;
    const summary = {
      ok:
        unauthenticated.status === 401 &&
        status.status === 200 &&
        capture.status === 202 &&
        Boolean(capture.body?.artifact_ref),
      chrome_binary: chromeBinary,
      chrome_args: chromeArgs,
      debugging_port: debuggingPort,
      extension_id: extensionId,
      extension_root: extensionRoot,
      manifest: {
        name: localManifest.name,
        version: localManifest.version,
        manifest_version: localManifest.manifest_version,
      },
      profile_dir: profileDir,
      receiver_base_url: receiverBaseUrl,
      service_worker_url: worker.url,
      unauthenticated,
      status,
      capture,
    };
    if (outputPath) writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    process.stdout.write(`${JSON.stringify(summary)}\n`);
    if (!summary.ok) process.exitCode = 1;
  } finally {
    await terminateProcess(chrome);
    if (!keepProfile && !process.env.POLYLOGUE_BROWSER_SMOKE_PROFILE_DIR) {
      rmSync(profileDir, { recursive: true, force: true });
    }
    if (stderr && process.exitCode) process.stderr.write(stderr);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message || error}\n`);
  process.exit(1);
});
