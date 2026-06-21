#!/usr/bin/env node
// Exercise the extension service-worker receiver path against a real local
// Polylogue browser-capture receiver. This is intentionally runnable without a
// GUI browser: it imports src/background.js with a small Chrome API mock and
// sends the same runtime messages the popup/content scripts use.

import { writeFileSync } from "node:fs";

const receiverBaseUrl = (process.env.POLYLOGUE_EXTENSION_RECEIVER_URL || "http://127.0.0.1:8765").replace(/\/+$/, "");
const receiverAuthToken = process.env.POLYLOGUE_EXTENSION_RECEIVER_TOKEN || "";
const outputPath = process.env.POLYLOGUE_EXTENSION_SMOKE_OUT || "";

let messageListener = null;
let stored = {
  receiverAuthToken: "",
  receiverBaseUrl,
};

globalThis.chrome = {
  action: {
    setBadgeBackgroundColor: async () => undefined,
    setBadgeText: async () => undefined,
  },
  runtime: {
    onMessage: {
      addListener: (fn) => {
        messageListener = fn;
      },
    },
  },
  storage: {
    local: {
      get: async (defaults) => ({ ...defaults, ...stored }),
      set: async (patch) => {
        stored = { ...stored, ...patch };
      },
    },
  },
};

await import(`../src/background.js?dev-loop-smoke=${Date.now()}`);
if (typeof messageListener !== "function") {
  throw new Error("background listener was not registered");
}

function fixtureEnvelope() {
  return {
    polylogue_capture_kind: "browser_llm_session",
    schema_version: 1,
    provenance: {
      source_url: "https://chatgpt.com/c/dev-loop-extension-smoke",
      page_title: "Polylogue extension dev-loop smoke",
      captured_at: "2026-06-21T00:00:00+00:00",
      adapter_name: "dev-loop-extension-smoke",
    },
    session: {
      provider: "chatgpt",
      provider_session_id: "dev-loop-extension-smoke",
      title: "Polylogue extension dev-loop smoke",
      turns: [{ provider_turn_id: "turn-1", role: "user", text: "extension smoke" }],
    },
  };
}

async function sendRuntimeMessage(message) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const keepAlive = messageListener(message, {}, (payload) => {
      settled = true;
      resolve(payload);
    });
    if (keepAlive !== true) {
      reject(new Error(`runtime listener did not keep channel alive for ${message.type}`));
      return;
    }
    setTimeout(() => {
      if (!settled) reject(new Error(`runtime message timed out: ${message.type}`));
    }, 5000);
  });
}

const rejected = await sendRuntimeMessage({
  type: "polylogue.capture",
  envelope: fixtureEnvelope(),
});
if (rejected.ok !== false || !rejected.receiver_request_id) {
  throw new Error(`expected unauthenticated rejection with receiver request id, got ${JSON.stringify(rejected)}`);
}

const configured = await sendRuntimeMessage({
  type: "polylogue.configureReceiver",
  receiverBaseUrl,
  receiverAuthToken,
});
if (configured.ok !== true || configured.authConfigured !== Boolean(receiverAuthToken)) {
  throw new Error(`receiver configuration failed: ${JSON.stringify(configured)}`);
}

const status = await sendRuntimeMessage({ type: "polylogue.status" });
if (status.ok !== true || !status.receiver_request_id) {
  throw new Error(`receiver status failed: ${JSON.stringify(status)}`);
}

const capture = await sendRuntimeMessage({
  type: "polylogue.capture",
  envelope: fixtureEnvelope(),
});
if (capture.ok !== true || capture.provider !== "chatgpt" || !capture.artifact_ref || !capture.receiver_request_id) {
  throw new Error(`receiver capture failed: ${JSON.stringify(capture)}`);
}

const summary = {
  ok: true,
  receiver_base_url: receiverBaseUrl,
  unauthenticated: {
    ok: rejected.ok,
    error: rejected.error,
    receiver_request_id: rejected.receiver_request_id,
  },
  status: {
    receiver_request_id: status.receiver_request_id,
  },
  capture: {
    artifact_ref: capture.artifact_ref,
    bytes_written: capture.bytes_written,
    provider: capture.provider,
    provider_session_id: capture.provider_session_id,
    receiver_request_id: capture.receiver_request_id,
  },
  stored_state: stored.polylogueState || null,
};

if (outputPath) {
  writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
}
process.stdout.write(`${JSON.stringify(summary)}\n`);
