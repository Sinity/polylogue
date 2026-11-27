// Logical requests to Polylogue's native messaging host. Each port carries
// one message and its whole shared answer; large messages use ordered parts.
import { decode, encode } from "./base64.js";

export const HOST_NAME = "com.polylogue.browser_capture";
const CHUNK_BYTES = 64 * 1024;

export class HostError extends Error {
  constructor(code, status = null, message = code, reason = null) {
    super(message);
    this.code = code;
    this.status = status;
    this.reason = reason;
  }
}

/** One logical message, reassembled before the caller sees the answer. */
export function exchange(runtime, message) {
  return new Promise((resolve, reject) => {
    let port;
    try { port = runtime.connectNative(HOST_NAME); }
    catch { reject(new HostError("host_missing")); return; }
    const parts = [];
    let sequence = 0;
    let done = false;
    const finish = (error, value) => {
      if (done) return;
      done = true;
      port.onMessage.removeListener(onMessage);
      port.onDisconnect.removeListener(onDisconnect);
      try { port.disconnect(); } catch { /* Already closed. */ }
      if (error) reject(error); else resolve(value);
    };
    const onMessage = (frame) => {
      try {
        if (frame?.version !== 2) throw new HostError("native_version_mismatch", null, "Reload the extension to use this browser host.");
        if (frame.type === "part") {
          if (frame.sequence !== sequence || typeof frame.last !== "boolean") throw new HostError("host_answer_unreadable");
          parts.push(decode(frame.data));
          sequence += 1;
          if (!frame.last) return;
          const bytes = new Uint8Array(parts.reduce((size, part) => size + part.length, 0));
          let offset = 0;
          for (const part of parts) { bytes.set(part, offset); offset += part.length; }
          frame = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
        } else if (sequence) throw new HostError("host_answer_unreadable");
        if (frame.version !== 2 || frame.type !== "answer" || !frame.answer || typeof frame.answer.outcome !== "string") {
          throw new HostError("host_answer_unreadable");
        }
        finish(null, frame.answer);
      } catch (error) { finish(error instanceof HostError ? error : new HostError("host_answer_unreadable")); }
    };
    const onDisconnect = () => finish(new HostError(runtime.lastError ? "host_missing" : "host_stopped"));
    port.onMessage.addListener(onMessage);
    port.onDisconnect.addListener(onDisconnect);
    try {
      const value = { ...message, version: 2 };
      const bytes = new TextEncoder().encode(JSON.stringify(value));
      if (bytes.length <= CHUNK_BYTES) port.postMessage(value);
      else for (let at = 0; at < bytes.length; at += CHUNK_BYTES) {
        port.postMessage({ version: 2, type: "part", sequence: at / CHUNK_BYTES,
          data: encode(bytes.subarray(at, at + CHUNK_BYTES)), last: at + CHUNK_BYTES >= bytes.length });
      }
    } catch (error) { finish(error); }
  });
}

/** The three messages accepted by the host. Operations use the daemon's names. */
export class Host {
  constructor(runtime) { this.runtime = runtime; }

  async message(message) {
    const answer = await exchange(this.runtime, message);
    if (answer.outcome === "error") {
      const failure = answer.error;
      if (!failure?.code) throw new HostError("host_answer_unreadable");
      throw new HostError(failure.code, null, failure.message, failure.reason ?? null);
    }
    if (!["ok", "empty", "degraded"].includes(answer.outcome) || !("value" in answer)) throw new HostError("host_answer_unreadable");
    return answer.value;
  }

  call(operation, args = {}, request_id = crypto.randomUUID()) {
    return this.message({ type: "call", operation, arguments: args, request_id });
  }

  /** Stores the capture's exact JSON bytes; answers once they are synced. */
  async storeCapture(body) {
    return this.message({ type: "capture", capture: body instanceof Blob ? await body.text() : body });
  }

  spooled(provider, ids) { return this.message({ type: "spooled", provider, ids }); }
}
