// Polylogue's native messaging host, `com.polylogue.browser_capture`, which
// is the `polylogue` binary. Chrome starts one host process per port. One
// exchange: `request` (method, path, headers), the body as `body` frames of
// at most 64 KiB, each acknowledged with `body_ack`, then `body_end`; the
// host answers `response` (status, headers), then one `response_body` frame
// per `response_next` until `response_end`. A refusal is an `error` frame.

export const HOST_NAME = "com.polylogue.browser_capture";
const APPENDED_BYTES = 64 * 1024;

export class HostError extends Error {
  constructor(code, status = null, message = code) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

function encode(bytes) {
  let text = "";
  for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(text);
}

function decode(data) {
  return Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
}

/** One request to the host; resolves `{status, headers, body: Blob}`. */
export function exchange(runtime, method, path, body = null) {
  return new Promise((resolve, reject) => {
    let port;
    try {
      port = runtime.connectNative(HOST_NAME);
    } catch {
      reject(new HostError("host_missing"));
      return;
    }
    const content = body === null ? null : body instanceof Blob ? body : new Blob([body]);
    let sent = 0;
    let answer = null;
    const responseBytes = [];
    let done = false;
    const finish = (error, value) => {
      if (done) return;
      done = true;
      port.onMessage.removeListener(onMessage);
      port.onDisconnect.removeListener(onDisconnect);
      try { port.disconnect(); } catch { /* Already closed. */ }
      if (error) reject(error); else resolve(value);
    };
    const sendNext = async () => {
      if (!content || sent >= content.size) {
        port.postMessage({ type: "body_end" });
        return;
      }
      const appendedBytes = new Uint8Array(await content.slice(sent, sent + APPENDED_BYTES).arrayBuffer());
      port.postMessage({ type: "body", sequence: Math.floor(sent / APPENDED_BYTES), data: encode(appendedBytes) });
      sent += appendedBytes.length;
    };
    const onMessage = (frame) => {
      if (frame?.type === "error") return finish(new HostError(frame.error));
      if (frame?.type === "body_ack") return void sendNext().catch((e) => finish(e));
      if (frame?.type === "response" && !answer) {
        answer = { status: frame.status, headers: frame.headers || {} };
        port.postMessage({ type: "response_next" });
        return;
      }
      if (frame?.type === "response_body" && answer) {
        responseBytes.push(decode(frame.data));
        port.postMessage({ type: "response_next" });
        return;
      }
      if (frame?.type === "response_end" && answer) {
        return finish(null, { ...answer, body: new Blob(responseBytes) });
      }
      finish(new HostError("host_answer_unreadable"));
    };
    const onDisconnect = () => {
      // Reading lastError tells a missing host from one that stopped.
      finish(new HostError(runtime.lastError ? "host_missing" : "host_stopped"));
    };
    port.onMessage.addListener(onMessage);
    port.onDisconnect.addListener(onDisconnect);
    port.postMessage({ type: "request", version: 1, method, path, headers: { "content-type": "application/json" } });
    void sendNext().catch((e) => finish(e));
  });
}

/** The host's routes, as the background uses them. */
export class Host {
  constructor(runtime) {
    this.runtime = runtime;
  }

  async json(method, path, body = null) {
    const sent = body === null || body instanceof Blob ? body : JSON.stringify(body);
    const answer = await exchange(this.runtime, method, path, sent);
    const value = await answer.body.text().then((text) => (text ? JSON.parse(text) : null)).catch(() => null);
    if (answer.status >= 400) {
      throw new HostError(value?.error || `host_http_${answer.status}`, answer.status, value?.message || value?.error);
    }
    return value;
  }

  status() { return this.json("GET", "/v1/status"); }
  reader() { return this.json("GET", "/v1/reader"); }
  /** Stores a capture; answers once it is synced to disk. */
  storeCapture(body) { return this.json("POST", "/v1/captures", body); }
  /** The conversation a capture holds and the files it names. */
  captureFiles(body) { return this.json("POST", "/v1/captures/files", body); }
  conversations(provider, ids) { return this.json("POST", "/v1/conversations", { provider, ids }); }
  saveNote(note) { return this.json("POST", "/v1/notes", note); }
  chatActions() { return this.json("GET", "/v1/chat-actions"); }
  claimChatAction(by) { return this.json("GET", `/v1/chat-actions?claim_by=${encodeURIComponent(by)}`); }
  reportChatAction(id, update) { return this.json("POST", `/v1/chat-actions/${encodeURIComponent(id)}/report`, update); }
  decideChatAction(id, by, approve) {
    return this.json("POST", `/v1/chat-actions/${encodeURIComponent(id)}/decision`, { by, approve });
  }
  async chatAttachment(id, attachment) {
    const answer = await exchange(this.runtime, "GET",
      `/v1/chat-actions/${encodeURIComponent(id)}/attachments/${encodeURIComponent(attachment)}`);
    if (answer.status >= 400) throw new HostError(`host_http_${answer.status}`, answer.status);
    return answer.body;
  }
  backfillJobs() { return this.json("GET", "/v1/backfill"); }
  claimBackfill(by) { return this.json("POST", "/v1/backfill/claim", { by }); }
  reportBackfill(id, by, report) { return this.json("POST", `/v1/backfill/${encodeURIComponent(id)}/report`, { by, ...report }); }
}
