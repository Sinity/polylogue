// Capturing a conversation: its JSON from the provider (through the page),
// the files it names, then one capture handed to the host. A capture goes
// into the outbox first and leaves it only when the host has synced it.

import { askPage, PageError } from "./tabs.js";

const KIND = "browser_llm_session";

export class CaptureError extends Error {
  constructor(code, { status = null, retryAfter = null } = {}) {
    super(code);
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function base64(bytes) {
  let text = "";
  for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(text);
}

/** A page answer that is not a success, as an error. */
function refused(answer) {
  return new CaptureError(answer.error || `provider_http_${answer.status}`, {
    status: answer.status,
    retryAfter: answer.retry_after ?? null,
  });
}

/**
 * The capture as JSON, with the provider's conversation JSON (`raw`, a
 * Blob) placed whole so it is never parsed and written again here.
 */
export function captureBody({ provider, conversation, url, raw = null, session = null, temporary = false, files = [], reason, version, now = new Date() }) {
  const provenance = { captured_at: now.toISOString(), source_url: url, extension_version: version, reason };
  const head = { polylogue_capture_kind: KIND, schema_version: 1, provenance };
  const attachments = files.map((file) => {
    const entry = {
      provider_attachment_id: file.id,
      message_provider_id: file.message,
      name: file.name ?? null,
      mime_type: file.media_type ?? null,
    };
    if (file.bytes) entry.content_base64 = base64(file.bytes);
    else entry.not_fetched = file.error || "unknown";
    return entry;
  });
  const sessionPart = {
    ...(session || {}),
    provider,
    provider_session_id: conversation,
    session_kind: temporary ? "temporary" : "standard",
    attachments,
  };
  if (!raw) return new Blob([JSON.stringify({ ...head, session: sessionPart })], { type: "application/json" });
  const opening = JSON.stringify(head).slice(0, -1);
  return new Blob([opening, ',"raw_provider_payload":', raw, ',"session":', JSON.stringify(sessionPart), "}"], {
    type: "application/json",
  });
}

async function sha256(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export class Capturer {
  /**
   * `chrome` and `host` are the browser and the native host; `outbox`
   * keeps captures until the host has them; `seen` remembers each
   * conversation's last stored provider JSON (a chrome.storage area).
   */
  constructor({ chrome, host, outbox, seen, log = () => {} }) {
    this.chrome = chrome;
    this.host = host;
    this.outbox = outbox;
    this.seen = seen;
    this.log = log;
    this.delivering = null;
  }

  /**
   * Captures the conversation `id` (or the tab's own) from `tabId`.
   * Answers `{outcome: "stored" | "unchanged", provider, conversation}`.
   */
  async capture(tabId, provider, { id = null, reason = "changed", force = false } = {}) {
    const page = await askPage(this.chrome, tabId, { type: "conversation", id });
    if (page.status !== 200) throw refused(page);
    const { info } = page;
    const conversation = info.id;
    if (!conversation) throw new CaptureError("no_conversation_on_page");
    const key = `${provider}:${conversation}`;
    const raw = page.body;
    const fingerprint = await sha256(raw || new Blob([JSON.stringify(info.session || {})]));
    const known = (await this.seen.get({ [key]: null }))[key];
    if (!force && known === fingerprint) return { outcome: "unchanged", provider, conversation };

    const version = this.chrome.runtime.getManifest().version;
    const base = { provider, conversation, url: info.url, raw, session: info.session, temporary: info.temporary, reason, version };
    let files = [];
    if (raw) {
      const read = await this.host.captureFiles(captureBody(base));
      files = await this.fetchFiles(tabId, conversation, read.files || []);
    }
    const body = captureBody({ ...base, files });
    await this.outbox.add({ provider, conversation, body });
    await this.deliver();
    await this.seen.set({ [key]: fingerprint });
    return { outcome: "stored", provider, conversation };
  }

  /** The bytes of each file the provider's JSON does not already hold. */
  async fetchFiles(tabId, conversation, listed) {
    const files = [];
    for (const file of listed) {
      if (file.inline) continue;
      try {
        const answer = await askPage(this.chrome, tabId, { type: "file", file, conversation });
        if (answer.status !== 200 || !answer.body) {
          files.push({ ...file, error: answer.error || `http_${answer.status}` });
          continue;
        }
        files.push({
          ...file,
          name: file.name || answer.info?.name,
          media_type: file.media_type || answer.info?.media_type,
          bytes: new Uint8Array(await answer.body.arrayBuffer()),
        });
      } catch (error) {
        if (error instanceof PageError && error.code === "page_closed") throw error;
        files.push({ ...file, error: String(error.code || error.message || error) });
      }
    }
    return files;
  }

  /**
   * Hands every waiting capture to the host, oldest first, and answers
   * whether the outbox is empty. Stops at the first refusal; the rest wait
   * for the next try. A capture added while this runs is sent in the same
   * run.
   */
  deliver() {
    this.again = true;
    if (!this.delivering) {
      this.delivering = (async () => {
        try {
          while (this.again) {
            this.again = false;
            if (!(await this.deliverWaiting())) return false;
          }
          return true;
        } finally {
          this.delivering = null;
        }
      })();
    }
    return this.delivering;
  }

  async deliverWaiting() {
    for (const waiting of await this.outbox.all()) {
      try {
        await this.host.storeCapture(waiting.body);
      } catch (error) {
        await this.outbox.failed(waiting.key, error.code || error.message);
        this.log(`Could not store a ${waiting.provider} capture: ${error.message}`);
        return false;
      }
      await this.outbox.remove(waiting.key);
    }
    return true;
  }
}
