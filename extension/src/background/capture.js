// Capturing a conversation: its JSON from the provider (through the page),
// then one capture handed to the host. A capture goes into the outbox first
// and leaves it only when the host has synced it. Files the conversation
// names are downloaded after it is stored, never before: a large file
// downloads slowly, and the conversation must not wait for it. Once the
// files are in, the conversation is stored again with their bytes. A file
// whose bytes the archive already holds is named by its hash and not
// downloaded again.

import { askPage, PageError } from "./tabs.js";

const KIND = "browser_llm_session";
// Bytes per base64 piece: a multiple of 3, so the pieces join into one
// valid base64 text.
const BASE64_PIECE_BYTES = 3 * 256 * 1024;

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

/** A Blob's bytes as base64 pieces, so no file becomes one huge string. */
async function base64Pieces(blob) {
  const pieces = [];
  for (let at = 0; at < blob.size; at += BASE64_PIECE_BYTES) {
    pieces.push(base64(new Uint8Array(await blob.slice(at, at + BASE64_PIECE_BYTES).arrayBuffer())));
  }
  return pieces;
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
 * Blob) placed whole so it is never parsed and written again here. Each
 * file is given with its bytes (`body`, a Blob), as stored bytes named by
 * their hash (`blob`), or as not fetched with the reason (`error`).
 */
export async function captureBody({ provider, conversation, url, raw = null, session = null, temporary = false, files = [], reason, version, now = new Date() }) {
  const provenance = { captured_at: now.toISOString(), source_url: url, extension_version: version, reason };
  const head = { polylogue_capture_kind: KIND, schema_version: 1, provenance };
  // File bytes go in as Blob pieces at a marker the JSON text cannot
  // otherwise hold.
  const marker = `polylogue-file-${crypto.randomUUID()}`;
  const bodies = [];
  const attachments = files.map((file) => {
    const entry = {
      provider_attachment_id: file.id,
      message_provider_id: file.message,
      name: file.name ?? null,
      mime_type: file.media_type ?? null,
    };
    if (file.body) {
      entry.content_base64 = `${marker}-${bodies.length}`;
      bodies.push(file.body);
    } else if (file.blob) {
      entry.blob = file.blob;
      if (Number.isFinite(file.size)) entry.size_bytes = file.size;
    } else {
      entry.not_fetched = file.error || "unknown";
    }
    return entry;
  });
  const sessionPart = {
    ...(session || {}),
    provider,
    provider_session_id: conversation,
    session_kind: temporary ? "temporary" : "standard",
    attachments,
  };
  const sessionParts = [];
  const pattern = new RegExp(`${marker}-(\\d+)`, "g");
  const sessionText = JSON.stringify(sessionPart);
  let from = 0;
  for (const found of sessionText.matchAll(pattern)) {
    sessionParts.push(sessionText.slice(from, found.index), ...(await base64Pieces(bodies[Number(found[1])])));
    from = found.index + found[0].length;
  }
  sessionParts.push(sessionText.slice(from));
  const opening = JSON.stringify(head).slice(0, -1);
  const payload = raw ? [',"raw_provider_payload":', raw] : [];
  return new Blob([opening, ...payload, ',"session":', ...sessionParts, "}"], { type: "application/json" });
}

/**
 * The conversation as content, for telling whether it changed: keys in
 * order, and lists of plain values in order too, since providers return
 * some (ChatGPT's `safe_urls`) in a new order on every read. A key whose
 * value is null counts as absent: claude.ai sends `"flags": null` on some
 * reads of an unchanged conversation and leaves it out on others.
 */
function content(value) {
  if (Array.isArray(value)) {
    const items = value.map(content);
    return items.every((item) => item === null || typeof item !== "object")
      ? items.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)))
      : items;
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).filter((key) => value[key] !== null).sort()
      .map((key) => [key, content(value[key])]));
  }
  return value;
}

async function fingerprint(raw, session) {
  const value = raw ? JSON.parse(await raw.text()) : session || {};
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(content(value))));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export class Capturer {
  /**
   * `chrome` and `host` are the browser and the host; `outbox`
   * keeps captures until the host has them; `seen` remembers what each
   * conversation held when it was last captured (a chrome.storage area
   * that outlives restarts, so a restart does not capture it again).
   */
  constructor({ chrome, host, outbox, seen, log = () => {} }) {
    this.chrome = chrome;
    this.host = host;
    this.outbox = outbox;
    this.seen = seen;
    this.log = log;
    this.delivering = null;
    // Conversations whose files are waiting or downloading, by
    // `provider:conversation`. Downloads run one after another.
    this.downloading = new Map();
    this.lastDownload = Promise.resolve();
  }

  /**
   * Captures the conversation `id` (or the tab's own) from `tabId`.
   * Answers `{outcome, provider, conversation}`: `stored`, `waiting` (in
   * the outbox until the host takes it) or `unchanged`. Files the archive
   * lacks download afterwards (see `downloads`).
   */
  async capture(tabId, provider, { id = null, reason = "changed", force = false } = {}) {
    const page = await askPage(this.chrome, tabId, { type: "conversation", id });
    if (page.status !== 200) throw refused(page);
    const { info } = page;
    const conversation = info.id;
    if (!conversation) throw new CaptureError("no_conversation_on_page");
    const key = `${provider}:${conversation}`;
    const raw = page.body;
    const now = await fingerprint(raw, info.session);
    const seenKey = `seen ${key}`;
    const filesKey = `files ${key}`;
    const known = await this.seen.get({ [seenKey]: null, [filesKey]: false });
    const unchanged = !force && known[seenKey] === now;
    // An unchanged conversation is looked at again only while files it
    // names are still to be downloaded (after a restart cut a download off).
    if (unchanged && !(known[filesKey] && !this.downloading.has(key))) {
      return { outcome: "unchanged", provider, conversation };
    }

    const version = this.chrome.runtime.getManifest().version;
    const base = { provider, conversation, url: info.url, raw, session: info.session, temporary: info.temporary, reason, version };
    const files = [];
    const missing = [];
    if (raw) {
      const read = await this.host.captureFiles(await captureBody(base));
      for (const file of read.files || []) {
        if (file.inline) continue;
        if (file.stored) {
          files.push({ ...file, blob: file.stored.blob, size: file.stored.size });
        } else {
          files.push({ ...file, error: "downloading" });
          missing.push(file);
        }
      }
    }
    let outcome = "unchanged";
    if (!unchanged) {
      outcome = (await this.keep(base, files)) ? "stored" : "waiting";
      await this.seen.set({ [seenKey]: now });
    }
    await this.seen.set({ [filesKey]: missing.length > 0 });
    if (missing.length) this.download(tabId, key, base, files, missing);
    return { outcome, provider, conversation };
  }

  /** Puts one capture in the outbox and delivers; whether it was taken. */
  async keep(base, files) {
    const body = await captureBody({ ...base, files });
    await this.outbox.add({ provider: base.provider, conversation: base.conversation, body });
    return this.deliver();
  }

  /**
   * Downloads the files a stored capture lacks, then stores the
   * conversation again with their bytes. A conversation already waiting
   * for its files is not queued twice.
   */
  download(tabId, key, base, files, missing) {
    if (this.downloading.has(key)) return;
    const work = this.lastDownload
      .then(async () => {
        const fetched = await this.fetchFiles(tabId, base.conversation, missing);
        if (fetched.some((file) => file.body)) {
          const byId = new Map(fetched.map((file) => [file.id, file]));
          await this.keep({ ...base, reason: "files" }, files.map((file) => byId.get(file.id) || file));
        }
        // Files that could not be downloaded are tried again when the
        // conversation next changes.
        await this.seen.set({ [`files ${key}`]: false });
      })
      .catch((error) => this.log(`Files of ${key} were not stored: ${error.code || error.message}`))
      .finally(() => this.downloading.delete(key));
    this.downloading.set(key, work);
    this.lastDownload = work;
  }

  /** Resolves once every download running now has finished. */
  async downloads() {
    await Promise.all([...this.downloading.values()]);
  }

  /** The bytes of each listed file, or the reason it has none. */
  async fetchFiles(tabId, conversation, listed) {
    const files = [];
    for (const file of listed) {
      try {
        const answer = await askPage(this.chrome, tabId, { type: "file", file, conversation });
        if (answer.status !== 200 || !answer.body) {
          files.push({ ...file, error: answer.error || `http_${answer.status}` });
          this.log(`File ${file.name || file.id} of ${conversation} could not be downloaded: ${answer.error || answer.status}`);
          continue;
        }
        files.push({
          ...file,
          error: null,
          name: file.name || answer.info?.name,
          media_type: file.media_type || answer.info?.media_type,
          body: answer.body,
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
