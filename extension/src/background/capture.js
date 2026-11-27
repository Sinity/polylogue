// Capturing a conversation: its JSON from the provider (through the page),
// then one capture handed to the host. A capture goes into the outbox first
// and leaves it only when the host has synced it. Files the conversation
// names are downloaded after it is stored, never before: a large file
// downloads slowly, and the conversation must not wait for it. Once the
// files are in, the conversation is stored again with their bytes. A file
// whose bytes the archive already holds is named by its hash and not
// downloaded again.

import "../capture_state.js";
import { encode } from "./base64.js";
import { DownloadJob } from "./download_job.js";
import { ProviderScheduler } from "./provider.js";
import { askPage, PageError } from "./tabs.js";

const captureState = globalThis.polylogueCaptureState;
const KIND = "browser_llm_session";
// Bytes per base64 part: a multiple of 3, so the parts join into one
// valid base64 text.
const BASE64_PART_BYTES = 3 * 256 * 1024;

export class CaptureError extends Error {
  constructor(code, { status = null, retryAfter = null } = {}) {
    super(code);
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

/** A Blob's bytes as base64 parts, so no file becomes one huge string. */
async function base64Parts(blob) {
  const parts = [];
  for (let at = 0; at < blob.size; at += BASE64_PART_BYTES) {
    parts.push(encode(new Uint8Array(await blob.slice(at, at + BASE64_PART_BYTES).arrayBuffer())));
  }
  return parts;
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
export async function captureBody({ provider, conversation, url, raw = null, session = null, temporary = false, files = [], reason, version, revision, now = new Date() }) {
  const provenance = { captured_at: now.toISOString(), source_url: url, extension_version: version, reason };
  if (revision) provenance.extension_revision = revision;
  const head = { polylogue_capture_kind: KIND, schema_version: 1, provenance };
  // File bytes go in as Blob parts at a marker the JSON text cannot
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
    sessionParts.push(sessionText.slice(from, found.index), ...(await base64Parts(bodies[Number(found[1])])));
    from = found.index + found[0].length;
  }
  sessionParts.push(sessionText.slice(from));
  const opening = JSON.stringify(head).slice(0, -1);
  const payload = raw ? [',"raw_provider_payload":', raw] : [];
  return new Blob([opening, ...payload, ',"session":', ...sessionParts, "}"], { type: "application/json" });
}

/**
 * The conversation as content, for telling whether it changed: keys in
 * order, and ChatGPT's `safe_urls` in sorted order because the provider
 * reshuffles it on each read. Other arrays keep their meaningful order. A key whose
 * value is null counts as absent: claude.ai sends `"flags": null` on some
 * reads of an unchanged conversation and leaves it out on others.
 */
function content(value, key = null) {
  if (Array.isArray(value)) {
    const items = value.map((item) => content(item));
    return key === "safe_urls" && items.every((item) => item === null || typeof item !== "object")
      ? items.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)))
      : items;
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).filter((key) => value[key] !== null).sort()
      .map((key) => [key, content(value[key], key)]));
  }
  return value;
}

async function sha256(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fingerprint(raw, session) {
  return sha256(JSON.stringify(content(raw ? JSON.parse(await raw.text()) : session || {})));
}

export class Capturer {
  /**
   * `chrome` and `host` are the browser and the host; `outbox`
   * keeps captures until the host has them; `seen` remembers what each
   * conversation held when it was last captured across browser restarts; the daemon owns pending files.
   */
  constructor({ chrome, host, outbox, seen, scheduler = new ProviderScheduler(), log = () => {} }) {
    this.chrome = chrome;
    this.scheduler = scheduler;
    this.host = host;
    this.outbox = outbox;
    this.seen = seen;
    this.log = log;
    this.delivering = null;
    // The download job of each conversation whose files are waiting or
    // downloading, by `provider:conversation`. Downloads run one after
    // another.
    this.downloading = new Map();
    this.lastDownload = Promise.resolve();
    this.tabConversations = new Map();
  }

  /**
   * Captures the conversation `id` (or the tab's own) from `tabId`.
   * Answers `{outcome, provider, conversation, confirmedTurn}`: `stored`,
   * `waiting` (in the outbox until the host takes it) or `unchanged`;
   * `confirmedTurn` is the turn ID a ChatGPT snapshot confirms for
   * `confirm`, else null. Files the archive lacks download afterwards (see
   * `downloads`).
   */
  async capture(tabId, provider, options = {}) {
    const key = options.id ? `${provider}:${options.id}` : this.tabConversations.get(tabId);
    // A running download stores nothing until this capture has ended: it
    // may store a newer capture of the conversation.
    const downloading = this.downloading.get(key);
    downloading?.captureBegan();
    try { return await this.captureObserved(tabId, provider, options); }
    finally { downloading?.captureEnded(); }
  }

  /**
   * A tab's own conversation is first asked what its page shows. A sweep
   * whose page shows what it showed at the conversation's last capture,
   * which the archive holds, ends there without fetching from the provider.
   */
  async captureObserved(tabId, provider, options = {}) {
    const shown = options.id ? null : await this.shownOnPage(tabId);
    if (options.reason === "sweep" && !options.force && shown && (await this.shownUnchanged(provider, shown))) {
      return { outcome: "unchanged", provider, conversation: shown.conversation, confirmedTurn: null };
    }
    const done = await this.fetchAndStore(tabId, provider, options);
    // Read before the fetch, so the capture holds at least what the page showed.
    if (shown && shown.conversation === done.conversation) {
      await this.seen.set({ [`shown ${provider}:${done.conversation}`]: shown.digest });
    }
    return done;
  }

  /** The tab's conversation and a digest of what its page shows, or null. */
  async shownOnPage(tabId) {
    const page = await askPage(this.chrome, tabId, { type: "page" }).catch(() => null);
    if (page?.status !== 200 || !page.info?.id || typeof page.shown !== "string") return null;
    return { conversation: page.info.id, digest: await sha256(page.shown) };
  }

  async shownUnchanged(provider, shown) {
    const key = `${provider}:${shown.conversation}`;
    const known = await this.seen.get({ [`shown ${key}`]: null });
    if (known[`shown ${key}`] !== shown.digest) return false;
    const held = await this.held(provider, shown.conversation).catch(() => null);
    return held !== null && this.heldWhole(key, held);
  }

  /** The capture state of a conversation, as the host answers it. */
  async held(provider, conversation) {
    return captureState.ofArchiveAnswer(provider, await captureState.conversations(this.host, provider, [conversation]))[0];
  }

  /**
   * Whether a capture of the conversation is held and none of its files
   * is left for a later capture to download: they are all in, or their
   * download is running.
   */
  heldWhole(key, held) {
    return captureState.holdsCapture(held)
      && (held !== captureState.IN_ARCHIVE_FILES_DOWNLOADING || this.downloading.has(key));
  }

  async fetchAndStore(tabId, provider, { id = null, reason = "changed", force = false, confirm = null, before = null } = {}) {
    const capturedAt = new Date();
    const page = await this.scheduler.run(provider, () => askPage(this.chrome, tabId, { type: "conversation", id }), { background: reason === "import_job", owner: tabId, before });
    if (page.status !== 200) throw refused(page);
    const { info } = page;
    const conversation = info.id;
    if (!conversation) throw new CaptureError("no_conversation_on_page");
    const key = `${provider}:${conversation}`;
    this.tabConversations.set(tabId, key);
    const raw = page.body;
    // A capture without the provider's JSON or page turns holds no
    // conversation. Storing it would archive an empty session, and its
    // fingerprint (of nothing) would make every later read look unchanged.
    if (!raw && !(info.session?.turns?.length > 0)) throw new CaptureError("no_conversation_content");
    let confirmedTurn = null;
    if (confirm && provider === "chatgpt" && raw) {
      const snapshot = JSON.parse(await raw.text());
      const normalize = (text) => String(text || "").replace(/\s+/g, " ").trim();
      const message = Object.values(snapshot.mapping || {}).map((node) => node?.message)
        .find((message) => message?.id === confirm.turn && message.author?.role === "user"
          && normalize((message.content?.parts || []).filter((part) => typeof part === "string").join("\n")) === normalize(confirm.text));
      if (message && snapshot.conversation_id === conversation) confirmedTurn = message.id;
    }
    const contentFingerprint = await fingerprint(raw, info.session);
    const seenKey = `seen ${key}`;
    const known = await this.seen.get({ [seenKey]: null });
    let unchanged = !force && known[seenKey] === contentFingerprint;
    if (unchanged) {
      const held = await this.held(provider, conversation);
      if (!captureState.holdsCapture(held)) {
        unchanged = false;
      } else if (this.heldWhole(key, held)) {
        return { outcome: "unchanged", provider, conversation };
      }
    }

    const { version, version_name: revision } = this.chrome.runtime.getManifest();
    // Later file delivery must not make this conversation's text look newer.
    const base = { provider, conversation, url: info.url, raw, session: info.session, temporary: info.temporary, reason, version, revision, now: capturedAt };
    const files = [];
    const missing = [];
    if (raw) {
      let read;
      try {
        read = await captureState.captureFiles(this.host, await captureBody(base));
      } catch (error) {
        // The raw conversation must reach the outbox even when the host
        // cannot list its files. Try discovery again on the next capture.
        // A running download was for an older capture and this one's files
        // are unknown, so the downloaded bytes are not stored.
        if (!unchanged) this.downloading.get(key)?.replace(null);
        const outcome = unchanged ? "unchanged" : (await this.keep(base, [])) ? "stored" : "waiting";
        this.log(`Files of ${key} could not be listed: ${error.code || error.message}`);
        return { outcome, provider, conversation, confirmedTurn };
      }
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
    if (!unchanged || (raw && missing.length === 0)) {
      outcome = (await this.keep(base, files)) ? "stored" : "waiting";
      await this.seen.set({ [seenKey]: contentFingerprint });
    }
    if (missing.length || this.downloading.has(key)) this.download(tabId, key, base, files, missing);
    return { outcome, provider, conversation, confirmedTurn };
  }

  /** Puts one capture in the outbox and delivers; whether it was taken. */
  async keep(base, files, current = () => true) {
    const body = await captureBody({ ...base, files });
    if (!current()) return false;
    const key = await this.outbox.add({ provider: base.provider, conversation: base.conversation, body }, current);
    if (key === null) return false;
    return this.deliver();
  }

  /**
   * Downloads the files a stored capture lacks, then stores the
   * conversation again with their bytes. A conversation already waiting
   * for its files is not queued twice: the newer capture replaces the one
   * its job was downloading for.
   */
  download(tabId, key, base, files, missing) {
    const work = { tabId, base, files, missing };
    const running = this.downloading.get(key);
    if (running) {
      running.replace(work);
      return;
    }
    const job = new DownloadJob(work);
    job.finished = this.lastDownload.then(() => this.runDownloads(key, job));
    this.downloading.set(key, job);
    this.lastDownload = job.finished;
  }

  async runDownloads(key, job) {
    try {
      for (let work = job.startNext(); work; work = job.startNext()) {
        try {
          const fetched = await this.fetchFiles(work.tabId, work.base.conversation,
            work.missing.filter((file) => !job.available.has(file.id)), work.base.provider, work.base.reason === "import_job");
          for (const file of fetched) if (file.body) job.available.set(file.id, file);
          const complete = work.files.map((file) => job.available.get(file.id)
            || fetched.find((f) => f.id === file.id) || file);
          for (;;) {
            await job.capturesEnded();
            // A newer capture was stored during the download. Its files can
            // use the bytes just fetched, but its text must stay newest.
            if (job.replaced || work.missing.length === 0) break;
            job.startStoring();
            let interrupted = false;
            await this.keep({ ...work.base, reason: "files" }, complete, () => {
              if (!job.mayStore()) interrupted = true;
              return !interrupted;
            });
            // A capture that began before the bytes reached the outbox may
            // end without storing anything; the bytes are then stored after it.
            if (!interrupted) break;
          }
        } catch (error) {
          // A newer capture already queued still runs if this one fails.
          this.log(`Files of ${key} were not stored: ${error.code || error.message}`);
        }
      }
    } finally {
      job.end();
      this.downloading.delete(key);
    }
  }

  /** Resolves once every download running now has finished. */
  async downloads() {
    await Promise.all([...this.downloading.values()].map((job) => job.finished));
  }

  /** The bytes of each listed file, or the reason it has none. */
  async fetchFiles(tabId, conversation, listed, provider, background = false) {
    const files = [];
    for (const file of listed) {
      try {
        const answer = await this.scheduler.run(provider, () => askPage(this.chrome, tabId, { type: "file", file, conversation }), { background, owner: tabId });
        if (answer.status === 429 || answer.status >= 500) throw refused(answer);
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
        if ((error instanceof PageError && error.code === "page_closed") || error.until || error.status === 429 || error.status >= 500) throw error;
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
