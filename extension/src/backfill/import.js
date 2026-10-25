import JSONParser from "../vendor/streamparser-json/jsonparser.js";
import { createSha256 } from "../vendor/sha256.js";

const fileId = /^[a-f0-9-]+$/;
const referenceFields = new Set(["id", "body_ref", "raw_ref", "stage_id", "record_ref", "capture_record_ref", "capture_bundle_ref", "staged_raw_json", "staged_turns", "staged_session_attachments", "staged_asset_failures", "job_id"]);
function rewrite(value, ids, tokens, key = "") {
  if (Array.isArray(value)) return value.map((child, index) => rewrite(child, ids, tokens,
    ["source_refs", "capture_source_refs", "asset_refs", "assets"].includes(key) || (key === "acquisition" && index === 1) ? "id" : ""));
  if (!value || typeof value !== "object") return typeof value === "string" && referenceFields.has(key) ? ids.get(value) || value : value;
  const result = Object.fromEntries(Object.entries(value).map(([field, child]) => [field, rewrite(child, ids, tokens, field)]));
  if (tokens.has(value.id) && typeof value.token === "string") result.token = tokens.get(value.id);
  return result;
}
function validateFile(evidence) {
  const meta = evidence?.metadata;
  if (!meta || !fileId.test(meta.id || "") || !Number.isSafeInteger(meta.bytes) || meta.bytes < 0 || !Array.isArray(evidence.parts)) throw new Error("checkpoint_import_file_invalid");
  if (evidence.parts.length !== (meta.state === "acquiring" ? meta.sequence : 1)) throw new Error("checkpoint_import_part_order_invalid");
  const hash = createSha256(); let bytes = 0;
  for (const [index, part] of evidence.parts.entries()) {
    if (part.sequence !== index || !Number.isSafeInteger(part.size_bytes) || part.size_bytes < 0 || typeof part.inline_base64 !== "string") throw new Error("checkpoint_import_part_invalid");
    const text = atob(part.inline_base64); const data = Uint8Array.from(text, c => c.charCodeAt(0));
    if (data.length !== part.size_bytes || createSha256().update(data).hex() !== part.sha256) throw new Error("checkpoint_import_part_corrupt");
    hash.update(data); bytes += data.length;
  }
  if (bytes !== evidence.size_bytes || bytes !== meta.bytes || hash.hex() !== evidence.sha256 ||
      (meta.state !== "acquiring" && hash.hex() !== meta.sha256)) throw new Error("checkpoint_import_file_corrupt");
  return meta;
}

/** Import the existing export format. Publication is one paused IDB transaction.
 * The tokenizer retains one acquired entry at a time, including that entry's
 * encoded file bytes; it does not promise a fixed memory bound for a large file.
 */
export async function importCheckpoint(file, { staging, store, signal = null }) {
  const hash = createSha256(); const reader = file.stream().getReader();
  const shape = {}; const ledger = { jobs: [], queue: [], revisions: [] }; const acquired = [];
  const parser = new JSONParser({ paths: ["$.schema", "$.ledger.version", "$.ledger.jobs", "$.ledger.jobs.*", "$.ledger.queue", "$.ledger.queue.*", "$.ledger.revisions", "$.ledger.revisions.*", "$.acquired", "$.acquired.*"], keepStack: false, stringBufferSize: 64 * 1024 });
  const pending = [];
  parser.onValue = ({ key, value, parent, stack }) => {
    if (stack.length === 1 && key === "schema") shape.schema = value;
    else if (stack.length === 2 && stack[1].key === "ledger") shape[key] = key === "version" ? value : Array.isArray(value) ? "array" : "invalid";
    else if (stack.length === 1 && key === "acquired") shape.acquired = Array.isArray(value) ? "array" : "invalid";
    else if (stack.length === 3 && ["jobs", "queue", "revisions"].includes(stack[2].key)) {
      ledger[stack[2].key].push(value); parent.length = 0;
    } else if (stack.length === 2 && stack[1].key === "acquired") { pending.push(value); parent.length = 0; }
  };
  const staged = []; const ids = new Map(); const tokens = new Map(); const fileMetadata = new Map();
  const cancel = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal?.addEventListener("abort", cancel, { once: true });
  async function accept(entry) {
    if (!entry || typeof entry.queue_id !== "string" || !Array.isArray(entry.files) || (entry.records && !Array.isArray(entry.records))) throw new Error("checkpoint_import_acquired_invalid");
    const files = [];
    for (const evidence of entry.files) {
      signal?.throwIfAborted();
      const meta = validateFile(evidence);
      const prior = fileMetadata.get(meta.id);
      if (prior) { if (prior.sha256 !== evidence.sha256 || prior.bytes !== meta.bytes) throw new Error("checkpoint_import_duplicate_file"); continue; }
      const ref = await staging.begin(meta.owner || {}, { kind: "checkpoint-import-evidence" });
      staged.push(ref.id); ids.set(meta.id, ref.id); tokens.set(meta.id, ref.token);
      await staging.save({ ...await staging.metadata(ref.id), import_staged: true });
      for (const part of evidence.parts) {
        signal?.throwIfAborted();
        // Existing append owns immutable parts and their replay checks.
        await staging.append(ref, meta.owner || {}, part.sequence, part.inline_base64);
      }
      if (meta.state !== "acquiring") await staging.seal(ref, meta.owner || {});
      fileMetadata.set(meta.id, { ...meta, sha256: evidence.sha256 }); files.push(meta.id);
    }
    acquired.push({ ...entry, files: files.length });
  }
  try {
    for (;;) {
      signal?.throwIfAborted(); const chunk = await reader.read(); if (chunk.done) break;
      hash.update(chunk.value);
      for (let offset = 0; offset < chunk.value.length; offset += 48 * 1024) {
        parser.write(chunk.value.subarray(offset, offset + 48 * 1024));
        for (const entry of pending.splice(0)) await accept(entry);
      }
    }
    if (!parser.isEnded) parser.end();
    for (const entry of pending.splice(0)) await accept(entry);
    signal?.throwIfAborted();
    if (shape.schema !== "polylogue.backfill-export.v2" || shape.version !== 1 || shape.acquired !== "array" ||
        !["jobs", "queue", "revisions"].every(k => shape[k] === "array") || ledger.jobs.length !== 1) throw new Error("checkpoint_import_shape_invalid");
    const job = ledger.jobs[0];
    if (typeof job.id !== "string" || !job.id || typeof job.provider !== "string" || typeof job.account_scope !== "string" || !job.account_scope) throw new Error("checkpoint_import_job_invalid");
    const prefix = `import:${hash.hex()}:`;
    if (ids.has(job.id)) throw new Error("checkpoint_import_id_collision");
    ids.set(job.id, prefix + job.id);
    for (const item of ledger.queue) {
      if (typeof item.id !== "string" || item.job_id !== job.id || item.provider !== job.provider) throw new Error("checkpoint_import_queue_invalid");
      if (ids.has(item.id)) throw new Error("checkpoint_import_id_collision");
      ids.set(item.id, prefix + item.id);
    }
    for (const entry of acquired) {
      if (!ledger.queue.some(item => item.id === entry.queue_id)) throw new Error("checkpoint_import_queue_missing");
      if (entry.capture) {
        if (typeof entry.capture.id !== "string" || entry.capture.provider !== job.provider) throw new Error("checkpoint_import_capture_invalid");
        // Typed captures may deliberately share their body file's ID.
        if (!ids.has(entry.capture.id)) ids.set(entry.capture.id, prefix + entry.capture.id);
      }
    }
    const requireFiles = refs => {
      for (const ref of refs || []) if (typeof ref !== "string" || !fileId.test(ref) || !fileMetadata.has(ref)) throw new Error("checkpoint_import_file_missing");
    };
    const requireRef = ref => { if (ref != null) requireFiles([typeof ref === "string" ? ref : ref.id]); };
    const requireAttachments = attachments => { for (const attachment of attachments || []) requireRef(attachment.staged_asset); };
    for (const meta of fileMetadata.values()) {
      requireFiles(meta.source_refs); requireFiles(meta.assets);
      if (meta.record_ref && !ids.has(meta.record_ref)) throw new Error("checkpoint_import_record_missing");
    }
    const captures = [{ id: prefix + "backup", kind: "checkpoint-import", source_refs: staged, digest: hash.hex() }];
    const records = [];
    for (const value of ledger.revisions) {
      if (typeof value.id !== "string" || value.provider !== job.provider) throw new Error("checkpoint_import_revision_invalid");
      captures.push({ id: prefix + "revision:" + value.id, kind: "imported-revision", value });
    }
    for (const entry of acquired) {
      if (entry.capture) {
        requireFiles(entry.capture.source_refs);
        requireRef(entry.capture.raw_ref);
        for (const ref of Object.values(entry.capture.related_refs || {})) requireRef(ref);
        const capture = rewrite(entry.capture, ids, tokens);
        delete capture.delivery_kind;
        captures.push(capture);
        for (const record of entry.records || []) {
          if (record.capture_id !== entry.capture.id) throw new Error("checkpoint_import_record_invalid");
          requireFiles(record.asset_refs);
          requireAttachments(record.turn?.attachments);
          requireAttachments(record.session_attachment ? [record.session_attachment] : []);
          records.push({ ...rewrite(record, ids, tokens), capture_id: capture.id });
        }
      }
    }
    const queue = ledger.queue.map(value => {
      requireFiles(value.source_refs); requireFiles(value.capture_source_refs);
      const row = rewrite(value, ids, tokens); delete row.envelope?.receiver_native;
      row.claimed_by = null; row.claim_expires_ms = null;
      if (row.body_ref && !staged.includes(row.body_ref)) throw new Error("checkpoint_import_file_missing");
      const entry = acquired.find(e => e.queue_id === value.id);
      if ((row.capture_record_ref || row.record_ref) && !entry?.records && !row.body_ref) {
        row.state = "recovery_required"; row.last_error = "checkpoint_import_records_missing";
      }
      return row;
    });
    // All files are verified before the database can refer to them. Imported
    // metadata cannot recreate automatic native-acquisition work at startup.
    for (const [original, meta] of fileMetadata) {
      const ref = ids.get(original); const current = await staging.metadata(ref);
      await staging.save({ ...rewrite(meta, ids, tokens), id: ref, token: current.token,
        kind: "checkpoint-import-evidence", imported_kind: meta.kind, import_staged: true });
    }
    signal?.throwIfAborted();
    const result = await store.commitImportedCheckpoint({ job: { ...rewrite(job, ids, tokens), status: "paused",
      cooldown_reason: "checkpoint_import_review_required", execution_owner: null, execution_expires_at_ms: null }, queue, revisions: ledger.revisions, captures, records }, signal);
    return { ...result, digest: hash.hex() };
  } catch (error) {
    for (const id of staged) await staging.discard(id).catch(() => undefined);
    throw error;
  } finally { signal?.removeEventListener("abort", cancel); await reader.cancel().catch(() => undefined); reader.releaseLock(); }
}
