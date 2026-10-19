import { createSha256 } from "../vendor/sha256.js";
import { receiverAckContractError } from "../backfill/models.js";
import { canonicalJson } from "../backfill/capture_jobs.js";

const CHUNK_BYTES = 48 * 1024;
const encoder = new TextEncoder();
function stageError(code) { const error = new Error(code); error.code = code; return error; }
function id() { return globalThis.crypto.randomUUID(); }
function base64(bytes) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(text);
}
function cancelStagedReader(reader, signal) {
  const abort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) abort();
  return () => signal?.removeEventListener("abort", abort);
}
async function digestFile(file, signal = null) {
  const hash = createSha256(); const reader = file.stream().getReader();
  const stopReader = cancelStagedReader(reader, signal);
  try { for (;;) { signal?.throwIfAborted(); const chunk = await reader.read(); signal?.throwIfAborted(); if (chunk.done) return hash.hex(); hash.update(chunk.value); } }
  finally { await reader.cancel().catch(() => undefined); stopReader(); reader.releaseLock(); }
}

/** One owner for extension-acquired bytes and immutable receiver bodies. */
export class CaptureStaging {
  constructor(storage = globalThis.navigator?.storage, records = null) { this.storage = storage; this.records = records; this.chains = new Map(); this.seals = new Map(); }
  async directory() {
    if (!this.storage?.getDirectory) throw stageError("capture_staging_unavailable");
    return (await this.storage.getDirectory()).getDirectoryHandle("polylogue-capture", { create: true });
  }
  async metadata(stageId) {
    if (!/^[a-f0-9-]+$/.test(stageId)) throw stageError("capture_staging_invalid_ref");
    const dir = await this.directory();
    try { return JSON.parse(await (await (await dir.getFileHandle(`${stageId}.json`)).getFile()).text()); }
    catch (error) {
      if (error?.name === "NotFoundError" || error instanceof SyntaxError) {
        const interrupted = stageError("capture_staging_interrupted");
        // Recovery distinguishes an unpublished file from malformed existing
        // metadata. Preserve the filesystem classification beside our token.
        interrupted.name = error.name;
        throw interrupted;
      }
      throw error;
    }
  }
  async *metadataEntries() {
    const dir = await this.directory();
    for await (const [name, handle] of dir.entries()) {
      if (!name.endsWith(".json")) continue;
      try { yield JSON.parse(await (await handle.getFile()).text()); }
      catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        yield { id: name.slice(0, -5), state: "metadata-interrupted" };
      }
    }
  }
  async unpublishedMetadata(stageId, owner, purpose) {
    const directory = await this.directory();
    const file = await (await directory.getFileHandle(`${stageId}.json`)).getFile();
    // OPFS creates an empty entry before the first writable close. Recreate
    // only that exact unpublished state, never malformed acquired metadata.
    if (file.size !== 0) return false;
    for await (const [name] of directory.entries()) {
      if (name.startsWith(`${stageId}.`) && name !== `${stageId}.json`) return false;
    }
    if (!await this.records?.captureReferences(stageId)) return true;
    if (!owner.checkpoint_snapshot_id) return false;
    const snapshot = await this.records.getCapture(owner.checkpoint_snapshot_id);
    if (purpose.kind === "checkpoint-export-evidence" && typeof purpose.export_record_key === "string" && snapshot?.kind === "checkpoint-export" && snapshot.state === "ready") {
      const record = await this.records.getCaptureRecord(snapshot.id, purpose.export_record_key);
      return record?.record_kind === "checkpoint:evidence" && record.copied_id === stageId && record.state === "copying";
    }
    return purpose.kind === "checkpoint" && snapshot?.artifact_id === stageId && snapshot.state === "ready" &&
      ["checkpoint-snapshot", "checkpoint-export"].includes(snapshot.kind);
  }
  async save(meta) {
    const dir = await this.directory();
    const writer = await (await dir.getFileHandle(`${meta.id}.json`, { create: true })).createWritable();
    try { await writer.write(JSON.stringify(meta)); await writer.close(); }
    catch (error) { await writer.abort().catch(() => undefined); throw error; }
  }
  serialize(stageId, operation) {
    const previous = this.chains.get(stageId) || Promise.resolve();
    const next = previous.then(operation, operation);
    this.chains.set(stageId, next);
    return next.finally(() => { if (this.chains.get(stageId) === next) this.chains.delete(stageId); });
  }
  producerStageId(owner, producerId) { return createSha256().update(JSON.stringify([owner, producerId])).hex(); }
  async begin(owner, purpose = {}, producerId = null) {
    const stageId = producerId ? this.producerStageId(owner, producerId) : id();
    return this.serialize(stageId, async () => {
      try {
        const existing = await this.metadata(stageId);
        if (JSON.stringify(existing.owner) !== JSON.stringify(owner) || JSON.stringify(existing.acquisition) !== JSON.stringify(purpose.acquisition || null) || JSON.stringify(existing.capture_bundle) !== JSON.stringify(purpose.capture_bundle || null) ||
            JSON.stringify(existing.invocation_ref) !== JSON.stringify(purpose.invocation_ref || null) ||
            existing.source_url !== (purpose.source_url || null) || existing.kind !== (purpose.kind || "asset") ||
            existing.observation_only !== (purpose.observation_only === true)) throw stageError("capture_staging_owner_mismatch");
        if (purpose.invocation_ref) {
          const invocation = await this.records.getCapture(purpose.invocation_ref.id);
          if (!invocation || invocation.kind !== "native-invocation" || invocation.state !== "open" ||
              invocation.token !== purpose.invocation_ref.token || JSON.stringify(invocation.owner) !== JSON.stringify(owner)) {
            throw stageError("native_invocation_owner_mismatch");
          }
        }
        return { id: existing.id, token: existing.token };
      } catch (error) {
        if (error.name !== "NotFoundError" && !(error.name === "SyntaxError" && await this.unpublishedMetadata(stageId, owner, purpose))) throw error;
      }
    const bundle = purpose.capture_bundle ? await this.records.getCapture(purpose.capture_bundle.id) : null;
    const invocation = purpose.invocation_ref ? await this.records.getCapture(purpose.invocation_ref.id) : null;
    if (purpose.invocation_ref && (!invocation || invocation.kind !== "native-invocation" || invocation.state !== "open" ||
        invocation.token !== purpose.invocation_ref.token || JSON.stringify(invocation.owner) !== JSON.stringify(owner))) throw stageError("native_invocation_owner_mismatch");
    const acquisitionSequence = purpose.invocation_ref ? invocation.acquisition_sequence : purpose.capture_bundle
      ? bundle?.acquisition_sequence
      : purpose.kind === "native-response" && !purpose.observation_only ? await this.records.nextNativeAcquisitionSequence() : null;
    if (purpose.kind === "native-response" && !purpose.observation_only && !Number.isSafeInteger(acquisitionSequence)) throw stageError("native_acquisition_sequence_invalid");
    const createdAt = new Date().toISOString();
    const meta = { id: stageId, token: id(), state: "acquiring", owner, sequence: 0, bytes: 0, acquisition_sequence: acquisitionSequence, observed_at: invocation?.observed_at || bundle?.observed_at || createdAt,
      extension_instance_id: purpose.invocation_ref ? invocation.extension_instance_id || null : purpose.capture_bundle ? bundle?.extension_instance_id || null : purpose.extension_instance_id || null,
      response_metadata: purpose.response_metadata || null, observation_only: purpose.observation_only === true,
      invocation_ref: purpose.invocation_ref || null, invocation_native_id: invocation?.native_id || null,
      queue_context: purpose.queue_context || null, kind: purpose.kind || "asset", acquisition: purpose.acquisition || null,
      capture_bundle: purpose.capture_bundle || null, source_url: purpose.source_url || null, created_at: createdAt };
    await this.save(meta);
    if (meta.kind === "native-response" && !meta.capture_bundle) await this.publishNativeAcquisition(meta);
    return { id: meta.id, token: meta.token };
    });
  }
  requireOwner(meta, ref, owner) {
    if (meta.token !== ref.token || JSON.stringify(meta.owner) !== JSON.stringify(owner)) throw stageError("capture_staging_owner_mismatch");
  }
  async append(ref, owner, sequence, encoded) {
    return this.serialize(ref.id, async () => {
      const meta = await this.metadata(ref.id); this.requireOwner(meta, ref, owner);
      if (meta.state !== "acquiring") throw stageError("capture_staging_sequence_mismatch");
      // The worker may terminate after committing a chunk but before its ACK.
      // Accept only the same bytes retransmitted for the last committed chunk.
      const text = atob(encoded); const bytes = Uint8Array.from(text, (char) => char.charCodeAt(0));
      const digest = createSha256().update(bytes).hex();
      if (sequence === meta.sequence - 1 && meta.last_chunk_sha256 === digest) return { sequence: meta.sequence, size_bytes: meta.bytes };
      if (sequence !== meta.sequence) throw stageError("capture_staging_sequence_mismatch");
      const dir = await this.directory();
      // A committed part is immutable. Opening the growing whole file for
      // every chunk would copy all prior bytes on each atomic OPFS close.
      const writer = await (await dir.getFileHandle(`${meta.id}.${sequence}.part`, { create: true })).createWritable();
      try { await writer.write(bytes); await writer.close(); }
      catch (error) { await writer.abort().catch(() => undefined); throw error; }
      meta.sequence += 1; meta.bytes += bytes.length; meta.last_chunk_sha256 = digest; await this.save(meta);
      return { sequence: meta.sequence, size_bytes: meta.bytes };
    });
  }
  async seal(ref, owner, acquisitionResult = null) {
    const controller = new AbortController();
    const active = this.seals.get(ref.id) || new Set(); active.add(controller); this.seals.set(ref.id, active);
    try { return await this.serialize(ref.id, async () => {
      controller.signal.throwIfAborted();
      const meta = await this.metadata(ref.id); this.requireOwner(meta, ref, owner);
      if (meta.kind === "native-response" && acquisitionResult?.response_metadata) {
        const metadata = acquisitionResult.response_metadata;
        if (!Number.isInteger(metadata.status) || metadata.status < 100 || metadata.status > 599 ||
            (metadata.content_type !== null && typeof metadata.content_type !== "string") ||
            (metadata.retry_after !== null && typeof metadata.retry_after !== "string")) throw stageError("capture_response_metadata_invalid");
        if (meta.response_metadata && JSON.stringify(meta.response_metadata) !== JSON.stringify(metadata)) throw stageError("capture_response_metadata_conflict");
        meta.response_metadata = metadata; await this.save(meta);
      }
      if (meta.acquisition && acquisitionResult) {
        if (acquisitionResult.status !== "acquired" || acquisitionResult.asset?.streamed !== true) throw stageError("capture_acquisition_result_invalid");
        meta.acquisition_result = acquisitionResult; await this.save(meta);
      }
      const publishAcquisition = async () => {
        if (meta.kind === "native-response" && !meta.capture_bundle) await this.publishNativeAcquisition(meta);
        if (meta.capture_bundle) {
          const bundle = await this.records.getCapture(meta.capture_bundle.id);
          if (bundle) await this.records.publishNativeBundleReply(meta.capture_bundle.id, owner, meta.capture_bundle.name,
            ref, { ok: true, status: meta.response_metadata?.status || 200 });
          else if (!await this.referenced(meta.id)) throw stageError("native_bundle_recovery_missing");
        }
        if (!meta.acquisition) return;
        if (!meta.acquisition_result || !this.records) throw stageError("capture_acquisition_result_missing");
        await this.records.publishAcquisition(meta.acquisition, { ...meta.acquisition_result,
          asset: { ...meta.acquisition_result.asset, staged_asset: ref, size_bytes: meta.bytes, sha256: meta.sha256 } });
      };
      if (meta.state === "sealed") {
        const file = await this.file(meta.id);
        if (file.size !== meta.bytes) throw stageError("capture_staging_incomplete");
        await publishAcquisition();
        await this.discardParts(meta.id);
        return { staged_asset: ref, size_bytes: meta.bytes, sha256: meta.sha256 };
      }
      if (meta.state !== "acquiring") throw stageError("capture_staging_sequence_mismatch");
      const dir = await this.directory();
      const writer = await (await dir.getFileHandle(`${meta.id}.bytes`, { create: true })).createWritable();
      const hash = createSha256(); let written = 0;
      try {
        for (let sequence = 0; sequence < meta.sequence; sequence += 1) {
          controller.signal.throwIfAborted();
          const part = await (await dir.getFileHandle(`${meta.id}.${sequence}.part`)).getFile();
          const reader = part.stream().getReader();
          const cancelReader = () => { void reader.cancel(controller.signal.reason).catch(() => undefined); };
          controller.signal.addEventListener("abort", cancelReader, { once: true });
          if (controller.signal.aborted) cancelReader();
          try {
            for (;;) {
              controller.signal.throwIfAborted();
              const chunk = await reader.read(); if (chunk.done) break;
              hash.update(chunk.value); written += chunk.value.length; await writer.write(chunk.value);
            }
          } finally {
            await reader.cancel().catch(() => undefined);
            controller.signal.removeEventListener("abort", cancelReader); reader.releaseLock();
          }
        }
        if (written !== meta.bytes) throw stageError("capture_staging_incomplete");
        controller.signal.throwIfAborted();
        await writer.close();
      } catch (error) { await writer.abort().catch(() => undefined); throw error; }
      // Until this publication succeeds every committed part remains the
      // recovery authority. A sealed restart reads final bytes without rewrite.
      meta.sha256 = hash.hex(); meta.state = "sealed"; await this.save(meta);
      await publishAcquisition();
      await this.discardParts(meta.id);
      return { staged_asset: ref, size_bytes: meta.bytes, sha256: meta.sha256 };
    }); } finally { active.delete(controller); if (!active.size) this.seals.delete(ref.id); }
  }
  async discardParts(stageId) {
    const dir = await this.directory();
    for await (const [name] of dir.entries()) {
      if (name.startsWith(`${stageId}.`) && name.endsWith(".part")) await dir.removeEntry(name);
    }
  }
  async file(stageId) {
    const dir = await this.directory();
    try { return await (await dir.getFileHandle(`${stageId}.bytes`)).getFile(); }
    catch (error) { if (error.name === "NotFoundError") throw Object.assign(stageError("capture_staging_missing_bytes"), { name: error.name, cause: error }); throw error; }
  }
  async cancel(ref, owner) {
    const current = await this.metadata(ref.id); this.requireOwner(current, ref, owner);
    const sealing = Boolean(this.seals.get(ref.id)?.size);
    for (const controller of this.seals.get(ref.id) || []) controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    return this.serialize(ref.id, async () => {
      const meta = await this.metadata(ref.id); this.requireOwner(meta, ref, owner);
      // An acquired result is immutable evidence even if its seal response was
      // lost. Only removal of its durable roots can retire sealed bytes.
      if (meta.state === "sealed" || meta.acquisition_result || sealing) return;
      if (meta.kind === "native-response" && !meta.capture_bundle) await this.records.retireNativeAcquisition(meta.id);
      if (meta.acquisition) await this.records.releaseAcquisition(meta.acquisition, meta.id);
      await this.discard(meta.id);
    });
  }
  async reconcileNativeAcquisition(stageId) {
    return this.serialize(stageId, async () => {
      let meta;
      try { meta = await this.metadata(stageId); }
      catch (error) { if (error?.name === "NotFoundError") return; throw error; }
      await this.publishNativeAcquisition(meta);
    });
  }
  async publishNativeAcquisition(meta) {
    const existing = await this.records.getCapture(`raw:${meta.id}`);
    if (meta.retired || (!existing && await this.records.hasNormalizedNativeSource(meta.id))) return;
    await this.records.putCapture({ id: `raw:${meta.id}`, kind: "native-acquisition", owner: meta.owner, provider: meta.owner.provider,
      raw_ref: { id: meta.id, token: meta.token }, source_refs: [meta.id], state: meta.state === "sealed" ? "pending-normalization" : "acquiring",
      observed_at: meta.observed_at, acquisition_sequence: meta.acquisition_sequence, source_url: meta.source_url,
      queue_context: meta.queue_context || null, invocation_ref: meta.invocation_ref || null, invocation_native_id: meta.invocation_native_id || null });
  }
  async discard(stageId) {
    await this.discardParts(stageId);
    const dir = await this.directory();
    await Promise.all([dir.removeEntry(`${stageId}.json`), dir.removeEntry(`${stageId}.bytes`)].map((p) => p.catch((error) => {
      if (error?.name !== "NotFoundError") throw error;
    })));
  }
  async *json(value, signal = null, canonical = false) {
    signal?.throwIfAborted();
    if (Array.isArray(value)) {
      yield "[";
      for (let i = 0; i < value.length; i++) { if (i) yield ","; yield* this.json(value[i] ?? null, signal, canonical); }
      yield "]";
    } else if (value && typeof value === "object") {
      yield "{"; let first = true;
      for (const key of canonical ? Object.keys(value).sort() : Object.keys(value)) {
        const child = value[key];
        if (child === undefined) continue;
        if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`; yield* this.json(child, signal, canonical);
      }
      yield "}";
    } else {
      const encoded = JSON.stringify(value);
      for (let offset = 0; offset < encoded.length;) {
        let end = Math.min(encoded.length, offset + CHUNK_BYTES);
        if (end < encoded.length && /[\uD800-\uDBFF]/.test(encoded[end - 1])) end -= 1;
        yield encoded.slice(offset, end); offset = end;
      }
    }
  }
  async *assetJson(asset, signal = null, canonical = false) {
    const keys = Object.keys(asset).filter((key) => key !== "staged_asset" && asset[key] !== undefined);
    if (asset.staged_asset && !keys.includes("inline_base64")) keys.push("inline_base64");
    if (canonical) keys.sort();
    yield "{"; let first = true;
    for (const key of keys) {
      if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`;
      if (key !== "inline_base64" || !asset.staged_asset) { yield* this.json(asset[key], signal, canonical); continue; }
      const ref = asset.staged_asset; const meta = await this.metadata(ref.id);
      if (meta.token !== ref.token || meta.state !== "sealed") throw stageError("capture_staging_asset_unsealed");
      yield '"';
      const file = await this.file(ref.id);
      for (let offset = 0; offset < file.size; offset += CHUNK_BYTES) {
        signal?.throwIfAborted();
        yield base64(new Uint8Array(await file.slice(offset, offset + CHUNK_BYTES).arrayBuffer()));
      }
      yield '"';
    }
    yield "}";
  }
  async *attachmentsJson(attachments, signal = null, canonical = false) {
    yield "[";
    for (let i = 0; i < attachments.length; i++) {
      if (i) yield ","; yield* this.assetJson(attachments[i], signal, canonical);
    }
    yield "]";
  }
  async *turnJson(turn, ordinal, signal = null) {
    yield "{"; let first = true; let hasOrdinal = false;
    for (const [key, child] of Object.entries(turn)) {
      if (child === undefined) continue;
      if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`;
      if (key === "attachments") yield* this.attachmentsJson(child || [], signal);
      else if (key === "ordinal") { hasOrdinal = true; yield* this.json(ordinal, signal); }
      else yield* this.json(child, signal);
    }
    if (!hasOrdinal) { if (!first) yield ","; yield `"ordinal":${ordinal}`; }
    yield "}";
  }
  async *recordsJson(captureId, channel, signal = null) {
    yield "["; let first = true; let ordinal = 0;
    for await (const record of this.records.captureRecords(captureId)) {
      signal?.throwIfAborted();
      if (channel === "turns" && record.turn) {
        if (!first) yield ","; first = false; yield* this.turnJson(record.turn, ordinal++, signal);
      } else if (channel === "attachments" && record.session_attachment) {
        if (!first) yield ","; first = false; yield* this.assetJson(record.session_attachment, signal);
      } else if (channel === "failures") {
        for (const failure of record.asset_failures || []) {
          if (!first) yield ","; first = false; yield* this.json(failure, signal);
        }
      }
    }
    yield "]";
  }
  async *providerMetaJson(meta, recordRef, signal = null) {
    yield "{"; let first = true;
    for (const [key, value] of Object.entries(meta)) {
      if (value === undefined) continue;
      if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`;
      if (key !== "asset_acquisition" || !recordRef || value?.failed?.staged_asset_failures !== recordRef) {
        yield* this.json(value, signal); continue;
      }
      yield "{"; let acquisitionFirst = true;
      for (const [field, child] of Object.entries(value)) {
        if (child === undefined) continue;
        if (!acquisitionFirst) yield ","; acquisitionFirst = false; yield `${JSON.stringify(field)}:`;
        if (field === "failed") yield* this.recordsJson(recordRef, "failures", signal);
        else yield* this.json(child, signal);
      }
      yield "}";
    }
    yield "}";
  }
  async *rawJson(value, sources, signal = null) {
    if (value && Object.keys(value).length === 1 && typeof value.staged_raw_json === "string" && sources.includes(value.staged_raw_json)) {
      const meta = await this.metadata(value.staged_raw_json);
      if (meta.state !== "sealed") throw stageError("capture_staging_asset_unsealed");
      const reader = (await this.file(meta.id)).stream().getReader();
      const stopReader = cancelStagedReader(reader, signal);
      try { for (;;) { signal?.throwIfAborted(); const chunk = await reader.read(); signal?.throwIfAborted(); if (chunk.done) break; yield chunk.value; } }
      finally { await reader.cancel().catch(() => undefined); stopReader(); reader.releaseLock(); }
    } else if (sources.length && value?.acquisition?.kind === "grok-endpoint-bundle") {
      yield "{"; let first = true;
      for (const [key, child] of Object.entries(value)) {
        if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`;
        if (["conversation", "responses", "response_nodes"].includes(key)) yield* this.rawJson(child, sources, signal);
        else yield* this.json(child, signal);
      }
      yield "}";
    } else yield* this.json(value, signal);
  }
  async *envelopeJson(envelope, signal = null) {
    yield "{"; let first = true;
    const recordRef = envelope.capture_record_ref;
    for (const [key, value] of Object.entries(envelope)) {
      if (value === undefined || ["capture_body_ref", "capture_source_refs", "capture_record_ref", "capture_summary", "capture_observation_ref"].includes(key)) continue;
      if (!first) yield ","; first = false; yield `${JSON.stringify(key)}:`;
      if (key === "raw_provider_payload") yield* this.rawJson(value, envelope.capture_source_refs || [], signal);
      else if (key === "provider_meta" && value) yield* this.providerMetaJson(value, recordRef, signal);
      else if (key === "session" && value) {
        yield "{"; let sessionFirst = true;
        for (const [field, child] of Object.entries(value)) {
          if (child === undefined) continue;
          if (!sessionFirst) yield ","; sessionFirst = false; yield `${JSON.stringify(field)}:`;
          if (field === "turns" && child?.staged_turns === recordRef) yield* this.recordsJson(recordRef, "turns", signal);
          else if (field === "attachments" && child?.staged_session_attachments === recordRef) yield* this.recordsJson(recordRef, "attachments", signal);
          else if (field === "provider_meta" && child) yield* this.providerMetaJson(child, recordRef, signal);
          else yield* this.json(child, signal);
        }
        yield "}";
      } else yield* this.json(value, signal);
    }
    yield "}";
  }
  async semanticTurnDigest(turn, signal = null) {
    const hash = createSha256();
    const append = (part) => hash.update(typeof part === "string" ? encoder.encode(part) : part);
    append("[");
    const fields = [turn.role ?? null, turn.text ?? null, turn.timestamp ?? null, Array.isArray(turn.blocks) ? turn.blocks : []];
    for (let i = 0; i < fields.length; i++) {
      if (i) append(","); for await (const part of this.json(fields[i], signal, true)) append(part);
    }
    append(",");
    for await (const part of this.attachmentsJson(Array.isArray(turn.attachments) ? turn.attachments : [], signal, true)) append(part);
    append("]"); return hash.hex();
  }
  async nativeRevisionId(provider, nativeId, rawRef, relatedRefs, acquisition) {
    const replies = {};
    for (const [name, ref] of Object.entries({ raw: rawRef, ...relatedRefs })) {
      const meta = await this.metadata(ref.id);
      replies[name] = { sha256: meta.sha256, source_url: meta.source_url, response_metadata: meta.response_metadata };
    }
    const hash = createSha256();
    for await (const part of this.json([provider, nativeId, replies, acquisition], null, true)) hash.update(typeof part === "string" ? encoder.encode(part) : part);
    return hash.hex();
  }
  async snapshotExportEvidence(snapshot, sourceRef, signal) {
    const sourceId = typeof sourceRef === "string" ? sourceRef : sourceRef.id;
    return this.serialize(sourceId, async () => {
      const key = `export-evidence:${sourceId}`;
      let record = await this.records.getCaptureRecord(snapshot.id, key);
      const current = await this.metadata(sourceId);
      if (sourceRef.token && sourceRef.token !== current.token) throw stageError("checkpoint_export_evidence_owner_mismatch");
      if (!record) {
        const copiedId = this.producerStageId({ checkpoint_snapshot_id: snapshot.id }, key);
        record = { capture_id: snapshot.id, key, record_kind: "checkpoint:evidence", order_time: 0,
          order_id: sourceId, occurrence: 0, original_metadata: current, copied_id: copiedId,
          asset_refs: [sourceId, copiedId], state: "copying" };
        // The original metadata and both custody roots commit before any OPFS
        // await. A retry freezes this same prefix even after its source seals.
        await this.records.putCaptureRecord(record);
      }
      const original = record.original_metadata;
      if (original.id !== current.id || original.token !== current.token ||
          JSON.stringify(original.owner) !== JSON.stringify(current.owner) ||
          JSON.stringify(original.acquisition) !== JSON.stringify(current.acquisition)) {
        throw stageError("checkpoint_export_evidence_owner_mismatch");
      }
      const owner = { checkpoint_snapshot_id: snapshot.id };
      const ref = await this.begin(owner, { kind: "checkpoint-export-evidence", export_record_key: key }, key);
      if (ref.id !== record.copied_id) throw stageError("checkpoint_export_evidence_owner_mismatch");
      let meta = await this.metadata(ref.id);
      if (meta.state === "export-copy-preparing") {
        try {
          const file = await this.file(ref.id);
          if (file.size === meta.bytes && await digestFile(file, signal) === meta.sha256) {
            meta = { ...meta, state: "sealed" }; await this.save(meta);
          }
        } catch (error) { if (error?.name !== "NotFoundError") throw error; }
      }
      if (meta.state !== "sealed") {
        const directory = await this.directory();
        const writer = await (await directory.getFileHandle(`${ref.id}.bytes`, { create: true })).createWritable();
        const hash = createSha256(); let bytes = 0;
        try {
          const acquiring = current.state === "acquiring";
          const count = acquiring ? original.sequence : 1;
          for (let sequence = 0; sequence < count && bytes < original.bytes; sequence++) {
            const file = acquiring
              ? await (await directory.getFileHandle(`${sourceId}.${sequence}.part`)).getFile()
              : await this.file(sourceId);
            for (let offset = 0; offset < file.size && bytes < original.bytes; offset += CHUNK_BYTES) {
              signal?.throwIfAborted();
              const chunk = new Uint8Array(await file.slice(offset, Math.min(file.size, offset + CHUNK_BYTES, offset + original.bytes - bytes)).arrayBuffer());
              hash.update(chunk); bytes += chunk.length; await writer.write(chunk);
            }
          }
          if (bytes !== original.bytes || (original.state !== "acquiring" && hash.hex() !== original.sha256)) throw stageError("checkpoint_export_evidence_mismatch");
          meta = { ...meta, state: "export-copy-preparing", bytes, sha256: hash.hex(), export_record_key: key };
          await this.save(meta); signal?.throwIfAborted(); await writer.close();
          meta = { ...meta, state: "sealed" }; await this.save(meta);
        } catch (error) { await writer.abort().catch(() => undefined); throw error; }
      }
      const file = await this.file(ref.id);
      if (file.size !== meta.bytes || meta.bytes !== original.bytes || await digestFile(file, signal) !== meta.sha256) throw stageError("checkpoint_export_evidence_mismatch");
      if (record.state !== "sealed") await this.records.putCaptureRecord({ ...record, state: "sealed", sha256: meta.sha256, bytes: meta.bytes });
      return { ref, metadata: original, sha256: meta.sha256, bytes: meta.bytes };
    });
  }

  async writeExportEvidence(stageId, append, signal, originalMetadata = null) {
    // Serialize with append/seal so an acquiring prefix cannot change or lose
    // its immutable part files while its export snapshot is written.
    return this.serialize(stageId, async () => {
      const meta = await this.metadata(stageId);
      const directory = await this.directory();
      await append('{"metadata":');
      for await (const text of this.json(originalMetadata || meta, signal)) await append(text);
      await append(',"parts":[');
      const aggregate = createSha256(); let total = 0;
      const acquiring = meta.state === "acquiring";
      const count = acquiring ? meta.sequence : 1;
      for (let sequence = 0; sequence < count; sequence++) {
        signal?.throwIfAborted();
        const file = acquiring
          ? await (await directory.getFileHandle(`${meta.id}.${sequence}.part`)).getFile()
          : await this.file(stageId);
        const hash = createSha256();
        await append(`${sequence ? "," : ""}{"sequence":${sequence},"size_bytes":${file.size},"inline_base64":"`);
        for (let offset = 0; offset < file.size; offset += CHUNK_BYTES) {
          signal?.throwIfAborted();
          const bytes = new Uint8Array(await file.slice(offset, offset + CHUNK_BYTES).arrayBuffer());
          hash.update(bytes); aggregate.update(bytes); total += bytes.length;
          await append(base64(bytes));
        }
        await append(`","sha256":"${hash.hex()}"}`);
      }
      const digest = aggregate.hex();
      if (total !== meta.bytes || (!acquiring && digest !== meta.sha256)) throw stageError("checkpoint_export_evidence_mismatch");
      await append(`],"size_bytes":${total},"sha256":"${digest}"}`);
    });
  }

  async prepareCheckpoint(snapshot, signal = null) {
    signal?.throwIfAborted();
    const owner = { checkpoint_snapshot_id: snapshot.id };
    const ref = await this.begin(owner, { kind: "checkpoint" }, snapshot.id);
    if (ref.id !== snapshot.artifact_id) throw stageError("checkpoint_artifact_owner_mismatch");
    return this.serialize(ref.id, async () => {
      const meta = await this.metadata(ref.id); this.requireOwner(meta, ref, owner);
      if (["sealed", "checkpoint-acknowledged"].includes(meta.state)) {
        const body = await this.file(meta.id);
        if (body.size !== meta.bytes) throw stageError("capture_staging_incomplete");
        return { ref: meta.id, body, digest: `sha256:${meta.sha256}`, sizeBytes: meta.bytes };
      }
      if (meta.state !== "acquiring") throw stageError("capture_staging_sequence_mismatch");
      const directory = await this.directory();
      const writer = await (await directory.getFileHandle(`${meta.id}.bytes`, { create: true })).createWritable();
      const hash = createSha256(); let bytes = 0;
      const append = async (text) => {
        for (let offset = 0; offset < text.length;) {
          signal?.throwIfAborted();
          let end = Math.min(text.length, offset + CHUNK_BYTES);
          if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1])) end -= 1;
          const chunk = encoder.encode(text.slice(offset, end));
          hash.update(chunk); bytes += chunk.length; await writer.write(chunk); offset = end;
        }
      };
      try {
        const exporting = snapshot.kind === "checkpoint-export";
        await append(exporting ? '{"schema":"polylogue.backfill-export.v2","ledger":{' : "{");
        for (const collection of ["jobs", "queue", "revisions"]) {
          await append(`${collection === "jobs" ? "" : ","}${JSON.stringify(collection)}:[`);
          let first = true;
          for await (const record of this.records.captureRecords(snapshot.id)) {
            if (record.record_kind !== `checkpoint:${collection}`) continue;
            await append(first ? "" : ",");
            if (exporting) {
              for await (const text of this.json(record.checkpoint_value, signal)) await append(text);
            } else await append(canonicalJson(record.checkpoint_value));
            first = false;
          }
          await append("]");
        }
        await append(`,"version":${canonicalJson(snapshot.version)}}`);
        if (exporting) {
          await append(',"acquired":['); let first = true;
          for await (const record of this.records.captureRecords(snapshot.id)) {
            if (record.record_kind !== "checkpoint:queue") continue;
            await append(`${first ? "" : ","}{"queue_id":${JSON.stringify(record.checkpoint_value.id)},"capture":`); first = false;
            for await (const text of this.json(record.export_capture, signal)) await append(text);
            await append(',"files":['); let firstFile = true;
            const evidence = async (ref) => {
              if (!ref) return;
              await append(firstFile ? "" : ","); firstFile = false;
              const evidence = await this.snapshotExportEvidence(snapshot, ref, signal);
              await this.writeExportEvidence(evidence.ref.id, append, signal, evidence.metadata);
            };
            for (const ref of record.asset_refs || []) await evidence(ref);
            // The snapshot's record-root index pins all original normalized
            // records and their asset custody until the exact export ACK.
            if (record.record_ref) {
              for await (const native of this.records.captureRecords(record.record_ref)) {
                for (const ref of native.asset_refs || []) await evidence(ref);
              }
            }
            await append("]}");
          }
          await append("]}");
        }
        signal?.throwIfAborted(); await writer.close();
      } catch (error) { await writer.abort().catch(() => undefined); throw error; }
      await this.save({ ...meta, state: "sealed", bytes, sha256: hash.hex() });
      return { ref: meta.id, body: await this.file(meta.id), digest: `sha256:${hash.hex()}`, sizeBytes: bytes };
    });
  }

  async releaseCheckpoint(snapshot, receipt) {
    const root = snapshot.kind === "checkpoint-export" ? await this.records.getCapture(snapshot.id) : null;
    if (root?.state === "acknowledged") {
      // Validate the exact durable receipt even when the first ACK already
      // removed the artifact. Remaining copy cleanup is restartable below.
      await this.records.acknowledgeExportSnapshot(snapshot, receipt);
    } else {
      const meta = await this.metadata(snapshot.artifact_id);
      await this.save({ ...meta, state: "checkpoint-acknowledged", receiver_receipt: receipt });
    }
    // The receipt is committed before releasing source custody. Removing a
    // snapshot does not retire the raw/assets it borrowed from delivery roots.
    if (snapshot.kind === "checkpoint-export") {
      for await (const record of this.records.captureRecords(snapshot.id)) {
        if (record.record_kind !== "checkpoint:evidence") continue;
        const copy = await this.metadata(record.copied_id);
        await this.save({ ...copy, retired: true });
      }
      await this.records.acknowledgeExportSnapshot(snapshot, receipt);
      // A crash after the ACK leaves retired copies for the existing orphan
      // sweep; no original acquisition or borrowed delivery is retired here.
      for await (const record of this.records.captureRecords(snapshot.id)) {
        if (record.record_kind === "checkpoint:evidence") await this.discardUnreferenced(record.copied_id);
      }
    }
    else await this.records.discardCapture(snapshot.id);
    await this.discardUnreferenced(snapshot.artifact_id);
  }

  async receiveCheckpoint(root, response, signal = null) {
    const owner = { checkpoint_recovery_id: root.id };
    const ref = await this.begin(owner, { kind: "checkpoint-recovery" }, root.id);
    if (ref.id !== root.artifact_id) throw stageError("checkpoint_artifact_owner_mismatch");
    return this.serialize(ref.id, async () => {
      const meta = await this.metadata(ref.id);
      if (meta.state === "sealed") {
        await response.body?.cancel();
        if (`sha256:${meta.sha256}` !== root.digest) throw stageError("checkpoint_digest_mismatch");
        return this.file(meta.id);
      }
      const reader = response.body?.getReader();
      if (!reader) throw stageError("checkpoint_body_unavailable");
      const abort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
      signal?.addEventListener("abort", abort, { once: true });
      const directory = await this.directory();
      const writer = await (await directory.getFileHandle(`${meta.id}.bytes`, { create: true })).createWritable();
      const hash = createSha256(); let bytes = 0;
      try {
        for (;;) {
          signal?.throwIfAborted();
          const { value, done } = await reader.read();
          if (done) break;
          hash.update(value); bytes += value.length; await writer.write(value);
        }
        signal?.throwIfAborted();
        if (`sha256:${hash.hex()}` !== root.digest || bytes !== root.size_bytes) throw stageError("checkpoint_digest_mismatch");
        await writer.close();
        await this.save({ ...meta, state: "sealed", bytes, sha256: hash.hex() });
        return this.file(meta.id);
      } catch (error) {
        await reader.cancel(error).catch(() => undefined);
        await writer.abort().catch(() => undefined);
        throw error;
      } finally {
        signal?.removeEventListener("abort", abort);
        reader.releaseLock();
      }
    });
  }

  conversionId(queueId) { return createSha256().update(`queue:${queueId}`).hex(); }
  async prepare(envelope, stageId = null, delivery = null, signal = null) {
    signal?.throwIfAborted();
    if (!this.records) throw stageError("capture_staging_records_unavailable");
    const suppliedEnvelope = envelope;
    if (envelope.capture_body_ref) {
      const meta = await this.completePreparedBody(envelope.capture_body_ref, signal);
      if (!["receiver-preparing", "receiver-ready", "receiver-acknowledged"].includes(meta.state)) throw stageError("capture_staging_interrupted");
      return { ref: meta.id, body: await this.file(meta.id), contentHash: meta.sha256, sizeBytes: meta.bytes, acknowledgedReceipt: meta.receiver_receipt || null };
    }
    if (stageId) {
      try {
        const meta = await this.completePreparedBody(stageId, signal);
        if (["receiver-ready", "receiver-acknowledged"].includes(meta.state)) {
          if (delivery?.delivery_kind === "foreground" && meta.state !== "receiver-acknowledged") await this.records.putDelivery({ ...delivery, body_ref: stageId, preparing: false });
          envelope.capture_body_ref = stageId;
          return { ref: stageId, body: await this.file(stageId), contentHash: meta.sha256, sizeBytes: meta.bytes, acknowledgedReceipt: meta.receiver_receipt || null };
        }
      } catch (error) {
        if (error.name === "SyntaxError" || (error.name !== "NotFoundError" && error.code !== "capture_staging_interrupted")) throw error;
      }
    }
    const ref = stageId ? { id: stageId } : await this.begin({ kind: "receiver-body" });
    const dir = await this.directory();
    const typedTurns = !envelope.capture_record_ref && Array.isArray(envelope.session?.turns) ? envelope.session.turns : null;
    const typedAttachments = Array.isArray(envelope.session?.attachments) ? envelope.session.attachments : null;
    const typedCaptureId = envelope.capture_record_ref || ref.id;
    if (typedTurns || typedAttachments) {
      envelope = { ...envelope, capture_record_ref: typedCaptureId, session: { ...envelope.session,
        ...(typedTurns ? { turns: { staged_turns: typedCaptureId } } : {}),
        ...(typedAttachments ? { attachments: { staged_session_attachments: typedCaptureId } } : {}),
      } };
    }
    const recordRef = envelope.capture_record_ref || null;
    const sources = envelope.capture_source_refs || [];
    const bodyRoot = { id: `body:${ref.id}`, kind: "receiver-body", state: "preparing",
      record_ref: recordRef, source_refs: sources, stage_id: ref.id, delivery, envelope };
    if (typedTurns || typedAttachments) await this.records.prepareTypedBody({
      captureId: typedCaptureId, createCapture: !suppliedEnvelope.capture_record_ref,
      turns: typedTurns, attachments: typedAttachments, bodyRoot, signal,
      delivery: delivery?.delivery_kind === "foreground" ? { ...delivery, body_ref: ref.id, preparing: true } : null,
    });
    else if (delivery?.delivery_kind === "foreground") await this.records.putDelivery({ ...delivery, body_ref: ref.id, preparing: true }, bodyRoot);
    else await this.records.putCapture(bodyRoot);
    const writer = await (await dir.getFileHandle(`${ref.id}.bytes`, { create: true })).createWritable();
    const hash = createSha256(); let bytes = 0;
    try {
      for await (const part of this.envelopeJson(envelope, signal)) {
        signal?.throwIfAborted();
        const chunk = typeof part === "string" ? encoder.encode(part) : part; hash.update(chunk); bytes += chunk.length; await writer.write(chunk);
      }
      const prepared = { id: ref.id, state: "receiver-preparing", bytes, sha256: hash.hex(), assets: sources,
        record_ref: recordRef, delivery };
      await this.save(prepared);
      signal?.throwIfAborted();
      await writer.close();
      await this.save({ ...prepared, state: "receiver-ready" });
      await this.records.putCapture({ id: `body:${ref.id}`, kind: "receiver-body", state: "ready", record_ref: recordRef, source_refs: prepared.assets });
      if (delivery?.delivery_kind === "foreground") await this.records.putDelivery({ ...delivery, body_ref: ref.id, preparing: false });
      suppliedEnvelope.capture_body_ref = ref.id;
      return { ref: ref.id, body: await this.file(ref.id), contentHash: hash.hex(), sizeBytes: bytes };
    } catch (error) { await writer.abort().catch(() => undefined); throw error; }
  }
  async resumeDelivery(entry, signal = null) {
    if (!entry.preparing) return;
    let meta = null;
    try { meta = await this.metadata(entry.body_ref); }
    catch (error) { if (error.code !== "capture_staging_interrupted") throw error; }
    if (["receiver-ready", "receiver-acknowledged"].includes(meta?.state) && meta.delivery?.id === entry.id) {
      await this.prepare({ capture_body_ref: entry.body_ref }, null, null, signal);
      await this.records.putDelivery({ ...entry, preparing: false }); return;
    }
    const root = await this.records.getCapture(`body:${entry.body_ref}`);
    if (!root?.envelope || root.stage_id !== entry.body_ref || root.delivery?.id !== entry.id) {
      const error = new Error("capture_delivery_preparation_missing"); error.code = "capture_delivery_preparation_missing"; throw error;
    }
    await this.prepare(root.envelope, entry.body_ref, root.delivery, signal);
    await this.records.putDelivery({ ...entry, preparing: false });
  }
  async completePreparedBody(stageId, signal = null) {
    signal?.throwIfAborted();
    const meta = await this.metadata(stageId);
    if (meta.state !== "receiver-preparing") return meta;
    let file;
    try { file = await this.file(stageId); }
    catch (error) { if (error?.name === "NotFoundError") throw stageError("capture_staging_interrupted"); throw error; }
    if (file.size !== meta.bytes || await digestFile(file, signal) !== meta.sha256) throw stageError("capture_staging_interrupted");
    await this.save({ ...meta, state: "receiver-ready" });
    await this.records.putCapture({ id: `body:${stageId}`, kind: "receiver-body", state: "ready",
      record_ref: meta.record_ref, source_refs: meta.assets || [] });
    return { ...meta, state: "receiver-ready" };
  }
  async markAcknowledged(stageId, receipt) {
    const meta = await this.metadata(stageId);
    const error = receiverAckContractError(receipt, meta.sha256);
    if (error) throw error;
    await this.save({ ...meta, state: "receiver-acknowledged", receiver_receipt: receipt });
  }
  async referenced(stageId) {
    if (!this.records) throw stageError("capture_staging_records_unavailable");
    return this.records.captureReferences(stageId);
  }
  async discardUnreferenced(stageId) {
    return this.serialize(stageId, async () => {
      let meta;
      try { meta = await this.metadata(stageId); }
      catch (error) { if (error?.name !== "NotFoundError") throw error; }
      if (meta?.kind === "checkpoint-export-evidence" && meta.retired) {
        const root = await this.records.getCapture(meta.owner.checkpoint_snapshot_id);
        if (root?.kind === "checkpoint-export" && root.state === "acknowledged" && meta.export_record_key) {
          const record = await this.records.getCaptureRecord(root.id, meta.export_record_key);
          if (record?.record_kind === "checkpoint:evidence" && record.copied_id === stageId) {
            await this.records.deleteCaptureRecord(root.id, record.key);
          }
        }
      }
      if (await this.referenced(stageId)) return;
      for await (const acquisition of this.records.acquisitions(stageId)) {
        await this.records.releaseAcquisition(acquisition.identity, acquisition.ref.id);
        await this.discardUnreferenced(acquisition.ref.id);
      }
      await this.discard(stageId);
    });
  }
  async acknowledgeNative(captureId) {
    const capture = await this.records.getCapture(captureId);
    if (!capture) return;
    if (receiverAckContractError(capture.receiver_receipt, capture.receiver_native?.sha256)) throw stageError("capture_staging_ack_required");
    if (await this.records.recordRootCount(captureId)) return;
    await this.retireFailedNormalizations(capture.raw_ref, capture.id);
    for (const ref of capture.source_refs) {
      const meta = await this.metadata(ref); await this.save({ ...meta, retired: true });
    }
    // Persist retirement before dropping the last owner. Restart's existing
    // orphan sweep can finish physical cleanup after any interruption here.
    await this.records.discardCapture(captureId);
    for (const ref of capture.source_refs) await this.discardUnreferenced(ref);
  }
  async acknowledge(stageId) {
    const meta = await this.metadata(stageId);
    if (meta.state !== "receiver-acknowledged" || receiverAckContractError(meta.receiver_receipt, meta.sha256)) throw stageError("capture_staging_ack_required");
    // Source retirement can fail after delivery publication. Its original input
    // still owns the acknowledged bytes/receipt until conversion settles.
    if (meta.delivery?.source_retry_id && await this.records.hasCaptureRetry(meta.delivery.source_retry_id)) return;
    if (meta.record_ref) {
      const capture = await this.records.getCapture(meta.record_ref);
      if (capture?.state === "ready" && capture.raw_ref) await this.retireFailedNormalizations(capture.raw_ref, capture.id);
    }
    if (meta.record_ref && await this.records.recordRootCount(meta.record_ref) <= 1) {
      for await (const record of this.records.captureRecords(meta.record_ref)) {
        for (const asset of record.asset_refs || []) {
          const acquired = await this.metadata(asset); await this.save({ ...acquired, retired: true });
        }
        await this.records.deleteCaptureRecord(meta.record_ref, record.key);
        for (const asset of record.asset_refs || []) await this.discardUnreferenced(asset);
      }
      await this.records.discardCapture(meta.record_ref);
    }
    await this.records.discardCapture(`body:${stageId}`);
    for (const asset of meta.assets || []) {
      const source = await this.metadata(asset); await this.save({ ...source, retired: true });
      await this.discardUnreferenced(asset);
    }
    await this.discard(stageId);
  }
  async retireFailedNormalizations(rawId, currentId, signal = null) {
    signal?.throwIfAborted();
    const current = await this.records.getCapture(currentId);
    if (!current || current.raw_ref !== rawId || current.state !== "ready") throw stageError("native_normalization_publication_missing");
    for await (const failed of this.records.failedNativeCaptures(current)) {
      signal?.throwIfAborted();
      if (await this.records.recordRootCount(failed.id)) continue;
      for await (const record of this.records.captureRecords(failed.id)) {
        signal?.throwIfAborted();
        for (const asset of record.asset_refs || []) {
          const metadata = await this.metadata(asset); await this.save({ ...metadata, retired: true });
        }
        await this.records.deleteCaptureRecord(failed.id, record.key);
        for (const asset of record.asset_refs || []) await this.discardUnreferenced(asset);
      }
      await this.records.discardCapture(failed.id);
      for (const ref of failed.source_refs || []) await this.discardUnreferenced(ref);
    }
  }
}
