import { File } from "node:buffer";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { IndexedDbBackfillStore } from "../../src/backfill/storage.js";
import { NativeCaptureNormalizer } from "../../src/capture/native.js";
import { CaptureStaging } from "../../src/capture/staging.js";

/** Receiver-contract fixture: no provider content is parsed in JavaScript.
 * Canonical parser and final-artifact parity belong to the Python integration
 * selection. These explicit plans exercise browser byte/occurrence custody.
 */
export function receiverContractPreparation(staging, store, contract = {}) {
  return async (capture, { rawRef, relatedRefs, signal, summaryOnly = false }) => {
    signal?.throwIfAborted();
    const members = capture.provider === "grok" ? { responses: rawRef, ...relatedRefs } : { conversation: rawRef };
    const retained = {};
    for (const [name, ref] of Object.entries(members)) {
      const meta = await staging.metadata(ref.id);
      staging.requireOwner(meta, ref, capture.owner);
      retained[name] = { sha256: meta.sha256, size_bytes: meta.bytes };
    }
    const reference = { adopted: { job: { job_id: "synthetic-receiver-job", provider: capture.provider },
      lease: { lease_id: "synthetic-lease", generation: 1, proof: "synthetic-proof" },
      scope: { kind: "invocation", resume_capability: "synthetic-receiver-resume-capability" } },
      acquisition_id: capture.acquisition_id, preparation_instance_id: capture.preparation_instance_id,
      plan_digest: "a".repeat(64), sha256: "b".repeat(64), size_bytes: 1234 };
    await store.putCapture({ ...capture, receiver_native: reference });
    await contract.onPrepare?.({ capture, members: retained, reference, signal });
    for (const asset of summaryOnly ? [] : contract.plan || []) {
      signal?.throwIfAborted();
      const result = await contract.acquire?.({ capture, rawRef, asset, signal });
      contract.receipts?.push({ ordinal: asset.ordinal, result });
    }
    return { rawRevision: "c".repeat(64), reference,
      summary: { title: null, turn_count: 1, attachment_count: (contract.plan || []).length,
        session_kind: "standard", needs_follow_up: false, ...contract.summary } };
  };
}

/** Synthetic OPFS with atomic close, restart persistence, and explicit faults. */
export function memoryOriginStorage() {
  const files = new Map();
  const writes = [];
  const directory = {
    async getDirectoryHandle() { return directory; },
    async getFileHandle(name, { create = false } = {}) {
      if (!files.has(name)) {
        if (!create) throw new globalThis.DOMException("Missing file", "NotFoundError");
        files.set(name, new Uint8Array());
      }
      return {
        async getFile() { return new File([files.get(name)], name); },
        async createWritable({ keepExistingData = false } = {}) {
          let size = keepExistingData ? files.get(name).length : 0;
          let segments = keepExistingData ? [{ offset: 0, bytes: files.get(name).slice() }] : [];
          let offset = 0;
          return {
            async seek(position) { offset = position; },
            async truncate(length) {
              segments = segments.filter((part) => part.offset < length).map((part) => ({
                ...part, bytes: part.bytes.subarray(0, Math.min(part.bytes.length, length - part.offset)),
              }));
              size = length;
            },
            async write(value) {
              const input = typeof value === "string" ? new TextEncoder().encode(value) : new Uint8Array(value);
              segments.push({ offset, bytes: input.slice() });
              size = Math.max(size, offset + input.length); offset += input.length;
            },
            async close() {
              if (directory.failClose?.(name)) throw new globalThis.DOMException("Synthetic disk quota", "QuotaExceededError");
              const bytes = new Uint8Array(size);
              for (const part of segments) bytes.set(part.bytes.subarray(0, size - part.offset), part.offset);
              files.set(name, bytes); writes.push(name);
            },
            async abort() {},
          };
        },
      };
    },
    async removeEntry(name) {
      if (!files.delete(name)) throw new globalThis.DOMException("Missing file", "NotFoundError");
    },
    async *entries() { for (const name of files.keys()) yield [name, await directory.getFileHandle(name)]; },
  };
  return { getDirectory: async () => directory, files, writes, directory };
}

export function stagingRuntime(storage = memoryOriginStorage(), owner = { tab_id: 42, document_id: "synthetic-document", provider: "chatgpt" }) {
  globalThis.IDBKeyRange = IDBKeyRange;
  const store = new IndexedDbBackfillStore(new IDBFactory());
  const staging = new CaptureStaging(storage, store);
  let dispatch = null;
  const operations = new Map();
  const contract = { plan: [], receipts: [], acquire: async ({ capture, rawRef, asset, signal }) => {
    signal?.throwIfAborted();
    const result = await dispatch({ type: "polylogue.acquireRecordAssets", provider: capture.provider,
      nativeId: capture.native_id, recordKey: asset.descriptor.original_record_key ?? String(asset.descriptor.original_record_ordinal),
      attachmentOrdinal: asset.ordinal, attachments: [asset.descriptor], capture_ref: rawRef.id });
    if (!result?.ok) throw new Error(result?.error || "fixture_assets_failed");
    return result.acquisition;
  } };
  const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store, contract) });
  return {
    staging, storage, store,
    nativeContract: contract,
    async retainedNativeReplies(envelope) {
      const capture = await store.getCapture(envelope.capture_record_ref);
      const read = async (ref) => JSON.parse(await (await staging.file(ref.id || ref)).text());
      if (capture.provider !== "grok") return read(capture.raw_ref);
      const replies = { responses: await read(capture.raw_ref) };
      for (const [name, ref] of Object.entries(capture.related_refs)) replies[name] = await read(ref);
      return replies;
    },
    setDispatch(value) { dispatch = value; },
    async materialize(envelope) {
      if (envelope.receiver_native) return envelope;
      const prepared = await staging.prepare(envelope);
      return JSON.parse(await prepared.body.text());
    },
    async sendMessage(message) {
      if (message.type === "polylogue.providerRateLimited") {
        if (message.provider !== owner.provider || message.provider_response?.status !== 429 || !message.request_id) {
          return { ok: false, error: "provider_rate_limit_response_invalid" };
        }
        if (message.claim) {
          const meta = await staging.metadata(message.claim.id);
          staging.requireOwner(meta, message.claim, owner);
          const source = new URL(message.provider_response.url);
          const providerHost = owner.provider === "chatgpt" ? "chatgpt.com" : owner.provider === "claude-ai" ? "claude.ai" : "grok.com";
          if (source.protocol !== "https:" || source.hostname !== providerHost) return { ok: false, error: "provider_rate_limit_response_invalid" };
        }
        return { ok: true };
      }
      if (message.type === "polylogue.nativeBundle.begin") {
        const bundle = await store.beginNativeBundle({ owner, provider: message.provider, nativeId: message.native_id,
          bundleId: message.bundle_id, requiredReplies: ["conversation", "responses"] });
        return { ok: true, bundle_ref: bundle.id, replies: bundle.replies };
      }
      if (message.type === "polylogue.nativeBundle.outcome") {
        await store.publishNativeBundleReply(message.bundle_ref, owner, message.name, null, message.outcome);
        await store.finishNativeBundle(message.bundle_ref, owner); return { ok: true };
      }
      if (message.type === "polylogue.nativeBundle.finish") {
        const controller = new globalThis.AbortController();
        const promise = normalizer.finishBundle(message.bundle_ref, owner, { pin: true, signal: controller.signal });
        operations.set(message.bundle_ref, { controller, promise });
        try { const result = await promise; return { ok: true, acquisition: result.acquisition }; }
        catch (error) { return { ok: false, error: error.code || error.message }; }
        finally { operations.delete(message.bundle_ref); }
      }
      if (message.type === "polylogue.nativeBundle.cancel") {
        const operation = operations.get(message.bundle_ref);
        if (operation) { operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError")); await operation.promise.catch(() => undefined); }
        return { ok: true };
      }
      if (message.type === "polylogue.capture" && message.envelope?.capture_record_ref) {
        message.envelope = await this.materialize(message.envelope);
        return undefined;
      }
      if (message.type === "polylogue.cancelNativeCapture") {
        const operation = operations.get(message.raw_ref.id);
        if (operation) {
          operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
          await dispatch?.({ type: "polylogue.cancelRecordAssets", capture_ref: message.raw_ref.id });
          await operation.promise.catch(() => undefined);
        }
        return { ok: true, outcome: "cancelled" };
      }
      if (message.type === "polylogue.restoreNativeCapture") {
        let selected = null;
        for await (const meta of staging.metadataEntries()) {
          if (meta.state !== "sealed" || meta.kind !== "native-response" || meta.owner.provider !== message.provider || meta.capture_bundle) continue;
          const ref = { id: meta.id, token: meta.token }; const headers = await normalizer.headers(ref);
          if (String(headers.uuid || headers.conversation_id || headers.id || "") !== message.native_id) continue;
          if (!selected || selected.acquisitionSequence < meta.acquisition_sequence) selected = { ok: true, bodyRef: ref, headers,
            url: meta.source_url, capturedAt: meta.created_at, acquisitionSequence: meta.acquisition_sequence };
        }
        return { ok: true, capture: selected };
      }
      if (message.type === "polylogue.nativeCaptureHeader") {
        const headers = await normalizer.headers(message.related_refs?.conversation || message.raw_ref);
        const meta = await staging.metadata(message.raw_ref.id);
        const nativeId = String(headers.uuid || headers.conversation_id || headers.conversationId || headers.id);
        const pinned = await store.pinNativeCache({ owner, provider: message.provider, nativeId, rawRef: message.raw_ref,
          relatedRefs: message.related_refs || {}, acquisition: message.acquisition || null, headers, observedAt: meta.created_at, acquisitionSequence: meta.acquisition_sequence });
        const current = pinned.current; const selected = await staging.metadata(current.raw_ref.id);
        const describe = (ref, related, acquisition, revisionHeaders, revisionMeta) => ({ ok: true, bodyRef: ref,
          relatedRefs: related, acquisition, nativeId,
          providerUpdatedAt: revisionHeaders.update_time ?? revisionHeaders.updated_at ?? revisionHeaders.updatedAt ?? revisionHeaders.modifyTime ?? null,
          url: revisionMeta.source_url, capturedAt: revisionMeta.created_at, acquisitionSequence: revisionMeta.acquisition_sequence });
        return { ok: true, headers, content_sha256: meta.sha256,
          capture: describe(message.raw_ref, message.related_refs || {}, message.acquisition || null, headers, meta),
          cache: { headers: current.headers, capture: describe(current.raw_ref, current.related_refs, current.acquisition, current.headers, selected) } };
      }
      if (message.type === "polylogue.releaseNativeCache") return { ok: true };
      if (message.type === "polylogue.normalizeNativeCapture" || message.type === "polylogue.nativeCaptureSummary") {
        const controller = new globalThis.AbortController();
        const promise = normalizer.normalize({ provider: message.provider, rawRef: message.raw_ref, nativeId: message.native_id, extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", attribution: message.attribution, relatedRefs: message.related_refs, acquisition: message.acquisition,
          signal: controller.signal, summaryOnly: message.type === "polylogue.nativeCaptureSummary" });
        operations.set(message.raw_ref.id, { controller, promise });
        try { const result = await promise; return message.type === "polylogue.nativeCaptureSummary" ? { ok: true, ...result } : { ok: true, envelope: result }; }
        catch (error) { return { ok: false, error: error.code || error.message }; }
        finally { operations.delete(message.raw_ref.id); }
      }
      if (!message.type?.startsWith("polylogue.asset.")) return undefined;
      try {
        if (message.type === "polylogue.asset.begin") {
          const acquisition = message.acquisition;
          const identity = acquisition ? [message.provider, acquisition.raw_id, acquisition.native_id, acquisition.record_key,
            acquisition.attachment_id, acquisition.attachment_ordinal] : null;
          const ref = await staging.begin(owner, { ...message, acquisition: identity }, message.request_id);
          if (identity) {
            const claim = await store.reserveAcquisition(identity, ref, message.request_id);
            if (claim.result) return { ok: true, result: claim.result };
            if (claim.producer_id !== message.request_id) return { ok: false, error: "capture_acquisition_in_progress" };
          }
          return { ok: true, ref };
        }
        if (message.type === "polylogue.asset.chunk") return { ok: true, ...await staging.append(message.ref, owner, message.sequence, message.base64) };
        if (message.type === "polylogue.asset.seal") return { ok: true, asset: await staging.seal(message.ref, owner, message.result) };
        if (message.type === "polylogue.asset.discard") {
          await staging.cancel(message.ref, owner); return { ok: true };
        }
      } catch (error) { return { ok: false, error: error.code || error.name || error.message }; }
      return undefined;
    },
  };
}

export async function attachmentBytes(staging, asset) {
  return new Uint8Array(await (await staging.file(asset.staged_asset.id)).arrayBuffer());
}
