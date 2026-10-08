import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { CaptureStaging } from "../src/capture/staging.js";
import { NativeCaptureNormalizer } from "../src/capture/native.js";
import { memoryOriginStorage, receiverContractPreparation, stagingRuntime } from "./infra/capture-staging.js";

const owner = { tab_id: 42, document_id: "synthetic-document", provider: "chatgpt" };
function digest(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function acceptedReceipt(requestId, contentHash) { return { receiver_request_id: requestId, content_hash: contentHash, submitted_content_hash: contentHash, outcome: "accepted" }; }

describe("durable capture staging", () => {
  it.each([
    '{"id":"session","is_temporary":true}',
    '{"id":"session","is_temporary":true,"mapping":null}',
    '{"id":"session","is_temporary":true,"mapping":[]}',
    '{"id":"session","is_temporary":true,"metadata":{"mapping":{}}}',
    '{"id":"session","mapping":{},"mapping":false}',
  ])("refuses a malformed root mapping before advertising native identity: %s", async (raw) => {
    const { staging, store } = stagingRuntime();
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(raw).toString("base64")); await staging.seal(ref, owner);
    const normalizer = new NativeCaptureNormalizer({ staging, store });
    await expect(normalizer.headers(ref)).rejects.toMatchObject({ code: "native_capture_mapping_invalid" });
    expect(await (await staging.file(ref.id)).text()).toBe(raw);
  });
  it.each([
    '{"id":"session","mapping":{}}',
    '{"mapping":{"node":{"message":{"content":{"parts":["neutral"]}}}},"id":"session"}',
    '{"id":"session","mapping":null,"mapping":{}}',
  ])("reads only header facts from a valid root mapping: %s", async (raw) => {
    const { staging, store } = stagingRuntime();
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(raw).toString("base64")); await staging.seal(ref, owner);
    expect(await new NativeCaptureNormalizer({ staging, store }).headers(ref)).toEqual({ id: "session" });
    expect(await (await staging.file(ref.id)).text()).toBe(raw);
  });
  it("refuses missing original source evidence and never attributes old bytes to the current tab", async () => {
    const { staging, store } = stagingRuntime();
    const raw = await staging.begin(owner, { kind: "native-response" });
    await staging.append(raw, owner, 0, globalThis.Buffer.from('{"id":"session","mapping":{}}').toString("base64"));
    await staging.seal(raw, owner);
    let prepared = false;
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: async () => { prepared = true; } });
    await expect(normalizer.normalize({ provider: "chatgpt", rawRef: raw, nativeId: "session",
      extensionVersion: "test", instanceId: "current-preparation-instance" }))
      .rejects.toThrow("native_original_document_unavailable");
    expect(prepared).toBe(false);
    expect(await (await staging.file(raw.id)).text()).toBe('{"id":"session","mapping":{}}');
    expect(await store.captureReferences(raw.id)).toBe(true);
  });

  it("binds raw admission to the durable invocation and retains acquired bytes after invocation close", async () => {
    const { staging, store } = stagingRuntime();
    const invocation = await store.reserveCaptureObservation(owner, "requested-session", "native-invocation", "synthetic-worker");
    const claim = await store.getCapture(invocation.id);
    const purpose = { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/requested-session", invocation_ref: invocation };
    const raw = await staging.begin(owner, purpose, "synthetic-native-request");
    expect(await staging.metadata(raw.id)).toMatchObject({
      invocation_ref: invocation, invocation_native_id: "requested-session",
      acquisition_sequence: claim.acquisition_sequence, observed_at: claim.observed_at,
    });
    await expect(staging.begin(owner, { ...purpose, invocation_ref: { ...invocation, token: "wrong-token" } }, "synthetic-native-request"))
      .rejects.toThrow("capture_staging_owner_mismatch");
    await store.closeNativeInvocation(invocation, false);
    await expect(staging.begin(owner, purpose, "synthetic-native-request")).rejects.toThrow("native_invocation_owner_mismatch");
    await expect(staging.begin(owner, purpose, "different-native-request")).rejects.toThrow("native_invocation_owner_mismatch");
    await staging.append(raw, owner, 0, globalThis.Buffer.from('{"id":"requested-session"}').toString("base64"));
    await staging.seal(raw, owner);
    await store.closeNativeInvocation(invocation);
    expect(await store.getCapture(invocation.id)).toBeUndefined();
    expect(await store.getCapture(`raw:${raw.id}`)).toMatchObject({ state: "pending-normalization", invocation_ref: invocation });
    const restarted = new CaptureStaging(staging.storage, store);
    expect(await (await restarted.file(raw.id)).text()).toBe('{"id":"requested-session"}');
  });

  it("cancels a pending staged raw read instead of reporting a completed payload", async () => {
    const { staging } = stagingRuntime();
    const ref = await staging.begin(owner);
    await staging.append(ref, owner, 0, "eA=="); await staging.seal(ref, owner);
    let cancelled = false;
    const pending = new globalThis.ReadableStream({ cancel() { cancelled = true; } });
    let opened;
    const fileOpened = new Promise((resolve) => { opened = resolve; });
    staging.file = async () => { opened(); return { stream: () => pending }; };
    const controller = new globalThis.AbortController();
    const reading = staging.rawJson({ staged_raw_json: ref.id }, [ref.id], controller.signal).next();
    const refused = expect(reading).rejects.toMatchObject({ name: "AbortError" });
    await fileOpened;
    controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    await refused;
    expect(cancelled).toBe(true);
    expect(pending.locked).toBe(false);
  });

  it.each(["chatgpt", "claude-ai"])("rejects mismatched %s native identity before acquiring assets", async (provider) => {
    const { staging, store } = stagingRuntime(); const rawOwner = { ...owner, provider };
    const ref = await staging.begin(rawOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    const payload = provider === "chatgpt" ? { conversation_id: "other-session", mapping: { node: { message: { id: "message", author: { role: "assistant" }, content: { parts: ["synthetic"] } } } } }
      : { uuid: "other-session", chat_messages: [{ uuid: "message", sender: "assistant", text: "synthetic" }] };
    await staging.append(ref, rawOwner, 0, globalThis.Buffer.from(JSON.stringify(payload)).toString("base64")); await staging.seal(ref, rawOwner);
    let assets = 0;
    await expect(new NativeCaptureNormalizer({ staging, store, prepareNative: async () => { assets++; throw new Error("unreachable receiver preparation"); } }).normalize({
      provider, rawRef: ref, nativeId: "declared-session", extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal,
    })).rejects.toThrow("native_capture_identity_mismatch");
    expect(assets).toBe(0);
    expect(await (await staging.file(ref.id)).text()).toBe(JSON.stringify(payload));
    expect(await store.getCapture(`raw:${ref.id}`)).toMatchObject({ state: "pending-normalization" });
  });

  it("acknowledges same-byte retransmission after restart regardless of base64 padding and refuses conflicting bytes", async () => {
    const { staging, store, storage } = stagingRuntime();
    const ref = await staging.begin(owner, {}, "synthetic-retransmission");
    await staging.append(ref, owner, 0, "eA==");
    const restarted = new CaptureStaging(storage, store);
    expect(await restarted.append(ref, owner, 0, "eA")).toEqual({ sequence: 1, size_bytes: 1 });
    await expect(restarted.append(ref, owner, 0, "eQ==")).rejects.toMatchObject({ code: "capture_staging_sequence_mismatch" });
    await restarted.seal(ref, owner);
    expect(await (await restarted.file(ref.id)).text()).toBe("x");
  });

  it("retries an empty unpublished metadata close but preserves malformed metadata with acquired parts and custody", async () => {
    const { staging, store, storage } = stagingRuntime();
    storage.directory.failClose = (name) => name.endsWith(".json");
    await expect(staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" }, "synthetic-operation")).rejects.toMatchObject({ name: "QuotaExceededError" });
    storage.directory.failClose = null;
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" }, "synthetic-operation");
    await staging.append(ref, owner, 0, globalThis.Buffer.from("synthetic retained bytes").toString("base64"));
    const part = storage.files.get(`${ref.id}.0.part`).slice();
    storage.files.set(`${ref.id}.json`, new globalThis.TextEncoder().encode("{"));
    await expect(new CaptureStaging(storage, store).begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" }, "synthetic-operation"))
      .rejects.toMatchObject({ code: "capture_staging_interrupted", name: "SyntaxError" });
    expect(storage.files.get(`${ref.id}.0.part`)).toEqual(part);
    expect(new globalThis.TextDecoder().decode(storage.files.get(`${ref.id}.json`))).toBe("{");
    expect(await store.getCapture(`raw:${ref.id}`)).toMatchObject({ raw_ref: ref, state: "acquiring" });
  });

  it("reconciles the current serialized raw state without resurrecting cancelled acquisitions", async () => {
    const { staging, store } = stagingRuntime();
    const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    const stale = await staging.metadata(raw.id);
    await staging.append(raw, owner, 0, globalThis.Buffer.from('{"conversation_id":"session","mapping":{}}').toString("base64"));
    await staging.seal(raw, owner);
    await staging.publishNativeAcquisition(stale);
    expect(await store.getCapture(`raw:${raw.id}`)).toMatchObject({ state: "pending-normalization" });
    const cancelled = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    await staging.cancel(cancelled, owner);
    await staging.reconcileNativeAcquisition(cancelled.id);
    expect(await store.getCapture(`raw:${cancelled.id}`)).toBeUndefined();
  });

  it("recovers sealed single-response custody after publication loss and retains bytes after document loss until ACK", async () => {
    const { staging, store, storage } = stagingRuntime();
    const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/session" });
    await staging.append(raw, owner, 0, globalThis.Buffer.from('{"conversation_id":"session","mapping":{}}').toString("base64"));
    const publish = staging.publishNativeAcquisition.bind(staging); let interrupted = true;
    staging.publishNativeAcquisition = async (meta) => {
      if (meta.state === "sealed" && interrupted) { interrupted = false; throw new Error("synthetic_publication_interrupted"); }
      return publish(meta);
    };
    await expect(staging.seal(raw, owner)).rejects.toThrow("synthetic_publication_interrupted");
    const writes = storage.writes.length;
    const restarted = new CaptureStaging(storage, store);
    await restarted.seal(raw, owner);
    expect(storage.writes.length).toBe(writes);
    expect((await store.getCapture(`raw:${raw.id}`)).state).toBe("pending-normalization");
    for await (const ref of store.releaseNativeCaches(owner)) await restarted.discardUnreferenced(ref?.id || ref);
    await restarted.discardUnreferenced(raw.id);
    expect(await (await restarted.file(raw.id)).text()).toBe('{"conversation_id":"session","mapping":{}}');
    const envelope = await new NativeCaptureNormalizer({ staging: restarted, store, prepareNative: receiverContractPreparation(restarted, store) }).normalize({ provider: owner.provider, rawRef: raw,
      nativeId: "session", extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    expect(await store.getCapture(`raw:${raw.id}`)).toBeUndefined();
    const capture = await store.getCapture(envelope.capture_record_ref);
    await store.putCapture({ ...capture, receiver_receipt: acceptedReceipt("synthetic-ack", envelope.receiver_native.sha256) });
    await restarted.acknowledgeNative(capture.id);
    expect(storage.files.size).toBe(0);
  });

  it("preserves an older acquired bundle until foreground normalization transfers custody despite a newer cache", async () => {
    const { staging, store } = stagingRuntime();
    const grokOwner = { ...owner, provider: "grok" };
    const bundle = await store.beginNativeBundle({ owner: grokOwner, provider: "grok", nativeId: "session", bundleId: "older-operation",
      requiredReplies: ["conversation", "responses"] });
    const refs = {};
    for (const [name, value] of Object.entries({ conversation: { conversationId: "session" }, responses: { responses: [] } })) {
      const ref = await staging.begin(grokOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native", capture_bundle: { id: bundle.id, name } });
      await staging.append(ref, grokOwner, 0, globalThis.Buffer.from(JSON.stringify(value)).toString("base64"));
      await staging.seal(ref, grokOwner); refs[name] = ref;
    }
    const newer = await staging.begin(grokOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    await staging.append(newer, grokOwner, 0, globalThis.Buffer.from('{"responses":[]}').toString("base64")); await staging.seal(newer, grokOwner);
    const newerMeta = await staging.metadata(newer.id);
    await store.pinNativeCache({ owner: grokOwner, provider: "grok", nativeId: "session", rawRef: newer,
      relatedRefs: { conversation: refs.conversation }, headers: { conversationId: "session" }, observedAt: newerMeta.created_at,
      acquisitionSequence: newerMeta.acquisition_sequence });
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) });
    const completed = await normalizer.finishBundle(bundle.id, grokOwner, { pin: true });
    expect(completed.rawRef).toEqual(refs.responses);
    expect((await store.getCapture(bundle.id)).state).toBe("ready");
    await staging.discardUnreferenced(refs.responses.id);
    expect(await staging.file(refs.responses.id)).toBeDefined();
    const envelope = await normalizer.normalize({ provider: "grok", rawRef: completed.rawRef, relatedRefs: completed.relatedRefs,
      acquisition: completed.acquisition, nativeId: "session", extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance",
      signal: new globalThis.AbortController().signal });
    expect(await store.getCapture(bundle.id)).toBeUndefined();
    expect((await store.getCapture(envelope.capture_record_ref)).delivery_kind).toBe("foreground");
    expect(await store.captureReferences(refs.responses.id)).toBe(true);
  });

  it.each(["chatgpt", "claude-ai", "grok"])("retains original %s members after canonical receiver refusal", async (provider) => {
    const { staging, store } = stagingRuntime();
    const rawOwner = { ...owner, provider };
    const payload = { id: "session", uuid: "session", conversationId: "session", opaque: [null, false, []], ...(provider === "chatgpt" ? { mapping: {} } : {}) };
    const raw = await staging.begin(rawOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    await staging.append(raw, rawOwner, 0, globalThis.Buffer.from(JSON.stringify(payload)).toString("base64"));
    await staging.seal(raw, rawOwner);
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: async () => {
      throw new Error("native capture has no canonical messages");
    } });
    await expect(normalizer.normalize({ provider, nativeId: "session", rawRef: raw, extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance",
      signal: new globalThis.AbortController().signal })).rejects.toThrow("native capture has no canonical messages");
    expect(await (await staging.file(raw.id)).text()).toBe(JSON.stringify(payload));
    expect(await store.captureReferences(raw.id)).toBe(true);
  });

  it("retries interrupted receiver preparation with the retained acquisition and original document", async () => {
    const { staging, store } = stagingRuntime();
    const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    const payload = { id: "session", mapping: { node: { message: { id: "message", author: { role: "assistant" }, content: { parts: ["synthetic"] } } } } };
    await staging.append(raw, owner, 0, globalThis.Buffer.from(JSON.stringify(payload)).toString("base64"));
    await staging.seal(raw, owner);
    const options = { provider: "chatgpt", rawRef: raw, nativeId: "session",
      extensionVersion: "0.1.0", instanceId: "first-preparation-instance", signal: new globalThis.AbortController().signal };
    let retained;
    const contract = { onPrepare: ({ capture, reference }) => {
      retained = { capture, reference }; throw new Error("synthetic-preparation-interrupted");
    } };
    await expect(new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store, contract) })
      .normalize(options)).rejects.toThrow("synthetic-preparation-interrupted");
    const interrupted = await store.getCapture(retained.capture.id);
    expect(interrupted.receiver_native).toEqual(retained.reference);
    expect(await store.captureReferences(raw.id)).toBe(true);
    const envelope = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) })
      .normalize({ ...options, instanceId: "later-worker-instance" });
    expect(envelope.receiver_native.acquisition_id).toBe(retained.capture.acquisition_id);
    expect(envelope.receiver_native.preparation_instance_id).toBe("first-preparation-instance");
    expect((await store.getCapture(envelope.capture_record_ref)).owner).toEqual(owner);
    expect(envelope.provenance.extension_instance_id).toBeNull();
    expect(envelope.provenance.acquisition_sequence).toBeNull();
    expect(await (await staging.file(raw.id)).text()).toBe(JSON.stringify(payload));
  });

  it("recovers a sealed bundle reply when worker loss precedes membership publication", async () => {
    const storage = memoryOriginStorage(); const { staging, store } = stagingRuntime(storage);
    const grokOwner = { ...owner, provider: "grok" };
    const bundle = await store.beginNativeBundle({ owner: grokOwner, provider: "grok", nativeId: "session", bundleId: "interrupted-operation", requiredReplies: ["conversation", "responses"] });
    const ref = await staging.begin(grokOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native", capture_bundle: { id: bundle.id, name: "conversation" } });
    const bytes = JSON.stringify({ conversationId: "session" });
    await staging.append(ref, grokOwner, 0, globalThis.Buffer.from(bytes).toString("base64"));
    const publish = store.publishNativeBundleReply.bind(store);
    store.publishNativeBundleReply = async () => { throw new Error("synthetic-worker-loss"); };
    await expect(staging.seal(ref, grokOwner)).rejects.toThrow("synthetic-worker-loss");
    expect((await staging.metadata(ref.id)).state).toBe("sealed");
    expect((await store.getCapture(bundle.id)).replies).toEqual({});
    const writes = storage.writes.filter((name) => name.endsWith(".bytes")).length;
    store.publishNativeBundleReply = publish;
    const restarted = new CaptureStaging(storage, store);
    await restarted.seal(ref, grokOwner);
    expect((await store.getCapture(bundle.id)).replies.conversation).toEqual(ref);
    const resumed = await store.beginNativeBundle({ owner: grokOwner, provider: "grok", nativeId: "session", bundleId: "next-attempt",
      requiredReplies: ["conversation", "responses"] });
    expect(resumed.id).toBe(bundle.id);
    expect(resumed.replies.conversation).toEqual(ref);
    expect(await (await restarted.file(ref.id)).text()).toBe(bytes);
    expect(storage.writes.filter((name) => name.endsWith(".bytes")).length).toBe(writes);
  });

  it("keeps the newer same-clock cache revision when older header parsing completes later", async () => {
    const { staging, store } = stagingRuntime();
    const old = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    const next = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    const oldMeta = await staging.metadata(old.id); const nextMeta = await staging.metadata(next.id);
    const common = { owner, provider: "chatgpt", nativeId: "session", headers: { id: "session" }, observedAt: "2026-01-01T00:00:00Z" };
    await store.pinNativeCache({ ...common, rawRef: next, acquisitionSequence: nextMeta.acquisition_sequence });
    const delayed = await store.pinNativeCache({ ...common, rawRef: old, acquisitionSequence: oldMeta.acquisition_sequence });
    expect(delayed.current.raw_ref).toEqual(next);
    expect(delayed.retired).toEqual([]);
    expect(await store.captureReferences(next.id)).toBe(true);
  });

  it("retains sealed bundle replies after lease loss and atomically transfers resumed custody", async () => {
    const storage = memoryOriginStorage(); const { staging, store } = stagingRuntime(storage);
    const grokOwner = { ...owner, provider: "grok" };
    const job = { id: "bundle-job", provider: "grok", status: "running", execution_owner: "first", execution_generation: 1 };
    const item = { id: "bundle-item", job_id: job.id, provider: "grok", native_id: "synthetic-conversation", state: "fetching", lease_owner: "first" };
    await store.putJob(job); await store.putQueue(item);
    const context = { jobId: job.id, itemId: item.id, owner: "first", generation: 1 };
    const bundle = await store.beginNativeBundle({ owner: grokOwner, provider: "grok", nativeId: item.native_id, bundleId: "operation",
      requiredReplies: ["conversation", "responses"], queueContext: context });
    expect((await store.getQueue(item.id)).capture_bundle_ref).toBe(bundle.id);
    await store.putJob({ ...job, execution_owner: "second", execution_generation: 2 });
    await store.putQueue({ ...await store.getQueue(item.id), lease_owner: "second" });
    const refs = {};
    for (const [name, value] of Object.entries({ conversation: { conversationId: item.native_id }, responses: { responses: [] } })) {
      const ref = await staging.begin(grokOwner, { kind: "native-response", source_url: "https://synthetic.invalid/native", capture_bundle: { id: bundle.id, name } });
      await staging.append(ref, grokOwner, 0, globalThis.Buffer.from(JSON.stringify(value)).toString("base64"));
      await staging.seal(ref, grokOwner); refs[name] = ref;
    }
    expect((await store.getQueue(item.id)).capture_bundle_replies).toBeUndefined();
    expect((await store.getCapture(bundle.id)).replies).toEqual(refs);
    const nextContext = { ...context, owner: "second", generation: 2 };
    const resumed = await store.beginNativeBundle({ owner: grokOwner, provider: "grok", nativeId: item.native_id,
      bundleId: "new-request-must-not-replace-operation", requiredReplies: ["conversation", "responses"], queueContext: nextContext });
    expect(resumed.id).toBe(bundle.id);
    const writes = storage.writes.length;
    const normalizer = new NativeCaptureNormalizer({ staging: new CaptureStaging(storage, store), store, prepareNative: receiverContractPreparation(staging, store) });
    await normalizer.finishBundle(bundle.id, grokOwner);
    const envelope = await normalizer.normalize({ provider: "grok", rawRef: refs.responses, relatedRefs: { conversation: refs.conversation },
      nativeId: item.native_id, extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal,
      acquisition: { kind: "grok-endpoint-bundle" }, queueContext: nextContext });
    expect(storage.writes.length).toBe(writes);
    expect(await store.getCapture(bundle.id)).toBeUndefined();
    const acquired = await store.getQueue(item.id);
    expect(acquired.capture_record_ref).toBe(envelope.capture_record_ref);
    expect(acquired.capture_bundle_ref).toBeNull();
    expect(acquired.capture_source_refs).toEqual(expect.arrayContaining(Object.values(refs).map((ref) => ref.id)));
    for (const ref of Object.values(refs)) expect(await store.captureReferences(ref.id)).toBe(true);
  });

  it("preserves literal staging-like keys in retained native bytes through receiver preparation", async () => {
    const { staging, store } = stagingRuntime();
    const literal = Object.fromEntries(["staged_raw_json", "staged_asset", "staged_turns", "staged_asset_failures", "staged_session_attachments", "capture_body_ref", "capture_source_refs", "capture_record_ref", "capture_summary"].map((key) => [key, { nested: [key, { staged_raw_json: "ordinary provider value" }] }]));
    const payload = { conversation_id: "synthetic-session", current_node: "message", mapping: {
      message: { parent: null, children: [], message: { id: "message", author: { role: "assistant" }, recipient: "synthetic_tool", content: { content_type: "text", parts: [JSON.stringify(literal)] } } },
    } };
    const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    await staging.append(raw, owner, 0, globalThis.Buffer.from(JSON.stringify(payload)).toString("base64"));
    await staging.seal(raw, owner);
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) });
    const envelope = await normalizer.normalize({ provider: "chatgpt", rawRef: raw, nativeId: "synthetic-session", extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    expect(envelope.capture_source_refs).toContain(raw.id);
    expect(envelope.session.turns).toEqual([]);
    expect(envelope.receiver_native).toMatchObject({ acquisition_id: expect.any(String), sha256: "b".repeat(64) });
    expect(JSON.parse(await (await staging.file(raw.id)).text())).toEqual(payload);
    expect(await store.captureReferences(raw.id)).toBe(true);
  });

  it("hashes attachment bytes independently of staging identity and keeps nested metadata literal", async () => {
    const { staging } = stagingRuntime();
    const assets = [];
    for (let i = 0; i < 2; i++) {
      const ref = await staging.begin(owner);
      await staging.append(ref, owner, 0, "Ynl0ZXM=");
      assets.push({ ...await staging.seal(ref, owner), provider_meta: { staged_asset: { id: "provider identifier" }, capture_summary: "provider metadata" } });
    }
    const turns = assets.map((asset) => ({ role: "assistant", text: "same", blocks: [], attachments: [asset] }));
    expect(await staging.semanticTurnDigest(turns[0])).toBe(await staging.semanticTurnDigest(turns[1]));
    const inline = { ...assets[0] };
    delete inline.staged_asset;
    inline.inline_base64 = "Ynl0ZXM=";
    expect(await staging.semanticTurnDigest(turns[0])).toBe(await staging.semanticTurnDigest({ ...turns[0], attachments: [inline] }));
    const prepared = await staging.prepare({ session: { turns } });
    const delivered = JSON.parse(await prepared.body.text());
    for (const turn of delivered.session.turns) {
      expect(turn.attachments[0].inline_base64).toBe("Ynl0ZXM=");
      expect(turn.attachments[0].provider_meta).toEqual(assets[0].provider_meta);
    }
  });

  it("retries only an identical committed chunk after worker loss and preserves immutable receiver bytes", async () => {
    const storage = memoryOriginStorage(); const { store } = stagingRuntime(storage); let staging = new CaptureStaging(storage, store);
    const ref = await staging.begin(owner); const bytes = new globalThis.TextEncoder().encode("synthetic bytes 😀");
    const encoded = globalThis.Buffer.from(bytes).toString("base64");
    await staging.append(ref, owner, 0, encoded);
    staging = new CaptureStaging(storage, store);
    expect(await staging.append(ref, owner, 0, encoded)).toEqual({ sequence: 1, size_bytes: bytes.length });
    await expect(staging.append(ref, owner, 0, "YQ==")).rejects.toMatchObject({ code: "capture_staging_sequence_mismatch" });
    await expect(staging.append(ref, { ...owner, document_id: "other" }, 1, encoded)).rejects.toMatchObject({ code: "capture_staging_owner_mismatch" });
    const asset = await staging.seal(ref, owner);
    expect(asset.sha256).toBe(digest(bytes));
    const envelope = { session: { turns: [{ text: "😀".repeat(25000), attachments: [asset] }] } };
    const prepared = await staging.prepare(envelope); const canonical = await prepared.body.text();
    expect(JSON.parse(canonical).session.turns[0].attachments[0].inline_base64).toBe(encoded);
    expect(prepared.contentHash).toBe(digest(canonical));
    const writes = storage.writes.length;
    staging = new CaptureStaging(storage, store);
    const resumed = await staging.prepare(envelope);
    expect(await resumed.body.text()).toBe(canonical); expect(storage.writes).toHaveLength(writes);
    await staging.markAcknowledged(prepared.ref, acceptedReceipt("ack", prepared.contentHash));
    await staging.acknowledge(prepared.ref); expect(storage.files.size).toBe(0);
  });

  it("publishes a resumable conversion once and reports a missing immutable file", async () => {
    const storage = memoryOriginStorage(); const { store } = stagingRuntime(storage); let staging = new CaptureStaging(storage, store);
    const original = { session: { turns: [{ text: "acquired offline evidence" }] } };
    const id = staging.conversionId("polylogue-ext-synthetic");
    const first = await staging.prepare(globalThis.structuredClone(original), id);
    const writes = storage.writes.length;
    staging = new CaptureStaging(storage, store);
    const second = await staging.prepare(globalThis.structuredClone(original), id);
    expect(second.ref).toBe(first.ref); expect(storage.writes).toHaveLength(writes);
    expect(await second.body.text()).toBe(await first.body.text());
    storage.files.delete(`${id}.bytes`);
    await expect(staging.prepare({ capture_body_ref: id })).rejects.toMatchObject({ code: "capture_staging_missing_bytes", cause: expect.objectContaining({ name: "NotFoundError" }) });
  });

  it("does not republish an ACKed delivery while resuming its conversion", async () => {
    const { staging, storage, store } = stagingRuntime();
    const envelope = { session: { turns: [{ text: "already delivered" }] } };
    const delivery = { id: "converted-delivery", delivery_kind: "foreground", queued_at: "2026-01-01T00:00:00Z" };
    const id = staging.conversionId(delivery.id);
    const first = await staging.prepare(globalThis.structuredClone(envelope), id, delivery);
    const receipt = acceptedReceipt("actual-ack", first.contentHash);
    await staging.markAcknowledged(id, receipt);
    await store.deleteDelivery(delivery.id);
    const writes = storage.writes.length;
    const restarted = new CaptureStaging(storage, store);
    const resumed = await restarted.prepare(globalThis.structuredClone(envelope), id, delivery);
    expect(resumed.acknowledgedReceipt).toEqual(receipt);
    expect(await store.getDelivery(delivery.id)).toBeNull();
    expect(storage.writes).toHaveLength(writes);
    expect(await resumed.body.text()).toBe(await first.body.text());
  });

  it("quota refusal leaves prior acquired files and never publishes a partial body", async () => {
    const storage = memoryOriginStorage(); const { store } = stagingRuntime(storage); const staging = new CaptureStaging(storage, store);
    const good = await staging.prepare({ session: { turns: [{ text: "retained" }] } });
    const original = await good.body.text();
    storage.directory.failClose = (name) => name.endsWith(".bytes") && !name.startsWith(good.ref);
    await expect(staging.prepare({ session: { turns: [{ text: "new" }] } })).rejects.toMatchObject({ name: "QuotaExceededError" });
    expect(await (await staging.file(good.ref)).text()).toBe(original);
  });
  it("resumes a seal after close failure and never rewrites sealed evidence after result publication interruption", async () => {
    const storage = memoryOriginStorage(); const { store } = stagingRuntime(storage);
    let staging = new CaptureStaging(storage, store);
    const identity = ["chatgpt", "raw-revision", "native-session", "native-record", "attachment"];
    const ref = await staging.begin(owner, { acquisition: identity }, "request-nonce");
    expect(await staging.begin(owner, { acquisition: identity }, "request-nonce")).toEqual(ref);
    await store.reserveAcquisition(identity, ref, "request-nonce");
    const parts = [globalThis.Buffer.from("first"), globalThis.Buffer.from("second"), globalThis.Buffer.from("last")];
    for (let sequence = 0; sequence < parts.length; sequence++) await staging.append(ref, owner, sequence, parts[sequence].toString("base64"));
    const result = { status: "acquired", asset: { streamed: true, media_type: "text/plain" } };
    storage.directory.failClose = (name) => name === `${ref.id}.bytes`;
    await expect(staging.seal(ref, owner, result)).rejects.toMatchObject({ name: "QuotaExceededError" });
    expect((await staging.metadata(ref.id)).state).toBe("acquiring");
    expect(parts.every((_part, sequence) => storage.files.has(`${ref.id}.${sequence}.part`))).toBe(true);
    storage.directory.failClose = null;
    const publish = store.publishAcquisition.bind(store);
    store.publishAcquisition = async () => { throw new Error("synthetic_worker_loss"); };
    staging = new CaptureStaging(storage, store);
    await expect(staging.seal(ref, owner)).rejects.toThrow("synthetic_worker_loss");
    expect((await staging.metadata(ref.id)).state).toBe("sealed");
    const finalWrites = storage.writes.filter((name) => name === `${ref.id}.bytes`).length;
    store.publishAcquisition = publish;
    staging = new CaptureStaging(storage, store);
    await staging.seal(ref, owner);
    expect(storage.writes.filter((name) => name === `${ref.id}.bytes`)).toHaveLength(finalWrites);
    expect(await (await staging.file(ref.id)).text()).toBe("firstsecondlast");
    expect(storage.files.has(`${ref.id}.0.part`)).toBe(false);
    expect((await store.getCapture(`asset:${JSON.stringify(identity)}`)).result.asset.sha256).toBe(digest(globalThis.Buffer.concat(parts)));
  });

  it("cancels and drains an active seal while preserving committed acquisition evidence for restart", async () => {
    const { staging, store, storage } = stagingRuntime();
    const identity = [owner.provider, "raw-revision", "session", "record", "attachment"];
    const ref = await staging.begin(owner, { acquisition: identity }, "cancel-seal");
    await store.reserveAcquisition(identity, ref, "cancel-seal");
    await staging.append(ref, owner, 0, "Zmlyc3Q=");
    await staging.append(ref, owner, 1, "c2Vjb25k");
    let started; const writing = new Promise((resolve) => { started = resolve; });
    let release; const blocked = new Promise((resolve) => { release = resolve; });
    const fileHandle = storage.directory.getFileHandle.bind(storage.directory);
    storage.directory.getFileHandle = async (name, options) => {
      const handle = await fileHandle(name, options);
      if (name !== `${ref.id}.bytes`) return handle;
      const writable = handle.createWritable.bind(handle);
      return { ...handle, createWritable: async (...args) => {
        const writer = await writable(...args); const write = writer.write.bind(writer);
        return { ...writer, write: async (value) => { started(); await blocked; return write(value); } };
      } };
    };
    const sealing = staging.seal(ref, owner, { status: "acquired", asset: { streamed: true } }).then((asset) => ({ asset }), (error) => ({ error }));
    await writing;
    await expect(staging.cancel(ref, { ...owner, document_id: "unrelated" })).rejects.toMatchObject({ code: "capture_staging_owner_mismatch" });
    let entered; const cancellationOwned = new Promise((resolve) => { entered = resolve; });
    const serialize = staging.serialize.bind(staging);
    staging.serialize = (stageId, work) => { entered(); return serialize(stageId, work); };
    const cancellation = staging.cancel(ref, owner);
    await cancellationOwned;
    release();
    expect((await sealing).error).toMatchObject({ name: "AbortError" });
    await cancellation;
    expect((await staging.metadata(ref.id)).state).toBe("acquiring");
    expect(storage.files.has(`${ref.id}.0.part`)).toBe(true);
    expect(storage.files.has(`${ref.id}.1.part`)).toBe(true);
    const restarted = new CaptureStaging(storage, store);
    const asset = await restarted.seal(ref, owner);
    expect(asset.sha256).toBe(digest("firstsecond"));
    expect(await (await restarted.file(ref.id)).text()).toBe("firstsecond");
  });

  it("keeps ACKed acquisition bytes while the page cache or another delivery owns them", async () => {
    const { staging, store, storage } = stagingRuntime();
    const raw = await staging.begin(owner, { kind: "native-response", source_url: "https://synthetic.invalid/native" });
    await staging.append(raw, owner, 0, globalThis.Buffer.from('{"conversation_id":"session","mapping":{}}').toString("base64")); await staging.seal(raw, owner);
    const cache = await store.pinNativeCache({ owner, provider: owner.provider, nativeId: "session", rawRef: raw, headers: {}, observedAt: "2026-01-01T00:00:00Z", acquisitionSequence: await store.nextNativeAcquisitionSequence() });
    const native = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) }).normalize({ provider: owner.provider, nativeId: "session", rawRef: raw, extensionVersion: "0.1.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    const retainedNative = await store.getCapture(native.capture_record_ref);
    await store.putCapture({ ...retainedNative, receiver_receipt: acceptedReceipt("native", native.receiver_native.sha256) });
    await staging.acknowledgeNative(retainedNative.id);
    const identity = [owner.provider, raw.id, "session", "message", "attachment"];
    const asset = await staging.begin(owner, { acquisition: identity }, "asset-request");
    await store.reserveAcquisition(identity, asset, "asset-request");
    await staging.append(asset, owner, 0, "Ynl0ZXM=");
    const sealed = await staging.seal(asset, owner, { status: "acquired", asset: { streamed: true } });
    const first = await staging.prepare({ session: { turns: [{ text: "one", attachments: [sealed] }] } });
    const second = await staging.prepare({ session: { turns: [{ text: "two", attachments: [sealed] }] } });
    await staging.markAcknowledged(first.ref, acceptedReceipt("first", first.contentHash)); await staging.acknowledge(first.ref);
    expect(await (await staging.file(asset.id)).text()).toBe("bytes");
    await store.discardCapture(cache.current.id); await staging.discardUnreferenced(raw.id);
    expect(await (await staging.file(asset.id)).text()).toBe("bytes");
    await staging.markAcknowledged(second.ref, acceptedReceipt("second", second.contentHash)); await staging.acknowledge(second.ref);
    expect(storage.files.size).toBe(0);
  });

  it("retains a single durable acquisition owner before any provider read and cancels only incomplete evidence", async () => {
    const { staging, store } = stagingRuntime();
    const identity = [owner.provider, "raw", "session", "message", "asset"];
    const first = await staging.begin(owner, { acquisition: identity }, "first");
    const second = await staging.begin(owner, { acquisition: identity }, "second");
    const initial = await store.reserveAcquisition(identity, first, "first");
    const competing = await store.reserveAcquisition(identity, second, "second");
    expect(competing.ref).toEqual(initial.ref); expect(competing.producer_id).toBe("first");
    await staging.cancel(first, owner);
    expect(await store.getCapture(initial.id)).toBeUndefined();
    expect((await store.reserveAcquisition(identity, second, "second")).producer_id).toBe("second");
    await staging.append(second, owner, 0, "YQ==");
    await staging.seal(second, owner, { status: "acquired", asset: { streamed: true } });
    await staging.cancel(second, owner);
    expect(await (await staging.file(second.id)).text()).toBe("a");
  });

  it("checks all shared acquisition roots and failed-revision siblings without losing a later owner", async () => {
    const { store } = stagingRuntime();
    const current = { id: "current", provider: "chatgpt", native_id: "session", raw_revision_sha256: "revision" };
    const expected = [];
    for (let index = 0; index < 64; index++) {
      const id = `failed-${String(index).padStart(3, "0")}`;
      await store.putCapture({ ...current, id, state: "failed" }); expected.push(id);
      await store.reserveAcquisition(["chatgpt", `raw-${index}`, "session", `message-${index}`, "attachment"], { id: "shared-asset" }, `producer-${index}`);
    }
    expect(await store.captureReferences("shared-asset")).toBe(false);
    await store.putCapture({ id: "last-live-root", source_refs: ["raw-63"] });
    expect(await store.captureReferences("shared-asset")).toBe(true);
    const observed = [];
    for await (const row of store.failedNativeCaptures(current)) observed.push(row.id);
    expect(observed).toEqual(expected);
    await store.discardCapture("last-live-root");
    expect(await store.captureReferences("shared-asset")).toBe(false);
  });

  it("resumes exact-record ACK retirement with many shared-asset owners", async () => {
    const { staging, store, storage } = stagingRuntime();
    const ref = await staging.begin(owner);
    await staging.append(ref, owner, 0, globalThis.Buffer.from("shared evidence").toString("base64"));
    const asset = await staging.seal(ref, owner);
    const count = 1024;
    const turns = Array.from({ length: count }, (_, index) => ({
      provider_turn_id: `message-${index}`, text: `record-${index}`, attachments: [asset],
    }));
    const prepared = await staging.prepare({ session: { turns } });
    expect(JSON.parse(await prepared.body.text()).session.turns).toHaveLength(count);
    await staging.markAcknowledged(prepared.ref, acceptedReceipt("many-records", prepared.contentHash));
    const remove = store.deleteCaptureRecord.bind(store);
    let retired = 0;
    store.deleteCaptureRecord = async (...args) => {
      if (retired === 257) throw new Error("synthetic_ack_worker_loss");
      await remove(...args); retired++;
    };
    await expect(staging.acknowledge(prepared.ref)).rejects.toThrow("synthetic_ack_worker_loss");
    expect(retired).toBe(257);
    expect(await (await staging.file(ref.id)).text()).toBe("shared evidence");
    expect(await store.captureReferences(ref.id)).toBe(true);
    store.deleteCaptureRecord = remove;
    await new CaptureStaging(storage, store).acknowledge(prepared.ref);
    expect(storage.files.size).toBe(0);
    expect(await store.captureReferences(ref.id)).toBe(false);
  }, 30_000);

  it("retains publication FIFO at an equal clock through retries and restart", async () => {
    const { staging, store, storage } = stagingRuntime();
    for (const id of ["z-first", "a-second", "m-third"]) {
      const delivery = { id, delivery_kind: "foreground", enqueued_at: "2026-01-01T00:00:00Z", attempts: 0 };
      const prepared = await staging.prepare({ session: { turns: [{ text: id }] } }, null, delivery);
      await store.putDelivery({ ...delivery, body_ref: prepared.ref });
    }
    const first = await store.getDelivery("z-first");
    await store.putDelivery({ ...first, attempts: 99, next_attempt_at: "2026-01-02T00:00:00Z" });
    const restarted = new store.constructor(store.indexedDb, store.databaseName);
    const ids = [];
    for await (const { entry } of restarted.deliveries()) {
      ids.push(entry.id);
      expect(await (await new CaptureStaging(storage, restarted).file(entry.body_ref)).text()).toContain(entry.id);
    }
    expect(ids).toEqual(["z-first", "a-second", "m-third"]);
    expect((await restarted.getDelivery("z-first")).delivery_sequence).toBe(first.delivery_sequence);
    const due = [];
    for await (const { entry } of restarted.deliveries({ dueAt: Date.parse("2026-01-01T01:00:00Z") })) due.push(entry.id);
    expect(due).not.toContain("z-first");
    expect(due).toHaveLength(2);
  });

  it.each(["record-failure", "cancel"])("rolls back typed preparation at a mid-record %s", async (boundary) => {
    const { staging, store } = stagingRuntime();
    const db = await store.database(); const transaction = db.transaction.bind(db);
    const controller = new globalThis.AbortController(); let recordsWritten = 0;
    db.transaction = (...args) => {
      const tx = transaction(...args);
      if (Array.isArray(args[0]) && args[0].includes("capture_records") && args[0].includes("queue")) {
        const objectStore = tx.objectStore.bind(tx);
        tx.objectStore = (name) => {
          const records = objectStore(name);
          if (name === "capture_records") {
            const put = records.put.bind(records);
            records.put = (row) => {
              recordsWritten++;
              if (recordsWritten === 2) {
                if (boundary === "record-failure") throw new Error("synthetic_record_failure");
                controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
              }
              return put(row);
            };
          }
          return records;
        };
      }
      return tx;
    };
    const { id } = await staging.begin({ kind: "receiver-body" });
    await expect(staging.prepare({ session: { turns: [{ text: "first" }, { text: "second" }] } }, id,
      { id: "atomic-delivery", delivery_kind: "foreground" }, controller.signal))
      .rejects.toMatchObject(boundary === "cancel" ? { name: "AbortError" } : { message: "synthetic_record_failure" });
    db.transaction = transaction;
    expect(recordsWritten).toBe(2);
    expect(await store.getDelivery("atomic-delivery")).toBeNull();
    expect(await store.getCapture(id)).toBeUndefined();
    expect(await store.getCapture(`body:${id}`)).toBeUndefined();
    expect(await store.getCaptureRecord(id, "0")).toBeUndefined();
    expect(await store.getCaptureRecord(id, "1")).toBeUndefined();
  });

  it("recovers a committed typed preparation before its OPFS writer opens", async () => {
    const { staging, store, storage } = stagingRuntime();
    const { id } = await staging.begin({ kind: "receiver-body" }); const getFileHandle = storage.directory.getFileHandle.bind(storage.directory);
    let refused = false;
    storage.directory.getFileHandle = async (name, options) => {
      if (name === `${id}.bytes` && !refused) { refused = true; throw new Error("synthetic_writer_loss"); }
      return getFileHandle(name, options);
    };
    await expect(staging.prepare({ session: { turns: [{ text: "retained input" }] } }, id,
      { id: "committed-delivery", delivery_kind: "foreground" })).rejects.toThrow("synthetic_writer_loss");
    const entry = await store.getDelivery("committed-delivery");
    expect(entry).toMatchObject({ preparing: true, body_ref: id });
    expect(await store.getCaptureRecord(id, "0")).toMatchObject({ turn: { text: "retained input" } });
    const restarted = new CaptureStaging(storage, store);
    await restarted.resumeDelivery(entry);
    expect(await store.getDelivery(entry.id)).toMatchObject({ preparing: false, delivery_sequence: entry.delivery_sequence });
    expect(JSON.parse(await (await restarted.file(id)).text()).session.turns).toEqual([{ text: "retained input", ordinal: 0 }]);
    await restarted.markAcknowledged(id, acceptedReceipt("committed-typed-ack", (await restarted.metadata(id)).sha256));
    await store.deleteDelivery(entry.id);
    await restarted.acknowledge(id);
    expect(await store.getCaptureRecord(id, "0")).toBeUndefined();
    expect(await store.getCapture(id)).toBeUndefined();
    expect(await store.getCapture(`body:${id}`)).toBeUndefined();
    expect(storage.files.has(`${id}.bytes`)).toBe(false);
  });

  it.each(["before-close", "ready-metadata", "ready-root", "ready-queue"])(
    "resumes foreground publication after worker loss at %s from durable refs", async (boundary) => {
      const { staging, store, storage } = stagingRuntime();
      const assetRef = await staging.begin(owner);
      await staging.append(assetRef, owner, 0, globalThis.Buffer.from("asset evidence").toString("base64"));
      const asset = await staging.seal(assetRef, owner);
      const bodyId = globalThis.crypto.randomUUID();
      const delivery = { id: "interrupted-delivery", delivery_kind: "foreground", enqueued_at: "2026-01-01T00:00:00Z", attempts: 0 };
      const originalSave = staging.save.bind(staging);
      const originalRoot = store.putCapture.bind(store);
      const originalDelivery = store.putDelivery.bind(store);
      let failed = false;
      const fail = () => { failed = true; throw new Error("synthetic_publication_loss"); };
      staging.save = async (meta) => {
        if (!failed && boundary === "before-close" && meta.state === "receiver-preparing") fail();
        if (!failed && boundary === "ready-metadata" && meta.state === "receiver-ready") fail();
        return originalSave(meta);
      };
      store.putCapture = async (row) => {
        if (!failed && boundary === "ready-root" && row.id === `body:${bodyId}` && row.state === "ready") fail();
        return originalRoot(row);
      };
      store.putDelivery = async (row, root) => {
        if (!failed && boundary === "ready-queue" && row.preparing === false) fail();
        return originalDelivery(row, root);
      };
      await expect(staging.prepare({ session: { turns: [{ text: "retained transcript" }], attachments: [asset] } }, bodyId, delivery))
        .rejects.toThrow("synthetic_publication_loss");
      const entry = await store.getDelivery(delivery.id);
      expect(entry).toMatchObject({ preparing: true, body_ref: bodyId });
      expect(await store.captureReferences(bodyId)).toBe(true);
      await staging.discardUnreferenced(bodyId);
      expect(await store.getCapture(`body:${bodyId}`)).toBeDefined();
      const root = await store.getCapture(`body:${bodyId}`);
      if (root.envelope) {
        expect(Array.isArray(root.envelope.session.turns)).toBe(false);
        expect(Array.isArray(root.envelope.session.attachments)).toBe(false);
      }
      const sealedWrites = storage.writes.filter((name) => name === `${bodyId}.bytes`).length;
      store.putCapture = originalRoot; store.putDelivery = originalDelivery;
      const restarted = new CaptureStaging(storage, store);
      await restarted.resumeDelivery(entry);
      const completed = await store.getDelivery(delivery.id);
      expect(completed).toMatchObject({ preparing: false, delivery_sequence: entry.delivery_sequence });
      const bytes = await (await restarted.file(bodyId)).text();
      expect(JSON.parse(bytes).session.attachments[0].inline_base64).toBe(globalThis.Buffer.from("asset evidence").toString("base64"));
      expect(JSON.parse(bytes).session.turns[0].text).toBe("retained transcript");
      expect((await restarted.metadata(bodyId)).sha256).toBe(digest(bytes));
      if (boundary !== "before-close") expect(storage.writes.filter((name) => name === `${bodyId}.bytes`)).toHaveLength(sealedWrites);
    },
  );

});
