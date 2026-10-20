import { describe, expect, it } from "vitest";
import { NativeCaptureNormalizer } from "../src/capture/native.js";
import { nativeEnvelope, prepareNativeCapture } from "../src/capture/native_transport.js";
import { stagingRuntime } from "./infra/capture-staging.js";

const owner = { tab_id: 42, document_id: "synthetic-document", provider: "chatgpt" };
async function fixture() {
  const runtime = stagingRuntime();
  const raw = '{ "id":"session", "mapping":{"node":{"message":{"id":"message","content":{"parts":["hello"]}}}} }';
  const ref = await runtime.staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/c/session" });
  await runtime.staging.append(ref, owner, 0, Buffer.from(raw).toString("base64"));
  await runtime.staging.seal(ref, owner);
  return { ...runtime, raw, ref };
}
const summary = { title: "Synthetic conversation", turn_count: 1, attachment_count: 1, session_kind: "main", needs_follow_up: false };

describe("native capture through the shared host parser", () => {
  it("retains raw bytes and by-value attachments in the ordinary durable capture envelope", async () => {
    const { staging, store, raw, ref } = await fixture();
    let inspected;
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: (capture, options) => prepareNativeCapture({
      capture, ...options, staging, store,
      inspect: async (body) => {
        inspected = await body.text();
        return { ok: true, summary, attachments: [{ provider_attachment_id: "file", message_provider_id: "message" }] };
      },
      acquire: async (descriptor) => {
        const asset = await staging.begin(owner, { kind: "asset" });
        await staging.append(asset, owner, 0, Buffer.from("attachment bytes").toString("base64"));
        await staging.seal(asset, owner);
        return { attachments: [{ ...descriptor, staged_asset: asset }], outcome: { failed: [] } };
      },
    }) });
    const envelope = await normalizer.normalize({ provider: owner.provider, rawRef: ref, nativeId: "session", extensionVersion: "test", instanceId: "instance" });
    expect(inspected).toContain(`"raw_provider_payload":${raw}`);
    expect(envelope.receiver_native).toBeUndefined();
    const prepared = await staging.prepare(envelope);
    const final = await prepared.body.text();
    expect(final).toContain(`"raw_provider_payload":${raw}`);
    const parsed = JSON.parse(final);
    expect(parsed.session.attachments).toEqual([{ provider_attachment_id: "file", message_provider_id: "message", content_base64: Buffer.from("attachment bytes").toString("base64") }]);
    expect(parsed.session.title).toBe(summary.title);
    const roots = [];
    for await (const capture of store.captures()) if (capture.kind === "receiver-body") roots.push(capture.id);
    expect(roots).toEqual([`body:${prepared.ref}`]);
    expect(await (await staging.file(ref.id)).text()).toBe(raw);
  });

  it("substitutes all Grok raw members even when the caller has no acquisition metadata", async () => {
    const { staging, store } = stagingRuntime();
    const owner = { tab_id: 42, document_id: "document", provider: "grok" };
    const refs = {};
    for (const [name, value] of Object.entries({ conversation: { conversationId: "session" }, responses: { responses: [{ message: "exact" }] } })) {
      const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://grok.com/synthetic" });
      await staging.append(ref, owner, 0, Buffer.from(JSON.stringify(value)).toString("base64")); await staging.seal(ref, owner); refs[name] = ref;
    }
    const capture = { id: "capture", provider: "grok", native_id: "session", raw_ref: refs.responses.id,
      related_refs: { conversation: refs.conversation }, source_refs: Object.values(refs).map(ref => ref.id), owner };
    await store.putCapture(capture);
    const prepared = await staging.prepare(nativeEnvelope(capture));
    expect(JSON.parse(await prepared.body.text()).raw_provider_payload).toEqual({ conversation: { conversationId: "session" },
      responses: { responses: [{ message: "exact" }] }, acquisition: { kind: "grok-endpoint-bundle" } });
  });

  it("keeps the raw acquisition after an inspection failure and removes its temporary body", async () => {
    const { staging, store, raw, ref } = await fixture();
    const normalizer = new NativeCaptureNormalizer({ staging, store, prepareNative: (capture, options) => prepareNativeCapture({
      capture, ...options, staging, store, inspect: async () => { throw new Error("host unavailable"); }, acquire: async () => { throw new Error("must not acquire"); },
    }) });
    await expect(normalizer.normalize({ provider: owner.provider, rawRef: ref, nativeId: "session", extensionVersion: "test", instanceId: "instance" })).rejects.toThrow("host unavailable");
    expect(await (await staging.file(ref.id)).text()).toBe(raw);
    for await (const capture of store.captures()) expect(capture.kind).not.toBe("receiver-body");
  });
});
