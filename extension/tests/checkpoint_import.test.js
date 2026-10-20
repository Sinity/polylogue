import { File } from "node:buffer";
import { describe, expect, it } from "vitest";
import { importCheckpoint } from "../src/backfill/import.js";
import { NativeCaptureNormalizer } from "../src/capture/native.js";
import { stagingRuntime, receiverContractPreparation } from "./infra/capture-staging.js";

async function backup() {
  const source = stagingRuntime(); const { staging, store } = source;
  const owner = { tab_id: 42, document_id: "old-document", provider: "chatgpt" };
  const raw = '{ "id":"session", "mapping":{} }';
  const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/c/session" });
  await staging.append(ref, owner, 0, globalThis.Buffer.from(raw).toString("base64")); await staging.seal(ref, owner);
  const envelope = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) })
    .normalize({ provider: "chatgpt", rawRef: ref, nativeId: "session", extensionVersion: "test", instanceId: "old-instance" });
  const asset = await staging.begin(owner, { kind: "asset" });
  await staging.append(asset, owner, 0, globalThis.Buffer.from("exact attachment").toString("base64")); await staging.seal(asset, owner);
  await store.putCaptureRecord({ capture_id: envelope.capture_record_ref, key: "asset", record_kind: "session-attachment", order_time: 0, order_id: "", occurrence: 0,
    session_attachment: { provider_attachment_id: "file", message_provider_id: "message", staged_asset: asset } });
  const body = await staging.prepare(envelope);
  await store.putJob({ id: "job", provider: "chatgpt", account_scope: "h1:old-profile-scope", status: "paused", cutoff: "2026-01-01T00:00:00Z" });
  await store.putQueue({ id: "item", job_id: "job", provider: "chatgpt", native_id: "session", state: "captured_waiting_receiver", body_ref: body.ref,
    capture_record_ref: envelope.capture_record_ref, source_refs: [ref.id] });
  const id = "export:synthetic"; const artifact = staging.producerStageId({ checkpoint_snapshot_id: id }, id);
  const snapshot = await store.createRecoverySnapshot("job", id, artifact, { kind: "checkpoint-export", token: "token" });
  const exported = await staging.prepareCheckpoint(snapshot);
  return { file: exported.body, raw, body: await body.body.text() };
}
const asFile = data => new File([JSON.stringify(data)], "backup.json", { type: "application/json" });
async function empty(store) { const jobs = []; for await (const job of store.jobs()) jobs.push(job); expect(jobs).toEqual([]); }

describe("local backfill backup import", () => {
  it("restores exact raw and attachment evidence, records and body while keeping the old account scope paused", async () => {
    const exported = await backup(); const target = stagingRuntime();
    const result = await importCheckpoint(exported.file, target);
    const job = await target.store.getJob(result.job_id);
    expect(job).toMatchObject({ status: "paused", account_scope: "h1:old-profile-scope", cooldown_reason: "checkpoint_import_review_required", execution_owner: null });
    const { items } = await target.store.queuePage(job.id);
    expect(items).toHaveLength(1);
    const body = await target.staging.prepare({ capture_body_ref: items[0].body_ref });
    expect(await body.body.text()).toBe(exported.body);
    const capture = await target.store.getCapture(items[0].capture_record_ref);
    expect(await (await target.staging.file(capture.raw_ref)).text()).toBe(exported.raw);
    const records = []; for await (const record of target.store.captureRecords(capture.id)) records.push(record);
    expect(records).toHaveLength(1);
    expect(await (await target.staging.file(records[0].session_attachment.staged_asset.id)).text()).toBe("exact attachment");
    const before = target.storage.files.size;
    await expect(importCheckpoint(exported.file, target)).rejects.toThrow("checkpoint_import_job_exists");
    expect(target.storage.files.size).toBe(before);
    expect(await (await target.staging.file(capture.raw_ref)).text()).toBe(exported.raw);
  });

  it("preserves a typed capture sharing its staged body ID", async () => {
    const exported = await backup(); const parsed = JSON.parse(await exported.file.text());
    const entry = parsed.acquired[0]; const row = parsed.ledger.queue[0];
    const old = entry.capture.id; const shared = row.body_ref;
    entry.capture.id = shared; row.capture_record_ref = shared;
    for (const record of entry.records) record.capture_id = shared;
    for (const file of entry.files) if (file.metadata.record_ref === old) file.metadata.record_ref = shared;
    const target = stagingRuntime(); const result = await importCheckpoint(asFile(parsed), target);
    const { items } = await target.store.queuePage(result.job_id);
    expect(items[0].capture_record_ref).toBe(items[0].body_ref);
    expect(await target.store.getCapture(items[0].body_ref)).toBeDefined();
    expect(await (await target.staging.prepare({ capture_body_ref: items[0].body_ref })).body.text()).toBe(exported.body);
  });

  it.each(["hash", "order", "path", "missing", "attachment", "raw", "related", "metadata", "truncated"])("refuses %s corruption without publishing a job", async kind => {
    const exported = await backup(); const parsed = JSON.parse(await exported.file.text());
    const evidence = parsed.acquired[0].files[0];
    if (kind === "hash") evidence.parts[0].sha256 = "0".repeat(64);
    if (kind === "order") evidence.parts[0].sequence = 2;
    if (kind === "path") evidence.metadata.id = "../outside";
    if (kind === "missing") parsed.acquired[0].files = [];
    if (kind === "attachment") parsed.acquired[0].records[0].session_attachment.staged_asset.id = "abcdef";
    if (kind === "raw") parsed.acquired[0].capture.raw_ref = "abcdef";
    if (kind === "related") parsed.acquired[0].capture.related_refs = { responses: { id: "abcdef" } };
    if (kind === "metadata") evidence.metadata.record_ref = "absent-capture";
    const file = kind === "truncated" ? new File([JSON.stringify(parsed).slice(0, -5)], "broken.json") : asFile(parsed);
    const target = stagingRuntime();
    await expect(importCheckpoint(file, target)).rejects.toThrow();
    await empty(target.store); expect(target.storage.files.size).toBe(0);
  });

  it("cancels during evidence installation without publishing a runnable partial job", async () => {
    const exported = await backup(); const target = stagingRuntime(); const controller = new globalThis.AbortController();
    const append = target.staging.append.bind(target.staging);
    target.staging.append = async (...args) => { const result = await append(...args); controller.abort(new globalThis.DOMException("Cancelled", "AbortError")); return result; };
    await expect(importCheckpoint(exported.file, { ...target, signal: controller.signal })).rejects.toThrow("Cancelled");
    await empty(target.store); expect(target.storage.files.size).toBe(0);
  });
});
