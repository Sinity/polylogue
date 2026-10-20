import { createSha256 } from "../vendor/sha256.js";

/** Raw provider replies stay in OPFS; the host supplies their interpretation. */
export function nativeEnvelope(capture, attachments = false) {
  const raw = (id) => ({ staged_raw_json: id });
  const payload = capture.provider === "grok"
    ? { conversation: raw(capture.related_refs.conversation.id), responses: raw(capture.raw_ref),
      ...(capture.related_refs.response_nodes ? { response_nodes: raw(capture.related_refs.response_nodes.id) } : {}),
      acquisition: capture.acquisition }
    : raw(capture.raw_ref);
  return { schema_version: 1, polylogue_capture_kind: "browser_llm_session", source: "browser-extension",
    provider_meta: { capture_fidelity: "native_full", ...capture.attribution,
      ...(attachments ? { asset_acquisition: { failed: { staged_asset_failures: capture.id } } } : {}) },
    provenance: { captured_at: capture.observed_at, extension_instance_id: capture.extension_instance_id,
      acquisition_sequence: capture.acquisition_sequence, source_url: capture.source_url },
    raw_provider_payload: payload,
    session: { provider: capture.provider, provider_session_id: capture.native_id,
      title: capture.title || capture.native_id, session_kind: capture.session_kind,
      turns: [], ...(attachments ? { attachments: { staged_session_attachments: capture.id } } : {}) },
    capture_source_refs: capture.source_refs, capture_record_ref: capture.id };
}

export async function prepareNativeCapture({ capture, staging, store, inspect, acquire, signal, summaryOnly = false, onProgress }) {
  const progress = (phase, state) => { try { onProgress?.(phase, state); } catch { /* Diagnostics cannot change capture. */ } };
  const revisions = [];
  for (const id of capture.source_refs) {
    const meta = await staging.metadata(id);
    staging.requireOwner(meta, { id, token: meta.token }, capture.owner);
    if (meta.state !== "sealed") throw new Error("capture_staging_asset_unsealed");
    revisions.push([id, meta.sha256]);
  }
  const rawRevision = createSha256().update(JSON.stringify(revisions)).hex();
  progress("native_prepare", "BEGIN");
  const envelope = nativeEnvelope(capture);
  let body;
  let plan;
  try {
    body = await staging.prepare(envelope, null, null, signal);
    plan = await inspect(body.body, signal);
    if (!plan?.ok || !plan.summary || !Array.isArray(plan.attachments)) throw new Error("native_inspection_invalid");
  } finally {
    // This body is an inspection request, never a pending delivery. Raw replies
    // have their independent capture roots throughout this temporary copy.
    const id = body?.ref || envelope.capture_body_ref;
    if (id) { await store.discardCapture(`body:${id}`); await staging.discard(id); }
  }
  progress("native_prepare", "END");
  if (summaryOnly) return { rawRevision, summary: plan.summary };
  progress("native_assets", "BEGIN");
  for (const [ordinal, descriptor] of plan.attachments.entries()) {
    signal?.throwIfAborted();
    const key = `native-attachment:${ordinal}`;
    const retained = await store.getCaptureRecord(capture.id, key);
    if (retained?.session_attachment) continue;
    const result = descriptor.provider_meta?.native_inline_sha256
      ? { attachments: [], outcome: { failed: [], retained_native_bytes: true } }
      : await acquire(descriptor, ordinal, signal);
    const attachment = result.attachments?.[0] || null;
    if (attachment?.staged_asset) {
      const meta = await staging.metadata(attachment.staged_asset.id);
      staging.requireOwner(meta, attachment.staged_asset, capture.owner);
      if (meta.state !== "sealed") throw new Error("capture_staging_asset_unsealed");
    }
    await store.putCaptureRecord({ capture_id: capture.id, key, record_kind: "session-attachment",
      occurrence: ordinal, order_time: 0, order_id: "", turn: null,
      session_attachment: attachment, asset_failures: result.outcome?.failed || [] });
  }
  progress("native_assets", "END");
  return { rawRevision, summary: plan.summary };
}
