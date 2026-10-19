import JSONParser from "../vendor/streamparser-json/jsonparser.js";
import TokenType from "../vendor/streamparser-json/utils/types/tokenType.js";

const HEADERS = ["id", "uuid", "conversation_id", "conversationId", "create_time", "update_time", "created_at", "updated_at", "createTime", "modifyTime", "is_temporary", "temporary", "conversation_template_id", "gizmo_id"];
function cancelNativeReader(reader, signal) {
  const cancel = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  return () => signal?.removeEventListener("abort", cancel);
}

/** Retains raw reply custody and delegates canonical preparation to receiver.
 * The vetted tokenizer selects acquisition header facts for identity/cache
 * control. A selected header scalar can still be arbitrarily large.
 */
export class NativeCaptureNormalizer {
  constructor({ staging, store, prepareNative }) {
    this.staging = staging; this.store = store; this.prepareNative = prepareNative;
  }
  async finishBundle(bundleId, owner, { pin = false, signal = null } = {}) {
    const bundle = await this.store.getCapture(bundleId);
    if (bundle?.kind !== "native-bundle" || JSON.stringify(bundle.owner) !== JSON.stringify(owner)) throw new Error("native_bundle_owner_mismatch");
    if (!bundle.replies.conversation || !bundle.replies.responses) throw new Error("native_bundle_incomplete");
    for (const ref of Object.values(bundle.replies)) this.staging.requireOwner(await this.staging.metadata(ref.id), ref, owner);
    const headers = await this.headers(bundle.replies.conversation, signal);
    signal?.throwIfAborted();
    if (String(headers.conversationId || headers.id || "") !== bundle.native_id) throw new Error("native_capture_identity_mismatch");
    const complete = await this.store.finishNativeBundle(bundle.id, owner);
    if (complete.state !== "ready") throw new Error(complete.error);
    const relatedRefs = { conversation: bundle.replies.conversation, ...(bundle.replies.response_nodes ? { response_nodes: bundle.replies.response_nodes } : {}) };
    const acquisition = { kind: "grok-endpoint-bundle", response_node_status: bundle.outcomes.response_nodes?.status || null,
      response_node_retry_after: bundle.outcomes.response_nodes?.retry_after || null };
    if (pin) {
      const raw = await this.staging.metadata(bundle.replies.responses.id);
      const pinned = await this.store.pinNativeCache({ owner, provider: bundle.provider, nativeId: bundle.native_id,
        rawRef: bundle.replies.responses, relatedRefs, acquisition, headers, observedAt: raw.created_at, acquisitionSequence: bundle.acquisition_sequence });
      for (const ref of pinned.retired) await this.staging.discardUnreferenced(ref?.id || ref);
      // The cache may already own a newer revision. This acquisition still
      // belongs to its bundle until normalization publishes its own custody.
    }
    return { headers, rawRef: bundle.replies.responses, relatedRefs, acquisition, acquisition_sequence: bundle.acquisition_sequence };
  }
  async headers(rawRef, signal = null) {
    const meta = await this.staging.metadata(rawRef.id);
    if (meta.token !== rawRef.token || meta.state !== "sealed") throw new Error("capture_staging_asset_unsealed");
    const headers = {}; const parser = new JSONParser({ paths: HEADERS.map((key) => `$.${key}`), keepStack: false, stringBufferSize: 64 * 1024 });
    parser.onValue = ({ key, value }) => { headers[key] = value; };
    let mappingObject = false; let mappingValue = false;
    // Observe the root member's value token through the existing JSON parser;
    // never select/materialize its potentially large mapping as a header.
    parser.onToken = ({ token }) => {
      if (mappingValue) { mappingObject = token === TokenType.LEFT_BRACE; mappingValue = false; }
      else if (token === TokenType.COLON && parser.tokenParser.stack.length === 1 && parser.tokenParser.key === "mapping") mappingValue = true;
    };
    const reader = (await this.staging.file(rawRef.id)).stream().getReader();
    const stopReader = cancelNativeReader(reader, signal);
    try {
      for (;;) {
        signal?.throwIfAborted(); const chunk = await reader.read(); if (chunk.done) break;
        for (let offset = 0; offset < chunk.value.length; offset += 48 * 1024) parser.write(chunk.value.subarray(offset, offset + 48 * 1024));
      }
      signal?.throwIfAborted(); if (!parser.isEnded) parser.end();
    } finally { await reader.cancel().catch(() => undefined); stopReader(); reader.releaseLock(); }
    if (meta.owner.provider === "chatgpt" && !mappingObject) {
      const error = new Error("native_capture_mapping_invalid"); error.code = "native_capture_mapping_invalid"; throw error;
    }
    return headers;
  }
  async normalize({ provider, rawRef, nativeId, extensionVersion, instanceId, attribution = {}, signal, relatedRefs = {}, acquisition = null, requireTemporary = false, queueContext = null, summaryOnly = false, onProgress = null }) {
    if (!["chatgpt", "claude-ai", "grok"].includes(provider)) throw new Error(`native_provider_unsupported:${provider}`);
    const raw = await this.staging.metadata(rawRef.id);
    if (raw.token !== rawRef.token || raw.state !== "sealed") throw new Error("capture_staging_asset_unsealed");
    const headers = await this.headers(relatedRefs.conversation || rawRef, signal);
    const observedId = provider === "chatgpt" ? headers.conversation_id || headers.id || nativeId
      : provider === "claude-ai" ? headers.uuid || headers.id || nativeId : headers.conversationId || nativeId;
    if (observedId !== nativeId) throw new Error("native_capture_identity_mismatch");
    if (requireTemporary && headers.is_temporary !== true) throw new Error("native_temporary_identity_mismatch");
    if (!raw.owner.document_id || !raw.source_url) throw new Error("native_original_document_unavailable");
    if (!instanceId || typeof instanceId !== "string") throw new Error("native_preparation_instance_unavailable");
    const captureId = `native-preparation:${rawRef.id}`;
    let capture = await this.store.getCapture(captureId);
    if (capture && (capture.provider !== provider || capture.native_id !== nativeId || JSON.stringify(capture.owner) !== JSON.stringify(raw.owner))) throw new Error("native_preparation_owner_mismatch");
    if (!capture) {
      // Preparation identity is independently retained. Missing historical
      // observation identity stays absent even after a worker/profile reload.
      capture = { id: captureId, state: "normalizing", raw_ref: rawRef.id,
        source_refs: [rawRef.id, ...Object.values(relatedRefs).map((ref) => ref.id)], owner: raw.owner,
        provider, native_id: nativeId, headers, related_refs: relatedRefs, acquisition,
        source_url: raw.source_url, extension_version: extensionVersion, attribution,
        observed_at: raw.observed_at, extension_instance_id: raw.extension_instance_id || null,
        acquisition_sequence: raw.extension_instance_id ? raw.acquisition_sequence : null,
        invocation_id: raw.invocation_ref?.id || null, preparation_instance_id: instanceId,
        preparation_token: crypto.randomUUID(), acquisition_id: crypto.randomUUID(),
        turn_count: 0, record_count: 0, attachment_count: 0 };
      await this.store.putCapture(capture);
    }
    signal?.throwIfAborted();
    const prepared = await this.prepareNative(capture, { rawRef, relatedRefs, queueContext, signal, summaryOnly, onProgress });
    if (summaryOnly) return { summary: prepared.summary, rawRevision: prepared.rawRevision };
    capture = { ...capture, state: "ready", receiver_native: prepared.reference,
      raw_revision_sha256: prepared.rawRevision, turn_count: prepared.summary.turn_count,
      record_count: prepared.summary.turn_count, attachment_count: prepared.summary.attachment_count,
      session_kind: prepared.summary.session_kind,
      headers: { ...capture.headers, needs_follow_up: prepared.summary.needs_follow_up } };
    const envelope = this.envelope(capture);
    if (queueContext) await this.store.publishNativeCapture(capture, envelope, queueContext);
    else await this.store.publishForegroundNativeCapture(capture, raw.capture_bundle?.id || null);
    return envelope;
  }
  envelope(capture) {
    return { schema_version: 1, polylogue_capture_kind: "browser_llm_session", source: "browser-extension",
      provider_meta: { capture_fidelity: "native_full", ...capture.attribution },
      provenance: { captured_at: capture.observed_at, extension_instance_id: capture.extension_instance_id,
        acquisition_sequence: capture.acquisition_sequence, source_url: capture.source_url },
      session: { provider: capture.provider, provider_session_id: capture.native_id,
        title: capture.native_id,
        session_kind: capture.session_kind,
        provider_meta: { capture_fidelity: "native_full", ...capture.attribution }, turns: [] },
      receiver_native: capture.receiver_native, capture_source_refs: capture.source_refs,
      capture_record_ref: capture.id,
      capture_summary: { title: capture.native_id,
        provider: capture.provider, providerSessionId: capture.native_id, captureMode: "native_full",
        needsFollowUp: capture.headers.needs_follow_up, turnCount: capture.turn_count, attachmentCount: capture.attachment_count } };
  }
}
