import { nativeFetch } from "./src/background/native_fetch.js";
import { createSha256 } from "./src/vendor/sha256.js";

// This module runs only in the isolated proof page, with no capture worker.
export async function proveNativeTransport({ receiverUrl, receiverId, attachmentUrl, attachmentSha256 }) {
  if (new URL(attachmentUrl).origin !== new URL(receiverUrl).origin || !receiverId || !/^[a-f0-9]{64}$/.test(attachmentSha256)) throw new Error("proof_native_inputs_invalid");
  const operation = (url, init = {}) => nativeFetch(chrome.runtime, url, { ...init, receiverId });
  const bytes = Uint8Array.from({ length: 1024 * 1024 + 17 }, (_, index) => (index * 31 + 7) % 256);
  const digest = createSha256().update(bytes).hex();
  const upload = await operation(`${receiverUrl}/v1/browser-action-attachments`, { method: "PUT", body: new Blob([bytes]) });
  const uploaded = await upload.json();
  if (upload.status !== 201 || uploaded.attachment_ref !== digest || uploaded.size_bytes !== bytes.length) throw new Error("proof_native_upload_mismatch");

  let refusal = null;
  try { await nativeFetch(chrome.runtime, `${receiverUrl}/v1/status`, { receiverId: `${receiverId}-wrong` }); }
  catch (error) { refusal = error.code; }
  if (refusal !== "receiver_identity_mismatch") throw new Error("proof_native_refusal_missing");

  const cancellation = new AbortController();
  const interrupted = await operation(attachmentUrl, { signal: cancellation.signal });
  const interruptedReader = interrupted.body.getReader();
  if ((await interruptedReader.read()).done) throw new Error("proof_native_cancel_empty");
  cancellation.abort();
  let cancelled = false;
  try { await interruptedReader.read(); } catch (error) { cancelled = error.name === "AbortError"; }
  if (!cancelled) throw new Error("proof_native_cancel_missing");

  const response = await operation(attachmentUrl);
  if (response.status !== 200) throw new Error("proof_native_download_refused");
  const hash = createSha256();
  let size = 0; let chunks = 0; let maxChunk = 0;
  for await (const chunk of response.body) {
    hash.update(chunk); size += chunk.length; chunks += 1; maxChunk = Math.max(maxChunk, chunk.length);
  }
  const receivedDigest = hash.hex();
  if (receivedDigest !== attachmentSha256 || size <= 1024 * 1024 || chunks < 2 || maxChunk > 64 * 1024) throw new Error("proof_native_download_mismatch");
  return { upload_bytes: bytes.length, upload_sha256: digest, response_bytes: size, response_sha256: receivedDigest,
    response_chunks: chunks, maximum_response_chunk_bytes: maxChunk, refusal, cancellation: true };
}
