const NATIVE_HOST = "com.polylogue.browser_capture";
const CHUNK_BYTES = 64 * 1024;
const FORBIDDEN_HEADERS = new Set([
  "authorization", "host", "connection", "content-length", "transfer-encoding",
  "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
]);

function refusal(code) {
  const error = new Error(code);
  error.code = code;
  return error;
}

function deferred() {
  let resolve; let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function encode(bytes) {
  let text = "";
  for (const byte of bytes) text += String.fromCharCode(byte);
  return btoa(text);
}

function decode(data) {
  if (typeof data !== "string") throw refusal("native_response_frame_invalid");
  const text = atob(data);
  if (text.length > CHUNK_BYTES || btoa(text) !== data) throw refusal("native_response_frame_invalid");
  return Uint8Array.from(text, char => char.charCodeAt(0));
}

async function* bodyChunks(body) {
  if (body === null || body === undefined) return;
  if (typeof body === "string") body = new Blob([body]);
  if (body instanceof ArrayBuffer) body = new Uint8Array(body);
  if (ArrayBuffer.isView(body)) {
    const bytes = new Uint8Array(body.buffer, body.byteOffset, body.byteLength);
    for (let offset = 0; offset < bytes.length; offset += CHUNK_BYTES) yield bytes.subarray(offset, offset + CHUNK_BYTES);
    return;
  }
  if (!(body instanceof Blob)) throw refusal("native_request_body_invalid");
  // Slice the already-owned Blob rather than materializing its whole body.
  for (let offset = 0; offset < body.size; offset += CHUNK_BYTES) {
    yield new Uint8Array(await body.slice(offset, offset + CHUNK_BYTES).arrayBuffer());
  }
}

/** One native port owns one receiver operation and its streamed response. */
export async function nativeFetch(runtime, input, init = {}) {
  const signal = init.signal;
  signal?.throwIfAborted();
  const url = new URL(input);
  const method = String(init.method || "GET").toUpperCase();
  if (!["GET", "POST", "PUT"].includes(method) || url.username || url.password || url.hash) throw refusal("native_request_invalid");
  if (url.pathname === "/v1/pairing/redeem") throw refusal("native_route_forbidden");
  if (method === "GET" && init.body !== undefined && init.body !== null) throw refusal("native_request_body_invalid");
  const headers = Object.fromEntries(new Headers(init.headers).entries());
  if (Object.keys(headers).some(key => FORBIDDEN_HEADERS.has(key))) throw refusal("native_request_header_forbidden");
  if (!runtime?.connectNative) throw refusal("native_messaging_unavailable");
  const port = runtime.connectNative(NATIVE_HOST);
  const responseReady = deferred();
  let ack = null; let next = null; let controller = null;
  let receivedHeaders = false; let uploadComplete = false; let closed = false;
  let bodySequence = 0; let responseSequence = 0; let noBody = false;

  function disconnect() {
    if (closed) return;
    closed = true;
    signal?.removeEventListener("abort", abort);
    port.onMessage.removeListener(onMessage);
    port.onDisconnect.removeListener(onDisconnect);
    port.disconnect();
  }
  function fail(error, cancel = true) {
    if (closed) return;
    if (cancel) { try { port.postMessage({ type: "cancel" }); } catch { /* EOF releases the same custody. */ } }
    if (!receivedHeaders) responseReady.reject(error);
    controller?.error(error);
    ack?.reject(error); next?.reject(error);
    disconnect();
  }
  function abort() { fail(signal.reason); }
  function onDisconnect() {
    // Reading lastError owns Chrome's missing-host diagnostic without exposing
    // its host-specific text to callers or leaving an unchecked error behind.
    const missingHost = runtime.lastError;
    fail(refusal(missingHost ? "native_messaging_unavailable" : "native_transport_disconnected"), false);
  }
  function onMessage(frame) {
    if (closed) return;
    try {
      if (frame?.type === "error" && typeof frame.error === "string") { fail(refusal(frame.error)); return; }
      if (frame?.type === "body_ack" && ack && frame.sequence === bodySequence) {
        const pending = ack; ack = null; bodySequence += 1; pending.resolve(); return;
      }
      if (frame?.type === "response" && uploadComplete && !receivedHeaders
          && Number.isInteger(frame.status) && frame.status >= 200 && frame.status <= 599
          && typeof frame.receiver_id === "string" && frame.receiver_id
          && frame.api_schema === "polylogue-browser-capture/v1"
          && (!init.receiverId || frame.receiver_id === init.receiverId)
          && frame.headers && typeof frame.headers === "object" && !Array.isArray(frame.headers)
          && Object.values(frame.headers).every(value => typeof value === "string")) {
        const stream = new ReadableStream({
          start(value) { controller = value; },
          pull() {
            if (closed) return;
            next = deferred();
            port.postMessage({ type: "response_next" });
            return next?.promise;
          },
          cancel(reason) { fail(reason || refusal("native_response_cancelled")); },
        }, { highWaterMark: 0 });
        noBody = [204, 205, 304].includes(frame.status);
        const response = new Response(noBody ? null : stream, { status: frame.status, headers: frame.headers });
        receivedHeaders = true;
        responseReady.resolve(response);
        if (noBody) {
          // Even a bodyless response has a terminal native frame to consume.
          next = deferred(); next.promise.catch(() => undefined);
          port.postMessage({ type: "response_next" });
        }
        return;
      }
      if (receivedHeaders && next && frame?.type === "response_body" && frame.sequence === responseSequence) {
        const bytes = decode(frame.data);
        if (noBody || !bytes.length) throw refusal("native_response_frame_invalid");
        responseSequence += 1;
        const pending = next; next = null; controller.enqueue(bytes); pending.resolve(); return;
      }
      if (receivedHeaders && next && frame?.type === "response_end") {
        const pending = next; next = null; controller.close(); pending.resolve(); disconnect(); return;
      }
      throw refusal("native_response_frame_invalid");
    } catch (error) { fail(error); }
  }
  port.onMessage.addListener(onMessage);
  port.onDisconnect.addListener(onDisconnect);
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) abort();
  void (async () => {
    try {
      if (closed) return;
      port.postMessage({ type: "request", version: 1, endpoint: url.origin,
        receiver_id: init.receiverId || null, method, path: `${url.pathname}${url.search}`, headers });
      for await (const bytes of bodyChunks(init.body)) {
        signal?.throwIfAborted();
        if (closed) return;
        ack = deferred();
        const acknowledged = ack.promise;
        port.postMessage({ type: "body", sequence: bodySequence, data: encode(bytes) });
        await acknowledged;
      }
      if (closed) return;
      uploadComplete = true;
      port.postMessage({ type: "body_end" });
    } catch (error) { fail(error); }
  })();
  return responseReady.promise;
}
