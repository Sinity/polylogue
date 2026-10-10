// @vitest-environment node
import { Buffer } from "node:buffer";
import { setImmediate } from "node:timers";
import { describe, expect, it, vi } from "vitest";
import { nativeFetch } from "../src/background/native_fetch.js";

function fixture(handler = () => {}) {
  const messages = new Set(); const disconnects = new Set(); const sent = [];
  const port = {
    onMessage: { addListener: fn => messages.add(fn), removeListener: fn => messages.delete(fn) },
    onDisconnect: { addListener: fn => disconnects.add(fn), removeListener: fn => disconnects.delete(fn) },
    postMessage(frame) { sent.push(frame); handler(frame, emit); },
    disconnect: vi.fn(),
  };
  const emit = frame => { for (const fn of messages) fn(frame); };
  return { runtime: { connectNative: vi.fn(() => port) }, port, sent, emit,
    eof() { for (const fn of disconnects) fn(); } };
}
const headers = { type: "response", status: 200, headers: { "X-Request-ID": "neutral-request" },
  receiver_id: "neutral-receiver", api_schema: "polylogue-browser-capture/v1" };
const turn = () => new Promise(resolve => setImmediate(resolve));

describe("one owned native receiver operation", () => {
  it("owns Chrome's missing-host diagnostic without exposing its text or falling back to HTTP", async () => {
    const f = fixture();
    const diagnostic = vi.fn(() => ({ message: "private host installation path" }));
    Object.defineProperty(f.runtime, "lastError", { get: diagnostic });
    const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    f.eof();
    await expect(pending).rejects.toMatchObject({ code: "native_messaging_unavailable", message: "native_messaging_unavailable" });
    expect(diagnostic).toHaveBeenCalledOnce();
    expect(f.port.disconnect).toHaveBeenCalledOnce();
  });
  it.each(["string", "blob", "bytes", "buffer"])("preserves exact %s body above native frame size, paced by ACK", async kind => {
    const bytes = new TextEncoder().encode("neutral α content ".repeat(90000));
    const body = kind === "string" ? new globalThis.TextDecoder().decode(bytes) : kind === "blob" ? new globalThis.Blob([bytes]) : kind === "buffer" ? bytes.buffer : bytes;
    const f = fixture(frame => { if (frame.type === "body_end") f.emit(headers); });
    const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/capture-jobs?after=3", {
      method: "PUT", receiverId: "neutral-receiver", headers: { "X-Polylogue-Native": "neutral" }, body,
    });
    const received = [];
    while (!f.sent.some(frame => frame.type === "body_end")) {
      await turn();
      const frames = f.sent.filter(frame => frame.type === "body");
      if (frames.length === received.length) continue;
      expect(frames.length).toBe(received.length + 1);
      const frame = frames.at(-1); const chunk = Buffer.from(frame.data, "base64");
      expect(chunk.length).toBeLessThanOrEqual(64 * 1024);
      expect(frame.sequence).toBe(received.length);
      received.push(chunk);
      await turn(); expect(f.sent.filter(frame => frame.type === "body")).toHaveLength(received.length);
      f.emit({ type: "body_ack", sequence: frame.sequence });
    }
    expect(Buffer.concat(received)).toEqual(Buffer.from(bytes));
    expect(f.runtime.connectNative).toHaveBeenCalledOnce();
    expect(f.sent[0]).toEqual({ type: "request", version: 1, endpoint: "http://127.0.0.1:8765",
      receiver_id: "neutral-receiver", method: "PUT", path: "/v1/capture-jobs?after=3", headers: { "x-polylogue-native": "neutral" } });
    const response = await pending;
    expect(response).toBeInstanceOf(globalThis.Response);
    expect(response.headers.get("X-Request-ID")).toBe("neutral-request");
    expect(f.sent.some(frame => frame.type === "response_next")).toBe(false);
    await response.body.cancel(); expect(f.port.disconnect).toHaveBeenCalledOnce();
  });

  it("pulls exactly one response quantum at a time and preserves HTTP failures", async () => {
    let quantum = 0;
    const f = fixture((frame, emit) => {
      if (frame.type === "body_end") emit({ ...headers, status: 409 });
      if (frame.type === "response_next") {
        emit(quantum < 2 ? { type: "response_body", sequence: quantum, data: Buffer.from([quantum++]).toString("base64") } : { type: "response_end" });
      }
    });
    const response = await nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    expect(response.ok).toBe(false); expect(response.status).toBe(409);
    expect(quantum).toBe(0);
    const reader = response.body.getReader();
    expect((await reader.read()).value).toEqual(new globalThis.Uint8Array([0]));
    await turn(); expect(quantum).toBe(1);
    expect((await reader.read()).value).toEqual(new globalThis.Uint8Array([1]));
    expect((await reader.read()).done).toBe(true);
    expect(f.port.disconnect).toHaveBeenCalledOnce();
  });

  it.each(["before", "after"])("propagates cancellation %s response headers and closes custody", async phase => {
    const controller = new globalThis.AbortController();
    const f = fixture((frame, emit) => { if (phase === "after" && frame.type === "body_end") emit(headers); });
    const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status", { signal: controller.signal });
    const error = new globalThis.DOMException("neutral cancellation", "AbortError");
    if (phase === "before") {
      const rejected = expect(pending).rejects.toBe(error); controller.abort(error); await rejected;
    } else {
      const response = await pending; const reading = response.body.getReader().read();
      const rejected = expect(reading).rejects.toBe(error); controller.abort(error); await rejected;
    }
    expect(f.sent.at(-1)).toEqual({ type: "cancel" });
    expect(f.port.disconnect).toHaveBeenCalledOnce();
  });

  it.each(["before", "after"])("propagates named host refusal %s headers without HTTP fallback", async phase => {
    const f = fixture((frame, emit) => { if (phase === "after" && frame.type === "body_end") emit(headers); });
    const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    if (phase === "before") {
      const failed = expect(pending).rejects.toMatchObject({ code: "receiver_identity_mismatch" });
      f.emit({ type: "error", error: "receiver_identity_mismatch" }); await failed;
    } else {
      const response = await pending; f.emit({ type: "error", error: "receiver_identity_mismatch" });
      await expect(response.text()).rejects.toMatchObject({ code: "receiver_identity_mismatch" });
    }
    expect(f.port.disconnect).toHaveBeenCalledOnce();
    expect(f.runtime.connectNative).toHaveBeenCalledOnce();
  });

  it("rejects port EOF before completion", async () => {
    const f = fixture(); const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    const failed = expect(pending).rejects.toMatchObject({ code: "native_transport_disconnected" }); f.eof(); await failed;
  });

  it("settles a bodyless response's terminal frame", async () => {
    const f = fixture((frame, emit) => {
      if (frame.type === "body_end") emit({ ...headers, status: 204 });
      if (frame.type === "response_next") emit({ type: "response_end" });
    });
    const response = await nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    expect(response.status).toBe(204); expect(response.body).toBeNull();
    expect(f.port.disconnect).toHaveBeenCalledOnce();
  });

  it("refuses a response sequence gap and closes its stream", async () => {
    const f = fixture((frame, emit) => {
      if (frame.type === "body_end") emit(headers);
      if (frame.type === "response_next") emit({ type: "response_body", sequence: 1, data: "AA==" });
    });
    const response = await nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status");
    await expect(response.text()).rejects.toMatchObject({ code: "native_response_frame_invalid" });
    expect(f.port.disconnect).toHaveBeenCalledOnce();
  });

  it("propagates cancellation while an upload awaits its ACK", async () => {
    const controller = new globalThis.AbortController(); const f = fixture();
    const pending = nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/capture-jobs", {
      method: "PUT", body: new globalThis.Uint8Array(131072), signal: controller.signal,
    });
    await turn(); expect(f.sent.filter(frame => frame.type === "body")).toHaveLength(1);
    const error = new Error("neutral cancellation"); const failed = expect(pending).rejects.toBe(error);
    controller.abort(error); await failed;
    expect(f.port.disconnect).toHaveBeenCalledOnce();
    expect(f.sent.some(frame => frame.type === "body_end")).toBe(false);
  });

  it.each(["authorization", "Host", "Content-Length", "Connection"])("refuses caller %s before opening a port", async header => {
    const f = fixture();
    await expect(nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/status", { headers: { [header]: "neutral" } })).rejects.toMatchObject({ code: "native_request_header_forbidden" });
    expect(f.runtime.connectNative).not.toHaveBeenCalled();
  });
  it("forbids secret-bearing pairing redemption", async () => {
    const f = fixture();
    await expect(nativeFetch(f.runtime, "http://127.0.0.1:8765/v1/pairing/redeem", { method: "POST" })).rejects.toMatchObject({ code: "native_route_forbidden" });
    expect(f.runtime.connectNative).not.toHaveBeenCalled();
  });
});
