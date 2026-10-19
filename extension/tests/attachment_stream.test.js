import { createHash } from "node:crypto";
import { describe, expect, it, afterEach } from "vitest";
import { AttachmentSha256 } from "../src/actions/sha256.js";
import { transferBrowserActionAttachmentInPage } from "../src/actions/chatgpt.js";

const transfer = (...args) => transferBrowserActionAttachmentInPage("action", "lease-owner", ...args);
afterEach(() => { delete globalThis.__polylogueBrowserActionAttachments; });

describe("streamed action attachment integrity", () => {
  it.each([0, 1, 55, 56, 63, 64, 65, 127, 128, 65537, 17 * 1024 * 1024])("hashes %i bytes through bounded updates", (length) => {
    const hash = new AttachmentSha256();
    const oracle = createHash("sha256");
    const chunk = new Uint8Array(16381).map((_, index) => index % 251);
    let remaining = length;
    while (remaining) {
      const bytes = chunk.subarray(0, Math.min(remaining, chunk.length));
      hash.update(bytes);
      oracle.update(bytes);
      remaining -= bytes.length;
    }
    expect(hash.digestHex()).toBe(oracle.digest("hex"));
    expect(() => hash.update(chunk)).toThrow("hash_finished");
  });

  it("assembles the exact provider File while refusing foreign ownership, skipped chunks and premature finish", async () => {
    const item = { attachment_id: "a1", name: "neutral.txt", mime_type: "text/plain", sha256: "11".repeat(32), size_bytes: 3 };
    transfer("begin");
    transfer("start", item);
    expect(() => transferBrowserActionAttachmentInPage("foreign", "lease-owner", "discard")).toThrow("owner_mismatch");
    expect(() => transfer("append", item, 1, btoa("abc"))).toThrow("chunk_mismatch");
    expect(() => transfer("finish", item)).toThrow("size_mismatch");
    transfer("append", item, 0, btoa("abc"));
    transfer("finish", item);
    const file = globalThis.__polylogueBrowserActionAttachments.entries.get("a1").file;
    expect(file.name).toBe("neutral.txt");
    expect(file.type).toBe("text/plain");
    const content = await new Promise((resolve, reject) => {
      const reader = new globalThis.FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsText(file);
    });
    expect(content).toBe("abc");
    transfer("discard");
    expect(globalThis.__polylogueBrowserActionAttachments).toBeUndefined();
  });

  it("refuses IPC chunks above the declared transfer unit without retaining bytes", () => {
    const item = { attachment_id: "a1", size_bytes: 100000, sha256: "11".repeat(32) };
    transfer("begin"); transfer("start", item);
    expect(() => transfer("append", item, 0, btoa("x".repeat(65537)))).toThrow("chunk_mismatch");
    expect(globalThis.__polylogueBrowserActionAttachments.entries.get("a1").size).toBe(0);
  });
});
