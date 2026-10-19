import { describe, expect, it, vi } from "vitest";
import { createBackgroundAdapters } from "../src/background/adapters.js";

describe("background adapter network seam", () => {
  it("keeps an old worker bound to its explicit test network", async () => {
    const oldFetch = vi.fn(async () => "old");
    const newFetch = vi.fn(async () => "new");
    globalThis.fetch = oldFetch;
    const oldAdapters = createBackgroundAdapters({}, oldFetch);
    globalThis.fetch = newFetch;

    await expect(oldAdapters.network("/old")).resolves.toBe("old");
    expect(oldFetch).toHaveBeenCalledWith("/old");
    expect(newFetch).not.toHaveBeenCalled();
  });
  it("uses native messaging by default without calling HTTP fetch", async () => {
    const fetch = vi.fn();
    globalThis.fetch = fetch;
    const adapters = createBackgroundAdapters({ runtime: {} });
    await expect(adapters.network("http://127.0.0.1:8765/v1/status")).rejects.toMatchObject({ code: "native_messaging_unavailable" });
    expect(fetch).not.toHaveBeenCalled();
  });
});
