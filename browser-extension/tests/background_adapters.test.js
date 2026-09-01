import { describe, expect, it, vi } from "vitest";
import { createBackgroundAdapters } from "../src/background/adapters.js";

describe("background adapter network seam", () => {
  it("keeps an old worker bound to the fetch implementation it created with", async () => {
    const oldFetch = vi.fn(async () => "old");
    const newFetch = vi.fn(async () => "new");
    globalThis.fetch = oldFetch;
    const oldAdapters = createBackgroundAdapters({});
    globalThis.fetch = newFetch;

    await expect(oldAdapters.network("/old")).resolves.toBe("old");
    expect(oldFetch).toHaveBeenCalledWith("/old");
    expect(newFetch).not.toHaveBeenCalled();
  });
});
