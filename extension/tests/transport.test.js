import { expect, it } from "vitest";
import { askPage } from "../src/background/tabs.js";

function chromeFrames(frames) {
  const listeners = new Set();
  return { runtime: {}, tabs: { connect: () => ({
    onMessage: { addListener: (fn) => listeners.add(fn), removeListener: (fn) => listeners.delete(fn) },
    onDisconnect: { addListener: () => {}, removeListener: () => {} },
    disconnect: () => {},
    postMessage: () => queueMicrotask(() => { for (const frame of frames) for (const fn of listeners) fn(frame); }),
  }) } };
}

it.each([
  [{ type: "end" }],
  [{ type: "chunk", data: "eA==" }],
  [{ type: "facts", status: 200, has_body: false }, { type: "chunk", data: "eA==" }],
  [{ type: "facts", status: 200, has_body: true }, { type: "facts", status: 200, has_body: true }],
].map((frames) => [frames]))("refuses out-of-phase page replies", async (frames) => {
  await expect(askPage(chromeFrames(frames), 1, { type: "conversation" })).rejects.toMatchObject({ code: "invalid_page_reply" });
});
