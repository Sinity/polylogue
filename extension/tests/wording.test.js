import { describe, expect, it } from "vitest";
import "../src/wording.js";

const words = globalThis.polylogueWording;
const page = { provider: "chatgpt", conversation: "c1", host: "ok", daemon: "running" };
const now = () => new Date().toISOString();

describe("what the page's conversation is called", () => {
  it("says when the page is not a chat or has no conversation open", () => {
    expect(words.conversation(null)).toMatchObject({ tone: "neutral", label: "Not a chat page" });
    expect(words.conversation({ provider: "claude-ai", conversation: null }).label).toBe("Claude: no conversation open");
  });

  it("says that the extension could not answer instead of calling the page not a chat", () => {
    expect(words.conversation({ error: "host_stopped" })).toMatchObject({
      tone: "bad", label: "The extension could not check this page", detail: "host_stopped",
    });
  });

  it("counts the archived messages in the singular and the plural", () => {
    expect(words.conversation({ ...page, state: "archived", messages: 1 }).label).toBe("In the archive: 1 message");
    expect(words.conversation({ ...page, state: "archived", messages: 12 })).toMatchObject({ tone: "ok", label: "In the archive: 12 messages" });
  });

  it("says what is stored while the daemon reads it or is silent", () => {
    expect(words.conversation({ ...page, state: "spooled" }).label).toBe("Stored; Polylogue is reading it");
    expect(words.conversation({ ...page, state: "spooled", daemon: "no_answer" })).toMatchObject({
      tone: "warn", label: "Stored; Polylogue reads it when its daemon answers",
    });
    expect(words.conversation({ ...page, state: "unknown", daemon: "no_answer" }).label).toBe("The daemon did not answer");
    expect(words.conversation({ ...page, state: "unknown", last: { outcome: "stored", at: now(), text: "Stored." } }).label)
      .toBe("Stored; the daemon did not answer");
  });

  it("says a conversation is not captured yet", () => {
    expect(words.conversation({ ...page, state: "missing" })).toMatchObject({ tone: "warn", label: "Not captured yet" });
  });

  it("names the command that installs the host when it is missing", () => {
    const said = words.conversation({ ...page, host: "missing" });
    expect(said.label).toBe("Polylogue is not installed for this browser");
    expect(said.detail).toBe("Run `polylogue install chrome --extension-id <id>`.");
  });

  it("puts the last failed capture first, with what it said", () => {
    const said = words.conversation({ ...page, state: "archived", messages: 3, last: { outcome: "failed", at: now(), text: "No answer." } });
    expect(said).toEqual({ tone: "bad", label: "The last capture failed", detail: "Last capture just now: No answer." });
  });
});
