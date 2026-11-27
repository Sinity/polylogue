import { describe, expect, it } from "vitest";
import "../src/capture_state.js";
import "../src/wording.js";

const words = globalThis.polylogueWording;
const states = globalThis.polylogueCaptureState;
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
    expect(words.conversation({ ...page, state: "unknown", daemon: "no_answer" })).toMatchObject({
      tone: "warn", label: "The daemon did not answer", detail: "It is stopped (start it with `polylogue daemon`) or busy.",
    });
    // Nothing in the spool is not called stored; the last capture is the line under it.
    expect(words.conversation({ ...page, state: "unknown", daemon: "no_answer", last: { outcome: "stored", at: now(), text: "stored" } }))
      .toMatchObject({ tone: "warn", label: "The daemon did not answer", detail: "Last capture just now: stored" });
  });

  it("says a conversation is not captured yet", () => {
    expect(words.conversation({ ...page, state: "missing" })).toMatchObject({ tone: "warn", label: "Not captured yet" });
  });

  it("names the command that installs the host when it is missing", () => {
    const said = words.conversation({ ...page, host: "missing" });
    expect(said.label).toBe("Polylogue is not installed for this browser");
    expect(said.detail).toBe("Run `polylogue install chrome`.");
  });

  it("says that Polylogue did not answer when the installed host fails, as the popup's header does", () => {
    expect(words.conversation({ ...page, host: "failed", state: "unknown", daemon: "no_answer" })).toEqual({
      state: "host_did_not_answer", tone: "bad", label: "Polylogue did not answer", detail: "",
    });
  });

  const failure = { outcome: "failed", at: now(), text: "No answer." };

  it("keeps what is held as the label when the last capture failed, with the failure under it", () => {
    expect(words.conversation({ ...page, state: "archived", messages: 3, last: failure })).toEqual({
      state: "in_archive", tone: "warn", label: "In the archive: 3 messages", detail: "The last capture failed just now: No answer.",
    });
    expect(words.conversation({ ...page, state: "spooled", last: failure })).toMatchObject({
      tone: "warn", label: "Stored; Polylogue is reading it", detail: "The last capture failed just now: No answer.",
    });
    expect(words.conversation({ ...page, state: "spooled", daemon: "no_answer", last: failure })).toMatchObject({
      tone: "warn", label: "Stored; Polylogue reads it when its daemon answers", detail: "The last capture failed just now: No answer.",
    });
  });

  it("says the last capture failed when nothing holds the conversation", () => {
    expect(words.conversation({ ...page, state: "missing", last: failure })).toEqual({
      state: "last_capture_failed", tone: "bad", label: "The last capture failed", detail: "Last capture just now: No answer.",
    });
    expect(words.conversation({ ...page, state: "unknown", daemon: "no_answer", last: failure }).label).toBe("The last capture failed");
  });
});

describe("a conversation's capture state", () => {
  const failed = { outcome: "failed" };

  it("is one of a closed set, decided by what is held before how the last capture went", () => {
    const cases = [
      [{ error: "host_stopped" }, "could_not_check"],
      [null, "not_a_chat_page"],
      [{ provider: null }, "not_a_chat_page"],
      [{ provider: "grok", conversation: null }, "no_conversation_open"],
      [{ ...page, host: "missing", state: "unknown" }, "host_not_installed"],
      [{ ...page, host: "failed", state: "unknown", last: failed }, "host_did_not_answer"],
      [{ ...page, state: "archived" }, "in_archive"],
      [{ ...page, state: "archived", last: failed }, "in_archive"],
      [{ ...page, state: "archived", downloading: true }, "in_archive_files_downloading"],
      [{ ...page, state: "spooled" }, "stored_being_read"],
      [{ ...page, state: "spooled", last: failed }, "stored_being_read"],
      [{ ...page, state: "spooled", daemon: "no_answer" }, "stored_daemon_not_answering"],
      [{ ...page, state: "missing", last: failed }, "last_capture_failed"],
      [{ ...page, state: "unknown", daemon: "no_answer", last: failed }, "last_capture_failed"],
      [{ ...page, state: "missing" }, "not_captured"],
      [{ ...page, state: "unknown", daemon: "no_answer" }, "daemon_did_not_answer"],
    ];
    for (const [facts, state] of cases) expect([facts, states.of(facts)]).toEqual([facts, state]);
    expect([...new Set(cases.map(([, state]) => state))].sort()).toEqual([...states.STATES].sort());
  });

  it("has words for every state", () => {
    const facts = { ...page, error: undefined, messages: 2 };
    for (const [given] of [
      [{ error: "x" }], [null], [{ provider: "grok" }], [{ ...facts, host: "missing" }], [{ ...facts, host: "failed" }],
      [{ ...facts, state: "archived" }], [{ ...facts, state: "archived", downloading: true }], [{ ...facts, state: "spooled" }],
      [{ ...facts, state: "spooled", daemon: "no_answer" }], [{ ...facts, state: "missing", last: { outcome: "failed", at: now(), text: "x" } }],
      [{ ...facts, state: "missing" }], [{ ...facts, state: "unknown", daemon: "no_answer" }],
    ]) {
      const said = words.conversation(given);
      expect(said.state).toBe(states.of(given));
      expect(said.label).not.toBe("");
    }
  });

  it("reads the host's answer about several conversations, one state each", () => {
    const answer = { daemon: "running", conversations: [
      { id: "a", state: "archived", messages: 2, downloading: false },
      { id: "b", state: "archived", messages: 2, downloading: true },
      { id: "c", state: "spooled", messages: null, downloading: null },
      { id: "d", state: "missing" },
    ] };
    const read = states.ofArchiveAnswer("chatgpt", answer);
    expect(read).toEqual(["in_archive", "in_archive_files_downloading", "stored_being_read", "not_captured"]);
    expect(read.map(states.inArchive)).toEqual([true, true, false, false]);
    expect(read.map(states.holdsCapture)).toEqual([true, true, true, false]);
    expect(states.ofArchiveAnswer("chatgpt", { daemon: "no_answer", conversations: [{ id: "a", state: "spooled" }, { id: "b", state: "unknown" }] }))
      .toEqual(["stored_daemon_not_answering", "daemon_did_not_answer"]);
  });
});
