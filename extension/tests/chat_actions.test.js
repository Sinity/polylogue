import { describe, expect, it } from "vitest";
import { ChatActions, actionUrl } from "../src/background/chat_actions.js";
import { Host } from "../src/background/host.js";
import { attachmentInPage, failureOutcome } from "../src/actions/chatgpt.js";
import { fakeChrome, json } from "./support/fake_chrome.js";

const ACTION = {
  id: "a1", provider: "chatgpt", conversation: null, text: "Draft this", model: "GPT X", submit: false,
  attachments: [{ id: "f1", name: "notes.txt", media_type: "text/plain", size: 5 }],
};

function setup(pageResult) {
  const reports = [];
  const routes = {
    "GET /v1/chat-actions": (_body, path) => ({ json: { action: path.includes("claim_by=ext-1") ? ACTION : null } }),
    "GET /v1/chat-actions/a1/attachments/f1": () => ({ bytes: new TextEncoder().encode("hello"), type: "text/plain" }),
    "POST /v1/chat-actions/a1/report": (body) => { reports.push(json(body)); return { json: { action: {} } }; },
  };
  const chrome = fakeChrome({ routes });
  const ran = [];
  chrome.scripting.executeScript = async ({ func, args }) => {
    ran.push(func.name);
    if (func.name === "chatActionInPage") return [{ result: pageResult(globalThis.__polylogueChatAttachments, args) }];
    return [{ result: func(...args) }];
  };
  const captured = [];
  const actions = new ChatActions({ chrome, host: new Host(chrome.runtime), instance: async () => "ext-1", capture: async (tabId) => captured.push(tabId) });
  return { actions, chrome, reports, ran, captured };
}

describe("a chat action queued with `polylogue chat send`", () => {
  it("is drafted in a background tab with its file, and the draft is reported", async () => {
    let files = null;
    const { actions, chrome, reports, ran } = setup((attachments, [action, runId]) => {
      files = attachments;
      expect(attachments.actionId).toBe(action.id);
      expect(attachments.runId).toBe(runId);
      return { ok: true, outcome: "drafted", provider_conversation_url: "https://chatgpt.com/", observed_model: "GPT X" };
    });
    await actions.step();
    expect(chrome.created[0]).toMatchObject({ url: "https://chatgpt.com/", active: false });
    expect(ran.filter((name) => name === "attachmentInPage").length).toBeGreaterThanOrEqual(4);
    const file = files.entries.get("f1").file;
    expect(await file.text()).toBe("hello");
    expect(file.name).toBe("notes.txt");
    expect(reports.map((r) => r.outcome)).toEqual(["progress", "drafted"]);
    expect(reports[1]).toMatchObject({ by: "ext-1", receipt: { model: "GPT X" } });
    // The draft stays open for the operator.
    expect(chrome.tabs.list).toHaveLength(1);
    delete globalThis.__polylogueChatAttachments;
  });

  it("reports an unknown outcome when sending may have happened", async () => {
    const { actions, reports } = setup(() => ({ ok: false, detail: "provider_turn_receipt_timeout", submission_may_have_occurred: true }));
    ACTION.submit = true;
    try {
      await actions.step();
    } finally {
      ACTION.submit = false;
      delete globalThis.__polylogueChatAttachments;
    }
    expect(reports.at(-1).outcome).toBe("outcome_unknown");
  });

  it("names its target page and its failures in the outcomes Polylogue knows", () => {
    expect(actionUrl({ conversation: "c 1" })).toBe("https://chatgpt.com/c/c%201");
    expect(actionUrl({ project: "https://chatgpt.com/g/g-p-abc/" })).toBe("https://chatgpt.com/g/g-p-abc/project");
    expect(failureOutcome(new Error("HTTP 429 Too Many Requests")).outcome).toBe("rate_limited");
    expect(failureOutcome(new Error("protocol_selector_missing")).outcome).toBe("provider_drift");
    expect(failureOutcome(new Error("Cloudflare challenge")).outcome).toBe("auth_challenge");
  });

  it("keeps each run's files apart", () => {
    attachmentInPage("a1", "run-1", "begin");
    expect(() => attachmentInPage("a1", "run-2", "start", { id: "f", size: 1, name: "x", media_type: "text/plain" }))
      .toThrow("protocol_attachment_run_mismatch");
    expect(() => attachmentInPage("a1", "run-2", "discard")).toThrow("protocol_attachment_run_mismatch");
    expect(globalThis.__polylogueChatAttachments.runId).toBe("run-1");
    attachmentInPage("a1", "run-1", "discard");
    expect(globalThis.__polylogueChatAttachments).toBeUndefined();
  });
});
