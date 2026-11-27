// What the popup and the in-page panel say about the page's conversation
// and about Polylogue, in plain words. Loaded as a classic script by both.
(function (root) {
  const PROVIDERS = { chatgpt: "ChatGPT", "claude-ai": "Claude", grok: "Grok", gemini: "Gemini" };

  function providerName(provider) {
    return PROVIDERS[provider] || provider || "this site";
  }

  function ago(at) {
    const ms = Date.now() - Date.parse(at || "");
    if (!Number.isFinite(ms)) return "";
    if (ms < 60_000) return "just now";
    if (ms < 3_600_000) return `${Math.floor(ms / 60_000)} min ago`;
    if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)} h ago`;
    return new Date(at).toLocaleString();
  }

  // `page` is the background's answer about a tab (see capture_state.js).
  // Answers the capture state with the words for it: what is held is the
  // label, and a failed last capture is a line under it.
  function conversation(page) {
    const states = root.polylogueCaptureState;
    const state = states.of(page);
    const last = page?.last;
    const failed = last?.outcome === "failed";
    const lastLine = last ? `Last capture ${ago(last.at)}: ${last.text}` : "";
    const lastOrFailureLine = failed ? `The last capture failed ${ago(last.at)}: ${last.text}` : lastLine;
    const said = (tone, label, detail) => ({ state, tone, label, detail });
    switch (state) {
      case states.COULD_NOT_CHECK:
        return said("bad", "The extension could not check this page", String(page.error));
      case states.NOT_A_CHAT_PAGE:
        return said("neutral", "Not a chat page", "");
      case states.NO_CONVERSATION_OPEN:
        return said("neutral", `${providerName(page.provider)}: no conversation open`, "");
      case states.HOST_NOT_INSTALLED:
        return said("bad", "Polylogue is not installed for this browser", "Run `polylogue install chrome`.");
      case states.HOST_DID_NOT_ANSWER:
        return said("bad", "Polylogue did not answer", lastOrFailureLine);
      case states.IN_ARCHIVE:
      case states.IN_ARCHIVE_FILES_DOWNLOADING:
        return said(failed ? "warn" : "ok", `In the archive: ${page.messages} message${page.messages === 1 ? "" : "s"}`, lastOrFailureLine);
      case states.STORED_BEING_READ:
        return said(failed ? "warn" : "ok", "Stored; Polylogue is reading it", lastOrFailureLine);
      case states.STORED_DAEMON_NOT_ANSWERING:
        return said("warn", "Stored; Polylogue reads it when its daemon answers", lastOrFailureLine);
      case states.LAST_CAPTURE_FAILED:
        return said("bad", "The last capture failed", lastLine);
      case states.NOT_CAPTURED:
        return said("warn", "Not captured yet", lastLine);
      case states.DAEMON_DID_NOT_ANSWER:
        return said("warn", "The daemon did not answer", lastLine || "It is stopped (start it with `polylogue daemon`) or busy.");
      default:
        throw new Error(`unknown capture state: ${state}`);
    }
  }

  root.polylogueWording = Object.freeze({ conversation, providerName, ago });
})(globalThis);
