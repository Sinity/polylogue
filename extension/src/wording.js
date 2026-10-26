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
    if (ms < 3_600_000) return `${Math.round(ms / 60_000)} min ago`;
    if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)} h ago`;
    return new Date(at).toLocaleString();
  }

  // `page` is the background's answer about a tab: the conversation on
  // it, what the archive holds of it, and the extension's last capture.
  function conversation(page) {
    if (!page?.provider) return { tone: "neutral", label: "Not a chat page", detail: "" };
    if (!page.conversation) {
      return { tone: "neutral", label: `${providerName(page.provider)}: no conversation open`, detail: "" };
    }
    const last = page.last;
    const lastLine = last ? `Last capture ${ago(last.at)}: ${last.text}` : "";
    if (page.host === "missing") {
      return {
        tone: "bad",
        label: "Polylogue is not installed for this browser",
        detail: "Run `polylogue install-agent chrome --extension-id <id>`.",
      };
    }
    if (last?.outcome === "failed") {
      return { tone: "bad", label: "The last capture failed", detail: lastLine };
    }
    switch (page.state) {
      case "archived":
        return {
          tone: "ok",
          label: `In the archive: ${page.messages} message${page.messages === 1 ? "" : "s"}`,
          detail: lastLine,
        };
      case "spooled":
        return {
          tone: page.daemon === "running" ? "ok" : "warn",
          label: page.daemon === "running" ? "Stored; Polylogue is reading it" : "Stored; Polylogue reads it when its daemon answers",
          detail: lastLine,
        };
      case "missing":
        return { tone: "warn", label: "Not captured yet", detail: lastLine };
      default:
        return {
          tone: last?.outcome === "stored" ? "ok" : "warn",
          label: last?.outcome === "stored" ? "Stored; the daemon did not answer" : "The daemon did not answer",
          detail: lastLine || "It is stopped (start it with `polylogue daemon`) or busy.",
        };
    }
  }

  root.polylogueWording = Object.freeze({ conversation, providerName, ago });
})(globalThis);
