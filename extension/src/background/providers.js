// Written by `cargo xtask extension-providers` from Polylogue's origin
// registry (crates/sources/src/origin.rs); edit the registry, not this file.
//
// Each provider the extension captures, by the name the extension gives it:
// the hosts that serve its conversation pages, the page opened to work in
// when none is open, and the path segment before a conversation's ID.
export const PROVIDERS = {
  "chatgpt": { hosts: ["chatgpt.com"], home: "https://chatgpt.com/", path: "c" },
  "claude-ai": { hosts: ["claude.ai"], home: "https://claude.ai/recents", path: "chat" },
  "gemini": { hosts: ["gemini.google.com"], home: "https://gemini.google.com/app", path: "app" },
  "grok": { hosts: ["grok.com"], home: "https://grok.com/", path: "c" },
};
