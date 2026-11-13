// Release inputs and browser-specific manifest transformations. Registration
// remains in the committed manifest; packaging never changes source files.
export const FIREFOX_GECKO_ID = "polylogue-browser-capture@sinity.dev";
export const runtimeFiles = Object.freeze([
  "manifest.json",
  "src/actions/chatgpt.js", "src/background.js",
  "src/background/backfill.js", "src/background/capture.js", "src/background/chat_actions.js",
  "src/background/host.js", "src/background/main.js", "src/background/notes.js",
  "src/background/outbox.js", "src/background/provider.js", "src/background/tabs.js",
  "src/content/chatgpt.js", "src/content/claude.js", "src/content/gemini.js",
  "src/content/grok.js", "src/content/page.js", "src/content/panel.js",
  "src/popup.html", "src/popup.js", "src/wording.js",
]);

export function browserVersion(raw) {
  if (typeof raw !== "string") throw new Error(`invalid browser version: ${raw}`);
  const version = raw.replace(/[+-].*$/, "").replace(/\.dev\d*$/, "");
  if (!/^(0|[1-9]\d*)(\.(0|[1-9]\d*)){0,3}$/.test(version)
      || version.split(".").some((part) => Number(part) > 65535)) {
    throw new Error(`invalid browser version: ${raw}`);
  }
  return version;
}

export function browserManifest(source, rawVersion, browser) {
  const manifest = structuredClone(source);
  manifest.version = browserVersion(rawVersion);
  if (browser === "firefox") {
    if (manifest.background?.service_worker) {
      manifest.background.scripts = [manifest.background.service_worker];
    }
    manifest.browser_specific_settings = {
      gecko: { id: FIREFOX_GECKO_ID, strict_min_version: "121.0" },
    };
  } else if (browser !== "chrome") throw new Error(`unknown browser: ${browser}`);
  return manifest;
}
