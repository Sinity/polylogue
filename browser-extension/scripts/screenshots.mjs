#!/usr/bin/env node
// Capture store-submission screenshots of the extension popup.
//
// Renders src/popup.html under headless Chromium via Playwright, with the
// `chrome.storage.local` / `chrome.runtime` APIs stubbed so the popup
// shows representative state (online, supported page, last capture).
// Captures the Chrome Web Store + AMO required aspect ratios:
//
//   1280x800  (Chrome Web Store small tile)
//    640x400  (Chrome Web Store small tile alt)
//    750x1334 (AMO mobile)
//
// Usage:
//   node scripts/screenshots.mjs [--out DIR]
//
// Skips gracefully if Playwright is not installed. CI installs it via
// `npx playwright install chromium`.

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = { out: join(EXT_ROOT, "dist", "screenshots") };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--out") args.out = resolve(argv[++i]);
  }
  return args;
}

const STATES = [
  {
    name: "online-captured",
    storage: {
      receiverBaseUrl: "http://127.0.0.1:8765",
      polylogueState: {
        online: true,
        captured: true,
        provider: "chatgpt",
        title: "Designing a SQLite blob store",
        captured_at: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
        site_supported: true,
        updated_at: new Date().toISOString(),
      },
    },
  },
  {
    name: "online-unsupported",
    storage: {
      receiverBaseUrl: "http://127.0.0.1:8765",
      polylogueState: {
        online: true,
        captured: false,
        site_supported: false,
        updated_at: new Date().toISOString(),
      },
    },
  },
  {
    name: "offline",
    storage: {
      receiverBaseUrl: "http://127.0.0.1:8765",
      polylogueState: {
        online: false,
        captured: false,
        site_supported: true,
        updated_at: new Date().toISOString(),
      },
    },
  },
];

const SIZES = [
  { name: "1280x800", width: 1280, height: 800 },
  { name: "640x400", width: 640, height: 400 },
  { name: "750x1334", width: 750, height: 1334 },
];

async function main() {
  let playwright;
  try {
    playwright = await import("playwright");
  } catch {
    process.stdout.write("playwright not installed — skipping screenshots\n");
    process.stdout.write("install with: npx playwright install chromium\n");
    return;
  }

  const args = parseArgs(process.argv.slice(2));
  if (!existsSync(args.out)) mkdirSync(args.out, { recursive: true });

  const popupPath = join(EXT_ROOT, "src", "popup.html");
  const popupUrl = pathToFileURL(popupPath).toString();
  const popupJsPath = join(EXT_ROOT, "src", "popup.js");
  const popupJsSource = readFileSync(popupJsPath, "utf8");

  const browser = await playwright.chromium.launch();
  try {
    for (const state of STATES) {
      for (const size of SIZES) {
        const context = await browser.newContext({
          viewport: { width: size.width, height: size.height },
          deviceScaleFactor: 2,
        });
        const page = await context.newPage();
        // Inject a minimal chrome.* stub before any popup script runs.
        await page.addInitScript({
          content: `
            const __storage = ${JSON.stringify(state.storage)};
            globalThis.chrome = {
              storage: {
                local: {
                  get: async (defaults) => {
                    const out = { ...defaults };
                    for (const key of Object.keys(defaults)) {
                      if (key in __storage) out[key] = __storage[key];
                    }
                    return out;
                  },
                  set: async () => {},
                },
              },
              runtime: {
                sendMessage: async () => ({ ok: true }),
                onMessage: { addListener: () => {} },
              },
              tabs: {
                query: async () => [{ id: 1, url: "https://chatgpt.com/c/example" }],
              },
              action: {
                setBadgeText: async () => {},
                setBadgeBackgroundColor: async () => {},
              },
            };
          `,
        });
        await page.goto(popupUrl);
        // The popup is normally driven by chrome.runtime messaging; run
        // its actual script so we screenshot real layout, not a mock.
        await page.addScriptTag({ content: popupJsSource });
        // Allow async fetch-skipping to settle.
        await page.waitForTimeout(150);
        const outFile = join(args.out, `popup-${state.name}-${size.name}.png`);
        await page.screenshot({ path: outFile, fullPage: false });
        process.stdout.write(`captured ${outFile}\n`);
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stderr.write(`screenshot capture failed: ${err.stack || err.message}\n`);
  process.exit(1);
});
