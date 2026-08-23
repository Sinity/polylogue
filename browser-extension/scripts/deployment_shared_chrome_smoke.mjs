#!/usr/bin/env node
// Shared-Chrome deployment render proof. It uses only Sinnix's control boundary
// for the existing operator Chrome and owns one parked agentbrowser target.

import { pathToFileURL } from "node:url";

import { assertAgentWindow, firstControlJson, runChromeControlBytes } from "./shared_chrome_control.mjs";

const DEPLOYED_ROOT_URL = "http://127.0.0.1:8766/";
const RENDER_TIMEOUT_S = 30;

export function assertDeployedRoot(dom) {
  const document = Buffer.from(dom).toString("utf8");
  if (!/<title>\s*Polylogue\s*<\/title>/i.test(document) || !/id=["']conv-header["']/.test(document)) {
    throw new Error("shared Chrome deployment render did not contain the Polylogue root marker");
  }
}

export async function runDeploymentSharedChromeSmoke({ control = runChromeControlBytes } = {}) {
  await control(["status"]);
  let targetId = null;
  try {
    const target = firstControlJson(await control(["agent-window", "--url", DEPLOYED_ROOT_URL]));
    if (target === null) throw new Error("shared Chrome control returned invalid agent-window JSON");
    if (typeof target.id === "string" && /^[A-F0-9]{32}$/i.test(target.id)) targetId = target.id;
    targetId = assertAgentWindow(target, DEPLOYED_ROOT_URL);
    await control([
      "await",
      targetId,
      "--js",
      "document.readyState === 'complete'",
      "--timeout-sec",
      String(RENDER_TIMEOUT_S),
    ]);
    const dom = await control(["get-html", targetId]);
    const screenshot = await control(["screenshot", targetId, "--format", "png"]);
    assertDeployedRoot(dom);
    if (screenshot.length === 0) throw new Error("shared Chrome deployment render produced empty evidence");
    return {
      ok: true,
      render: {
        url: DEPLOYED_ROOT_URL,
        dom_bytes: dom.length,
        screenshot_bytes: screenshot.length,
        target_closed: true,
      },
    };
  } finally {
    if (targetId !== null) await control(["close", targetId]);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runDeploymentSharedChromeSmoke()
    .then((result) => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch((error) => {
      process.stderr.write(`${error.stack || error.message || error}\n`);
      process.exitCode = 1;
    });
}
