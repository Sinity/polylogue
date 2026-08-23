#!/usr/bin/env node
// Shared-Chrome control proof for the deterministic dev-loop operation. It
// never launches a browser, allocates a debugging port, or creates a profile.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { assertAgentWindow, firstControlJson, runChromeControlBytes } from "./shared_chrome_control.mjs";

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be supplied by the declared dev-loop service`);
  return value;
}

function requireExpectedServiceContext() {
  const jobId = process.env.SINNIXD_JOB_ID || "";
  const unit = `sinnixd-job-${jobId}.service`;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(jobId)) {
    throw new Error("shared-Chrome dev-loop proof requires a Sinnixd job UUID");
  }
  if (process.env.SINNIXD_PROJECT_ID !== "polylogue" || process.env.SINNIXD_OPERATION !== "dev_loop_proof") {
    throw new Error("shared-Chrome dev-loop proof rejects execution outside the fixed dev-loop service context");
  }
  const cgroup = readFileSync("/proc/self/cgroup", "utf8").split("\n").find((line) => line.includes("::"))?.split("::", 2)[1] || "";
  if (!cgroup.includes(`/agent.slice/${unit}`)) {
    throw new Error("shared-Chrome dev-loop proof is not inside its matching Sinnixd transient unit");
  }
  const unitExecStart = spawnSync(
    "systemctl",
    ["--user", "show", unit, "--property=ExecStart", "--value"],
    { encoding: "utf8", timeout: 2000 },
  );
  const childCommand = unitExecStart.stdout.slice(unitExecStart.stdout.indexOf("/env -i") + "/env -i".length);
  if (unitExecStart.status !== 0 || !unitExecStart.stdout.includes("/env -i") || !["SINNIXD_JOB_ID", "SINNIXD_PROJECT_ID", "SINNIXD_OPERATION"].every((name) => childCommand.includes(`${name}=${process.env[name]}`))) {
    throw new Error("shared-Chrome dev-loop transient unit does not match the declared operation");
  }
}

export { assertAgentWindow } from "./shared_chrome_control.mjs";

export async function runChromeControl(args, timeoutMs, spawnCommand) {
  return firstControlJson(await runChromeControlBytes(args, timeoutMs, spawnCommand)) || {};
}

export async function runSharedChromeControlWorkflow({ extensionRoot, control = runChromeControl }) {
  if (!existsSync(path.join(extensionRoot, "manifest.json"))) {
    throw new Error("dev-loop extension root has no manifest.json");
  }
  await control(["status"]);
  await control(["load-extension", "--path", extensionRoot]);
  let createdTargetId = null;
  try {
    const target = await control(["agent-window", "--url", "about:blank"]);
    if (typeof target?.id === "string" && /^[A-F0-9]{32}$/i.test(target.id)) createdTargetId = target.id;
    assertAgentWindow(target, "about:blank");
    return { ok: true, shared_chrome: { extension_loaded: true, target_closed: true } };
  } finally {
    if (createdTargetId !== null) await control(["close", createdTargetId]);
  }
}

export async function runDevLoopSharedChromeProof() {
  requireExpectedServiceContext();
  return runSharedChromeControlWorkflow({ extensionRoot: path.resolve(requiredEnvironment("POLYLOGUE_DEV_LOOP_EXTENSION_ROOT")) });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runDevLoopSharedChromeProof()
    .then((result) => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch((error) => {
      process.stderr.write(`${error.stack || error.message || error}\n`);
      process.exitCode = 1;
    });
}
