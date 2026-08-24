import { spawn } from "node:child_process";

const CONTROL_COMMAND = "/home/sinity/.local/bin/sinnix-chrome-control";
const AGENT_WORKSPACE = "agentbrowser";
const CONTROL_TIMEOUT_MS = 30_000;

export function firstControlJson(bytes) {
  for (const line of Buffer.from(bytes).toString("utf8").split("\n")) {
    try {
      const value = JSON.parse(line);
      if (value && typeof value === "object") return value;
    } catch {
      // The control command may emit diagnostics before its JSON response.
    }
  }
  return null;
}

export function assertAgentWindow(candidate, expectedUrl) {
  if (!candidate || typeof candidate !== "object" || !/^[A-F0-9]{32}$/i.test(candidate.id || "")) {
    throw new Error("shared Chrome control returned an invalid proof target");
  }
  if (candidate.url !== expectedUrl || candidate.parked !== true || candidate.workspace !== AGENT_WORKSPACE || candidate.show_with !== "F7") {
    throw new Error("shared Chrome proof target was not verified hidden on agentbrowser");
  }
  return candidate.id;
}

export function runChromeControlBytes(args, timeoutMs = CONTROL_TIMEOUT_MS, spawnCommand = spawn) {
  return new Promise((resolve, reject) => {
    const child = spawnCommand(CONTROL_COMMAND, args, { stdio: ["ignore", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const finish = (callback) => (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      finish(reject)(new Error(`sinnix-chrome-control ${args[0]} timed out`));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => stdout.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk) => stderr.push(Buffer.from(chunk)));
    child.once("error", finish(reject));
    child.once("close", (code) => {
      if (code !== 0) {
        finish(reject)(new Error(`sinnix-chrome-control ${args[0]} failed: ${Buffer.concat(stderr).toString("utf8") || Buffer.concat(stdout).toString("utf8") || `exit ${code}`}`));
        return;
      }
      finish(resolve)(Buffer.concat(stdout));
    });
  });
}
