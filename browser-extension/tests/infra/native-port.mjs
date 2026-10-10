// Actual native-host stdio framing for the neutral Python integration fixture.
import { spawn } from "node:child_process";
import { Buffer } from "node:buffer";

export function nativeRuntime(command) {
  return { connectNative(name) {
    if (name !== "com.polylogue.browser_capture") throw new Error("native_host_unknown");
    const child = spawn(command[0], command.slice(1), { stdio: ["pipe", "pipe", "ignore"] });
    const messages = new Set(); const disconnects = new Set();
    let buffer = Buffer.alloc(0); let ended = false;
    const eof = () => { if (!ended) { ended = true; for (const fn of disconnects) fn(); } };
    child.on("error", eof); child.on("exit", eof);
    child.stdin.on("error", eof);
    child.stdout.on("data", bytes => {
      buffer = Buffer.concat([buffer, bytes]);
      while (buffer.length >= 4) {
        const length = buffer.readUInt32LE(0);
        if (length > 1024 * 1024) { child.stdin.end(); eof(); return; }
        if (buffer.length < length + 4) return;
        let frame;
        try { frame = JSON.parse(buffer.subarray(4, length + 4).toString("utf8")); }
        catch { child.stdin.end(); eof(); return; }
        buffer = buffer.subarray(length + 4);
        for (const fn of messages) fn(frame);
      }
    });
    return {
      onMessage: { addListener: fn => messages.add(fn), removeListener: fn => messages.delete(fn) },
      onDisconnect: { addListener: fn => disconnects.add(fn), removeListener: fn => disconnects.delete(fn) },
      postMessage(frame) {
        const bytes = Buffer.from(JSON.stringify(frame)); const prefix = Buffer.alloc(4);
        prefix.writeUInt32LE(bytes.length); child.stdin.write(prefix); child.stdin.write(bytes);
      },
      disconnect() { ended = true; child.stdin.end(); },
    };
  } };
}
