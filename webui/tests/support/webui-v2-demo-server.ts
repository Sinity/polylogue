import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import readline from 'node:readline';

const repoRoot = path.resolve(process.cwd(), '..');

export interface WebuiV2DemoServer {
  baseUrl: string;
  demoSessionIds: readonly string[];
  largeSessionId: string;
  largeSessionMessageCount: number;
  process: ChildProcessWithoutNullStreams;
  output: string[];
  tempRoot: string;
}

interface ReadyReceipt {
  kind: 'ready';
  base_url: string;
  session_count: number;
  message_count: number;
  demo_session_ids: string[];
  large_session_id: string;
  large_session_message_count: number;
}

export async function startWebuiV2DemoServer(): Promise<WebuiV2DemoServer> {
  const scratchRoot = existsSync('/realm/tmp') ? '/realm/tmp' : tmpdir();
  const tempRoot = await mkdtemp(path.join(scratchRoot, 'polylogue-webui-v2-e2e-'));
  const output: string[] = [];
  const child = spawn('uv', ['run', 'python', 'tests/browser/webui_v2_demo_server.py'], {
    cwd: repoRoot,
    env: { ...process.env, POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT: path.join(tempRoot, 'archive') },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string) => output.push(chunk));

  const receipt = await new Promise<ReadyReceipt>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`webui-v2 demo server timed out: ${output.join('')}`)), 60_000);
    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      output.push(line);
      let payload: unknown;
      try { payload = JSON.parse(line); } catch { return; }
      if (typeof payload === 'object' && payload !== null && (payload as { kind?: string }).kind === 'ready') {
        clearTimeout(timer);
        resolve(payload as ReadyReceipt);
      }
    });
    child.once('exit', (code: number | null) => {
      clearTimeout(timer);
      reject(new Error(`webui-v2 demo server exited ${code}: ${output.join('')}`));
    });
  });

  return {
    baseUrl: receipt.base_url,
    demoSessionIds: receipt.demo_session_ids,
    largeSessionId: receipt.large_session_id,
    largeSessionMessageCount: receipt.large_session_message_count,
    process: child,
    output,
    tempRoot,
  };
}

export async function stopWebuiV2DemoServer(server: WebuiV2DemoServer | undefined): Promise<void> {
  if (!server) return;
  if (server.process.exitCode === null) {
    server.process.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 5_000);
      server.process.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
  await rm(server.tempRoot, { recursive: true, force: true });
}
