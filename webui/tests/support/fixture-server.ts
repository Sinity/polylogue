import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import readline from 'node:readline';

export interface FixtureServer {
  baseUrl: string;
  process: ChildProcessWithoutNullStreams;
  output: string[];
}

export async function startFixtureServer(): Promise<FixtureServer> {
  const output: string[] = [];
  const child = spawn(process.execPath, ['scripts/fixture-server.mjs'], {
    cwd: process.cwd(),
    env: { ...process.env },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string) => output.push(chunk));

  const baseUrl = await new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`design-system fixture timed out: ${output.join('')}`)), 15_000);
    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      output.push(line);
      let payload: unknown;
      try { payload = JSON.parse(line); } catch { return; }
      if (
        typeof payload === 'object' && payload !== null
        && (payload as { kind?: string }).kind === 'ready'
        && typeof (payload as { base_url?: unknown }).base_url === 'string'
      ) {
        clearTimeout(timer);
        resolve((payload as { base_url: string }).base_url);
      }
    });
    child.once('exit', (code: number | null) => {
      clearTimeout(timer);
      reject(new Error(`design-system fixture exited ${code}: ${output.join('')}`));
    });
  });
  return { baseUrl, process: child, output };
}

export async function stopFixtureServer(server: FixtureServer | undefined): Promise<void> {
  if (!server || server.process.exitCode !== null) return;
  server.process.kill('SIGTERM');
  await new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, 5_000);
    server.process.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
  });
}
