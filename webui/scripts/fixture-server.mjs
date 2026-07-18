import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const siteRoot = resolve(root, 'dist/site');
const ssrEntry = resolve(root, 'dist/server/entry.mjs');
const { renderDocument } = await import(`${pathToFileURL(ssrEntry).href}?v=${Date.now()}`);
const contentTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
]);

function headers(contentType) {
  return {
    'cache-control': 'no-store',
    'content-type': contentType,
    'content-security-policy': "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    'referrer-policy': 'no-referrer',
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
  };
}

async function serveAsset(pathname, response, headOnly) {
  const relativePath = pathname.replace(/^\/assets\//, 'assets/');
  const filePath = resolve(siteRoot, relativePath);
  if (!filePath.startsWith(`${siteRoot}${sep}`)) return false;
  try {
    const info = await stat(filePath);
    if (!info.isFile()) return false;
    const body = await readFile(filePath);
    response.writeHead(200, { ...headers(contentTypes.get(extname(filePath)) ?? 'application/octet-stream'), 'content-length': body.length });
    response.end(headOnly ? undefined : body);
    return true;
  } catch {
    return false;
  }
}

const server = createServer(async (request, response) => {
  const method = request.method ?? 'GET';
  if (method !== 'GET' && method !== 'HEAD') {
    response.writeHead(405, { ...headers('text/plain; charset=utf-8'), allow: 'GET, HEAD' });
    response.end('Method not allowed');
    return;
  }
  const target = request.url ?? '/';
  const url = new URL(target, 'http://fixture.invalid');
  if (url.pathname.startsWith('/assets/')) {
    if (await serveAsset(url.pathname, response, method === 'HEAD')) return;
    response.writeHead(404, headers('text/plain; charset=utf-8'));
    response.end('Asset not found');
    return;
  }
  const rendered = renderDocument(target);
  const body = Buffer.from(rendered.html, 'utf8');
  response.writeHead(rendered.status, { ...headers('text/html; charset=utf-8'), 'content-length': body.length });
  response.end(method === 'HEAD' ? undefined : body);
});

server.listen(0, '127.0.0.1', () => {
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Fixture server failed to bind a TCP port');
  process.stdout.write(`${JSON.stringify({ kind: 'ready', base_url: `http://127.0.0.1:${address.port}` })}\n`);
});

function close() {
  server.close(() => process.exit(0));
}
process.on('SIGINT', close);
process.on('SIGTERM', close);
