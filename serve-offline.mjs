#!/usr/bin/env node
// Zero-dependency static file server for running the built app offline.
//
//   node serve-offline.mjs            -> http://localhost:8080/
//   node serve-offline.mjs 9000       -> http://localhost:9000/
//   node serve-offline.mjs --port 9000 --host 0.0.0.0
//
// Serves the folder this file lives in (dist/ after `npm run build`).
// http://localhost counts as a secure context, so the webcam works offline.
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { dirname, extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.woff2': 'font/woff2',
};

function parseArgs(argv) {
  let port = Number(process.env.PORT) || 8080;
  let host = '127.0.0.1';
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--port' || a === '-p') port = Number(argv[++i]);
    else if (a === '--host') host = argv[++i] ?? host;
    else if (/^\d+$/.test(a)) port = Number(a);
  }
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error(`Invalid port. Usage: node serve-offline.mjs [port] [--host 0.0.0.0]`);
    process.exit(1);
  }
  return { port, host };
}

/** Map a request URL path to a file inside ROOT, or null if it escapes ROOT. */
function resolvePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath);
  } catch {
    return null;
  }
  if (decoded.includes('\0')) return null;
  const full = normalize(join(ROOT, decoded));
  if (full !== ROOT && !full.startsWith(ROOT + sep)) return null;
  return full;
}

function send(res, status, body, type = 'text/plain; charset=utf-8') {
  res.writeHead(status, { 'Content-Type': type, 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

const server = createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'Method not allowed');

  const { pathname } = new URL(req.url ?? '/', 'http://localhost');
  let file = resolvePath(pathname);
  if (file === null) return send(res, 403, 'Forbidden');

  try {
    let info = await stat(file);
    if (info.isDirectory()) {
      file = join(file, 'index.html');
      info = await stat(file);
    }
    res.writeHead(200, {
      'Content-Type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
      'Content-Length': info.size,
      'Cache-Control': 'no-cache',
    });
    if (req.method === 'HEAD') return res.end();
    createReadStream(file)
      .on('error', () => res.destroy())
      .pipe(res);
  } catch {
    send(res, 404, 'Not found');
  }
});

const { port, host } = parseArgs(process.argv.slice(2));

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Try: node serve-offline.mjs ${port + 1}`);
  } else {
    console.error(err.message);
  }
  process.exit(1);
});

server.listen(port, host, () => {
  console.log(`Serving ${ROOT}`);
  console.log(`Open  http://localhost:${port}/  (Ctrl+C to stop)`);
});
