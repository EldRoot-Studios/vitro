"""Local web server for the Vitro app (`vitro web [FILE ...]`).

Serves the built app from `web/dist/` and, on request, the trace files it was told about: the files given on the
command line and the traces in `samples/`. The browser parses the traces itself; nothing is uploaded and the
server needs neither PyTorch nor NumPy beyond the container header reader.

Routes (GET and HEAD only):
    /                 the app
    /<asset>          a file under web/dist/ with an allow-listed extension
    /api/library      JSON list of the registered trace files, headers re-read on every call
    /files/<id>       raw bytes of registered file <id>

Safety: the socket is bound to 127.0.0.1, the Host header must name this server (a page on another site cannot
reach it through DNS rebinding), only files registered at start-up are served, by number rather than by path,
and every response carries a strict Content-Security-Policy.

Run:  .venv/Scripts/python.exe -m vitro.cli web samples/t5_small.vtdump samples/t5_small.vtdemo
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit

from .container import read_header

ROOT = Path(__file__).resolve().parents[2]
# The built app. `web/dist` is committed, so `vitro web` needs no Node; `npm run build` in web/ renews it.
WEB_DIR = ROOT / 'web' / 'dist'
SAMPLES_DIR = ROOT / 'samples'
DEFAULT_PORT = 8640
TRACE_SUFFIXES = ('.vtdump', '.vtdemo')

# Explicit on purpose: Windows takes MIME types from the registry, where .js is sometimes text/plain, and a
# browser refuses to run an ES module served as text/plain.
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.woff2': 'font/woff2',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class TraceFileError(ValueError):
    """A file that was asked for is not a readable Vitro trace; the message names the file and the problem."""


@dataclass(frozen=True)
class Entry:
    """A trace file the server may serve, fixed at start-up."""

    path: Path
    source: str  # 'argument' or 'sample'


def describe(entry: Entry, file_id: int) -> dict:
    """Library record of one entry, from its container header. Raises TraceFileError if it cannot be read."""
    try:
        header = read_header(str(entry.path))
        size = entry.path.stat().st_size
    except (OSError, ValueError) as e:
        raise TraceFileError(str(e)) from e
    trace = header.get('trace', {})
    return {
        'id': file_id,
        'name': entry.path.name,
        'kind': header.get('kind', 'dump'),
        'size': size,
        'traceId': trace.get('id', ''),
        'title': trace.get('title', entry.path.stem),
        'model': trace.get('model_name', ''),
        'arch': trace.get('arch', ''),
        'source': entry.source,
    }


def collect_entries(paths: Sequence[str | Path], samples_dir: str | Path | None = SAMPLES_DIR) -> list[Entry]:
    """The files to register: the explicit `paths` first, then the traces of `samples_dir`, each once.

    An explicit path that is not a readable trace is an error: the user asked for it by name.
    """
    entries: list[Entry] = []
    seen: set[Path] = set()
    samples = None if samples_dir is None else Path(samples_dir)
    for p in paths:
        path = Path(p).resolve()
        if not path.is_file():
            raise TraceFileError(f'{p}: no such file')
        describe(Entry(path, 'argument'), 0)  # raises with the container's own message when it is not a trace
        if path not in seen:
            seen.add(path)
            entries.append(Entry(path, 'argument'))
    if samples is not None and samples.is_dir():
        for path in sorted(samples.iterdir()):
            if path.suffix in TRACE_SUFFIXES and path.resolve() not in seen:
                seen.add(path.resolve())
                entries.append(Entry(path.resolve(), 'sample'))
    return entries


def library_json(entries: Sequence[Entry]) -> dict:
    """Body of /api/library. A file that has become unreadable is listed with an `error`, never dropped."""
    files = []
    for i, entry in enumerate(entries):
        try:
            files.append(describe(entry, i))
        except TraceFileError as e:
            files.append({'id': i, 'name': entry.path.name, 'source': entry.source, 'error': str(e)})
    return {'files': files}


class _Handler(BaseHTTPRequestHandler):
    server_version = 'VitroWeb'
    protocol_version = 'HTTP/1.1'

    server: 'VitroServer'

    def log_request(self, code='-', size='-') -> None:
        # Quiet unless something went wrong: a page load is dozens of requests.
        if isinstance(code, HTTPStatus):
            code = code.value
        if str(code).isdigit() and int(code) >= 400:
            super().log_request(code, size)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, head_only: bool, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', CSP)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _fail(self, status: HTTPStatus, message: str, head_only: bool = False) -> None:
        self._send(status, f'{status.value} {message}\n'.encode('utf-8'), 'text/plain; charset=utf-8', head_only)

    def _serve_file(self, path: Path, content_type: str, head_only: bool) -> None:
        try:
            body = path.read_bytes()
        except OSError as e:
            self._fail(HTTPStatus.NOT_FOUND, f'cannot read {path.name}: {e.strerror}', head_only)
            return
        self._send(HTTPStatus.OK, body, content_type, head_only)

    def _handle(self, head_only: bool) -> None:
        host = (self.headers.get('Host') or '').lower()
        if host not in self.server.allowed_hosts:
            self._fail(HTTPStatus.FORBIDDEN, 'Host not allowed', head_only)
            return
        path = unquote(urlsplit(self.path).path)
        if path == '/api/library':
            body = json.dumps(library_json(self.server.entries), ensure_ascii=False).encode('utf-8')
            self._send(HTTPStatus.OK, body, 'application/json; charset=utf-8', head_only)
        elif path.startswith('/files/'):
            tail = path[len('/files/'):]
            if not tail.isascii() or not tail.isdigit() or int(tail) >= len(self.server.entries):
                self._fail(HTTPStatus.NOT_FOUND, 'no such trace file', head_only)
                return
            self._serve_file(self.server.entries[int(tail)].path, 'application/octet-stream', head_only)
        else:
            self._serve_asset(path, head_only)

    def _serve_asset(self, path: str, head_only: bool) -> None:
        rel = 'index.html' if path == '/' else path[1:]
        parts = rel.split('/')
        if any(p in ('', '.', '..') or '\\' in p or ':' in p or '\0' in p for p in parts):
            self._fail(HTTPStatus.NOT_FOUND, 'not found', head_only)
            return
        content_type = CONTENT_TYPES.get(Path(rel).suffix.lower())
        target = (self.server.web_dir / rel).resolve()
        if content_type is None or not target.is_relative_to(self.server.web_dir) or not target.is_file():
            self._fail(HTTPStatus.NOT_FOUND, 'not found', head_only)
            return
        self._serve_file(target, content_type, head_only)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        self._handle(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head_only=True)

    def _refuse_method(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, b'405 only GET and HEAD are served\n', 'text/plain; charset=utf-8',
                   False, {'Allow': 'GET, HEAD'})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse_method


class VitroServer(ThreadingHTTPServer):
    """HTTP server bound to the loopback interface that knows its trace files and its web root."""

    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that another one is listening on, and both would
    # answer. Without it a taken port is an error, which `serve` reports; elsewhere it only frees a port in TIME_WAIT.
    allow_reuse_address = sys.platform != 'win32'

    def __init__(self, entries: Sequence[Entry], port: int = 0, web_dir: Path = WEB_DIR) -> None:
        self.entries = list(entries)
        self.web_dir = Path(web_dir).resolve()
        super().__init__(('127.0.0.1', port), _Handler)
        bound = self.server_address[1]
        self.allowed_hosts = {f'127.0.0.1:{bound}', f'localhost:{bound}'}

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.server_address[1]}/'


def serve(paths: Sequence[str | Path], port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    """Runs the app until interrupted. Returns the process exit code."""
    if not (WEB_DIR / 'index.html').is_file():
        print(f'vitro web: the app is missing: {WEB_DIR / "index.html"} not found. Use a Vitro checkout, or run `npm run build` in web/.', file=sys.stderr)
        return 2
    try:
        entries = collect_entries(paths)
    except TraceFileError as e:
        print(f'vitro web: {e}', file=sys.stderr)
        return 2
    for entry in entries:
        if entry.source == 'sample':
            try:
                describe(entry, 0)
            except TraceFileError as e:
                print(f'vitro web: warning: sample not usable: {e}', file=sys.stderr)
    try:
        server = VitroServer(entries, port)
    except OSError as e:
        print(f'vitro web: cannot listen on 127.0.0.1:{port}: {e.strerror}. Choose another port with --port.', file=sys.stderr)
        return 2
    opened = [i for i, e in enumerate(entries) if e.source == 'argument']
    url = server.url + (f'#open={",".join(map(str, opened))}' if opened else '')
    print(f'Vitro is running at {server.url}  (Ctrl+C to stop)')
    for i in opened:
        print(f'  opened: {entries[i].path}')
    if open_browser:
        timer = threading.Timer(0.3, lambda: webbrowser.open(url) or print(f'Could not start a browser; open {url} yourself.'))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        server.server_close()
    return 0
