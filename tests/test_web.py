"""Verification of the web app: `vitro web` (the server and the built app) and the reader that runs in the browser.

What it proves, and how:
  * the server serves the app and the registered trace files and nothing else (paths, methods, the Host header,
    security headers), and tells a bad file from a good one the way the container reader does;
  * the browser's reader and NumPy agree: the same numbers from the sample traces to 1e-9, the same refusal
    message for every kind of broken file, the same float32 for all 65536 float16 bit patterns;
  * the TypeScript tests of the app pass (vitest), and the committed `web/dist` is what the sources build to;
  * `import vitro` does not pull in PyTorch, so `vitro web` and `vitro info` start in a fraction of a second.

The parts that run the app's TypeScript need Node 22.18 or newer (it runs .ts files directly). Without it they
are reported as SKIPPED, loudly; set VITRO_REQUIRE_NODE=1 to make a missing Node an error.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, 'web')
DIST = os.path.join(WEB, 'dist')
SAMPLES = os.path.join(ROOT, 'samples')
PY_ROOT = os.path.join(ROOT, 'python')
for p in (PY_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_demo  # noqa: E402
import test_viewer  # noqa: E402
from vitro import webserver  # noqa: E402
from vitro.container import ContainerFile, Writer, save  # noqa: E402

SKIPPED: list[str] = []


def node_exe() -> str | None:
    """Path of a Node new enough to run TypeScript files, or None (the caller reports the skip)."""
    exe = shutil.which('node')
    if exe is None:
        return None
    out = subprocess.run([exe, '--version'], capture_output=True, text=True, encoding='utf-8', timeout=30).stdout.strip()
    m = re.fullmatch(r'v(\d+)\.(\d+)\.\d+', out)
    if m is None or (int(m.group(1)), int(m.group(2))) < (22, 18):
        return None
    return exe


def need_node(what: str) -> str | None:
    exe = node_exe()
    if exe is None:
        if os.environ.get('VITRO_REQUIRE_NODE') == '1':
            raise AssertionError(f'{what}: Node 22.18 or newer is required (VITRO_REQUIRE_NODE=1)')
        SKIPPED.append(what)
        print(f'test_web: SKIPPED {what}: Node 22.18 or newer was not found', flush=True)
    return exe


def scratch_dir() -> tempfile.TemporaryDirectory:
    """A temporary folder inside the project, not on the system drive."""
    base = os.path.join(ROOT, '_test_tmp')
    os.makedirs(base, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='web_', dir=base)


def node_run(exe: str, *args: str, cwd: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([exe, *args], capture_output=True, text=True, encoding='utf-8', cwd=cwd, timeout=timeout)


# ---- the built app ----


def check_dist_files() -> None:
    index = os.path.join(DIST, 'index.html')
    assert os.path.isfile(index), f'{index} is missing: run `npm run build` in web/ and commit web/dist'
    with open(index, encoding='utf-8') as fh:
        html = fh.read()
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, 'index.html refers to no script or stylesheet'
    for ref in refs:
        assert ref.startswith('./'), f'index.html: {ref!r} is not a relative URL, so the app would not work from a sub-path'
        assert os.path.isfile(os.path.join(DIST, ref[2:])), f'index.html refers to {ref}, which is not in web/dist'
    # the server's Content-Security-Policy allows only its own origin: anything fetched from elsewhere would be blocked
    for base, _dirs, files in os.walk(DIST):
        for fn in files:
            if os.path.splitext(fn)[1] in ('.html', '.css'):
                with open(os.path.join(base, fn), encoding='utf-8') as fh:
                    text = fh.read()
                external = re.findall(r'https?://[^\s"\')]+', text)
                assert not external, f'{fn} refers to other sites: {external[:3]}'
    served = {os.path.splitext(fn)[1].lower() for _b, _d, files in os.walk(DIST) for fn in files}
    unserved = served - set(webserver.CONTENT_TYPES)
    assert not unserved, f'web/dist has files of kinds the server refuses to serve: {sorted(unserved)}'
    print(f'test_web: web/dist has {sum(len(f) for _b, _d, f in os.walk(DIST))} files, all relative, all servable', flush=True)


def tree_digest(root: str) -> dict[str, str]:
    """Relative path -> SHA-256, with line endings of text files made equal (a checkout may convert them)."""
    out = {}
    for base, _dirs, files in os.walk(root):
        for fn in files:
            path = os.path.join(base, fn)
            with open(path, 'rb') as fh:
                data = fh.read()
            if os.path.splitext(fn)[1] in ('.html', '.js', '.css', '.svg'):
                data = data.replace(b'\r\n', b'\n')
            out[os.path.relpath(path, root).replace('\\', '/')] = hashlib.sha256(data).hexdigest()
    return out


def check_dist_is_current(node: str) -> None:
    vite = os.path.join(WEB, 'node_modules', 'vite', 'bin', 'vite.js')
    if not os.path.isfile(vite):
        SKIPPED.append('web/dist is what the sources build to')
        print('test_web: SKIPPED the check that web/dist is current: run `npm install` in web/ to enable it', flush=True)
        return
    with scratch_dir() as tmp:
        out = os.path.join(tmp, 'dist')
        r = node_run(node, vite, 'build', '--outDir', out, '--emptyOutDir', cwd=WEB)
        assert r.returncode == 0, f'vite build failed:\n{r.stdout}\n{r.stderr}'
        built, committed = tree_digest(out), tree_digest(DIST)
        assert built == committed, (
            'web/dist is not what the sources build to: run `npm run build` in web/ and commit web/dist.\n'
            f'  only built: {sorted(set(built) - set(committed))}\n  only committed: {sorted(set(committed) - set(built))}\n'
            f'  different: {sorted(k for k in set(built) & set(committed) if built[k] != committed[k])}'
        )
    print(f'test_web: web/dist is exactly what the sources build to ({len(committed)} files)', flush=True)


def check_vitest(node: str) -> None:
    vitest = os.path.join(WEB, 'node_modules', 'vitest', 'vitest.mjs')
    if not os.path.isfile(vitest):
        SKIPPED.append('the TypeScript tests (vitest)')
        print('test_web: SKIPPED the TypeScript tests: run `npm install` in web/ to enable them', flush=True)
        return
    r = node_run(node, vitest, 'run', cwd=WEB)
    assert r.returncode == 0, f'vitest failed:\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}'
    m = re.search(r'Tests\s+(\d+) passed \((\d+)\)', r.stdout)
    assert m and m.group(1) == m.group(2), f'vitest did not report all tests passed:\n{r.stdout[-1500:]}'
    print(f'test_web: vitest: {m.group(1)} TypeScript tests passed', flush=True)


# ---- the browser's reader against NumPy ----


def check_digests(node: str) -> None:
    """Both readers, written independently, give the same numbers from every sample trace."""
    script = os.path.join(WEB, 'tests', 'digest.ts')
    cases = [
        ('t5_small.vtdump', lambda p: test_viewer.python_digest(p)),
        ('qwen3_0_6b.vtdump', lambda p: test_viewer.python_digest(p)),
        ('t5_small.vtdemo', lambda p: test_demo.check_tape(p, True)),
        ('qwen3_0_6b.vtdemo', lambda p: test_demo.check_tape(p, False)),
    ]
    for name, python_side in cases:
        path = os.path.join(SAMPLES, name)
        r = node_run(node, script, path, timeout=120)
        assert r.returncode == 0, f'{name}: digest.ts failed:\n{r.stderr}'
        test_viewer.same(json.loads(r.stdout), python_side(path), name)
    print(f'test_web: browser reader == NumPy to 1e-9 on {len(cases)} sample traces', flush=True)


def check_refusal_parity(node: str) -> None:
    """A broken file is refused with the same message by the Python reader and by the browser's."""
    script = os.path.join(WEB, 'tests', 'refuse.ts')
    good_header = json.dumps({'version': 1}).encode()
    good = b'VITRO01\n' + len(good_header).to_bytes(8, 'little') + good_header + b'\0' * (-len(good_header) % 16)

    def blob(magic: bytes = b'VITRO01\n', length: int | None = None, text: bytes = good_header) -> bytes:
        n = len(text) if length is None else length
        return magic + n.to_bytes(8, 'little') + text + b'\0' * (-len(text) % 16)

    broken = {
        'empty.vtdump': b'',
        'short.vtdump': good[:10],
        'magic.vtdump': blob(magic=b'NOTVITRO'),
        'length.vtdump': blob(length=99999),
        'zero_length.vtdump': blob(length=0),
        'version.vtdump': blob(text=b'{"version":2}'),
        'json.vtdump': blob(text=b'{"x":'),
        'not_object.vtdump': blob(text=b'[1,2]'),
        'utf8.vtdump': blob(text=b'{"v":"\xff\xfe"}'),
    }
    with scratch_dir() as tmp:
        for fn, data in {**broken, 'good.vtdump': good}.items():
            path = os.path.join(tmp, fn)
            with open(path, 'wb') as fh:
                fh.write(data)
            r = node_run(node, script, path, timeout=60)
            assert r.returncode == 0, f'{fn}: refuse.ts failed:\n{r.stderr}'
            try:
                ContainerFile(path)
                python_says = 'OK'
            except ValueError as e:
                python_says = f'REFUSED {e}'
            js_says = r.stdout.strip()
            if fn == 'utf8.vtdump':
                # the two readers name a bad byte differently (each in its runtime's words); both must refuse
                assert python_says.startswith('REFUSED ') and js_says.startswith('REFUSED '), f'{fn}: python {python_says!r}, browser {js_says!r}'
                assert 'not valid' in python_says and 'not valid' in js_says, f'{fn}: python {python_says!r}, browser {js_says!r}'
                continue
            if fn == 'json.vtdump':
                # same refusal, the parser's own explanation differs after "header is not valid JSON"
                assert python_says.split(' (')[0] == js_says.split(' (')[0], f'{fn}: python {python_says!r}, browser {js_says!r}'
                continue
            assert python_says == js_says, f'{fn}: python {python_says!r}, browser {js_says!r}'
    print(f'test_web: browser reader refuses {len(broken)} kinds of broken file in the words of the Python reader', flush=True)


def check_float16_parity(node: str) -> None:
    """All 65536 float16 bit patterns decode to the same float32 in NumPy and in the browser."""
    pattern = np.arange(65536, dtype=np.uint16)
    w = Writer()
    desc = w.add(pattern.view(np.float16), 'f2')
    values = pattern.view(np.float16).astype(np.float32)
    bits = values.view(np.uint32).copy()
    bits[np.isnan(values)] = 0x7FC00000  # the payload of a NaN means nothing
    want = f'{hashlib.sha256(bits.tobytes()).hexdigest()} 65536'
    with scratch_dir() as tmp:
        path = os.path.join(tmp, 'half.vtdump')
        save(path, {'h': desc}, w)
        r = node_run(node, os.path.join(WEB, 'tests', 'half.ts'), path, timeout=60)
    assert r.returncode == 0, f'half.ts failed:\n{r.stderr}'
    assert r.stdout.strip() == want, f'float16 decoders differ: browser {r.stdout.strip()}, NumPy {want}'
    print('test_web: all 65536 float16 patterns decode identically in the browser and in NumPy', flush=True)


# ---- the server ----


class Client:
    """A bare HTTP client that lets the test choose the Host header and the method."""

    def __init__(self, port: int) -> None:
        self.port = port

    def get(self, path: str, host: str | None = None, method: str = 'GET') -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        try:
            conn.request(method, path, headers={'Host': host or f'127.0.0.1:{self.port}'})
            res = conn.getresponse()
            return res.status, {k.lower(): v for k, v in res.getheaders()}, res.read()
        finally:
            conn.close()


def check_server() -> None:
    # refusals are logged to stderr by the server; the test expects dozens of them and checks them itself
    webserver._Handler.log_message = lambda self, *args, **kwargs: None  # type: ignore[method-assign]
    entries = webserver.collect_entries([], SAMPLES)
    assert [e.path.name for e in entries] == sorted(e.path.name for e in entries) and len(entries) == 4, 'the four samples, in name order'
    srv = webserver.VitroServer(entries, 0, web_dir=DIST)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = Client(srv.server_address[1])

        status, headers, body = c.get('/')
        assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8' and b'<div id="root">' in body
        assert headers['content-security-policy'] == webserver.CSP and "default-src 'none'" in webserver.CSP
        assert headers['x-content-type-options'] == 'nosniff' and headers['cache-control'] == 'no-store'
        assert headers['referrer-policy'] == 'no-referrer'

        # the library lists every sample with its header facts
        status, headers, body = c.get('/api/library')
        lib = json.loads(body)['files']
        assert status == 200 and headers['content-type'].startswith('application/json') and len(lib) == 4
        by_name = {f['name']: f for f in lib}
        assert by_name['t5_small.vtdump']['kind'] == 'dump' and by_name['t5_small.vtdemo']['kind'] == 'demo'
        assert by_name['t5_small.vtdump']['traceId'] == by_name['t5_small.vtdemo']['traceId'] == 't5_small_translation'
        assert by_name['qwen3_0_6b.vtdump']['size'] == os.path.getsize(os.path.join(SAMPLES, 'qwen3_0_6b.vtdump'))

        # a registered file comes back byte for byte, by number
        for f in lib:
            status, headers, body = c.get(f"/files/{f['id']}")
            with open(os.path.join(SAMPLES, f['name']), 'rb') as fh:
                assert status == 200 and body == fh.read(), f['name']
            assert headers['content-type'] == 'application/octet-stream'

        # everything else under /files is refused
        for path in ('/files/4', '/files/999', '/files/-1', '/files/1.0', '/files/x', '/files/', '/files/0/extra', '/files/%30%2e%2e', '/files/%D9%A3'):
            assert c.get(path)[0] == 404, path

        # the app's own assets are served with explicit types
        refs = re.findall(r'(?:src|href)="\./([^"]+)"', body_of(c, '/'))
        assert refs
        types = {'.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.woff2': 'font/woff2'}
        for ref in refs:
            status, headers, _ = c.get('/' + ref)
            assert status == 200 and headers['content-type'] == types[os.path.splitext(ref)[1]], ref
        fonts = [fn for fn in sorted(os.listdir(os.path.join(DIST, 'assets'))) if fn.endswith('.woff2')]
        assert fonts and c.get('/assets/' + fonts[0])[1]['content-type'] == 'font/woff2'

        # nothing outside web/dist, however the path is written
        for path in ('/../README.md', '/%2e%2e/README.md', '/assets/../../README.md', '/..%2fREADME.md', '/assets/..%2f..%2fREADME.md',
                     '/index.html/..', '/assets//x', '/assets\\x.js', '/C:/Windows/win.ini', '/%00', '/assets/%00.js',
                     '/pyproject.toml', '/package.json', '/node_modules/x.js', '/tests/digest.ts', '/index.ts', '/assets/missing.js'):
            assert c.get(path)[0] == 404, path

        # only GET and HEAD
        for method in ('POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'):
            status, headers, _ = c.get('/', method=method)
            assert status == 405 and headers['allow'] == 'GET, HEAD', method
        status, headers, body = c.get('/', method='HEAD')
        assert status == 200 and body == b'' and int(headers['content-length']) == len(body_of(c, '/').encode('utf-8'))

        # DNS rebinding: a page on another site reaches the loopback address with its own name in Host
        for host in ('evil.example', f'evil.example:{c.port}', f'127.0.0.1:{c.port + 1}', '127.0.0.1', '', 'localhost:1'):
            status, _, body = c.get('/api/library', host=host or ' ')
            assert status == 403, host
        assert c.get('/api/library', host=f'localhost:{c.port}')[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()
    print('test_web: the server serves the app and the registered files only, and refuses the rest', flush=True)


def body_of(c: Client, path: str) -> str:
    return c.get(path)[2].decode('utf-8')


def check_server_files() -> None:
    """Explicit files are checked at start-up; a sample that cannot be read is listed with its error, not dropped."""
    with scratch_dir() as tmp:
        bad = os.path.join(tmp, 'broken.vtdump')
        with open(bad, 'wb') as fh:
            fh.write(b'not a trace file at all, but long enough to have a header')
        try:
            webserver.collect_entries([bad], None)
            raise AssertionError('a file named on the command line that is not a trace must be refused')
        except webserver.TraceFileError as e:
            assert 'not a Vitro trace file' in str(e) and 'broken.vtdump' in str(e), str(e)
        try:
            webserver.collect_entries([os.path.join(tmp, 'missing.vtdump')], None)
            raise AssertionError('a missing file must be refused')
        except webserver.TraceFileError as e:
            assert 'no such file' in str(e), str(e)

        samples = os.path.join(tmp, 'samples')
        os.makedirs(samples)
        shutil.copy(os.path.join(SAMPLES, 't5_small.vtdump'), samples)
        shutil.copy(bad, os.path.join(samples, 'zz_broken.vtdump'))
        with open(os.path.join(samples, 'readme.txt'), 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('not a trace')
        entries = webserver.collect_entries([os.path.join(SAMPLES, 'qwen3_0_6b.vtdemo')], samples)
        lib = webserver.library_json(entries)['files']
        assert [(f['name'], f['source']) for f in lib] == [('qwen3_0_6b.vtdemo', 'argument'), ('t5_small.vtdump', 'sample'), ('zz_broken.vtdump', 'sample')]
        assert 'error' not in lib[0] and 'error' not in lib[1] and 'not a Vitro trace file' in lib[2]['error']
        # the same file given twice, or also present among the samples, is registered once
        twice = webserver.collect_entries([os.path.join(SAMPLES, 't5_small.vtdump')] * 2, SAMPLES)
        assert len(twice) == 4 and twice[0].source == 'argument'
    print('test_web: files given by name are checked at start-up; unreadable samples are listed with their error', flush=True)


def check_cli_and_imports() -> None:
    env = {**os.environ, 'PYTHONPATH': PY_ROOT, 'PYTHONIOENCODING': 'utf-8'}
    r = subprocess.run([sys.executable, '-m', 'vitro.cli', 'web', 'missing.vtdump', '--no-browser'], capture_output=True, text=True, encoding='utf-8', env=env, timeout=60)
    assert r.returncode == 2 and 'no such file' in r.stderr, (r.returncode, r.stderr)
    r = subprocess.run([sys.executable, '-m', 'vitro.cli', 'web', '--bogus'], capture_output=True, text=True, encoding='utf-8', env=env, timeout=60)
    assert r.returncode == 2 and 'unrecognized arguments' in r.stderr, (r.returncode, r.stderr)
    # a port that is taken is a message, not a traceback
    holder = webserver.VitroServer([], 0, web_dir=DIST)
    try:
        r = subprocess.run([sys.executable, '-m', 'vitro.cli', 'web', '--no-browser', '--port', str(holder.server_address[1])], capture_output=True, text=True, encoding='utf-8', env=env, timeout=60)
    finally:
        holder.server_close()
    assert r.returncode == 2 and 'cannot listen' in r.stderr and 'Traceback' not in r.stderr, (r.returncode, r.stderr)

    code = (
        'import sys; import vitro; '
        'assert "torch" not in sys.modules, "import vitro pulled in torch"; '
        'assert "vitro.exporter" not in sys.modules and "vitro.adapters" not in sys.modules; '
        'assert callable(vitro.export_dump) and "torch" in sys.modules; '
        'assert "export_dump" in dir(vitro) and "ContainerFile" in dir(vitro); '
        'import vitro.cli, vitro.webserver; print("lazy ok")'
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
    assert r.returncode == 0 and 'lazy ok' in r.stdout, (r.stdout, r.stderr)
    try:
        subprocess.run([sys.executable, '-c', 'import vitro; vitro.no_such_name'], check=True, capture_output=True, env=env, timeout=60)
        raise AssertionError('an unknown name must raise AttributeError')
    except subprocess.CalledProcessError as e:
        assert b"has no attribute 'no_such_name'" in e.stderr, e.stderr
    print('test_web: vitro web refuses bad arguments with a message; `import vitro` leaves PyTorch alone', flush=True)


def run() -> None:
    SKIPPED.clear()
    check_dist_files()
    check_server()
    check_server_files()
    check_cli_and_imports()
    node = need_node('the browser reader against NumPy, the TypeScript tests and the check that web/dist is current')
    if node is not None:
        check_digests(node)
        check_refusal_parity(node)
        check_float16_parity(node)
        check_vitest(node)
        check_dist_is_current(node)
    if SKIPPED:
        print(f'test_web: passed with {len(SKIPPED)} part(s) SKIPPED: {"; ".join(SKIPPED)}', flush=True)
    else:
        print('test_web: every part ran', flush=True)


if __name__ == '__main__':
    run()
