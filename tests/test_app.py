"""Verification of bin/vitro.exe, the app in one window.

The program is the web app of web/ inside a window of the Microsoft Edge engine (WebView2), carrying the app, the
sample traces and the licence notices inside itself. This runs the real program, with its real window, and checks:
  * the command line: --version, --help, --licenses, bad options, files that are missing or are not traces (the
    message must be the one the Python reader gives for the same file);
  * what the page can fetch: the library of traces and the trace files, byte for byte, the app's own files, the
    refusals, the security headers;
  * the app in the window: open a run from the start screen, visit all six views without an error, play the
    replay, and see that a file that is not a trace is refused with its reason;
  * a picture of the window is made (--screenshot), and is not black;
  * files named on the command line are opened.

It opens a window for a few seconds per check, so it needs a Windows desktop session. The program is rebuilt first
when it is older than its sources (`bash build.sh app`; the first build downloads the WebView2 SDK).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ROOT = os.path.join(ROOT, 'python')
for p in (PY_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_viewer import _find_bash  # noqa: E402
from vitro.container import ContainerFile  # noqa: E402

EXE = os.path.join(ROOT, 'bin', 'vitro.exe')
SAMPLES = os.path.join(ROOT, 'samples')


def newest_source() -> float:
    patterns = [
        'cpp/src/app_*', 'cpp/include/vitro/*.hpp', 'cpp/res/*', 'web/dist/**/*', 'samples/*', 'THIRD_PARTY_NOTICES.txt',
        'tools/make_app_resources.py', 'tools/fetch_webview2.py', 'build.sh', 'pyproject.toml',
    ]
    files = [f for pat in patterns for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)) if os.path.isfile(f)]
    return max(os.path.getmtime(f) for f in files)


def ensure_built() -> None:
    if os.path.isfile(EXE) and os.path.getmtime(EXE) >= newest_source():
        return
    env = dict(os.environ)
    env['PYTHON'] = sys.executable  # the build runs the same Python as the tests
    r = subprocess.run([_find_bash(), os.path.join(ROOT, 'build.sh'), 'app'], capture_output=True, text=True, encoding='utf-8', env=env, timeout=900)
    assert r.returncode == 0 and os.path.isfile(EXE), f'build failed:\n{r.stdout}\n{r.stderr}'


def run_app(data_dir: str, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env['VITRO_DATA_DIR'] = data_dir  # the engine's profile goes to the scratch folder, not the user's
    return subprocess.run([EXE, *args], capture_output=True, text=True, encoding='utf-8', env=env, timeout=timeout)


def eval_page(data_dir: str, script: str, *args: str, timeout_ms: int = 90000) -> dict:
    """Runs `script` (the body of an async function) in the app's page and returns what it returned."""
    with tempfile.NamedTemporaryFile('w', suffix='.js', dir=data_dir, delete=False, encoding='utf-8', newline='\n') as fh:
        fh.write(script)
        path = fh.name
    try:
        r = run_app(data_dir, '--eval-file', path, '--timeout', str(timeout_ms), *args, timeout=timeout_ms // 1000 + 60)
    finally:
        os.remove(path)
    assert r.returncode == 0, f'the page script failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}'
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result['ok'], f'the page script threw: {result.get("error")}'
    return result['value']


def sha256_of(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---- the command line ----


def check_command_line(tmp: str) -> None:
    r = run_app(tmp, '--version')
    assert r.returncode == 0 and r.stdout.startswith('vitro 0.1.0 (WebView2 SDK 1.0.'), r.stdout
    r = run_app(tmp, '--help')
    assert r.returncode == 0 and 'usage: vitro.exe' in r.stdout and '--eval-file' in r.stdout, r.stdout
    r = run_app(tmp, '--licenses')
    assert r.returncode == 0 and 'Microsoft' in r.stdout and 'Geist' in r.stdout and 'React' in r.stdout, 'licence notices are missing'
    for args, want in (
        (['--bogus'], 'unknown option --bogus'),
        (['--size', '12'], '--size needs WIDTHxHEIGHT'),
        (['--wait', 'soon'], '--wait needs a positive whole number'),
        (['--screenshot'], '--screenshot needs a value'),
    ):
        r = run_app(tmp, *args)
        assert r.returncode == 2 and want in r.stderr, (args, r.returncode, r.stderr)

    # a file that is missing or is not a trace stops the program before a window opens, in the Python reader's words
    r = run_app(tmp, os.path.join(tmp, 'missing.vtdump'))
    assert r.returncode == 2 and 'missing.vtdump: no such file' in r.stderr, r.stderr
    magic = b'VITRO01\n'
    good_header = json.dumps({'version': 1}).encode()

    def blob(text: bytes, mag: bytes = magic, length: int | None = None) -> bytes:
        return mag + (len(text) if length is None else length).to_bytes(8, 'little') + text + b'\0' * (-len(text) % 16)

    broken = {
        'short.vtdump': b'tiny',
        'magic.vtdump': blob(b'{"x"', mag=b'NOTVITRO'),
        'length.vtdump': blob(good_header, length=99999),
        'version.vtdump': blob(b'{"version":2}'),
        'notjson.vtdump': blob(b'{"x":'),
        'notutf8.vtdump': blob(b'{"v":"\xff\xfe"}'),
        'notobject.vtdump': blob(b'[1,2]'),
    }
    for name, data in broken.items():
        path = os.path.join(tmp, name)
        with open(path, 'wb') as fh:
            fh.write(data)
        try:
            ContainerFile(path)
            raise AssertionError(f'{name}: the Python reader accepted it')
        except ValueError as e:
            want = str(e)
        r = run_app(tmp, path)
        got = r.stderr.strip()
        assert r.returncode == 2 and got.startswith('vitro: '), (name, r.returncode, got)
        got = got[len('vitro: '):]
        if name == 'notjson.vtdump':  # the parsers explain a bad JSON text in their own words after the same first part
            got, want = got.split(' (')[0], want.split(' (')[0]
        if name == 'notutf8.vtdump':
            got, want = got.split(' (')[0], want.split(' (')[0]
        assert got == want, f'{name}: program says {got!r}, Python says {want!r}'
    print(f'test_app: command line: version, help, licences, 4 bad options, 1 missing file and {len(broken)} broken files in the Python reader\'s words', flush=True)


# ---- the page ----

ROUTES = r"""
const out = { title: document.title, home: !!document.querySelector('.home') };
const lib = await (await fetch('api/library')).json();
out.files = lib.files;
out.hashes = {};
for (const f of lib.files) {
  const buf = await (await fetch('files/' + f.id)).arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buf);
  out.hashes[f.name] = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
const status = async (path, init) => (await fetch(path, init)).status;
out.status = {
  index: await status('/'),
  asset: await status(document.querySelector('script[src]').getAttribute('src')),
  missing: await status('assets/missing.js'),
  traversal: await status('..%2fREADME.md'),
  backslash: await status('assets%5Cx.js'),
  colon: await status('/C:/Windows/win.ini'),
  badId: await status('files/999'),
  badId2: await status('files/x'),
  post: await status('api/library', { method: 'POST' }),
  put: await status('files/0', { method: 'PUT' }),
};
const res = await fetch('/');
out.csp = res.headers.get('content-security-policy');
out.type = res.headers.get('content-type');
out.nosniff = res.headers.get('x-content-type-options');
const font = [...document.styleSheets].flatMap((s) => [...s.cssRules]).filter((r) => r instanceof CSSFontFaceRule).length;
out.fontRules = font;
out.fonts = [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family);
return out;
"""

FLOW = r"""
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const waitFor = async (fn, what, ms = 10000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { const v = fn(); if (v) return v; await sleep(50); }
  throw new Error('timed out waiting for ' + what);
};
const out = {};
const card = await waitFor(() => [...document.querySelectorAll('.run')].find((r) => r.innerText.includes('t5-small')), 'the T5 card on the start screen');
card.querySelector('button').click();
await waitFor(() => document.querySelector('.app'), 'the workspace');
await waitFor(() => document.title.includes('t5-small'), 'the title of the trace');
out.title = document.title;
out.tabs = [...document.querySelectorAll('[role=tab]')].map((t) => t.innerText);
out.views = {};
for (const tab of [...document.querySelectorAll('[role=tab]')]) {
  tab.click();
  await sleep(600);
  out.views[tab.innerText] = {
    canvases: document.querySelectorAll('canvas.scene').length,
    scroll: !!document.querySelector('.free-scroll'),
    errors: [...document.querySelectorAll('.toast[data-kind=err]')].map((t) => t.innerText),
    rail: !!document.querySelector('.pane.left'),
    inspector: !!document.querySelector('.pane.right'),
  };
}
// the replay plays: the playhead moves while the window is visible
const play = document.querySelector('.tbtn.play');
if (play.getAttribute('aria-label').startsWith('Play')) play.click(); // paused when the system asks for less motion
await sleep(200);
const head = document.querySelector('.tl .head');
const at0 = parseFloat(head.style.left);
await sleep(1500);
out.playheadMoved = parseFloat(head.style.left) !== at0; // it may have looped round, so not just greater
// a file that is not a trace is refused, with its reason
const input = document.querySelector('input[type=file]');
const dt = new DataTransfer();
dt.items.add(new File([new Uint8Array(40)], 'bad.vtdump'));
input.files = dt.files;
input.dispatchEvent(new Event('change', { bubbles: true }));
out.refusal = await waitFor(() => [...document.querySelectorAll('.toast')].map((t) => t.innerText).find((t) => t.startsWith('bad.vtdump')), 'the refusal of a bad file');
// the keyboard reaches the page
window.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true }));
await sleep(300);
out.tabAfterKey = document.querySelector('[role=tab][aria-selected=true]').innerText;
return out;
"""

OPENED = r"""
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (let i = 0; i < 100 && !(document.querySelector('.app') && document.title.includes('-')); i++) await sleep(50); // the title follows the state a moment later
return {
  title: document.title,
  workspace: !!document.querySelector('.app'),
  tab: document.querySelector('[role=tab][aria-selected=true]')?.innerText,
  hash: decodeURIComponent(location.hash),
  tokens: document.querySelectorAll('.cell').length,
  library: (await (await fetch('api/library')).json()).files.map((f) => f.name + ':' + f.source),
};
"""


def check_page(tmp: str) -> None:
    out = eval_page(tmp, ROUTES)
    assert out['title'] == 'Vitro' and out['home'], out
    names = sorted(f['name'] for f in out['files'])
    assert names == sorted(os.path.basename(p) for p in glob.glob(os.path.join(SAMPLES, '*.vt*'))), names
    by_name = {f['name']: f for f in out['files']}
    assert [f['id'] for f in out['files']] == [0, 1, 2, 3] and all(f['source'] == 'sample' for f in out['files'])
    assert by_name['t5_small.vtdump']['kind'] == 'dump' and by_name['t5_small.vtdemo']['kind'] == 'demo'
    assert by_name['t5_small.vtdump']['traceId'] == by_name['t5_small.vtdemo']['traceId'] == 't5_small_translation'
    assert by_name['qwen3_0_6b.vtdump']['model'] == 'Qwen/Qwen3-0.6B' and by_name['qwen3_0_6b.vtdump']['arch'] == 'decoder_only'
    for name, digest in out['hashes'].items():
        path = os.path.join(SAMPLES, name)
        assert digest == sha256_of(path), f'{name}: the page received other bytes than the file holds'
        assert by_name[name]['size'] == os.path.getsize(path), name
    st = out['status']
    assert (st['index'], st['asset']) == (200, 200), st
    for k in ('missing', 'traversal', 'backslash', 'colon', 'badId', 'badId2'):
        assert st[k] == 404, (k, st)
    assert st['post'] == 405 and st['put'] == 405, st
    assert out['csp'].startswith("default-src 'none'") and "frame-ancestors 'none'" in out['csp'], out['csp']
    assert out['type'] == 'text/html; charset=utf-8' and out['nosniff'] == 'nosniff', out
    assert {'Geist', 'Geist Mono'} <= set(out['fonts']), f'the fonts did not load: {out["fonts"]}'
    print(f'test_app: page: library of {len(names)} samples, {len(out["hashes"])} files byte for byte, refusals, headers, fonts', flush=True)

    flow = eval_page(tmp, FLOW)
    assert flow['tabs'] == ['Tokens', 'Attention', 'Activations', 'Spans', '3D Net', 'Replay'], flow['tabs']
    assert 'google-t5/t5-small' in flow['title'], flow['title']
    for tab, view in flow['views'].items():
        assert view['errors'] == [], (tab, view['errors'])
        assert view['rail'] and view['inspector'] == (tab != 'Spans'), (tab, view)
        assert view['scroll'] == (tab in ('Tokens', 'Spans')) and view['canvases'] == (0 if tab in ('Tokens', 'Spans') else 1), (tab, view)
    assert flow['playheadMoved'], 'the replay did not play'
    assert flow['refusal'] == 'bad.vtdump: not a Vitro trace file', flow['refusal']
    assert flow['tabAfterKey'] == 'Tokens', flow['tabAfterKey']
    print('test_app: the app in the window: a run opened, six views without an error, the replay played, a bad file refused', flush=True)


def png_size(path: str) -> tuple[int, int]:
    with open(path, 'rb') as fh:
        head = fh.read(33)
    assert head[:8] == b'\x89PNG\r\n\x1a\n' and head[12:16] == b'IHDR', f'{path} is not a PNG file'
    return struct.unpack('>II', head[16:24])


def check_screenshot(tmp: str) -> None:
    path = os.path.join(tmp, 'window.png')
    r = run_app(tmp, '--screenshot', path, '--size', '1000x640', '--hash', '#open=3,2&view=attention&token=7&fam=2', '--wait', '800')
    assert r.returncode == 0 and os.path.isfile(path), (r.returncode, r.stderr)
    w, h = png_size(path)
    assert w >= 1000 and h >= 640, (w, h)
    size = os.path.getsize(path)
    assert size > 20_000, f'the picture is only {size} bytes: the window was blank (a real one is about 55 KB, a black one 3 KB)'
    try:
        from PIL import Image, ImageStat
    except ImportError:
        print('test_app: the pixels of the picture were not examined (Pillow is not installed); its size was checked', flush=True)
    else:
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            brightest = max(rgb.getextrema(), key=lambda lo_hi: lo_hi[1])[1]
            pixels = rgb.resize((100, 64)).tobytes()
            share = sum(1 for i in range(0, len(pixels), 3) if max(pixels[i:i + 3]) > 60) / 6400
        assert brightest > 200 and share > 0.02, f'the picture is almost black (brightest {brightest}, lit share {share:.3f})'
        print(f'test_app: screenshot {w}x{h}, {size // 1024} KB, {share:.0%} of it lit', flush=True)
        return
    print(f'test_app: screenshot {w}x{h}, {size // 1024} KB', flush=True)


def check_files_on_command_line(tmp: str) -> None:
    dump, demo = (os.path.join(SAMPLES, n) for n in ('t5_small.vtdump', 't5_small.vtdemo'))
    out = eval_page(tmp, OPENED, dump, demo)
    assert out['workspace'] and out['tab'] == 'Tokens' and out['hash'].startswith('#open=0,1'), out
    assert 'google-t5/t5-small' in out['title'] and out['tokens'] == 15, out
    assert out['library'][:2] == ['t5_small.vtdump:argument', 't5_small.vtdemo:argument'] and len(out['library']) == 6, out['library']
    # the same file twice is one file
    out = eval_page(tmp, OPENED, dump, dump)
    assert out['library'].count('t5_small.vtdump:argument') == 1 and len(out['library']) == 5, out['library']
    print('test_app: files named on the command line are opened, listed first, and listed once', flush=True)


def remove_when_free(path: str) -> None:
    """Deletes the scratch folder. The engine's helper processes keep files of its profile open for a moment after
    the program has exited, so deleting is retried for a while; a folder that stays is reported, not hidden."""
    for _ in range(40):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.5)
    print(f'test_app: could not remove {path}: the engine still holds files in it', flush=True)


def run() -> None:
    ensure_built()
    base = os.path.join(ROOT, '_test_tmp')
    os.makedirs(base, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='app_', dir=base)
    try:
        check_command_line(tmp)
        check_page(tmp)
        check_screenshot(tmp)
        check_files_on_command_line(tmp)
    finally:
        remove_when_free(tmp)
    print('test_app: every part ran', flush=True)


if __name__ == '__main__':
    run()
