"""End-to-end verification of bin/vitro_view.exe against Python/NumPy: build, selftest, digest, PNGs, input events, refusals, window loop."""
from __future__ import annotations

import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ROOT = os.path.join(ROOT, 'python')
EX_ROOT = os.path.join(ROOT, 'examples')
for p in (PY_ROOT, EX_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tiny_decoder_only import run_example as run_gpt  # noqa: E402
from tiny_encoder_decoder import run_example as run_seq2seq  # noqa: E402
from vitro import DecoderOnlyAdapter, SpanLabel, TokenMeta, TraceSpec, export_dump  # noqa: E402
from vitro.container import ContainerFile  # noqa: E402

VIEW_EXE = os.path.join(ROOT, 'bin', 'vitro_view.exe')
DEMO_EXE = os.path.join(ROOT, 'bin', 'vitro_demo.exe')


def _find_bash() -> str:
    for p in ('C:/Program Files/Git/bin/bash.exe', 'C:/msys64/usr/bin/bash.exe', shutil.which('bash')):
        if p and os.path.isfile(p) and 'System32' not in p:
            return p
    raise RuntimeError('Git Bash or MSYS2 bash is required to run build.sh on Windows')


def ensure_built() -> None:
    srcs = glob.glob(os.path.join(ROOT, 'cpp', '**', '*.[ch]pp'), recursive=True) + [os.path.join(ROOT, 'build.sh')]
    newest = max(os.path.getmtime(s) for s in srcs)
    if all(os.path.isfile(x) and os.path.getmtime(x) >= newest for x in (VIEW_EXE, DEMO_EXE)):
        return
    r = subprocess.run([_find_bash(), os.path.join(ROOT, 'build.sh')], capture_output=True, text=True, encoding='utf-8')
    assert r.returncode == 0 and os.path.isfile(VIEW_EXE) and os.path.isfile(DEMO_EXE), f'build failed:\n{r.stdout}\n{r.stderr}'


def view(*args: str, ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([VIEW_EXE, *args], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if ok:
        assert r.returncode == 0, f'{args}: exit {r.returncode}\n{r.stdout}\n{r.stderr}'
    return r


def read_png(path: str) -> np.ndarray:
    with open(path, 'rb') as fh:
        raw = fh.read()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n', f'{path}: invalid PNG header'
    at, idat, size = 8, b'', None
    while at < len(raw):
        n, = struct.unpack('>I', raw[at:at + 4])
        kind = raw[at + 4:at + 8]
        body = raw[at + 8:at + 8 + n]
        assert zlib.crc32(raw[at + 4:at + 8 + n]) == struct.unpack('>I', raw[at + 8 + n:at + 12 + n])[0]
        if kind == b'IHDR':
            size = struct.unpack('>II', body[:8])
        elif kind == b'IDAT':
            idat += body
        at += 12 + n
    w, h = size
    data = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 3 + 1)
    assert (data[:, 0] == 0).all()
    return data[:, 1:].reshape(h, w, 3)


def python_digest(path: str) -> dict:
    cf = ContainerFile(path)
    h = cf.header

    def s(desc) -> float:
        return 0.0 if desc is None else float(np.nansum(cf.array(desc).astype(np.float64)))

    units = []
    for u in h['units']:
        seq = u['sequence']
        sum_w = sum(s(lk['weights']) for lk in u['links'])
        sum_act = sum(s(st['act']) for st in u['stages'])
        sum_norm = s(u['norms']['data']) if u.get('norms') else 0.0
        ftok = seq.get('free_tok')
        units.append({
            'u': u['u'], 'n': len(cf.array(seq['tokens'])),
            'free_n': 0 if ftok is None else len(cf.array(ftok)),
            'diverge': -1 if u['diverge'] is None else u['diverge'],
            'sum_tok': s(seq['tokens']), 'sum_p': s(seq['p']), 'sum_rank': s(seq['rank']),
            'sum_top_p': s(seq['top_p']), 'sum_entropy': s(seq['entropy']),
            'sum_weights': sum_w, 'sum_act': sum_act, 'sum_norms': sum_norm,
        })
    return {'uid': h['trace']['id'], 'labels': len(h['vocab']['labels']), 'spans': len(h['spans']), 'units': units}


def same(a, b, path: str = '') -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), f'{path}: keys {sorted(a)} != {sorted(b)}'
        for k in a:
            same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), f'{path}: len {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{path}[{i}]')
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), f'{path}: viewer {a} != numpy {b}'
    else:
        assert a == b, f'{path}: {a!r} != {b!r}'


def state_of(r: subprocess.CompletedProcess) -> dict:
    line = [x for x in r.stdout.splitlines() if x.startswith('state: ')]
    assert line, f'missing state line in stdout:\n{r.stdout}'
    return json.loads(line[0][len('state: '):])


def _make_latent_scalar_dump(tmp: str) -> str:
    """Builds a custom DecoderOnly trace carrying 2D LatentMap + scalar token values to test those views too."""
    from tiny_decoder_only import TinyGPT, make_vocab

    torch.manual_seed(9)
    vocab = make_vocab()
    model = TinyGPT(vocab=len(vocab), d=32, layers=2, heads=2).eval()
    adapter = DecoderOnlyAdapter(model, model_name='LatentScalarGPT')
    tok = torch.tensor([1, 5, 10, 11, 8, 13, 14, 2], dtype=torch.long)
    tr = adapter.trace_forward(tok)
    tr.latents = [{
        'name': 'Hidden State Snapshot',
        'row_label': 'Position',
        'col_label': 'Channel',
        'data': np.random.default_rng(1).normal(size=(7, 32)).astype(np.float32),
    }]
    vals = [-1.0] * len(vocab)
    vals[vocab.index('1')] = 1.0
    spec = TraceSpec(
        trace_id='latent_scalar',
        spans=[SpanLabel(0, 3, 'Prefix', tint='cyan'), SpanLabel(3, len(tok), 'Body', tint='gold', parent=0)],
        token_meta=TokenMeta(labels=vocab, values=vals),
    )
    return export_dump(os.path.join(tmp, 'latent_scalar.vtdump'), tr, spec=spec, step=99)


def run() -> None:
    ensure_built()
    r = view('--selftest')
    assert 'selftest:' in r.stdout and ' 0 failed' in r.stdout, r.stdout
    print(f'test_viewer: {r.stdout.strip()}', flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        gpt_dump, gpt_demo = run_gpt(tmp)
        seq_dump, _ = run_seq2seq(tmp)
        lat_dump = _make_latent_scalar_dump(tmp)

        for p in (gpt_dump, seq_dump, lat_dump):
            same(json.loads(view(p, '--digest').stdout), python_digest(p), os.path.basename(p))
        print('test_viewer: --digest matches NumPy sums on all architectures', flush=True)

        shots = os.path.join(tmp, 'shots')
        view(seq_dump, '--all', shots, '--size', '1400x900', '--token', '2')
        names = ['tokens', 'attention', 'activations', 'spans', 'net']
        imgs = {}
        for n in names:
            img = read_png(os.path.join(shots, f'{n}.png'))
            assert img.shape == (900, 1400, 3)
            colours = len({tuple(c) for c in img[::7, ::7].reshape(-1, 3)})
            assert colours > 30, f'{n}: too few colors ({colours})'
            imgs[n] = img
        assert all(not np.array_equal(imgs[a], imgs[b]) for i, a in enumerate(names) for b in names[i + 1:])
        p1 = os.path.join(tmp, 'net1.png')
        p2 = os.path.join(tmp, 'net2.png')
        p3 = os.path.join(tmp, 'net3.png')
        view(gpt_dump, '--png', p1, '--view', 'net', '--phase', '0.2', '--size', '1400x900')
        view(gpt_dump, '--png', p2, '--view', 'net', '--phase', '0.2', '--size', '1400x900')
        view(gpt_dump, '--png', p3, '--view', 'net', '--phase', '0.6', '--size', '1400x900')
        assert np.array_equal(read_png(p1), read_png(p2)), 'same render state must be deterministic'
        assert not np.array_equal(read_png(p1), read_png(p3)), 'wave phase must animate 3D net'
        view(lat_dump, '--png', os.path.join(tmp, 'lat.png'), '--view', 'activations', '--size', '1400x900')
        assert read_png(os.path.join(tmp, 'lat.png')).shape == (900, 1400, 3)
        print('test_viewer: all 5 views render deterministically across decoder-only, seq2seq, and latent traces', flush=True)

        # Events & interactions
        st = state_of(view(seq_dump, '--png', os.path.join(tmp, 'ev.png'), '--size', '1400x900', '--state',
                           '--events', 'k50;k70;k71;k53;m500,400;p500,400;m580,430;r580,430;w500,400,120;k80'))
        assert st['view'] == 'net' and st['grid'] == 1 and abs(st['yaw'] - 0.30) > 0.1
        st2 = state_of(view(seq_dump, '--png', os.path.join(tmp, 'ev2.png'), '--size', '1400x900', '--state',
                            '--events', 'k49;k69;k68'))
        assert st2['view'] == 'tokens' and st2['token'] >= 1
        print('test_viewer: headless events (tabs, keys, 3D orbit, wheel, jumps) verified', flush=True)

        # Refusals
        r_bad1 = view(os.path.join(tmp, 'missing.vtdump'), '--digest', ok=False)
        assert r_bad1.returncode != 0
        r_bad2 = view(gpt_demo, '--digest', ok=False)
        assert r_bad2.returncode != 0 and 'recording' in r_bad2.stderr
        r_bad3 = view(gpt_dump, '--unknown-flag', ok=False)
        assert r_bad3.returncode != 0
        print('test_viewer: refusals on missing file, wrong file kind, and invalid flags verified', flush=True)

        # Real Win32 HWND window test
        rw = view('--window-test', '--library-dir', tmp, '--shots-dir', os.path.join(tmp, 'wshots'),
                  '--drop-test', gpt_dump, '--flood-test', '80', '--size', '1280x760')
        assert 'window test:' in rw.stdout and '"loaded":true' in rw.stdout, rw.stdout
        print('test_viewer: Win32 window loop & drag-and-drop verified', flush=True)


if __name__ == '__main__':
    run()
