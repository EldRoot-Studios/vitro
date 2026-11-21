"""End-to-end verification of bin/vitro_demo.exe: tape math, digest parity, frames, input events, refusals, window loop."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ROOT = os.path.join(ROOT, 'python')
EX_ROOT = os.path.join(ROOT, 'examples')
for p in (PY_ROOT, EX_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_viewer import ensure_built, read_png, same, state_of  # noqa: E402
from tiny_decoder_only import run_example as run_gpt  # noqa: E402
from tiny_encoder_decoder import run_example as run_seq2seq  # noqa: E402
from vitro.container import ContainerFile  # noqa: E402

DEMO_EXE = os.path.join(ROOT, 'bin', 'vitro_demo.exe')


def demo(*args: str, ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([DEMO_EXE, *args], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if ok:
        assert r.returncode == 0, f'{args}: exit {r.returncode}\n{r.stdout}\n{r.stderr}'
    return r


def check_tape(path: str, expect_cross: bool) -> dict:
    cf = ContainerFile(path)
    h = cf.header
    assert h['kind'] == 'demo'
    out = {'uid': h['trace']['id'], 'segments': []}
    for sg in h['segments']:
        ftok = cf.array(sg['free_tok'])
        fp = cf.array(sg['free_p'])
        tp = sg['tape']
        top_id = cf.array(tp['top_id'])
        top_p = cf.array(tp['top_p']).astype(np.float32)
        ent = cf.array(tp['entropy']).astype(np.float32)
        sid = cf.array(tp['self_id'])
        sw = cf.array(tp['self_w']).astype(np.float32)
        nm = cf.array(tp['norm']).astype(np.float32)
        t0, M = sg['t0'], len(ftok)
        assert M > t0
        assert (top_id[t0:, 0] == ftok[t0:]).all()
        assert (np.diff(top_p[t0:], axis=1) <= 1e-3).all() and (ent[t0:] >= -1e-3).all()
        assert np.isfinite(nm[t0:]).all()
        for t in range(t0, M):
            v = sid[t] >= 0
            assert (sid[t][v] <= t).all()
        if expect_cross:
            cid = cf.array(tp['cross_id'])
            cw = cf.array(tp['cross_w']).astype(np.float32)
            assert (cid[t0:] >= 0).any() and (cw[t0:] >= 0).all()
        def f64_sum(a: np.ndarray) -> float:
            b = a.astype(np.float64)
            return float(np.nansum(b))

        out['segments'].append({
            'k': sg['k'], 'm': M,
            'sum_ftok': f64_sum(ftok), 'sum_fp': f64_sum(fp),
            'sum_top_p': f64_sum(top_p), 'sum_self_w': f64_sum(sw),
            'sum_norm': f64_sum(nm),
        })
    return out


def run() -> None:
    ensure_built()
    with tempfile.TemporaryDirectory() as tmp:
        gpt_dump, gpt_demo = run_gpt(tmp)
        _, seq_demo = run_seq2seq(tmp)

        d_gpt = check_tape(gpt_demo, expect_cross=False)
        d_seq = check_tape(seq_demo, expect_cross=True)
        for p, want in ((gpt_demo, d_gpt), (seq_demo, d_seq)):
            got = json.loads(demo(p, '--digest').stdout)
            same(got['segments'], want['segments'], os.path.basename(p))
        print('test_demo: tape invariants and --digest parity verified for causal and seq2seq demos', flush=True)

        f1 = os.path.join(tmp, 'read.png')
        f2 = os.path.join(tmp, 'step1.png')
        f3 = os.path.join(tmp, 'step2.png')
        demo(seq_demo, '--png', f1, '--read', '0.4', '--size', '1400x860')
        demo(seq_demo, '--png', f2, '--token', '1', '--frac', '0.5', '--size', '1400x860')
        demo(seq_demo, '--png', f3, '--token', '3', '--frac', '0.7', '--size', '1400x860')
        assert read_png(f1).shape == (860, 1400, 3)
        assert not np.array_equal(read_png(f1), read_png(f2))
        assert not np.array_equal(read_png(f2), read_png(f3))
        print('test_demo: reading wave and token step PNG frames differ cleanly', flush=True)

        st = state_of(demo(gpt_demo, '--png', os.path.join(tmp, 'ev.png'), '--size', '1400x860', '--state',
                           '--events', 'k32;k39;k39;k38;k76;k71'))
        assert st['playing'] == 0 and st['loop'] == 0 and st['ghost'] == 0 and st['speed'] > 1.0
        print('test_demo: headless player events verified', flush=True)

        r_bad = demo(gpt_dump, '--digest', ok=False)
        assert r_bad.returncode != 0 and 'not a generation demo' in r_bad.stderr

        rw = demo('--window-test', '--library-dir', tmp, '--shots-dir', os.path.join(tmp, 'shots'),
                  '--drop-test', seq_demo, '--flood-test', '80', '--size', '1160x720')
        assert 'window test:' in rw.stdout and '"loaded":true' in rw.stdout, rw.stdout
        print('test_demo: Win32 player window loop & drag-and-drop verified', flush=True)


if __name__ == '__main__':
    run()
