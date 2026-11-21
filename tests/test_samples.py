"""Verifies pre-recorded real text transformer traces in vitro/samples/ end-to-end in C++ viewer/player & Python."""
from __future__ import annotations

import json
import os
import re
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES_DIR = os.path.join(ROOT, 'samples')

from test_demo import check_tape, demo  # noqa: E402
from test_leak_audit import FORBIDDEN  # noqa: E402
from test_viewer import ensure_built, python_digest, read_png, same, view  # noqa: E402
from vitro.container import ContainerFile  # noqa: E402


def run() -> None:
    ensure_built()
    pat = re.compile('|'.join(FORBIDDEN), re.IGNORECASE)
    dumps = sorted(os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.vtdump'))
    demos = sorted(os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.vtdemo'))
    assert dumps and demos, f'expected sample .vtdump and .vtdemo files in {SAMPLES_DIR}'

    with tempfile.TemporaryDirectory() as tmp:
        for dp in dumps:
            cf = ContainerFile(dp)
            hdr_text = json.dumps(cf.header, ensure_ascii=False)
            assert pat.search(hdr_text) is None, f'leak match inside {dp} header'
            got = json.loads(view(dp, '--digest').stdout)
            want = python_digest(dp)
            same(got, want, os.path.basename(dp))

            stem = os.path.splitext(os.path.basename(dp))[0]
            shots = os.path.join(tmp, f'{stem}_shots')
            view(dp, '--all', shots, '--size', '1400x900', '--token', '2')
            for vn in ('tokens', 'attention', 'activations', 'spans', 'net'):
                img = read_png(os.path.join(shots, f'{vn}.png'))
                assert img.shape == (900, 1400, 3) and len(np.unique(img.reshape(-1, 3), axis=0)) > 20

        for dm in demos:
            cf = ContainerFile(dm)
            hdr_text = json.dumps(cf.header, ensure_ascii=False)
            assert pat.search(hdr_text) is None, f'leak match inside {dm} header'
            has_cross = cf.header['trace'].get('arch') == 'encoder_decoder'
            want = check_tape(dm, expect_cross=has_cross)
            got = json.loads(demo(dm, '--digest').stdout)
            same(got['segments'], want['segments'], os.path.basename(dm))

            stem = os.path.splitext(os.path.basename(dm))[0]
            p0 = os.path.join(tmp, f'{stem}_a.png')
            p1 = os.path.join(tmp, f'{stem}_b.png')
            demo(dm, '--png', p0, '--token', '2', '--frac', '0.3', '--size', '1400x860')
            demo(dm, '--png', p1, '--token', '3', '--frac', '0.9', '--size', '1400x860')
            i0, i1 = read_png(p0), read_png(p1)
            assert i0.shape == (860, 1400, 3) and i1.shape == (860, 1400, 3) and not np.array_equal(i0, i1)

    print(f'test_samples: verified {len(dumps)} sample dumps & {len(demos)} sample demos cleanly', flush=True)


if __name__ == '__main__':
    run()
