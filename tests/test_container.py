"""Checks of vitro.container: binary roundtrip, 16-byte alignment, custom magic, and strict refusals."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.join(os.path.dirname(HERE), 'python')
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from vitro.container import ALIGN, DTYPES, MAGIC, VERSION, ContainerFile, Writer, save  # noqa: E402


def _raises(exc, fn) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    raise AssertionError(f'expected {exc.__name__} to be raised')


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        w = Writer()
        rng = np.random.default_rng(11)
        f4 = rng.normal(size=(3, 5)).astype(np.float32)
        f4[1, 2] = np.nan
        f2 = rng.uniform(0, 1, size=(2, 3, 4)).astype(np.float32)
        i4 = np.arange(-5, 7, dtype=np.int32)
        i2 = np.asarray([[-3, 4], [32767, -32768]])
        u1 = np.asarray([True, False, True])
        empty = np.zeros((0, 4), np.float32)
        desc = {
            'f4': w.add(f4, 'f4'), 'f2': w.add(f2, 'f2'), 'i4': w.add(i4, 'i4'),
            'i2': w.add(i2, 'i2'), 'u1': w.add(u1, 'u1'), 'empty': w.add(empty, 'f4'),
            'text': 'Label ✓ Токен',
        }
        p = os.path.join(tmp, 'test.vtdump')
        save(p, {'desc': desc, 'kind': 'dump'}, w)
        cf = ContainerFile(p)
        assert cf.header['version'] == VERSION and cf.header['desc']['text'] == 'Label ✓ Токен'
        for k, want in (('f4', f4), ('i4', i4), ('i2', i2)):
            assert np.array_equal(cf.array(cf.header['desc'][k]), want, equal_nan=True), k
        assert np.array_equal(cf.array(cf.header['desc']['u1']), u1.astype(np.uint8))
        assert np.array_equal(cf.array(cf.header['desc']['f2']), f2.astype(np.float16))
        assert cf.array(cf.header['desc']['empty']).shape == (0, 4)
        offs = [v['off'] for v in cf.header['desc'].values() if isinstance(v, dict)]
        assert all(o % ALIGN == 0 for o in offs), offs

        # Custom magic plugin compatibility
        custom_magic = b'PLUGN1\n\x00'
        cp = os.path.join(tmp, 'custom.bin')
        save(cp, {'ok': True}, Writer(), magic=custom_magic)
        assert ContainerFile(cp, magic=custom_magic).header['ok'] is True
        assert 'not a Vitro trace' in _raises(ValueError, lambda: ContainerFile(cp, magic=MAGIC))

        # Refusals
        w2 = Writer()
        assert 'do not fit' in _raises(ValueError, lambda: w2.add([40000], 'i2'))
        assert 'integers only' in _raises(ValueError, lambda: w2.add([1.5], 'i4'))
        assert 'dtype' in _raises(ValueError, lambda: w2.add([1], 'f8'))
        bad = os.path.join(tmp, 'bad.vtdump')
        assert 'NaN' in _raises(ValueError, lambda: save(bad, {'x': float('nan')}, Writer()))
        assert not os.path.exists(bad)

        with open(p, 'rb') as fh:
            raw = fh.read()
        cut = os.path.join(tmp, 'cut.vtdump')
        with open(cut, 'wb') as fh:
            fh.write(raw[:20])
        assert 'header says' in _raises(ValueError, lambda: ContainerFile(cut))
        assert 'outside the data section' in _raises(ValueError, lambda: cf.array({'dtype': 'f4', 'off': 16, 'shape': [99999]}))
        assert 'not an array descriptor' in _raises(ValueError, lambda: cf.array({'dtype': 'f4'}))
        n = int.from_bytes(raw[8:16], 'little')
        h = json.loads(raw[16:16 + n].decode('utf-8'))
        h['version'] = 99
        raw_h = json.dumps(h, separators=(',', ':')).encode('utf-8')
        bad_v = os.path.join(tmp, 'v99.vtdump')
        with open(bad_v, 'wb') as fh:
            fh.write(raw[:8] + len(raw_h).to_bytes(8, 'little') + raw_h + b'\0' * (-len(raw_h) % 16) + raw[16 + n + (-n % 16):])
        assert 'version 99' in _raises(ValueError, lambda: ContainerFile(bad_v))

        # a header that is not JSON, not UTF-8 or not an object is refused with the file named (it used to surface as
        # a bare parser message, or as an AttributeError for a list)
        for name, text, want in (
            ('notjson.vtdump', b'{"x":', 'header is not valid JSON'),
            ('notutf8.vtdump', b'{"v":"\xff\xfe"}', 'header is not valid UTF-8'),
            ('notobject.vtdump', b'[1,2]', 'header must be a JSON object'),
        ):
            odd = os.path.join(tmp, name)
            with open(odd, 'wb') as fh:
                fh.write(MAGIC + len(text).to_bytes(8, 'little') + text + b'\0' * (-len(text) % 16))
            msg = _raises(ValueError, lambda: ContainerFile(odd))
            assert msg.startswith(f'{odd}: ') and want in msg, msg

    print('test_container: all roundtrip, alignment, and refusal checks passed', flush=True)


if __name__ == '__main__':
    run()
