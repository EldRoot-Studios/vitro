"""Aligned binary trace container (`VITRO01`) for headers and raw C-order tensors.

Layout (little-endian):
    bytes 0..7    8-byte format magic (default ``b'VITRO01\\n'``)
    bytes 8..15   unsigned 64-bit JSON header length
    bytes 16..    UTF-8 JSON header (zero-padded to 16-byte alignment)
    data          raw tensors, each starting on a 16-byte boundary

Shared across Vitro open tools and downstream plugins via configurable ``magic``.
"""
from __future__ import annotations

import json
import os

import numpy as np

MAGIC = b'VITRO01\n'
ALIGN = 16
VERSION = 1
DTYPES = {
    'f4': np.dtype('<f4'),
    'f2': np.dtype('<f2'),
    'i4': np.dtype('<i4'),
    'i2': np.dtype('<i2'),
    'i1': np.dtype('i1'),
    'u1': np.dtype('u1'),
}


def _pad(n: int) -> int:
    return -n % ALIGN


def is_descriptor(x) -> bool:
    return isinstance(x, dict) and set(x) == {'dtype', 'off', 'shape'}


class Writer:
    """Packs numpy arrays sequentially at 16-byte offsets and emits JSON descriptors."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self.size = 0

    def add(self, array, dtype: str) -> dict:
        if dtype not in DTYPES:
            raise ValueError(f'dtype {dtype!r}: must be one of {sorted(DTYPES)}')
        a = np.asarray(array)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        want = DTYPES[dtype]
        if np.issubdtype(want, np.integer):
            if a.dtype.kind not in 'iub':
                raise ValueError(f'array of {a.dtype} cannot be stored as {dtype} (integers only)')
            info = np.iinfo(want)
            if a.size and (int(a.min()) < info.min or int(a.max()) > info.max):
                raise ValueError(f'values {int(a.min())} .. {int(a.max())} do not fit in {dtype}')
        elif a.dtype.kind not in 'fiub':
            raise ValueError(f'array of {a.dtype} cannot be stored as {dtype}')
        data = np.ascontiguousarray(a.astype(want, copy=False))
        off = self.size
        raw = data.tobytes()
        self._parts.append(raw + b'\0' * _pad(len(raw)))
        self.size += len(raw) + _pad(len(raw))
        return {'dtype': dtype, 'off': off, 'shape': [int(s) for s in data.shape]}

    def blob(self) -> bytes:
        return b''.join(self._parts)


def save(path: str, header: dict, writer: Writer, magic: bytes = MAGIC) -> None:
    """Atomically saves JSON header + binary tensor section via a flushed temp file."""
    if len(magic) != 8:
        raise ValueError(f'magic must be 8 bytes, got {len(magic)}')
    head = dict(header)
    head['version'] = VERSION
    try:
        text = json.dumps(head, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except ValueError as e:
        raise ValueError(f'header contains NaN, Infinity, or unserialisable object: {e}') from e
    raw = text.encode('utf-8')
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.part'
    try:
        with open(tmp, 'wb') as fh:
            fh.write(magic)
            fh.write(len(raw).to_bytes(8, 'little'))
            fh.write(raw + b'\0' * _pad(len(raw)))
            fh.write(writer.blob())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


class ContainerFile:
    """Validates and maps a trace container file, offering zero-copy read-only numpy views."""

    def __init__(self, path: str, magic: bytes = MAGIC, wrong_magic_msg: str = 'not a Vitro trace file') -> None:
        self.path = path
        with open(path, 'rb') as fh:
            self._buf = memoryview(fh.read())
        if len(self._buf) < 16:
            raise ValueError(f'{path}: truncated file (less than 16 bytes)')
        if bytes(self._buf[:8]) != magic:
            raise ValueError(f'{path}: {wrong_magic_msg}')
        hl = int.from_bytes(self._buf[8:16], 'little')
        if hl == 0 or 16 + hl > len(self._buf):
            raise ValueError(f'{path}: header says {hl} bytes, file has {len(self._buf) - 16}')
        self.header = json.loads(bytes(self._buf[16:16 + hl]).decode('utf-8'))
        if self.header.get('version') != VERSION:
            raise ValueError(f'{path}: container version {self.header.get("version")}, expected {VERSION}')
        self._data_at = 16 + hl + _pad(hl)
        self.data_size = len(self._buf) - self._data_at
        if self.data_size < 0:
            raise ValueError(f'{path}: padding runs past end of file')

    def array(self, desc: dict) -> np.ndarray:
        if not is_descriptor(desc):
            raise ValueError(f'{desc!r} is not an array descriptor')
        dt_name, off, shape = desc['dtype'], desc['off'], desc['shape']
        if dt_name not in DTYPES:
            raise ValueError(f'unknown dtype {dt_name!r}')
        dt = DTYPES[dt_name]
        if not isinstance(off, int) or off < 0 or off % ALIGN:
            raise ValueError(f'offset {off} lies outside the data section (must be >= 0 and {ALIGN}-aligned)')
        if not isinstance(shape, list) or any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid shape {shape!r}')
        count = int(np.prod(shape, dtype=np.int64))
        nbytes = count * dt.itemsize
        if off + nbytes > self.data_size:
            raise ValueError(f'array {dt_name}{shape} at {off} lies outside the data section ({self.data_size} bytes)')
        a = np.frombuffer(self._buf, dtype=dt, count=count, offset=self._data_at + off).reshape(shape)
        a.setflags(write=False)
        return a
