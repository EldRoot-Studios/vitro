"""Isolation & open-source hygiene audit: ensures no proprietary identifiers leaked into Vitro."""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FORBIDDEN = [
    r'\btwomem\b',
    r'\bpoleweave\b',
    r'\bostov\b',
    r'\btmdump\b',
    r'\btmdemo\b',
    r'\bdeep128\b',
    r'\bnumin\b',
    r'\bnumcount\b',
    r'\bossuary\b',
    r'Mesh3D_Project',
    r'\b1017\b',
]
SCAN_EXTS = {'.py', '.cpp', '.hpp', '.h', '.sh', '.md', '.toml', '.txt'}


def run() -> None:
    self_path = os.path.abspath(__file__)
    pat = re.compile('|'.join(FORBIDDEN), re.IGNORECASE)
    hits: list[str] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git', 'bin', '_obj', '__pycache__', '_test_tmp')]
        for fn in files:
            p = os.path.join(base, fn)
            if os.path.abspath(p) == self_path:
                continue
            if os.path.splitext(fn)[1].lower() not in SCAN_EXTS:
                continue
            with open(p, 'r', encoding='utf-8', errors='replace') as fh:
                for ln, line in enumerate(fh, 1):
                    m = pat.search(line)
                    if m:
                        rel = os.path.relpath(p, ROOT)
                        hits.append(f'{rel}:{ln}: matched {m.group(0)!r}')
    assert not hits, 'Proprietary/internal leakage detected in Vitro:\n' + '\n'.join(hits[:15])
    print('test_leak_audit: 0 proprietary/internal symbols found in Vitro', flush=True)


if __name__ == '__main__':
    run()
