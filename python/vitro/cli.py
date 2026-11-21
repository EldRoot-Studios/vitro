"""Command-line interface for Vitro: inspect trace containers, launch viewer/player, and run selftests."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from .container import ContainerFile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_DIR = os.path.join(ROOT, 'bin')


def _find_exe(name: str) -> str:
    exe = os.path.join(BIN_DIR, name)
    if os.path.isfile(exe):
        return exe
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f'{name} not found in {BIN_DIR} — run "bash build.sh" first')


def cmd_info(args) -> int:
    f = ContainerFile(args.file)
    h = f.header
    tr = h.get('trace', {})
    print(f"file:   {args.file} ({h.get('kind')}, version {h.get('version')})")
    print(f"model:  {tr.get('model_name')} ({tr.get('arch')}), title: {tr.get('title')}")
    print(f"vocab:  {len(h.get('vocab', {}).get('labels', []))} tokens, spans: {len(h.get('spans', []))}")
    for u in h.get('units', []):
        stages = [s.get('label') for s in u.get('stages', [])]
        print(f"  unit {u.get('u')} ({u.get('label')}): {len(stages)} stages -> {stages}, bits={u.get('bits'):.2f}")
    for s in h.get('segments', []):
        stages = [st.get('label') for st in s.get('stages', [])]
        print(f"  segment {s.get('k')} ({s.get('label')}): {len(stages)} stages -> {stages}")
    return 0


def cmd_launch(exe_name: str, extra: list[str]) -> int:
    return subprocess.call([_find_exe(exe_name), *extra])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='vitro', description='Glass-box PyTorch trace inspector and 3D player')
    sub = ap.add_subparsers(dest='cmd', required=True)
    inf = sub.add_parser('info', help='Print summary of a .vtdump or .vtdemo file')
    inf.add_argument('file')
    sub.add_parser('view', help='Launch vitro_view.exe (forward extra args after --)')
    sub.add_parser('demo', help='Launch vitro_demo.exe (forward extra args after --)')
    sub.add_parser('selftest', help='Run built-in C++ selftest')
    a, rest = ap.parse_known_args(argv)
    if a.cmd == 'info':
        return cmd_info(a)
    if a.cmd == 'selftest':
        return cmd_launch('vitro_view.exe', ['--selftest'])
    if a.cmd == 'view':
        return cmd_launch('vitro_view.exe', rest)
    if a.cmd == 'demo':
        return cmd_launch('vitro_demo.exe', rest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
