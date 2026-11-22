"""Command-line interface for Vitro: inspect trace containers, launch viewer/player, and run selftests."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from .container import ContainerFile
from .webserver import DEFAULT_PORT, serve

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


def cmd_app(files: list[str]) -> int:
    """Opens the app (bin/vitro.exe), with the given trace files, and waits until its window is closed."""
    return subprocess.call([_find_exe('vitro.exe'), *files])


def cmd_web(args) -> int:
    return serve(args.files, port=args.port, open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='vitro', description='Glass-box PyTorch trace inspector and 3D player')
    sub = ap.add_subparsers(dest='cmd', required=True)
    inf = sub.add_parser('info', help='Print summary of a .vtdump or .vtdemo file')
    inf.add_argument('file')
    app = sub.add_parser('app', help='Open the Vitro app (bin/vitro.exe): inspector and generation replay in one window')
    app.add_argument('files', nargs='*', help='.vtdump and/or .vtdemo files to open (the samples are built in)')
    web = sub.add_parser('web', help='Serve the same app to your browser, for systems without vitro.exe (Linux, macOS)')
    web.add_argument('files', nargs='*', help='.vtdump and/or .vtdemo files to open (the samples are always listed)')
    web.add_argument('--port', type=int, default=DEFAULT_PORT, help='port on 127.0.0.1 (default: %(default)s)')
    web.add_argument('--no-browser', action='store_true', help='only print the address, do not open a browser')
    sub.add_parser('view', help='Launch the legacy vitro_view.exe (forward extra args after --)')
    sub.add_parser('demo', help='Launch the legacy vitro_demo.exe (forward extra args after --)')
    sub.add_parser('selftest', help='Run built-in C++ selftest')
    a, rest = ap.parse_known_args(argv)
    if a.cmd in ('web', 'app') and rest:
        ap.error(f'unrecognized arguments: {" ".join(rest)}')
    if a.cmd == 'app':
        return cmd_app(a.files)
    if a.cmd == 'web':
        return cmd_web(a)
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
