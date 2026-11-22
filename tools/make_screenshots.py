"""Takes the pictures of the README with the real program: bin/vitro.exe, in its own window, one picture per entry of PICTURES.

Each picture is a file name and the address fragment (the same as `vitro.exe --hash`) that puts the app in the state to
be shown. The sample traces are named by their number in the program's library: 0 and 1 are the recording and the dump
of Qwen3-0.6B, 2 and 3 those of T5-small. A position in the replay (`pos`) is counted in ticks: for T5 the context is
read for 8 ticks and then one tick goes to each generated token, for Qwen3 each token takes one tick from the start.
A replay that is left playing loops, so a still frame needs `playing=0` as well as `pos`.

`social` is the picture of link previews: twice as wide as it is high, which is what social networks crop to and the
size GitHub asks for a repository's social preview. tools/make_demo_site.py puts it on the browser demo's page.

The pictures are made by the program itself, so they show what it draws and not a mock-up. Every picture is written to
a scratch file first and replaces its place in assets/ only when it is a PNG of the expected size that is not almost
empty, so a failed run never leaves a half picture in the repository. Run it after `bash build.sh app`, then look at
the pictures before committing them:

    python tools/make_screenshots.py                      every picture, into assets/
    python tools/make_screenshots.py net replay_t5        only these (names without .png)

The program's profile goes to a scratch folder inside the project (_test_tmp/), removed at the end. Windows only.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'bin', 'vitro.exe')
ASSETS = os.path.join(ROOT, 'assets')
SCRATCH_PARENT = os.path.join(ROOT, '_test_tmp')

WIDTH, HEIGHT = 1280, 800
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
MIN_BYTES = 20_000  # a window that never drew is a flat colour and compresses to a few kilobytes


@dataclass(frozen=True)
class Picture:
    name: str  # the file in assets/, without .png
    fragment: str  # what to open: '' is the start screen
    wait_ms: int = 1000  # how long the window settles before it is captured
    size: tuple[int, int] = (WIDTH, HEIGHT)  # width and height of the window's client area, in pixels
    script: str = ''  # the body of an async function to run in the page first (--eval-file); if it throws, the picture fails


# Waits for the workspace and closes its left panel, so that the stage is wide and the panel's list is not cut off by
# the low window of the link-preview picture. The label is the one the panel's button carries in web/src/ui/Shell.tsx.
CLOSE_LEFT_PANEL = """\
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const find = () => document.querySelector('button[aria-label="Show or hide the left panel"]');
const started = Date.now();
while (find() === null) {
  if (Date.now() - started > 20000) throw new Error('the button of the left panel did not appear within 20 s');
  await sleep(50);
}
const button = find();
if (button.getAttribute('aria-pressed') === 'true') button.click();
await sleep(300);
if (button.getAttribute('aria-pressed') !== 'false') throw new Error('the left panel did not close');
return 'closed';
"""

PICTURES = (
    Picture('home', ''),
    Picture('tokens', '#open=1,0&view=tokens&token=3'),
    Picture('attention', '#open=3,2&view=attention&token=7&fam=2&grid=1'),
    Picture('activations', '#open=3,2&view=activations&token=7'),
    Picture('spans', '#open=3,2&view=spans'),
    Picture('net', '#open=3,2&view=net&token=7&nplay=0&ntime=3'),
    Picture('replay_t5', '#open=3,2&view=replay&pos=14.5&playing=0'),
    Picture('replay_qwen', '#open=1,0&view=replay&pos=5.5&playing=0'),
    Picture('social', '#open=3,2&view=replay&pos=14.5&playing=0', size=(1280, 640), script=CLOSE_LEFT_PANEL),
)


def png_size(path: str) -> tuple[int, int]:
    """Width and height from the header of a PNG file; refuses a file that is not one."""
    with open(path, 'rb') as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b'IHDR':
        raise SystemExit(f'{path} is not a PNG file')
    return struct.unpack('>II', head[16:24])


def take(picture: Picture, profile: str) -> str:
    """Makes one picture with the program and puts it in assets/. Returns a line for the report."""
    width, height = picture.size
    scratch = os.path.join(profile, picture.name + '.png')
    args = [EXE, '--screenshot', scratch, '--size', f'{width}x{height}', '--wait', str(picture.wait_ms), '--timeout', '90000']
    if picture.fragment:
        args += ['--hash', picture.fragment]
    if picture.script:
        script_path = os.path.join(profile, picture.name + '.js')
        with open(script_path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(picture.script)
        args += ['--eval-file', script_path]
    env = {**os.environ, 'VITRO_DATA_DIR': os.path.join(profile, 'engine')}
    done = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', env=env, timeout=180)
    if done.returncode != 0:
        # a script that threw says so on stdout (as JSON), the program's own errors on stderr
        raise SystemExit(f'{picture.name}: vitro.exe exited with {done.returncode}: {done.stderr.strip()} {done.stdout.strip()}')
    if not os.path.isfile(scratch):
        raise SystemExit(f'{picture.name}: vitro.exe reported success but made no picture')
    size = png_size(scratch)
    if size != (width, height):
        raise SystemExit(f'{picture.name}: the picture is {size[0]}x{size[1]}, expected {width}x{height} (is the display scaled?)')
    nbytes = os.path.getsize(scratch)
    if nbytes < MIN_BYTES:
        raise SystemExit(f'{picture.name}: the picture is only {nbytes} bytes, so the window drew almost nothing')
    os.replace(scratch, os.path.join(ASSETS, picture.name + '.png'))
    return f'{picture.name:12} {size[0]}x{size[1]} {nbytes:8} bytes'


def remove_when_free(path: str) -> None:
    """Removes a folder the engine may still hold for a moment; says so when it cannot."""
    for attempt in range(20):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as err:
            if attempt == 19:
                print(f'make_screenshots: could not remove {path}: {err}', file=sys.stderr)
                return
            time.sleep(0.5)


def main(argv: list[str]) -> int:
    if sys.platform != 'win32':
        raise SystemExit('make_screenshots.py drives bin/vitro.exe, which runs on Windows only')
    if not os.path.isfile(EXE):
        raise SystemExit(f'{EXE} not found: run `bash build.sh app` first')
    by_name = {p.name: p for p in PICTURES}
    unknown = [n for n in argv if n not in by_name]
    if unknown:
        raise SystemExit(f'unknown picture {unknown[0]!r}; the pictures are: {", ".join(by_name)}')
    wanted = [by_name[n] for n in argv] if argv else list(PICTURES)
    os.makedirs(SCRATCH_PARENT, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    profile = tempfile.mkdtemp(prefix='shots_', dir=SCRATCH_PARENT)
    try:
        for picture in wanted:
            print(take(picture, profile), flush=True)
    finally:
        remove_when_free(profile)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
