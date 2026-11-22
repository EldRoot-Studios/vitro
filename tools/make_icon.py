"""Draws the program icon, cpp/res/vitro.ico: the cube of the app (web/public/favicon.svg) on a dark tile.

The icon is committed; run this only to change it. It needs Pillow (`pip install pillow`). Every size is drawn on its
own at eight times the size and shrunk, so the small ones stay crisp, and stored as PNG inside the .ico, which Windows
10 and 11 read at every size.

    python tools/make_icon.py
"""
from __future__ import annotations

import io
import os
import struct
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'cpp', 'res', 'vitro.ico')
SIZES = (16, 24, 32, 48, 64, 128, 256)
BG = (5, 5, 6, 255)
INK = (244, 244, 245, 255)

# The geometry of favicon.svg, in its 24 x 24 box.
OUTLINE = [(12, 3.6), (19.2, 7.8), (19.2, 16.2), (12, 20.4), (4.8, 16.2), (4.8, 7.8)]
EDGES = [[(4.8, 7.8), (12, 12), (19.2, 7.8)], [(12, 12), (12, 20.4)]]
STROKE = 1.4
TILE_RADIUS = 6


def draw(size: int) -> Image.Image:
    k = 8
    big = size * k
    unit = big / 24
    img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, big - 1, big - 1], radius=TILE_RADIUS * unit, fill=BG)
    width = max(1, round(STROKE * unit))

    def line(points: list[tuple[float, float]], closed: bool = False) -> None:
        pts = [(x * unit, y * unit) for x, y in points] + ([(points[0][0] * unit, points[0][1] * unit)] if closed else [])
        d.line(pts, fill=INK, width=width, joint='curve')
        for x, y in pts:  # round joins and caps
            d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=INK)

    line(OUTLINE, closed=True)
    for edge in EDGES:
        line(edge)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main() -> int:
    blobs = []
    for size in SIZES:
        buf = io.BytesIO()
        draw(size).save(buf, format='PNG', optimize=True)
        blobs.append((size, buf.getvalue()))
    header = struct.pack('<HHH', 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries = b''
    for size, data in blobs:
        entries += struct.pack('<BBBBHHII', size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    with open(OUT, 'wb') as fh:
        fh.write(header + entries + b''.join(data for _s, data in blobs))
    print(f'make_icon: {OUT} ({offset} bytes, sizes {SIZES})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
