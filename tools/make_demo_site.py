"""Builds the browser demo of Vitro: the app and its sample runs as plain files that any static web host can serve.

The app asks its server for two things: the list of trace files (`api/library`) and each file (`files/<id>`).
`vitro web` answers them on the fly; this script writes the same answers as files, so GitHub Pages (or any other
static host) can serve the app with the sample runs built in. The rest is the built app, `web/dist` (committed),
copied as it is. Only `index.html` changes: it gets the content security policy of `vitro web` as a <meta> tag (a
static host cannot send the header), the tags that make a link to the page preview well (title, description, picture)
and the project's address, which the start screen shows as the demo's links (web/src/ui/demo.ts).

What is written into OUT, a folder that does not exist yet or is empty:
    index.html, favicon.svg, assets/   the app
    api/library                        the list of the sample traces, as `vitro web` sends it
    files/<id>                         the sample traces, byte for byte, under the numbers api/library gives
    preview.png                        the picture of link previews (assets/social.png)
    .nojekyll                          tells a branch-based GitHub Pages to serve the files as they are

    .venv/Scripts/python.exe tools/make_demo_site.py --out _site

--site-url is the address the demo will have (link previews need absolute addresses) and --repo-url the project's.
Both must be plain https addresses and default to this project's own; the workflow passes the real ones. All input is
checked before anything is written, so a refusal leaves no half-built site. Needs NumPy (the container reader that
the list is made with) and nothing else.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'python', ROOT / 'tools'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from make_screenshots import png_size  # noqa: E402
from vitro import webserver  # noqa: E402

DIST = ROOT / 'web' / 'dist'
SOCIAL = ROOT / 'assets' / 'social.png'
DEFAULT_SITE_URL = 'https://eldroot-studios.github.io/vitro/'
DEFAULT_REPO_URL = 'https://github.com/EldRoot-Studios/vitro'

TITLE = 'Vitro: a glass-box inspector for PyTorch models'
DESCRIPTION = (
    'Open a sample trace and follow a prediction through the network: tokens, attention, activations and a '
    'step-by-step 3D replay of the generation. It runs in your browser and nothing is uploaded.'
)
PREVIEW_ALT = 'The Vitro window replaying a generation in 3D: the network, the generated tokens and the candidates of one step'

# Where the built page is changed. Vite writes these tags exactly so; a page that does not have them is refused.
CHARSET_TAG = '<meta charset="utf-8" />'
DESCRIPTION_TAG = re.compile(r'<meta name="description" content="[^"]*" />')
INDENT = '\n    '


def https_url(flag: str, value: str) -> str:
    """`value` if it is a plain https address: a host, no credentials, query or fragment, nothing a page must escape.

    Raises SystemExit naming `flag` and the value otherwise.
    """
    try:
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError as e:
        raise SystemExit(f'{flag}: "{value}" is not a valid address ({e})') from e
    if parts.scheme != 'https' or not host:
        raise SystemExit(f'{flag}: "{value}" is not an https address')
    if parts.query or parts.fragment or parts.username is not None or parts.password is not None:
        raise SystemExit(f'{flag}: "{value}" must not carry a query, a fragment or credentials')
    if re.search(r'[\s"\'<>]', value):
        raise SystemExit(f'{flag}: "{value}" holds a space, a quote or an angle bracket')
    return value


def meta_policy() -> str:
    """The content security policy of `vitro web` for a <meta> tag: without frame-ancestors, which a <meta> cannot carry."""
    directives = [d.strip() for d in webserver.CSP.split(';') if d.strip()]
    return '; '.join(d for d in directives if not d.startswith('frame-ancestors'))


def _meta(attribute: str, key: str, content: str) -> str:
    return f'<meta {attribute}="{key}" content="{html.escape(content, quote=True)}" />'


def demo_index(page: str, *, site_url: str, repo_url: str, preview_size: tuple[int, int]) -> str:
    """The built app's index.html with the demo's additions. Refuses a page that is not shaped as expected.

    The policy goes right after the charset tag, so it comes before any script; the description tag is replaced by
    itself, the demo marker, the canonical address and the link-preview tags.
    """
    if page.count(CHARSET_TAG) != 1:
        raise SystemExit(f'index.html: expected {CHARSET_TAG} once, found it {page.count(CHARSET_TAG)} times')
    found = DESCRIPTION_TAG.findall(page)
    if len(found) != 1:
        raise SystemExit(f'index.html: expected one <meta name="description"> tag, found {len(found)}')
    width, height = preview_size
    tags = [
        _meta('name', 'description', DESCRIPTION),
        _meta('name', 'vitro-demo', repo_url),
        f'<link rel="canonical" href="{html.escape(site_url, quote=True)}" />',
        _meta('property', 'og:type', 'website'),
        _meta('property', 'og:site_name', 'Vitro'),
        _meta('property', 'og:title', TITLE),
        _meta('property', 'og:description', DESCRIPTION),
        _meta('property', 'og:url', site_url),
        _meta('property', 'og:image', site_url + 'preview.png'),
        _meta('property', 'og:image:width', str(width)),
        _meta('property', 'og:image:height', str(height)),
        _meta('property', 'og:image:alt', PREVIEW_ALT),
        _meta('name', 'twitter:card', 'summary_large_image'),
    ]
    policy = f'<meta http-equiv="Content-Security-Policy" content="{html.escape(meta_policy(), quote=True)}" />'
    page = page.replace(CHARSET_TAG, CHARSET_TAG + INDENT + policy, 1)
    # a function as the replacement: the text is used as it is, with no backslash or group processing
    return DESCRIPTION_TAG.sub(lambda _match: INDENT.join(tags), page, count=1)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8', newline='\n')


def build(
    out: Path,
    site_url: str,
    repo_url: str,
    *,
    dist: Path = DIST,
    samples_dir: Path = webserver.SAMPLES_DIR,
    social: Path = SOCIAL,
) -> dict[str, int]:
    """Writes the demo site into `out`, a folder that does not exist yet or is empty; returns counts for the report.

    `site_url` ends with a slash, `repo_url` does not; both are plain https addresses. Everything that can be
    refused is checked before `out` is touched.
    """
    index = dist / 'index.html'
    if not index.is_file():
        raise SystemExit(f'{index} is missing: run `npm run build` in web/ and commit web/dist')
    if not social.is_file():
        raise SystemExit(f'{social} is missing: run `python tools/make_screenshots.py social` (needs bin/vitro.exe)')
    preview_size = png_size(str(social))
    entries = webserver.collect_entries([], samples_dir)
    if not entries:
        raise SystemExit(f'no sample traces (.vtdump, .vtdemo) in {samples_dir}: the demo would open with nothing to show')
    library = webserver.library_json(entries)
    broken = [f for f in library['files'] if 'error' in f]
    if broken:
        raise SystemExit('unreadable sample traces: ' + '; '.join(f'{f["name"]}: {f["error"]}' for f in broken))
    page = demo_index(index.read_text(encoding='utf-8'), site_url=site_url, repo_url=repo_url, preview_size=preview_size)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f'--out: {out} already exists and is not an empty folder; remove it or choose another')

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist, out, dirs_exist_ok=True)
    _write_text(out / 'index.html', page)
    (out / 'api').mkdir()
    # the same call as the server's, so the file holds what `vitro web` would send
    _write_text(out / 'api' / 'library', json.dumps(library, ensure_ascii=False))
    (out / 'files').mkdir()
    for number, entry in enumerate(entries):
        shutil.copyfile(entry.path, out / 'files' / str(number))
    shutil.copyfile(social, out / 'preview.png')
    _write_text(out / '.nojekyll', '')
    written = sorted(p for p in out.rglob('*') if p.is_file())
    return {'files': len(written), 'bytes': sum(p.stat().st_size for p in written), 'traces': len(entries)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Builds the static browser demo of Vitro (see the top of this file).')
    parser.add_argument('--out', required=True, type=Path, help='folder to write; it must not exist yet or must be empty')
    parser.add_argument('--site-url', default=DEFAULT_SITE_URL, help='the address the demo will have (https)')
    parser.add_argument('--repo-url', default=DEFAULT_REPO_URL, help="the project's address (https)")
    args = parser.parse_args(argv)
    site_url = https_url('--site-url', args.site_url)
    if not site_url.endswith('/'):
        site_url += '/'
    repo_url = https_url('--repo-url', args.repo_url).rstrip('/')
    out = args.out.resolve()
    counts = build(out, site_url, repo_url)
    print(
        f'make_demo_site: {counts["files"]} files, {counts["bytes"] / 1048576:.1f} MB, {counts["traces"]} sample traces '
        f'written to {out}; site {site_url}, project {repo_url}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
