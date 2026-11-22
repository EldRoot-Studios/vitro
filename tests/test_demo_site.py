"""Verification of the browser demo: tools/make_demo_site.py writes a static copy of the app that any file server can serve.

It proves, on a site built into a scratch folder:
  * every file of the built app (web/dist) is there byte for byte, except index.html, and nothing else is there apart
    from what the demo adds (api/library, files/<id>, preview.png, .nojekyll);
  * api/library says what `vitro web` says about the sample traces (the same JSON), and files/<id> is the sample trace
    that the list names under that number, byte for byte, so the app finds what it asks for;
  * index.html carries the server's content security policy as a <meta> tag before the first script, refers to its files
    by relative paths that exist (a project site lives under /<name>/, so a path from the root would be a 404), names the
    project's address for the start screen, and has absolute, matching addresses in the link-preview tags;
  * a second build, from the command line, gives the same bytes;
  * bad input is refused with a message that names the flag or the file, and a refusal writes nothing: addresses that
    are not plain https, an output folder that is not empty, a missing app, a missing or broken preview picture, no
    sample traces, an unreadable one, an index.html that is not shaped as expected.
Needs NumPy (the container reader). It needs no Node, PyTorch or window, so the workflow runs it before it publishes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT / 'python', ROOT / 'tools'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import make_demo_site  # noqa: E402
from make_screenshots import png_size  # noqa: E402
from vitro import webserver  # noqa: E402

# Not this project's own addresses, so that a build which ignored its arguments and used the defaults would be caught.
SITE_URL = 'https://example.org/demo/'
REPO_URL = 'https://github.com/example/project'
CHARSET = '<meta charset="utf-8" />'
DESCRIPTION = '<meta name="description" content="x" />'


class Tags(HTMLParser):
    """Collects the attributes of every <meta>, <link> and <script> tag, with the character references decoded."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ('meta', 'link', 'script'):
            self.tags.append((tag, {key: value or '' for key, value in attrs}))


def parse_tags(page: str) -> list[tuple[str, dict[str, str]]]:
    parser = Tags()
    parser.feed(page)
    return parser.tags


def tree(base: Path) -> dict[str, Path]:
    """Every file under `base`, by its path written with forward slashes."""
    return {p.relative_to(base).as_posix(): p for p in sorted(base.rglob('*')) if p.is_file()}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refuses(action: Callable[[], object], pattern: str) -> None:
    """Fails unless `action` raises SystemExit with a message matching `pattern`."""
    try:
        action()
    except SystemExit as refusal:
        assert re.search(pattern, str(refusal)), f'the refusal "{refusal}" does not match {pattern!r}'
        return
    raise AssertionError(f'expected a refusal matching {pattern!r}, but nothing was refused')


def scratch_dir() -> tempfile.TemporaryDirectory:
    """A temporary folder inside the project, not on the system drive."""
    base = ROOT / '_test_tmp'
    base.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='site_', dir=base)


# ---- the built site ----


def check_files(site: Path, entries: list[webserver.Entry]) -> int:
    dist = tree(make_demo_site.DIST)
    built = tree(site)
    added = {'api/library', 'preview.png', '.nojekyll', *(f'files/{number}' for number in range(len(entries)))}
    assert set(built) == set(dist) | added, f'the site holds {sorted(set(built) ^ (set(dist) | added))} that should not differ'
    for rel, path in dist.items():
        if rel != 'index.html':
            assert digest(built[rel]) == digest(path), f'{rel} differs from web/dist'
    assert digest(built['preview.png']) == digest(make_demo_site.SOCIAL), 'preview.png is not assets/social.png'
    assert built['.nojekyll'].stat().st_size == 0, '.nojekyll must be empty'
    return len(dist) - 1


def check_library(site: Path, entries: list[webserver.Entry]) -> None:
    library = json.loads((site / 'api' / 'library').read_text(encoding='utf-8'))
    assert library == webserver.library_json(entries), 'api/library is not what the server sends'
    files = library['files']
    assert [f['id'] for f in files] == list(range(len(entries))), 'the numbers of api/library are not 0, 1, 2, ...'
    assert {f['kind'] for f in files} == {'dump', 'demo'}, 'the samples should hold both kinds of file'
    for f, entry in zip(files, entries, strict=True):
        assert 'error' not in f, f'{f["name"]} is listed as unreadable: {f.get("error")}'
        assert f['name'] == entry.path.name and f['source'] == 'sample'
        assert (site / 'files' / str(f['id'])).stat().st_size == f['size'], f'files/{f["id"]} has another size than the list says'
        assert digest(site / 'files' / str(f['id'])) == digest(entry.path), f'files/{f["id"]} is not {entry.path.name}'


def check_index(site: Path) -> None:
    page = (site / 'index.html').read_text(encoding='utf-8')
    tags = parse_tags(page)
    metas = [attrs for tag, attrs in tags if tag == 'meta']
    by_name = {m['name']: m['content'] for m in metas if 'name' in m}
    by_property = {m['property']: m['content'] for m in metas if 'property' in m}

    policies = [m['content'] for m in metas if m.get('http-equiv') == 'Content-Security-Policy']
    assert len(policies) == 1, f'expected one Content-Security-Policy <meta>, found {len(policies)}'
    want = [d.strip() for d in webserver.CSP.split(';') if d.strip() and not d.strip().startswith('frame-ancestors')]
    assert [d.strip() for d in policies[0].split(';')] == want, "the policy is not the server's without frame-ancestors"
    assert 'frame-ancestors' not in policies[0], 'a <meta> policy cannot carry frame-ancestors'
    assert page.index('Content-Security-Policy') < page.index('<script'), 'the policy must come before the first script'

    refs = [a['src'] for tag, a in tags if tag == 'script' and 'src' in a] + [a['href'] for tag, a in tags if tag == 'link' and 'href' in a]
    for ref in refs:
        if ref.startswith('https://'):
            assert ref == SITE_URL, f'unexpected absolute reference {ref}'
        else:
            assert ref.startswith('./') and (site / ref[2:]).is_file(), f'{ref} is not a relative path to a file of the site'
    assert refs.count(SITE_URL) == 1, 'the canonical address should appear once'
    assert any(r.startswith('./assets/') and r.endswith('.js') for r in refs), 'the app script is not referred to'
    assert any(r.startswith('./assets/') and r.endswith('.css') for r in refs), 'the app stylesheet is not referred to'

    assert by_name['vitro-demo'] == REPO_URL, 'the demo marker does not name the project'
    assert by_name['twitter:card'] == 'summary_large_image'
    assert by_property['og:url'] == SITE_URL, f'og:url is {by_property["og:url"]}, expected {SITE_URL}'
    assert by_property['og:image'] == SITE_URL + 'preview.png', f'og:image is {by_property["og:image"]}, expected {SITE_URL}preview.png'
    shown = (int(by_property['og:image:width']), int(by_property['og:image:height']))
    assert shown == png_size(str(site / 'preview.png')), f'og:image:width/height say {shown}, the picture is {png_size(str(site / "preview.png"))}'
    assert by_name['description'] and by_name['description'] == by_property['og:description']
    for key in ('og:type', 'og:site_name', 'og:title', 'og:image:alt'):
        assert by_property.get(key), f'{key} is missing or empty'


def check_escaping() -> None:
    """An address with an ampersand reaches the marker as it is written, not as a broken attribute."""
    page = make_demo_site.demo_index(CHARSET + DESCRIPTION, site_url=SITE_URL, repo_url='https://example.org/a&b', preview_size=(2, 1))
    marker = [attrs for tag, attrs in parse_tags(page) if tag == 'meta' and attrs.get('name') == 'vitro-demo']
    assert len(marker) == 1 and marker[0]['content'] == 'https://example.org/a&b', marker
    assert 'a&amp;b' in page, 'the ampersand was not escaped in the page'


# ---- refusals ----


def check_addresses() -> int:
    cases = (
        ('http://example.org/', 'not an https address'),
        ('javascript:alert(1)', 'not an https address'),
        ('//example.org/x', 'not an https address'),
        ('example.org', 'not an https address'),
        ('https://user:secret@example.org/', 'credentials'),
        ('https://example.org/?a=1', 'query'),
        ('https://example.org/#x', 'fragment'),
        ('https://example.org/a"b', 'quote'),
        ('https://example.org/a b', 'space'),
        ('https://[::1', 'not a valid address'),
    )
    for value, reason in cases:
        refuses(lambda v=value: make_demo_site.https_url('--site-url', v), rf'--site-url: "{re.escape(value)}".*{reason}')
    assert make_demo_site.https_url('--repo-url', REPO_URL) == REPO_URL
    return len(cases)


def check_refusals(base: Path) -> int:
    """Each refusal names its reason and leaves the output folder alone."""
    count = 0

    def refused_build(name: str, pattern: str, **kwargs: Path) -> None:
        nonlocal count
        out = base / name
        refuses(lambda: make_demo_site.build(out, SITE_URL, REPO_URL, **kwargs), pattern)
        assert not out.exists(), f'a refused build left {out} behind'
        count += 1

    no_dist = base / 'no_dist'
    no_dist.mkdir()
    refused_build('r_dist', r'index\.html is missing', dist=no_dist)
    refused_build('r_social', r'none\.png is missing', social=base / 'none.png')
    not_png = base / 'not_a_picture.png'
    not_png.write_text('this is not a PNG file', encoding='utf-8', newline='\n')
    refused_build('r_png', r'is not a PNG file', social=not_png)
    no_samples = base / 'no_samples'
    no_samples.mkdir()
    refused_build('r_empty', r'no sample traces', samples_dir=no_samples)
    broken = base / 'broken_samples'
    broken.mkdir()
    (broken / 'bad.vtdump').write_bytes(b'this is not a Vitro trace')
    refused_build('r_broken', r'unreadable sample traces: bad\.vtdump', samples_dir=broken)

    busy = base / 'busy'
    busy.mkdir()
    (busy / 'keep.txt').write_text('mine', encoding='utf-8', newline='\n')
    refuses(lambda: make_demo_site.build(busy, SITE_URL, REPO_URL), r'not an empty folder')
    assert sorted(p.name for p in busy.iterdir()) == ['keep.txt'], 'a refused build changed the folder that was in the way'
    a_file = base / 'a_file'
    a_file.write_text('x', encoding='utf-8', newline='\n')
    refuses(lambda: make_demo_site.build(a_file, SITE_URL, REPO_URL), r'not an empty folder')
    assert a_file.read_text(encoding='utf-8') == 'x'
    count += 2

    kwargs = {'site_url': SITE_URL, 'repo_url': REPO_URL, 'preview_size': (10, 5)}
    refuses(lambda: make_demo_site.demo_index('<head></head>', **kwargs), r'expected <meta charset="utf-8" /> once, found it 0 times')
    refuses(lambda: make_demo_site.demo_index(CHARSET * 2 + DESCRIPTION, **kwargs), r'found it 2 times')
    refuses(lambda: make_demo_site.demo_index(CHARSET, **kwargs), r'expected one <meta name="description"> tag, found 0')
    refuses(lambda: make_demo_site.demo_index(CHARSET + DESCRIPTION * 2, **kwargs), r'expected one <meta name="description"> tag, found 2')
    count += 4

    # an existing folder that is empty is fine: it is what a fresh checkout of a deploy folder looks like
    empty_out = base / 'empty_out'
    empty_out.mkdir()
    make_demo_site.build(empty_out, SITE_URL, REPO_URL)
    assert (empty_out / 'index.html').is_file()
    return count


def run() -> None:
    entries = webserver.collect_entries([], webserver.SAMPLES_DIR)
    assert len(entries) >= 2, f'expected the sample traces in {webserver.SAMPLES_DIR}, found {len(entries)} files'
    with scratch_dir() as scratch:
        base = Path(scratch)
        site = base / 'site'
        counts = make_demo_site.build(site, SITE_URL, REPO_URL)
        app_files = check_files(site, entries)
        check_library(site, entries)
        check_index(site)
        check_escaping()
        print(f'test_demo_site: {app_files} app files byte for byte, library and {len(entries)} traces as the server has them, index.html as expected', flush=True)

        # same input, same bytes, and the addresses are normalised the way the command line promises
        again = base / 'again'
        code = make_demo_site.main(['--out', str(again), '--site-url', SITE_URL.rstrip('/'), '--repo-url', REPO_URL + '/'])
        assert code == 0
        first = {rel: digest(path) for rel, path in tree(site).items()}
        second = {rel: digest(path) for rel, path in tree(again).items()}
        assert first == second, f'a second build differs in {sorted(k for k in first.keys() | second.keys() if first.get(k) != second.get(k))}'
        assert counts['files'] == len(first) and counts['traces'] == len(entries), counts
        print(f'test_demo_site: a second build, from the command line, gives the same {len(first)} files', flush=True)

        bad_addresses = check_addresses()
        other_refusals = check_refusals(base)
        print(f'test_demo_site: {bad_addresses} bad addresses and {other_refusals} other refusals, each with its reason', flush=True)
    print('test_demo_site: every part ran', flush=True)


if __name__ == '__main__':
    run()
