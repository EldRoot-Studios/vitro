"""Release hygiene: one version everywhere, and a changelog entry for it.

The version is written in pyproject.toml (the source of truth: `bash build.sh app` builds it into vitro.exe), in
python/vitro/__init__.py, in web/package.json and twice in web/package-lock.json. They must agree, and CHANGELOG.md
must have a section "## [VERSION] - YYYY-MM-DD" for it, so a release is never cut with a stale number or without notes.
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return fh.read()


def declared(pattern: str, rel: str) -> str:
    found = re.search(pattern, read(rel), re.MULTILINE)
    assert found, f'{rel}: no version line (pattern {pattern})'
    return found.group(1)


def run() -> None:
    versions = {
        'pyproject.toml': declared(r'^version *= *"([^"]+)"', 'pyproject.toml'),
        'python/vitro/__init__.py': declared(r"^__version__ = '([^']+)'", 'python/vitro/__init__.py'),
        'web/package.json': json.loads(read('web/package.json'))['version'],
    }
    lock = json.loads(read('web/package-lock.json'))
    versions['web/package-lock.json'] = lock['version']
    versions['web/package-lock.json (packages[""])'] = lock['packages']['']['version']
    wanted = versions['pyproject.toml']
    assert re.fullmatch(r'\d+\.\d+\.\d+', wanted), f'pyproject.toml: "{wanted}" is not MAJOR.MINOR.PATCH'
    stale = {name: v for name, v in versions.items() if v != wanted}
    assert not stale, f'these disagree with pyproject.toml ({wanted}): {stale}'
    entry = re.compile(rf'^## \[{re.escape(wanted)}\] - \d{{4}}-\d{{2}}-\d{{2}}$', re.MULTILINE)
    assert entry.search(read('CHANGELOG.md')), f'CHANGELOG.md has no section "## [{wanted}] - YYYY-MM-DD"'
    print(f'test_release: version {wanted} is the same in {len(versions)} places and has a changelog entry', flush=True)


if __name__ == '__main__':
    run()
