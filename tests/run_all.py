"""Mandatory verification suite for Vitro. Runs every check cleanly and exits non-zero on failure."""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_app  # noqa: E402
import test_container  # noqa: E402
import test_demo  # noqa: E402
import test_leak_audit  # noqa: E402
import test_release  # noqa: E402
import test_samples  # noqa: E402
import test_spec_and_capture  # noqa: E402
import test_viewer  # noqa: E402
import test_web  # noqa: E402


def main() -> int:
    t0 = time.time()
    suites = [
        ('leak_audit', test_leak_audit.run),
        ('release', test_release.run),
        ('container', test_container.run),
        ('spec_and_capture', test_spec_and_capture.run),
        ('viewer', test_viewer.run),
        ('demo', test_demo.run),
        ('samples', test_samples.run),
        ('web', test_web.run),
        ('app', test_app.run),
    ]
    for name, fn in suites:
        s0 = time.time()
        print(f'=== [{name}] ===', flush=True)
        fn()
        print(f'=== [{name}] passed ({time.time() - s0:.1f} s) ===\n', flush=True)
    print(f'ALL {len(suites)} VITRO VERIFICATION SUITES PASSED IN {time.time() - t0:.1f} s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
