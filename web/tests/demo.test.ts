/** The demo marker of the hosted page: which project addresses the start screen accepts and which it refuses. */

import { afterEach, expect, test, vi } from 'vitest';

import { demoHomepage, type MetaSource } from '../src/ui/demo.ts';

const SELECTOR = 'meta[name="vitro-demo"]';

/** A stand-in for `document` with one marker tag whose content is `content` (a tag with no attribute when undefined), or no tag when null. */
function pageWith(content: string | null | undefined): MetaSource {
  return {
    querySelector: (selector) => {
      if (selector !== SELECTOR || content === null) return null;
      return { getAttribute: (name) => (name === 'content' && content !== undefined ? content : null) };
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

test('a page without the marker is not the demo, and says nothing', () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  expect(demoHomepage(pageWith(null))).toBeNull();
  expect(warn).not.toHaveBeenCalled();
});

test('a plain https address is returned without spaces and without a trailing slash', () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  expect(demoHomepage(pageWith('https://github.com/EldRoot-Studios/vitro'))).toBe('https://github.com/EldRoot-Studios/vitro');
  expect(demoHomepage(pageWith('  https://github.com/EldRoot-Studios/vitro//  '))).toBe('https://github.com/EldRoot-Studios/vitro');
  expect(demoHomepage(pageWith('https://example.org'))).toBe('https://example.org');
  expect(warn).not.toHaveBeenCalled();
});

test.each([
  ['http://github.com/EldRoot-Studios/vitro', 'not https'],
  ['javascript:alert(1)', 'a script address'],
  ['//github.com/EldRoot-Studios/vitro', 'no scheme'],
  ['/EldRoot-Studios/vitro', 'a relative path'],
  ['https://user:secret@github.com/EldRoot-Studios/vitro', 'credentials'],
  ['https://github.com/EldRoot-Studios/vitro?tab=readme', 'a query'],
  ['https://github.com/EldRoot-Studios/vitro#readme', 'a fragment'],
  ['', 'an empty value'],
])('%s is refused (%s) and the refusal names the value', (value) => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  expect(demoHomepage(pageWith(value))).toBeNull();
  expect(warn).toHaveBeenCalledTimes(1);
  expect(String(warn.mock.calls[0][0])).toContain(`"${value}"`);
});

test('a marker without a content attribute is refused like an empty one', () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  expect(demoHomepage(pageWith(undefined))).toBeNull();
  expect(warn).toHaveBeenCalledTimes(1);
});
