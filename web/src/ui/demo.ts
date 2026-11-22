/**
 * The hosted browser demo. The page that serves it names the project's address in `<meta name="vitro-demo">`
 * (tools/make_demo_site.py writes the tag); the program (vitro.exe) and `vitro web` do not carry the tag, so the
 * start screen shows the demo's links only on the hosted page.
 */

/** The part of `document` this file reads: the page is passed in, so that a test can give a stand-in and no global is touched. */
export interface MetaSource {
  querySelector(selector: string): { getAttribute(name: string): string | null } | null;
}

/** The address as a URL if it is a plain https address (no credentials, query or fragment), else null. */
function parsePlainHttps(text: string): URL | null {
  try {
    const url = new URL(text);
    const plain = url.protocol === 'https:' && url.hostname !== '' && url.username === '' && url.password === '' && url.search === '' && url.hash === '';
    return plain ? url : null;
  } catch {
    return null;
  }
}

/**
 * The project address (https, without a trailing slash) that the page names, or null when the page is not the hosted
 * demo. The value becomes a link, so anything but a plain https address is refused: that is reported in the console
 * with the value, and the links are left out.
 */
export function demoHomepage(doc: MetaSource): string | null {
  const tag = doc.querySelector('meta[name="vitro-demo"]');
  if (tag === null) return null;
  const text = (tag.getAttribute('content') ?? '').trim();
  const url = parsePlainHttps(text);
  if (url === null) {
    console.warn(`<meta name="vitro-demo"> must hold a plain https address, not "${text}"; the demo links are left out`);
    return null;
  }
  return url.href.replace(/\/+$/, '');
}
