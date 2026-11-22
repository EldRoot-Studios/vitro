/**
 * The pointer tooltip. It changes on every pointer move, so it is moved by hand on one element instead of
 * through React state: `TooltipLayer` renders the element and hands it over with `attach`.
 */

let el: HTMLElement | null = null;

export const tooltip = {
  /** Called by `TooltipLayer` with its element, and with null when it unmounts. */
  attach(node: HTMLElement | null): void {
    el = node;
  },

  /** Shows `text` near the client position (x, y), keeping the tooltip inside the window. */
  show(text: string, x: number, y: number): void {
    if (!el) return;
    if (el.textContent !== text) el.textContent = text;
    el.dataset.show = '';
    const pad = 14;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const left = x + pad + w > window.innerWidth - 8 ? Math.max(8, x - pad - w) : x + pad;
    const top = y + pad + h > window.innerHeight - 8 ? Math.max(8, y - pad - h) : y + pad;
    el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  },

  hide(): void {
    if (el) delete el.dataset.show;
  },
};
