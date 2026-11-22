/**
 * Corner notifications. Errors stay for eight seconds, other messages for four; at most four are shown, the
 * oldest goes first. A message can be dismissed by clicking it.
 */

import { createStore } from './store.ts';

export type ToastKind = 'info' | 'ok' | 'warn' | 'err';

export interface Toast {
  id: number;
  message: string;
  kind: ToastKind;
  /** True during the short fade-out before the toast is removed. */
  leaving: boolean;
}

export const toastStore = createStore<{ items: Toast[] }>({ items: [] });

const SHOWN_AT_MOST = 4;
const FADE_MS = 250;
let nextId = 1;
const timers = new Map<number, ReturnType<typeof setTimeout>>();

function forget(id: number): void {
  clearTimeout(timers.get(id));
  timers.delete(id);
}

function remove(id: number): void {
  forget(id);
  toastStore.set({ items: toastStore.get().items.filter((t) => t.id !== id) });
}

export const toasts = {
  show(message: string, kind: ToastKind = 'info'): void {
    const id = nextId++;
    const items = [...toastStore.get().items, { id, message, kind, leaving: false }];
    while (items.length > SHOWN_AT_MOST) forget(items.shift()!.id);
    toastStore.set({ items });
    timers.set(id, setTimeout(() => toasts.dismiss(id), kind === 'err' ? 8000 : 4000));
  },

  /** Fades the toast out and removes it. */
  dismiss(id: number): void {
    forget(id);
    toastStore.set({ items: toastStore.get().items.map((t) => (t.id === id ? { ...t, leaving: true } : t)) });
    timers.set(id, setTimeout(() => remove(id), FADE_MS));
  },
};
