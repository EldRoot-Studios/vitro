/**
 * A tiny observable store: one state object, shallow patches, subscribers told which keys changed.
 * Components keep no copy of the state; they read it from here and redraw what depends on the changed keys.
 */

export interface Store<S extends object> {
  get(): S;
  /** Merges `patch`; keys whose value is the same (by Object.is) are not changes, and no change tells nobody. */
  set(patch: Partial<S>): void;
  /** Calls `fn` after every change with the new state, the previous one and the keys that changed. */
  watch(fn: (state: S, prev: S, changed: (keyof S)[]) => void): () => void;
  /** The form `useSyncExternalStore` wants: a callback without arguments. */
  subscribe(fn: () => void): () => void;
}

export function createStore<S extends object>(initial: S): Store<S> {
  let state: S = { ...initial };
  const watchers = new Set<(state: S, prev: S, changed: (keyof S)[]) => void>();
  return {
    get: () => state,
    set(patch) {
      const changed = (Object.keys(patch) as (keyof S)[]).filter((k) => !Object.is(state[k], patch[k]));
      if (changed.length === 0) return;
      const prev = state;
      state = { ...state, ...patch };
      for (const fn of [...watchers]) fn(state, prev, changed);
    },
    watch(fn) {
      watchers.add(fn);
      return () => watchers.delete(fn);
    },
    subscribe(fn) {
      const w = (): void => fn();
      watchers.add(w);
      return () => watchers.delete(w);
    },
  };
}
