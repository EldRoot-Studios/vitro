/**
 * Everything that belongs to the whole window: the start-up from the address bar, the address bar itself, the
 * keyboard, dropped files, and the dialogs. The screens are Home and Workspace.
 */

import { useCallback, useEffect, type ChangeEvent } from 'react';

import { withClamp } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { loadLibrary, openFiles, openServerFiles } from '../state/session.ts';
import { VIEWS } from '../state/state.ts';
import { toasts } from '../state/toasts.ts';
import { formatHash, parseHash } from '../state/urlState.ts';
import { FilesDialog, HelpDialog } from './Dialogs.tsx';
import { Home } from './Home.tsx';
import { handleViewKey } from './keys.ts';
import { DropOverlay, Toasts, TooltipLayer, useFileDrop } from './Overlays.tsx';
import { attachPicker } from './picker.ts';
import { Workspace, snapshot } from './Shell.tsx';

const TEXT_TARGETS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/** The keys of the whole window; a key it does not use goes to the open view. */
function onKeyDown(e: KeyboardEvent): void {
  const target = e.target as HTMLElement;
  if (e.defaultPrevented || TEXT_TARGETS.has(target.tagName) || target.isContentEditable) return;
  const s = app.get();
  if (s.helpOpen || s.filesOpen) {
    if (e.key === 'Escape') app.set({ helpOpen: false, filesOpen: false });
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
    e.preventDefault();
    app.set({ filesOpen: true });
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  // Space and Enter belong to a focused button or link: it will click itself.
  if ((e.key === ' ' || e.key === 'Enter') && ['BUTTON', 'A'].includes(target.tagName)) return;
  const act = (fn: () => void): void => {
    e.preventDefault();
    fn();
  };
  if (e.key === '?' || e.key === 'F1') return act(() => app.set({ helpOpen: true }));
  if (e.key === 'o' || e.key === 'O') return act(() => app.set({ filesOpen: true }));
  const hasTrace = Boolean(s.dump || s.demo);
  if (s.screen === 'home' || !hasTrace) {
    if (hasTrace && (e.key === 'Escape' || e.key === 'b' || e.key === 'B')) act(() => app.set({ screen: 'app' }));
    return;
  }
  const digit = '123456'.indexOf(e.key);
  if (digit >= 0 && e.key.length === 1) return act(() => app.set({ view: VIEWS[digit].id }));
  if (e.key === 'b' || e.key === 'B') return act(() => app.set({ screen: 'home' }));
  if (e.key === 's' || e.key === 'S') return act(snapshot);
  if (handleViewKey(e)) e.preventDefault();
}

/** Opens what the address names, once: the files by their number on the server, then the view state. */
async function startFromAddress(): Promise<void> {
  const { patch, open, ignored } = parseHash(location.hash);
  if (ignored.length) toasts.show(`Ignored in the address: ${ignored.join(', ')}`, 'warn');
  await loadLibrary();
  if (open.length === 0) return;
  const { library } = app.get();
  if (library.status !== 'ready') {
    toasts.show(`Cannot open the files named in the address: the file list is unavailable (${library.error}).`, 'err');
    return;
  }
  const entries = open.map((id) => library.files.find((f) => f.id === id) ?? null);
  open.forEach((id, i) => entries[i] || toasts.show(`The address names file ${id}, which the server does not have.`, 'err'));
  const opened = await openServerFiles(entries.filter((f) => f !== null && !f.error));
  if (opened > 0) app.set(withClamp(app.get(), patch));
}

let started = false;

export function App() {
  const screen = useApp((s) => (s.screen === 'app' && (s.dump || s.demo) ? 'app' : 'home'));
  const helpOpen = useApp((s) => s.helpOpen);
  const filesOpen = useApp((s) => s.filesOpen);

  const drop = useCallback((files: File[]) => void openFiles(files), []);
  const dragging = useFileDrop(drop);

  useEffect(() => {
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // the address bar mirrors the view, a little after the last change
  useEffect(() => {
    let timer = 0;
    const write = (): void => {
      const s = app.get();
      const open = [s.dumpFile?.id, s.demoFile?.id].filter((id): id is number => id !== null && id !== undefined);
      const hash = s.dump || s.demo ? formatHash(s, open) : '';
      history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
      document.title = s.dump || s.demo ? `${(s.dump ?? s.demo)!.title} - Vitro` : 'Vitro';
    };
    const off = app.watch(() => {
      clearTimeout(timer);
      timer = window.setTimeout(write, 250);
    });
    return () => {
      off();
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (started) return;
    started = true;
    void startFromAddress();
  }, []);

  const onPick = (e: ChangeEvent<HTMLInputElement>): void => {
    const files = [...(e.target.files ?? [])];
    e.target.value = '';
    if (files.length) void openFiles(files);
  };

  return (
    <>
      {screen === 'app' ? <Workspace /> : <Home />}
      {filesOpen && <FilesDialog />}
      {helpOpen && <HelpDialog />}
      <input ref={attachPicker} type="file" multiple accept=".vtdump,.vtdemo" hidden aria-hidden="true" tabIndex={-1} onChange={onPick} />
      <DropOverlay show={dragging} />
      <Toasts />
      <TooltipLayer />
    </>
  );
}
