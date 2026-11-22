/** Floating things that belong to no view: the pointer tooltip, the toasts, the drop overlay and the error panel. */

import { Component, useEffect, useState, useSyncExternalStore, type ReactNode } from 'react';

import { toastStore, toasts } from '../state/toasts.ts';
import { tooltip } from './tooltip.ts';

export function TooltipLayer() {
  return <div className="tip glass" role="tooltip" ref={tooltip.attach} />;
}

export function Toasts() {
  const items = useSyncExternalStore(toastStore.subscribe, () => toastStore.get().items);
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <button key={t.id} type="button" className="toast glass" data-kind={t.kind} data-leaving={t.leaving || undefined} onClick={() => toasts.dismiss(t.id)}>
          {t.message}
        </button>
      ))}
    </div>
  );
}

const carriesFiles = (e: DragEvent): boolean => [...(e.dataTransfer?.types ?? [])].includes('Files');

/**
 * Tracks files dragged over the window. The drop itself is handled here too: the files go to `onFiles`.
 * Returns whether a file is being dragged right now, for the overlay.
 */
export function useFileDrop(onFiles: (files: File[]) => void): boolean {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const enter = (e: DragEvent): void => {
      if (!carriesFiles(e)) return;
      depth++;
      setOver(true);
    };
    const leave = (e: DragEvent): void => {
      if (!carriesFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setOver(false);
    };
    const dragOver = (e: DragEvent): void => {
      if (carriesFiles(e)) e.preventDefault();
    };
    const drop = (e: DragEvent): void => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setOver(false);
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) onFiles(files);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', dragOver);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', dragOver);
      window.removeEventListener('drop', drop);
    };
  }, [onFiles]);
  return over;
}

export function DropOverlay({ show }: { show: boolean }) {
  return (
    <div className="dropzone" data-show={show || undefined} aria-hidden="true">
      <div className="glass">
        <b>Drop to open</b>
        <span>.vtdump and .vtdemo files</span>
      </div>
    </div>
  );
}

interface BoundaryProps {
  /** Names what failed in the message shown to the person. */
  label: string;
  children: ReactNode;
}

/** Catches an error thrown while a view is drawn and shows it, instead of leaving a blank window. */
export class ErrorBoundary extends Component<BoundaryProps, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  override componentDidCatch(error: Error): void {
    toasts.show(`${this.props.label}: ${error.message}`, 'err');
    console.error(error);
  }

  override render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="empty glass">
        <h2>{this.props.label} stopped working</h2>
        <p className="mono">{error.message}</p>
        <button type="button" className="btn ghost" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}
