/**
 * The shell: top bar, the panels and the open view. It owns no data; it asks the state which view is open,
 * mounts that view with its panels, and mounts it afresh when the view or the file behind it changes.
 *
 * Layout: the picture fills the window and the panels lie on it as glass. The stylesheet reads the data
 * attributes of `.app` to decide which panels exist and sets the insets of `.free`, the rectangle that is not
 * covered; DOM views live inside it and canvas views frame their content in it (see layout.ts).
 */

import { useEffect, useRef, useState } from 'react';

import { VIEW_COMPONENTS } from '../views/index.ts';
import { app, useApp } from '../state/app.ts';
import { VIEWS } from '../state/state.ts';
import { toasts } from '../state/toasts.ts';
import { EmptyView } from './EmptyView.tsx';
import { Icon, Logo } from './icons.tsx';
import { viewSnapshot } from './keys.ts';
import { ErrorBoundary } from './Overlays.tsx';
import { Rail } from './Rail.tsx';

const traceIds = new WeakMap<object, number>();
let lastTraceId = 0;

/** A number that identifies a parsed file, so that a view is mounted afresh when its file is replaced. */
function traceKey(trace: object): number {
  let id = traceIds.get(trace);
  if (id === undefined) traceIds.set(trace, (id = ++lastTraceId));
  return id;
}

/** Saves a canvas as a PNG on the black of the app; the scene canvases are transparent. */
function saveCanvas(canvas: HTMLCanvasElement, filename: string): void {
  const out = document.createElement('canvas');
  out.width = canvas.width;
  out.height = canvas.height;
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('this browser cannot draw on a canvas');
  ctx.fillStyle = '#050506';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(canvas, 0, 0);
  out.toBlob((blob) => {
    if (!blob) {
      toasts.show('The picture could not be made.', 'err');
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, 'image/png');
}

/** Saves the picture of the open view; says so when the view has none. */
export function snapshot(): void {
  const canvas = viewSnapshot();
  if (!canvas) {
    toasts.show('This view has no picture to save. Try Attention, Activations, 3D Net or Replay.', 'warn');
    return;
  }
  const s = app.get();
  saveCanvas(canvas, `vitro-${s.view}-${(s.dump ?? s.demo)!.uid}.png`);
}

export function toggleFullscreen(): void {
  const done = document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  done.catch((e: unknown) => toasts.show(`Full screen is not available: ${e instanceof Error ? e.message : String(e)}`, 'err'));
}

function Topbar() {
  const view = useApp((s) => s.view);
  const title = useApp((s) => (s.dump ?? s.demo)?.title ?? 'No trace open');
  const railOpen = useApp((s) => s.railOpen);
  const [full, setFull] = useState(false);
  const tabs = useRef<HTMLElement>(null);
  // in a narrow window the tabs scroll sideways; keep the open one in sight
  useEffect(() => {
    tabs.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [view]);
  return (
    <header className="topbar glass">
      <a
        className="brand"
        href="#"
        aria-label="Start screen"
        onClick={(e) => {
          e.preventDefault();
          app.set({ screen: 'home' });
        }}
      >
        <Logo />
        Vitro
      </a>
      <nav className="tabs" role="tablist" aria-label="Views" ref={tabs}>
        {VIEWS.map((v, i) => (
          <button key={v.id} type="button" role="tab" className="tab" aria-selected={v.id === view} title={`${v.label} (${i + 1})`} onClick={() => app.set({ view: v.id })}>
            {v.label}
          </button>
        ))}
      </nav>
      <div className="bar-r">
        <button type="button" className="session" aria-label="Open traces (O)" title="Open traces (O)" onClick={() => app.set({ filesOpen: true })}>
          {title}
        </button>
        <button type="button" className="ib" aria-label="Show or hide the left panel" aria-pressed={railOpen} title="Left panel" onClick={() => app.set({ railOpen: !railOpen })}>
          <Icon name="rail" />
        </button>
        <button type="button" className="ib" aria-label="Save the picture of this view (S)" title="Save the picture (S)" onClick={snapshot}>
          <Icon name="camera" />
        </button>
        <button
          type="button"
          className="ib"
          aria-label="Full screen"
          aria-pressed={full}
          title="Full screen"
          onClick={() => {
            toggleFullscreen();
            setFull(!full);
          }}
        >
          <Icon name="fullscreen" />
        </button>
        <button type="button" className="ib" aria-label="Help and shortcuts (?)" title="Help (?)" onClick={() => app.set({ helpOpen: true })}>
          <Icon name="help" />
        </button>
      </div>
    </header>
  );
}

/** The app screen: the open view with its panels. */
export function Workspace() {
  const view = useApp((s) => s.view);
  const dump = useApp((s) => s.dump);
  const demo = useApp((s) => s.demo);
  const railOpen = useApp((s) => s.railOpen);
  const info = VIEWS.find((v) => v.id === view)!;
  const data = info.needs === 'dump' ? dump : demo;
  const parts = data ? VIEW_COMPONENTS[view] : null;
  const { Stage, Inspector, Dock } = parts ?? { Stage: null, Inspector: null, Dock: null };
  const key = `${view}:${data ? traceKey(data) : 'none'}`;
  return (
    <div className="app" data-rail={railOpen} data-inspector={Inspector !== null} data-dock={Dock !== null}>
      <div className="free" aria-hidden="true" />
      <div className="stage" role="tabpanel" aria-label={info.label}>
        {Stage ? (
          <ErrorBoundary key={key} label={info.label}>
            <Stage />
          </ErrorBoundary>
        ) : (
          <EmptyView view={info} />
        )}
      </div>
      <Topbar />
      {railOpen && (
        <aside className="pane left glass" aria-label="Trace">
          <Rail />
        </aside>
      )}
      {Inspector && (
        <aside className="pane right glass" aria-label="Details">
          <ErrorBoundary key={key} label={`${info.label} details`}>
            <Inspector />
          </ErrorBoundary>
        </aside>
      )}
      {Dock && (
        <div className="dock glass">
          <ErrorBoundary key={key} label={`${info.label} controls`}>
            <Dock />
          </ErrorBoundary>
        </div>
      )}
    </div>
  );
}
