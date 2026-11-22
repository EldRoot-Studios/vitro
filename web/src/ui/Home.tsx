/**
 * The start screen: a glass card over a slowly turning network, a place to drop files, and the sample runs.
 * It is also where the brand mark leads back to, with a way to continue the session that is open.
 */

import { useEffect, useRef } from 'react';

import { app, useApp } from '../state/app.ts';
import { fitCanvas, onFontsLoaded } from './canvas.ts';
import { drawHomeScene } from './homeScene.ts';
import { Icon, Logo } from './icons.tsx';
import { reducedMotion, useLoop } from './loop.ts';
import { Pick, RunList } from './parts.tsx';

export function Home() {
  const hasTrace = useApp((s) => s.dump !== null || s.demo !== null);
  const title = useApp((s) => (s.dump ?? s.demo)?.title ?? '');
  const library = useApp((s) => s.library);
  const canvas = useRef<HTMLCanvasElement>(null);
  const still = useRef(reducedMotion());
  const start = useRef(performance.now());

  const request = useLoop((now) => {
    const c = canvas.current;
    if (!c) return false;
    const { ctx, w, h } = fitCanvas(c);
    ctx.clearRect(0, 0, w, h);
    drawHomeScene(ctx, { x: 0, y: 0, w, h }, still.current ? 3.2 : (now - start.current) / 1000);
    return !still.current;
  });
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const resize = new ResizeObserver(request);
    resize.observe(c);
    return () => resize.disconnect();
  }, [request]);
  useEffect(() => onFontsLoaded(request), [request]);

  return (
    <section className="home" aria-label="Start">
      <canvas ref={canvas} className="scene" role="img" aria-label="A network of glowing points, turning slowly" />
      <div className="home-actions">
        <button type="button" className="ib" aria-label="Help and shortcuts (?)" onClick={() => app.set({ helpOpen: true })}>
          <Icon name="help" />
        </button>
      </div>
      <div className="card glass">
        <div className="brand">
          <Logo />
          Vitro
        </div>
        <h1>See inside the model</h1>
        <p>Open a saved trace and follow a prediction through the network: tokens, attention, activations and a step-by-step 3D replay of the generation.</p>
        <Pick />
        <RunList library={library} />
        {hasTrace && (
          <button type="button" className="btn ghost back" onClick={() => app.set({ screen: 'app' })}>
            Back to {title}
            <Icon name="arrow" size={15} />
          </button>
        )}
        <div className="foot">Nothing is uploaded: traces are read by your browser and never leave this machine.</div>
      </div>
    </section>
  );
}
