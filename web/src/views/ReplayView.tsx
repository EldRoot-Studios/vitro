/**
 * Replay: a recorded generation played back in 3D, token by token. First the source context is read (a wave of
 * light over the attention links), then each token is written: its layers light up one after another, the
 * strongest attention sources glow, and the candidates for the next token appear. Port of cpp/src/demo_app.cpp.
 *
 * The picture owns the clock (see replayData.ts); the transport and the panel follow it.
 */

import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from 'react';

import { css, heat, hex, textOn, type RGB } from '../core/colors.ts';
import { TICKS_PER_SECOND } from '../core/demo.ts';
import { frameCamera } from '../core/math3d.ts';
import { pickNode, projectNodes, type Projection } from '../core/net.ts';
import { PULSE_COLOR, SPEEDS, buildScene, composeFrame, segmentTone, type Scene } from '../core/replay.ts';
import { app, useApp } from '../state/app.ts';
import { REPLAY_CAMERA } from '../state/state.ts';
import { FONT, addLine, fitCanvas, glow, onFontsLoaded, spark } from '../ui/canvas.ts';
import { clamp, formatProb, oneLine, visibleLabel } from '../ui/format.ts';
import { Icon } from '../ui/icons.tsx';
import { useSnapshotSource, useViewKeys } from '../ui/keys.ts';
import { freeRect } from '../ui/layout.ts';
import { useLoop } from '../ui/loop.ts';
import { drawBrackets, useOrbit } from '../ui/orbit.ts';
import { Candidate, Facts, Heading } from '../ui/parts.tsx';
import { tooltip } from '../ui/tooltip.ts';
import { useDemo, useResizeRedraw, vars, withLabelRoom } from './hooks.ts';
import { clock, jumpSegment, jumpTo, seekDivergence, seekFraction, stepBy, togglePlay } from './replayData.ts';

/** Keys of the state that change what the picture shows. */
const WATCHED = ['playing', 'speed', 'loop', 'ghost', 'yaw', 'pitch', 'zoom'];
const TONE_COLORS = { plain: 'rgba(255,255,255,0.3)', exact: 'rgba(127,207,159,0.5)', differs: 'rgba(232,123,123,0.5)' } as const;

export function ReplayStage() {
  const demo = useDemo();
  const canvas = useRef<HTMLCanvasElement>(null);
  const caption = useRef<HTMLElement>(null);
  const screen = useRef<Projection | null>(null);
  const committed = useRef<{ seg: number; reading: boolean | null; token: number }>({ seg: -1, reading: null, token: -1 });
  const scene = useRef<{ seg: number; scene: Scene } | null>(null);

  const request = useLoop((_now, dt) => {
    const c = canvas.current;
    if (!c) return false;
    const s = app.get();
    if (s.playing) {
      clock.sway += dt;
      clock.pos += dt * TICKS_PER_SECOND * s.speed;
      if (clock.pos >= demo.total) {
        if (s.loop) clock.pos %= Math.max(1e-6, demo.total);
        else {
          clock.pos = Math.max(0, demo.total - 1e-6);
          app.set({ pos: clock.pos, playing: false });
        }
      }
    }
    const nw = demo.locate(clock.pos);
    const cm = committed.current;
    if (nw.seg !== cm.seg || nw.reading !== cm.reading || nw.token !== cm.token) {
      committed.current = { seg: nw.seg, reading: nw.reading, token: nw.token };
      app.set({ pos: clock.pos });
    }
    if (scene.current?.seg !== nw.seg) scene.current = { seg: nw.seg, scene: buildScene(demo.segments[nw.seg]) };
    const sc = scene.current.scene;
    const sg = demo.segments[nw.seg];
    const { ctx, w, h } = fitCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const yaw = s.yaw + (s.playing ? 0.16 * Math.sin(clock.sway * 0.45) : 0);
    const cam = frameCamera({ yaw, pitch: s.pitch, zoom: s.zoom, halfSpan: sc.halfSpan, view: withLabelRoom(freeRect(c)), xs: sc.nodes.x, ys: sc.nodes.y, zs: sc.nodes.z, fitH: 0.7, dropY: 6 });
    const proj = projectNodes(sc.nodes, cam);
    screen.current = proj;
    const frame = composeFrame(demo, sg, sc, nw);
    if (caption.current && caption.current.textContent !== frame.caption) caption.current.textContent = frame.caption;

    ctx.globalCompositeOperation = 'lighter';
    ctx.lineWidth = 1;
    for (const l of frame.lines) addLine(ctx, proj.x[l.a], proj.y[l.a], proj.x[l.b], proj.y[l.b], l.color, l.alpha);
    for (let i = 0; i < sc.nodes.n; i++) {
      const look = frame.nodes[i];
      if (!look) continue;
      const r = look.radius * Math.min(1.8, Math.max(0.7, proj.scale[i] * 1.1));
      glow(ctx, proj.x[i], proj.y[i], r * 1.9, look.color, look.alpha * 0.55);
      spark(ctx, proj.x[i], proj.y[i], look.color, Math.min(1, look.alpha + 0.2));
    }
    for (const p of frame.pulses) glow(ctx, proj.x[p.a] + (proj.x[p.b] - proj.x[p.a]) * p.t, proj.y[p.a] + (proj.y[p.b] - proj.y[p.a]) * p.t, 3, PULSE_COLOR, p.alpha);
    for (const hl of frame.halos) glow(ctx, proj.x[hl.node], proj.y[hl.node], hl.radius * Math.max(0.8, proj.scale[hl.node]), hl.color, hl.alpha);
    ctx.globalCompositeOperation = 'source-over';
    ctx.font = FONT.sans(12.5);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (const hl of frame.halos) {
      if (hl.label === undefined) continue;
      ctx.fillStyle = hl.labelDim ? 'rgba(250,249,245,0.6)' : css(hl.color);
      ctx.fillText(oneLine(hl.label), proj.x[hl.node] + 11, proj.y[hl.node]);
    }
    drawBrackets(ctx, cam, sc.groups, sc.stageX, frame.groupLit);
    for (const f of clock.frame) f();
    return app.get().playing;
  });

  useOrbit(canvas, {
    get: () => {
      const s = app.get();
      return { yaw: s.yaw, pitch: s.pitch, zoom: s.zoom };
    },
    set: (p) => app.set({ ...(p.yaw !== undefined && { yaw: p.yaw }), ...(p.pitch !== undefined && { pitch: p.pitch }), ...(p.zoom !== undefined && { zoom: p.zoom }) }),
    hover: (x, y, e) => {
      const proj = screen.current;
      const sc = scene.current;
      if (!proj || !sc) return;
      const i = pickNode(proj, x, y);
      if (i < 0) return tooltip.hide();
      const sg = demo.segments[sc.seg];
      const st = sg.stages[sc.scene.nodes.stage[i]];
      const idx = sc.scene.nodes.idx[i];
      let text = `${st.label} [${idx}]`;
      if (st.tokenSource === 'target' && idx + st.tokenOffset < sg.ftok.length) text += ` "${demo.label(sg.ftok[idx + st.tokenOffset]).trim()}"`;
      else if (st.tokenSource === 'source' && sg.srcTok && idx + st.tokenOffset < sg.srcTok.length) text += ` "${demo.label(sg.srcTok[idx + st.tokenOffset]).trim()}"`;
      tooltip.show(text, e.clientX, e.clientY);
    },
    leave: () => tooltip.hide(),
  });

  // The clock starts at the committed position and follows the state when something else moves it (a click on a
  // segment, a link from the Tokens view). The check is made inside the notification, before the clock moves on.
  useEffect(() => {
    clock.pos = app.get().pos;
    clock.sway = 0;
    clock.request = request;
    const off = app.watch((s, _prev, changed) => {
      if (changed.includes('pos') && Math.abs(s.pos - clock.pos) > 1e-9) clock.pos = s.pos;
      if (changed.includes('pos') || changed.some((k) => WATCHED.includes(k as string))) request();
    });
    return () => {
      off();
      if (clock.request === request) clock.request = null;
      tooltip.hide();
    };
  }, [request]);
  useEffect(() => onFontsLoaded(request), [request]);
  useResizeRedraw(canvas, request);
  useSnapshotSource(() => canvas.current);

  useViewKeys((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const s = app.get();
    const speedAt = (SPEEDS as readonly number[]).indexOf(s.speed);
    switch (e.key) {
      case ' ':
        togglePlay();
        return true;
      case 'ArrowLeft':
        stepBy(e.shiftKey ? -10 : -1);
        return true;
      case 'ArrowRight':
        stepBy(e.shiftKey ? 10 : 1);
        return true;
      case 'ArrowUp':
        app.set({ speed: SPEEDS[Math.min(SPEEDS.length - 1, speedAt + 1)] });
        return true;
      case 'ArrowDown':
        app.set({ speed: SPEEDS[Math.max(0, speedAt - 1)] });
        return true;
      case '[':
        jumpSegment(-1);
        return true;
      case ']':
        jumpSegment(1);
        return true;
      case 'Home':
        jumpTo(0);
        return true;
      case 'End':
        jumpTo(Math.max(0, demo.total - 1e-6), { playing: false });
        return true;
      case 'l':
      case 'L':
        app.set({ loop: !s.loop });
        return true;
      case 'g':
      case 'G':
        app.set({ ghost: !s.ghost });
        return true;
      case 'r':
      case 'R':
        app.set({ yaw: REPLAY_CAMERA.yaw, pitch: REPLAY_CAMERA.pitch, zoom: REPLAY_CAMERA.zoom });
        return true;
      case 'd':
      case 'D':
        seekDivergence();
        return true;
      default:
        return false;
    }
  });

  return (
    <>
      <canvas ref={canvas} className="scene orbit" role="img" aria-label="3D replay of the generation" />
      <div className="over">
        <b ref={caption} />
        <span>Drag to orbit · scroll to zoom</span>
      </div>
    </>
  );
}

/** The transport: step and play buttons, speed, loop and reference toggles, and the timeline. */
export function ReplayDock() {
  const demo = useDemo();
  const playing = useApp((s) => s.playing);
  const speed = useApp((s) => s.speed);
  const loop = useApp((s) => s.loop);
  const ghost = useApp((s) => s.ghost);
  const timeline = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLElement>(null);
  const scrubbing = useRef(false);
  const total = Math.max(1e-6, demo.total);

  useEffect(() => {
    const update = (): void => {
      const pos = clamp(clock.pos, 0, demo.total);
      if (head.current) head.current.style.left = `${(pos / total) * 100}%`;
      const nw = demo.locate(pos);
      const sg = demo.segments[nw.seg];
      const text = `${sg.label}: ${nw.reading ? 'reading the source' : `token ${nw.token} of ${sg.ftok.length - 1}`}`;
      const slider = timeline.current;
      if (slider) {
        slider.setAttribute('aria-valuenow', String(Math.round(pos)));
        slider.setAttribute('aria-valuetext', text);
      }
    };
    clock.frame.add(update);
    update();
    return () => {
      clock.frame.delete(update);
    };
  }, [demo, total]);

  const scrub = (e: ReactPointerEvent): void => {
    const box = timeline.current!.getBoundingClientRect();
    seekFraction((e.clientX - box.left) / box.width);
  };

  return (
    <>
      <div className="tb">
        <button type="button" className="tbtn" aria-label="Previous segment ([)" title="Previous segment ([)" onClick={() => jumpSegment(-1)}>
          <Icon name="segmentBack" />
        </button>
        <button type="button" className="tbtn" aria-label="Step back (Left)" title="Step back (←)" onClick={() => stepBy(-1)}>
          <Icon name="stepBack" />
        </button>
        <button type="button" className="tbtn play" aria-label={playing ? 'Pause (Space)' : 'Play (Space)'} title="Play or pause (Space)" onClick={togglePlay}>
          <Icon name={playing ? 'pause' : 'play'} size={16} />
        </button>
        <button type="button" className="tbtn" aria-label="Step forward (Right)" title="Step forward (→)" onClick={() => stepBy(1)}>
          <Icon name="stepForward" />
        </button>
        <button type="button" className="tbtn" aria-label="Next segment (])" title="Next segment (])" onClick={() => jumpSegment(1)}>
          <Icon name="segmentForward" />
        </button>
      </div>
      <div
        className="tl"
        ref={timeline}
        role="slider"
        tabIndex={0}
        aria-label="Position in the recording"
        aria-valuemin={0}
        aria-valuemax={Math.round(demo.total)}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          scrubbing.current = true;
          scrub(e);
        }}
        onPointerMove={(e) => scrubbing.current && scrub(e)}
        onPointerUp={() => (scrubbing.current = false)}
        onPointerCancel={() => (scrubbing.current = false)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault();
            stepBy((e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1));
          }
        }}
      >
        <div className="track">
          {demo.segments.flatMap((sg, k) => {
            const tone = TONE_COLORS[segmentTone(sg)];
            const parts = [];
            if (sg.readTicks > 0) parts.push(<i key={`r${k}`} className="read" style={{ left: `${(sg.start / total) * 100}%`, width: `${(sg.readTicks / total) * 100}%` }} />);
            parts.push(<i key={`w${k}`} style={{ left: `${((sg.start + sg.readTicks) / total) * 100}%`, width: `${(sg.writeTicks / total) * 100}%`, background: tone }} />);
            return parts;
          })}
        </div>
        {demo.segments.map((sg, k) =>
          sg.diverge >= sg.t0 && sg.diverge < sg.ftok.length ? <i key={`d${k}`} className="mark" title="free generation diverges here" style={{ left: `${((sg.start + sg.readTicks + (sg.diverge - sg.t0)) / total) * 100}%` }} /> : null,
        )}
        <i className="head" ref={head} />
      </div>
      <div className="rate">
        <div className="speeds" role="group" aria-label="Speed">
          {SPEEDS.map((v) => (
            <button key={v} type="button" aria-pressed={v === speed} onClick={() => app.set({ speed: v })}>
              {v}×
            </button>
          ))}
        </div>
        <button type="button" className="tbtn" aria-pressed={loop} aria-label="Loop (L)" title="Loop (L)" onClick={() => app.set({ loop: !loop })}>
          <Icon name="loop" />
        </button>
        <button type="button" className="tbtn" aria-pressed={ghost} aria-label="Show the reference tokens (G)" title="Show the reference tokens (G)" onClick={() => app.set({ ghost: !ghost })}>
          <Icon name="eye" />
        </button>
      </div>
    </>
  );
}

const CONTEXT_CELL: RGB = hex('#1d1d1b');
const MISSED_CELL: RGB = hex('#5a2a2c');

/** The panel: the written tokens as a strip, and the step that is being written (or the reading progress). */
export function ReplayInspector() {
  const demo = useDemo();
  const dumpOpen = useApp((s) => s.dump !== null);
  const pos = useApp((s) => s.pos);
  const ghost = useApp((s) => s.ghost);
  const fill = useRef<HTMLElement>(null);
  const nw = demo.locate(pos);
  const sg = demo.segments[nw.seg];

  // the reading progress moves with the clock, not with the committed position
  useEffect(() => {
    if (!nw.reading) return;
    const update = (): void => {
      if (fill.current) fill.current.style.width = `${demo.locate(clock.pos).readP * 100}%`;
    };
    clock.frame.add(update);
    update();
    return () => {
      clock.frame.delete(update);
    };
  }, [demo, nw.reading]);

  const t = nw.reading ? sg.t0 - 1 : nw.token;
  const nRef = sg.refTok ? sg.refTok.length : 0;
  const count = ghost ? Math.max(sg.ftok.length, nRef) : sg.ftok.length;
  const sub = nRef > 0 ? `generated ${sg.ftok.length} tokens (reference ${nRef})${sg.freeExact ? ' · exact' : ''}` : `generated ${sg.ftok.length} tokens`;
  const k = demo.topK;
  const written = sg.ftok[nw.token];
  const topId = sg.tape.topId.int32();
  const topP = sg.tape.topP.float32();
  return (
    <>
      <div className="title">{sg.label}</div>
      <div className="sub">{sub}</div>
      <div className="rstrip">
        {Array.from({ length: count }, (_, i) => {
          const isWritten = i < sg.ftok.length;
          const word = isWritten ? demo.label(sg.ftok[i]) : demo.label(sg.refTok![i]);
          const same = i < nRef && isWritten && sg.refTok![i] === sg.ftok[i];
          const base: RGB = i < sg.t0 ? CONTEXT_CELL : nRef === 0 || same ? heat(0.12 + 0.55 * (isWritten ? sg.fp[i] : 0)) : MISSED_CELL;
          const done = i <= t && i < sg.ftok.length;
          return (
            <span key={i} className="rcell" data-state={done ? (i === t && !nw.reading ? 'current' : 'written') : 'pending'} data-diverge={i === sg.diverge || undefined} style={vars({ '--bg': css(base), '--fg': css(textOn(base)) })}>
              {visibleLabel(word)}
            </span>
          );
        })}
      </div>
      {nw.reading ? (
        <>
          <Heading>Reading the source</Heading>
          <div className="progress">
            <i ref={fill} />
          </div>
          <p className="sub">{sg.srcTok ? sg.srcTok.length : '?'} source tokens are encoded before the first token is written.</p>
        </>
      ) : (
        <>
          <Heading>
            Step {nw.token} <span className="mono tok">{demo.label(written)}</span>
          </Heading>
          <Facts
            rows={[
              ['Probability', formatProb(sg.fp[nw.token])],
              ['Entropy', `${sg.tape.entropy.float32()[nw.token].toFixed(2)} bits`],
            ]}
          />
          <Heading>Top-{k} candidates</Heading>
          <div className="cands">
            {Array.from({ length: k }, (_, o) => {
              const id = topId[nw.token * k + o];
              if (id < 0) return null;
              return <Candidate key={o} rank={o + 1} label={demo.label(id)} p={topP[nw.token * k + o]} ok={id === written} />;
            })}
          </div>
          {dumpOpen && (
            <div className="actions">
              <button type="button" className="btn ghost" onClick={() => app.set({ view: 'tokens', token: nw.token })}>
                Inspect this token
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}

