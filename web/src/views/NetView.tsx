/**
 * 3D Net: the network of the trace as columns of lit nodes. Pick a token and a backward attention rollout
 * lights the path of influence that led to its prediction; a wave of light runs through the stages.
 * Drag to orbit, wheel to zoom, click a node to pick the token behind it.
 * Port of cpp/src/view_net.cpp (draw_net).
 */

import { useEffect, useRef, useSyncExternalStore } from 'react';

import { tokenText } from '../core/analysis.ts';
import { mix } from '../core/colors.ts';
import { frameCamera, hash01 } from '../core/math3d.ts';
import { EDGE, NET_KS, WAVE_PERIOD, describeNode, pickNode, projectNodes, selectedToken, type Projection } from '../core/net.ts';
import { selectTokenPatch, selectUnitPatch } from '../state/actions.ts';
import { app, useApp } from '../state/app.ts';
import { FONT, addLine, fitCanvas, glow, onFontsLoaded, spark } from '../ui/canvas.ts';
import { clamp, formatProb } from '../ui/format.ts';
import { Icon } from '../ui/icons.tsx';
import { useSnapshotSource, useViewKeys } from '../ui/keys.ts';
import { freeRect } from '../ui/layout.ts';
import { useLoop } from '../ui/loop.ts';
import { drawBrackets, useOrbit } from '../ui/orbit.ts';
import { CandidateList, Chips, Facts, Heading } from '../ui/parts.tsx';
import { tooltip } from '../ui/tooltip.ts';
import { compact, useResizeRedraw, useUnit, withLabelRoom } from './hooks.ts';
import { NET_BASE, NET_OK, NET_OUT_OTHER, NET_WHITE, netClock, netFor, netHover, resetNetCamera, toggleNetPlay } from './netData.ts';

/** Keys of the state that change what the picture shows without changing the graph. */
const WATCHED = ['netYaw', 'netPitch', 'netZoom', 'netPath', 'netPlay', 'netTime'] as const;

export function NetStage() {
  const { dump, unit } = useUnit();
  const token = useApp((s) => s.token);
  const head = useApp((s) => s.head);
  const netK = useApp((s) => s.netK);
  const net = netFor(dump, unit, token, head, netK);
  const j = selectedToken(unit, token);
  const canvas = useRef<HTMLCanvasElement>(null);
  const screen = useRef<Projection | null>(null);
  const hovered = useRef(-1);
  const perf = useRef({ lod: 1, emaMs: 0 });

  const request = useLoop((_now, dt) => {
    const c = canvas.current;
    if (!c) return false;
    const s = app.get();
    if (s.netPlay) netClock.time += dt;
    const { ctx, w, h } = fitCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const cam = frameCamera({ yaw: s.netYaw, pitch: s.netPitch, zoom: s.netZoom, halfSpan: net.halfSpan, view: withLabelRoom(freeRect(c)), xs: net.nodes.x, ys: net.nodes.y, zs: net.nodes.z, fitH: 0.74, dropY: 8 });
    const proj = projectNodes(net.nodes, cam);
    screen.current = proj;
    const t0 = performance.now();
    const dragging = c.dataset.dragging !== undefined;
    const p = perf.current;
    const eff = Math.max(0.08, dragging ? p.lod * 0.5 : p.lod);
    const faintGain = Math.min(6, 1 / eff);
    const wave = s.netPlay || netClock.time > 0;
    const wavePos = wave ? ((netClock.time / WAVE_PERIOD) % 1) * Math.max(1, net.nStages - 1) : -10;
    const { edges, nodes } = net;

    ctx.globalCompositeOperation = 'lighter';
    ctx.lineWidth = 1;
    for (let e = 0; e < edges.n; e++) {
      const imp = edges.imp[e];
      if (s.netPath && imp < 0.1) continue;
      const kind = edges.kind[e];
      const faint = imp < 0.1 || (kind === EDGE.RESID && imp < 0.12);
      if (faint && eff < 1 && hash01(e, 5) >= eff) continue;
      const gain = faint ? faintGain : 1;
      const a = edges.a[e];
      const b = edges.b[e];
      const boost = wave ? 1 + 1.7 * Math.exp(-((edges.stage[e] - 0.5 - wavePos) ** 2) / 1.4) : 1;
      let color;
      let alpha;
      if (kind === EDGE.RESID && imp < 0.12) {
        color = NET_BASE;
        alpha = 0.022 * boost * gain;
      } else if (imp < 0.1) {
        color = net.tints[edges.tint[e]];
        alpha = (0.04 + 0.12 * Math.sqrt(Math.min(1, edges.w[e]))) * boost * gain;
      } else {
        const g = imp ** 0.9;
        color = mix(net.tints[edges.tint[e]], NET_WHITE, Math.min(1, g * 0.7));
        alpha = (0.09 + 0.62 * g) * boost * (kind === EDGE.RESID ? 0.5 : 1);
      }
      addLine(ctx, proj.x[a], proj.y[a], proj.x[b], proj.y[b], color, Math.min(1, alpha));
      if (!faint && imp ** 0.9 > 0.7) addLine(ctx, proj.x[a] + 1, proj.y[a], proj.x[b] + 1, proj.y[b], color, Math.min(1, alpha * 0.6));
    }

    for (let i = 0; i < nodes.n; i++) {
      const stage = nodes.stage[i];
      const isOut = stage === net.sOut;
      const latent = !isOut && unit.stages[stage].kind === 'latent';
      const g = isOut ? 1 : nodes.imp[i] ** 0.6;
      if (dragging && g < 0.2 && i & 1) continue;
      const color = isOut ? (i === net.trueOut ? NET_OK : NET_OUT_OTHER) : mix(NET_BASE, NET_WHITE, g * 0.6);
      const crowd = Math.min(Math.max(70 / Math.max(1, net.count[stage]), 0.22), 1);
      const scale = Math.min(1.8, Math.max(0.7, proj.scale[i] * 1.1));
      const r = (latent ? 3 : (0.8 + 1.1 * g) * Math.sqrt(crowd) + 0.3) * scale;
      const alpha = isOut ? 0.9 : Math.min(1, (0.08 + 0.3 * nodes.act[i] + 0.7 * g) * (latent ? 1.4 : Math.sqrt(crowd) + 0.15));
      glow(ctx, proj.x[i], proj.y[i], r * 1.9, color, alpha * 0.5);
      spark(ctx, proj.x[i], proj.y[i], color, Math.min(1, alpha + 0.2));
    }
    // keep the frame time near what a screen can show: draw fewer faint edges when it runs long
    p.emaMs = 0.7 * p.emaMs + 0.3 * (performance.now() - t0);
    if (p.emaMs > 12.5) p.lod = Math.max(0.1, p.lod * 0.85);
    else if (p.emaMs < 7) p.lod = Math.min(1, p.lod * 1.1);

    if (net.sPred < net.sOut && net.count[net.sPred] > 0) {
      const sel = net.first[net.sPred] + net.selP;
      glow(ctx, proj.x[sel], proj.y[sel], 11 * Math.max(0.8, proj.scale[sel]), NET_WHITE, 0.7);
      if (net.trueOut >= 0) glow(ctx, proj.x[net.trueOut], proj.y[net.trueOut], 9 * Math.max(0.8, proj.scale[net.trueOut]), NET_OK, 0.7);
    }
    if (wave) {
      for (const e of net.hot) {
        const a = edges.a[e];
        const b = edges.b[e];
        const ph = (netClock.time * 0.42 + hash01(e, 77)) % 1;
        for (let tail = 0; tail < 4; tail++) {
          let tt = ph - 0.025 * tail;
          if (tt < 0) tt += 1;
          glow(ctx, proj.x[a] + (proj.x[b] - proj.x[a]) * tt, proj.y[a] + (proj.y[b] - proj.y[a]) * tt, 3 - 0.5 * tail, NET_WHITE, (0.75 - 0.2 * tail) * Math.min(1, 0.4 + edges.imp[e]));
        }
      }
    }
    ctx.globalCompositeOperation = 'source-over';

    ctx.font = FONT.sans(12.5);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (let o = 0; o < net.count[net.sOut]; o++) {
      const i = net.outFirst + o;
      ctx.fillStyle = i === net.trueOut ? '#7fcf9f' : 'rgba(250,249,245,0.82)';
      ctx.fillText(compact(dump.label(nodes.idx[i])), proj.x[i] + 11, proj.y[i]);
    }
    const hv = hovered.current;
    if (hv >= 0 && hv < nodes.n) {
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(proj.x[hv], proj.y[hv], 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    drawBrackets(
      ctx,
      cam,
      net.groups,
      net.stageX,
      net.groups.map((g) => wave && wavePos + 0.5 >= g.a && wavePos - 0.5 <= g.b),
    );
    return s.netPlay;
  });

  useOrbit(canvas, {
    get: () => {
      const s = app.get();
      return { yaw: s.netYaw, pitch: s.netPitch, zoom: s.netZoom };
    },
    set: (p) => app.set({ ...(p.yaw !== undefined && { netYaw: p.yaw }), ...(p.pitch !== undefined && { netPitch: p.pitch }), ...(p.zoom !== undefined && { netZoom: p.zoom }) }),
    click: (x, y) => {
      const proj = screen.current;
      if (!proj) return;
      const i = pickNode(proj, x, y);
      if (i < 0) return;
      const stage = net.nodes.stage[i];
      if (stage >= unit.stages.length) return;
      const st = unit.stages[stage];
      if (st.tokenSource !== 'target') return;
      const tok = net.nodes.idx[i] + st.tokenOffset;
      if (tok >= unit.t0 && tok < unit.nTokens) app.set(selectTokenPatch(app.get(), tok));
    },
    hover: (x, y, e) => {
      const proj = screen.current;
      if (!proj) return;
      const i = pickNode(proj, x, y);
      if (i !== hovered.current) {
        hovered.current = i;
        netHover.set({ text: i >= 0 ? describeNode(dump, unit, net, i) : '' });
        request();
      }
      if (i >= 0) tooltip.show(describeNode(dump, unit, net, i), e.clientX, e.clientY);
      else tooltip.hide();
    },
    leave: () => {
      hovered.current = -1;
      netHover.set({ text: '' });
      tooltip.hide();
      request();
    },
  });

  // a new graph: forget the hovered node; the wave clock follows the state while the wave is paused
  useEffect(() => {
    hovered.current = -1;
    netHover.set({ text: '' });
    request();
  }, [request, net]);
  useEffect(() => {
    netClock.time = app.get().netTime;
    return app.watch((s, _prev, changed) => {
      if (!changed.some((k) => (WATCHED as readonly string[]).includes(k))) return;
      if (changed.includes('netTime') && !s.netPlay) netClock.time = s.netTime;
      request();
    });
  }, [request]);
  useEffect(() => onFontsLoaded(request), [request]);
  useEffect(() => () => tooltip.hide(), []);
  useResizeRedraw(canvas, request);
  useSnapshotSource(() => canvas.current);

  useViewKeys((e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const s = app.get();
    switch (e.key) {
      case ' ':
        toggleNetPlay();
        return true;
      case 'p':
      case 'P':
        app.set({ netPath: !s.netPath });
        return true;
      case 'r':
      case 'R':
        resetNetCamera();
        return true;
      case 'k':
      case 'K':
        app.set({ netK: NET_KS[(NET_KS.indexOf(s.netK as (typeof NET_KS)[number]) + 1) % NET_KS.length] });
        return true;
      case 'ArrowLeft':
      case 'ArrowRight': {
        const at = s.token < 0 ? unit.t0 : s.token + (e.key === 'ArrowLeft' ? -1 : 1);
        app.set(selectTokenPatch(s, clamp(at, unit.t0, unit.nTokens - 1)));
        return true;
      }
      case '[':
        app.set(selectUnitPatch(s, Math.max(0, s.unit - 1)));
        return true;
      case ']':
        app.set(selectUnitPatch(s, Math.min(dump.units.length - 1, s.unit + 1)));
        return true;
      default:
        return false;
    }
  });

  return (
    <>
      <canvas ref={canvas} className="scene orbit" role="img" aria-label="3D graph of the network with the rollout path of the selected token" />
      <div className="over">
        <b>
          Rollout from token {j} “{compact(tokenText(dump, unit, j))}”
        </b>
        <span>Drag to orbit · scroll to zoom · click a node to pick its token</span>
      </div>
    </>
  );
}

export function NetInspector() {
  const { dump, unit } = useUnit();
  const token = useApp((s) => s.token);
  const head = useApp((s) => s.head);
  const netK = useApp((s) => s.netK);
  const hover = useSyncExternalStore(netHover.subscribe, () => netHover.get().text);
  const net = netFor(dump, unit, token, head, netK);
  const j = selectedToken(unit, token);
  const predicted = j >= unit.t0 && j < unit.nTokens;
  return (
    <>
      <div className="big">
        {j} <span className="mono tok">{tokenText(dump, unit, j)}</span>
      </div>
      <div className="sub">{unit.label}</div>
      {predicted ? (
        <>
          <Facts
            rows={[
              ['Target probability', formatProb(unit.p[j])],
              ['Rank', <span key="rank" data-state={unit.rank[j] > 1 ? 'bad' : undefined}>{unit.rank[j]}</span>],
              ['Entropy', `${unit.entropy[j].toFixed(2)} bits`],
            ]}
          />
          <Heading>Top predictions</Heading>
          <CandidateList dump={dump} unit={unit} j={j} />
        </>
      ) : (
        <p className="note">This token is context: there is no prediction to explain. Pick a later token.</p>
      )}
      <Heading>Reading the picture</Heading>
      <div className="legend">
        <div>
          <i style={{ background: '#fff', height: 3 }} />
          Backward rollout: what the prediction leaned on
        </div>
        <div>
          <i style={{ background: 'rgba(255,255,255,0.4)' }} />
          Strongest attention links
        </div>
        <div>
          <i style={{ background: '#a2b8d4' }} />
          Cross-attention (encoder to decoder)
        </div>
        <div>
          <i style={{ background: 'var(--ok)' }} />
          The target token among the candidates
        </div>
      </div>
      <Heading>Node</Heading>
      <p className="note hover-note">{hover || 'Hover a node to read what it is.'}</p>
      <p className="sub">
        {net.nStages} stages · {net.nodes.n} nodes · {net.edges.n} edges
      </p>
    </>
  );
}

export function NetDock() {
  const netPlay = useApp((s) => s.netPlay);
  const netPath = useApp((s) => s.netPath);
  const netK = useApp((s) => s.netK);
  return (
    <>
      <div className="tb">
        <button type="button" className="tbtn play" aria-label={netPlay ? 'Pause the wave (Space)' : 'Play the wave (Space)'} title="Play or pause the wave (Space)" onClick={toggleNetPlay}>
          <Icon name={netPlay ? 'pause' : 'play'} size={16} />
        </button>
      </div>
      <button type="button" className="btn ghost" aria-pressed={netPath} title="Rollout path only (P)" onClick={() => app.set({ netPath: !netPath })}>
        Rollout only
      </button>
      <div className="dock-group">
        <span className="label">Edges per node</span>
        <Chips label="Edges per node" value={netK} onChange={(k) => app.set({ netK: k })} options={NET_KS.map((k) => ({ value: k as number, label: String(k) }))} />
      </div>
      <button type="button" className="tbtn" aria-label="Reset camera (R)" title="Reset camera (R)" onClick={resetNetCamera}>
        <Icon name="reset" size={16} />
      </button>
    </>
  );
}
