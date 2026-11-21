"""Serializes TraceOutput / DemoTraceOutput records into aligned .vtdump / .vtdemo bundles."""
from __future__ import annotations

import time
import numpy as np
import torch

from .adapters.base import DemoTraceOutput, TraceOutput
from .capture import compute_token_stats
from .container import MAGIC, Writer, save
from .spec import TraceSpec


def _pad_per_token(arr: np.ndarray, total_len: int, t0: int, fill_val) -> np.ndarray:
    shape = (total_len,) + arr.shape[1:]
    out = np.full(shape, fill_val, dtype=arr.dtype)
    n = min(len(arr), max(0, total_len - t0))
    if n > 0:
        out[t0:t0 + n] = arr[:n]
    return out


def _vocab_block(writer: Writer, spec: TraceSpec, vocab_size: int) -> dict:
    if spec.token_meta is not None:
        labels = list(spec.token_meta.labels)
        if len(labels) < vocab_size:
            labels.extend(f't{i}' for i in range(len(labels), vocab_size))
        tag_names = list(spec.token_meta.tag_names)
        values = spec.token_meta.values
    else:
        labels = [f't{i}' for i in range(vocab_size)]
        tag_names = []
        values = None
    out = {'labels': labels}
    if tag_names:
        out['tag_names'] = tag_names
    if values is not None:
        out['values'] = writer.add(np.asarray(values, np.float32), 'f4')
    return out


def export_dump(path: str, trace: TraceOutput | list[TraceOutput], spec: TraceSpec | None = None,
                free_tokens: list[np.ndarray] | None = None,
                free_probs: list[np.ndarray] | None = None,
                step: int | None = None, device: str = 'cpu', precision: str = 'fp32') -> str:
    """Packs one or more forward TraceOutputs + optional free rollout & user spans into a .vtdump file."""
    spec = spec or TraceSpec()
    traces = [trace] if isinstance(trace, TraceOutput) else list(trace)
    if not traces:
        raise ValueError('export_dump requires at least one TraceOutput')
    vocab_size = int(traces[0].logits.shape[-1])
    w = Writer()

    value_ids = None
    if spec.token_meta is not None and spec.token_meta.values is not None:
        vals = np.asarray(spec.token_meta.values, np.float32)
        idx = np.flatnonzero(np.isfinite(vals) & (vals >= 0))
        if len(idx):
            value_ids = torch.from_numpy(idx)

    unit_recs = []
    for ui, tr in enumerate(traces):
        tok = np.asarray(tr.tok, np.int32)
        T = len(tok)
        t0 = int(tr.t0)
        targets = torch.from_numpy(tok[t0:].astype(np.int64))
        rows = tr.logits[t0 - 1:T - 1]
        if len(rows) != len(targets):
            raise ValueError(f'logits rows ({len(rows)}) != targets ({len(targets)})')
        st = compute_token_stats(rows, targets, top_k=spec.top_k, value_ids=value_ids)
        bits = float(st['bits'].sum())
        forced_exact = bool(st['right'].all())

        seq_block = {
            'tokens': w.add(tok, 'i4'),
            'p': w.add(_pad_per_token(st['p'], T, t0, np.nan), 'f4'),
            'rank': w.add(_pad_per_token(st['rank'], T, t0, 0), 'i4'),
            'top_id': w.add(_pad_per_token(st['top_id'], T, t0, -1), 'i4'),
            'top_p': w.add(_pad_per_token(st['top_p'], T, t0, 0.0), 'f4'),
            'entropy': w.add(_pad_per_token(st['entropy'], T, t0, 0.0), 'f4'),
            'n_allowed': w.add(_pad_per_token(st['n_allowed'], T, t0, vocab_size), 'i4'),
            'right': w.add(_pad_per_token(st['right'], T, t0, False), 'u1'),
        }
        if 'val_mean' in st:
            seq_block['val_mean'] = w.add(_pad_per_token(st['val_mean'], T, t0, np.nan), 'f4')
            seq_block['val_std'] = w.add(_pad_per_token(st['val_std'], T, t0, np.nan), 'f4')
        if tr.src_tok is not None:
            seq_block['source_tokens'] = w.add(np.asarray(tr.src_tok, np.int32), 'i4')
        if spec.token_meta is not None and spec.token_meta.token_tags is not None:
            seq_block['token_tags'] = w.add(np.asarray(spec.token_meta.token_tags[:T], np.int32), 'i4')

        div = -1
        free_exact = False
        if free_tokens is not None and ui < len(free_tokens):
            ft = np.asarray(free_tokens[ui], np.int32)
            fp = np.asarray(free_probs[ui], np.float32) if free_probs is not None else np.ones(len(ft), np.float32)
            seq_block['free_tok'] = w.add(ft, 'i4')
            seq_block['free_p'] = w.add(fp, 'f4')
            for j in range(t0, min(T, len(ft))):
                if ft[j] != tok[j]:
                    div = j
                    break
            free_exact = div < 0 and len(ft) == T

        stages_json = []
        for sspec, act in tr.stages:
            stages_json.append({
                'name': sspec.name, 'label': sspec.label, 'group': sspec.group, 'kind': sspec.kind,
                'count': int(sspec.count), 'token_source': sspec.token_source, 'token_offset': int(sspec.token_offset),
                'span_y': float(sspec.span_y), 'z_jitter': float(sspec.z_jitter), 'x_jitter': float(sspec.x_jitter),
                'gap_before': float(sspec.gap_before),
                'act': w.add(np.asarray(act, np.float32), 'f4') if act is not None else None,
            })

        links_json = []
        for lspec, wt in tr.links:
            links_json.append({
                'src': int(lspec.src), 'dst': int(lspec.dst), 'name': lspec.name, 'family': lspec.family,
                'kind': lspec.kind, 'tint': lspec.tint, 'layer_idx': int(lspec.layer_idx),
                'row_offset': int(lspec.row_offset), 'causal': bool(lspec.causal),
                'weights': w.add(np.asarray(wt, np.float16), 'f2') if wt is not None else None,
            })

        norms_json = None
        if tr.norm_data is not None:
            norms_json = {
                'name': tr.norm_name,
                'layers': list(tr.norm_layers),
                'components': list(tr.norm_components),
                'data': w.add(np.asarray(tr.norm_data, np.float32), 'f4'),
                'row_offset': 0,
            }

        latents_json = []
        for lm in tr.latents:
            latents_json.append({
                'name': lm['name'],
                'row_label': lm.get('row_label', 'Row'),
                'col_label': lm.get('col_label', 'Dim'),
                'data': w.add(np.asarray(lm['data'], np.float32), 'f4'),
                'norms': w.add(np.asarray(lm['norms'], np.float32), 'f4') if lm.get('norms') is not None else None,
            })

        unit_recs.append({
            'u': ui,
            'label': spec.unit_label if len(traces) == 1 else f'{spec.unit_label} {ui}',
            'kind': tr.arch,
            't0': t0,
            'bits': bits,
            'forced_exact': forced_exact,
            'free_exact': free_exact,
            'free_ended': True,
            'diverge': div,
            'anomalies': 0,
            'check_logit_diff': float(tr.check_logit_diff),
            'sequence': seq_block,
            'stages': stages_json,
            'links': links_json,
            'norms': norms_json,
            'latents': latents_json,
        })

    header = {
        'kind': 'dump',
        'tool': 'vitro',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'trace': {
            'id': spec.trace_id,
            'title': spec.title or spec.trace_id,
            'model_name': spec.model_name or traces[0].model_name,
            'arch': traces[0].arch,
            'precision': precision,
            'device': device,
            'step': step,
            'top_k': spec.top_k,
            'delta_red_at': spec.delta_red_at,
        },
        'vocab': _vocab_block(w, spec, vocab_size),
        'spans': [sp.to_json(i) for i, sp in enumerate(spec.spans)],
        'units': unit_recs,
    }
    save(path, header, w, magic=MAGIC)
    return path


def export_demo(path: str, segment: DemoTraceOutput | list[DemoTraceOutput], spec: TraceSpec | None = None,
                vocab_size: int | None = None, precision: str = 'fp32', step: int | None = None) -> str:
    """Packs one or more DemoTraceOutputs into an aligned .vtdemo recording."""
    spec = spec or TraceSpec()
    segs = [segment] if isinstance(segment, DemoTraceOutput) else list(segment)
    if not segs:
        raise ValueError('export_demo requires at least one DemoTraceOutput')
    inferred_v = int(max(int(segs[0].ftok.max()) + 1, int(segs[0].top_id.max()) + 1, 16))
    vsize = vocab_size or (len(spec.token_meta.labels) if spec.token_meta else inferred_v)
    w = Writer()

    seg_recs = []
    for k, sg in enumerate(segs):
        stages_json = []
        for sspec, act in sg.stages:
            stages_json.append({
                'name': sspec.name, 'label': sspec.label, 'group': sspec.group, 'kind': sspec.kind,
                'count': int(sspec.count), 'token_source': sspec.token_source, 'token_offset': int(sspec.token_offset),
                'span_y': float(sspec.span_y), 'z_jitter': float(sspec.z_jitter), 'x_jitter': float(sspec.x_jitter),
                'gap_before': float(sspec.gap_before),
                'act': w.add(np.asarray(act, np.float32), 'f4') if act is not None else None,
            })
        prefill_json = []
        for pl in sg.prefill_links:
            prefill_json.append({
                'src': int(pl['src']), 'dst': int(pl['dst']), 'kind': pl['kind'], 'tint': pl['tint'],
                'ids': w.add(np.asarray(pl['ids'], np.int32), 'i4'),
                'weights': w.add(np.asarray(pl['weights'], np.float16), 'f2'),
            })
        tape_json = {
            'top_id': w.add(np.asarray(sg.top_id, np.int32), 'i4'),
            'top_p': w.add(np.asarray(sg.top_p, np.float32), 'f4'),
            'entropy': w.add(np.asarray(sg.entropy, np.float32), 'f4'),
            'p_ref': w.add(np.asarray(sg.p_ref, np.float32), 'f4') if sg.p_ref is not None else None,
            'self_id': w.add(np.asarray(sg.self_id, np.int32), 'i4'),
            'self_w': w.add(np.asarray(sg.self_w, np.float16), 'f2'),
            'cross_id': w.add(np.asarray(sg.cross_id, np.int32), 'i4') if sg.cross_id is not None else None,
            'cross_w': w.add(np.asarray(sg.cross_w, np.float16), 'f2') if sg.cross_w is not None else None,
            'norm': w.add(np.asarray(sg.norm, np.float16), 'f2'),
        }
        seg_recs.append({
            'k': k,
            'label': sg.label,
            'kind': sg.kind,
            't0': int(sg.t0),
            'diverge': int(sg.diverge),
            'free_exact': bool(sg.free_exact),
            'free_ended': bool(sg.free_ended),
            'has_read': bool(sg.has_read),
            'read_ticks': float(sg.read_ticks),
            'source_tokens': w.add(np.asarray(sg.src_tok, np.int32), 'i4') if sg.src_tok is not None else None,
            'ref_tokens': w.add(np.asarray(sg.ref_tok, np.int32), 'i4') if sg.ref_tok is not None else None,
            'free_tok': w.add(np.asarray(sg.ftok, np.int32), 'i4'),
            'free_p': w.add(np.asarray(sg.fp, np.float32), 'f4'),
            'stages': stages_json,
            'prefill_links': prefill_json,
            'decode': {
                'in_stage': int(sg.in_stage),
                'out_stage': int(sg.out_stage),
                'cross_stage': int(sg.cross_stage),
                'layer_stages': [int(x) for x in sg.layer_stages],
            },
            'tape': tape_json,
        })

    header = {
        'kind': 'demo',
        'tool': 'vitro',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'trace': {
            'id': spec.trace_id,
            'title': spec.title or spec.trace_id,
            'model_name': spec.model_name or segs[0].model_name,
            'arch': segs[0].arch,
            'precision': precision,
            'step': step,
            'top_k': spec.top_k,
        },
        'vocab': _vocab_block(w, spec, vsize),
        'spans': [sp.to_json(i) for i, sp in enumerate(spec.spans)],
        'segments': seg_recs,
    }
    save(path, header, w, magic=MAGIC)
    return path
