"""Universal PyTorch adapters for Decoder-Only, Encoder-Decoder, and HuggingFace transformers."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..capture import HookSession, extract_tensor, no_amp, top_k_edges
from ..spec import LinkSpec, StageSpec, TraceSpec
from .base import DemoTraceOutput, TraceOutput


def _find_attr_module_list(root: nn.Module, candidates: tuple[str, ...]) -> list[nn.Module] | None:
    for path in candidates:
        cur = root
        ok = True
        for part in path.split('.'):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
            return list(cur)
    return None


def _find_attn_submodule(block: nn.Module, preferred: tuple[str, ...] = ('self_attn', 'attn', 'attention')) -> nn.Module | None:
    for name in preferred:
        if hasattr(block, name) and isinstance(getattr(block, name), nn.Module):
            return getattr(block, name)
    for mod in block.modules():
        if isinstance(mod, nn.MultiheadAttention):
            return mod
    return None


def _find_mlp_submodule(block: nn.Module) -> nn.Module | None:
    for name in ('mlp', 'ffn', 'feed_forward', 'linear2'):
        if hasattr(block, name) and isinstance(getattr(block, name), nn.Module):
            return getattr(block, name)
    return None


def _avg_heads(w: np.ndarray) -> np.ndarray:
    return w.mean(0) if w.ndim == 3 else w


def _build_norm_array(layers_n: int, rec: dict[str, torch.Tensor], keys_per_layer: list[list[tuple[int, str]]],
                      positions: int) -> np.ndarray:
    n_comp = max((max(idx for idx, _ in row) for row in keys_per_layer), default=0) + 1
    out = np.zeros((layers_n, n_comp, positions), np.float32)
    for li, row in enumerate(keys_per_layer):
        for ci, key in row:
            if key in rec:
                arr = rec[key][0].numpy()
                m = min(positions, arr.shape[-1])
                out[li, ci, :m] = arr[:m]
    return out


class DecoderOnlyAdapter:
    """Traces any causal PyTorch decoder/transformer model with user-customizable stage & link labels."""

    def __init__(self, model: nn.Module, blocks: list[nn.Module] | None = None,
                 attn_attr: str | None = None, mlp_attr: str | None = None,
                 forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 model_name: str = '') -> None:
        self.model = model
        self.blocks = list(blocks) if blocks is not None else _find_attr_module_list(
            model, ('blocks', 'layers', 'h', 'decoder.layers', 'transformer.h', 'model.layers', 'body.layers')
        )
        if not self.blocks:
            raise ValueError('could not auto-discover transformer blocks; pass blocks=[...] explicitly')
        self.attn_attr = attn_attr
        self.mlp_attr = mlp_attr
        self.forward_fn = forward_fn or (lambda tok: extract_tensor(model(tok)))
        self.model_name = model_name or model.__class__.__name__

    def _resolve_block(self, block: nn.Module) -> tuple[nn.Module, nn.Module | None]:
        attn = getattr(block, self.attn_attr) if self.attn_attr else _find_attn_submodule(block)
        if attn is None:
            raise ValueError(f'no attention submodule found on {block.__class__.__name__}')
        mlp = getattr(block, self.mlp_attr) if self.mlp_attr else _find_mlp_submodule(block)
        return attn, mlp

    def _install(self, sess: HookSession) -> None:
        for i, blk in enumerate(self.blocks):
            attn, mlp = self._resolve_block(blk)
            sess.watch_attention(f'self.{i}', attn)
            sess.watch_norm(f'h.{i}', blk)
            sess.watch_norm(f'sa.{i}', attn)
            if mlp is not None:
                sess.watch_norm(f'mlp.{i}', mlp)

    @torch.no_grad()
    def trace_forward(self, tokens: torch.Tensor, spec: TraceSpec | None = None,
                      t0: int = 1, amp=no_amp) -> TraceOutput:
        spec = spec or TraceSpec()
        tok = tokens.detach()
        if tok.ndim == 1:
            tok = tok[None]
        inp = tok[:, :-1]
        P = int(inp.shape[1])

        with amp():
            logits_plain = self.forward_fn(inp)
        with HookSession(self._install) as sess, amp():
            logits_hook = self.forward_fn(inp)
        diff = float((logits_plain.float() - logits_hook.float()).abs().max())

        L = len(self.blocks)
        has_mlp = all(f'mlp.{i}' in sess.rec for i in range(L))
        comps = [spec.label_for_comp('resid', 'Residual'), spec.label_for_comp('self', 'Self-Attn')]
        if has_mlp:
            comps.append(spec.label_for_comp('mlp', 'MLP'))

        norm_keys = []
        for i in range(L):
            row = [(0, f'h.{i}'), (1, f'sa.{i}')]
            if has_mlp:
                row.append((2, f'mlp.{i}'))
            norm_keys.append(row)
        norm_data = _build_norm_array(L, sess.rec, norm_keys, P)

        in_group = spec.label_for_group('input', 'Input')
        dec_group = spec.label_for_group('decoder', 'Transformer')
        stages: list[tuple[StageSpec, np.ndarray | None]] = [
            (StageSpec('input', spec.label_for_stage('input', 'Tokens'), in_group, 'input', P, 'target', 0),
             np.full((P,), 0.45, np.float32))
        ]
        for i in range(L):
            st_name = f'block_{i}'
            st_label = spec.label_for_stage(st_name, f'Block {i + 1}')
            act = norm_data[i, 0].copy()
            stages.append((StageSpec(st_name, st_label, dec_group, 'layer', P, 'target', 0), act))

        fam_name = spec.label_for_attn('self', 'Causal Self-Attn')
        links: list[tuple[LinkSpec, np.ndarray | None]] = []
        for i in range(L):
            src, dst = i, i + 1
            links.append((LinkSpec(src, dst, f'Residual L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
            w = sess.rec[f'self.{i}'][0].numpy().astype(np.float16)
            links.append((LinkSpec(src, dst, f'{fam_name} L{i + 1}', fam_name, 'self', 'violet', i, 0, True), w))

        return TraceOutput(
            arch='decoder_only',
            model_name=spec.model_name or self.model_name,
            tok=tok[0].detach().cpu().numpy().astype(np.int32),
            logits=logits_plain[0].detach().float().cpu(),
            t0=t0,
            stages=stages,
            links=links,
            norm_name=f'{spec.model_name or self.model_name} Layer Norms',
            norm_layers=[spec.label_for_stage(f'block_{i}', f'Block {i + 1}') for i in range(L)],
            norm_components=comps,
            norm_data=norm_data,
            check_logit_diff=diff,
        )

    @torch.no_grad()
    def record_generation(self, prompt: torch.Tensor, max_new_tokens: int, eos_id: int | None = None,
                          ref_tokens: torch.Tensor | None = None, spec: TraceSpec | None = None,
                          top_k: int = 5, edge_k: int = 4, label: str = 'Generation', amp=no_amp) -> DemoTraceOutput:
        spec = spec or TraceSpec()
        dev = next(self.model.parameters()).device
        seq = prompt.detach().to(dev).reshape(1, -1)
        t0 = int(seq.shape[1])
        probs = [1.0] * t0
        L = len(self.blocks)
        top_ids, top_ps, ents = [], [], []
        self_ids, self_ws, norms = [], [], []
        ended = False

        for _ in range(max_new_tokens):
            with HookSession(self._install) as sess, amp():
                logits = self.forward_fn(seq)[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            kt = min(top_k, p.numel())
            tlp, tid = lp.topk(kt)
            tp = tlp.exp().cpu().numpy().astype(np.float32)
            tid_np = tid.cpu().numpy().astype(np.int32)
            if kt < top_k:
                tid_np = np.pad(tid_np, (0, top_k - kt), constant_values=-1)
                tp = np.pad(tp, (0, top_k - kt), constant_values=0.0)
            ent = float((-(p * lp).where(p > 0, torch.zeros_like(lp))).sum().item() / math.log(2))

            nxt = int(tid_np[0])
            top_ids.append(tid_np)
            top_ps.append(tp)
            ents.append(ent)
            probs.append(float(tp[0]))

            cur_len = int(seq.shape[1])
            s_id = np.full((L, edge_k), -1, np.int32)
            s_w = np.zeros((L, edge_k), np.float16)
            nm = np.zeros((L, 3), np.float16)
            for li in range(L):
                w_last = _avg_heads(sess.rec[f'self.{li}'][0].numpy())[-1]  # (cur_len,)
                kk = min(edge_k, len(w_last))
                top_src = np.argsort(-w_last)[:kk]
                s_id[li, :kk] = top_src.astype(np.int32)
                s_w[li, :kk] = w_last[top_src].astype(np.float16)
                nm[li, 0] = float(sess.rec[f'h.{li}'][0, -1])
                nm[li, 1] = float(sess.rec[f'sa.{li}'][0, -1])
                if f'mlp.{li}' in sess.rec:
                    nm[li, 2] = float(sess.rec[f'mlp.{li}'][0, -1])
            self_ids.append(s_id)
            self_ws.append(s_w)
            norms.append(nm)

            seq = torch.cat([seq, torch.tensor([[nxt]], device=dev, dtype=seq.dtype)], dim=1)
            if eos_id is not None and nxt == eos_id:
                ended = True
                break

        ftok = seq[0].detach().cpu().numpy().astype(np.int32)
        M = len(ftok)
        steps_n = M - t0
        full_top_id = np.full((M, top_k), -1, np.int32)
        full_top_p = np.zeros((M, top_k), np.float32)
        full_ent = np.zeros((M,), np.float32)
        full_sid = np.full((M, L, edge_k), -1, np.int32)
        full_sw = np.zeros((M, L, edge_k), np.float16)
        full_nm = np.zeros((M, L, 3), np.float16)
        full_top_id[t0:] = np.stack(top_ids)
        full_top_p[t0:] = np.stack(top_ps)
        full_ent[t0:] = np.asarray(ents, np.float32)
        full_sid[t0:] = np.stack(self_ids)
        full_sw[t0:] = np.stack(self_ws)
        full_nm[t0:] = np.stack(norms)

        in_g = spec.label_for_group('input', 'Tokens')
        dec_g = spec.label_for_group('decoder', 'Transformer')
        stages: list[tuple[StageSpec, np.ndarray | None]] = [
            (StageSpec('input', spec.label_for_stage('input', 'Input'), in_g, 'input', M - 1, 'target', 0), None)
        ]
        for li in range(L):
            stages.append((StageSpec(f'block_{li}', spec.label_for_stage(f'block_{li}', f'Block {li + 1}'), dec_g, 'layer', M - 1, 'target', 0), None))
        stages.append((StageSpec('output', 'Candidates', 'Output', 'output', top_k, 'none', 0, span_y=90.0, z_jitter=0.0, x_jitter=0.0, gap_before=20.0), None))

        ref_np = ref_tokens.detach().cpu().numpy().astype(np.int32) if ref_tokens is not None else None
        div = -1
        exact = False
        if ref_np is not None:
            for j in range(t0, min(M, len(ref_np))):
                if ftok[j] != ref_np[j]:
                    div = j
                    break
            exact = div < 0 and M == len(ref_np)

        return DemoTraceOutput(
            arch='decoder_only',
            model_name=spec.model_name or self.model_name,
            label=label,
            kind='causal',
            t0=t0,
            ftok=ftok,
            fp=np.asarray(probs[:M], np.float32),
            stages=stages,
            in_stage=0,
            out_stage=len(stages) - 1,
            layer_stages=list(range(1, L + 1)),
            top_id=full_top_id,
            top_p=full_top_p,
            entropy=full_ent,
            src_tok=None,
            ref_tok=ref_np,
            self_id=full_sid,
            self_w=full_sw,
            norm=full_nm,
            has_read=False,
            diverge=div,
            free_exact=exact,
            free_ended=ended or steps_n == max_new_tokens,
        )


class EncoderDecoderAdapter:
    """Traces any PyTorch seq2seq encoder-decoder model with cross-attention & dual stages."""

    def __init__(self, model: nn.Module, enc_layers: list[nn.Module] | None = None,
                 dec_layers: list[nn.Module] | None = None,
                 forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 model_name: str = '') -> None:
        self.model = model
        self.enc_layers = list(enc_layers) if enc_layers is not None else _find_attr_module_list(
            model, ('encoder.layers', 'enc.layers', 'encoder.blocks')
        )
        self.dec_layers = list(dec_layers) if dec_layers is not None else _find_attr_module_list(
            model, ('decoder.layers', 'dec.layers', 'decoder.blocks')
        )
        if not self.enc_layers or not self.dec_layers:
            raise ValueError('could not auto-discover encoder & decoder layers; pass enc_layers and dec_layers')
        self.forward_fn = forward_fn or (lambda src, tgt: extract_tensor(model(src, tgt)))
        self.model_name = model_name or model.__class__.__name__

    def _install(self, sess: HookSession) -> None:
        for i, blk in enumerate(self.enc_layers):
            sa = _find_attn_submodule(blk, ('self_attn', 'attn'))
            if sa is None:
                raise ValueError(f'encoder block {i} has no self-attention submodule')
            sess.watch_attention(f'enc_self.{i}', sa)
            sess.watch_norm(f'enc_h.{i}', blk)
        for i, blk in enumerate(self.dec_layers):
            sa = _find_attn_submodule(blk, ('self_attn', 'attn'))
            ca = _find_attn_submodule(blk, ('multihead_attn', 'cross_attn', 'encoder_attn'))
            if sa is None or ca is None:
                raise ValueError(f'decoder block {i} needs self-attn and cross-attn submodules')
            sess.watch_attention(f'dec_self.{i}', sa)
            sess.watch_attention(f'dec_cross.{i}', ca)
            sess.watch_norm(f'dec_h.{i}', blk)
            sess.watch_norm(f'dec_sa.{i}', sa)
            sess.watch_norm(f'dec_ca.{i}', ca)
            mlp = _find_mlp_submodule(blk)
            if mlp is not None:
                sess.watch_norm(f'dec_mlp.{i}', mlp)

    @torch.no_grad()
    def trace_forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor,
                      spec: TraceSpec | None = None, t0: int = 1, amp=no_amp) -> TraceOutput:
        spec = spec or TraceSpec()
        src = src_tokens.detach()
        tgt = tgt_tokens.detach()
        if src.ndim == 1:
            src = src[None]
        if tgt.ndim == 1:
            tgt = tgt[None]
        inp_tgt = tgt[:, :-1]
        E = int(src.shape[1])
        P = int(inp_tgt.shape[1])

        with amp():
            logits_plain = self.forward_fn(src, inp_tgt)
        with HookSession(self._install) as sess, amp():
            logits_hook = self.forward_fn(src, inp_tgt)
        diff = float((logits_plain.float() - logits_hook.float()).abs().max())

        Le, Ld = len(self.enc_layers), len(self.dec_layers)
        has_mlp = all(f'dec_mlp.{i}' in sess.rec for i in range(Ld))
        comps = [spec.label_for_comp('resid', 'Residual'), spec.label_for_comp('self', 'Self-Attn'), spec.label_for_comp('cross', 'Cross-Attn')]
        if has_mlp:
            comps.append(spec.label_for_comp('mlp', 'MLP'))
        norm_keys = []
        for i in range(Ld):
            row = [(0, f'dec_h.{i}'), (1, f'dec_sa.{i}'), (2, f'dec_ca.{i}')]
            if has_mlp:
                row.append((3, f'dec_mlp.{i}'))
            norm_keys.append(row)
        norm_data = _build_norm_array(Ld, sess.rec, norm_keys, P)

        src_g = spec.label_for_group('source', 'Source')
        enc_g = spec.label_for_group('encoder', 'Encoder')
        din_g = spec.label_for_group('dec_in', 'Target In')
        dec_g = spec.label_for_group('decoder', 'Decoder')

        stages: list[tuple[StageSpec, np.ndarray | None]] = [
            (StageSpec('src_in', spec.label_for_stage('src_in', 'Source'), src_g, 'input', E, 'source', 0), np.full((E,), 0.45, np.float32))
        ]
        for i in range(Le):
            act = sess.rec[f'enc_h.{i}'][0].numpy().astype(np.float32)
            stages.append((StageSpec(f'enc_{i}', spec.label_for_stage(f'enc_{i}', f'Encoder {i + 1}'), enc_g, 'layer', E, 'source', 0), act))
        enc_last_stage = len(stages) - 1
        dec_in_stage = len(stages)
        stages.append((StageSpec('dec_in', spec.label_for_stage('dec_in', 'Dec Input'), din_g, 'input', P, 'target', 0, gap_before=28.0),
                       np.full((P,), 0.35, np.float32)))
        for i in range(Ld):
            act = norm_data[i, 0].copy()
            stages.append((StageSpec(f'dec_{i}', spec.label_for_stage(f'dec_{i}', f'Decoder {i + 1}'), dec_g, 'layer', P, 'target', 0), act))

        efam = spec.label_for_attn('enc_self', 'Encoder Self-Attn')
        sfam = spec.label_for_attn('dec_self', 'Decoder Self-Attn')
        cfam = spec.label_for_attn('dec_cross', 'Encoder-Decoder Cross-Attn')
        links: list[tuple[LinkSpec, np.ndarray | None]] = []
        for i in range(Le):
            links.append((LinkSpec(i, i + 1, f'Enc Resid L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
            links.append((LinkSpec(i, i + 1, f'{efam} L{i + 1}', efam, 'self', 'blue', i, 0, False),
                          sess.rec[f'enc_self.{i}'][0].numpy().astype(np.float16)))
        for i in range(Ld):
            src_st = dec_in_stage + i
            dst_st = dec_in_stage + 1 + i
            links.append((LinkSpec(src_st, dst_st, f'Dec Resid L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
            links.append((LinkSpec(src_st, dst_st, f'{sfam} L{i + 1}', sfam, 'self', 'violet', i, 0, True),
                          sess.rec[f'dec_self.{i}'][0].numpy().astype(np.float16)))
            links.append((LinkSpec(enc_last_stage, dst_st, f'{cfam} L{i + 1}', cfam, 'cross', 'magenta', i, 0, False),
                          sess.rec[f'dec_cross.{i}'][0].numpy().astype(np.float16)))

        return TraceOutput(
            arch='encoder_decoder',
            model_name=spec.model_name or self.model_name,
            tok=tgt[0].detach().cpu().numpy().astype(np.int32),
            src_tok=src[0].detach().cpu().numpy().astype(np.int32),
            logits=logits_plain[0].detach().float().cpu(),
            t0=t0,
            stages=stages,
            links=links,
            norm_name=f'{spec.model_name or self.model_name} Decoder Norms',
            norm_layers=[spec.label_for_stage(f'dec_{i}', f'Decoder {i + 1}') for i in range(Ld)],
            norm_components=comps,
            norm_data=norm_data,
            check_logit_diff=diff,
        )

    @torch.no_grad()
    def record_generation(self, src_tokens: torch.Tensor, start_tgt: torch.Tensor, max_new_tokens: int,
                          eos_id: int | None = None, ref_tokens: torch.Tensor | None = None,
                          spec: TraceSpec | None = None, top_k: int = 5, edge_k: int = 4,
                          label: str = 'Seq2Seq Generation', amp=no_amp) -> DemoTraceOutput:
        spec = spec or TraceSpec()
        dev = next(self.model.parameters()).device
        src = src_tokens.detach().to(dev).reshape(1, -1)
        seq = start_tgt.detach().to(dev).reshape(1, -1)
        E = int(src.shape[1])
        t0 = int(seq.shape[1])
        probs = [1.0] * t0
        Le, Ld = len(self.enc_layers), len(self.dec_layers)
        top_ids, top_ps, ents = [], [], []
        self_ids, self_ws, cross_ids, cross_ws, norms = [], [], [], [], []
        first_sess: dict[str, torch.Tensor] | None = None
        ended = False

        for _ in range(max_new_tokens):
            with HookSession(self._install) as sess, amp():
                logits = self.forward_fn(src, seq)[0, -1].float()
            if first_sess is None:
                first_sess = {k: v for k, v in sess.rec.items() if k.startswith('enc_')}
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            kt = min(top_k, p.numel())
            tlp, tid = lp.topk(kt)
            tp = tlp.exp().cpu().numpy().astype(np.float32)
            tid_np = tid.cpu().numpy().astype(np.int32)
            if kt < top_k:
                tid_np = np.pad(tid_np, (0, top_k - kt), constant_values=-1)
                tp = np.pad(tp, (0, top_k - kt), constant_values=0.0)
            ent = float((-(p * lp).where(p > 0, torch.zeros_like(lp))).sum().item() / math.log(2))

            nxt = int(tid_np[0])
            top_ids.append(tid_np)
            top_ps.append(tp)
            ents.append(ent)
            probs.append(float(tp[0]))

            cur_len = int(seq.shape[1])
            s_id = np.full((Ld, edge_k), -1, np.int32)
            s_w = np.zeros((Ld, edge_k), np.float16)
            c_id = np.full((Ld, edge_k), -1, np.int32)
            c_w = np.zeros((Ld, edge_k), np.float16)
            nm = np.zeros((Ld, 3), np.float16)
            for li in range(Ld):
                wl = _avg_heads(sess.rec[f'dec_self.{li}'][0].numpy())[-1]
                kks = min(edge_k, len(wl))
                ts = np.argsort(-wl)[:kks]
                s_id[li, :kks] = ts.astype(np.int32)
                s_w[li, :kks] = wl[ts].astype(np.float16)

                wc = _avg_heads(sess.rec[f'dec_cross.{li}'][0].numpy())[-1]
                kkc = min(edge_k, len(wc))
                tc = np.argsort(-wc)[:kkc]
                c_id[li, :kkc] = tc.astype(np.int32)
                c_w[li, :kkc] = wc[tc].astype(np.float16)

                nm[li, 0] = float(sess.rec[f'dec_h.{li}'][0, -1])
                nm[li, 1] = float(sess.rec[f'dec_sa.{li}'][0, -1])
                nm[li, 2] = float(sess.rec[f'dec_ca.{li}'][0, -1])
            self_ids.append(s_id); self_ws.append(s_w)
            cross_ids.append(c_id); cross_ws.append(c_w)
            norms.append(nm)

            seq = torch.cat([seq, torch.tensor([[nxt]], device=dev, dtype=seq.dtype)], dim=1)
            if eos_id is not None and nxt == eos_id:
                ended = True
                break

        ftok = seq[0].detach().cpu().numpy().astype(np.int32)
        M = len(ftok)
        full_top_id = np.full((M, top_k), -1, np.int32)
        full_top_p = np.zeros((M, top_k), np.float32)
        full_ent = np.zeros((M,), np.float32)
        full_sid = np.full((M, Ld, edge_k), -1, np.int32)
        full_sw = np.zeros((M, Ld, edge_k), np.float16)
        full_cid = np.full((M, Ld, edge_k), -1, np.int32)
        full_cw = np.zeros((M, Ld, edge_k), np.float16)
        full_nm = np.zeros((M, Ld, 3), np.float16)
        full_top_id[t0:] = np.stack(top_ids); full_top_p[t0:] = np.stack(top_ps); full_ent[t0:] = np.asarray(ents, np.float32)
        full_sid[t0:] = np.stack(self_ids); full_sw[t0:] = np.stack(self_ws)
        full_cid[t0:] = np.stack(cross_ids); full_cw[t0:] = np.stack(cross_ws); full_nm[t0:] = np.stack(norms)

        src_g = spec.label_for_group('source', 'Source')
        enc_g = spec.label_for_group('encoder', 'Encoder')
        din_g = spec.label_for_group('dec_in', 'Dec In')
        dec_g = spec.label_for_group('decoder', 'Decoder')
        stages: list[tuple[StageSpec, np.ndarray | None]] = [
            (StageSpec('src_in', spec.label_for_stage('src_in', 'Source'), src_g, 'input', E, 'source', 0), np.full((E,), 0.4, np.float32))
        ]
        for li in range(Le):
            act = first_sess[f'enc_h.{li}'][0].numpy().astype(np.float32) if first_sess else None
            stages.append((StageSpec(f'enc_{li}', spec.label_for_stage(f'enc_{li}', f'Encoder {li + 1}'), enc_g, 'layer', E, 'source', 0), act))
        enc_last = len(stages) - 1
        in_stage = len(stages)
        stages.append((StageSpec('dec_in', spec.label_for_stage('dec_in', 'Dec Input'), din_g, 'input', M - 1, 'target', 0, gap_before=28.0), None))
        for li in range(Ld):
            stages.append((StageSpec(f'dec_{li}', spec.label_for_stage(f'dec_{li}', f'Decoder {li + 1}'), dec_g, 'layer', M - 1, 'target', 0), None))
        out_stage = len(stages)
        stages.append((StageSpec('output', 'Candidates', 'Output', 'output', top_k, 'none', 0, span_y=90.0, z_jitter=0.0, x_jitter=0.0, gap_before=20.0), None))

        prefill_links = []
        if first_sess is not None:
            for li in range(Le):
                w = _avg_heads(first_sess[f'enc_self.{li}'][0].numpy())
                idx, vals = top_k_edges(w, min(3, E))
                prefill_links.append({'src': li, 'dst': li + 1, 'kind': 'self', 'tint': 'blue', 'ids': idx, 'weights': vals})

        ref_np = ref_tokens.detach().cpu().numpy().astype(np.int32) if ref_tokens is not None else None
        div = -1
        exact = False
        if ref_np is not None:
            for j in range(t0, min(M, len(ref_np))):
                if ftok[j] != ref_np[j]:
                    div = j
                    break
            exact = div < 0 and M == len(ref_np)

        return DemoTraceOutput(
            arch='encoder_decoder',
            model_name=spec.model_name or self.model_name,
            label=label,
            kind='seq2seq',
            t0=t0,
            ftok=ftok,
            fp=np.asarray(probs[:M], np.float32),
            stages=stages,
            in_stage=in_stage,
            out_stage=out_stage,
            layer_stages=list(range(in_stage + 1, out_stage)),
            top_id=full_top_id,
            top_p=full_top_p,
            entropy=full_ent,
            src_tok=src[0].detach().cpu().numpy().astype(np.int32),
            ref_tok=ref_np,
            self_id=full_sid,
            self_w=full_sw,
            norm=full_nm,
            has_read=True,
            cross_stage=enc_last,
            cross_id=full_cid,
            cross_w=full_cw,
            prefill_links=prefill_links,
            diverge=div,
            free_exact=exact,
            free_ended=ended or (M - t0) == max_new_tokens,
        )


def _ensure_eager_attention(model: nn.Module) -> None:
    """Switches HuggingFace models (including encoder/decoder subconfigs) to eager attention so weights can be captured."""
    if hasattr(model, 'set_attn_implementation'):
        try:
            model.set_attn_implementation('eager')
        except Exception:
            pass
    for mod in model.modules():
        cfg = getattr(mod, 'config', None)
        if cfg is not None and hasattr(cfg, '_attn_implementation'):
            cfg._attn_implementation = 'eager'


class HuggingFaceAdapter:
    """Traces HuggingFace causal or seq2seq models offline via ``output_attentions=True``."""

    def __init__(self, model: nn.Module, model_name: str = '') -> None:
        self.model = model
        self.is_enc_dec = bool(getattr(getattr(model, 'config', None), 'is_encoder_decoder', False))
        self.model_name = model_name or getattr(getattr(model, 'config', None), 'model_type', '') or model.__class__.__name__
        _ensure_eager_attention(self.model)

    @torch.no_grad()
    def trace_forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor | None = None,
                      spec: TraceSpec | None = None, t0: int = 1, amp=no_amp) -> TraceOutput:
        spec = spec or TraceSpec()
        _ensure_eager_attention(self.model)
        tok = (decoder_input_ids if decoder_input_ids is not None else input_ids).detach()
        if tok.ndim == 1:
            tok = tok[None]
        src = input_ids.detach()
        if src.ndim == 1:
            src = src[None]
        inp_tok = tok[:, :-1]
        kw = {'output_attentions': True, 'output_hidden_states': True, 'use_cache': False, 'return_dict': True}
        with amp():
            if self.is_enc_dec:
                out_plain = self.model(input_ids=src, decoder_input_ids=inp_tok, use_cache=False, return_dict=True)
                out = self.model(input_ids=src, decoder_input_ids=inp_tok, **kw)
            else:
                out_plain = self.model(input_ids=inp_tok, use_cache=False, return_dict=True)
                out = self.model(input_ids=inp_tok, **kw)
        diff = float((out_plain.logits.float() - out.logits.float()).abs().max())

        P = int(inp_tok.shape[1])
        if not self.is_enc_dec:
            if not getattr(out, 'attentions', None):
                raise RuntimeError('HuggingFace causal model returned no attentions; load with attn_implementation="eager"')
            attns = list(out.attentions)
            hs = list(out.hidden_states) if out.hidden_states is not None else None
            L = len(attns)
            norm_data = np.stack([hs[i + 1][0].float().norm(dim=-1).cpu().numpy() for i in range(L)])[:, None, :] if hs else np.ones((L, 1, P), np.float32)
            in_g = spec.label_for_group('input', 'Input')
            dec_g = spec.label_for_group('decoder', 'Transformer')
            stages = [(StageSpec('input', spec.label_for_stage('input', 'Tokens'), in_g, 'input', P, 'target', 0), np.full((P,), 0.45, np.float32))]
            for i in range(L):
                stages.append((StageSpec(f'block_{i}', spec.label_for_stage(f'block_{i}', f'Block {i + 1}'), dec_g, 'layer', P, 'target', 0), norm_data[i, 0].copy()))
            fam = spec.label_for_attn('self', 'Causal Self-Attn')
            links = []
            for i in range(L):
                links.append((LinkSpec(i, i + 1, f'Residual L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
                links.append((LinkSpec(i, i + 1, f'{fam} L{i + 1}', fam, 'self', 'violet', i, 0, True), attns[i][0].float().cpu().numpy().astype(np.float16)))
            return TraceOutput(
                arch='decoder_only',
                model_name=spec.model_name or self.model_name,
                tok=tok[0].detach().cpu().numpy().astype(np.int32),
                logits=out_plain.logits[0].detach().float().cpu(),
                t0=t0,
                stages=stages,
                links=links,
                norm_name=f'{spec.model_name or self.model_name} Hidden Norms',
                norm_layers=[spec.label_for_stage(f'block_{i}', f'Block {i + 1}') for i in range(L)],
                norm_components=[spec.label_for_comp('resid', 'Hidden Norm')],
                norm_data=norm_data.astype(np.float32),
                check_logit_diff=diff,
            )

        if not getattr(out, 'encoder_attentions', None) or not getattr(out, 'decoder_attentions', None):
            raise RuntimeError('HuggingFace seq2seq model returned no attentions; load with attn_implementation="eager"')
        e_attns = list(out.encoder_attentions)
        d_attns = list(out.decoder_attentions)
        c_attns = list(out.cross_attentions)
        E, Le, Ld = int(src.shape[1]), len(e_attns), len(d_attns)
        e_hs = list(out.encoder_hidden_states) if getattr(out, 'encoder_hidden_states', None) is not None else None
        d_hs = list(out.decoder_hidden_states) if out.decoder_hidden_states is not None else None
        norm_data = np.stack([d_hs[i + 1][0].float().norm(dim=-1).cpu().numpy() for i in range(Ld)])[:, None, :] if d_hs else np.ones((Ld, 1, P), np.float32)

        stages = [(StageSpec('src_in', spec.label_for_stage('src_in', 'Source'), spec.label_for_group('source', 'Source'), 'input', E, 'source', 0), np.full((E,), 0.45, np.float32))]
        for i in range(Le):
            act_e = e_hs[i + 1][0].float().norm(dim=-1).cpu().numpy() if e_hs else None
            stages.append((StageSpec(f'enc_{i}', spec.label_for_stage(f'enc_{i}', f'Encoder {i + 1}'), spec.label_for_group('encoder', 'Encoder'), 'layer', E, 'source', 0), act_e))
        enc_last = len(stages) - 1
        din = len(stages)
        stages.append((StageSpec('dec_in', spec.label_for_stage('dec_in', 'Dec Input'), spec.label_for_group('dec_in', 'Target In'), 'input', P, 'target', 0, gap_before=28.0), None))
        for i in range(Ld):
            stages.append((StageSpec(f'dec_{i}', spec.label_for_stage(f'dec_{i}', f'Decoder {i + 1}'), spec.label_for_group('decoder', 'Decoder'), 'layer', P, 'target', 0), norm_data[i, 0].copy()))
        links = []
        efam = spec.label_for_attn('enc_self', 'Encoder Self-Attn')
        sfam = spec.label_for_attn('dec_self', 'Decoder Self-Attn')
        cfam = spec.label_for_attn('dec_cross', 'Encoder-Decoder Cross-Attn')
        for i in range(Le):
            links.append((LinkSpec(i, i + 1, f'Enc Resid L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
            links.append((LinkSpec(i, i + 1, f'{efam} L{i + 1}', efam, 'self', 'blue', i, 0, False), e_attns[i][0].float().cpu().numpy().astype(np.float16)))
        for i in range(Ld):
            links.append((LinkSpec(din + i, din + 1 + i, f'Dec Resid L{i + 1}', '', 'resid', 'violet', i, 0, False), None))
            links.append((LinkSpec(din + i, din + 1 + i, f'{sfam} L{i + 1}', sfam, 'self', 'violet', i, 0, True), d_attns[i][0].float().cpu().numpy().astype(np.float16)))
            links.append((LinkSpec(enc_last, din + 1 + i, f'{cfam} L{i + 1}', cfam, 'cross', 'magenta', i, 0, False), c_attns[i][0].float().cpu().numpy().astype(np.float16)))
        return TraceOutput(
            arch='encoder_decoder',
            model_name=spec.model_name or self.model_name,
            tok=tok[0].detach().cpu().numpy().astype(np.int32),
            src_tok=src[0].detach().cpu().numpy().astype(np.int32),
            logits=out_plain.logits[0].detach().float().cpu(),
            t0=t0,
            stages=stages,
            links=links,
            norm_name=f'{spec.model_name or self.model_name} Decoder Norms',
            norm_layers=[spec.label_for_stage(f'dec_{i}', f'Decoder {i + 1}') for i in range(Ld)],
            norm_components=[spec.label_for_comp('resid', 'Hidden Norm')],
            norm_data=norm_data.astype(np.float32),
            check_logit_diff=diff,
        )

    @torch.no_grad()
    def record_generation(self, tokens: torch.Tensor, max_new_tokens: int,
                          dec_prompt: torch.Tensor | None = None,
                          eos_id: int | list[int] | None = None,
                          ref_tokens: torch.Tensor | None = None,
                          spec: TraceSpec | None = None,
                          top_k: int = 5, edge_k: int = 4, label: str = '', amp=no_amp) -> DemoTraceOutput:
        """Records greedy autoregressive generation (causal or seq2seq) step-by-step into a DemoTraceOutput."""
        spec = spec or TraceSpec()
        _ensure_eager_attention(self.model)
        dev = next(self.model.parameters()).device
        eos_set = set(eos_id) if isinstance(eos_id, (list, tuple, set)) else ({int(eos_id)} if eos_id is not None else set())

        if not self.is_enc_dec:
            seq = tokens.detach().to(dev).reshape(1, -1)
            t0 = int(seq.shape[1])
            probs = [1.0] * t0
            top_ids, top_ps, ents, self_ids, self_ws, norms = [], [], [], [], [], []
            ended = False
            L = 0
            for _ in range(max_new_tokens):
                with amp():
                    out = self.model(input_ids=seq, use_cache=False, output_attentions=True, output_hidden_states=True, return_dict=True)
                if not getattr(out, 'attentions', None):
                    raise RuntimeError('HuggingFace causal model returned no attentions; load with attn_implementation="eager"')
                attns = list(out.attentions)
                hs = list(out.hidden_states) if out.hidden_states is not None else None
                L = len(attns)
                logits = out.logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                p = lp.exp()
                kt = min(top_k, p.numel())
                tlp, tid = lp.topk(kt)
                tp = tlp.exp().cpu().numpy().astype(np.float32)
                tid_np = tid.cpu().numpy().astype(np.int32)
                if kt < top_k:
                    tid_np = np.pad(tid_np, (0, top_k - kt), constant_values=-1)
                    tp = np.pad(tp, (0, top_k - kt), constant_values=0.0)
                ent = float((-(p * lp).where(p > 0, torch.zeros_like(lp))).sum().item() / math.log(2))
                nxt = int(tid_np[0])
                top_ids.append(tid_np); top_ps.append(tp); ents.append(ent); probs.append(float(tp[0]))

                s_id = np.full((L, edge_k), -1, np.int32)
                s_w = np.zeros((L, edge_k), np.float16)
                nm = np.zeros((L, 1), np.float16)
                for li in range(L):
                    wl = _avg_heads(attns[li][0].float().cpu().numpy())[-1]
                    kk = min(edge_k, len(wl))
                    ts = np.argsort(-wl)[:kk]
                    s_id[li, :kk] = ts.astype(np.int32)
                    s_w[li, :kk] = wl[ts].astype(np.float16)
                    nm[li, 0] = float(hs[li + 1][0, -1].float().norm().item()) if hs else 1.0
                self_ids.append(s_id); self_ws.append(s_w); norms.append(nm)

                seq = torch.cat([seq, torch.tensor([[nxt]], device=dev, dtype=seq.dtype)], dim=1)
                if nxt in eos_set:
                    ended = True
                    break

            ftok = seq[0].detach().cpu().numpy().astype(np.int32)
            M = len(ftok)
            full_top_id = np.full((M, top_k), -1, np.int32)
            full_top_p = np.zeros((M, top_k), np.float32)
            full_ent = np.zeros((M,), np.float32)
            full_sid = np.full((M, L, edge_k), -1, np.int32)
            full_sw = np.zeros((M, L, edge_k), np.float16)
            full_nm = np.zeros((M, L, 1), np.float16)
            full_top_id[t0:] = np.stack(top_ids); full_top_p[t0:] = np.stack(top_ps); full_ent[t0:] = np.asarray(ents, np.float32)
            full_sid[t0:] = np.stack(self_ids); full_sw[t0:] = np.stack(self_ws); full_nm[t0:] = np.stack(norms)

            in_g = spec.label_for_group('input', 'Tokens')
            dec_g = spec.label_for_group('decoder', 'Transformer')
            stages = [(StageSpec('input', spec.label_for_stage('input', 'Input'), in_g, 'input', M - 1, 'target', 0), None)]
            for li in range(L):
                stages.append((StageSpec(f'block_{li}', spec.label_for_stage(f'block_{li}', f'Block {li + 1}'), dec_g, 'layer', M - 1, 'target', 0), None))
            stages.append((StageSpec('output', 'Candidates', 'Output', 'output', top_k, 'none', 0, span_y=90.0, z_jitter=0.0, x_jitter=0.0, gap_before=20.0), None))

            ref_np = ref_tokens.detach().cpu().numpy().astype(np.int32) if ref_tokens is not None else None
            div = -1
            exact = False
            if ref_np is not None:
                for j in range(t0, min(M, len(ref_np))):
                    if ftok[j] != ref_np[j]:
                        div = j
                        break
                exact = div < 0 and M == len(ref_np)

            return DemoTraceOutput(
                arch='decoder_only',
                model_name=spec.model_name or self.model_name,
                label=label or f'{spec.model_name or self.model_name} Rollout',
                kind='causal',
                t0=t0, ftok=ftok, fp=np.asarray(probs[:M], np.float32),
                stages=stages, in_stage=0, out_stage=len(stages) - 1, layer_stages=list(range(1, L + 1)),
                top_id=full_top_id, top_p=full_top_p, entropy=full_ent,
                src_tok=None, ref_tok=ref_np, self_id=full_sid, self_w=full_sw, norm=full_nm,
                has_read=False, diverge=div, free_exact=exact, free_ended=ended or (M - t0) == max_new_tokens,
            )

        src = tokens.detach().to(dev).reshape(1, -1)
        E = int(src.shape[1])
        start_id = getattr(getattr(self.model, 'config', None), 'decoder_start_token_id', None)
        if start_id is None:
            start_id = getattr(getattr(self.model, 'config', None), 'bos_token_id', None) or 0
        seq = torch.tensor([[int(start_id)]], device=dev, dtype=src.dtype) if dec_prompt is None else dec_prompt.detach().to(dev).reshape(1, -1)
        t0 = int(seq.shape[1])
        probs = [1.0] * t0

        enc_mod = self.model.get_encoder()
        with amp():
            enc_out = enc_mod(input_ids=src, output_attentions=True, output_hidden_states=True, return_dict=True)
        if not getattr(enc_out, 'attentions', None):
            raise RuntimeError('HuggingFace seq2seq encoder returned no attentions; load with attn_implementation="eager"')
        e_attns = list(enc_out.attentions)
        e_hs = list(enc_out.hidden_states) if getattr(enc_out, 'hidden_states', None) is not None else None
        Le = len(e_attns)

        top_ids, top_ps, ents, self_ids, self_ws, cross_ids, cross_ws, norms = [], [], [], [], [], [], [], []
        ended = False
        Ld = 0
        for _ in range(max_new_tokens):
            with amp():
                out = self.model(encoder_outputs=enc_out, decoder_input_ids=seq, use_cache=False,
                                 output_attentions=True, output_hidden_states=True, return_dict=True)
            if not getattr(out, 'decoder_attentions', None) or not getattr(out, 'cross_attentions', None):
                raise RuntimeError('HuggingFace seq2seq decoder returned no attentions; load with attn_implementation="eager"')
            d_attns = list(out.decoder_attentions)
            c_attns = list(out.cross_attentions)
            d_hs = list(out.decoder_hidden_states) if getattr(out, 'decoder_hidden_states', None) is not None else None
            Ld = len(d_attns)
            logits = out.logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            kt = min(top_k, p.numel())
            tlp, tid = lp.topk(kt)
            tp = tlp.exp().cpu().numpy().astype(np.float32)
            tid_np = tid.cpu().numpy().astype(np.int32)
            if kt < top_k:
                tid_np = np.pad(tid_np, (0, top_k - kt), constant_values=-1)
                tp = np.pad(tp, (0, top_k - kt), constant_values=0.0)
            ent = float((-(p * lp).where(p > 0, torch.zeros_like(lp))).sum().item() / math.log(2))
            nxt = int(tid_np[0])
            top_ids.append(tid_np); top_ps.append(tp); ents.append(ent); probs.append(float(tp[0]))

            s_id = np.full((Ld, edge_k), -1, np.int32); s_w = np.zeros((Ld, edge_k), np.float16)
            c_id = np.full((Ld, edge_k), -1, np.int32); c_w = np.zeros((Ld, edge_k), np.float16)
            nm = np.zeros((Ld, 1), np.float16)
            for li in range(Ld):
                wl = _avg_heads(d_attns[li][0].float().cpu().numpy())[-1]
                kks = min(edge_k, len(wl)); ts = np.argsort(-wl)[:kks]
                s_id[li, :kks] = ts.astype(np.int32); s_w[li, :kks] = wl[ts].astype(np.float16)

                wc = _avg_heads(c_attns[li][0].float().cpu().numpy())[-1]
                kkc = min(edge_k, len(wc)); tc = np.argsort(-wc)[:kkc]
                c_id[li, :kkc] = tc.astype(np.int32); c_w[li, :kkc] = wc[tc].astype(np.float16)
                nm[li, 0] = float(d_hs[li + 1][0, -1].float().norm().item()) if d_hs else 1.0
            self_ids.append(s_id); self_ws.append(s_w)
            cross_ids.append(c_id); cross_ws.append(c_w); norms.append(nm)

            seq = torch.cat([seq, torch.tensor([[nxt]], device=dev, dtype=seq.dtype)], dim=1)
            if nxt in eos_set:
                ended = True
                break

        ftok = seq[0].detach().cpu().numpy().astype(np.int32)
        M = len(ftok)
        full_top_id = np.full((M, top_k), -1, np.int32); full_top_p = np.zeros((M, top_k), np.float32); full_ent = np.zeros((M,), np.float32)
        full_sid = np.full((M, Ld, edge_k), -1, np.int32); full_sw = np.zeros((M, Ld, edge_k), np.float16)
        full_cid = np.full((M, Ld, edge_k), -1, np.int32); full_cw = np.zeros((M, Ld, edge_k), np.float16)
        full_nm = np.zeros((M, Ld, 1), np.float16)
        full_top_id[t0:] = np.stack(top_ids); full_top_p[t0:] = np.stack(top_ps); full_ent[t0:] = np.asarray(ents, np.float32)
        full_sid[t0:] = np.stack(self_ids); full_sw[t0:] = np.stack(self_ws)
        full_cid[t0:] = np.stack(cross_ids); full_cw[t0:] = np.stack(cross_ws); full_nm[t0:] = np.stack(norms)

        src_g = spec.label_for_group('source', 'Source')
        enc_g = spec.label_for_group('encoder', 'Encoder')
        din_g = spec.label_for_group('dec_in', 'Dec In')
        dec_g = spec.label_for_group('decoder', 'Decoder')
        stages: list[tuple[StageSpec, np.ndarray | None]] = [
            (StageSpec('src_in', spec.label_for_stage('src_in', 'Source'), src_g, 'input', E, 'source', 0), np.full((E,), 0.4, np.float32))
        ]
        for li in range(Le):
            act_e = e_hs[li + 1][0].float().norm(dim=-1).cpu().numpy().astype(np.float32) if e_hs else None
            stages.append((StageSpec(f'enc_{li}', spec.label_for_stage(f'enc_{li}', f'Encoder {li + 1}'), enc_g, 'layer', E, 'source', 0), act_e))
        enc_last = len(stages) - 1
        in_stage = len(stages)
        stages.append((StageSpec('dec_in', spec.label_for_stage('dec_in', 'Dec Input'), din_g, 'input', M - 1, 'target', 0, gap_before=28.0), None))
        for li in range(Ld):
            stages.append((StageSpec(f'dec_{li}', spec.label_for_stage(f'dec_{li}', f'Decoder {li + 1}'), dec_g, 'layer', M - 1, 'target', 0), None))
        out_stage = len(stages)
        stages.append((StageSpec('output', 'Candidates', 'Output', 'output', top_k, 'none', 0, span_y=90.0, z_jitter=0.0, x_jitter=0.0, gap_before=20.0), None))

        prefill_links = []
        for li in range(Le):
            w = _avg_heads(e_attns[li][0].float().cpu().numpy())
            idx, vals = top_k_edges(w, min(3, E))
            prefill_links.append({'src': li, 'dst': li + 1, 'kind': 'self', 'tint': 'blue', 'ids': idx, 'weights': vals})

        ref_np = ref_tokens.detach().cpu().numpy().astype(np.int32) if ref_tokens is not None else None
        div = -1
        exact = False
        if ref_np is not None:
            for j in range(t0, min(M, len(ref_np))):
                if ftok[j] != ref_np[j]:
                    div = j
                    break
            exact = div < 0 and M == len(ref_np)

        return DemoTraceOutput(
            arch='encoder_decoder',
            model_name=spec.model_name or self.model_name,
            label=label or f'{spec.model_name or self.model_name} Rollout',
            kind='seq2seq',
            t0=t0, ftok=ftok, fp=np.asarray(probs[:M], np.float32),
            stages=stages, in_stage=in_stage, out_stage=out_stage, layer_stages=list(range(in_stage + 1, out_stage)),
            top_id=full_top_id, top_p=full_top_p, entropy=full_ent,
            src_tok=src[0].detach().cpu().numpy().astype(np.int32), ref_tok=ref_np,
            self_id=full_sid, self_w=full_sw, norm=full_nm,
            has_read=True, cross_stage=enc_last, cross_id=full_cid, cross_w=full_cw, prefill_links=prefill_links,
            diverge=div, free_exact=exact, free_ended=ended or (M - t0) == max_new_tokens,
        )


def auto_adapter(model: nn.Module, **kwargs):
    """Automatically selects the most suitable PyTorch adapter for a given model."""
    cfg = getattr(model, 'config', None)
    if cfg is not None and hasattr(cfg, 'model_type'):
        return HuggingFaceAdapter(model, **kwargs)
    enc = _find_attr_module_list(model, ('encoder.layers', 'enc.layers', 'encoder.blocks'))
    dec = _find_attr_module_list(model, ('decoder.layers', 'dec.layers', 'decoder.blocks'))
    if enc and dec:
        return EncoderDecoderAdapter(model, enc_layers=enc, dec_layers=dec, **kwargs)
    return DecoderOnlyAdapter(model, **kwargs)
