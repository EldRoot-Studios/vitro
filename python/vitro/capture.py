"""Model-agnostic PyTorch hook primitives & per-token statistical computations."""
from __future__ import annotations

import contextlib
import math
from typing import Callable

import numpy as np
import torch


def no_amp():
    """Null precision context manager usable as a fallback AMP context."""
    return contextlib.nullcontext()


def extract_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f'expected Tensor or tuple starting with Tensor, got {type(output).__name__}')


def extract_attn_weights(output) -> torch.Tensor | None:
    if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
        return output[1]
    return None


def register_attention_hook(handles: list, expected: set[str], store: Callable[[str, torch.Tensor], None],
                            key: str, module, rows_from: int = 0) -> None:
    """Registers forward hooks capturing MultiheadAttention / tuple-weight attention maps."""
    expected.add(key)

    def pre(_mod, args, kwargs):
        kw = dict(kwargs)
        if 'need_weights' in _mod.forward.__code__.co_varnames:
            kw['need_weights'] = True
            if 'average_attn_weights' in _mod.forward.__code__.co_varnames:
                kw['average_attn_weights'] = False
        return args, kw

    def post(_mod, _args, _kwargs, output):
        w = extract_attn_weights(output)
        if w is None:
            raise RuntimeError(f'{key}: attention module did not return weights')
        if w.ndim == 3:
            w = w.unsqueeze(1)
        store(key, w[:, :, rows_from:, :])

    handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
    handles.append(module.register_forward_hook(post, with_kwargs=True))


def register_norm_hook(handles: list, expected: set[str], store: Callable[[str, torch.Tensor], None],
                       key: str, module, pick: Callable = extract_tensor, rows_from: int = 0) -> None:
    """Registers a forward hook recording L2 activation norm per sequence position."""
    expected.add(key)

    def post(_mod, _args, output):
        t = pick(output)
        store(key, t[:, rows_from:].float().norm(dim=-1))

    handles.append(module.register_forward_hook(post))


class HookSession:
    """Manages PyTorch MHA fast-path disabling, hook lifecycle cleanup, and completeness assertions."""

    def __init__(self, install_fn: Callable[['HookSession'], None]) -> None:
        self._install_fn = install_fn
        self.handles: list = []
        self.expected: set[str] = set()
        self.rec: dict[str, torch.Tensor] = {}
        self._fast: bool | None = None

    def store(self, key: str, value: torch.Tensor) -> None:
        if key in self.rec:
            raise RuntimeError(f'{key}: recorded twice during one forward pass')
        self.rec[key] = value.detach().float().cpu()

    def watch_attention(self, key: str, module, rows_from: int = 0) -> None:
        register_attention_hook(self.handles, self.expected, self.store, key, module, rows_from)

    def watch_norm(self, key: str, module, pick: Callable = extract_tensor, rows_from: int = 0) -> None:
        register_norm_hook(self.handles, self.expected, self.store, key, module, pick, rows_from)

    def cleanup(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        if self._fast is not None:
            torch.backends.mha.set_fastpath_enabled(self._fast)
            self._fast = None

    def __enter__(self) -> 'HookSession':
        self._fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            self._install_fn(self)
        except BaseException:
            self.cleanup()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
        if exc[0] is None:
            missing = sorted(self.expected - set(self.rec))
            if missing:
                raise RuntimeError(f'{len(missing)} hooked module(s) never executed: {missing[:5]}')


def compute_token_stats(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 5,
                        value_ids: torch.Tensor | None = None,
                        is_value_mask: torch.Tensor | None = None,
                        require_finite_targets: bool = True) -> dict[str, np.ndarray]:
    """Computes per-step probability, rank, top-K candidates, entropy, and optional scalar metrics.

    Args:
        logits: ``(R, V)`` float logits at each predicted position.
        targets: ``(R,)`` integer target token IDs.
    """
    rows = logits.float()
    tgt = targets.long()
    lp = torch.log_softmax(rows, -1)
    lp_true = lp.gather(1, tgt[:, None])[:, 0]
    if require_finite_targets and not bool(torch.isfinite(lp_true).all()):
        bad = (~torch.isfinite(lp_true)).nonzero()[:, 0]
        raise ValueError(f'model assigns -inf probability to reference targets at positions {bad.tolist()[:8]}')

    p = lp.exp()
    k_take = min(top_k, lp.shape[1])
    top_lp, top_id = lp.topk(k_take, dim=1)
    top_p = top_lp.exp()
    top_id = top_id.masked_fill(~torch.isfinite(top_lp), -1)
    if k_take < top_k:
        pad = top_k - k_take
        top_id = torch.nn.functional.pad(top_id, (0, pad), value=-1)
        top_p = torch.nn.functional.pad(top_p, (0, pad), value=0.0)

    plogp = torch.where(p > 0, p * lp, torch.zeros_like(lp))
    out = {
        'p': lp_true.exp(),
        'bits': -lp_true.double() / math.log(2),
        'rank': 1 + (lp > lp_true[:, None]).sum(1),
        'top_id': top_id,
        'top_p': top_p,
        'entropy': -plogp.sum(1) / math.log(2),
        'n_allowed': torch.isfinite(rows).sum(1),
        'right': rows.argmax(-1) == tgt,
    }
    if value_ids is not None:
        levels = torch.arange(len(value_ids), device=lp.device, dtype=torch.float32)
        pv = p[:, value_ids.long()]
        mass = pv.sum(1)
        safe = mass.clamp_min(1e-30)
        mean = (pv * levels).sum(1) / safe
        spread = ((pv * (levels - mean[:, None]) ** 2).sum(1) / safe).sqrt()
        mask = (mass > 1e-6) if is_value_mask is None else (is_value_mask.bool() & (mass > 1e-6))
        nan = torch.full_like(mean, float('nan'))
        out['val_mean'] = torch.where(mask, mean, nan)
        out['val_std'] = torch.where(mask, spread, nan)

    dtypes = {
        'p': np.float32, 'bits': np.float64, 'rank': np.int32, 'top_id': np.int32,
        'top_p': np.float32, 'entropy': np.float32, 'n_allowed': np.int32, 'right': np.bool_,
        'val_mean': np.float32, 'val_std': np.float32,
    }
    return {k: v.detach().cpu().numpy().astype(dtypes[k]) for k, v in out.items()}


def top_k_edges(weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compresses a ``(C_dst, C_src)`` weight matrix to top-``k`` sources per row."""
    if weights.ndim != 2:
        raise ValueError(f'expected 2D matrix, got {weights.shape}')
    dst, src = weights.shape
    kk = min(max(1, k), src)
    idx = np.argsort(-weights, axis=1)[:, :kk].astype(np.int32)
    vals = np.take_along_axis(weights, idx, axis=1).astype(np.float16)
    return idx, vals
