"""Adapter base dataclasses describing model topology, forward activations, and rollout steps."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..spec import LinkSpec, StageSpec


@dataclass
class TraceOutput:
    """Standardized output of tracing one forward pass on a PyTorch model."""

    arch: str
    model_name: str
    tok: np.ndarray                                    # (T_out,) int64
    logits: torch.Tensor                               # (T_out - 1, V) float32 on CPU or device
    t0: int = 1
    src_tok: np.ndarray | None = None                  # optional (E_src,) for encoder-decoder
    stages: list[tuple[StageSpec, np.ndarray | None]] = field(default_factory=list)
    links: list[tuple[LinkSpec, np.ndarray | None]] = field(default_factory=list)
    norm_name: str = 'Layer Activations'
    norm_layers: list[str] = field(default_factory=list)
    norm_components: list[str] = field(default_factory=list)
    norm_data: np.ndarray | None = None                # (L, C, R) float32
    latents: list[dict] = field(default_factory=list)
    check_logit_diff: float = 0.0


@dataclass
class DemoTraceOutput:
    """Standardized autoregressive generation recording with dynamic stages & sparse ray tapes."""

    arch: str
    model_name: str
    label: str
    kind: str
    t0: int
    ftok: np.ndarray                                   # (M,) int64
    fp: np.ndarray                                     # (M,) float32
    stages: list[tuple[StageSpec, np.ndarray | None]]
    in_stage: int
    out_stage: int
    layer_stages: list[int]
    top_id: np.ndarray                                 # (M, top_k) int32
    top_p: np.ndarray                                  # (M, top_k) float32
    entropy: np.ndarray                                # (M,) float32
    self_id: np.ndarray                                # (M, Ld, K_self) int32
    self_w: np.ndarray                                 # (M, Ld, K_self) float16
    norm: np.ndarray                                   # (M, Ld, C) float16
    src_tok: np.ndarray | None = None
    ref_tok: np.ndarray | None = None
    p_ref: np.ndarray | None = None
    has_read: bool = False
    read_ticks: float = 8.0
    cross_stage: int = -1
    cross_id: np.ndarray | None = None                 # (M, Ld, K_cross) int32
    cross_w: np.ndarray | None = None                  # (M, Ld, K_cross) float16
    prefill_links: list[dict] = field(default_factory=list)
    diverge: int = -1
    free_exact: bool = False
    free_ended: bool = True
