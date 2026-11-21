"""Declarative trace annotations & user-customizable 3D architecture labels.

Lets users explicitly name and group:
  - Architectural stages & connections in the 3D view (`StageSpec`, `LinkSpec`)
  - Sequence segments / prompt sections / tree parts (`SpanLabel`)
  - Token display names, category tags, and scalar numeric values (`TokenMeta`)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpanLabel:
    """User-defined labeled token interval ``[start, end)`` shown in Tokens, Spans, and Demo views."""

    start: int
    end: int
    label: str
    kind: str = 'region'
    parent: int = -1
    tint: str = ''
    meta: str = ''

    def to_json(self, idx: int) -> dict:
        if self.end < self.start or self.start < 0:
            raise ValueError(f'invalid span range [{self.start}, {self.end}) for {self.label!r}')
        return {
            'id': idx,
            'start': int(self.start),
            'end': int(self.end),
            'label': str(self.label),
            'kind': str(self.kind),
            'parent': int(self.parent),
            'tint': str(self.tint),
            'meta': str(self.meta),
        }


@dataclass
class StageSpec:
    """Describes one column in Vitro's dynamic 3D network graph."""

    name: str
    label: str
    group: str = ''
    kind: str = 'layer'                # input | layer | latent | output
    count: int = 0
    token_source: str = 'none'         # none | source | target
    token_offset: int = 0
    span_y: float = 150.0
    z_jitter: float = 44.0
    x_jitter: float = 5.0
    gap_before: float = 0.0


@dataclass
class LinkSpec:
    """Describes a directed edge set (residual or attention matrix) between two 3D stages."""

    src: int
    dst: int
    name: str
    family: str = ''
    kind: str = 'self'                 # resid | self | cross | out
    tint: str = 'violet'
    layer_idx: int = -1
    row_offset: int = 0
    causal: bool = False


@dataclass
class TokenMeta:
    """Optional user vocabulary labels, token category tags, and numeric values."""

    labels: list[str]
    tag_names: list[str] = field(default_factory=list)
    token_tags: list[int] | None = None
    values: list[float] | None = None

    @classmethod
    def from_tokenizer(cls, tokenizer, vocab_size: int | None = None, readable: bool = True, **extra) -> 'TokenMeta':
        n = vocab_size or getattr(tokenizer, 'vocab_size', None) or len(tokenizer)
        decode = getattr(tokenizer, 'convert_ids_to_tokens', None)
        if callable(decode):
            raw = decode(list(range(n)))
            labs = [str(t) if t is not None else f'<{i}>' for i, t in enumerate(raw)]
        else:
            labs = [f't{i}' for i in range(n)]
        if readable:
            labs = [_clean_token_label(t) for t in labs]
        return cls(labels=labs, **extra)


def _clean_token_label(t: str) -> str:
    if t in ('\u2581', '\u0120', ' '):
        return '·'
    t = t.replace('\u2581', ' ').replace('\u0120', ' ').replace('\u010a', '\\n').replace('\u0109', '\\t')
    return ''.join(ch if ord(ch) >= 0x20 and ord(ch) != 0x7f else f'\\x{ord(ch):02x}' for ch in t)


@dataclass
class TraceSpec:
    """High-level user configuration controlling names, sections, and custom labels."""

    trace_id: str = 'trace'
    title: str = ''
    model_name: str = ''
    unit_label: str = 'Forward Pass'
    stage_labels: dict[str, str] = field(default_factory=dict)
    group_labels: dict[str, str] = field(default_factory=dict)
    attn_names: dict[str, str] = field(default_factory=dict)
    component_labels: dict[str, str] = field(default_factory=dict)
    spans: list[SpanLabel] = field(default_factory=list)
    token_meta: TokenMeta | None = None
    delta_red_at: float = 12.0
    top_k: int = 5

    def label_for_stage(self, key: str, default: str) -> str:
        return self.stage_labels.get(key, default)

    def label_for_group(self, key: str, default: str) -> str:
        return self.group_labels.get(key, default)

    def label_for_attn(self, key: str, default: str) -> str:
        return self.attn_names.get(key, default)

    def label_for_comp(self, key: str, default: str) -> str:
        return self.component_labels.get(key, default)
