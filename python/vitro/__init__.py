"""Vitro: Glass-box PyTorch trace inspector & 3D activation player."""
from .adapters import (
    DecoderOnlyAdapter,
    DemoTraceOutput,
    EncoderDecoderAdapter,
    HuggingFaceAdapter,
    TraceOutput,
    auto_adapter,
)
from .capture import HookSession, compute_token_stats, no_amp, register_attention_hook, register_norm_hook, top_k_edges
from .container import ContainerFile, MAGIC, Writer, save
from .exporter import export_demo, export_dump
from .spec import LinkSpec, SpanLabel, StageSpec, TokenMeta, TraceSpec

__version__ = '0.1.0'

__all__ = [
    'ContainerFile', 'MAGIC', 'Writer', 'save',
    'SpanLabel', 'StageSpec', 'LinkSpec', 'TokenMeta', 'TraceSpec',
    'HookSession', 'no_amp', 'register_attention_hook', 'register_norm_hook', 'compute_token_stats', 'top_k_edges',
    'TraceOutput', 'DemoTraceOutput', 'DecoderOnlyAdapter', 'EncoderDecoderAdapter', 'HuggingFaceAdapter', 'auto_adapter',
    'export_dump', 'export_demo',
]
