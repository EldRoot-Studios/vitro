"""Vitro: Glass-box PyTorch trace inspector & 3D activation player.

Importing the package is cheap on purpose (NumPy only). The parts that need PyTorch - the model adapters, the
capture hooks and the exporters - are loaded the first time one of their names is used, so `vitro web` and
`vitro info`, which only read saved traces, start in a fraction of a second instead of waiting for torch.
"""
from __future__ import annotations

import importlib

from .container import ContainerFile, MAGIC, Writer, save
from .spec import LinkSpec, SpanLabel, StageSpec, TokenMeta, TraceSpec

__version__ = '0.2.0'

# Public names that need PyTorch, mapped to the submodule that defines them.
_LAZY = {
    'DecoderOnlyAdapter': 'adapters',
    'DemoTraceOutput': 'adapters',
    'EncoderDecoderAdapter': 'adapters',
    'HuggingFaceAdapter': 'adapters',
    'TraceOutput': 'adapters',
    'auto_adapter': 'adapters',
    'HookSession': 'capture',
    'compute_token_stats': 'capture',
    'no_amp': 'capture',
    'register_attention_hook': 'capture',
    'register_norm_hook': 'capture',
    'top_k_edges': 'capture',
    'export_demo': 'exporter',
    'export_dump': 'exporter',
}
# Submodules that used to be attributes of the package because it imported them eagerly.
_SUBMODULES = ('adapters', 'capture', 'exporter')

__all__ = [
    'ContainerFile', 'MAGIC', 'Writer', 'save',
    'SpanLabel', 'StageSpec', 'LinkSpec', 'TokenMeta', 'TraceSpec',
    'HookSession', 'no_amp', 'register_attention_hook', 'register_norm_hook', 'compute_token_stats', 'top_k_edges',
    'TraceOutput', 'DemoTraceOutput', 'DecoderOnlyAdapter', 'EncoderDecoderAdapter', 'HuggingFaceAdapter', 'auto_adapter',
    'export_dump', 'export_demo',
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', __name__)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(f'.{module}', __name__), name)
    globals()[name] = value  # the next access finds it directly
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))
