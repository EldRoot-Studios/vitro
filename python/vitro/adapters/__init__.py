"""PyTorch model adapters for Vitro."""
from .base import DemoTraceOutput, TraceOutput
from .pytorch import DecoderOnlyAdapter, EncoderDecoderAdapter, HuggingFaceAdapter, auto_adapter

__all__ = [
    'TraceOutput',
    'DemoTraceOutput',
    'DecoderOnlyAdapter',
    'EncoderDecoderAdapter',
    'HuggingFaceAdapter',
    'auto_adapter',
]
