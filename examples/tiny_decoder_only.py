"""Example: trace and record a tiny PyTorch causal GPT-like model with custom stage & span labels."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.join(os.path.dirname(HERE), 'python')
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from vitro import DecoderOnlyAdapter, SpanLabel, TokenMeta, TraceSpec, export_demo, export_dump  # noqa: E402


class TinyCausalBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + a
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 48, layers: int = 3, heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([TinyCausalBlock(d, heads) for _ in range(layers)])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)[None]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).triu(1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm_f(x))


def make_vocab() -> list[str]:
    special = ['<pad>', '<bos>', '<eos>']
    words = ['sys', ':', 'task', 'sort', 'list', 'if', 'len', '<=', '1', 'return',
             'x', 'for', 'in', 'range', 'min', 'max', 'sum', 'add', 'out', '=',
             '(', ')', ',', '->', 'done', 'ok', 'err', 'val', 'n']
    return special + words


def run_example(out_dir: str) -> tuple[str, str]:
    torch.manual_seed(42)
    vocab = make_vocab()
    model = TinyGPT(vocab=len(vocab)).eval()

    # Reference sequence with user-labeled regions ("spans") and token tags
    seq_words = ['<bos>', 'sys', ':', 'task', 'sort', 'list', '->', 'if', 'len', '(', 'x', ')', '<=', '1', 'return', 'x', ',', 'done', '<eos>']
    tok_ids = torch.tensor([vocab.index(w) for w in seq_words], dtype=torch.long)
    tags = [0 if w.startswith('<') else 1 if i < 6 else 2 for i, w in enumerate(seq_words)]

    spec = TraceSpec(
        trace_id='tiny_gpt_example',
        title='TinyGPT - User Labeled Trace',
        model_name='TinyGPT-3L',
        unit_label='Code Sequence',
        stage_labels={
            'input': 'Code Tokens',
            'block_0': 'Local Syntax',
            'block_1': 'Control Flow',
            'block_2': 'Next-Token Mix',
        },
        group_labels={'input': 'Tokens', 'decoder': 'Causal Blocks'},
        attn_names={'self': 'Causal Self-Attn'},
        component_labels={'resid': 'Block Output', 'self': 'Self-Attn', 'mlp': 'FFN'},
        spans=[
            SpanLabel(1, 6, 'System Header', kind='prompt', tint='cyan', meta='Instruction prefix'),
            SpanLabel(6, 16, 'Base Case Guard', kind='code', tint='gold', parent=0, meta='Conditional return'),
            SpanLabel(16, len(seq_words), 'Completion', kind='output', tint='emerald'),
        ],
        token_meta=TokenMeta(labels=vocab, tag_names=['special', 'instruction', 'code'], token_tags=tags),
    )

    adapter = DecoderOnlyAdapter(model, model_name='TinyGPT-3L')
    tr = adapter.trace_forward(tok_ids, spec=spec)
    demo = adapter.record_generation(tok_ids[:7], max_new_tokens=10, eos_id=vocab.index('<eos>'),
                                     ref_tokens=tok_ids, spec=spec, label='TinyGPT Rollout')

    os.makedirs(out_dir, exist_ok=True)
    dump_path = export_dump(os.path.join(out_dir, 'tiny_gpt.vtdump'), tr, spec=spec,
                            free_tokens=[demo.ftok], free_probs=[demo.fp], step=1200)
    demo_path = export_demo(os.path.join(out_dir, 'tiny_gpt.vtdemo'), demo, spec=spec,
                            vocab_size=len(vocab), step=1200)
    return dump_path, demo_path


if __name__ == '__main__':
    dp, dm = run_example(os.path.join(os.path.dirname(HERE), 'out_examples'))
    print(f'wrote {dp}\nwrote {dm}')
