"""Example: trace and record a tiny PyTorch Encoder-Decoder transformer with cross-attention."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.join(os.path.dirname(HERE), 'python')
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from vitro import EncoderDecoderAdapter, SpanLabel, TokenMeta, TraceSpec, export_demo, export_dump  # noqa: E402


class TinySeq2Seq(nn.Module):
    def __init__(self, vocab: int = 24, d: int = 32, heads: int = 2, n_enc: int = 2, n_dec: int = 2) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(vocab, d)
        self.tgt_emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(64, d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.head = nn.Linear(d, vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        es = self.src_emb(src) + self.pos(torch.arange(src.shape[1], device=src.device)[None])
        mem = self.encoder(es)
        et = self.tgt_emb(tgt) + self.pos(torch.arange(tgt.shape[1], device=tgt.device)[None])
        mask = torch.ones(tgt.shape[1], tgt.shape[1], dtype=torch.bool, device=tgt.device).triu(1)
        h = self.decoder(et, mem, tgt_mask=mask, tgt_is_causal=True)
        return self.head(h)


def run_example(out_dir: str) -> tuple[str, str]:
    torch.manual_seed(7)
    vocab = ['<pad>', '<bos>', '<eos>'] + [f'w{i}' for i in range(21)]
    model = TinySeq2Seq(vocab=len(vocab)).eval()
    src = torch.tensor([1, 4, 7, 9, 11, 5, 2], dtype=torch.long)
    tgt = torch.tensor([1, 6, 8, 10, 12, 14, 15, 2], dtype=torch.long)

    spec = TraceSpec(
        trace_id='tiny_seq2seq',
        title='TinySeq2Seq - Cross Attention Trace',
        model_name='TinySeq2Seq-2x2',
        stage_labels={
            'src_in': 'Source Tokens', 'enc_0': 'Encoder 1', 'enc_1': 'Encoder 2',
            'dec_in': 'Target Prefix', 'dec_0': 'Cross Block 1', 'dec_1': 'Cross Block 2',
        },
        spans=[
            SpanLabel(0, 1, 'BOS', kind='special', tint='cyan'),
            SpanLabel(1, 5, 'Translated Core', kind='target', tint='gold'),
            SpanLabel(5, len(tgt), 'Closing Clause', kind='target', tint='magenta'),
        ],
        token_meta=TokenMeta(labels=vocab),
    )

    adapter = EncoderDecoderAdapter(model, model_name='TinySeq2Seq-2x2')
    tr = adapter.trace_forward(src, tgt, spec=spec)
    demo = adapter.record_generation(src, tgt[:1], max_new_tokens=7, eos_id=2,
                                     ref_tokens=tgt, spec=spec, label='Seq2Seq Beam')

    os.makedirs(out_dir, exist_ok=True)
    dp = export_dump(os.path.join(out_dir, 'tiny_seq2seq.vtdump'), tr, spec=spec,
                     free_tokens=[demo.ftok], free_probs=[demo.fp], step=400)
    dm = export_demo(os.path.join(out_dir, 'tiny_seq2seq.vtdemo'), demo, spec=spec,
                     vocab_size=len(vocab), step=400)
    return dp, dm


if __name__ == '__main__':
    dp, dm = run_example(os.path.join(os.path.dirname(HERE), 'out_examples'))
    print(f'wrote {dp}\nwrote {dm}')
