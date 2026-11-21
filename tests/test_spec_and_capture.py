"""Checks universal PyTorch tracing across Decoder-Only, Encoder-Decoder, HuggingFace, and custom user labels."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ROOT = os.path.join(ROOT, 'python')
EX_ROOT = os.path.join(ROOT, 'examples')
for p in (PY_ROOT, EX_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tiny_decoder_only import TinyGPT, make_vocab  # noqa: E402
from tiny_encoder_decoder import TinySeq2Seq  # noqa: E402
from vitro import (  # noqa: E402
    DecoderOnlyAdapter,
    EncoderDecoderAdapter,
    HuggingFaceAdapter,
    SpanLabel,
    TokenMeta,
    TraceSpec,
    auto_adapter,
    compute_token_stats,
)


def check_decoder_only() -> None:
    torch.manual_seed(1)
    vocab = make_vocab()
    model = TinyGPT(vocab=len(vocab), d=32, layers=2, heads=2).eval()
    adapter = auto_adapter(model)
    assert isinstance(adapter, DecoderOnlyAdapter)
    spec = TraceSpec(
        model_name='CustomGPT',
        stage_labels={'input': 'User Input', 'block_0': 'Custom L0', 'block_1': 'Custom L1'},
        group_labels={'decoder': 'Custom Stack'},
        attn_names={'self': 'Custom Self'},
        spans=[SpanLabel(0, 4, 'Prefix', kind='prompt'), SpanLabel(4, 10, 'Body', kind='code', parent=0)],
        token_meta=TokenMeta(labels=vocab, tag_names=['tok'], token_tags=[0] * 10),
    )
    tok = torch.randint(0, len(vocab), (10,))
    tr = adapter.trace_forward(tok, spec=spec)
    assert tr.check_logit_diff < 1e-5, f'hooked logits differed by {tr.check_logit_diff}'
    assert [s.label for s, _ in tr.stages] == ['User Input', 'Custom L0', 'Custom L1']
    assert all(s.group == 'Custom Stack' for s, _ in tr.stages[1:])
    attn_links = [(lk, w) for lk, w in tr.links if w is not None]
    assert len(attn_links) == 2 and all(lk.family == 'Custom Self' for lk, _ in attn_links)
    for lk, w in attn_links:
        assert w.shape == (2, 9, 9)
        assert np.allclose(w.astype(np.float32).sum(-1), 1.0, atol=2e-3)
        # Strict causal mask check: j > i is zero
        for i in range(9):
            assert float(w[:, i, i + 1:].max(initial=0.0)) == 0.0
    st = compute_token_stats(tr.logits, tok[1:])
    assert st['rank'].shape == (9,) and (st['rank'] >= 1).all()
    print('test_capture_universal: DecoderOnlyAdapter parity & causality verified', flush=True)


def check_encoder_decoder() -> None:
    torch.manual_seed(2)
    model = TinySeq2Seq(vocab=24, d=32, heads=2, n_enc=2, n_dec=2).eval()
    adapter = auto_adapter(model)
    assert isinstance(adapter, EncoderDecoderAdapter)
    src = torch.tensor([1, 4, 6, 8, 2])
    tgt = torch.tensor([1, 5, 7, 9, 10, 2])
    spec = TraceSpec(stage_labels={'enc_0': 'MyEnc0', 'dec_1': 'MyDec1'}, attn_names={'dec_cross': 'MyCross'})
    tr = adapter.trace_forward(src, tgt, spec=spec)
    assert tr.check_logit_diff < 1e-5, f'seq2seq hooked logits differed by {tr.check_logit_diff}'
    names = [s.label for s, _ in tr.stages]
    assert 'MyEnc0' in names and 'MyDec1' in names
    cross = [(lk, w) for lk, w in tr.links if lk.kind == 'cross']
    assert len(cross) == 2 and all(lk.family == 'MyCross' for lk, _ in cross)
    for _, w in cross:
        assert w.shape == (2, 5, 5)
        assert np.allclose(w.astype(np.float32).sum(-1), 1.0, atol=2e-3)
    print('test_capture_universal: EncoderDecoderAdapter parity & cross-attn verified', flush=True)


def check_huggingface_offline() -> None:
    from transformers import BartConfig, BartForConditionalGeneration, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(3)
    gpt_cfg = GPT2Config(vocab_size=32, n_positions=32, n_embd=32, n_layer=2, n_head=2, attn_implementation='eager')
    gpt = GPT2LMHeadModel(gpt_cfg).eval()
    ha = auto_adapter(gpt)
    assert isinstance(ha, HuggingFaceAdapter)
    tok = torch.randint(0, 32, (8,))
    tr = ha.trace_forward(tok, spec=TraceSpec(model_name='HF-GPT2-Tiny', stage_labels={'block_0': 'HF Block 0'}))
    assert tr.check_logit_diff < 1e-5, f'HF GPT2 logit diff {tr.check_logit_diff}'
    assert [s.label for s, _ in tr.stages] == ['Tokens', 'HF Block 0', 'Block 2']
    assert tr.links[1][1].shape == (2, 7, 7)
    dm_gpt = ha.record_generation(tok[:4], max_new_tokens=4, eos_id=0, spec=TraceSpec(model_name='HF-GPT2-Tiny'))
    assert dm_gpt.kind == 'causal' and len(dm_gpt.ftok) > 4 and dm_gpt.self_w.shape[1] == 2

    bart_cfg = BartConfig(vocab_size=32, max_position_embeddings=32, d_model=32, encoder_layers=2, decoder_layers=2,
                          encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                          attn_implementation='eager')
    bart = BartForConditionalGeneration(bart_cfg).eval()
    hb = auto_adapter(bart)
    assert isinstance(hb, HuggingFaceAdapter) and hb.is_enc_dec
    src = torch.tensor([1, 5, 6, 2])
    tgt = torch.tensor([1, 7, 8, 9, 2])
    trb = hb.trace_forward(src, tgt, spec=TraceSpec(attn_names={'dec_cross': 'BART Cross'}))
    assert trb.check_logit_diff < 1e-5
    cross = [w for lk, w in trb.links if lk.kind == 'cross']
    assert len(cross) == 2 and cross[0].shape == (2, 4, 4)
    dm_bart = hb.record_generation(src, max_new_tokens=4, eos_id=2, spec=TraceSpec(model_name='HF-BART-Tiny'))
    assert dm_bart.kind == 'seq2seq' and dm_bart.has_read and dm_bart.cross_w is not None and len(dm_bart.prefill_links) == 2
    print('test_capture_universal: offline HuggingFace GPT2 & BART adapters (+ generation) verified', flush=True)


def check_local_pretrained_text() -> None:
    import tempfile
    from hf_text_transformer import trace_causal, trace_t5
    from vitro.container import ContainerFile

    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained('google-t5/t5-small', local_files_only=True)
        has_t5 = True
    except Exception:
        has_t5 = False
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
        has_qwen = True
    except Exception:
        has_qwen = False

    if not (has_t5 or has_qwen):
        print('test_capture_universal: local HF pretrained weights not cached (skipping live weight check)', flush=True)
        return
    with tempfile.TemporaryDirectory() as tmp:
        if has_t5:
            dp, dm = trace_t5(tmp, local_only=True)
            assert ContainerFile(dp).header['units'][0]['check_logit_diff'] < 1e-5
            assert ContainerFile(dm).header['trace']['arch'] == 'encoder_decoder'
        if has_qwen:
            qp, qm = trace_causal(tmp, max_new_tokens=3, local_only=True)
            assert ContainerFile(qp).header['units'][0]['check_logit_diff'] < 1e-5
            assert ContainerFile(qm).header['trace']['arch'] == 'decoder_only'
        print(f'test_capture_universal: live cached text transformers verified (t5={has_t5}, qwen3={has_qwen})', flush=True)


def run() -> None:
    check_decoder_only()
    check_encoder_decoder()
    check_huggingface_offline()
    check_local_pretrained_text()


if __name__ == '__main__':
    run()
