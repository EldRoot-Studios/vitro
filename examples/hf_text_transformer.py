"""Example: trace and record real pretrained HuggingFace text transformers (T5-small seq2seq & Qwen3-0.6B causal)."""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ROOT = os.path.join(ROOT, 'python')
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from vitro import SpanLabel, TokenMeta, TraceSpec, auto_adapter, export_demo, export_dump  # noqa: E402


def trace_t5(out_dir: str, model_id: str = 'google-t5/t5-small', local_only: bool = True) -> tuple[str, str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=local_only).eval()
    prompt_text = 'translate English to French: The glass box reveals every layer.'
    src = tokenizer(prompt_text, return_tensors='pt').input_ids

    adapter = auto_adapter(model, model_name=model_id)
    base_spec = TraceSpec(
        trace_id='t5_small_translation',
        title=f'{model_id} - English to French',
        model_name=model_id,
        unit_label='French Translation',
        stage_labels={'src_in': 'English Tokens', 'dec_in': 'French Prefix'},
        group_labels={'source': 'Prompt', 'encoder': 'T5 Encoder', 'decoder': 'T5 Decoder'},
        attn_names={'enc_self': 'Encoder Self-Attn', 'dec_self': 'Decoder Self-Attn', 'dec_cross': 'Cross-Attn'},
        token_meta=TokenMeta.from_tokenizer(tokenizer, vocab_size=model.config.vocab_size),
    )

    demo = adapter.record_generation(
        src, max_new_tokens=20, eos_id=tokenizer.eos_token_id, spec=base_spec, label='T5-Small Rollout'
    )
    ftok = torch.from_numpy(demo.ftok[None].astype('int64'))
    spans = [SpanLabel(0, 1, 'Decoder Start', kind='special', tint='cyan')]
    if len(demo.ftok) > 2:
        spans.append(SpanLabel(1, len(demo.ftok) - 1, 'Translated Output', kind='output', tint='gold'))
        spans.append(SpanLabel(len(demo.ftok) - 1, len(demo.ftok), 'EOS', kind='special', tint='violet'))
    spec = TraceSpec(
        trace_id=base_spec.trace_id, title=base_spec.title, model_name=base_spec.model_name,
        unit_label=base_spec.unit_label, stage_labels=base_spec.stage_labels,
        group_labels=base_spec.group_labels, attn_names=base_spec.attn_names,
        spans=spans, token_meta=base_spec.token_meta,
    )
    tr = adapter.trace_forward(src, ftok, spec=spec)
    os.makedirs(out_dir, exist_ok=True)
    dp = export_dump(os.path.join(out_dir, 't5_small.vtdump'), tr, spec=spec,
                     free_tokens=[demo.ftok], free_probs=[demo.fp])
    dm = export_demo(os.path.join(out_dir, 't5_small.vtdemo'), demo, spec=spec,
                     vocab_size=model.config.vocab_size)
    return dp, dm


def trace_causal(out_dir: str, model_id: str = 'Qwen/Qwen3-0.6B', local_only: bool = True,
                 prompt: str = 'Self-attention routes context across', max_new_tokens: int = 10) -> tuple[str, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only).eval()
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prompt_len = int(prompt_ids.shape[1])

    adapter = auto_adapter(model, model_name=model_id)
    base_spec = TraceSpec(
        trace_id='qwen3_0_6b_causal',
        title=f'{model_id} - Next-Token Continuation',
        model_name=model_id,
        unit_label='Continuation',
        stage_labels={'input': 'Prompt + Output'},
        group_labels={'decoder': 'Qwen3 Layers'},
        attn_names={'self': 'Causal Self-Attn (GQA avg)'},
        token_meta=TokenMeta.from_tokenizer(tokenizer, vocab_size=model.config.vocab_size),
    )

    demo = adapter.record_generation(
        prompt_ids, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id,
        spec=base_spec, label='Qwen3-0.6B Rollout',
    )
    ftok = torch.from_numpy(demo.ftok[None].astype('int64'))
    spans = [SpanLabel(0, prompt_len, 'Prompt', kind='prompt', tint='cyan')]
    if len(demo.ftok) > prompt_len:
        spans.append(SpanLabel(prompt_len, len(demo.ftok), 'Generated Tokens', kind='output', tint='gold'))
    spec = TraceSpec(
        trace_id=base_spec.trace_id, title=base_spec.title, model_name=base_spec.model_name,
        unit_label=base_spec.unit_label, stage_labels=base_spec.stage_labels,
        group_labels=base_spec.group_labels, attn_names=base_spec.attn_names,
        spans=spans, token_meta=base_spec.token_meta,
    )
    tr = adapter.trace_forward(ftok, spec=spec)
    os.makedirs(out_dir, exist_ok=True)
    dp = export_dump(os.path.join(out_dir, 'qwen3_0_6b.vtdump'), tr, spec=spec,
                     free_tokens=[demo.ftok], free_probs=[demo.fp])
    dm = export_demo(os.path.join(out_dir, 'qwen3_0_6b.vtdemo'), demo, spec=spec,
                     vocab_size=model.config.vocab_size)
    return dp, dm


def main() -> int:
    ap = argparse.ArgumentParser(description='Trace local HuggingFace text transformers into Vitro .vtdump / .vtdemo')
    ap.add_argument('--model', choices=('t5', 'qwen3', 'both'), default='both')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'samples'))
    ap.add_argument('--allow-download', action='store_true')
    args = ap.parse_args()
    local_only = not args.allow_download
    if args.model in ('t5', 'both'):
        for p in trace_t5(args.out_dir, local_only=local_only):
            print('wrote', p)
    if args.model in ('qwen3', 'both'):
        for p in trace_causal(args.out_dir, local_only=local_only):
            print('wrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
