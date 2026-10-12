"""Repeatable GPU smoke benchmark; no project data or checkpoints are modified."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Optional Hugging Face generation benchmark (downloads weights).')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--output', default='gpu_benchmark.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available in this Python environment.')
    torch.manual_seed(42)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda_runtime': torch.version.cuda,
              'packages': {m: bool(importlib.util.find_spec(m)) for m in ['transformers', 'peft', 'accelerate']}}
    print(json.dumps(report, indent=2), flush=True)
    n = 8192
    a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    with torch.inference_mode():
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    report['bf16_matmul'] = {'matrix_size': n, 'iterations': 50,
                            'seconds': elapsed, 'tflops': 2*n**3*50/elapsed/1e12}
    print(json.dumps(report['bf16_matmul'], indent=2), flush=True)
    del a, b, c
    torch.cuda.empty_cache()
    if args.model:
        # Interactive pod shells may enable an optional downloader that is not
        # installed in this isolated benchmark environment. Fall back before
        # importing Hugging Face, which reads this setting at import time.
        if (os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '').upper() in {'1', 'ON', 'YES', 'TRUE'}
                and importlib.util.find_spec('hf_transfer') is None):
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
            print('hf_transfer is unavailable; using the standard downloader.', flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        start = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        report['model'] = args.model
        report['download_and_load_seconds'] = time.perf_counter() - start
        report['generation'] = []
        prompts = ['Explain why the sky looks blue in simple language.',
                   'বাংলায় ব্যাখ্যা করুন: আকাশ নীল দেখায় কেন?']
        with torch.inference_mode():
            for batch in args.batches:
                texts = [tok.apply_chat_template([{'role': 'user', 'content': prompts[i % 2]}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                         for i in range(batch)]
                inputs = tok(texts, return_tensors='pt', padding=True).to('cuda')
                model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    # Fixed work for throughput comparison: force exactly N output tokens.
                    out = model.generate(**inputs, min_new_tokens=args.tokens, max_new_tokens=args.tokens,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                seconds = sorted(timings)[1]
                row = {'batch': batch, 'padded_input_tokens': inputs.input_ids.shape[1],
                       'output_tokens_per_sequence': args.tokens, 'seconds_median': seconds,
                       'tokens_per_second': batch*args.tokens/seconds,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                       'timings_seconds': timings,
                       'sample': tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)}
                report['generation'].append(row)
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved ' + args.output, flush=True)


if __name__ == '__main__':
    main()
