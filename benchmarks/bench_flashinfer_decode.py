#!/usr/bin/env python3
import argparse
import math
import random
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FlashInfer paged dense decode attention."
    )
    parser.add_argument("top_k_pages", type=int, nargs="?", default=8)
    parser.add_argument("batch_size", type=int, nargs="?", default=16)
    parser.add_argument("min_ctx", type=int, nargs="?", default=512)
    parser.add_argument("max_ctx", type=int, nargs="?", default=1024)
    parser.add_argument("seed", type=int, nargs="?", default=7)
    parser.add_argument("iterations", type=int, nargs="?", default=10)
    parser.add_argument("warmup", type=int, nargs="?", default=2)
    parser.add_argument("num_heads", type=int, nargs="?", default=32)
    parser.add_argument("head_dim", type=int, nargs="?", default=128)
    parser.add_argument("page_size", type=int, nargs="?", default=16)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="KV/query dtype for FlashInfer benchmark.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="FlashInfer backend override, e.g. auto/fa2/fa3/cudnn.",
    )
    return parser.parse_args()


def make_context_lengths(batch_size, min_ctx, max_ctx, seed):
    rng = random.Random(seed)
    return [rng.randint(min_ctx, max_ctx) for _ in range(batch_size)]


def ceil_div(a, b):
    return (a + b - 1) // b


def main():
    args = parse_args()
    import flashinfer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run FlashInfer benchmark.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device("cuda:0")

    context_lengths = make_context_lengths(
        args.batch_size, args.min_ctx, args.max_ctx, args.seed
    )
    page_counts = [ceil_div(x, args.page_size) for x in context_lengths]
    total_pages = sum(page_counts)
    total_tokens = sum(context_lengths)
    selected_tokens = args.batch_size * args.top_k_pages * args.page_size

    kv_page_indptr_host = [0]
    for pages in page_counts:
        kv_page_indptr_host.append(kv_page_indptr_host[-1] + pages)
    kv_last_page_len_host = [
        ((ctx - 1) % args.page_size) + 1 if ctx > 0 else 1 for ctx in context_lengths
    ]

    kv_page_indptr = torch.tensor(
        kv_page_indptr_host, dtype=torch.int32, device=device
    )
    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        kv_last_page_len_host, dtype=torch.int32, device=device
    )

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend=args.backend
    )

    kv_cache = torch.randn(
        total_pages,
        2,
        args.page_size,
        args.num_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )

    torch.cuda.synchronize()
    plan_start = time.perf_counter()
    wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        args.num_heads,
        args.num_heads,
        args.head_dim,
        args.page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
    )
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - plan_start) * 1000.0

    for _ in range(args.warmup):
        q = torch.randn(
            args.batch_size,
            args.num_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        _ = wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(args.iterations):
        q = torch.randn(
            args.batch_size,
            args.num_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        start.record()
        out = wrapper.run(q, kv_cache)
        end.record()
        end.synchronize()
        total_ms += start.elapsed_time(end)

    avg_ms = total_ms / max(args.iterations, 1)
    bytes_dense = 2 * total_tokens * args.num_heads * args.head_dim * 4
    h100_bw = 3.35e12
    bw_floor_ms = (bytes_dense / h100_bw) * 1e3

    print("== FlashInfer Decode Benchmark ==")
    print(
        f"batch_size={args.batch_size} total_pages={total_pages} "
        f"top_k_pages={args.top_k_pages} num_heads={args.num_heads} "
        f"head_dim={args.head_dim} page_size={args.page_size} "
        f"min_ctx={args.min_ctx} max_ctx={args.max_ctx} "
        f"seed={args.seed} iterations={args.iterations} warmup={args.warmup} "
        f"dtype={args.dtype} backend={args.backend}"
    )
    print()
    print("== Batch Stats ==")
    print(f"total_context_tokens={total_tokens}")
    print(f"total_selected_tokens={selected_tokens}")
    print(f"dense_bytes_mb={bytes_dense / (1024.0 * 1024.0):.3f}")
    print(f"h100_dense_bw_floor_ms={bw_floor_ms:.6f}")
    print()
    print("== FlashInfer Timings ==")
    print(f"plan_ms={plan_ms:.3f}")
    print(f"run_ms={avg_ms:.3f}")
    print(f"output_shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
