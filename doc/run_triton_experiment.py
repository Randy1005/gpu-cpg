from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cuda-e2e-ms", type=float, required=True)
    parser.add_argument("--cuda-relax-ms", type=float, required=True)
    args = parser.parse_args()

    from triton_exp.benchmark import benchmark_e2e, benchmark_relax_only
    from triton_exp.gates import gate_correctness, gate_csr, gate_preliminary
    from triton_exp.io import build_backward_csr_np, load_dumps
    from triton_exp.kernel import run_suffix_bfs

    dumps = load_dumps(args.dump_dir)
    gate_preliminary(dumps)
    csr = build_backward_csr_np(dumps)
    gate_csr(csr, dumps.n_nodes, dumps.n_edges)
    triton_scaled = run_suffix_bfs(csr, dumps.n_nodes, args.block_size)
    _, max_diff = gate_correctness(triton_scaled, dumps)

    triton_relax_ms = benchmark_relax_only(args.dump_dir, args.block_size, args.warmup, args.runs)
    triton_e2e_ms = benchmark_e2e(args.dump_dir, args.block_size, args.warmup, args.runs)

    print("=== Triton Suffix BFS Experiment Summary ===")
    print(f"nodes={dumps.n_nodes}")
    print(f"edges={dumps.n_edges}")
    print(f"max_diff={max_diff:.6f}")
    print(f"cuda_e2e_ms={args.cuda_e2e_ms:.3f}")
    print(f"triton_e2e_ms={triton_e2e_ms:.3f}")
    print(f"e2e_ratio={triton_e2e_ms / args.cuda_e2e_ms:.3f}")
    print(f"cuda_relax_ms={args.cuda_relax_ms:.3f}")
    print(f"triton_relax_ms={triton_relax_ms:.3f}")
    print(f"kernel_ratio={triton_relax_ms / args.cuda_relax_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
