from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from .io import build_backward_csr_np, load_dumps
from .kernel import run_suffix_bfs


def median_ms(values: list[float]) -> float:
    return statistics.median(values) * 1000.0


def benchmark_relax_only(directory: str | Path, block_size: int, warmup: int, runs: int) -> float:
    dumps = load_dumps(directory)
    csr = build_backward_csr_np(dumps)
    for _ in range(warmup):
        run_suffix_bfs(csr, dumps.n_nodes, block_size)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_suffix_bfs(csr, dumps.n_nodes, block_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return median_ms(times)


def benchmark_e2e(directory: str | Path, block_size: int, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        dumps = load_dumps(directory)
        csr = build_backward_csr_np(dumps)
        run_suffix_bfs(csr, dumps.n_nodes, block_size)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dumps = load_dumps(directory)
        csr = build_backward_csr_np(dumps)
        run_suffix_bfs(csr, dumps.n_nodes, block_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return median_ms(times)
