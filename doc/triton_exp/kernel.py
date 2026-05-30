from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .io import Csr

INT_MAX = 2147483647


@triton.jit
def suffix_bfs_kernel(row_ptr, col_idx, weights, suffix, nodes, n_nodes_level,
                      BLOCK_SIZE: tl.constexpr, MAX_DEGREE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_nodes_level
    node_ids = tl.load(nodes + offsets, mask=mask, other=0)
    row_start = tl.load(row_ptr + node_ids, mask=mask, other=0)
    row_end = tl.load(row_ptr + node_ids + 1, mask=mask, other=0)
    degree = row_end - row_start
    min_val = tl.full([BLOCK_SIZE], INT_MAX, dtype=tl.int32)

    for d in range(MAX_DEGREE):
        valid = (d < degree) & mask
        edge_idx = row_start + d
        succ = tl.load(col_idx + edge_idx, mask=valid, other=0)
        w = tl.load(weights + edge_idx, mask=valid, other=INT_MAX)
        succ_dist = tl.load(suffix + succ, mask=valid, other=INT_MAX)
        reachable = valid & (succ_dist != INT_MAX)
        safe_dist = tl.where(reachable, succ_dist, 0)
        candidate = tl.where(reachable, w + safe_dist, INT_MAX)
        min_val = tl.where(reachable & (candidate < min_val), candidate, min_val)

    tl.store(suffix + node_ids, min_val, mask=mask)


def to_gpu(csr: Csr) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(csr.row_ptr, dtype=torch.int32, device="cuda"),
        torch.tensor(csr.col_idx, dtype=torch.int32, device="cuda"),
        torch.tensor(csr.weights_scaled, dtype=torch.int32, device="cuda"),
    )


def run_suffix_bfs(csr: Csr, n_nodes: int, block_size: int = 128) -> np.ndarray:
    row_ptr, col_idx, weights = to_gpu(csr)
    suffix = torch.full((n_nodes,), INT_MAX, dtype=torch.int32, device="cuda")
    for sink in csr.level_groups[0]:
        suffix[int(sink)] = 0

    max_degree = 1
    while max_degree < max(1, csr.max_out_degree):
        max_degree *= 2

    for level in range(1, len(csr.level_groups)):
        group = csr.level_groups[level]
        if group.size == 0:
            continue
        nodes = torch.tensor(group, dtype=torch.int32, device="cuda")
        grid = (triton.cdiv(group.size, block_size),)
        suffix_bfs_kernel[grid](
            row_ptr, col_idx, weights, suffix, nodes, group.size,
            BLOCK_SIZE=block_size, MAX_DEGREE=max_degree)

    torch.cuda.synchronize()
    return suffix.cpu().numpy()
