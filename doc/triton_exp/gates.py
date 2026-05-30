from __future__ import annotations

import numpy as np

from .io import Csr, GpuCpgDumps


def gate_preliminary(dumps: GpuCpgDumps) -> None:
    if dumps.suffix_scaled.shape[0] != dumps.levels.shape[0]:
        raise ValueError("suffix and levels length mismatch")
    if np.any(dumps.edge_src < 0) or np.any(dumps.edge_dst < 0):
        raise ValueError("negative vertex id in edges")
    if np.any(dumps.edge_src >= dumps.n_nodes) or np.any(dumps.edge_dst >= dumps.n_nodes):
        raise ValueError("edge endpoint out of range")
    for src, dst in zip(dumps.edge_src, dumps.edge_dst):
        if dumps.levels[int(src)] <= dumps.levels[int(dst)]:
            raise ValueError(f"level violation on edge {src}->{dst}")


def gate_csr(csr: Csr, n_nodes: int, n_edges: int) -> None:
    if csr.row_ptr.size != n_nodes + 1:
        raise ValueError(f"row_ptr.size={csr.row_ptr.size} != n_nodes + 1={n_nodes + 1}")
    if not np.all(csr.row_ptr[:-1] <= csr.row_ptr[1:]):
        raise ValueError("row_ptr is not monotone")
    if int(csr.row_ptr[-1]) != n_edges:
        raise ValueError(f"row_ptr[-1]={csr.row_ptr[-1]} != n_edges={n_edges}")
    if csr.col_idx.size and (csr.col_idx.min() < 0 or csr.col_idx.max() >= n_nodes):
        raise ValueError("col_idx out of range")
    if sum(len(g) for g in csr.level_groups) != n_nodes:
        raise ValueError("level groups do not cover all nodes")
    for sink in csr.level_groups[0]:
        if csr.row_ptr[int(sink) + 1] - csr.row_ptr[int(sink)] != 0:
            raise ValueError(f"sink {sink} has outgoing edge")


def gate_correctness(triton_scaled: np.ndarray, dumps: GpuCpgDumps) -> tuple[int, float]:
    if triton_scaled.shape != dumps.suffix_scaled.shape:
        raise ValueError(
            f"triton and gpu_cpg suffix shape mismatch: "
            f"{triton_scaled.shape} != {dumps.suffix_scaled.shape}"
        )
    diff = np.abs(triton_scaled.astype(np.int64) - dumps.suffix_scaled.astype(np.int64))
    max_scaled = int(diff.max()) if diff.size else 0
    max_float = max_scaled / 10000.0
    if max_scaled != 0:
        worst = int(np.argmax(diff))
        raise ValueError(
            f"max_scaled_diff={max_scaled}; worst node={worst}; "
            f"triton={triton_scaled[worst]} gpu_cpg={dumps.suffix_scaled[worst]}"
        )
    return max_scaled, max_float
