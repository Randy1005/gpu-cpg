from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCALE_UP = 10000


@dataclass(frozen=True)
class GpuCpgDumps:
    suffix_scaled: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_weights_scaled: np.ndarray
    levels: np.ndarray
    level_order: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.suffix_scaled.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])


@dataclass(frozen=True)
class Csr:
    row_ptr: np.ndarray
    col_idx: np.ndarray
    weights_scaled: np.ndarray
    level_groups: list[np.ndarray]
    max_level: int
    max_out_degree: int


def _scaled(value: str) -> int:
    return int(round(float(value) * SCALE_UP))


def load_dumps(directory: str | Path) -> GpuCpgDumps:
    d = Path(directory)

    suffix_entries: dict[int, int] = {}
    for line in (d / "benchmark_sfxt.txt").read_text(encoding="utf-8").splitlines():
        node, dist = line.split()
        suffix_entries[int(node)] = _scaled(dist)

    if not suffix_entries:
        raise ValueError("benchmark_sfxt.txt is empty")

    n_nodes = max(suffix_entries) + 1
    suffix = np.zeros(n_nodes, dtype=np.int32)
    for node, dist in suffix_entries.items():
        suffix[node] = dist

    src: list[int] = []
    dst: list[int] = []
    wts: list[int] = []
    for line in (d / "benchmark_edges_tfm.txt").read_text(encoding="utf-8").splitlines():
        s, t, w = line.split()
        src.append(int(s))
        dst.append(int(t))
        wts.append(_scaled(w))

    levels_entries: dict[int, int] = {}
    for line in (d / "benchmark_levels.txt").read_text(encoding="utf-8").splitlines():
        node, level = line.split()
        levels_entries[int(node)] = int(level)

    if len(levels_entries) != n_nodes:
        raise ValueError(f"levels cover {len(levels_entries)} nodes but suffix has {n_nodes}")

    levels = np.zeros(n_nodes, dtype=np.int32)
    for node, level in levels_entries.items():
        levels[node] = level

    order = [
        int(line)
        for line in (d / "benchmark_level_order.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(order) != n_nodes:
        raise ValueError(f"level order has {len(order)} nodes but suffix has {n_nodes}")

    return GpuCpgDumps(
        suffix_scaled=suffix,
        edge_src=np.asarray(src, dtype=np.int32),
        edge_dst=np.asarray(dst, dtype=np.int32),
        edge_weights_scaled=np.asarray(wts, dtype=np.int32),
        levels=levels,
        level_order=np.asarray(order, dtype=np.int32),
    )


def build_backward_csr_np(dumps: GpuCpgDumps) -> Csr:
    order = np.argsort(dumps.edge_src, kind="stable")
    src = dumps.edge_src[order]
    dst = dumps.edge_dst[order]
    weights = dumps.edge_weights_scaled[order]

    row_ptr = np.zeros(dumps.n_nodes + 1, dtype=np.int32)
    for s in src:
        row_ptr[int(s) + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    max_level = int(dumps.levels.max()) if dumps.n_nodes else 0
    level_groups = [
        np.where(dumps.levels == level)[0].astype(np.int32)
        for level in range(max_level + 1)
    ]
    degrees = row_ptr[1:] - row_ptr[:-1]

    return Csr(
        row_ptr=row_ptr,
        col_idx=dst.astype(np.int32),
        weights_scaled=weights.astype(np.int32),
        level_groups=level_groups,
        max_level=max_level,
        max_out_degree=int(degrees.max()) if degrees.size else 0,
    )
