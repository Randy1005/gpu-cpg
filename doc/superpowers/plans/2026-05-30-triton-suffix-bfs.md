# Triton Suffix BFS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Triton suffix-distance experiment that compares correctness and runtime against gpu-cpg's CUDA suffix-distance build.

**Architecture:** Add a small gpu-cpg dump/timing executable that emits the exact inputs and reference outputs needed by Python. Keep Triton code in focused Python modules: dump loading/CSR construction, kernel execution, correctness gates, and performance reporting. Use scaled integer distances to match gpu-cpg's `SCALE_UP=10000` semantics.

**Tech Stack:** C++20/CUDA, existing gpu-cpg `CpGen`, Python 3.9+, NumPy, PyTorch CUDA tensors, Triton, pytest.

---

## File Structure

- Modify: `/home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh`
  - Add read-only dump helpers for suffix distances, fanout weighted edges, per-node levels, and level order.
- Modify: `/home/cchang289/Research/gpu-cpg/examples/CMakeLists.txt`
  - Add `triton-sfxt-dump` executable.
- Create: `/home/cchang289/Research/gpu-cpg/examples/triton-sfxt-dump.cu`
  - Runs suffix-distance build, writes dumps, prints CUDA e2e/relax timings.
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/__init__.py`
  - Package marker.
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/io.py`
  - Load gpu-cpg dumps and build scaled-int CSR/levels.
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/kernel.py`
  - Triton suffix relaxation kernel and runner.
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/gates.py`
  - Gate P, Gate 1, Gate 2 validation functions.
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/benchmark.py`
  - Triton e2e and relax-only timing helpers.
- Create: `/home/cchang289/Research/gpu-cpg/doc/run_triton_experiment.py`
  - CLI orchestration for one benchmark dump directory.
- Create: `/home/cchang289/Research/gpu-cpg/doc/tests/test_triton_io.py`
  - CPU-only tests for dump parsing, CSR, levels, scaled-int behavior.

---

### Task 1: Add CPU-Only Python Dump Parsing Tests

**Files:**
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/__init__.py`
- Create: `/home/cchang289/Research/gpu-cpg/doc/tests/test_triton_io.py`
- Create later: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/io.py`

- [ ] **Step 1: Create package marker**

```python
# /home/cchang289/Research/gpu-cpg/doc/triton_exp/__init__.py
```

- [ ] **Step 2: Write failing tests**

```python
# /home/cchang289/Research/gpu-cpg/doc/tests/test_triton_io.py
from pathlib import Path

import numpy as np

from triton_exp.io import SCALE_UP, build_backward_csr_np, load_dumps


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_dumps_preserves_scaled_int_semantics(tmp_path):
    write(tmp_path / "benchmark_sfxt.txt", "0 0\n1 1.2345\n2 2.0001\n")
    write(tmp_path / "benchmark_edges_tfm.txt", "2 1 0.5000\n1 0 1.2345\n")
    write(tmp_path / "benchmark_levels.txt", "0 0\n1 1\n2 2\n")
    write(tmp_path / "benchmark_level_order.txt", "0\n1\n2\n")

    dumps = load_dumps(tmp_path)

    assert SCALE_UP == 10000
    assert dumps.n_nodes == 3
    assert dumps.suffix_scaled.tolist() == [0, 12345, 20001]
    assert dumps.edge_weights_scaled.tolist() == [5000, 12345]
    assert dumps.levels.tolist() == [0, 1, 2]
    assert dumps.level_order.tolist() == [0, 1, 2]


def test_build_backward_csr_groups_successors_by_source(tmp_path):
    write(tmp_path / "benchmark_sfxt.txt", "0 0\n1 1\n2 2\n3 3\n")
    write(tmp_path / "benchmark_edges_tfm.txt", "2 1 0.5\n2 0 1.5\n3 2 2.0\n")
    write(tmp_path / "benchmark_levels.txt", "0 0\n1 0\n2 1\n3 2\n")
    write(tmp_path / "benchmark_level_order.txt", "0\n1\n2\n3\n")

    dumps = load_dumps(tmp_path)
    csr = build_backward_csr_np(dumps)

    assert csr.row_ptr.tolist() == [0, 0, 0, 2, 3]
    assert csr.col_idx.tolist() == [1, 0, 2]
    assert csr.weights_scaled.tolist() == [5000, 15000, 20000]
    assert [group.tolist() for group in csr.level_groups] == [[0, 1], [2], [3]]
    assert int(csr.max_out_degree) == 2


def test_load_dumps_rejects_missing_vertex_level(tmp_path):
    write(tmp_path / "benchmark_sfxt.txt", "0 0\n1 1\n")
    write(tmp_path / "benchmark_edges_tfm.txt", "1 0 1\n")
    write(tmp_path / "benchmark_levels.txt", "0 0\n")
    write(tmp_path / "benchmark_level_order.txt", "0\n1\n")

    try:
        load_dumps(tmp_path)
    except ValueError as exc:
        assert "levels cover 1 nodes but suffix has 2" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 3: Run tests and verify red**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 -m pytest tests/test_triton_io.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'triton_exp.io'`.

---

### Task 2: Implement Dump Parsing and CSR Construction

**Files:**
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/io.py`
- Test: `/home/cchang289/Research/gpu-cpg/doc/tests/test_triton_io.py`

- [ ] **Step 1: Implement `io.py`**

```python
# /home/cchang289/Research/gpu-cpg/doc/triton_exp/io.py
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
```

- [ ] **Step 2: Run tests and verify green**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 -m pytest tests/test_triton_io.py -q
```

Expected: `3 passed`.

---

### Task 3: Add gpu-cpg Dump and Timing Executable

**Files:**
- Modify: `/home/cchang289/Research/gpu-cpg/gpucpg/gpucpg.cuh`
- Create: `/home/cchang289/Research/gpu-cpg/examples/triton-sfxt-dump.cu`
- Modify: `/home/cchang289/Research/gpu-cpg/examples/CMakeLists.txt`

- [ ] **Step 1: Add dump helpers to `CpGen`**

Insert these public methods near existing `dump_succ_dists()` in `gpucpg.cuh`:

```cpp
  void dump_sfxt_by_vertex(std::ostream& os) const {
    for (size_t i = 0; i < _h_dists.size(); ++i) {
      os << i << ' ' << static_cast<float>(_h_dists[i]) / SCALE_UP << '\n';
    }
  }

  void dump_fanout_edges_tfm(std::ostream& os) const {
    for (size_t src = 0; src + 1 < _h_fanout_adjp.size(); ++src) {
      for (int e = _h_fanout_adjp[src]; e < _h_fanout_adjp[src + 1]; ++e) {
        os << src << ' ' << _h_fanout_adjncy[e] << ' ' << _h_fanout_wgts[e] << '\n';
      }
    }
  }

  void dump_node_levels(std::ostream& os) const {
    for (size_t level = 0; level + 1 < _h_verts_lvlp.size(); ++level) {
      for (int pos = _h_verts_lvlp[level]; pos < _h_verts_lvlp[level + 1]; ++pos) {
        os << _h_queue[pos] << ' ' << level << '\n';
      }
    }
  }

  void dump_level_order(std::ostream& os) const {
    for (const auto& v : _h_queue) {
      os << v << '\n';
    }
  }
```

- [ ] **Step 2: Add executable source**

```cpp
// /home/cchang289/Research/gpu-cpg/examples/triton-sfxt-dump.cu
#include "gpucpg.cuh"

#include <filesystem>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: triton-sfxt-dump [benchmark] [k] [output_dir]\n";
    return EXIT_FAILURE;
  }

  const std::string benchmark = argv[1];
  const int k = std::stoi(argv[2]);
  const std::filesystem::path out_dir = argv[3];
  std::filesystem::create_directories(out_dir);

  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);
  cpgen.report_paths(
    k, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX,
    gpucpg::PfxtExpMethod::SHORT_LONG,
    false, 0.005f, 5.0f, 8,
    false, false, false, true,
    gpucpg::CsrReorderMethod::E_ORIENTED,
    true);

  {
    std::ofstream os(out_dir / "benchmark_sfxt.txt");
    cpgen.dump_sfxt_by_vertex(os);
  }
  {
    std::ofstream os(out_dir / "benchmark_edges_tfm.txt");
    cpgen.dump_fanout_edges_tfm(os);
  }
  {
    std::ofstream os(out_dir / "benchmark_levels.txt");
    cpgen.dump_node_levels(os);
  }
  {
    std::ofstream os(out_dir / "benchmark_level_order.txt");
    cpgen.dump_level_order(os);
  }

  const auto cuda_e2e_ms =
    (cpgen.lvlize_time + cpgen.relax_time) / 1ms;
  const auto cuda_relax_ms = cpgen.relax_time / 1ms;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "nodes=" << cpgen.num_verts() << '\n';
  std::cout << "edges=" << cpgen.num_edges() << '\n';
  std::cout << "cuda_e2e_ms=" << cuda_e2e_ms << '\n';
  std::cout << "cuda_relax_ms=" << cuda_relax_ms << '\n';
  return EXIT_SUCCESS;
}
```

- [ ] **Step 3: Register executable in CMake**

Add:

```cmake
add_executable(triton-sfxt-dump ${GPUCPG_EXAMPLE_DIR}/triton-sfxt-dump.cu)
```

Append `triton-sfxt-dump` to `GPUCPG_EXAMPLES`.

- [ ] **Step 4: Build only the new executable**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg
cmake --build build --target triton-sfxt-dump -j
```

Expected: target builds without compile errors.

---

### Task 4: Add Triton Kernel and Correctness Gates

**Files:**
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/kernel.py`
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/gates.py`

- [ ] **Step 1: Implement Triton kernel**

```python
# /home/cchang289/Research/gpu-cpg/doc/triton_exp/kernel.py
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
        candidate = w + succ_dist
        min_val = tl.where(valid & (candidate < min_val), candidate, min_val)

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
```

- [ ] **Step 2: Implement gates**

```python
# /home/cchang289/Research/gpu-cpg/doc/triton_exp/gates.py
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
    for src, dst in zip(dumps.edge_src[:10000], dumps.edge_dst[:10000]):
        if dumps.levels[int(src)] <= dumps.levels[int(dst)]:
            raise ValueError(f"level violation on edge {src}->{dst}")


def gate_csr(csr: Csr, n_nodes: int, n_edges: int) -> None:
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
```

- [ ] **Step 3: Run import check**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 - <<'PY'
from triton_exp.io import load_dumps, build_backward_csr_np
from triton_exp.kernel import run_suffix_bfs
from triton_exp.gates import gate_preliminary, gate_csr, gate_correctness
print("imports ok")
PY
```

Expected: `imports ok`.

---

### Task 5: Add Benchmark CLI

**Files:**
- Create: `/home/cchang289/Research/gpu-cpg/doc/triton_exp/benchmark.py`
- Create: `/home/cchang289/Research/gpu-cpg/doc/run_triton_experiment.py`

- [ ] **Step 1: Implement benchmark helpers**

```python
# /home/cchang289/Research/gpu-cpg/doc/triton_exp/benchmark.py
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
```

- [ ] **Step 2: Implement CLI**

```python
# /home/cchang289/Research/gpu-cpg/doc/run_triton_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path

from triton_exp.benchmark import benchmark_e2e, benchmark_relax_only
from triton_exp.gates import gate_correctness, gate_csr, gate_preliminary
from triton_exp.io import build_backward_csr_np, load_dumps
from triton_exp.kernel import run_suffix_bfs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cuda-e2e-ms", type=float, required=True)
    parser.add_argument("--cuda-relax-ms", type=float, required=True)
    args = parser.parse_args()

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
```

- [ ] **Step 3: Run CLI help**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 run_triton_experiment.py --help
```

Expected: help output lists `dump_dir`, `--cuda-e2e-ms`, `--cuda-relax-ms`.

---

### Task 6: Run vga_lcd Gate Pass

**Files:**
- Uses built executable and Python CLI.

- [ ] **Step 1: Generate dumps**

Run:

```bash
cd /home/cchang289/Research/gpu-cpg
./build/examples/triton-sfxt-dump benchmarks/vga_lcd_random_wgts_dense40.txt 1000 doc/triton_runs/vga_lcd
```

Expected output includes:

```text
nodes=...
edges=...
cuda_e2e_ms=...
cuda_relax_ms=...
```

- [ ] **Step 2: Run Triton gates and benchmark**

Substitute CUDA numbers printed by Step 1:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 run_triton_experiment.py triton_runs/vga_lcd \
  --cuda-e2e-ms <CUDA_E2E_MS> \
  --cuda-relax-ms <CUDA_RELAX_MS>
```

Expected:

```text
=== Triton Suffix BFS Experiment Summary ===
max_diff=0.000000
e2e_ratio=...
kernel_ratio=...
```

- [ ] **Step 3: Stop if correctness fails**

If `gate_correctness` raises `ValueError`, inspect:

```bash
cd /home/cchang289/Research/gpu-cpg/doc
python3 -m pytest tests/test_triton_io.py -q
```

Expected: Python parser tests still pass. Then inspect max out-degree and level order before changing kernel.

---

## Self-Review

Spec coverage:
- gpu-cpg dumps: Task 3.
- scaled-int correctness: Tasks 1, 2, 4.
- CSR construction and integrity: Tasks 1, 2, 4.
- Triton kernel: Task 4.
- primary e2e and secondary relax-only metrics: Task 5.
- vga_lcd gate before larger benchmarks: Task 6.

Known gap:
- This plan executes `vga_lcd` first. After it passes, repeat Task 6 for `leon2` and `netcard` with their benchmark file paths and add rows to the summary table in `triton-experiment.md` or a separate run log.

No placeholders: commands, files, and code snippets are concrete. `<CUDA_E2E_MS>` and `<CUDA_RELAX_MS>` are runtime values printed by the preceding command, not design placeholders.
