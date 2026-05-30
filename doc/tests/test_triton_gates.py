import numpy as np
import pytest

from triton_exp.gates import gate_correctness, gate_csr, gate_preliminary
from triton_exp.io import Csr, GpuCpgDumps


def make_dumps(
    suffix_scaled: np.ndarray,
    edge_src: np.ndarray | None = None,
    edge_dst: np.ndarray | None = None,
    levels: np.ndarray | None = None,
) -> GpuCpgDumps:
    if edge_src is None:
        edge_src = np.asarray([], dtype=np.int32)
    if edge_dst is None:
        edge_dst = np.asarray([], dtype=np.int32)
    if levels is None:
        levels = np.zeros(suffix_scaled.shape[0], dtype=np.int32)
    return GpuCpgDumps(
        suffix_scaled=suffix_scaled.astype(np.int32),
        edge_src=edge_src.astype(np.int32),
        edge_dst=edge_dst.astype(np.int32),
        edge_weights_scaled=np.ones(edge_src.shape[0], dtype=np.int32),
        levels=levels.astype(np.int32),
        level_order=np.arange(suffix_scaled.shape[0], dtype=np.int32),
    )


def test_gate_correctness_rejects_shape_mismatch_before_diffing():
    dumps = make_dumps(np.asarray([0, 10], dtype=np.int32))

    with pytest.raises(ValueError, match="triton and gpu_cpg suffix shape mismatch"):
        gate_correctness(np.asarray([[0], [10]], dtype=np.int32), dumps)


def test_gate_preliminary_checks_level_ordering_for_all_edges():
    n_edges = 10001
    edge_src = np.ones(n_edges, dtype=np.int32)
    edge_dst = np.zeros(n_edges, dtype=np.int32)
    edge_src[-1] = 0
    edge_dst[-1] = 1
    dumps = make_dumps(
        suffix_scaled=np.asarray([0, 1], dtype=np.int32),
        edge_src=edge_src,
        edge_dst=edge_dst,
        levels=np.asarray([0, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="level violation on edge 0->1"):
        gate_preliminary(dumps)


def test_gate_csr_rejects_wrong_row_ptr_size():
    csr = Csr(
        row_ptr=np.asarray([0, 0], dtype=np.int32),
        col_idx=np.asarray([], dtype=np.int32),
        weights_scaled=np.asarray([], dtype=np.int32),
        level_groups=[np.asarray([0, 1], dtype=np.int32)],
        max_level=0,
        max_out_degree=0,
    )

    with pytest.raises(ValueError, match="row_ptr.size=2 != n_nodes \\+ 1=3"):
        gate_csr(csr, n_nodes=2, n_edges=0)
