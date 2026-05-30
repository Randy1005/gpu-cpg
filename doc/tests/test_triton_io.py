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
