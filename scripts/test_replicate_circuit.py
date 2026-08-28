#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from replicate_circuit import (
    build_macro_dag,
    connector_shape,
    inspect_circuit,
    parse_edges,
    read_vertex_count,
    replicate,
    skip_vertex_labels,
)


def read_edges(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    with path.open("r", encoding="utf-8") as stream:
        vertices = read_vertex_count(stream)
        skip_vertex_labels(stream, vertices)
        edges = [
            (src, dst, float(weight))
            for src, dst, weight in parse_edges(stream, vertices)
        ]
    return vertices, edges


def is_acyclic(vertices: int, edges: list[tuple[int, int, float]]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(vertices)]
    indegree = [0] * vertices
    for src, dst, _ in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1
    queue = [vertex for vertex, degree in enumerate(indegree) if degree == 0]
    visited = 0
    for src in queue:
        visited += 1
        for dst in adjacency[src]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return visited == vertices


class ReplicateCircuitTest(unittest.TestCase):
    def test_copies_form_deep_macro_dag_with_bounded_full_coverage(self) -> None:
        fixture = """4
\"a\";
\"b\";
\"c\";
\"d\";
\"0\" -> \"1\", 1.5;
\"1\" -> \"2\", 2;
\"0\" -> \"3\", 4.25;
"""
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            output_path = Path(directory) / "output.txt"
            second_output_path = Path(directory) / "output-again.txt"
            input_path.write_text(fixture, encoding="utf-8")

            stats = replicate(input_path, output_path, 8)
            repeat_stats = replicate(input_path, second_output_path, 8)
            layers, macro_edges = build_macro_dag(8, 289, 0.35)
            connector_vertices, connector_edges = connector_shape(2, 1)

            self.assertEqual(len(layers), 5)
            self.assertEqual(stats["macro_layers"], 5)
            self.assertEqual(stats["macro_edges"], len(macro_edges))
            self.assertEqual(
                stats["output_vertices"], 32 + connector_vertices * len(macro_edges) + 2
            )
            self.assertEqual(
                stats["output_edges"], 8 * 3 + connector_edges * len(macro_edges) + 3
            )
            self.assertEqual(stats, repeat_stats)
            self.assertEqual(output_path.read_bytes(), second_output_path.read_bytes())
            vertices, edges, sources, sinks = inspect_circuit(output_path)
            self.assertEqual((vertices, edges),
                             (stats["output_vertices"], stats["output_edges"]))
            super_source = vertices - 2
            super_sink = vertices - 1
            self.assertEqual(sources, [super_source])
            self.assertEqual(sinks, [super_sink])

            parsed_vertices, parsed_edges = read_edges(output_path)
            self.assertTrue(is_acyclic(parsed_vertices, parsed_edges))
            copied_weights = [
                weight
                for src, dst, weight in parsed_edges
                if src % 4 == 0 and dst == src + 1 and src < 32
            ]
            self.assertEqual(len(copied_weights), 8)
            self.assertEqual(len(set(copied_weights)), 8)
            self.assertTrue(all(1.5 * 0.85 <= weight <= 1.5 * 1.15
                                for weight in copied_weights))

            adjacency: list[list[int]] = [[] for _ in range(parsed_vertices)]
            indegree = [0] * parsed_vertices
            outdegree = [0] * parsed_vertices
            for src, dst, _ in parsed_edges:
                adjacency[src].append(dst)
                indegree[dst] += 1
                outdegree[src] += 1

            connector_begin = 32
            for vertex in range(connector_begin, super_source):
                self.assertLessEqual(indegree[vertex], 2)
                self.assertLessEqual(outdegree[vertex], 2)

            # Only the macro root copy receives super-source edges and only the
            # macro leaf copy reaches the super-sink; intermediate copies cannot
            # bypass the macro task graph.
            super_children = set(adjacency[super_source])
            self.assertEqual(super_children, {0})
            super_parents = {
                src for src, dst, _ in parsed_edges if dst == super_sink
            }
            self.assertEqual(super_parents, {30, 31})
            for copy_index in range(1, 8):
                self.assertGreater(indegree[copy_index * 4], 0)
            for copy_index in range(7):
                self.assertGreater(outdegree[copy_index * 4 + 2], 0)
                self.assertGreater(outdegree[copy_index * 4 + 3], 0)

    def test_rejects_out_of_range_edge(self) -> None:
        fixture = '2\n"a";\n"b";\n"0" -> "2", 1;\n'
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            input_path.write_text(fixture, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds vertex range"):
                inspect_circuit(input_path)


if __name__ == "__main__":
    unittest.main()
