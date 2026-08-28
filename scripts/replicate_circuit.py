#!/usr/bin/env python3
"""Replicate and diversify a weighted circuit DAG behind shared terminals.

The benchmark reader uses the first line as the vertex count, skips that many
vertex-label lines, then expects edges of the form

    "SOURCE_ID" -> "DESTINATION_ID", WEIGHT;

Vertex labels are informational, so generated labels are compact numeric IDs.
The topology of each copy is retained, but internal weights receive a
deterministic bounded perturbation.  Copies become macro-nodes in a layered,
randomized task DAG.  Every macro-edge connects all upstream primary outputs
to all downstream primary inputs through bounded-degree merge/distribution
trees rather than an impractical complete bipartite product.  Only root copies
join the super-source and only leaf copies join the super-sink, so paths cannot
bypass the deeper task graph.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from array import array
from pathlib import Path
from typing import Iterator, TextIO


EDGE_RE = re.compile(
    r'^\s*"(?P<src>\d+)"\s*->\s*"(?P<dst>\d+)"'
    r'(?:\s*,\s*(?P<weight>[^;]+))?\s*;\s*$'
)


def read_vertex_count(stream: TextIO) -> int:
    line = stream.readline()
    try:
        count = int(line.strip())
    except ValueError as exc:
        raise ValueError(f"invalid vertex count: {line.rstrip()!r}") from exc
    if count <= 0:
        raise ValueError(f"vertex count must be positive, got {count}")
    return count


def skip_vertex_labels(stream: TextIO, count: int) -> None:
    for index in range(count):
        if not stream.readline():
            raise ValueError(f"missing vertex label {index} of {count}")


def parse_edges(stream: TextIO, vertex_count: int) -> Iterator[tuple[int, int, str]]:
    for line_number, line in enumerate(stream, start=vertex_count + 2):
        if not line.strip():
            continue
        match = EDGE_RE.match(line)
        if not match:
            raise ValueError(f"line {line_number}: malformed edge: {line.rstrip()!r}")
        src = int(match.group("src"))
        dst = int(match.group("dst"))
        if src >= vertex_count or dst >= vertex_count:
            raise ValueError(
                f"line {line_number}: edge ({src}, {dst}) exceeds "
                f"vertex range [0, {vertex_count})"
            )
        weight_text = match.group("weight") or "1"
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                f"line {line_number}: invalid edge weight {weight_text!r}"
            ) from exc
        if not math.isfinite(weight):
            raise ValueError(f"line {line_number}: non-finite edge weight")
        yield src, dst, weight_text.strip()


def inspect_circuit(path: Path) -> tuple[int, int, list[int], list[int]]:
    with path.open("r", encoding="utf-8") as stream:
        vertex_count = read_vertex_count(stream)
        skip_vertex_labels(stream, vertex_count)
        indegree = array("I", [0]) * vertex_count
        outdegree = array("I", [0]) * vertex_count
        edge_count = 0
        for src, dst, _ in parse_edges(stream, vertex_count):
            if indegree[dst] == 0xFFFFFFFF or outdegree[src] == 0xFFFFFFFF:
                raise OverflowError("a vertex degree exceeds uint32 capacity")
            indegree[dst] += 1
            outdegree[src] += 1
            edge_count += 1

    sources = [vertex for vertex, degree in enumerate(indegree) if degree == 0]
    sinks = [vertex for vertex, degree in enumerate(outdegree) if degree == 0]
    if not sources or not sinks:
        raise ValueError("input has no primary source or no primary sink")
    return vertex_count, edge_count, sources, sinks


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def perturb_weight(
    weight: float, seed: int, copy_index: int, edge_index: int, jitter: float
) -> float:
    if weight == 0.0 or jitter == 0.0:
        return weight
    bits = splitmix64(seed ^ (copy_index << 32) ^ edge_index)
    unit = (bits >> 11) * (1.0 / (1 << 53))
    return weight * (1.0 + jitter * (2.0 * unit - 1.0))


def format_weight(weight: float) -> str:
    return format(weight, ".9g")


def mean_nonzero_absolute_weight(path: Path, vertex_count: int) -> float:
    total = 0.0
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        actual_count = read_vertex_count(stream)
        if actual_count != vertex_count:
            raise RuntimeError("input changed while it was being inspected")
        skip_vertex_labels(stream, vertex_count)
        for _, _, weight_text in parse_edges(stream, vertex_count):
            weight = abs(float(weight_text))
            if weight > 0.0:
                total += weight
                count += 1
    return total / count if count else 1.0


def copy_edges(
    path: Path,
    output: TextIO,
    vertex_count: int,
    offset: int,
    copy_index: int,
    seed: int,
    weight_jitter: float,
) -> None:
    with path.open("r", encoding="utf-8") as stream:
        actual_count = read_vertex_count(stream)
        if actual_count != vertex_count:
            raise RuntimeError("input changed while it was being replicated")
        skip_vertex_labels(stream, vertex_count)
        for edge_index, (src, dst, weight_text) in enumerate(
            parse_edges(stream, vertex_count)
        ):
            weight = perturb_weight(
                float(weight_text), seed, copy_index, edge_index, weight_jitter
            )
            output.write(
                f'"{src + offset}" -> "{dst + offset}", '
                f'{format_weight(weight)};\n'
            )


def macro_layer_sizes(copies: int) -> tuple[int, ...]:
    if copies == 8:
        return (1, 2, 2, 2, 1)
    if copies == 16:
        return (1, 2, 4, 4, 2, 2, 1)
    raise ValueError("macro task graph supports exactly 8 or 16 copies")


def build_macro_dag(
    copies: int, seed: int, extra_edge_probability: float
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    if not 0.0 <= extra_edge_probability <= 1.0:
        raise ValueError("macro extra-edge probability must be in [0, 1]")
    sizes = macro_layer_sizes(copies)
    layers: list[list[int]] = []
    cursor = 0
    for size in sizes:
        layers.append(list(range(cursor, cursor + size)))
        cursor += size

    rng = random.Random(splitmix64(seed ^ 0xA0761D6478BD642F))
    edges: set[tuple[int, int]] = set()
    for left, right in zip(layers, layers[1:]):
        shuffled_left = left.copy()
        shuffled_right = right.copy()
        rng.shuffle(shuffled_left)
        rng.shuffle(shuffled_right)
        # Cover every downstream macro-node, then every upstream macro-node.
        # Adjacent layer ratios never exceed two, so the mandatory cover has
        # macro fan-in/fan-out at most two.
        for index, dst in enumerate(shuffled_right):
            edges.add((shuffled_left[index % len(shuffled_left)], dst))
        for index, src in enumerate(shuffled_left):
            edges.add((src, shuffled_right[index % len(shuffled_right)]))

        outdegree = {src: 0 for src in left}
        indegree = {dst: 0 for dst in right}
        for src, dst in edges:
            if src in outdegree and dst in indegree:
                outdegree[src] += 1
                indegree[dst] += 1
        candidates = [
            (src, dst) for src in left for dst in right
            if (src, dst) not in edges
        ]
        rng.shuffle(candidates)
        for src, dst in candidates:
            if (outdegree[src] < 2 and indegree[dst] < 2
                    and rng.random() < extra_edge_probability):
                edges.add((src, dst))
                outdegree[src] += 1
                indegree[dst] += 1
    return layers, sorted(edges)


def connector_shape(source_count: int, sink_count: int) -> tuple[int, int]:
    # Merge source_count upstream POs to one root, then distribute that root to
    # sink_count downstream PIs. The merge root doubles as the distribution
    # root, saving one connector vertex.
    vertices = max(0, source_count - 1) + max(0, sink_count - 2)
    merge_edges = 2 * max(0, source_count - 1)
    distribution_edges = 1 if sink_count == 1 else 2 * sink_count - 2
    return vertices, merge_edges + distribution_edges


def emit_full_coverage_connector(
    output: TextIO,
    upstream_outputs: list[int],
    downstream_inputs: list[int],
    next_vertex: int,
    base_weight: float,
    seed: int,
    macro_edge_index: int,
    weight_jitter: float,
) -> tuple[int, int]:
    edge_index = 0

    def write_edge(src: int, dst: int) -> None:
        nonlocal edge_index
        weight = perturb_weight(
            base_weight,
            seed ^ 0xD1B54A32D192ED03,
            macro_edge_index,
            edge_index,
            weight_jitter,
        )
        output.write(
            f'"{src}" -> "{dst}", {format_weight(weight)};\n'
        )
        edge_index += 1

    frontier = upstream_outputs.copy()
    while len(frontier) > 1:
        merged: list[int] = []
        for index in range(0, len(frontier), 2):
            if index + 1 == len(frontier):
                merged.append(frontier[index])
                continue
            parent = next_vertex
            next_vertex += 1
            write_edge(frontier[index], parent)
            write_edge(frontier[index + 1], parent)
            merged.append(parent)
        frontier = merged
    root = frontier[0]

    stack: list[tuple[int, list[int]]] = [(root, downstream_inputs)]
    while stack:
        parent, targets = stack.pop()
        if len(targets) == 1:
            write_edge(parent, targets[0])
            continue
        midpoint = len(targets) // 2
        for subset in (targets[:midpoint], targets[midpoint:]):
            if len(subset) == 1:
                write_edge(parent, subset[0])
            else:
                child = next_vertex
                next_vertex += 1
                write_edge(parent, child)
                stack.append((child, subset))
    return next_vertex, edge_index


def replicate(
    input_path: Path,
    output_path: Path,
    copies: int,
    *,
    seed: int = 289,
    weight_jitter: float = 0.15,
    macro_extra_edge_probability: float = 0.35,
) -> dict[str, int | float]:
    if copies < 2:
        raise ValueError("copies must be at least 2")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if not 0.0 <= weight_jitter < 1.0:
        raise ValueError("weight jitter must be in [0, 1)")

    vertex_count, edge_count, sources, sinks = inspect_circuit(input_path)
    replicated_vertices = vertex_count * copies
    layers, macro_edges = build_macro_dag(
        copies, seed, macro_extra_edge_probability
    )
    connector_vertices_per_edge, connector_edges_per_edge = connector_shape(
        len(sinks), len(sources)
    )
    connector_vertices = connector_vertices_per_edge * len(macro_edges)
    connector_edges = connector_edges_per_edge * len(macro_edges)
    output_vertices = replicated_vertices + connector_vertices + 2
    if output_vertices > 2_147_483_647:
        raise OverflowError("replicated vertex IDs exceed signed 32-bit capacity")
    super_source = output_vertices - 2
    super_sink = output_vertices - 1
    connector_weight = mean_nonzero_absolute_weight(input_path, vertex_count)
    terminal_edges = len(layers[0]) * len(sources) + len(layers[-1]) * len(sinks)
    output_edges = copies * edge_count + connector_edges + terminal_edges
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        output.write(f"{output_vertices}\n")
        for vertex in range(output_vertices):
            output.write(f'"{vertex}";\n')

        for copy_index in range(copies):
            offset = copy_index * vertex_count
            copy_edges(
                input_path,
                output,
                vertex_count,
                offset,
                copy_index,
                seed,
                weight_jitter,
            )
        for copy_index in layers[0]:
            offset = copy_index * vertex_count
            for source in sources:
                output.write(f'"{super_source}" -> "{source + offset}", 0;\n')
        for copy_index in layers[-1]:
            offset = copy_index * vertex_count
            for sink in sinks:
                output.write(f'"{sink + offset}" -> "{super_sink}", 0;\n')

        next_connector_vertex = replicated_vertices
        emitted_connector_edges = 0
        for macro_edge_index, (upstream, downstream) in enumerate(macro_edges):
            upstream_offset = upstream * vertex_count
            downstream_offset = downstream * vertex_count
            next_connector_vertex, emitted = emit_full_coverage_connector(
                output,
                [sink + upstream_offset for sink in sinks],
                [source + downstream_offset for source in sources],
                next_connector_vertex,
                connector_weight,
                seed,
                macro_edge_index,
                weight_jitter,
            )
            emitted_connector_edges += emitted
        if next_connector_vertex != super_source:
            raise RuntimeError("connector vertex accounting mismatch")
        if emitted_connector_edges != connector_edges:
            raise RuntimeError("connector edge accounting mismatch")

    return {
        "input_vertices": vertex_count,
        "input_edges": edge_count,
        "input_sources": len(sources),
        "input_sinks": len(sinks),
        "output_vertices": output_vertices,
        "output_edges": output_edges,
        "copies": copies,
        "seed": seed,
        "weight_jitter": weight_jitter,
        "macro_layers": len(layers),
        "macro_edges": len(macro_edges),
        "macro_extra_edge_probability": macro_extra_edge_probability,
        "connector_vertices": connector_vertices,
        "connector_edges": connector_edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("copies", type=int, choices=(8, 16))
    parser.add_argument("--seed", type=int, default=289)
    parser.add_argument("--weight-jitter", type=float, default=0.15)
    parser.add_argument("--macro-extra-edge-probability", type=float, default=0.35)
    args = parser.parse_args()

    try:
        stats = replicate(
            args.input,
            args.output,
            args.copies,
            seed=args.seed,
            weight_jitter=args.weight_jitter,
            macro_extra_edge_probability=args.macro_extra_edge_probability,
        )
    except (OSError, ValueError, OverflowError, RuntimeError) as exc:
        print(f"replicate_circuit: error: {exc}", file=sys.stderr)
        return 1

    print(
        "replicate_circuit"
        + "".join(f" {name}={value}" for name, value in stats.items())
        + f" output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
