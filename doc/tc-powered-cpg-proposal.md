# TC-Powered Critical Path Generation Proposal

This is a working proposal draft. The wording is intentionally lightweight so
we can revise it as the story changes.

## Motivation

Critical path generation repeatedly asks a simple question at very large scale:
for each active path, which off-tree timing edges can create the next candidate
path? In a conventional CUDA implementation, this becomes many small irregular
lookups and candidate writes. That is effective but leaves the GPU's tensor
cores mostly unused.

That matters because modern NVIDIA GPUs expose most of their peak arithmetic
through tensor-core paths. H100 SXM is roughly `67 TFLOPS` FP32, but almost
`4 PFLOPS` FP8 tensor-core throughput. HGX B200 is roughly `600 TFLOPS` FP32
for 8 GPUs, but `36 PFLOPS` dense FP8/FP6 tensor-core throughput. The exact
precision is not the point for CPG; the point is that the fastest hardware path
on newer GPUs is a tiled many-to-many engine, while ordinary CPG uses mostly
scalar CUDA graph traversal.

The proposal is to expose the hidden many-to-many structure inside path
generation. CPG is not a neural-network matrix multiply, but deviation
discovery repeatedly asks a matrix-like question:

> Which active spur sources can connect to which legal deviation destinations?

If we express that question as tiled bit-vector overlap, tensor cores become a
natural fit. They can test many source/destination relationships at once, while
CUDA handles the exact path-dependent candidate records after discovery.

## Core Technical Challenge

CPG is not naturally a dense matrix multiply. It is path dependent:

- only the current frontier paths are active;
- each active path reaches different spur sources;
- only non-suffix-tree fanout edges are valid deviations;
- exactness depends on preserving G-PathGen's PFXT window ordering.

The challenge is therefore not simply "run CPG on tensor cores." The challenge
is to find a formulation that gives tensor cores regular work while keeping the
same exact candidate set and path ranking.

## Core Innovation

We reformulate deviation discovery as a compressed semiring-style matrix
operation.

Informally, imagine a large checklist:

- rows represent possible deviation destinations;
- columns represent spur sources grouped into fixed-size bit-vector slots;
- the active frontier marks which spur sources are currently reachable.

For each destination, we ask: "Does this destination have any active spur source
that can reach it through a non-tree edge?" That is a set-intersection question.
Tensor cores can answer many of these overlap questions in parallel by treating
the bit-vector masks like tiny matrix tiles.

The operation is semiring-like because the math is not ordinary arithmetic for
path costs. The useful operations are closer to:

- combine: intersect active-source bits with destination fanout bits;
- reduce: report whether any bit survives, and which source/destination pairs
  survived.

The tensor core is used as a high-throughput engine for this batched overlap
test. The exact path costs and HPQ/LPQ admission decisions remain exact CUDA
logic after discovery.

## Figure: From CPG to Tensor-Core Tiles

A useful presentation figure is a three-panel transformation.

### Panel A: CPG Graph View

Show active paths reaching spur sources `u`, with legal deviation edges leaving
those sources:

```text
active paths                  legal deviations

  p1 -> u0 --------------------------> v1
  p2 -> u1 -------------> v0
  p3 -> u4 -------------> v2
  p4 -> u6 --------------------------> v1
```

Here `u` is a spur source reached by at least one active parent path. A legal
deviation edge is any fanout edge `u -> v` that is not the suffix-tree edge
`succs[u]`.

### Panel B: Matrix / Set-Overlap View

The same graph relation becomes a binary deviation matrix:

```text
                 spur source u
                 u0 u1 u2 u3 u4 u5 u6 u7
active alpha      1  1  0  0  1  0  1  0

A_dev row v0      0  1  0  0  0  0  0  1
A_dev row v1      1  0  1  0  0  0  1  0
A_dev row v2      0  0  0  1  1  0  0  0
A_dev row v3      1  1  0  0  0  1  0  0
```

`A_dev[v, u] = 1` means:

```text
edge u -> v exists AND edge u -> v is not succs[u]
```

`alpha[u] = 1` means:

```text
at least one active parent path in the current PFXT chain sub-step reaches u
```

So discovery computes:

```text
hit_mask[v, :] = A_dev[v, :] AND alpha[:]
```

Example:

```text
A_dev[v1, :]   = [1 0 1 0 0 0 1 0]
alpha[:]       = [1 1 0 0 1 0 1 0]
hit_mask[v1]   = [1 0 0 0 0 0 1 0]
```

This means `v1` has active deviation sources `u0` and `u6`.

### Panel C: Tensor-Core Tile View

The matrix is processed in small dense tiles:

```text
          active frontier alpha tile
          [u0 u1 u2 u3 u4 u5 u6 u7]

A_dev  [ v0 row bit mask ]       TC tile overlap
tile   [ v1 row bit mask ]  -->  overlap scores + hit masks
       [ v2 row bit mask ]
       [ v3 row bit mask ]
```

Conceptually, a TC tile computes overlap scores:

```text
score[v] = count(A_dev[v, :] AND alpha[:])
```

For the example above:

```text
score[v0] = 1
score[v1] = 2
score[v2] = 1
score[v3] = 2
```

The score only says how many active source hits a destination row has. Exact
pair emission still uses the bit mask:

```text
hit_mask[v1] = [1 0 0 0 0 0 1 0]
set bits     = u0, u6
emit         = (u0, v1), (u6, v1)
```

This distinction is important for correctness: tensor cores accelerate the
batched overlap/filtering work, but exact CUDA logic still enumerates the
surviving source bits and emits exact deviation pairs.

## Implementation Sketch

The current implementation builds a transposed deviation matrix `A_dev`.

- Row `v`: a deviation destination.
- Bit `u`: source `u` has a fanout edge to `v`, and that edge is not the suffix
  tree edge.
- Frontier vector `alpha`: active spur sources from the current PFXT chain
  sub-step.

Tensor-core discovery computes overlap between `A_dev` rows and `alpha`. Each
hit emits a deviation pair `(u, v)`.

That pair is not yet a full path. If many parent paths reach the same source
`u`, the same deviation edge `u -> v` creates many distinct candidates:

```text
candidate(p, u, v) =
  parent path p to u
  + deviation edge u -> v
  + suffix-tree path from v to sink
```

The source-local candidate path keeps the TC output grouped by source:

```text
source u
  parent paths reaching u: p0, p1, p2, ...
  deviation destinations:  v0, v1, v2, ...
  exact candidates:        Cartesian product of parents and deviations
```

This is the key bridge between TC discovery and candidate generation. The TC
tile tells us which `(u, v)` families are alive; candidate generation then
expands each family into exact path records only after applying the current
PFXT split/window rules. The current implementation still materializes those
records with CUDA kernels, which is why candidate generation remains the main
runtime bottleneck.

This keeps the exactness invariant: the deviation edge alone does not define a
path. Different parent paths reaching the same `u -> v` edge remain distinct
PFXT candidates.

## Current Result

The current TC implementation is not faster than GPG yet. It is the first
end-to-end exact implementation that uses tensor cores for deviation discovery,
and the latest compact source-local path with tile-native short-only candidate
emission is now close enough on the main K=1M density point to make
optimization meaningful.

Headline comparison against GPG:

| density | K | GPG ms | TC ms | TC/GPG |
|---|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 314.5 | 1.68x |
| d20 | 1M | 264.1 | 309.8 | 1.17x |
| d30 | 200K | 62.7 | 113.2 | 1.81x |
| d40 | 200K | 64.1 | 136.6 | 2.13x |
| d50 | 100K | 59.6 | 95.3 | 1.60x |

These are PFXT expansion times only. The compact static-deviation CSR is a
one-time graph setup structure and is not counted in either GPG or TC PFXT
runtime.

The compact path keeps source-local candidate generation enabled in the late
heavy steps instead of falling back to the older chunked path. Tile-native
short-only emission then removes one source-local product pass when LPQ writes
are already disabled. In the June 16 sweep this won on all five density points,
with the clearest gain on d20 K=1M: `315.7 ms` without tile-native emission and
`309.8 ms` with it. That is still slower than GPG, but only `1.17x` on the main
K=1M density point.

Candidate materialization is still the dominant cost. On the d20 K=1M exactness
run, TC materialized `233.1M` parent/deviation products and skipped `122.9M`
products that would not enter the current window. The next improvement must
reduce materialization work or fuse it more tightly with the TC-discovered
source-local tiles.

## Next TC-Specific Directions

1. **Fuse discovery with less candidate materialization.** Keep the tensor-core
   discovery result in a grouped/source-local form and avoid expanding work that
   will not enter HPQ/LPQ.

2. **Make candidate generation more tensor-core-shaped.** Treat candidate
   families as tiles of parent paths by deviation edges, so later work can reuse
   the same dense grouping that tensor cores already prefer.

The proposal goal is to move more of PFXT from irregular per-candidate CUDA work
into regular grouped operations without changing exactness.
