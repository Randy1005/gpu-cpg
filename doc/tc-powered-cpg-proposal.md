# TC-Powered Critical Path Generation Proposal

This is a working proposal draft. The wording is intentionally lightweight so
we can revise it as the story changes.

## Motivation

Critical path generation repeatedly asks a simple question at very large scale:
for each active path, which off-tree timing edges can create the next candidate
path? In a conventional CUDA implementation, this becomes many small irregular
lookups and candidate writes. That is effective but leaves the GPU's tensor
cores mostly unused.

Tensor cores are built for dense matrix operations. The proposal is to expose
the hidden matrix-like structure inside path generation and use tensor cores for
the part that looks like massive batched set intersection.

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

## Implementation Sketch

The current implementation builds a transposed deviation matrix `A_dev`.

- Row `v`: a deviation destination.
- Bit `u`: source `u` has a fanout edge to `v`, and that edge is not the suffix
  tree edge.
- Frontier vector `alpha`: active spur sources from the current PFXT window.

Tensor-core discovery computes overlap between `A_dev` rows and `alpha`. Each
hit emits a deviation pair `(u, v)`. The source-local candidate path then groups
work by active source and materializes exact `(parent path, u, v)` candidates.

This keeps the exactness invariant: the deviation edge alone does not define a
path. Different parent paths reaching the same `u -> v` edge remain distinct
PFXT candidates.

## Current Result

The current TC implementation is not faster than GPG yet. It is the first
end-to-end exact implementation that uses tensor cores for deviation discovery.

| density | K | GPG ms | TC ms | TC/GPG |
|---|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 415.2 | 2.22x |
| d20 | 1M | 264.1 | 468.9 | 1.78x |
| d30 | 200K | 62.7 | 199.4 | 3.18x |
| d40 | 200K | 64.1 | 226.9 | 3.54x |
| d50 | 100K | 59.6 | 172.3 | 2.89x |

This table is still useful: it shows where the first implementation pays extra
cost. Tensor-core deviation discovery is cheap; candidate materialization and
queue work dominate.

## Next TC-Specific Directions

1. **Fuse discovery with less candidate materialization.** Keep the tensor-core
   discovery result in a grouped/source-local form and avoid expanding work that
   will not enter HPQ/LPQ.

2. **Make candidate generation more tensor-core-shaped.** Treat candidate
   families as tiles of parent paths by deviation edges, so later work can reuse
   the same dense grouping that tensor cores already prefer.

The proposal goal is to move more of PFXT from irregular per-candidate CUDA work
into regular grouped operations without changing exactness.
