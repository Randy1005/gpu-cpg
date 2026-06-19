# TC-Powered Critical Path Generation Proposal

This is a working proposal draft for the current tensor-core CPG direction.

## Motivation

Critical path generation repeatedly asks a large many-to-many question:

```text
which active spur sources can take which legal deviation edges?
```

G-PathGen answers this efficiently with CUDA graph traversal, but the work is
mostly scalar and irregular. Modern NVIDIA GPUs put much of their peak
throughput behind tensor-core tile operations. CPG is not a neural network, but
one part of PFXT has the right shape: deviation discovery is a tiled set-overlap
problem.

The proposal is to expose that structure while preserving exact k-critical path
ordering.

## Core Challenge

PFXT is path dependent:

- each active parent path reaches a different spur source;
- only non-suffix-tree fanout edges are legal deviations;
- a deviation edge alone does not define a path;
- exactness depends on G-PathGen's window ordering invariants.

The challenge is therefore not simply to run a graph algorithm on tensor cores.
The challenge is to use tensor cores for the regular many-to-many part while
keeping the exact candidate identity:

```text
(parent path, deviation source u, deviation destination v)
```

## Core Idea

Build a transposed deviation matrix:

```text
A_dev[v, u] = 1 iff edge u -> v exists and edge u -> v is not succs[u]
```

For one PFXT suffix-chain walk step, the active frontier marks which spur
sources are currently reached:

```text
alpha[u] = 1 iff at least one active parent path currently reaches u
```

Deviation discovery becomes a set-overlap operation:

```text
hit_mask[v, :] = A_dev[v, :] AND alpha[:]
```

Tensor cores accelerate this tiled overlap. CUDA then decodes the surviving
bits and preserves exact candidate identity.

## Figure: From CPG to Tensor-Core Tiles

### Panel A: Graph View

```text
active parent paths              legal deviation edges

  p1 -> u0 ------------------------------> v1
  p2 -> u1 ----------------> v0
  p3 -> u4 ----------------> v2
  p4 -> u6 ------------------------------> v1
```

Here `u` is a spur source reached by at least one active parent path. A legal
deviation is any edge `u -> v` that is not the suffix-tree edge `succs[u]`.

### Panel B: Matrix / Set-Overlap View

```text
                 spur source u
                 u0 u1 u2 u3 u4 u5 u6 u7
active alpha      1  1  0  0  1  0  1  0

A_dev row v0      0  1  0  0  0  0  0  1
A_dev row v1      1  0  1  0  0  0  1  0
A_dev row v2      0  0  0  1  1  0  0  0
A_dev row v3      1  1  0  0  0  1  0  0
```

For `v1`:

```text
A_dev[v1, :] = [1 0 1 0 0 0 1 0]
alpha[:]     = [1 1 0 0 1 0 1 0]
hit_mask     = [1 0 0 0 0 0 1 0]
```

The set bits are `u0` and `u6`, so discovery emits:

```text
(u0, v1), (u6, v1)
```

### Panel C: Tensor-Core Tile View

```text
          active frontier alpha tile
          [u0 u1 u2 u3 u4 u5 u6 u7]

A_dev  [ v0 row bit mask ]       TC tile overlap
tile   [ v1 row bit mask ]  -->  overlap scores + hit masks
       [ v2 row bit mask ]
       [ v3 row bit mask ]
```

The score is only a compact way to find live rows:

```text
score[v] = count(A_dev[v, :] AND alpha[:])
```

Exact pair emission still decodes the hit mask. This is what keeps the TC stage
as a discovery accelerator rather than an approximate path generator.

## Implementation

The current meaningful implementation has three pieces.

### 1. Tensor-Core Deviation Discovery

The static deviation relation is stored as compressed bit-vector set storage.
During a PFXT suffix-chain walk, TC kernels evaluate active-source overlap with
destination rows and identify live deviation families `(u, v)`.

This is the TC-specific part of the implementation.

### 2. Spur-Source Grouped Candidate Generation

For every discovered deviation source `u`, candidate generation must combine:

```text
all parent paths currently at u
*
all deviation destinations v reachable from u
```

The same edge `u -> v` can produce many exact paths because the parent path is
part of the identity:

```text
candidate(parent, u, v)
```

Spur-source grouped candidate generation keeps the TC-discovered work grouped by
deviation source, then materializes the exact Cartesian product for that source.
This preserves exactness and avoids flattening the TC output into a less useful
global pair stream.

### 3. Compact Active-Source Grouping

The first spur-source grouped implementation still paid full-graph bookkeeping:
group arrays were indexed by all graph nodes even though each suffix-chain
substep only used a small active subset.

Compact active-source grouping replaces that with dense slots for active spur
sources only:

```text
active_sources = [u17, u42, u900000]
slot[u17]      = 0
slot[u42]      = 1
slot[u900000]  = 2
```

This reduced the hidden interface cost between fast TC discovery and CUDA
candidate materialization. It does not change the candidate set.

## Current Results

Latest in-process retime: one warmup, three measured trials. GPG numbers are
cached PFXT means from the same density sweep. TC uses the current best
configuration: single-pass, single-work, spur-source grouped candidate
generation, compact static deviations, tile-native short-only emission, and
compact active-source grouping.

| density | K | GPG ms | TC ms | TC/GPG |
|---|---:|---:|---:|---:|
| d10 | 1M | 186.9 | 323.3 | 1.73x |
| d20 | 1M | 264.1 | 266.7 | 1.01x |
| d30 | 200K | 62.7 | 68.7 | 1.09x |
| d40 | 200K | 64.1 | 104.3 | 1.63x |
| d50 | 100K | 59.6 | 60.2 | 1.01x |

PFXT time only. Graph input, SFXT construction, and static deviation-matrix
setup are cached outside the reported PFXT time in the in-process harness.

## Current Bottleneck

TC discovery is fast. The remaining cost is candidate product handling:

```text
parent paths at u * deviation destinations from u
```

On d20 K=1M, the current TC path still touches about `233M`
parent/deviation products. `122.9M` of those products are classified as skipped
for the current window. Compact active-source grouping removed a large setup
cost, but it did not remove the product work itself.

## Next Directions

1. **Keep TC tiles alive longer.** Avoid converting TC-discovered structure into
   expensive per-candidate records too early.
2. **Reduce materialized products without extra compaction cost.** Use cheap
   tile/source-level decisions to avoid visiting products that cannot enter
   HPQ/LPQ, while preserving exactness.

Failed optional paths and negative lessons are recorded separately in
[tc-pfxt-lessons-learned.md](tc-pfxt-lessons-learned.md).
