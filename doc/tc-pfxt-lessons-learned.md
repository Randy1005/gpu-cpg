# TC PFXT Lessons Learned

This note keeps failed or optional optimization attempts out of the proposal
story while preserving the technical lessons.

## Why This Exists

The clean proposal should explain the current implementation:

- tensor-core deviation discovery;
- spur-source grouped candidate generation;
- compact active-source grouping;
- current runtime and bottleneck.

The experiments below are useful, but they should not be mixed into the main
proposal because they are not current headline paths.

## Source-Major Candidate Scheduling

Goal: reorganize candidate materialization by source so repeated parent/deviation
work would be more local.

Observed results:

| experiment | result |
|---|---:|
| d10 K=100K, single-work | 91.2 ms mean |
| d10 K=100K, naive source-major | 176.9 ms mean |
| d10 K=100K, tiled source-major | 93.3 ms mean |
| d20 K=1M, tiled source-major with fallback | 519.8 ms mean |

Lesson: changing scheduling order alone is not enough. A useful design must
reduce product work or global memory traffic, not only reshuffle which block
fills candidates.

## Direct PairMeta Emission

Goal: have TC discovery emit `PairMeta {src,dst,edge_id,wgt}` directly, avoiding
the raw `(u, v)` pair array and the pair-to-metadata conversion kernel.

Observed d20 gates:

| K | exactness | TC PFXT ms | pair_meta ms removed | interpretation |
|---:|---|---:|---:|---|
| 10K | PASS | 265.3 | about 7.5 | too much extra discovery pressure |
| 100K | PASS | 476.7 | about 10.9 | still a regression |

Lesson: the conversion kernel was not the dominant cost. Moving metadata reads
and heavier writes into the discovery kernel made the TC stage slower.

## Fused Discovery Interface Shadow

Goal: estimate whether removing the global TC-to-CUDA pair interface would pay
off.

Observed d20 gates:

| K | exactness | shadow cost | interpretation |
|---:|---|---:|---|
| 10K | PASS | 115.9 ms | second traversal too expensive |
| 100K | PASS | 242.5 ms | still too expensive |

Lesson: fusion can only help if it happens inside the original discovery pass.
A second TC/BVSS traversal or reconstructed pair interface loses the benefit.

## In-Discovery Short-Only Fusion

Goal: materialize short-only candidates directly inside discovery for substeps
where LPQ writes are disabled.

Observed d20 K=10K:

| metric | value |
|---|---:|
| exactness | PASS |
| fused substeps | 24 |
| LPQ-active substeps skipped | 141 |
| fused pairs | 456,994 |
| parent visits | 6.9M |
| in-discovery time | 515.3 ms |
| total TC PFXT time | 654.0 ms |

Lesson: this destroyed the efficient one-block-per-pair materialization shape.
Even though it removed some interface writes, it moved irregular parent scanning
into the discovery warp and became much slower.

## Tile Bounds / Product Skipping

Goal: use tile-level bounds to avoid visiting skipped parent/deviation products.

Lesson: the skip statistics were compelling, but cheap bounds were either too
weak or required setup that ate the savings. Any future version must make the
skip decision with near-zero extra memory traffic.

## Queue / Window Micro-Optimizations

Goal: reduce split/window bookkeeping overhead.

What helped:

- scalar active-count cleanup simplified host reads;
- reusing the split-update promoted count removed a duplicate count.

Observed impact: correctness-safe but mostly timing-neutral once compact
active-source grouping was enabled.

Lesson: queue/window overhead matters, but the next large gain is still product
handling. Queue changes are secondary unless they remove a full pass over a
large LPQ.

## Compact Active-Source Grouping

This is not a failed path. It is the successful lesson from the cleanup work.

The previous spur-source grouped path still prepared source groups using
full-graph arrays. Compact grouping instead uses only active spur-source slots
per suffix-chain substep.

Observed d20 K=1M improvement:

```text
before compact grouping: about 555 ms
after compact grouping:  about 266 ms
```

Lesson: preserving TC-discovered structure is useful only if the handoff stays
compact. Hidden full-graph setup can dominate even when TC discovery itself is
fast.
