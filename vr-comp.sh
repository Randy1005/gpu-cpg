# run vr compare with other methods
#!/bin/bash

# ./build/examples/cpg-other-vrs [benchmark] [k] [vr_method] [vmap_file]
# [vr_method] 2 --> ours
# [vr_method] 4 --> rabbit order
# [vr_method] 5 --> Gorder
# [vr_method] 6 --> Corder
# k is just 1000000


# ours
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/des_perf_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/ldoor.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/cage15.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 1000000 2 none
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 1000000 2 none

# rabbit
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 1000000 4 c7522-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/des_perf_dense40.txt 1000000 4 des_perf-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 1000000 4 vga_lcd-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 1000000 4 leon3mp-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 1000000 4 netcard-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 1000000 4 leon2-d40.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/ldoor.txt 1000000 4 ldoor.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/cage15.txt 1000000 4 cage15.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 1000000 4 nlpkkt120.vmap
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 1000000 4 nlpkkt160.vmap

# Gorder
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 1000000 5 ~/Research/large-benchmarks/c7522-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/des_perf_dense40.txt 1000000 5 ~/Research/large-benchmarks/des_perf-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 1000000 5 ~/Research/large-benchmarks/vga_lcd-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 1000000 5 ~/Research/large-benchmarks/leon3mp-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 1000000 5 ~/Research/large-benchmarks/netcard-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 1000000 5 ~/Research/large-benchmarks/leon2-d40_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/ldoor.txt 1000000 5 ~/Research/large-benchmarks/DIMACS/ldoor_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/cage15.txt 1000000 5 ~/Research/large-benchmarks/DIMACS/cage15_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 1000000 5 ~/Research/large-benchmarks/DIMACS/nlpkkt120_Gorder.txt
./build/examples/cpg-other-vrs ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 1000000 5 ~/Research/large-benchmarks/DIMACS/nlpkkt160_Gorder.txt






