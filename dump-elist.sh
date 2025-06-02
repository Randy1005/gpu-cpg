# dump the edge lists so we can use rabbit order
#!/bin/bash

# ./a.out [benchmark] [elist file name]
# des_perf_dense40
./build/examples/dump-elist ~/Research/large-benchmarks/des_perf_dense40.txt ~/Research/large-benchmarks/des_perf-d40.elist

# vga_lcd_random_wgts_dense40
./build/examples/dump-elist ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt ~/Research/large-benchmarks/vga_lcd-d40.elist

# leon3mp_iccad_random_wgts_dense40
./build/examples/dump-elist ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt ~/Research/large-benchmarks/leon3mp-d40.elist

# netcard_random_wgts_dense40
./build/examples/dump-elist ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt ~/Research/large-benchmarks/netcard-d40.elist

# leon2_iccad_random_wgts_dense40
./build/examples/dump-elist ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt ~/Research/large-benchmarks/leon2-d40.elist

# ldoor
./build/examples/dump-elist ~/Research/large-benchmarks/DIMACS/ldoor.txt ~/Research/large-benchmarks/DIMACS/ldoor.elist

# cage15
./build/examples/dump-elist ~/Research/large-benchmarks/DIMACS/cage15.txt ~/Research/large-benchmarks/DIMACS/cage15.elist

# nlpkkt120
./build/examples/dump-elist ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt ~/Research/large-benchmarks/DIMACS/nlpkkt120.elist

# nlpkkt160
./build/examples/dump-elist ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt ~/Research/large-benchmarks/DIMACS/nlpkkt160.elist
