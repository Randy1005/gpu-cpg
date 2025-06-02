# csr reorder disabled, relax_bu_step
# -o cpg-vr-disabled-[graph]-sfxt
# c7522_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-c7522-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 10 0

# des_perf_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-des_perf-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/des_perf_dense40.txt 10 0

# vga_lcd_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-vga_lcd-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 10 0

# leon3mp_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-leon3mp-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 10 0

# netcard_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-netcard-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 10 0

# leon2_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-leon2-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 10 0

# DIMACS/ldoor.txt
sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-ldoor-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/ldoor.txt 10 0

# DIMACS/cage15.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-cage15-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/cage15.txt 10 0

# DIMACS/nlpkkt120.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-nlpkkt120-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 10 0

# DIMACS/nlpkkt160.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-nlpkkt160-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 10 0

# csr reorder disabled, expand_short_pile_tile_spur
# -o cpg-vr-disabled-[graph]-pfxt

# c7522_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-c7522-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 1000000 0

# des_perf_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-des_perf-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/des_perf_dense40.txt 1000000 0

# vga_lcd_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-vga_lcd-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 1000000 0

# leon3mp_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-leon3mp-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 1000000 0

# netcard_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-netcard-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 1000000 0

# leon2_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-leon2-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 1000000 0

# DIMACS/ldoor.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-ldoor-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/ldoor.txt 1000000 0

# DIMACS/cage15.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-cage15-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/cage15.txt 1000000 0

# DIMACS/nlpkkt120.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-nlpkkt120-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 1000000 0

# DIMACS/nlpkkt160.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-disabled-nlpkkt160-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 1000000 0


# csr reorder enabled, relax_bu_step
# -o cpg-vr-disabled-[graph]-sfxt
# c7522_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-c7522-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 10 1
# des_perf_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-des_perf-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/des_perf_dense40.txt 10 1
# vga_lcd_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-vga_lcd-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 10 1
# leon3mp_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-leon3mp-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 10 1
# netcard_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-netcard-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 10 1
# leon2_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-leon2-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 10 1
# DIMACS/ldoor.txt
sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-ldoor-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/ldoor.txt 10 1
# DIMACS/cage15.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-cage15-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/cage15.txt 10 1
# DIMACS/nlpkkt120.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-nlpkkt120-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 10 1
# DIMACS/nlpkkt160.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-nlpkkt160-sfxt -k relax_bu_step --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 10 1

# csr reorder enabled, expand_short_pile_tile_spur
# -o cpg-vr-enabled-[graph]-pfxt
# c7522_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-c7522-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/c7522_random_wgts_dense40.txt 1000000 1
# des_perf_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-des_perf-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/des_perf_dense40.txt 1000000 1
# vga_lcd_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-vga_lcd-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/vga_lcd_random_wgts_dense40.txt 1000000 1
# leon3mp_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-leon3mp-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon3mp_iccad_random_wgts_dense40.txt 1000000 1
# netcard_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-netcard-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/netcard_random_wgts_dense40.txt 1000000 1
# leon2_iccad_random_wgts_dense40.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-leon2-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/leon2_iccad_random_wgts_dense40.txt 1000000 1
# DIMACS/ldoor.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-ldoor-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/ldoor.txt 1000000 1
# DIMACS/cage15.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-cage15-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/cage15.txt 1000000 1
# DIMACS/nlpkkt120.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-nlpkkt120-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt120.txt 1000000 1
# DIMACS/nlpkkt160.txt
# sudo /usr/local/cuda-12.6/bin/ncu -f -o cpg-vr-enabled-nlpkkt160-pfxt -k expand_short_pile_tile_spur --nvtx --set full --import-source on ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/nlpkkt160.txt 1000000 1