#!/bin/bash

# generate corder csr-bin files
# /home/cchang289/Research/Corder-TPDS-21/corder -d <input_csr_bin> -o <output_csr_bin> -s 1024 -a 7 -t 20 1>> corder-console-info.txt

rm corder-console-info.txt

# c7522
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/c7522-d40.csr-bin -o ~/Research/large-benchmarks/c7522-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# des_perf
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/des_perf-d40.csr-bin -o ~/Research/large-benchmarks/des_perf-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# vga_lcd
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/vga_lcd-d40.csr-bin -o ~/Research/large-benchmarks/vga_lcd-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# leon3mp
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/leon3mp-d40.csr-bin -o ~/Research/large-benchmarks/leon3mp-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# netcard
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/netcard-d40.csr-bin -o ~/Research/large-benchmarks/netcard-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# leon2
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/leon2-d40.csr-bin -o ~/Research/large-benchmarks/leon2-d40-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# ldoor
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/DIMACS/ldoor.csr-bin -o ~/Research/large-benchmarks/DIMACS/ldoor-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# cage15
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/DIMACS/cage15.csr-bin -o ~/Research/large-benchmarks/DIMACS/cage15-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# nlpkkt120
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/DIMACS/nlpkkt120.csr-bin -o ~/Research/large-benchmarks/DIMACS/nlpkkt120-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

# nlpkkt160
/home/cchang289/Research/Corder-TPDS-21/corder -d ~/Research/large-benchmarks/DIMACS/nlpkkt160.csr-bin -o ~/Research/large-benchmarks/DIMACS/nlpkkt160-corder.csr-bin -s 1024 -a 7 -t 20 1>> corder-console-info.txt

