#!/bin/bash

# ./../pcpm/csr_gen/a.out <inputFile> <outputFile> -w 1

# c7522
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/c7522-d40.elist ~/Research/large-benchmarks/c7522-d40.csr-bin -w 1

# des_perf
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/des_perf-d40.elist ~/Research/large-benchmarks/des_perf-d40.csr-bin -w 1

# vga_lcd
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/vga_lcd-d40.elist ~/Research/large-benchmarks/vga_lcd-d40.csr-bin -w 1

# leon3mp
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/leon3mp-d40.elist ~/Research/large-benchmarks/leon3mp-d40.csr-bin -w 1

# netcard
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/netcard-d40.elist ~/Research/large-benchmarks/netcard-d40.csr-bin -w 1

# leon2
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/leon2-d40.elist ~/Research/large-benchmarks/leon2-d40.csr-bin -w 1

# ldoor
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/DIMACS/ldoor.elist ~/Research/large-benchmarks/DIMACS/ldoor.csr-bin -w 1

# cage15
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/DIMACS/cage15.elist ~/Research/large-benchmarks/DIMACS/cage15.csr-bin -w 1

# nlpkkt120
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/DIMACS/nlpkkt120.elist ~/Research/large-benchmarks/DIMACS/nlpkkt120.csr-bin -w 1

# nlpkkt160
./../pcpm/csr_gen/a.out ~/Research/large-benchmarks/DIMACS/nlpkkt160.elist ~/Research/large-benchmarks/DIMACS/nlpkkt160.csr-bin -w 1




