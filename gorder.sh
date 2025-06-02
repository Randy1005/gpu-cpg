#!/bin/bash

# generate gorder csr (only fanout) files
# /home/cchang289/Research/Gorder/Gorder <elist_file>
rm gorder-console-info.txt

# c7522
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/c7522-d40.elist 1>> gorder-console-info.txt

# des_perf
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/des_perf-d40.elist 1>> gorder-console-info.txt

# vga_lcd
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/vga_lcd-d40.elist 1>> gorder-console-info.txt

# leon3mp
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/leon3mp-d40.elist 1>> gorder-console-info.txt

# netcard
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/netcard-d40.elist 1>> gorder-console-info.txt

# leon2
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/leon2-d40.elist 1>> gorder-console-info.txt

# ldoor
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/DIMACS/ldoor.elist 1>> gorder-console-info.txt
# cage15
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/DIMACS/cage15.elist 1>> gorder-console-info.txt
# nlpkkt120
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/DIMACS/nlpkkt120.elist 1>> gorder-console-info.txt
# nlpkkt160
/home/cchang289/Research/Gorder/Gorder ~/Research/large-benchmarks/DIMACS/nlpkkt160.elist 1>> gorder-console-info.txt
