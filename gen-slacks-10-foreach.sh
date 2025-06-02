#!/bin/bash

for density in 70 80
do
    for (( i=3; i<=4; i++ ))
    do
        echo "gen slacks for $density, iteration $i"
        # cage15
        ./build/examples/cpg ~/Research/large-benchmarks/DIMACS/cage15_dense$density-$i.txt 1000000 ~/Research/large-benchmarks/DIMACS/cage15_dense$density-$i.slks
    done
done


