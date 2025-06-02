#!/bin/bash

# call densify 10 times for each density
# for loop in bash
for density in 70 80
do
    for (( i=3; i<=4; i++ ))
    do
        echo "Running densify with density $density, iteration $i"
        # cage15
        ./build/examples/densify $density ~/Research/large-benchmarks/DIMACS/cage15.txt ~/Research/large-benchmarks/DIMACS/cage15_dense$density-$i.txt
    done
done
