# call rabbit reorder on each elist file
#!/bin/bash

# ./../rabbit_order/demo/reorder [elist file] 1> [vmap file] 2>> [info file]
for elist_file in ~/Research/large-benchmarks/des_perf-d40.elist \
                  ~/Research/large-benchmarks/vga_lcd-d40.elist \
                  ~/Research/large-benchmarks/leon3mp-d40.elist \
                  ~/Research/large-benchmarks/netcard-d40.elist \
                  ~/Research/large-benchmarks/leon2-d40.elist \
                  ~/Research/large-benchmarks/DIMACS/ldoor.elist \
                  ~/Research/large-benchmarks/DIMACS/cage15.elist \
                  ~/Research/large-benchmarks/DIMACS/nlpkkt120.elist \
                  ~/Research/large-benchmarks/DIMACS/nlpkkt160.elist
do
    # Extract the base name of the elist file
    base_name=$(basename "$elist_file" .elist)
    
    # Define the output vmap file
    vmap_file=${base_name}.vmap
    
    # Run the rabbit reorder command
    echo "Reordering $elist_file to $vmap_file"
    ./../rabbit_order/demo/reorder "$elist_file" 1> "$vmap_file" 2>> "rabbit_console_info.txt"
done

