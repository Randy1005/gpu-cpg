./gen-runtime-err-vs-avgdeg.sh


./densify-10-foreach.sh
./gen-slacks-10-foreach.sh

# dense70
./build/examples/gen-runtime-error-vs-avgdeg ~/Research/large-benchmarks/DIMACS/cage15_dense70 1
./build/examples/gen-runtime-error-vs-avgdeg ~/Research/large-benchmarks/DIMACS/cage15_dense70 2

# dense80
./build/examples/gen-runtime-error-vs-avgdeg ~/Research/large-benchmarks/DIMACS/cage15_dense80 1
./build/examples/gen-runtime-error-vs-avgdeg ~/Research/large-benchmarks/DIMACS/cage15_dense80 2