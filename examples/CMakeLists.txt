add_executable(cpg ${GPUCPG_EXAMPLE_DIR}/cpg.cu)
add_executable(levelize ${GPUCPG_EXAMPLE_DIR}/levelize.cu)
add_executable(gen_benchmark ${GPUCPG_EXAMPLE_DIR}/gen_benchmark.cu)
add_executable(gen_sizedup_benchmark ${GPUCPG_EXAMPLE_DIR}/gen_sizedup_benchmark.cu)
add_executable(get_node_by_idx ${GPUCPG_EXAMPLE_DIR}/get_node_by_idx.cu)
add_executable(cpg-compare ${GPUCPG_EXAMPLE_DIR}/cpg-compare.cu)
add_executable(cpg-mteps ${GPUCPG_EXAMPLE_DIR}/cpg-mteps.cu)
add_executable(bfs_hybrid ${GPUCPG_EXAMPLE_DIR}/bfs_hybrid.cu)
add_executable(bfs_td ${GPUCPG_EXAMPLE_DIR}/bfs_td.cu)
add_executable(densify ${GPUCPG_EXAMPLE_DIR}/densify.cu)
add_executable(lvlize_then_relax ${GPUCPG_EXAMPLE_DIR}/lvlize_then_relax.cu)
add_executable(dac21-baseline ${GPUCPG_EXAMPLE_DIR}/dac21-baseline.cu)
add_executable(convert-dimacs ${GPUCPG_EXAMPLE_DIR}/convert-dimacs.cu)
add_executable(gen-big-table ${GPUCPG_EXAMPLE_DIR}/gen-big-table.cu)
add_executable(gen-runtime-error-vs-k ${GPUCPG_EXAMPLE_DIR}/gen-runtime-error-vs-k.cu)
add_executable(gen-runtime-error-vs-avgdeg ${GPUCPG_EXAMPLE_DIR}/gen-runtime-error-vs-avgdeg.cu)
add_executable(gen-runtime-error-vs-diam ${GPUCPG_EXAMPLE_DIR}/gen-runtime-error-vs-diam.cu)
add_executable(abl-study-gen-runtime-vs-delta ${GPUCPG_EXAMPLE_DIR}/abl-study-gen-runtime-vs-delta.cu)
add_executable(compare-warp-spur ${GPUCPG_EXAMPLE_DIR}/compare-warp-spur.cu)
add_executable(cpg-runtime-breakdown ${GPUCPG_EXAMPLE_DIR}/cpg-runtime-breakdown.cu)
add_executable(dump-graph-stats ${GPUCPG_EXAMPLE_DIR}/dump-graph-stats.cu)
add_executable(dump-elist ${GPUCPG_EXAMPLE_DIR}/dump-elist.cu)
add_executable(cpg-other-vrs ${GPUCPG_EXAMPLE_DIR}/cpg-other-vrs.cu)


list(APPEND GPUCPG_EXAMPLES
  cpg 
  cpg-other-vrs
  cpg-compare
  cpg-mteps
  levelize
  gen_benchmark
  gen_sizedup_benchmark
  get_node_by_idx
  bfs_hybrid
  bfs_td
  densify
  lvlize_then_relax
  dac21-baseline
  convert-dimacs
  gen-big-table
  gen-runtime-error-vs-k
  gen-runtime-error-vs-avgdeg
  gen-runtime-error-vs-diam
  abl-study-gen-runtime-vs-delta
  compare-warp-spur
  cpg-runtime-breakdown
  dump-graph-stats
  dump-elist)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg)

endforeach()
