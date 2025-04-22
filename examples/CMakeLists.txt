#add_executable(cpg ${GPUCPG_EXAMPLE_DIR}/cpg.cu)
add_executable(cpg-short-long ${GPUCPG_EXAMPLE_DIR}/cpg-short-long.cu)
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

list(APPEND GPUCPG_EXAMPLES
  #cpg
  cpg-short-long 
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
)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg)

endforeach()
