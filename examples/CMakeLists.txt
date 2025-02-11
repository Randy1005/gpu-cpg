add_executable(reduce ${GPUCPG_EXAMPLE_DIR}/reduce.cu)
add_executable(cpg ${GPUCPG_EXAMPLE_DIR}/cpg.cu)
add_executable(cpg-short-long ${GPUCPG_EXAMPLE_DIR}/cpg-short-long.cu)
add_executable(levelize ${GPUCPG_EXAMPLE_DIR}/levelize.cu)
add_executable(gen_benchmark ${GPUCPG_EXAMPLE_DIR}/gen_benchmark.cu)
add_executable(gen_sizedup_benchmark ${GPUCPG_EXAMPLE_DIR}/gen_sizedup_benchmark.cu)
add_executable(get_node_by_idx ${GPUCPG_EXAMPLE_DIR}/get_node_by_idx.cu)
add_executable(cpg-compare ${GPUCPG_EXAMPLE_DIR}/cpg-compare.cu)
add_executable(cpg-slack-error ${GPUCPG_EXAMPLE_DIR}/cpg-slack-error.cu)
add_executable(cpg-mteps ${GPUCPG_EXAMPLE_DIR}/cpg-mteps.cu)
add_executable(bfs_hybrid ${GPUCPG_EXAMPLE_DIR}/bfs_hybrid.cu)

list(APPEND GPUCPG_EXAMPLES
  reduce
  cpg
  cpg-short-long 
  cpg-compare
  cpg-slack-error
  cpg-mteps
  levelize
  gen_benchmark
  gen_sizedup_benchmark
  get_node_by_idx
  bfs_hybrid
)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg)

endforeach()
