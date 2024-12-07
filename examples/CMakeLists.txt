add_executable(reduce ${GPUCPG_EXAMPLE_DIR}/reduce.cu)
add_executable(cpg ${GPUCPG_EXAMPLE_DIR}/cpg.cpp)
add_executable(levelize ${GPUCPG_EXAMPLE_DIR}/levelize.cpp)

target_compile_options(cpg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--line-info>) 

list(APPEND GPUCPG_EXAMPLES
  reduce
  cpg 
  levelize
)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg)

endforeach()
