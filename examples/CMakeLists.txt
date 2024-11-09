set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)

add_executable(reduce ${GPUCPG_EXAMPLE_DIR}/reduce.cpp)

list(APPEND GPUCPG_EXAMPLES
  reduce
)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg CCCL::CCCL)

endforeach()
