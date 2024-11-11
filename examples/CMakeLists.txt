add_executable(reduce ${GPUCPG_EXAMPLE_DIR}/reduce.cu)
add_executable(read_input ${GPUCPG_EXAMPLE_DIR}/read_input.cpp)

list(APPEND GPUCPG_EXAMPLES
  reduce
  read_input
)

foreach(example IN LISTS GPUCPG_EXAMPLES)
  target_include_directories(${example}
    PUBLIC ${PROJECT_SOURCE_DIR}/gpucpg) 
  target_link_libraries(${example}
    PUBLIC gpucpg)

endforeach()
