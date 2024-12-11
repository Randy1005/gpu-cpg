enable_testing()
include(${GPUCPG_3RDPARTY_DIR}/doctest/doctest.cmake)
list(APPEND CU_UNITTESTS
  memcpy 
)

foreach(unittest IN LISTS CU_UNITTESTS)
  add_executable(${unittest} ${unittest}.cu)
	add_test(NAME ${unittest} COMMAND ${unittest})	
	set_property(TEST ${unittest} PROPERTY LABELS "unit")
  target_include_directories(${unittest} 
    PRIVATE ${PROJECT_SOURCE_DIR} ${GPUCPG_3RDPARTY_DIR}/doctest)

  target_link_libraries(${unittest} 
    LINK_PUBLIC gpucpg)
  
  doctest_discover_tests(${unittest})
endforeach()
