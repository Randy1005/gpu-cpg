add_library(gpucpg gpucpg.cu)

# find the pthreads library
find_package(Threads REQUIRED)

target_link_libraries(gpucpg 
  PUBLIC CCCL::CCCL
  PRIVATE cudart
)

target_compile_options(gpucpg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
option(TIME_BFS_STEP "Time each BFS steps" OFF)
if (TIME_BFS_STEP)
  target_compile_definitions(gpucpg PUBLIC TIME_BFS_STEP=${TIME_BFS_STEP})
endif()
