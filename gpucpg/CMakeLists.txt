add_library(gpucpg gpucpg.cu)
add_library(graph graph.cu)

# include the moderngpu path
# include_directories(${GPUCPG_3RDPARTY_DIR}/moderngpu/src)

# find the pthreads library
find_package(Threads REQUIRED)

# find the OpenMP library
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gpucpg PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gpucpg
  PUBLIC graph 
  PUBLIC CCCL::CCCL
  PRIVATE cudart
)

target_compile_options(gpucpg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       --relocatable-device-code=true
                       -fopenmp
                       -lineinfo
                      #  -Xptxas=-v
                       >)
