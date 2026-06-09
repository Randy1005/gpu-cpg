add_library(gpucpg gpucpg.cu)
set_target_properties(gpucpg PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(graph graph.cu)
set_target_properties(graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

# include the moderngpu path
# include_directories(${GPUCPG_3RDPARTY_DIR}/moderngpu/src)

# find the pthreads library
find_package(Threads REQUIRED)

# find the OpenMP library
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gpucpg PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(gpucpg
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_path(GPUCPG_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
  HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PATHS
    /usr/local/cuda/include
    /usr/local/cuda-12.6/targets/x86_64-linux/include
)
if(GPUCPG_NVTX_INCLUDE_DIR)
  target_include_directories(gpucpg PRIVATE ${GPUCPG_NVTX_INCLUDE_DIR})
  target_compile_definitions(gpucpg PRIVATE GPUCPG_HAS_NVTX=1)
else()
  target_compile_definitions(gpucpg PRIVATE GPUCPG_HAS_NVTX=0)
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
