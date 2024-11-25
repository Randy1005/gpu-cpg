add_library(gpucpg gpucpg.cu)

# find the pthreads library
find_package(Threads REQUIRED)

target_link_libraries(gpucpg 
  PUBLIC CCCL::CCCL
  PRIVATE cudart
)

target_compile_options(gpucpg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
