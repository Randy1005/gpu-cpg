add_library(gpucpg gpucpg.cu)

# find the pthreads library
find_package(Threads REQUIRED)

target_link_libraries(gpucpg 
  PUBLIC CCCL::CCCL
  PRIVATE cudart
)

