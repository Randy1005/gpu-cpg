cmake_minimum_required(VERSION 3.21.0)

project(gpuCPG VERSION 1.0 LANGUAGES CXX CUDA)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY "nvidia/cccl" 
    GIT_TAG "main"
)

set(CMAKE_C_STANDARD    17)
set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "-std=c++20")

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()


# Default to building for the GPU on the current system
set(CMAKE_CUDA_ARCHITECTURES 86)


# compile and link CUDA code separately from the rest of your code to
# speed up the build process and reduce the size of your final executable.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -Wall --extended-lambda")

# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release ")
  set(
    CMAKE_BUILD_TYPE Release
    CACHE
    STRING "Choose the type of build."
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(GPUCPG_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
set(GPUCPG_BENCHMARKS_DIR ${PROJECT_SOURCE_DIR}/benchmarks)
set(GPUCPG_3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------
include(CheckLanguage)

# Enable test
include(CTest)
enable_testing()
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_subdirectory(gpucpg)
add_subdirectory(examples)
add_subdirectory(unittests)

