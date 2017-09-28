project(CeresPro)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

file(GLOB SRCS ./cpp/*.cpp)
add_executable(Test ./cpp/test.cxx ${SRCS})

find_path(EIGEN3_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library PATH_SUFFIXES eigen3 eigen)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
