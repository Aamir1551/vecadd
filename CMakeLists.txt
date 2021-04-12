cmake_minimum_required(VERSION 3.17)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(opencl_prac VERSION 0.0.0 LANGUAGES C)

set(CMAKE_CXX_STANDARD 14)

add_executable(opencl_prac vectorAddition.c)
target_include_directories(opencl_prac PUBLIC "C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.2/include" "{CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(opencl_prac PUBLIC "C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.2/lib/Win32/OpenCL.lib")
#configure_file(./vectorAddition.cl COPYONLY)