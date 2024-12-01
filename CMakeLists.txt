cmake_minimum_required(VERSION 3.10)
project(RayTracing LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA language
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architectures (modify if targeting different hardware)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Include directories for headers
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Define the executable target
add_executable(pathfinding ${SOURCES})

# Enable CUDA separable compilation
set_target_properties(pathfinding PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CUDA flags for optimization and architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda --expt-relaxed-constexpr")

# Linker flags for libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed -pthread -ldl -lrt")