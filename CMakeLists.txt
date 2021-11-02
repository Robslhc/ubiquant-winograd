cmake_minimum_required(VERSION 3.2)

project(winograd)

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
option(USE_WINO_B4 "use winograd algorithm with block size 4" ON)
option(DEBUG_MODE "debug mode" OFF)

include_directories(include)
include_directories(src)

message(STATUS "Use winograd b4 ${USE_WINO_B4}")
message(STATUS "Debug mode ${DEBUG_MODE}")

if (USE_WINO_B4)
    add_definitions(-DWINO_B4)
endif()

if (DEBUG_MODE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    add_definitions(-D__DEBUG)
else ()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
endif()

# enable avx512
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx512f")

# enable openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SRC_FILES "*.c"
                    "src/wgb4f3/avx512/*.h"
                    "src/wgb4f3/avx512/*.c")

add_executable(winograd ${SRC_FILES})
