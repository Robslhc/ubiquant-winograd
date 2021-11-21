cmake_minimum_required(VERSION 3.2)

project(winograd)

set(CMAKE_C_STANDARD 11)
option(USE_WINO_B4 "use winograd algorithm with block size 4" ON)
option(USE_OPENMP "use openmp for winograd" ON)
option(USE_INLINE_ASM "use inline asm for kernels" OFF)
option(DEBUG_MODE "debug mode" OFF)
option(PROFILE_MODE "profile mode" OFF)
option(PARALLEL_MODE_OUTER "enable parallel outer optimize" OFF)
option(PARALLEL_MODE_INNER "enable parallel outer optimize" OFF)
option(AUTO_SELECT_TILE_BLK "auto select tile blk for parallel outer mode" OFF)

include_directories(include)
include_directories(src)

message(STATUS "Use winograd b4 ${USE_WINO_B4}")
message(STATUS "Debug mode ${DEBUG_MODE}")
message(STATUS "Profile mode ${PROFILE_MODE}")
message(STATUS "Enable parallel outer ${PARALLEL_MODE_OUTER}")
message(STATUS "Enable parallel inner ${PARALLEL_MODE_INNER}")
message(STATUS "Autoselect tile blk ${AUTO_SELECT_TILE_BLK}")

if (USE_WINO_B4)
    add_definitions(-DWINO_B4)
endif()

if(PARALLEL_MODE_OUTER)
    add_definitions(-DFORCE_PARALLEL_OUTER)
    if(AUTO_SELECT_TILE_BLK)
        add_definitions(-DAUTO_SELECT_TILE_BLK)
    endif()
endif()

if(PARALLEL_MODE_INNER)
    add_definitions(-DFORCE_PARALLEL_INNER)
endif()

if(USE_INLINE_ASM)
    add_definitions(-DUSE_INLINE_ASM)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-function-declaration")
endif()

if (DEBUG_MODE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    add_definitions(-D__DEBUG)
else ()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
endif()

if (PROFILE_MODE)
    add_definitions(-DPROFILE)
endif()

# enable avx512
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-discarded-qualifiers -mavx -mavx512f")

# enable openmp
if (USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

file(GLOB SRC_FILES "*.c"
                    "src/wgb4f3/avx512/*.h"
                    "src/wgb4f3/avx512/*.c")

add_executable(winograd ${SRC_FILES})
