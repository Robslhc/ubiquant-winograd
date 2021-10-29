cmake_minimum_required(VERSION 3.2)

project(winograd)

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# enable openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SRC_FILES "driver.c"
                    "winograd.c")

add_executable(winograd ${SRC_FILES})