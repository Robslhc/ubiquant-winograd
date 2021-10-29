#!/bin/bash

set -e

TOP_DIR=$(dirname $(realpath $0))

if [ $# -lt 1 ]; then
    echo "[Usage]: regression.sh testrelease/build/rebuild/validation/benchmark/clean [options]"
elif [ $1 == "testrelease" ]; then
    clang-format -i *.c
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir build; cd build;
    shift
    cmake $* .. -G Ninja
    cmake --build .
    cd ..

    build/winograd small.conf 1
    build/winograd realworld.conf 1
elif [ $1 == "build" ]; then
    if [ ! -d "build" ]; then
        mkdir build; cd build;
        shift
        cmake $* .. -G Ninja
        cmake --build .
        cd ..
    fi
    cd build
    if [ -f "build.ninja" ]; then
        ninja
    else
        make -j
    fi
    cd ..
elif [ $1 == "rebuild" ]; then
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir build; cd build;
    shift
    cmake $* .. -G Ninja
    cmake --build .
    cd ..
elif [ $1 == "validation" ]; then
    if [ ! -d "build" ]; then
        mkdir build; cd build;
        shift
        cmake $* .. -G Ninja
        cmake --build .
        cd ..
    fi
    build/winograd small.conf 1
elif [ $1 == "benchmark" ]; then
    if [ ! -d "build" ]; then
        mkdir build; cd build;
        shift
        cmake $* .. -G Ninja
        cmake --build .
        cd ..
    fi
    build/winograd small.conf 0
    build/winograd realworld.conf 0
elif [ $1 == "clean" ]; then
    if [ -d "build" ]; then
        rm -rf build
    fi
    if [ -f "winograd" ]; then
        rm winograd
    fi
fi