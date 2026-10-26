#!/bin/sh
# Builds bench/gk against the engine's kernel library (build it first: make -C eddoursul/build strata)
cd "$(dirname "$0")"
E=../../..
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -I$E/include gk.cu $E/build/libstrata_kernels.a -o gk "$@"
