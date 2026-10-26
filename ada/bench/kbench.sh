#!/bin/bash
# Build bench/kbench (the CPU expert kernels on their own) against the fork's built libraries.
E=${E:-$HOME/git/strata/eddoursul}
D=${D:-$HOME/git/strata/deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d}
g++ -O3 -std=c++20 -mavx2 -mfma -mf16c -DSTRATA_NATIVE_EXPERTS=1 -I$E/include -I$E/third_party/ggml -I$D/ggml/include \
  "$(dirname "$0")/kbench.cpp" -o "$(dirname "$0")/kbench" $E/build/libstrata_kernels_cpu.a \
  $E/build/ggml/src/libggml-cpu.a $E/build/ggml/src/libggml-base.a -lpthread -fopenmp
