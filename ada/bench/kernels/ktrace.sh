#!/bin/sh
# Builds bench/libktrace.so against the CUPTI the vLLM venv ships (the toolkit here has none).
V=/home/mikaeljo/git/vllm/qwen38-27b-rtx3090/venv/lib/python3.12/site-packages/nvidia/cu13
g++ -O2 -std=c++17 -shared -fPIC -o "$(dirname "$0")/libktrace.so" "$(dirname "$0")/ktrace.cpp" -I$V/include -I/usr/local/cuda/include -L$V/lib -l:libcupti.so.13 -Wl,-rpath,$V/lib
