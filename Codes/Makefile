NVCC ?= nvcc
CXX ?= c++
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
# Optional, e.g. make CUDA_ARCH=sm_80. Use the architecture of YOUR course GPU.
CUDA_ARCH ?=
ARCH_FLAGS = $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),)

.PHONY: all cpu-test quick full sanitize clean

all: test_step1 test_cpu

test_step1: tests/test_step1.cu include/block_scan.cuh include/validation.hpp
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) -Iinclude $< -o $@

test_cpu: tests/test_cpu.cpp include/validation.hpp
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

cpu-test: test_cpu
	./test_cpu

quick: test_step1
	./test_step1 --suite quick

full: test_step1
	./test_step1 --suite full

# Keep sanitizer runs small but cover all types, all patterns and integration.
# Additional B=1024 tests exercise the maximum number of warps per block.
sanitize: test_step1
	compute-sanitizer --error-exitcode 1 --tool memcheck ./test_step1 --suite quick --type all
	compute-sanitizer --error-exitcode 1 --tool racecheck ./test_step1 --suite quick --type all
	compute-sanitizer --error-exitcode 1 --tool synccheck ./test_step1 --suite quick --type all
	compute-sanitizer --error-exitcode 1 --tool memcheck ./test_step1 --type all --block 1024 --items 4 --n 4099
	compute-sanitizer --error-exitcode 1 --tool racecheck ./test_step1 --type all --block 1024 --items 4 --n 4099
	compute-sanitizer --error-exitcode 1 --tool synccheck ./test_step1 --type all --block 1024 --items 4 --n 4099

clean:
	rm -f test_step1 test_cpu
