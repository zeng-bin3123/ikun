# infccl Makefile for Iluvatar BI-V100 (CoreX toolchain)
#
# Usage:
#   make                    # build all tests
#   make test               # run all tests
#   make bench              # run allreduce benchmark
#   make clean

CXX       = /usr/local/corex/bin/clang++
CUDA_ARCH = --cuda-gpu-arch=ivcore10
CUDA_PATH = /usr/local/corex-3.2.3
CXXFLAGS  = -x cuda $(CUDA_ARCH) --cuda-path=$(CUDA_PATH) \
            -I$(CUDA_PATH)/include -Isrc -O2 -std=c++14
LDFLAGS   = -L$(CUDA_PATH)/lib64 -lcudart -lpthread

SRCS = src/core.cu src/all_reduce.cu src/reduce.cu \
       src/broadcast.cu src/all_gather.cu src/reduce_scatter.cu

TESTS = all_reduce_test reduce_scatter_test all_gather_test \
        broadcast_test reduce_test

.PHONY: all test bench clean

all: $(TESTS)

all_reduce_test: test/all_reduce_test.cu $(SRCS) src/*.h
	$(CXX) $(CXXFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

reduce_scatter_test: test/reduce_scatter_test.cu $(SRCS) src/*.h
	$(CXX) $(CXXFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

all_gather_test: test/all_gather_test.cu $(SRCS) src/*.h
	$(CXX) $(CXXFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

broadcast_test: test/broadcast_test.cu $(SRCS) src/*.h
	$(CXX) $(CXXFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

reduce_test: test/reduce_test.cu $(SRCS) src/*.h
	$(CXX) $(CXXFLAGS) $(SRCS) $< -o $@ $(LDFLAGS)

test: $(TESTS)
	@echo "=== Running all tests ==="
	@pass=0; fail=0; \
	for t in $(TESTS); do \
	  echo "--- $$t ---"; \
	  CUDA_VISIBLE_DEVICES=0,1,2,3 ./$$t && pass=$$((pass+1)) || fail=$$((fail+1)); \
	done; \
	echo "=== $$pass passed, $$fail failed ==="

bench: all_reduce_test
	CUDA_VISIBLE_DEVICES=0,1,2,3 ./all_reduce_test

clean:
	rm -f $(TESTS)
