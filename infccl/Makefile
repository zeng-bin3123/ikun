CXX      := /usr/local/corex/bin/clang++
ARCH     := ivcore10
CUDAPATH := /usr/local/corex-3.2.3
CFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) -I/usr/local/corex/include -Isrc
LDFLAGS  := -L/usr/local/corex/lib64 -lcudart

SRCS     := src/core.cu src/all_reduce.cu src/reduce.cu src/broadcast.cu src/all_gather.cu src/reduce_scatter.cu
TESTS    := all_reduce_test reduce_test broadcast_test all_gather_test reduce_scatter_test

all: $(TESTS)

all_reduce_test: $(SRCS) test/all_reduce_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

reduce_test: $(SRCS) test/reduce_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

broadcast_test: $(SRCS) test/broadcast_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

all_gather_test: $(SRCS) test/all_gather_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

reduce_scatter_test: $(SRCS) test/reduce_scatter_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TESTS)
	@for t in $(TESTS); do echo "=== $$t ===" && CUDA_VISIBLE_DEVICES=0,1,2 timeout 120 ./$$t && echo "" || exit 1; done

clean:
	rm -f $(TESTS)

.PHONY: all test clean
