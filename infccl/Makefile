CXX      := /usr/local/corex/bin/clang++
ARCH     := ivcore10
CUDAPATH := /usr/local/corex
CFLAGS   := -x cuda --cuda-gpu-arch=$(ARCH) --cuda-path=$(CUDAPATH) -I$(CUDAPATH)/include -Isrc -O2
LDFLAGS  := -L$(CUDAPATH)/lib64 -lcudart -lpthread
SRCS     := src/peer_transport.cu src/transfer_engine_c.cu src/collectives.cu
TESTS    := all_reduce_test transfer_test collective_test
BENCH    := transfer_bench

all: $(TESTS) $(BENCH)

all_reduce_test: $(SRCS) test/all_reduce_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)
transfer_test: src/peer_transport.cu test/transfer_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)
collective_test: $(SRCS) test/collective_test.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)
transfer_bench: src/peer_transport.cu src/collectives.cu src/transfer_engine_c.cu bench/transfer_bench.cu
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)
test: $(TESTS)
	@for t in $(TESTS); do echo "=== $$t ===" && CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 120 ./$$t && echo "" || exit 1; done
bench: $(BENCH)
	CUDA_VISIBLE_DEVICES=0,1,2,3 ./transfer_bench
clean:
	rm -f $(TESTS) $(BENCH)
.PHONY: all test bench clean
