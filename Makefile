# ikun - GPU-aware Distributed Collective Runner (C implementation)
#
# Pure C, no external dependencies. Compiles with gcc on Ubuntu 20.04.
# Local hardware: 4× Iluvatar BI-V100, CoreX 3.2.3, CUDA 10.2 compat
#
# Quick build:    make build   (just the C runner, gcc only)
# Full build:     make cmake   (C runner + core engine via cmake)
# Smoke test:     make run

CC := gcc
CFLAGS := -Wall -Wextra -O2 -std=c11 -I include
LDFLAGS :=
BINARY := bin/ikun
CMAKE := /usr/local/bin/cmake
GPUS ?= 0,1,2,3

SRC := src/ikun.c src/main.c

.PHONY: all build cmake run clean

all: build run

build: $(BINARY)

$(BINARY): $(SRC) include/ikun.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $(BINARY) $(SRC) $(LDFLAGS)

cmake:
	@mkdir -p build_full
	cd build_full && $(CMAKE) .. -DUSE_ILU=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
	cd build_full && $(MAKE) -j$$(nproc)

run: $(BINARY)
	CUDA_VISIBLE_DEVICES=$(GPUS) ./$(BINARY)

clean:
	rm -rf bin/ build_full/ build_core/ build_cmake/ build_diag/
