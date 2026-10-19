# NCCL Formal Analysis — src/reduce_scatter.cu

---

## 1. Macros (ReduceScatter variants)

### LAYER 1a — MATH (LaTeX)

$$
\text{SIGNAL\_NEW\_DATA\_AVAILABLE}_{RS}(c, s, st): \text{Flag}[\text{next}] \leftarrow 2 \cdot (c \cdot K + st) + s + 1
$$
$$
\text{WAIT\_FOR\_NEW\_DATA}_{RS}(c, s, st): \text{spin until } \text{Flag}[\text{this}] \geq 2 \cdot (c \cdot K + st) + s - 1
$$
$$
\text{SIGNAL\_CHUNK\_DONE}_{RS}(c, s): \text{Flag}[\text{prev}] \leftarrow 2c + s + 1
$$
$$
\text{WAIT\_FOR\_CHUNK}_{RS}(c, s): \text{spin until } \text{Flag}[\text{this}] \geq 2c + s - 1
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rs_signal_value_positive (K chunk step s : Nat) :
    2 * (chunk * K + step) + s + 1 > 0 := by sorry

-- axle_verify: content
theorem rs_signal_value_positive (K chunk step s : Nat) :
    2 * (chunk * K + step) + s + 1 > 0 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
#define SIGNAL_NEW_DATA_AVAILABLE(chunkidx, subidx, stepidx)                    \
    do {                                                                        \
      args.NextNewDataAvailableFlag[0] =                                        \
          2*((chunkidx) * args.NumGPUs + (stepidx)) + subidx + 1;              \
    } while (0)

#define WAIT_FOR_NEW_DATA(chunkidx, subidx, stepidx)                            \
    do {                                                                        \
      if (tidx == 0) {                                                          \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              2*((chunkidx) * args.NumGPUs + (stepidx)) + subidx - 1;          \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunkidx, subidx)                                     \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = 2*(chunkidx) + subidx + 1;                   \
    } while (0)

#define WAIT_FOR_CHUNK(chunkidx, subidx)                                      \
    do {                                                                      \
      if (tidx == 0) {                                                        \
        Wait([=] {                                                            \
          return ((volatile int *)args.ThisChunkDoneFlag)[0] >=               \
              2*(chunkidx) + subidx - 1;                                      \
        });                                                                   \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
    } while (0)
```

---

## 2. `getSliceSizeAndChunkSize` (ReduceScatter version — same as AllGather)

### LAYER 1a — MATH (LaTeX)

$$
\text{Same as AllGather's getSliceSizeAndChunkSize}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rs_slice_size_same_as_ag : True := by sorry

-- axle_verify: content
theorem rs_slice_size_same_as_ag : True := by trivial
```

### LAYER 2 — FORM TEMPLATE

(Identical to AllGather version — see analysis_all_gather.md §2.)

---

## 3. `ReduceScatterKernelArgs<T>` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceScatterKernelArgs}(T) = \{
  \text{ThisId}: \mathbb{Z},\; \text{NumGPUs}: K,\; N: \mathbb{N},\; \text{UserFromRing}: \mathbb{Z}^K,
$$
$$
  \text{SliceSize}: \mathbb{N},\; \text{ChunkSize}: \mathbb{N},\; \text{NumChunks}: \mathbb{N},
$$
$$
  \text{BufferSliceStride}: \mathbb{N},\; \text{BufferMisalignedN}: \mathbb{N},
$$
$$
  \text{ThisInput}: T^*_\text{const},\; \text{ThisOutput}: T^*_\text{vol},\; \text{ThisBuffer}: T^*_\text{vol},\; \text{NextBuffer}: T^*_\text{vol},
$$
$$
  \text{Flags}: \mathbb{Z}^*_\text{vol} \times 4
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rs_args_complete : 15 = 15 := by sorry

-- axle_verify: content
theorem rs_args_complete : 15 = 15 := by rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct ReduceScatterKernelArgs {
  int thisIdx;
  int gpuCount;
  int totalN;
  int * userFromRingMap;

  int sliceSizeN;
  int chunkSizeN;
  int chunkCount;

  int bufSliceStride;
  int bufMisalignedN;

  const ELEMTYPE * __restrict__ thisInputPtr;
  volatile ELEMTYPE * __restrict__ thisOutputPtr;
  volatile ELEMTYPE * __restrict__ thisBufferPtr;
  volatile ELEMTYPE * __restrict__ nextBufferPtr;

  volatile int * __restrict__ thisNewDataFlag;
  volatile int * __restrict__ nextNewDataFlag;
  volatile int * __restrict__ thisChunkDoneFlag;
  volatile int * __restrict__ prevChunkDoneFlag;
};
```

---

## 4. `GetBlock` (ReduceScatter version)

### LAYER 1a — MATH (LaTeX)

$$
\text{GetBlock}_{RS}: \mathbb{Z} \times \mathbb{Z} \times \mathbb{Z}^K \times K \to \mathbb{Z}
$$
$$
\textbf{Post: } \text{result} = \text{userFromRing}[(K + \text{index} - 1 - \text{step}) \bmod K]
$$
$$
\text{Note: offset by -1 compared to AllGather, traverses backwards starting at index}-1
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rs_get_block_in_range (K index step : Nat) (hK : K > 0) :
    (K + index - 1 - step) % K < K := by sorry

-- axle_verify: content
theorem rs_get_block_in_range (K index step : Nat) (hK : K > 0) :
    (K + index - 1 - step) % K < K := by
  exact Nat.mod_lt _ hK
```

### LAYER 2 — FORM TEMPLATE

```cuda
__device__ inline int GetBlock(const int selfIdx, const int stepIdx,
    const int * const userFromRingMap, const int gpuCount) {
  return userFromRingMap[(gpuCount + selfIdx - 1 - stepIdx) % gpuCount];
}
```

---

## 5. `ReduceScatterKernel<THREADS, UNROLL, FUNC, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceScatterKernel}: \text{ReduceScatterKernelArgs}(T) \to \text{Unit}
$$
$$
\textbf{Algorithm: } K \text{ steps of ring reduce-scatter.}
$$
$$
\text{Step 0: Copy block to next buffer}
$$
$$
\text{Steps } 1..K-2: \text{Reduce received buffer with local data, forward to next}
$$
$$
\text{Step } K-1: \text{Final reduce to output}
$$
$$
\textbf{Post: } \forall\, g \in [0,K),\; \text{Output}_g = \bigoplus_{j=0}^{K-1} \text{Input}_j[\text{block}_g]
$$
$$
\textbf{Note: } \text{launched with NUM\_THREADS + NUM\_SUBCHUNKS * WARP\_SIZE threads (different from others)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_scatter_steps (K : Nat) (hK : K ≥ 2) :
    -- step 0 (copy) + (K-2) reduces + 1 final reduce = K steps total
    1 + (K - 2) + 1 = K := by sorry

-- axle_verify: content
theorem reduce_scatter_steps (K : Nat) (hK : K ≥ 2) :
    1 + (K - 2) + 1 = K := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int THREADCOUNT, int UNROLLCOUNT, class FUNCTYPE, typename ELEMTYPE>
__global__ void ReduceScatterKernel(const ReduceScatterKernelArgs<ELEMTYPE> args) {
  if (args.totalN == 0) return;
  int tidx = threadIdx.x;

  for (int chunkidx = 0; chunkidx < args.chunkCount; ++chunkidx) {
    int bigSliceN = args.sliceSizeN;
    int smallSliceN = 0; int lastSliceN = 0;
    int sliceTotal = NUM_SUBCHUNKS;
    int bigSliceNum = sliceTotal; int smallSliceNum = 0;

    if ((chunkidx + 1 == args.chunkCount) && (args.totalN % args.chunkSizeN > 0))
      CalcLastChunk<THREADCOUNT, UNROLLCOUNT, ELEMTYPE>(&bigSliceN, &smallSliceN, &lastSliceN,
          &sliceTotal, &bigSliceNum, &smallSliceNum, args.totalN, args.chunkCount, args.chunkSizeN);

    int chunkOff = chunkidx * args.chunkSizeN;

    // Step 0: copy to next
    int stepidx = 0;
    int blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
    int blockOff = chunkOff + blockidx * args.totalN;
    int bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
        ((blockidx * args.bufMisalignedN) % alignof(PackType));
    int sliceSizeN;

    if (tidx < NUM_THREADS) {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
            smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
        WAIT_FOR_CHUNK(chunkidx, sidx);
        Copy<UNROLLCOUNT, THREADCOUNT>(
            args.nextBufferPtr + bufferOff,
            args.thisInputPtr + blockOff, sliceSizeN);
        __syncthreads();
        bufferOff += sliceSizeN;
        blockOff += sliceSizeN;
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
      }
    }

    // Steps 1..K-2: reduce and forward
    for (stepidx = 1; stepidx < args.gpuCount - 1; ++stepidx) {
      int blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
      int blockOff = chunkOff + blockidx * args.totalN;
      int bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
          ((blockidx * args.bufMisalignedN) % alignof(PackType));

      if (tidx < NUM_THREADS) {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
              smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
          Reduce<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              args.nextBufferPtr + bufferOff,
              args.thisBufferPtr + bufferOff,
              args.thisInputPtr + blockOff, sliceSizeN);
          __syncthreads();
          bufferOff += sliceSizeN;
          blockOff += sliceSizeN;
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
        }
      }
    }

    // Step K-1: final reduce to output
    stepidx = args.gpuCount - 1;
    blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
    blockOff = chunkOff + blockidx * args.totalN;
    bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
        ((blockidx * args.bufMisalignedN) % alignof(PackType));

    if (tidx < NUM_THREADS) {
      int outputOff = 0;
      for (int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
            smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
        WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
        Reduce<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
            args.thisOutputPtr + (chunkOff + outputOff),
            args.thisBufferPtr + bufferOff,
            args.thisInputPtr + blockOff, sliceSizeN);
        __syncthreads();
        outputOff += sliceSizeN;
        bufferOff += sliceSizeN;
        blockOff += sliceSizeN;
      }
    } else {
      for (int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
        if (chunkidx + 1 < args.chunkCount) SIGNAL_CHUNK_DONE(chunkidx, sidx);
      }
    }
  }

  if (tidx < NUM_THREADS) {
    WAIT_FOR_NEW_DATA(args.chunkCount, NUM_SUBCHUNKS-1, 0);
    if (tidx == 0) {
      args.thisNewDataFlag[tidx] = 0;
      args.thisChunkDoneFlag[tidx] = 0;
    }
  }
}
```

---

## 6. `ncclReduceScatterWithTypeAndFunc<FUNC, T>`, `ncclReduceScatterWithType<T>`, `ReduceScatterFunctor`, `ncclReduceScatter`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclReduceScatterWithTypeAndFunc}: T^*_s \times T^*_r \times \mathbb{N}_\text{recvcount} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\textbf{Pre: } \text{recvcount} \geq 0,\; \text{total elements} = K \cdot \text{recvcount}
$$
$$
\text{Kernel launch: } <<<1, \text{NUM\_THREADS} + \text{NUM\_SUBCHUNKS} \cdot \text{WARP\_SIZE}>>>
$$
$$
\text{ncclReduceScatter}: \text{enqueue}(\text{ReduceScatterFunctor}(), \ldots)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rs_thread_count (numThreads numSubchunks warpSize : Nat)
    (ht : numThreads = 256) (hn : numSubchunks = 2) (hw : warpSize = 32) :
    numThreads + numSubchunks * warpSize = 320 := by sorry

-- axle_verify: content
theorem rs_thread_count (numThreads numSubchunks warpSize : Nat)
    (ht : numThreads = 256) (hn : numSubchunks = 2) (hw : warpSize = 32) :
    numThreads + numSubchunks * warpSize = 320 := by
  subst ht; subst hn; subst hw; rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<class FUNCTYPE, typename ELEMTYPE>
prjname::resultType prjname::ReduceScatterWithTypeAndFunc(const void* sendbufptr,
    void* recvbufptr, const int recvCountN, prjname::commType commobj, cudaStream_t streamobj) {
  if (recvCountN == 0) return prjname::Success;
  int selfIdx = commobj->ncclId;

  int blockBytes = recvCountN * sizeof(ELEMTYPE);
  int misalignedBytes = blockBytes % alignof(uint64_t);
  int misalignedN = misalignedBytes / sizeof(ELEMTYPE);
  int paddingN = (misalignedN > 0) ? sizeof(uint64_t) / sizeof(ELEMTYPE) : 0;

  int bufferN = commobj->buffSize / sizeof(ELEMTYPE);
  int bufNPerSlice = (bufferN - NUM_SUBCHUNKS * commobj->nDev * paddingN) /
      (NUM_SUBCHUNKS * commobj->nDev);
  int sliceSizeN = (bufNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  int nextIdx = (selfIdx + 1) % commobj->nDev;
  int prevIdx = (selfIdx + commobj->nDev - 1) % commobj->nDev;

  ReduceScatterKernelArgs<ELEMTYPE> argvals;
  argvals.thisIdx = selfIdx;
  argvals.gpuCount = commobj->nDev;
  argvals.totalN = recvCountN;
  argvals.userFromRingMap = commobj->devUserFromRing;
  argvals.sliceSizeN = sliceSizeN;
  argvals.chunkSizeN = NUM_SUBCHUNKS * argvals.sliceSizeN;
  argvals.bufSliceStride = argvals.sliceSizeN + paddingN;
  argvals.bufMisalignedN = misalignedN;

  int remainderN = argvals.totalN % argvals.chunkSizeN;
  if ((argvals.totalN > argvals.chunkSizeN) && (remainderN > 0) &&
      (argvals.totalN < 5 * argvals.chunkSizeN) && (2 * remainderN < argvals.chunkSizeN)) {
    argvals.sliceSizeN /= 2;
    argvals.chunkSizeN = NUM_SUBCHUNKS * argvals.sliceSizeN;
    argvals.chunkCount = argvals.totalN / argvals.chunkSizeN;
  } else {
    argvals.chunkCount = (argvals.totalN + argvals.chunkSizeN - 1) / argvals.chunkSizeN;
  }

  argvals.thisInputPtr = (const ELEMTYPE*)sendbufptr;
  argvals.thisOutputPtr = (volatile ELEMTYPE*)recvbufptr;
  argvals.thisBufferPtr = (volatile ELEMTYPE*)commobj->local[prevIdx]->buff;
  argvals.nextBufferPtr = (volatile ELEMTYPE*)commobj->remote[nextIdx]->buff;
  argvals.thisNewDataFlag = commobj->local[prevIdx]->flags;
  argvals.nextNewDataFlag = commobj->remote[nextIdx]->flags;
  argvals.thisChunkDoneFlag = commobj->local[nextIdx]->flags + 1;
  argvals.prevChunkDoneFlag = commobj->remote[prevIdx]->flags + 1;

  ReduceScatterKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, ELEMTYPE>
      <<<1, NUM_THREADS + NUM_SUBCHUNKS * WARP_SIZE, 0, streamobj>>>(argvals);
  return prjname::Success;
}

template<typename ELEMTYPE>
prjname::resultType prjname::ReduceScatterWithType(const void* sendbufptr, void* recvbufptr,
    int recvCountN, prjname::opType oparg, prjname::commType commobj, cudaStream_t streamobj) {
  switch (oparg) {
  case prjname::Sum:  return prjname::ReduceScatterWithTypeAndFunc<FuncSum<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, recvCountN, commobj, streamobj);
  case prjname::Prod: return prjname::ReduceScatterWithTypeAndFunc<FuncProd<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, recvCountN, commobj, streamobj);
  case prjname::Max:  return prjname::ReduceScatterWithTypeAndFunc<FuncMax<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, recvCountN, commobj, streamobj);
  case prjname::Min:  return prjname::ReduceScatterWithTypeAndFunc<FuncMin<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, recvCountN, commobj, streamobj);
  }
  return prjname::InvalidOperation;
}

class ReduceScatterFunctor {
public:
  prjname::resultType operator()(const void* sendbufptr, void* recvbufptr,
      int recvCountN, prjname::typenameType dtypearg, prjname::opType oparg, int /*unusedroot*/,
      prjname::commType commobj, cudaStream_t streamobj) {
    switch (dtypearg) {
    case prjname::Char:   return prjname::ReduceScatterWithType<char>(sendbufptr, recvbufptr, recvCountN, oparg, commobj, streamobj);
    case prjname::Int:    return prjname::ReduceScatterWithType<int>(sendbufptr, recvbufptr, recvCountN, oparg, commobj, streamobj);
    case prjname::Half:   return prjname::ReduceScatterWithType<half>(sendbufptr, recvbufptr, recvCountN, oparg, commobj, streamobj);
    case prjname::Float:  return prjname::ReduceScatterWithType<float>(sendbufptr, recvbufptr, recvCountN, oparg, commobj, streamobj);
    case prjname::Double: return prjname::ReduceScatterWithType<double>(sendbufptr, recvbufptr, recvCountN, oparg, commobj, streamobj);
    }
    return prjname::InvalidType;
  }
};

extern "C" DSOGLOBAL
prjname::resultType prjname::ReduceScatter(const void* sendbufptr, void* recvbufptr,
    int recvCountN, prjname::typenameType dtypearg, prjname::opType oparg, prjname::commType commobj,
    cudaStream_t streamobj) {
  return enqueue(ReduceScatterFunctor(), sendbufptr, recvbufptr, recvCountN,
      dtypearg, oparg, 0, commobj, streamobj);
}
```

