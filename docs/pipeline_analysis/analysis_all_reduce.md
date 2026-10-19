# NCCL Formal Analysis — src/all_reduce.cu

---

## 1. Macros: SIGNAL_NEW_DATA_AVAILABLE, WAIT_FOR_NEW_DATA, SIGNAL_CHUNK_DONE, WAIT_FOR_CHUNK (AllReduce variants)

### LAYER 1a — MATH (LaTeX)

$$
\text{SIGNAL\_NEW\_DATA\_AVAILABLE}_{AR}(c, s, st): \text{Flag}[\text{next}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot (c \cdot (2K-2) + st + 1) + s
$$
$$
\text{WAIT\_FOR\_NEW\_DATA}_{AR}(c, s, st): \text{spin until } \text{Flag}[\text{this}] \geq 2 \cdot (c \cdot (2K-2) + st) + s
$$
$$
\text{SIGNAL\_CHUNK\_DONE}_{AR}(c, s): \text{Flag}[\text{prev}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot c + s + 1
$$
$$
\text{WAIT\_FOR\_CHUNK}_{AR}(c, s): \text{spin until } \text{Flag}[\text{this}] \geq \text{NUM\_SUBCHUNKS} \cdot c + s + 1 - \text{NUM\_SUBCHUNKS}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_total_steps (K : Nat) (hK : K ≥ 2) :
    2 * K - 2 = 2 * (K - 1) := by sorry

-- axle_verify: content
theorem allreduce_total_steps (K : Nat) (hK : K ≥ 2) :
    2 * K - 2 = 2 * (K - 1) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
#define SIGNAL_NEW_DATA_AVAILABLE(chunkidx, subidx, stepidx)                  \
    do {                                                                      \
      __threadfence_system();                                                 \
      args.NextNewDataAvailableFlag[0] =                                      \
          NUM_SUBCHUNKS*((chunkidx) * (2 * args.NumGPUs - 2) + (stepidx) + 1)+subidx; \
    } while (0)

#define WAIT_FOR_NEW_DATA(chunkidx, subidx, stepidx)                            \
    do {                                                                        \
      if (tidx == 0) {                                                          \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              2*((chunkidx) * (2 * args.NumGPUs - 2) + (stepidx))+subidx;      \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunkidx, subidx)                                     \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunkidx) + subidx + 1;        \
    } while (0)

#define WAIT_FOR_CHUNK(chunkidx, subidx)                                        \
    do {                                                                        \
      if (tidx == 0) {                                                          \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisChunkDoneFlag)[0] >=                 \
              NUM_SUBCHUNKS*(chunkidx) + subidx + 1 - NUM_SUBCHUNKS;            \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)
```

---

## 2. `getSliceSizeAndOffset` (AllReduce version)

### LAYER 1a — MATH (LaTeX)

$$
\text{getSliceSizeAndOffset}: \mathbb{N}^7 \to \mathbb{N} \times \mathbb{N}
$$
$$
\textbf{Post: } (\text{size}, \text{offset}) = \begin{cases}
  (\text{bigSliceN},\; \text{slice} \cdot \text{bigSliceN}) & \text{slice} < \text{numBigSlices} \\
  (\text{smallSliceN},\; B \cdot \text{bigSliceN} + (\text{slice}-B) \cdot \text{smallSliceN}) & B \leq \text{slice} < B+S \\
  (\text{lastSliceN},\; B \cdot \text{bigSliceN} + (\text{slice}-B) \cdot \text{smallSliceN}) & \text{slice}=\text{numSlices}-1 \\
  (0,\; \ldots) & \text{otherwise}
\end{cases}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem slice_offset_nonneg (slice numBig bigN : Nat) (h : slice < numBig) :
    slice * bigN ≥ 0 := by sorry

-- axle_verify: content
theorem slice_offset_nonneg (slice numBig bigN : Nat) (h : slice < numBig) :
    slice * bigN ≥ 0 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
__device__ inline void getSliceSizeAndOffset(int *sizeOut, int *offsetOut, int sliceidx,
    int sliceTotal, int bigSliceNum, int smallSliceNum, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (sliceidx < bigSliceNum) {
    *sizeOut = bigSliceN;
    *offsetOut = sliceidx * bigSliceN;
  } else {
    *sizeOut = (sliceidx < bigSliceNum + smallSliceNum) ? smallSliceN
        : ((sliceidx == sliceTotal - 1) ? lastSliceN : 0);
    *offsetOut = bigSliceNum * bigSliceN + (sliceidx - bigSliceNum) * smallSliceN;
  }
}
```

---

## 3. `AllReduceKernelArgs<T>` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{AllReduceKernelArgs}(T) = \{
  \text{ThisId}: \mathbb{Z},\; \text{NumGPUs}: K,\; N: \mathbb{N},
$$
$$
  \text{SliceSize}: \mathbb{N},\; \text{ChunkSize}: \mathbb{N},\; \text{NumChunks}: \mathbb{N},
$$
$$
  \text{ThisPtrToNextOutput}: T^{**},\; \text{PrevPtrToThisOutput}: T^{**},
$$
$$
  \text{ThisInput}: T^*,\; \text{ThisOutput}: T^*_\text{vol},\; \text{ThisBuffer}: T^*_\text{vol},\; \text{NextBuffer}: T^*_\text{vol},
$$
$$
  \text{Flags}: \mathbb{Z}^*_\text{vol} \times 4
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_chunk_covers (N chunkSize numSubchunks numGPUs sliceSize : Nat)
    (hs : sliceSize > 0) (hg : numGPUs > 0) (hn : numSubchunks > 0)
    (hc : chunkSize = numSubchunks * numGPUs * sliceSize) (hcs : chunkSize > 0) :
    (N + chunkSize - 1) / chunkSize * chunkSize ≥ N := by sorry

-- axle_verify: content
theorem allreduce_chunk_covers (N chunkSize numSubchunks numGPUs sliceSize : Nat)
    (hs : sliceSize > 0) (hg : numGPUs > 0) (hn : numSubchunks > 0)
    (hc : chunkSize = numSubchunks * numGPUs * sliceSize) (hcs : chunkSize > 0) :
    (N + chunkSize - 1) / chunkSize * chunkSize ≥ N := by
  have key := Nat.div_add_mod (N + chunkSize - 1) chunkSize
  have hmod := Nat.mod_lt (N + chunkSize - 1) hcs
  rw [mul_comm] at key
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct AllReduceKernelArgs {
  int thisIdx;
  int gpuCount;
  int totalN;

  int sliceSizeN;
  int chunkSizeN;
  int chunkCount;

  ELEMTYPE ** thisPtrToNextOut;
  ELEMTYPE ** prevPtrToThisOut;

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

## 4. `AllReduceKernel<THREADS, UNROLL, FUNC, PUSHRECV, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{AllReduceKernel}: \text{AllReduceKernelArgs}(T) \to \text{Unit}
$$
$$
\textbf{Algorithm: } \text{Ring reduce-scatter followed by ring all-gather, total } 2(K-1) \text{ steps.}
$$
$$
\text{Steps } 0..K-2: \text{reduce-scatter phase (reduce local data into ring)}
$$
$$
\text{Step } K-1: \text{final reduce, write to output and push to next}
$$
$$
\text{Steps } K..2K-3: \text{all-gather phase (propagate final result around ring)}
$$
$$
\textbf{Post: } \forall\, i \in [0,N),\; \text{Output}[i] = \bigoplus_{g=0}^{K-1} \text{Input}_g[i]
$$
$$
\textbf{Invariant: } \text{Flags reset to 0 and output ptr set to null at exit}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_phase_count (K : Nat) (hK : K ≥ 2) :
    (K - 1) + 1 + (K - 2) = 2 * K - 2 := by sorry

-- axle_verify: content
theorem allreduce_phase_count (K : Nat) (hK : K ≥ 2) :
    (K - 1) + 1 + (K - 2) = 2 * K - 2 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
__shared__ volatile void * nextOutputPtr;

template<int THREADCOUNT, int UNROLLCOUNT, class FUNCTYPE, bool PUSHRECVFLAG, typename ELEMTYPE>
__launch_bounds__(THREADCOUNT+WARP_SIZE, 1)
__global__ void AllReduceKernel(const AllReduceKernelArgs<ELEMTYPE> args) {
  if (args.totalN == 0) return;
  const int tidx = threadIdx.x;

  if (tidx == 0) {
    Wait([=] { return *((ELEMTYPE * volatile *)args.prevPtrToThisOut) == nullptr; });
    *((ELEMTYPE * volatile *)args.prevPtrToThisOut) = (ELEMTYPE*)args.thisOutputPtr;
    Wait([=] { return *((ELEMTYPE * volatile *)args.thisPtrToNextOut) != nullptr; });
    if (PUSHRECVFLAG)
      nextOutputPtr = *((volatile void * volatile *)args.thisPtrToNextOut);
  }
  __syncthreads();

  for (int chunkidx = 0; chunkidx < args.chunkCount; ++chunkidx) {
    int bigSliceN = args.sliceSizeN;
    int smallSliceN = 0; int lastSliceN = 0;
    int sliceTotal = args.gpuCount * NUM_SUBCHUNKS;
    int bigSliceNum = sliceTotal; int smallSliceNum = 0;

    if ((chunkidx + 1 == args.chunkCount) && (args.totalN % args.chunkSizeN > 0))
      CalcLastChunk<THREADCOUNT, UNROLLCOUNT, ELEMTYPE>(&bigSliceN, &smallSliceN, &lastSliceN,
          &sliceTotal, &bigSliceNum, &smallSliceNum, args.totalN, args.chunkCount, args.chunkSizeN);

    int chunkOff = chunkidx * args.chunkSizeN;

    // Step 0: push to next buffer
    int stepidx = 0;
    int sliceidx = args.thisIdx;
    int offsetN; int sliceSizeN;

    if (tidx < THREADCOUNT) {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        if (sidx > 0) { sliceidx += args.gpuCount; }
        getSliceSizeAndOffset(&sliceSizeN, &offsetN, sliceidx, sliceTotal,
            bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
        if (!PUSHRECVFLAG && chunkidx > 0) { WAIT_FOR_CHUNK(chunkidx, sidx); }
        Copy<UNROLLCOUNT, THREADCOUNT>(
            args.nextBufferPtr + offsetN, args.thisInputPtr + chunkOff + offsetN, sliceSizeN);
        __syncthreads();
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
      }
    }

    // Steps 1..K-2: reduce and forward
    for (stepidx = 1; stepidx < args.gpuCount - 1; ++stepidx) {
      if (tidx < THREADCOUNT) {
        sliceidx = (args.gpuCount + sliceidx - 1) % args.gpuCount;
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          if (sidx > 0) { sliceidx += args.gpuCount; }
          getSliceSizeAndOffset(&sliceSizeN, &offsetN, sliceidx, sliceTotal,
              bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
          Reduce<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              args.nextBufferPtr + offsetN, args.thisBufferPtr + offsetN,
              args.thisInputPtr + chunkOff + offsetN, sliceSizeN);
          __syncthreads();
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
        }
      }
    }

    // Step K-1: final reduce, write output
    stepidx = args.gpuCount - 1;
    if (tidx < THREADCOUNT) {
      sliceidx = (args.gpuCount + sliceidx - 1) % args.gpuCount;
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        if (sidx > 0) { sliceidx += args.gpuCount; }
        getSliceSizeAndOffset(&sliceSizeN, &offsetN, sliceidx, sliceTotal,
            bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
        WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
        if (PUSHRECVFLAG) {
          ReduceAndCopy<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              (volatile ELEMTYPE *)nextOutputPtr + chunkOff + offsetN,
              args.thisOutputPtr + chunkOff + offsetN,
              args.thisBufferPtr + offsetN,
              args.thisInputPtr + chunkOff + offsetN, sliceSizeN);
        } else {
          ReduceAndCopy<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              args.nextBufferPtr + offsetN,
              args.thisOutputPtr + chunkOff + offsetN,
              args.thisBufferPtr + offsetN,
              args.thisInputPtr + chunkOff + offsetN, sliceSizeN);
        }
        __syncthreads();
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
      }
    }

    // Steps K..2K-3: allgather phase
    for (stepidx = args.gpuCount; stepidx < 2 * args.gpuCount - 2; ++stepidx) {
      if (tidx < THREADCOUNT) {
        sliceidx = (args.gpuCount + sliceidx - 1) % args.gpuCount;
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          if (sidx > 0) { sliceidx += args.gpuCount; }
          getSliceSizeAndOffset(&sliceSizeN, &offsetN, sliceidx, sliceTotal,
              bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
          if (PUSHRECVFLAG) {
            Copy<UNROLLCOUNT, THREADCOUNT>(
                (volatile ELEMTYPE *)nextOutputPtr + chunkOff + offsetN,
                args.thisOutputPtr + chunkOff + offsetN, sliceSizeN);
          } else {
            DoubleCopy<UNROLLCOUNT, THREADCOUNT>(
                args.nextBufferPtr + offsetN,
                args.thisOutputPtr + chunkOff + offsetN,
                args.thisBufferPtr + offsetN, sliceSizeN);
          }
          __syncthreads();
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
        }
      }
    }

    if (!PUSHRECVFLAG) {
      if (tidx < THREADCOUNT) {
        sliceidx = (args.gpuCount + sliceidx - 1) % args.gpuCount;
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          if (sidx > 0) { sliceidx += args.gpuCount; }
          getSliceSizeAndOffset(&sliceSizeN, &offsetN, sliceidx, sliceTotal,
              bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
          Copy<UNROLLCOUNT, THREADCOUNT>(
              args.thisOutputPtr + chunkOff + offsetN,
              args.thisBufferPtr + offsetN, sliceSizeN);
          __syncthreads();
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          if(chunkidx+1 < args.chunkCount) { SIGNAL_CHUNK_DONE(chunkidx, sidx); }
        }
      }
    }
  }

  if (tidx < THREADCOUNT) {
    if(PUSHRECVFLAG) { WAIT_FOR_NEW_DATA(args.chunkCount, NUM_SUBCHUNKS-1, 0); }
    if (tidx == 0) {
      args.thisNewDataFlag[0] = 0;
      if(!PUSHRECVFLAG) { args.thisChunkDoneFlag[0] = 0; }
      *args.thisPtrToNextOut = nullptr;
    }
  }
}
```

---

## 5. `ncclAllReduceWithTypeAndFunc<FUNC, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllReduceWithTypeAndFunc}: T^*_s \times T^*_r \times \mathbb{N} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{sliceSize} = \lfloor \text{bufferNPerSlice} / \text{UNROLL\_SIZE} \rfloor \cdot \text{UNROLL\_SIZE}
$$
$$
\text{ChunkSize} = \text{NUM\_SUBCHUNKS} \cdot K \cdot \text{sliceSize}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_slice_aligned (bufNPerSlice unrollSize : Nat) (hu : unrollSize > 0) :
    (bufNPerSlice / unrollSize) * unrollSize ≤ bufNPerSlice := by sorry

-- axle_verify: content
theorem allreduce_slice_aligned (bufNPerSlice unrollSize : Nat) (hu : unrollSize > 0) :
    (bufNPerSlice / unrollSize) * unrollSize ≤ bufNPerSlice := by
  exact Nat.div_mul_le_self bufNPerSlice unrollSize
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<class FUNCTYPE, typename ELEMTYPE>
prjname::resultType prjname::AllReduceWithTypeAndFunc(const void* sendbufptr, void* recvbufptr,
    const int countN, prjname::commType commobj, cudaStream_t streamobj) {
  if (countN == 0) return prjname::Success;
  int selfIdx = commobj->ncclId;

  int bufferN = commobj->buffSize / sizeof(ELEMTYPE);
  int bufNPerSlice = bufferN / (NUM_SUBCHUNKS * commobj->nDev);
  int sliceSizeN = (bufNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  int nextIdx = (selfIdx + 1) % commobj->nDev;
  int prevIdx = (selfIdx + commobj->nDev - 1) % commobj->nDev;

  AllReduceKernelArgs<ELEMTYPE> argvals;
  argvals.thisIdx = selfIdx;
  argvals.gpuCount = commobj->nDev;
  argvals.totalN = countN;
  argvals.sliceSizeN = sliceSizeN;
  int subchunkN = commobj->nDev * argvals.sliceSizeN;
  argvals.chunkSizeN = NUM_SUBCHUNKS * subchunkN;

  int remainderN = argvals.totalN % argvals.chunkSizeN;
  if ((argvals.totalN > argvals.chunkSizeN) && (remainderN > 0) &&
      (argvals.totalN < 5 * argvals.chunkSizeN) && (2 * remainderN < argvals.chunkSizeN)) {
    argvals.sliceSizeN /= 2;
    subchunkN = commobj->nDev * argvals.sliceSizeN;
    argvals.chunkSizeN = NUM_SUBCHUNKS * subchunkN;
    argvals.chunkCount = argvals.totalN / argvals.chunkSizeN;
  } else {
    argvals.chunkCount = (argvals.totalN + argvals.chunkSizeN - 1) / argvals.chunkSizeN;
  }

  argvals.thisPtrToNextOut = (ELEMTYPE**)&(commobj->local[nextIdx]->recvPtrs[0]);
  argvals.prevPtrToThisOut = (ELEMTYPE**)&(commobj->remote[prevIdx]->recvPtrs[0]);
  argvals.thisInputPtr = (const ELEMTYPE*)sendbufptr;
  argvals.thisOutputPtr = (volatile ELEMTYPE*)recvbufptr;
  argvals.thisBufferPtr = (volatile ELEMTYPE*)commobj->local[prevIdx]->buff;
  argvals.nextBufferPtr = (volatile ELEMTYPE*)commobj->remote[nextIdx]->buff;
  argvals.thisNewDataFlag = commobj->local[prevIdx]->flags;
  argvals.nextNewDataFlag = commobj->remote[nextIdx]->flags;
  argvals.thisChunkDoneFlag = commobj->local[nextIdx]->flags + 1;
  argvals.prevChunkDoneFlag = commobj->remote[prevIdx]->flags + 1;

  if( commobj->useRemoteRecv ) {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, true, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  } else {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, false, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  }
  return prjname::Success;
}
```

---

## 6. `ncclAllReduceWithType<T>`, `AllReduceFunctor`, `ncclAllReduce`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllReduceWithType}: \text{dispatches by op} \in \{\text{Sum}, \text{Prod}, \text{Max}, \text{Min}\}
$$
$$
\text{AllReduceFunctor}: \text{dispatches by datatype} \in \{\text{char}, \text{int}, \text{half}, \text{float}, \text{double}\}
$$
$$
\text{ncclAllReduce}: \text{enqueue}(\text{AllReduceFunctor}(), \ldots)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_op_dispatch (op : Nat) (hop : op < 4) :
    op = 0 ∨ op = 1 ∨ op = 2 ∨ op = 3 := by sorry

-- axle_verify: content
theorem allreduce_op_dispatch (op : Nat) (hop : op < 4) :
    op = 0 ∨ op = 1 ∨ op = 2 ∨ op = 3 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
prjname::resultType prjname::AllReduceWithType(const void* sendbufptr,
    void* recvbufptr, int countN, prjname::opType oparg, prjname::commType commobj,
    cudaStream_t streamobj) {
  switch (oparg) {
  case prjname::Sum:
    return prjname::AllReduceWithTypeAndFunc<FuncSum<ELEMTYPE>, ELEMTYPE>(
        sendbufptr, recvbufptr, countN, commobj, streamobj);
  case prjname::Prod:
    return prjname::AllReduceWithTypeAndFunc<FuncProd<ELEMTYPE>, ELEMTYPE>(
        sendbufptr, recvbufptr, countN, commobj, streamobj);
  case prjname::Max:
    return prjname::AllReduceWithTypeAndFunc<FuncMax<ELEMTYPE>, ELEMTYPE>(
        sendbufptr, recvbufptr, countN, commobj, streamobj);
  case prjname::Min:
    return prjname::AllReduceWithTypeAndFunc<FuncMin<ELEMTYPE>, ELEMTYPE>(
        sendbufptr, recvbufptr, countN, commobj, streamobj);
  }
  return prjname::InvalidOperation;
}

class AllReduceFunctor {
public:
  prjname::resultType operator()(const void* sendbufptr, void* recvbufptr,
      int countN, prjname::typenameType dtypearg, prjname::opType oparg, int /*unusedroot*/,
      prjname::commType commobj, cudaStream_t streamobj) {
    switch (dtypearg) {
    case prjname::Char:
      return prjname::AllReduceWithType<char>(sendbufptr, recvbufptr, countN, oparg, commobj, streamobj);
    case prjname::Int:
      return prjname::AllReduceWithType<int>(sendbufptr, recvbufptr, countN, oparg, commobj, streamobj);
    case prjname::Half:
      return prjname::AllReduceWithType<half>(sendbufptr, recvbufptr, countN, oparg, commobj, streamobj);
    case prjname::Float:
      return prjname::AllReduceWithType<float>(sendbufptr, recvbufptr, countN, oparg, commobj, streamobj);
    case prjname::Double:
      return prjname::AllReduceWithType<double>(sendbufptr, recvbufptr, countN, oparg, commobj, streamobj);
    }
    return prjname::InvalidType;
  }
};

extern "C" DSOGLOBAL
prjname::resultType prjname::AllReduce(const void* sendbufptr, void* recvbufptr, int countN,
    prjname::typenameType dtypearg, prjname::opType oparg, prjname::commType commobj,
    cudaStream_t streamobj) {
  return enqueue(AllReduceFunctor(), sendbufptr, recvbufptr, countN, dtypearg, oparg, 0,
      commobj, streamobj);
}
```

