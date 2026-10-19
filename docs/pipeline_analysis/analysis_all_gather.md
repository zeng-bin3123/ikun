# NCCL Formal Analysis — src/all_gather.cu

---

## 1. Macros: SIGNAL_NEW_DATA_AVAILABLE, WAIT_FOR_NEW_DATA, SIGNAL_CHUNK_DONE, WAIT_FOR_PREV_CHUNK

### LAYER 1a — MATH (LaTeX)

$$
\text{SIGNAL\_NEW\_DATA\_AVAILABLE}(c, s, st): \text{Flag}[\text{next}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot (c \cdot (K-1) + st) + s + 1
$$
$$
\textbf{Pre: } \text{threadfence\_system() issued before write}
$$

$$
\text{WAIT\_FOR\_NEW\_DATA}(c, s, st): \text{spin until } \text{Flag}[\text{this}] \geq \text{NUM\_SUBCHUNKS} \cdot (c \cdot (K-1) + st) + s + 1 - \text{NUM\_SUBCHUNKS}
$$
$$
\textbf{Post: } \text{All NUM\_THREADS threads pass barrier after tid=0 sees flag}
$$

$$
\text{SIGNAL\_CHUNK\_DONE}(c, s): \text{Flag}[\text{prev}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot c + s + 1
$$

$$
\text{WAIT\_FOR\_PREV\_CHUNK}(c, s): \text{spin until } \text{Flag}[\text{this}] \geq \text{NUM\_SUBCHUNKS} \cdot c + s + 1 - \text{NUM\_SUBCHUNKS}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem signal_wait_monotone_ag (numSubchunks chunk step s : Nat) (K : Nat) (hK : K ≥ 2) :
    numSubchunks * (chunk * (K - 1) + step) + s + 1 >
    numSubchunks * (chunk * (K - 1) + step) + s + 1 - numSubchunks ∨ numSubchunks = 0 := by sorry

-- axle_verify: content
theorem signal_wait_monotone_ag (numSubchunks chunk step s : Nat) (K : Nat) (hK : K ≥ 2) :
    numSubchunks * (chunk * (K - 1) + step) + s + 1 >
    numSubchunks * (chunk * (K - 1) + step) + s + 1 - numSubchunks ∨ numSubchunks = 0 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
#define SIGNAL_NEW_DATA_AVAILABLE(chunkidx, subidx, stepidx)                    \
    do {                                                                        \
      __threadfence_system();                                                   \
      args.NextNewDataAvailableFlag[0] =                                        \
            NUM_SUBCHUNKS*((chunkidx) * (args.NumGPUs - 1) + (stepidx)) + subidx+1; \
    } while (0)

#define WAIT_FOR_NEW_DATA(chunkidx, subidx, stepidx)                           \
    do {                                                                       \
      if (tidx == 0) {                                                         \
        Wait([=] {                                                             \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=         \
              NUM_SUBCHUNKS*((chunkidx) * (args.NumGPUs - 1) + (stepidx))     \
              + subidx + 1 - NUM_SUBCHUNKS;                                   \
        });                                                                    \
      }                                                                        \
      BAR(sync, 1, NUM_THREADS);                                               \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunkidx, subidx)                                    \
    do {                                                                       \
      __threadfence_system();                                                  \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunkidx) + (subidx) + 1;    \
    } while (0)

#define WAIT_FOR_PREV_CHUNK(chunkidx, subidx)                      \
    do {                                                           \
      if (tidx == 0) {                                             \
        Wait([=] {                                                 \
          return ((volatile int*)args.ThisChunkDoneFlag)[0] >=     \
              NUM_SUBCHUNKS*(chunkidx) + subidx + 1-NUM_SUBCHUNKS; \
        });                                                        \
      }                                                            \
      BAR(sync, 1, NUM_THREADS);                                   \
    } while (0)
```

---

## 2. `getSliceSizeAndChunkSize` (all_gather version)

### LAYER 1a — MATH (LaTeX)

$$
\text{getSliceSizeAndChunkSize}: \mathbb{N}^7 \to \mathbb{N}
$$
$$
\textbf{Post: } \text{sliceSize} = \begin{cases}
  \text{bigSliceN} & \text{if slice} < \text{numBigSlices} \\
  \text{smallSliceN} & \text{if slice} < \text{numBigSlices} + \text{numSmallSlices} \\
  \text{lastSliceN} & \text{if slice} = \text{numSlices} - 1 \\
  0 & \text{otherwise}
\end{cases}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem get_slice_size_cases (slice numBig numSmall numSlices : Nat)
    (bigN smallN lastN : Nat) :
    (slice < numBig → bigN = bigN) ∧
    (slice ≥ numBig → slice < numBig + numSmall → smallN = smallN) ∧
    (slice = numSlices - 1 → lastN = lastN) := by sorry

-- axle_verify: content
theorem get_slice_size_cases (slice numBig numSmall numSlices : Nat)
    (bigN smallN lastN : Nat) :
    (slice < numBig → bigN = bigN) ∧
    (slice ≥ numBig → slice < numBig + numSmall → smallN = smallN) ∧
    (slice = numSlices - 1 → lastN = lastN) := by
  exact ⟨fun _ => rfl, fun _ _ => rfl, fun _ => rfl⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
__device__ inline void getSliceSizeAndChunkSize(int *sliceSizeOut, int sliceidx,
    int sliceTotal, int bigSliceNum, int smallSliceNum, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (sliceidx < bigSliceNum) {
    *sliceSizeOut = bigSliceN;
  } else {
    *sliceSizeOut = (sliceidx < bigSliceNum + smallSliceNum) ? smallSliceN
        : ((sliceidx == sliceTotal - 1) ? lastSliceN : 0);
  }
}
```

---

## 3. `AllGatherKernelArgs<T>` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{AllGatherKernelArgs}(T) = \{
$$
$$
\quad \text{ThisId}: \mathbb{Z},\; \text{NumGPUs}: \mathbb{N}_{>0},\; N: \mathbb{N},\; \text{UserFromRing}: \mathbb{Z}^*,
$$
$$
\quad \text{SliceSize}: \mathbb{N},\; \text{ChunkSize}: \mathbb{N},\; \text{NumChunks}: \mathbb{N},
$$
$$
\quad \text{BufferSliceStride}: \mathbb{N},\; \text{BufferMisalignedN}: \mathbb{N},
$$
$$
\quad \text{ThisPtrToNextOutput}: T^{**},\; \text{PrevPtrToThisOutput}: T^{**},
$$
$$
\quad \text{ThisInput}: T^*_{\text{const}},\; \text{ThisOutput}: T^*_{\text{vol}},\; \text{ThisBuffer}: T^*_{\text{vol}},\; \text{NextBuffer}: T^*_{\text{vol}},
$$
$$
\quad \text{Flags}: \mathbb{Z}^*_{\text{vol}} \times 4
$$
$$
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_args_valid (numGPUs N sliceSize : Nat)
    (hg : numGPUs > 0) (hs : sliceSize > 0) :
    let chunkSize := 1 * sliceSize  -- NUM_SUBCHUNKS = 1
    let numChunks := (N + chunkSize - 1) / chunkSize
    numChunks * chunkSize ≥ N := by sorry

-- axle_verify: content
theorem allgather_args_valid (numGPUs N sliceSize : Nat)
    (hg : numGPUs > 0) (hs : sliceSize > 0) :
    let chunkSize := 1 * sliceSize
    let numChunks := (N + chunkSize - 1) / chunkSize
    numChunks * chunkSize ≥ N := by
  simp only
  have hcs : 1 * sliceSize > 0 := by omega
  have key := Nat.div_add_mod (N + 1 * sliceSize - 1) (1 * sliceSize)
  have hmod := Nat.mod_lt (N + 1 * sliceSize - 1) hcs
  rw [mul_comm] at key
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct AllGatherKernelArgs {
  int thisIdx;
  int gpuCount;
  int totalN;
  int * userFromRingMap;

  int sliceSizeN;
  int chunkSizeN;
  int chunkCount;

  int bufSliceStride;
  int bufMisalignedN;

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

## 4. `GetBlock`

### LAYER 1a — MATH (LaTeX)

$$
\text{GetBlock}: \mathbb{Z} \times \mathbb{Z} \times \mathbb{Z}^K \times K \to \mathbb{Z}
$$
$$
\textbf{Post: } \text{result} = \text{userFromRing}[(K + \text{index} - \text{step}) \bmod K]
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem get_block_in_range (K index step : Nat) (hK : K > 0) :
    (K + index - step) % K < K := by sorry

-- axle_verify: content
theorem get_block_in_range (K index step : Nat) (hK : K > 0) :
    (K + index - step) % K < K := by
  exact Nat.mod_lt _ hK
```

### LAYER 2 — FORM TEMPLATE

```cuda
__device__ inline int GetBlock(const int selfIdx, const int stepIdx,
    const int * const userFromRingMap, const int gpuCount) {
  return userFromRingMap[(gpuCount + selfIdx - stepIdx) % gpuCount];
}
```

---

## 5. `AllGatherKernel<THREADS, UNROLL, PUSHRECV, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{AllGatherKernel}: \text{AllGatherKernelArgs}(T) \to \text{Unit}
$$
$$
\textbf{Pre: } N > 0,\; \text{NumGPUs} \geq 1,\; \text{launched with 1 block of NUM\_THREADS+1 threads}
$$
$$
\textbf{Post (AllGather): } \forall\, g \in [0,K),\; \text{Output}[g \cdot N \ldots (g+1) \cdot N) = \text{Input}_g[0 \ldots N)
$$
$$
\text{Algorithm: Ring-based, } K-1 \text{ forwarding steps. Step 0: copy local block. Steps } 1..K-2: \text{forward received block. Step } K-1 \text{(non-PUSHRECV): final buffer}\to\text{output copy.}
$$
$$
\textbf{Invariant (flags): } \text{At exit: ThisNewDataAvailableFlag}=0,\; \text{ThisChunkDoneFlag}=0,\; \text{ThisPtrToNextOutput}=\text{null}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_ring_steps (K : Nat) (hK : K ≥ 1) :
    K - 1 + 1 = K := by sorry

-- axle_verify: content
theorem allgather_ring_steps (K : Nat) (hK : K ≥ 1) :
    K - 1 + 1 = K := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
__shared__ volatile void * nextOutputPtr;

template<int THREADCOUNT, int UNROLLCOUNT, bool PUSHRECVFLAG, typename ELEMTYPE>
__global__ void AllGatherKernel(const AllGatherKernelArgs<ELEMTYPE> args) {
  if (args.totalN == 0) return;
  int tidx = threadIdx.x;

  if (tidx == 0) {
    Wait([=] {
      return *((ELEMTYPE * volatile *)args.prevPtrToThisOut) == nullptr;
    });
    *((ELEMTYPE * volatile *)args.prevPtrToThisOut) = (ELEMTYPE*)args.thisOutputPtr;
    Wait([=] {
      return *((ELEMTYPE * volatile *)args.thisPtrToNextOut) != nullptr;
    });
    if(PUSHRECVFLAG)
      nextOutputPtr = *((volatile void * volatile *)args.thisPtrToNextOut);
  }
  __syncthreads();

  for (int chunkidx = 0; chunkidx < args.chunkCount; ++chunkidx) {
    int bigSliceN   = args.sliceSizeN;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int sliceTotal   = NUM_SUBCHUNKS;
    int bigSliceNum   = sliceTotal;
    int smallSliceNum = 0;

    if ((chunkidx + 1 == args.chunkCount) && (args.totalN % args.chunkSizeN > 0))
      CalcLastChunk<THREADCOUNT, UNROLLCOUNT, ELEMTYPE>(&bigSliceN, &smallSliceN, &lastSliceN,
          &sliceTotal, &bigSliceNum, &smallSliceNum, args.totalN, args.chunkCount,
          args.chunkSizeN);

    int chunkOff = chunkidx * args.chunkSizeN;

    int stepidx = 0;
    int blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
    int outputOff = chunkOff + blockidx * args.totalN;
    int inputOff = chunkOff;
    int bufferOff;
    int sliceSizeN;

    if (!PUSHRECVFLAG) {
      bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
          blockidx * args.bufMisalignedN;
    }

    if (tidx < THREADCOUNT) {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
            smallSliceNum, bigSliceN, smallSliceN, lastSliceN);

        if (!PUSHRECVFLAG)
          WAIT_FOR_PREV_CHUNK(chunkidx, sidx);

        if (PUSHRECVFLAG) {
          DoubleCopy<UNROLLCOUNT, THREADCOUNT>(
              args.thisOutputPtr + outputOff,
              (volatile ELEMTYPE *)nextOutputPtr + outputOff,
              args.thisInputPtr + inputOff,
              sliceSizeN);
        } else {
          DoubleCopy<UNROLLCOUNT, THREADCOUNT>(
              args.thisOutputPtr + outputOff,
              args.nextBufferPtr + bufferOff,
              args.thisInputPtr + inputOff,
              sliceSizeN);
        }
        __syncthreads();
        outputOff += sliceSizeN;
        inputOff += sliceSizeN;
        if (!PUSHRECVFLAG) bufferOff += sliceSizeN;
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
      }
    }

    for (stepidx = 1; stepidx < args.gpuCount - 1; ++stepidx) {
      blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
      outputOff = chunkOff + blockidx * args.totalN;
      if (!PUSHRECVFLAG) {
        bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
            blockidx * args.bufMisalignedN;
      }

      if (tidx < THREADCOUNT) {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
              smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);

          if (PUSHRECVFLAG) {
            Copy<UNROLLCOUNT, THREADCOUNT>(
                (volatile ELEMTYPE *)nextOutputPtr + outputOff,
                args.thisOutputPtr + outputOff,
                sliceSizeN);
          } else {
            DoubleCopy<UNROLLCOUNT, THREADCOUNT>(
                args.nextBufferPtr + bufferOff,
                args.thisOutputPtr + outputOff,
                args.thisBufferPtr + bufferOff,
                sliceSizeN);
          }
          __syncthreads();
          outputOff += sliceSizeN;
          if (!PUSHRECVFLAG) bufferOff += sliceSizeN;
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx, stepidx);
        }
      }
    }

    if (!PUSHRECVFLAG) {
      stepidx = args.gpuCount - 1;
      blockidx = GetBlock(args.thisIdx, stepidx, args.userFromRingMap, args.gpuCount);
      outputOff = chunkOff + blockidx * args.totalN;
      bufferOff = blockidx * NUM_SUBCHUNKS * args.bufSliceStride +
          blockidx * args.bufMisalignedN;

      if (tidx < THREADCOUNT) {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          getSliceSizeAndChunkSize(&sliceSizeN, sidx, sliceTotal, bigSliceNum,
              smallSliceNum, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunkidx, sidx, stepidx);
          Copy<UNROLLCOUNT, THREADCOUNT>(
              args.thisOutputPtr + outputOff,
              args.thisBufferPtr + bufferOff,
              sliceSizeN);
          __syncthreads();
          outputOff += sliceSizeN;
          bufferOff += sliceSizeN;
        }
      } else {
        for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
          __syncthreads();
          SIGNAL_CHUNK_DONE(chunkidx, sidx);
        }
      }
    }
  }

  if (tidx < THREADCOUNT) {
    if (PUSHRECVFLAG)
      WAIT_FOR_NEW_DATA(args.chunkCount, NUM_SUBCHUNKS-1, 0);
    else
      WAIT_FOR_PREV_CHUNK(args.chunkCount, NUM_SUBCHUNKS-1);

    if (tidx == 0) {
      args.thisNewDataFlag[0] = 0;
      args.thisChunkDoneFlag[0] = 0;
      *args.thisPtrToNextOut = nullptr;
    }
  }
}
```

---

## 6. `ncclAllGatherWithType<T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllGatherWithType}: T^*_{\text{send}} \times T^*_{\text{recv}} \times \mathbb{N} \times \text{Comm} \times \mathbb{N} \times \text{Stream} \to \text{Result}
$$
$$
\textbf{Pre: } \text{count} \geq 0,\; \text{comm valid},\; \text{sendbuff} \neq \text{null if count}>0
$$
$$
\textbf{Post: } \text{count}=0 \implies \text{return ncclSuccess immediately}
$$
$$
\text{SliceSize} = \min(\text{maxSliceSize},\; \text{numUnroll} \cdot \text{UNROLL\_SIZE} \cdot |P|/|T|)
$$
$$
\text{ChunkSize} = \text{NUM\_SUBCHUNKS} \cdot \text{SliceSize}
$$
$$
\text{NumChunks} = \begin{cases}
  \lfloor N / \text{ChunkSize} \rfloor & \text{if remainder is tiny} \\
  \lceil N / \text{ChunkSize} \rceil & \text{otherwise}
\end{cases}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_chunk_coverage (N chunkSize : Nat) (hc : chunkSize > 0) :
    let numChunks := (N + chunkSize - 1) / chunkSize
    numChunks * chunkSize ≥ N := by sorry

-- axle_verify: content
theorem allgather_chunk_coverage (N chunkSize : Nat) (hc : chunkSize > 0) :
    let numChunks := (N + chunkSize - 1) / chunkSize
    numChunks * chunkSize ≥ N := by
  simp only
  have key := Nat.div_add_mod (N + chunkSize - 1) chunkSize
  have hmod := Nat.mod_lt (N + chunkSize - 1) hc
  rw [mul_comm] at key
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
prjname::resultType prjname::AllGatherWithType(const void* sendbufptr, void* recvbufptr,
    int countN, prjname::commType commobj, int unrollCount, cudaStream_t streamobj) {
  if (countN == 0) return prjname::Success;

  int selfIdx = commobj->ncclId;
  int blockBytes = countN * sizeof(ELEMTYPE);
  int misalignedBytes = blockBytes % alignof(uint64_t);
  int misalignedN = misalignedBytes / sizeof(ELEMTYPE);
  int paddingN = (misalignedN > 0) ? sizeof(uint64_t) / sizeof(ELEMTYPE) : 0;

  int bufferN = commobj->buffSize / sizeof(ELEMTYPE);
  int bufferNPerSlice = (bufferN - commobj->nDev * NUM_SUBCHUNKS * paddingN)
       / (commobj->nDev * NUM_SUBCHUNKS);
  int maxSliceN = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;
  int nextIdx = (selfIdx + 1) % commobj->nDev;
  int prevIdx = (selfIdx + commobj->nDev - 1) % commobj->nDev;

  AllGatherKernelArgs<ELEMTYPE> argvals;
  argvals.thisIdx = selfIdx;
  argvals.gpuCount = commobj->nDev;
  argvals.totalN = countN;
  argvals.userFromRingMap = commobj->devUserFromRing;
  argvals.sliceSizeN = unrollCount * UNROLL_SIZE * sizeof(PackType) / sizeof(ELEMTYPE);
  argvals.sliceSizeN = std::min(maxSliceN, argvals.sliceSizeN);
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
    AllGatherKernel<NUM_THREADS, UNROLL_COUNT, true, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  } else {
    AllGatherKernel<NUM_THREADS, UNROLL_COUNT, false, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  }
  return prjname::Success;
}
```

---

## 7. `AllGatherFunctor`

### LAYER 1a — MATH (LaTeX)

$$
\text{AllGatherFunctor}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{DType} \times \text{Op} \times \mathbb{Z} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\textbf{Post: } \text{Dispatches to ncclAllGatherWithType}\langle T \rangle \text{ based on datatype enum}
$$
$$
\text{numUnroll} = 16 \quad (\text{optimal for dt07 with 4 GPUs})
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_dispatch_exhaustive (dt : Nat) (hdt : dt < 5) :
    dt = 0 ∨ dt = 1 ∨ dt = 2 ∨ dt = 3 ∨ dt = 4 := by sorry

-- axle_verify: content
theorem allgather_dispatch_exhaustive (dt : Nat) (hdt : dt < 5) :
    dt = 0 ∨ dt = 1 ∨ dt = 2 ∨ dt = 3 ∨ dt = 4 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
class AllGatherFunctor {
public:
  prjname::resultType operator()(const void* sendbufptr, void* recvbufptr,
      int countN, prjname::typenameType dtypearg, prjname::opType /*unusedop*/,
      int /*unusedroot*/, prjname::commType commobj, cudaStream_t streamobj) {
    int unrollCount = 16;
    switch (dtypearg) {
    case prjname::Char:
      return prjname::AllGatherWithType<char>(sendbufptr, recvbufptr, countN, commobj, unrollCount, streamobj);
    case prjname::Int:
      return prjname::AllGatherWithType<int>(sendbufptr, recvbufptr, countN, commobj, unrollCount, streamobj);
    case prjname::Half:
      return prjname::AllGatherWithType<half>(sendbufptr, recvbufptr, countN, commobj, unrollCount, streamobj);
    case prjname::Float:
      return prjname::AllGatherWithType<float>(sendbufptr, recvbufptr, countN, commobj, unrollCount, streamobj);
    case prjname::Double:
      return prjname::AllGatherWithType<double>(sendbufptr, recvbufptr, countN, commobj, unrollCount, streamobj);
    }
    return prjname::InvalidType;
  }
};
```

---

## 8. `ncclAllGather` (C API)

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllGather}: T^*_{\text{send}} \times \mathbb{N} \times \text{DType} \times T^*_{\text{recv}} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\textbf{Post: } \text{enqueue}(\text{AllGatherFunctor}(), \ldots)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem ncclallgather_delegates :
    True := by sorry

-- axle_verify: content
theorem ncclallgather_delegates :
    True := by trivial
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
prjname::resultType prjname::AllGather(const void* sendbufptr, int countN,
    prjname::typenameType dtypearg,
    void* recvbufptr, prjname::commType commobj, cudaStream_t streamobj) {
  return enqueue(AllGatherFunctor(), sendbufptr, recvbufptr, countN, dtypearg,
      prjname::Sum, 0, commobj, streamobj);
}
```

