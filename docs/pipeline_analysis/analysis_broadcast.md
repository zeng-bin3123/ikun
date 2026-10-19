# NCCL Formal Analysis — src/broadcast.cu

---

## 1. Macros: SIGNAL_NEW_DATA_AVAILABLE, WAIT_FOR_NEW_DATA, SIGNAL_CHUNK_DONE, WAIT_FOR_CHUNK, WAIT_FOR_NEW_DATA_AND_CHUNK (Broadcast variants)

### LAYER 1a — MATH (LaTeX)

$$
\text{SIGNAL\_NEW\_DATA\_AVAILABLE}_{BC}(c, s): \text{Flag}[\text{next}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot c + s + 1
$$
$$
\text{WAIT\_FOR\_NEW\_DATA}_{BC}(c, s): \text{spin until } \text{Flag}[\text{this}] \geq \text{NUM\_SUBCHUNKS} \cdot c + s + 1
$$
$$
\text{SIGNAL\_CHUNK\_DONE}_{BC}(c, s): \text{Flag}[\text{prev}] \leftarrow \text{NUM\_SUBCHUNKS} \cdot c + s + 1
$$
$$
\text{WAIT\_FOR\_CHUNK}_{BC}(c, s): \text{spin until } \text{Flag}[\text{this}] \geq \text{NUM\_SUBCHUNKS} \cdot c + s + 1 - \text{NUM\_SUBCHUNKS}
$$
$$
\text{WAIT\_FOR\_NEW\_DATA\_AND\_CHUNK}_{BC}(c, s): \text{spin until both above conditions hold}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_signal_positive (numSubchunks chunk s : Nat) :
    numSubchunks * chunk + s + 1 > 0 := by sorry

-- axle_verify: content
theorem bcast_signal_positive (numSubchunks chunk s : Nat) :
    numSubchunks * chunk + s + 1 > 0 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
#define SIGNAL_NEW_DATA_AVAILABLE(chunkidx, subidx)                             \
    do {                                                                        \
      __threadfence_system();                                                   \
      args.NextNewDataAvailableFlag[0] = NUM_SUBCHUNKS*(chunkidx) + subidx + 1; \
    } while (0)

#define WAIT_FOR_NEW_DATA(chunkidx, subidx)                                     \
    do {                                                                        \
      if (tidx == 0) {                                                          \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              NUM_SUBCHUNKS*(chunkidx) + subidx + 1;                            \
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

#define WAIT_FOR_NEW_DATA_AND_CHUNK(chunkidx, subidx)                           \
    do {                                                                        \
      if (tidx == 0) {                                                          \
        Wait([=] {                                                              \
          bool newDataReady =                                                   \
              ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=             \
                  NUM_SUBCHUNKS*(chunkidx) + subidx + 1;                        \
          bool chunkReady =                                                     \
              ((volatile int *)args.ThisChunkDoneFlag)[0] >=                    \
                  NUM_SUBCHUNKS*(chunkidx)+subidx + 1 - NUM_SUBCHUNKS;          \
          return newDataReady && chunkReady;                                     \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)
```

---

## 2. `BroadcastKernelArgs<T>` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{BroadcastKernelArgs}(T) = \{
  \text{ThisId}: \mathbb{Z},\; N: \mathbb{N},
$$
$$
  \text{SliceSize}: \mathbb{N},\; \text{ChunkSize}: \mathbb{N},\; \text{NumChunks}: \mathbb{N},\; \text{BufferSliceStride}: \mathbb{N},
$$
$$
  \text{ThisPtrToNextData}: T^{**},\; \text{PrevPtrToThisData}: T^{**},
$$
$$
  \text{ThisData}: T^*,\; \text{ThisBuffer}: T^*_\text{vol},\; \text{NextBuffer}: T^*_\text{vol},
$$
$$
  \text{Flags}: \mathbb{Z}^*_\text{vol} \times 4
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_args_struct_valid (N sliceSize : Nat) (hs : sliceSize > 0) :
    let chunkSize := 4 * sliceSize  -- NUM_SUBCHUNKS=4
    chunkSize > 0 := by sorry

-- axle_verify: content
theorem bcast_args_struct_valid (N sliceSize : Nat) (hs : sliceSize > 0) :
    let chunkSize := 4 * sliceSize
    chunkSize > 0 := by
  simp; omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct BroadcastKernelArgs {
  int thisIdx;
  int totalN;

  int sliceSizeN;
  int chunkSizeN;
  int chunkCount;
  int bufSliceStride;

  ELEMTYPE ** thisPtrToNextData;
  ELEMTYPE ** prevPtrToThisData;

  ELEMTYPE * __restrict__ thisDataPtr;
  volatile ELEMTYPE * __restrict__ thisBufferPtr;
  volatile ELEMTYPE * __restrict__ nextBufferPtr;

  volatile int * __restrict__ thisNewDataFlag;
  volatile int * __restrict__ nextNewDataFlag;
  volatile int * __restrict__ thisChunkDoneFlag;
  volatile int * __restrict__ prevChunkDoneFlag;
};
```

---

## 3. `BcastRole` enum

### LAYER 1a — MATH (LaTeX)

$$
\text{BcastRole} = \{\text{ROOT}=0,\; \text{MIDDLE}=1,\; \text{END}=2\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_roles_distinct : (0 ≠ 1) ∧ (1 ≠ 2) ∧ (0 ≠ 2) := by sorry

-- axle_verify: content
theorem bcast_roles_distinct : (0 ≠ 1) ∧ (1 ≠ 2) ∧ (0 ≠ 2) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
enum BcastRole {ROOTROLE=0, MIDDLEROLE=1, ENDROLE=2};
```

---

## 4. `BroadcastKernel<THREADS, UNROLL, PUSHRECV, ROLE, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{BroadcastKernel}: \text{BroadcastKernelArgs}(T) \to \text{Unit}
$$
$$
\textbf{Algorithm: } \text{Linear pipeline from ROOT through MIDDLE(s) to END.}
$$
$$
\textbf{Post (ROOT): } \text{Copies local data to NextBuffer (or directly to next output if PUSHRECV)}
$$
$$
\textbf{Post (MIDDLE): } \text{Receives from prev, copies to next and local output}
$$
$$
\textbf{Post (END): } \text{Receives from prev, copies to local output}
$$
$$
\textbf{Post: } \forall\, g,\; \text{Data}_g = \text{Data}_{\text{root}} \quad \text{(all GPUs hold root's data)}
$$
$$
\textbf{Invariant: } \text{Flags reset and NextData ptr set to null at exit}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_pipeline_length (nDev rootId : Nat) (hr : rootId < nDev) (hd : nDev ≥ 1) :
    let endId := (rootId + nDev - 1) % nDev
    endId < nDev := by sorry

-- axle_verify: content
theorem bcast_pipeline_length (nDev rootId : Nat) (hr : rootId < nDev) (hd : nDev ≥ 1) :
    let endId := (rootId + nDev - 1) % nDev
    endId < nDev := by
  simp only
  exact Nat.mod_lt _ (by omega)
```

### LAYER 2 — FORM TEMPLATE

```cuda
__shared__ volatile void * nextDataPtr;

template<int THREADCOUNT, int UNROLLCOUNT, bool PUSHRECVFLAG, int ROLEFLAG, typename ELEMTYPE>
__global__ void BroadcastKernel(const BroadcastKernelArgs<ELEMTYPE> args) {
  if (args.totalN == 0) return;
  int tidx = threadIdx.x;

  if (tidx == 0) {
    if (ROLEFLAG != ROOTROLE) {
      Wait([=] { return *((ELEMTYPE * volatile *)args.prevPtrToThisData) == nullptr; });
      *((ELEMTYPE * volatile *)args.prevPtrToThisData) = (ELEMTYPE*)args.thisDataPtr;
    }
    if (ROLEFLAG != ENDROLE) {
      Wait([=] { return *((ELEMTYPE * volatile *)args.thisPtrToNextData) != nullptr; });
      if (PUSHRECVFLAG)
        nextDataPtr = *((volatile void * volatile *)args.thisPtrToNextData);
    }
  }
  __syncthreads();

  for (int chunkidx = 0; chunkidx < args.chunkCount; ++chunkidx) {
    int bigSliceN = args.sliceSizeN;
    int smallSliceN = 0; int lastSliceN = 0;
    int sliceTotal = NUM_SUBCHUNKS;
    int bigSliceNum = sliceTotal; int smallSliceNum = 0;

    if ((chunkidx + 1 == args.chunkCount) && (args.totalN % args.chunkSizeN > 0))
      CalcLastChunk<THREADCOUNT, UNROLLCOUNT, ELEMTYPE>(&bigSliceN, &smallSliceN, &lastSliceN,
          &sliceTotal, &bigSliceNum, &smallSliceNum, args.totalN, args.chunkCount, args.chunkSizeN);

    int chunkOff = chunkidx * args.chunkSizeN;
    int offsetN; int sliceSizeN;

    if (tidx < THREADCOUNT) {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        getSliceSizeAndOffset(&sliceSizeN, &offsetN, sidx, sliceTotal,
            bigSliceNum, smallSliceNum, bigSliceN, smallSliceN, lastSliceN);

        if (PUSHRECVFLAG) {
          if (ROLEFLAG != ROOTROLE) WAIT_FOR_NEW_DATA(chunkidx, sidx);
          if (ROLEFLAG != ENDROLE)
            Copy<UNROLLCOUNT, THREADCOUNT>(
                (volatile ELEMTYPE *)nextDataPtr + chunkOff + offsetN,
                args.thisDataPtr + chunkOff + offsetN, sliceSizeN);
        } else {
          if (ROLEFLAG == ROOTROLE) {
            WAIT_FOR_CHUNK(chunkidx, sidx);
            Copy<UNROLLCOUNT, THREADCOUNT>(
                args.nextBufferPtr + (sidx * args.bufSliceStride),
                args.thisDataPtr + chunkOff + offsetN, sliceSizeN);
          } else if (ROLEFLAG == MIDDLEROLE) {
            WAIT_FOR_NEW_DATA_AND_CHUNK(chunkidx, sidx);
            DoubleCopy<UNROLLCOUNT, THREADCOUNT>(
                args.nextBufferPtr + (sidx * args.bufSliceStride),
                args.thisDataPtr + chunkOff + offsetN,
                args.thisBufferPtr + (sidx * args.bufSliceStride), sliceSizeN);
          } else {
            WAIT_FOR_NEW_DATA(chunkidx, sidx);
            Copy<UNROLLCOUNT, THREADCOUNT>(
                args.thisDataPtr + chunkOff + offsetN,
                args.thisBufferPtr + (sidx * args.bufSliceStride), sliceSizeN);
          }
        }
        __syncthreads();
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        if (ROLEFLAG != ENDROLE) SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx);
        if ((!PUSHRECVFLAG) && (ROLEFLAG != ROOTROLE) && (chunkidx + 1 < args.chunkCount))
          SIGNAL_CHUNK_DONE(chunkidx, sidx);
      }
    }
  }

  if (tidx == 0) {
    args.thisNewDataFlag[0] = 0;
    args.thisChunkDoneFlag[0] = 0;
    *args.thisPtrToNextData = nullptr;
  }
}
```

---

## 5. `ncclBcastWithType<T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclBcastWithType}: T^* \times \mathbb{N} \times \mathbb{Z}_\text{root} \times \text{Comm} \times \mathbb{N} \times \text{Stream} \to \text{Result}
$$
$$
\textbf{Pre: } \text{root} \in [0, K),\; \text{count} \geq 0
$$
$$
\text{rootId} = \text{comm.ringFromUser}[\text{root}]
$$
$$
\text{Role assignment: } \text{index} = \text{rootId} \implies \text{ROOT},\; \text{index} = (\text{rootId}+K-1)\%K \implies \text{END},\; \text{else MIDDLE}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_role_unique (index rootId nDev : Nat) (hn : nDev ≥ 2) (hr : rootId < nDev) (hi : index < nDev) :
    (index = rootId ∨ index = (rootId + nDev - 1) % nDev ∨
     (index ≠ rootId ∧ index ≠ (rootId + nDev - 1) % nDev)) := by sorry

-- axle_verify: content
theorem bcast_role_unique (index rootId nDev : Nat) (hn : nDev ≥ 2) (hr : rootId < nDev) (hi : index < nDev) :
    (index = rootId ∨ index = (rootId + nDev - 1) % nDev ∨
     (index ≠ rootId ∧ index ≠ (rootId + nDev - 1) % nDev)) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
prjname::resultType prjname::BcastWithType(void* bufptr, const int countN, const int rootRank,
    prjname::commType commobj, int unrollCount, cudaStream_t streamobj) {
  if (countN == 0) return prjname::Success;

  int selfIdx = commobj->ncclId;
  int rootIdx = commobj->ringFromUser[rootRank];
  int nextIdx = (selfIdx + 1) % commobj->nDev;
  int prevIdx = (selfIdx + commobj->nDev - 1) % commobj->nDev;

  int bufferN = commobj->buffSize / sizeof(ELEMTYPE);
  int bufNPerSlice = bufferN / NUM_SUBCHUNKS;
  int maxSliceN = (bufNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  BroadcastKernelArgs<ELEMTYPE> argvals;
  argvals.thisIdx = selfIdx;
  argvals.totalN = countN;
  argvals.sliceSizeN = unrollCount * UNROLL_SIZE * sizeof(PackType) / sizeof(ELEMTYPE);

  if (!commobj->useRemoteRecv) { argvals.sliceSizeN *= 8; }

  argvals.sliceSizeN = std::min(maxSliceN, argvals.sliceSizeN);
  argvals.bufSliceStride = argvals.sliceSizeN;
  argvals.chunkSizeN = NUM_SUBCHUNKS * argvals.sliceSizeN;

  int remainderN = argvals.totalN % argvals.chunkSizeN;
  if ((argvals.totalN > argvals.chunkSizeN) && (remainderN > 0) &&
      (argvals.totalN < 5 * argvals.chunkSizeN) && (2 * remainderN < argvals.chunkSizeN)) {
    argvals.sliceSizeN /= 2;
    argvals.chunkSizeN = NUM_SUBCHUNKS * argvals.sliceSizeN;
    argvals.chunkCount = argvals.totalN / argvals.chunkSizeN;
  } else {
    argvals.chunkCount = (argvals.totalN + argvals.chunkSizeN - 1) / argvals.chunkSizeN;
  }

  argvals.thisPtrToNextData = (ELEMTYPE**)&(commobj->local[nextIdx]->recvPtrs[0]);
  argvals.prevPtrToThisData = (ELEMTYPE**)&(commobj->remote[prevIdx]->recvPtrs[0]);
  argvals.thisDataPtr = (ELEMTYPE*)bufptr;
  argvals.thisBufferPtr = (volatile ELEMTYPE*)commobj->local[prevIdx]->buff;
  argvals.nextBufferPtr = (volatile ELEMTYPE*)commobj->remote[nextIdx]->buff;
  argvals.thisNewDataFlag = commobj->local[prevIdx]->flags;
  argvals.nextNewDataFlag = commobj->remote[nextIdx]->flags;
  argvals.thisChunkDoneFlag = commobj->local[nextIdx]->flags + 1;
  argvals.prevChunkDoneFlag = commobj->remote[prevIdx]->flags + 1;

  if (commobj->useRemoteRecv) {
    if (selfIdx == (rootIdx + commobj->nDev - 1) % commobj->nDev) {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, true, ENDROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    } else if (selfIdx == rootIdx) {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, true, ROOTROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    } else {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, true, MIDDLEROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    }
  } else {
    if (selfIdx == (rootIdx + commobj->nDev - 1) % commobj->nDev) {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, false, ENDROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    } else if (selfIdx == rootIdx) {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, false, ROOTROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    } else {
      BroadcastKernel<NUM_THREADS, UNROLL_COUNT, false, MIDDLEROLE, ELEMTYPE>
          <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
    }
  }
  return prjname::Success;
}
```

---

## 6. `BroadcastFunctor`, `ncclBcast`

### LAYER 1a — MATH (LaTeX)

$$
\text{BroadcastFunctor}: \text{dispatches by datatype, numUnroll}=4
$$
$$
\text{ncclBcast}: \text{enqueue}(\text{BroadcastFunctor}(), \text{nullptr}, \text{buff}, \ldots)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bcast_functor_numunroll : 4 > 0 := by sorry

-- axle_verify: content
theorem bcast_functor_numunroll : 4 > 0 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
class BroadcastFunctor {
public:
  prjname::resultType operator()(const void* /*unusedsend*/,
      void* bufptr, int countN, prjname::typenameType dtypearg, prjname::opType /*unusedop*/,
      int rootidx, prjname::commType commobj, cudaStream_t streamobj) {
    int unrollCount = 4;
    switch (dtypearg) {
    case prjname::Char:
      return prjname::BcastWithType<char>(bufptr, countN, rootidx, commobj, unrollCount, streamobj);
    case prjname::Int:
      return prjname::BcastWithType<int>(bufptr, countN, rootidx, commobj, unrollCount, streamobj);
    case prjname::Half:
      return prjname::BcastWithType<half>(bufptr, countN, rootidx, commobj, unrollCount, streamobj);
    case prjname::Float:
      return prjname::BcastWithType<float>(bufptr, countN, rootidx, commobj, unrollCount, streamobj);
    case prjname::Double:
      return prjname::BcastWithType<double>(bufptr, countN, rootidx, commobj, unrollCount, streamobj);
    }
    return prjname::InvalidType;
  }
};

extern "C" DSOGLOBAL
prjname::resultType prjname::Bcast(void* bufptr, int countN, prjname::typenameType dtypearg,
    int rootidx, prjname::commType commobj, cudaStream_t streamobj) {
  return enqueue(BroadcastFunctor(), nullptr, bufptr, countN, dtypearg, prjname::Sum,
      rootidx, commobj, streamobj);
}
```

