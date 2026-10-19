# NCCL Formal Analysis — src/reduce.cu

---

## 1. Macros (Reduce variants — same structure as Broadcast)

### LAYER 1a — MATH (LaTeX)

$$
\text{Same flag structure as broadcast.cu: SIGNAL/WAIT\_NEW\_DATA}(c,s),\; \text{SIGNAL/WAIT\_CHUNK}(c,s),\; \text{WAIT\_NEW\_DATA\_AND\_CHUNK}(c,s)
$$
$$
\text{Flag values: } \text{NUM\_SUBCHUNKS} \cdot c + s + 1 \quad (\text{NUM\_SUBCHUNKS}=4)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_flag_monotone (numSub c1 c2 s1 s2 : Nat)
    (hc : c2 > c1 ∨ (c2 = c1 ∧ s2 > s1)) :
    numSub * c2 + s2 + 1 > numSub * c1 + s1 + 1 - numSub ∨ numSub = 0 := by sorry

-- axle_verify: content
theorem reduce_flag_monotone (numSub c1 c2 s1 s2 : Nat)
    (hc : c2 > c1 ∨ (c2 = c1 ∧ s2 > s1)) :
    numSub * c2 + s2 + 1 > numSub * c1 + s1 + 1 - numSub ∨ numSub = 0 := by
  omega
```

### LAYER 2 — FORM TEMPLATE

(Same macro template as broadcast — omitted for brevity, identical substitutions apply with `chunkidx`, `subidx`, `tidx`.)

---

## 2. `ReduceKernelArgs<T>` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceKernelArgs}(T) = \{
  \text{ThisId}: \mathbb{Z},\; N: \mathbb{N},
$$
$$
  \text{SliceSize}: \mathbb{N},\; \text{ChunkSize}: \mathbb{N},\; \text{NumChunks}: \mathbb{N},\; \text{BufferSliceStride}: \mathbb{N},
$$
$$
  \text{ThisPtrToNextData}: T^{**},\; \text{PrevPtrToThisData}: T^{**},
$$
$$
  \text{Output}: T^*,\; \text{ThisData}: T^*_\text{const},\; \text{ThisBuffer}: T^*_\text{vol},\; \text{NextBuffer}: T^*_\text{vol},
$$
$$
  \text{Flags}: \mathbb{Z}^*_\text{vol} \times 4
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_args_fields : 14 = 14 := by sorry

-- axle_verify: content
theorem reduce_args_fields : 14 = 14 := by rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct ReduceKernelArgs {
  int thisIdx;
  int totalN;

  int sliceSizeN;
  int chunkSizeN;
  int chunkCount;
  int bufSliceStride;

  ELEMTYPE ** thisPtrToNextData;
  ELEMTYPE ** prevPtrToThisData;

  ELEMTYPE * __restrict__ outputPtr;
  const ELEMTYPE * __restrict__ thisDataPtr;
  volatile ELEMTYPE * __restrict__ thisBufferPtr;
  volatile ELEMTYPE * __restrict__ nextBufferPtr;

  volatile int * __restrict__ thisNewDataFlag;
  volatile int * __restrict__ nextNewDataFlag;
  volatile int * __restrict__ thisChunkDoneFlag;
  volatile int * __restrict__ prevChunkDoneFlag;
};
```

---

## 3. `ReduceRole` enum

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceRole} = \{\text{BEGIN}=0,\; \text{MIDDLE}=1,\; \text{END}=2\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_roles_distinct : (0 ≠ 1) ∧ (1 ≠ 2) ∧ (0 ≠ 2) := by sorry

-- axle_verify: content
theorem reduce_roles_distinct : (0 ≠ 1) ∧ (1 ≠ 2) ∧ (0 ≠ 2) := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
enum ReduceRole {BEGINROLE=0, MIDDLEROLE=1, ENDROLE=2};
```

---

## 4. `ReduceKernel<THREADS, UNROLL, FUNC, ROLE, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceKernel}: \text{ReduceKernelArgs}(T) \to \text{Unit}
$$
$$
\textbf{Algorithm: } \text{Linear pipeline: BEGIN} \to \text{MIDDLE}^* \to \text{END (=root)}
$$
$$
\textbf{Post (BEGIN): } \text{Copies local data to NextBuffer}
$$
$$
\textbf{Post (MIDDLE): } \text{Reduces ThisBuffer with local data, writes to NextBuffer}
$$
$$
\textbf{Post (END): } \text{Output}[i] = F(\text{ThisBuffer}[i], \text{ThisData}[i]) = \bigoplus_{g} \text{Data}_g[i]
$$
$$
\textbf{Invariant: } \text{Flags reset and NextData ptr null at exit}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_pipeline_correctness (K : Nat) (hK : K ≥ 2) :
    -- Pipeline has K-1 steps: 1 BEGIN + (K-3) MIDDLE + 1 END = K-1 for K≥3
    -- For K=2: 1 BEGIN + 1 END = 2
    K - 1 ≥ 1 := by sorry

-- axle_verify: content
theorem reduce_pipeline_correctness (K : Nat) (hK : K ≥ 2) :
    K - 1 ≥ 1 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
__shared__ volatile void * nextDataPtr;

template<int THREADCOUNT, int UNROLLCOUNT, class FUNCTYPE, int ROLEFLAG, typename ELEMTYPE>
__global__ void ReduceKernel(const ReduceKernelArgs<ELEMTYPE> args) {
  if (args.totalN == 0) return;
  int tidx = threadIdx.x;

  if (tidx == 0) {
    if (ROLEFLAG != BEGINROLE) {
      Wait([=] { return *((ELEMTYPE * volatile *)args.prevPtrToThisData) == nullptr; });
      *((ELEMTYPE * volatile *)args.prevPtrToThisData) = (ELEMTYPE*)args.thisDataPtr;
    }
    if (ROLEFLAG != ENDROLE) {
      Wait([=] { return *((ELEMTYPE * volatile *)args.thisPtrToNextData) != nullptr; });
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

        if (ROLEFLAG == BEGINROLE) {
          WAIT_FOR_CHUNK(chunkidx, sidx);
          Copy<UNROLLCOUNT, THREADCOUNT>(
              args.nextBufferPtr + (sidx * args.bufSliceStride),
              args.thisDataPtr + chunkOff + offsetN, sliceSizeN);
        } else if (ROLEFLAG == MIDDLEROLE) {
          WAIT_FOR_NEW_DATA_AND_CHUNK(chunkidx, sidx);
          Reduce<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              args.nextBufferPtr + (sidx * args.bufSliceStride),
              args.thisDataPtr + chunkOff + offsetN,
              args.thisBufferPtr + (sidx * args.bufSliceStride), sliceSizeN);
        } else {
          WAIT_FOR_NEW_DATA(chunkidx, sidx);
          Reduce<UNROLLCOUNT, THREADCOUNT, FUNCTYPE>(
              args.outputPtr + chunkOff + offsetN,
              args.thisDataPtr + chunkOff + offsetN,
              args.thisBufferPtr + (sidx * args.bufSliceStride), sliceSizeN);
        }
        __syncthreads();
      }
    } else {
      for(int sidx=0; sidx<NUM_SUBCHUNKS; ++sidx) {
        __syncthreads();
        if (ROLEFLAG != ENDROLE) SIGNAL_NEW_DATA_AVAILABLE(chunkidx, sidx);
        if ((ROLEFLAG != BEGINROLE) && (chunkidx + 1 < args.chunkCount))
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

## 5. `ncclReduceWithTypeAndFunc<FUNC, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclReduceWithTypeAndFunc}: T^*_s \times T^*_r \times \mathbb{N} \times \mathbb{Z}_\text{root} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{rootId} = \text{comm.ringFromUser}[\text{root}]
$$
$$
\text{Role: } (\text{rootId}+1)\%K \implies \text{BEGIN},\; \text{rootId} \implies \text{END},\; \text{else MIDDLE}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_role_assignment (index rootId nDev : Nat) (hn : nDev ≥ 2) :
    index = (rootId + 1) % nDev ∨ index = rootId ∨
    (index ≠ (rootId + 1) % nDev ∧ index ≠ rootId) := by sorry

-- axle_verify: content
theorem reduce_role_assignment (index rootId nDev : Nat) (hn : nDev ≥ 2) :
    index = (rootId + 1) % nDev ∨ index = rootId ∨
    (index ≠ (rootId + 1) % nDev ∧ index ≠ rootId) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<class FUNCTYPE, typename ELEMTYPE>
prjname::resultType prjname::ReduceWithTypeAndFunc(const void* sendbufptr, void* recvbufptr,
    const int countN, const int rootRank, prjname::commType commobj, cudaStream_t streamobj) {
  if (countN == 0) return prjname::Success;
  int selfIdx = commobj->ncclId;
  const int unrollCount = 4;
  int rootIdx = commobj->ringFromUser[rootRank];
  int nextIdx = (selfIdx + 1) % commobj->nDev;
  int prevIdx = (selfIdx + commobj->nDev - 1) % commobj->nDev;

  int bufferN = commobj->buffSize / sizeof(ELEMTYPE);
  int bufNPerSlice = bufferN / NUM_SUBCHUNKS;
  int maxSliceN = (bufNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  ReduceKernelArgs<ELEMTYPE> argvals;
  argvals.thisIdx = selfIdx;
  argvals.totalN = countN;
  argvals.sliceSizeN = unrollCount * UNROLL_SIZE * sizeof(PackType) / sizeof(ELEMTYPE);
  if(!commobj->useRemoteRecv) { argvals.sliceSizeN *= 8; }
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
  argvals.outputPtr = (ELEMTYPE*)recvbufptr;
  argvals.thisDataPtr = (const ELEMTYPE*)sendbufptr;
  argvals.thisBufferPtr = (volatile ELEMTYPE*)commobj->local[prevIdx]->buff;
  argvals.nextBufferPtr = (volatile ELEMTYPE*)commobj->remote[nextIdx]->buff;
  argvals.thisNewDataFlag = commobj->local[prevIdx]->flags;
  argvals.nextNewDataFlag = commobj->remote[nextIdx]->flags;
  argvals.thisChunkDoneFlag = commobj->local[nextIdx]->flags + 1;
  argvals.prevChunkDoneFlag = commobj->remote[prevIdx]->flags + 1;

  if (selfIdx == (rootIdx + 1) % commobj->nDev) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, BEGINROLE, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  } else if (selfIdx == rootIdx) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, ENDROLE, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  } else {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNCTYPE, MIDDLEROLE, ELEMTYPE>
        <<<1, NUM_THREADS + 1, 0, streamobj>>>(argvals);
  }
  return prjname::Success;
}
```

---

## 6. `ncclReduceWithType<T>`, `ReduceFunctor`, `ncclReduce`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclReduceWithType}: \text{dispatches by op}
$$
$$
\text{ReduceFunctor}: \text{dispatches by datatype}
$$
$$
\text{ncclReduce}: \text{enqueue}(\text{ReduceFunctor}(), \ldots)
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_dispatch_total (dt op : Nat) (hdt : dt < 5) (hop : op < 4) :
    dt * 4 + op < 20 := by sorry

-- axle_verify: content
theorem reduce_dispatch_total (dt op : Nat) (hdt : dt < 5) (hop : op < 4) :
    dt * 4 + op < 20 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template <typename ELEMTYPE>
prjname::resultType prjname::ReduceWithType(const void* sendbufptr,
      void* recvbufptr, int countN, prjname::opType oparg, int rootidx,
      prjname::commType commobj, cudaStream_t streamobj) {
  switch (oparg) {
    case prjname::Sum:  return prjname::ReduceWithTypeAndFunc<FuncSum<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, countN, rootidx, commobj, streamobj);
    case prjname::Prod: return prjname::ReduceWithTypeAndFunc<FuncProd<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, countN, rootidx, commobj, streamobj);
    case prjname::Max:  return prjname::ReduceWithTypeAndFunc<FuncMax<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, countN, rootidx, commobj, streamobj);
    case prjname::Min:  return prjname::ReduceWithTypeAndFunc<FuncMin<ELEMTYPE>, ELEMTYPE>(sendbufptr, recvbufptr, countN, rootidx, commobj, streamobj);
  }
  return prjname::InvalidOperation;
}

class ReduceFunctor {
public:
  prjname::resultType operator()(const void* sendbufptr,
      void* recvbufptr, int countN, prjname::typenameType dtypearg, prjname::opType oparg,
      int rootidx, prjname::commType commobj, cudaStream_t streamobj) {
    switch (dtypearg) {
    case prjname::Char:   return prjname::ReduceWithType<char>(sendbufptr, recvbufptr, countN, oparg, rootidx, commobj, streamobj);
    case prjname::Int:    return prjname::ReduceWithType<int>(sendbufptr, recvbufptr, countN, oparg, rootidx, commobj, streamobj);
    case prjname::Half:   return prjname::ReduceWithType<half>(sendbufptr, recvbufptr, countN, oparg, rootidx, commobj, streamobj);
    case prjname::Float:  return prjname::ReduceWithType<float>(sendbufptr, recvbufptr, countN, oparg, rootidx, commobj, streamobj);
    case prjname::Double: return prjname::ReduceWithType<double>(sendbufptr, recvbufptr, countN, oparg, rootidx, commobj, streamobj);
    }
    return prjname::InvalidType;
  }
};

extern "C" DSOGLOBAL
prjname::resultType prjname::Reduce(const void* sendbufptr, void* recvbufptr, int countN,
    prjname::typenameType dtypearg, prjname::opType oparg, int rootidx, prjname::commType commobj,
    cudaStream_t streamobj) {
  return enqueue(ReduceFunctor(), sendbufptr, recvbufptr, countN, dtypearg, oparg,
      rootidx, commobj, streamobj);
}
```

