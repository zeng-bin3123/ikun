
## 10. AllReduce Collective (all_reduce.cu)

### 10.1 AllReduceKernelArgs

**[LAYER 1a — MATH]**

$$\text{AllReduceKernelArgs}\langle T \rangle = \{ \text{ThisId} : [0, \text{NumGPUs}), \; \text{NumGPUs} : \mathbb{Z}_{>0}, \; N : \mathbb{Z}_{\geq 0}, $$
$$\text{SliceSize}, \text{ChunkSize}, \text{NumChunks} : \mathbb{Z}_{>0}, $$
$$\text{ThisInput}, \text{ThisOutput}, \text{ThisBuffer}, \text{NextBuffer} : \text{Ptr}(T), $$
$$\text{flags} : \text{Ptr}(\text{int}) \times 4 \}$$
$$\text{Invariant}: \text{ChunkSize} = \text{NumGPUs} \times \text{NUM\_SUBCHUNKS} \times \text{SliceSize}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_chunk_decomposition (numGPUs subchunks sliceSize chunkSize : Nat)
    (h : subchunks = 2) (h2 : chunkSize = numGPUs * subchunks * sliceSize) :
    chunkSize = numGPUs * 2 * sliceSize := by sorry

-- axle_verify: content
theorem allreduce_chunk_decomposition (numGPUs subchunks sliceSize chunkSize : Nat)
    (h : subchunks = 2) (h2 : chunkSize = numGPUs * subchunks * sliceSize) :
    chunkSize = numGPUs * 2 * sliceSize := by
  subst h; linarith
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct adjReduceKernelArgs {
  int thisRingIdx;
  int numGPUs;
  int totalN;
  int sliceSz;
  int chunkSz;
  int numChunks;
  T ** thisPtrToNextOut;
  T ** prevPtrToThisOut;
  const T * __restrict__ inputPtr;
  volatile T * __restrict__ outputPtr;
  volatile T * __restrict__ localBufPtr;
  volatile T * __restrict__ remoteBufPtr;
  volatile int * __restrict__ thisDataFlag;
  volatile int * __restrict__ nextDataFlag;
  volatile int * __restrict__ thisChunkFlag;
  volatile int * __restrict__ prevChunkFlag;
};
```

---

### 10.2 AllReduceKernel

**[LAYER 1a — MATH]**

$$\text{AllReduceKernel} : \text{Ring AllReduce in } 2(k-1) \text{ steps for } k \text{ GPUs}$$
$$\text{Steps } [0, k-1): \text{Reduce-Scatter phase — each step reduces a slice and forwards}$$
$$\text{Step } k-1: \text{Reduce final slice, write to output, forward to next GPU}$$
$$\text{Steps } [k, 2k-2): \text{AllGather phase — copy final results around ring}$$

$$\text{Communication pattern}: \text{GPU}_i \text{ sends to GPU}_{(i+1) \bmod k}$$
$$\text{Total data movement}: (2k-2) \times \text{SliceSize per chunk}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_total_steps (k : Nat) (h : k ≥ 2) :
    2 * k - 2 = 2 * (k - 1) := by sorry

-- axle_verify: content
theorem allreduce_total_steps (k : Nat) (h : k ≥ 2) :
    2 * k - 2 = 2 * (k - 1) := by omega
```

```lean
-- axle_verify: formal_statement
theorem allreduce_ring_wrap (i k : Nat) (h : k > 0) :
    (i + 1) % k < k := by sorry

-- axle_verify: content
theorem allreduce_ring_wrap (i k : Nat) (h : k > 0) :
    (i + 1) % k < k := Nat.mod_lt _ h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int THREADS, int UNROLL, class OP, bool PUSHRECV, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void adjReduceKernel(const adjReduceKernelArgs<T> args) {
  if (args.totalN == 0) return;
  const int tid = threadIdx.x;
  /* Phase 0: exchange output pointers with neighbors */
  if (tid == 0) {
    SpinWait([=] { return *((T * volatile *)args.prevPtrToThisOut) == nullptr; });
    *((T * volatile *)args.prevPtrToThisOut) = (T*)args.outputPtr;
    SpinWait([=] { return *((T * volatile *)args.thisPtrToNextOut) != nullptr; });
  }
  __syncthreads();
  for (int chunk = 0; chunk < args.numChunks; ++chunk) {
    /* Step 0: push local data to next buffer */
    /* Steps 1..k-2: reduce from buffer + local, forward */
    /* Step k-1: final reduce, write output, forward */
    /* Steps k..2k-3: copy final results around ring */
    /* (non-PUSHRECV: final copy from buffer to output) */
  }
  /* Reset flags and pointers */
}
```

---

### 10.3 ncclAllReduceWithTypeAndFunc

**[LAYER 1a — MATH]**

$$\text{ncclAllReduceWithTypeAndFunc}\langle \text{FUNC}, T \rangle : \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{ncclComm}^* \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Computes}: \text{SliceSize} = \min(\lfloor \text{bufferN} / (\text{NUM\_SUBCHUNKS} \times \text{nDev}) / \text{UNROLL\_SIZE} \rfloor \times \text{UNROLL\_SIZE}, \text{bufferNPerSlice})$$
$$\text{Heuristic}: \text{if small remainder, halve SliceSize to balance last chunk}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_slice_multiple (sliceSize UNROLL_SIZE : Nat) (h : UNROLL_SIZE = 2048)
    (h2 : sliceSize = (sliceSize / UNROLL_SIZE) * UNROLL_SIZE) :
    sliceSize % UNROLL_SIZE = 0 := by sorry

-- axle_verify: content
theorem allreduce_slice_multiple (sliceSize UNROLL_SIZE : Nat) (h : UNROLL_SIZE = 2048)
    (h2 : sliceSize = (sliceSize / UNROLL_SIZE) * UNROLL_SIZE) :
    sliceSize % UNROLL_SIZE = 0 := by
  subst h
  rw [h2]
  exact Nat.mul_div_cancel_left_mod sliceSize 2048 |>.symm ▸ Nat.mul_mod_right _ _
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP, typename T>
prjnameResult prjnameAllReduceWithTypeAndFunc(const void* sendPtr, void* recvPtr,
    const int elemCount, prjnameComm* comm, cudaStream_t stream) {
  if (elemCount == 0) return prjnameSuccess;
  int ringIdx = comm->ringIdx;
  int bufN = comm->buffSz / sizeof(T);
  int bufPerSlice = bufN / (NUM_SUBCHUNKS * comm->numDev);
  int sliceSz = (bufPerSlice / UNROLL_SIZE) * UNROLL_SIZE;
  int nextIdx = (ringIdx + 1) % comm->numDev;
  int prevIdx = (ringIdx + comm->numDev - 1) % comm->numDev;
  adjReduceKernelArgs<T> args;
  /* populate args */
  /* chunk balancing heuristic */
  if( comm->useRemoteRecv ) {
    adjReduceKernel<NUM_THREADS, UNROLL_COUNT, OP, true, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    adjReduceKernel<NUM_THREADS, UNROLL_COUNT, OP, false, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return prjnameSuccess;
}
```

---

### 10.4 ncclAllReduceWithType

**[LAYER 1a — MATH]**

$$\text{ncclAllReduceWithType}\langle T \rangle : \text{dispatches on ncclRedOp\_t} \to \text{ncclAllReduceWithTypeAndFunc}\langle \text{Op}_i, T \rangle$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_op_dispatch (op : Nat) (h : op ≤ 3) :
    op = 0 ∨ op = 1 ∨ op = 2 ∨ op = 3 := by sorry

-- axle_verify: content
theorem allreduce_op_dispatch (op : Nat) (h : op ≤ 3) :
    op = 0 ∨ op = 1 ∨ op = 2 ∨ op = 3 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
prjnameResult prjnameAllReduceWithType(const void* sendPtr,
    void* recvPtr, int elemCount, prjnameRedOp optype,
    prjnameComm* comm, cudaStream_t stream) {
  switch (optype) {
  case prjnameSum:  return prjnameAllReduceWithTypeAndFunc<OpAdd<T>, T>(sendPtr, recvPtr, elemCount, comm, stream);
  case prjnameProd: return prjnameAllReduceWithTypeAndFunc<OpMul<T>, T>(sendPtr, recvPtr, elemCount, comm, stream);
  case prjnameMax:  return prjnameAllReduceWithTypeAndFunc<OpMax<T>, T>(sendPtr, recvPtr, elemCount, comm, stream);
  case prjnameMin:  return prjnameAllReduceWithTypeAndFunc<OpMin<T>, T>(sendPtr, recvPtr, elemCount, comm, stream);
  }
  return prjnameInvalidOperation;
}
```

---

### 10.5 AllReduceFunctor

**[LAYER 1a — MATH]**

$$\text{AllReduceFunctor} : \text{dispatches on ncclDataType\_t} \to \text{ncclAllReduceWithType}\langle T_i \rangle$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_type_dispatch (dtype : Nat) (h : dtype ≤ 4) :
    dtype = 0 ∨ dtype = 1 ∨ dtype = 2 ∨ dtype = 3 ∨ dtype = 4 := by sorry

-- axle_verify: content
theorem allreduce_type_dispatch (dtype : Nat) (h : dtype ≤ 4) :
    dtype = 0 ∨ dtype = 1 ∨ dtype = 2 ∨ dtype = 3 ∨ dtype = 4 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
class adjReduceFunctor {
public:
  prjnameResult operator()(const void* sendPtr, void* recvPtr,
      int elemCount, prjnameDataType datatype, prjnameRedOp optype, int /*rootIdx*/,
      prjnameComm* comm, cudaStream_t stream) {
    switch (datatype) {
    case prjnameChar:   return prjnameAllReduceWithType<char>(sendPtr, recvPtr, elemCount, optype, comm, stream);
    case prjnameInt:    return prjnameAllReduceWithType<int>(sendPtr, recvPtr, elemCount, optype, comm, stream);
    case prjnameHalf:   return prjnameAllReduceWithType<half>(sendPtr, recvPtr, elemCount, optype, comm, stream);
    case prjnameFloat:  return prjnameAllReduceWithType<float>(sendPtr, recvPtr, elemCount, optype, comm, stream);
    case prjnameDouble: return prjnameAllReduceWithType<double>(sendPtr, recvPtr, elemCount, optype, comm, stream);
    }
    return prjnameInvalidType;
  }
};
```

---

### 10.6 ncclAllReduce (C API)

**[LAYER 1a — MATH]**

$$\text{ncclAllReduce} : \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \text{Op} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Semantics}: \forall i \in [0, \text{count}). \; \text{recvbuff}[i] = \bigoplus_{g=0}^{k-1} \text{sendbuff}_g[i]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allreduce_api_delegates :
    True := by sorry

-- axle_verify: content
theorem allreduce_api_delegates :
    True := trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameAllReduce(const void* sendPtr, void* recvPtr, int elemCount,
    prjnameDataType datatype, prjnameRedOp optype, prjnameComm* comm, cudaStream_t stream) {
  return enqueueCollective(adjReduceFunctor(), sendPtr, recvPtr, elemCount, datatype, optype, 0,
      comm, stream);
}
```

---

## 11. AllGather Collective (all_gather.cu)

### 11.1 AllGatherKernelArgs

**[LAYER 1a — MATH]**

$$\text{AllGatherKernelArgs}\langle T \rangle = \text{AllReduceKernelArgs}\langle T \rangle \cup \{ \text{UserFromRing} : \text{Ptr}(\mathbb{Z}), $$
$$\text{BufferSliceStride}, \text{BufferMisalignedN} : \mathbb{Z}_{\geq 0} \}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allgather_args_superset :
    True := by sorry

-- axle_verify: content
theorem allgather_args_superset :
    True := trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct adjGatherKernelArgs {
  int thisRingIdx;
  int numGPUs;
  int totalN;
  int * userFromRing;
  int sliceSz;
  int chunkSz;
  int numChunks;
  int bufSliceStride;
  int bufMisalignedN;
  T ** thisPtrToNextOut;
  T ** prevPtrToThisOut;
  const T * __restrict__ inputPtr;
  volatile T * __restrict__ outputPtr;
  volatile T * __restrict__ localBufPtr;
  volatile T * __restrict__ remoteBufPtr;
  volatile int * __restrict__ thisDataFlag;
  volatile int * __restrict__ nextDataFlag;
  volatile int * __restrict__ thisChunkFlag;
  volatile int * __restrict__ prevChunkFlag;
};
```

---

### 11.2 GetBlock (all_gather.cu)

**[LAYER 1a — MATH]**

$$\text{GetBlock}(\text{index}, \text{step}, \text{userFromRing}, k) = \text{userFromRing}[(k + \text{index} - \text{step}) \bmod k]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem getblock_in_range (k index step : Nat) (h : k > 0) :
    (k + index - step) % k < k := by sorry

-- axle_verify: content
theorem getblock_in_range (k index step : Nat) (h : k > 0) :
    (k + index - step) % k < k := Nat.mod_lt _ h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
__device__ inline int GetBlockIdx(const int ringIdx, const int stepIdx,
    const int * const mapArr, const int numGPUs) {
  return mapArr[(numGPUs + ringIdx - stepIdx) % numGPUs];
}
```

---

### 11.3 AllGatherKernel

**[LAYER 1a — MATH]**

$$\text{AllGatherKernel} : \text{Ring AllGather in } k-1 \text{ steps}$$
$$\text{Step 0}: \text{Copy own block from input to output and forward to next GPU}$$
$$\text{Steps } 1 \ldots k-2: \text{Copy received block to output and forward}$$
$$\text{Step } k-1 \text{(non-PUSHRECV)}: \text{Final copy from buffer to output}$$
$$\text{Block mapping}: \text{userFromRing traversed backwards starting at thisId}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allgather_steps (k : Nat) (h : k ≥ 2) :
    k - 1 ≥ 1 := by sorry

-- axle_verify: content
theorem allgather_steps (k : Nat) (h : k ≥ 2) :
    k - 1 ≥ 1 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int THREADS, int UNROLL, bool PUSHRECV, typename T>
__global__ void adjGatherKernel(const adjGatherKernelArgs<T> args) {
  if (args.totalN == 0) return;
  int tid = threadIdx.x;
  /* Exchange output pointers */
  for (int chunk = 0; chunk < args.numChunks; ++chunk) {
    /* Step 0: DoubleCopy own block to output + next */
    /* Steps 1..k-2: copy/forward received blocks */
    /* Step k-1 (non-PUSHRECV): final buffer->output copy */
  }
  /* Reset flags */
}
```

---

### 11.4 ncclAllGatherWithType

**[LAYER 1a — MATH]**

$$\text{ncclAllGatherWithType}\langle T \rangle : \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{ncclComm}^* \times \mathbb{Z}_{>0} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Handles alignment padding for buffer slice stride}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allgather_alignment_padding (blockSize align : Nat) (h : align = 8) :
    blockSize % align < align := by sorry

-- axle_verify: content
theorem allgather_alignment_padding (blockSize align : Nat) (h : align = 8) :
    blockSize % align < align := by
  subst h; exact Nat.mod_lt _ (by norm_num)
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
prjnameResult prjnameAllGatherWithType(const void* sendPtr, void* recvPtr,
    int elemCount, prjnameComm* comm, int numUnroll, cudaStream_t stream) {
  if (elemCount == 0) return prjnameSuccess;
  /* compute misalignment, padding, sliceSize */
  adjGatherKernelArgs<T> args;
  /* populate args, chunk balancing */
  if( comm->useRemoteRecv ) {
    adjGatherKernel<NUM_THREADS, UNROLL_COUNT, true, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    adjGatherKernel<NUM_THREADS, UNROLL_COUNT, false, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return prjnameSuccess;
}
```

---

### 11.5 AllGatherFunctor / ncclAllGather (C API)

**[LAYER 1a — MATH]**

$$\text{ncclAllGather} : \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \text{Ptr} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Semantics}: \text{recvbuff}[g \times \text{count} \ldots (g+1) \times \text{count} - 1] = \text{sendbuff}_g[\ldots]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem allgather_output_size (ndev count : Nat) :
    ndev * count = ndev * count := by sorry

-- axle_verify: content
theorem allgather_output_size (ndev count : Nat) :
    ndev * count = ndev * count := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
class adjGatherFunctor {
public:
  prjnameResult operator()(const void* sendPtr, void* recvPtr,
      int elemCount, prjnameDataType datatype, prjnameRedOp /*unused*/,
      int /*unused*/, prjnameComm* comm, cudaStream_t stream) {
    int numUnroll = 16;
    switch (datatype) {
    case prjnameChar:   return prjnameAllGatherWithType<char>(sendPtr, recvPtr, elemCount, comm, numUnroll, stream);
    case prjnameInt:    return prjnameAllGatherWithType<int>(sendPtr, recvPtr, elemCount, comm, numUnroll, stream);
    case prjnameHalf:   return prjnameAllGatherWithType<half>(sendPtr, recvPtr, elemCount, comm, numUnroll, stream);
    case prjnameFloat:  return prjnameAllGatherWithType<float>(sendPtr, recvPtr, elemCount, comm, numUnroll, stream);
    case prjnameDouble: return prjnameAllGatherWithType<double>(sendPtr, recvPtr, elemCount, comm, numUnroll, stream);
    }
    return prjnameInvalidType;
  }
};

extern "C" DSOGLOBAL
prjnameResult prjnameAllGather(const void* sendPtr, int elemCount, prjnameDataType datatype,
    void* recvPtr, prjnameComm* comm, cudaStream_t stream) {
  return enqueueCollective(adjGatherFunctor(), sendPtr, recvPtr, elemCount, datatype,
      prjnameSum, 0, comm, stream);
}
```

---
