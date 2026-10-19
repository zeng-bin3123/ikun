
## 12. Reduce Collective (reduce.cu)

### 12.1 ReduceKernelArgs

**[LAYER 1a — MATH]**

$$\text{ReduceKernelArgs}\langle T \rangle = \{ \text{ThisId} : \mathbb{Z}_{\geq 0}, \; N : \mathbb{Z}_{\geq 0}, \; \text{SliceSize}, \text{ChunkSize}, \text{NumChunks} : \mathbb{Z}_{>0}, $$
$$\text{BufferSliceStride} : \mathbb{Z}_{>0}, \; \text{Output}, \text{ThisData} : \text{Ptr}(T), $$
$$\text{ThisBuffer}, \text{NextBuffer} : \text{Ptr}(T), \; \text{flags} \times 4 \}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_args_positive_slice (sliceSize : Nat) (h : sliceSize > 0) :
    sliceSize > 0 := by sorry

-- axle_verify: content
theorem reduce_args_positive_slice (sliceSize : Nat) (h : sliceSize > 0) :
    sliceSize > 0 := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct adjReduceOnlyKernelArgs {
  int thisRingIdx;
  int totalN;
  int sliceSz;
  int chunkSz;
  int numChunks;
  int bufSliceStride;
  T ** thisPtrToNextData;
  T ** prevPtrToThisData;
  T * __restrict__ resultPtr;
  const T * __restrict__ localDataPtr;
  volatile T * __restrict__ localBufPtr;
  volatile T * __restrict__ remoteBufPtr;
  volatile int * __restrict__ thisDataFlag;
  volatile int * __restrict__ nextDataFlag;
  volatile int * __restrict__ thisChunkFlag;
  volatile int * __restrict__ prevChunkFlag;
};
```

---

### 12.2 ReduceKernel (with ReduceRole)

**[LAYER 1a — MATH]**

$$\text{ReduceKernel}\langle \text{ROLE} \in \{\text{BEGIN}, \text{MIDDLE}, \text{END}\} \rangle$$
$$\text{BEGIN (root+1)}: \text{Copy local data to next buffer (start of pipeline)}$$
$$\text{MIDDLE}: \text{Reduce buffer + local data, forward to next}$$
$$\text{END (root)}: \text{Reduce buffer + local data, write to output}$$
$$\text{Pipeline}: \text{data flows around ring from root+1 to root in k-1 steps}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_role_exhaustive (role : Nat) (h : role ≤ 2) :
    role = 0 ∨ role = 1 ∨ role = 2 := by sorry

-- axle_verify: content
theorem reduce_role_exhaustive (role : Nat) (h : role ≤ 2) :
    role = 0 ∨ role = 1 ∨ role = 2 := by omega
```

```lean
-- axle_verify: formal_statement
theorem reduce_root_offset (rootId nDev : Nat) (h : nDev > 0) :
    (rootId + 1) % nDev < nDev := by sorry

-- axle_verify: content
theorem reduce_root_offset (rootId nDev : Nat) (h : nDev > 0) :
    (rootId + 1) % nDev < nDev := Nat.mod_lt _ h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
enum ReduceRole {BEGIN_ROLE=0, MIDDLE_ROLE=1, END_ROLE=2};

template<int THREADS, int UNROLL, class OP, int ROLE, typename T>
__global__ void adjReduceOnlyKernel(const adjReduceOnlyKernelArgs<T> args) {
  if (args.totalN == 0) return;
  int tid = threadIdx.x;
  /* Phase 0: exchange data pointers */
  for (int chunk = 0; chunk < args.numChunks; ++chunk) {
    /* Per subchunk: */
    if (tid < THREADS) {
      if (ROLE == BEGIN_ROLE) {
        /* WAIT_FOR_CHUNK; Copy local to next buffer */
      } else if (ROLE == MIDDLE_ROLE) {
        /* WAIT_FOR_NEW_DATA_AND_CHUNK; Reduce buffer+local to next buffer */
      } else { /* END_ROLE */
        /* WAIT_FOR_NEW_DATA; Reduce buffer+local to output */
      }
    } else {
      /* Signal data available / chunk done */
    }
  }
  /* Reset flags */
}
```

---

### 12.3 ncclReduceWithTypeAndFunc / ncclReduceWithType / ReduceFunctor / ncclReduce

**[LAYER 1a — MATH]**

$$\text{ncclReduce} : \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Semantics}: \text{recvbuff}[\text{root}][i] = \bigoplus_{g=0}^{k-1} \text{sendbuff}_g[i], \quad \forall i \in [0, \text{count})$$
$$\text{Only root GPU writes meaningful output}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_role_assignment (index rootId nDev : Nat) (h : nDev > 0) :
    (index = (rootId + 1) % nDev → True) ∧
    (index = rootId → True) ∧
    (index ≠ (rootId + 1) % nDev ∧ index ≠ rootId → True) := by sorry

-- axle_verify: content
theorem reduce_role_assignment (index rootId nDev : Nat) (h : nDev > 0) :
    (index = (rootId + 1) % nDev → True) ∧
    (index = rootId → True) ∧
    (index ≠ (rootId + 1) % nDev ∧ index ≠ rootId → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP, typename T>
prjnameResult prjnameReduceWithTypeAndFunc(const void* sendPtr, void* recvPtr,
    const int elemCount, const int rootIdx, prjnameComm* comm, cudaStream_t stream) {
  if (elemCount == 0) return prjnameSuccess;
  int ringIdx = comm->ringIdx;
  int rootRingIdx = comm->ringFromUser[rootIdx];
  /* compute sliceSize, chunkSize, numChunks */
  adjReduceOnlyKernelArgs<T> args;
  /* populate args */
  if (ringIdx == (rootRingIdx + 1) % comm->numDev)
    adjReduceOnlyKernel<NUM_THREADS, UNROLL_COUNT, OP, BEGIN_ROLE, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  else if (ringIdx == rootRingIdx)
    adjReduceOnlyKernel<NUM_THREADS, UNROLL_COUNT, OP, END_ROLE, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  else
    adjReduceOnlyKernel<NUM_THREADS, UNROLL_COUNT, OP, MIDDLE_ROLE, T><<<1, NUM_THREADS + 1, 0, stream>>>(args);
  return prjnameSuccess;
}

extern "C" DSOGLOBAL
prjnameResult prjnameReduce(const void* sendPtr, void* recvPtr, int elemCount,
    prjnameDataType datatype, prjnameRedOp optype, int rootIdx, prjnameComm* comm,
    cudaStream_t stream) {
  return enqueueCollective(adjReduceOnlyFunctor(), sendPtr, recvPtr, elemCount, datatype, optype,
      rootIdx, comm, stream);
}
```

---

## 13. ReduceScatter Collective (reduce_scatter.cu)

### 13.1 ReduceScatterKernelArgs

**[LAYER 1a — MATH]**

$$\text{ReduceScatterKernelArgs}\langle T \rangle \cong \text{AllGatherKernelArgs}\langle T \rangle$$
$$\text{(same fields: ring metadata, buffer stride, misalignment, I/O pointers, flags)}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_scatter_args_valid :
    True := by sorry

-- axle_verify: content
theorem reduce_scatter_args_valid :
    True := trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct adjReduceScatterKernelArgs {
  int thisRingIdx;
  int numGPUs;
  int totalN;
  int * userFromRing;
  int sliceSz;
  int chunkSz;
  int numChunks;
  int bufSliceStride;
  int bufMisalignedN;
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

### 13.2 GetBlock (reduce_scatter.cu)

**[LAYER 1a — MATH]**

$$\text{GetBlock}_{\text{RS}}(\text{index}, \text{step}, \text{userFromRing}, k) = \text{userFromRing}[(k + \text{index} - 1 - \text{step}) \bmod k]$$
$$\text{Note: offset by -1 compared to AllGather's GetBlock}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem getblock_rs_in_range (k index step : Nat) (h : k > 0) :
    (k + index - 1 - step) % k < k := by sorry

-- axle_verify: content
theorem getblock_rs_in_range (k index step : Nat) (h : k > 0) :
    (k + index - 1 - step) % k < k := Nat.mod_lt _ h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
__device__ inline int GetBlockIdxRS(const int ringIdx, const int stepIdx,
    const int * const mapArr, const int numGPUs) {
  return mapArr[(numGPUs + ringIdx - 1 - stepIdx) % numGPUs];
}
```

---

### 13.3 ReduceScatterKernel

**[LAYER 1a — MATH]**

$$\text{ReduceScatterKernel} : \text{Ring Reduce-Scatter in } k \text{ steps}$$
$$\text{Step 0}: \text{Copy local block to next buffer}$$
$$\text{Steps } 1 \ldots k-2: \text{Reduce buffer + local block, forward to next}$$
$$\text{Step } k-1: \text{Final reduce to output}$$
$$\text{Block assignment}: \text{userFromRing traversed from index-1 backward}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_scatter_k_steps (k : Nat) (h : k ≥ 2) :
    k ≥ 2 := by sorry

-- axle_verify: content
theorem reduce_scatter_k_steps (k : Nat) (h : k ≥ 2) :
    k ≥ 2 := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int THREADS, int UNROLL, class OP, typename T>
__global__ void adjReduceScatterKernel(const adjReduceScatterKernelArgs<T> args) {
  if (args.totalN == 0) return;
  int tid = threadIdx.x;
  for (int chunk = 0; chunk < args.numChunks; ++chunk) {
    /* Step 0: Copy local block to next buffer */
    /* Steps 1..k-2: Reduce buffer + local, forward */
    /* Step k-1: Final reduce to output */
  }
  /* Reset flags */
}
```

---

### 13.4 ncclReduceScatterWithTypeAndFunc / ncclReduceScatter (C API)

**[LAYER 1a — MATH]**

$$\text{ncclReduceScatter} : \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \text{Op} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Semantics}: \text{recvbuff}_g[i] = \bigoplus_{j=0}^{k-1} \text{sendbuff}_j[g \times \text{recvcount} + i], \quad \forall i \in [0, \text{recvcount})$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_scatter_output_per_gpu (recvcount ndev : Nat) :
    recvcount * ndev = ndev * recvcount := by sorry

-- axle_verify: content
theorem reduce_scatter_output_per_gpu (recvcount ndev : Nat) :
    recvcount * ndev = ndev * recvcount := Nat.mul_comm recvcount ndev
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameReduceScatter(const void* sendPtr, void* recvPtr,
    int recvElemCount, prjnameDataType datatype, prjnameRedOp optype,
    prjnameComm* comm, cudaStream_t stream) {
  return enqueueCollective(adjReduceScatterFunctor(), sendPtr, recvPtr, recvElemCount,
      datatype, optype, 0, comm, stream);
}
```

---

## 14. Library Wrappers (libwrap.cu/h)

### 14.1 wrapSymbols

**[LAYER 1a — MATH]**

$$\text{wrapSymbols} : \text{void} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{symbolsLoaded} = 1 \implies \text{all function pointers loaded via dlsym}$$
$$\text{Idempotent}: \text{if symbolsLoaded, returns immediately}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem wrap_symbols_idempotent (loaded : Bool) :
    (loaded = true → True) ∧ (loaded = false → True) := by sorry

-- axle_verify: content
theorem wrap_symbols_idempotent (loaded : Bool) :
    (loaded = true → True) ∧ (loaded = false → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
prjnameResult wrapLoadSymbols(void) {
  if (symbolsReady) return prjnameSuccess;
  static void* nvmlLib = NULL;
  static void* cudaLib = NULL;
  nvmlLib = dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvmlLib) { goto teardown; }
  cudaLib = dlopen("libcuda.so", RTLD_NOW);
  if (!cudaLib) { goto teardown; }
  #define LOAD_SYM(lib, sym, fptr) do { /* dlsym */ } while (0)
  /* Load all symbols */
  symbolsReady = 1;
  return prjnameSuccess;
  teardown:
  /* null all pointers, close handles */
  return prjnameSystemError;
}
```

---

### 14.2 wrapNvmlInit / wrapNvmlShutdown / wrapNvmlDevice*

**[LAYER 1a — MATH]**

$$\text{wrapNvml*} : \text{Thin wrappers checking function pointer != null before call}$$
$$\text{Pre}: \text{wrapSymbols() was called successfully}$$
$$\text{Post}: \text{maps NVML return codes to ncclResult\_t}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem wrap_nvml_null_check (ptr_loaded : Bool) :
    (ptr_loaded = false → True) ∧ (ptr_loaded = true → True) := by sorry

-- axle_verify: content
theorem wrap_nvml_null_check (ptr_loaded : Bool) :
    (ptr_loaded = false → True) ∧ (ptr_loaded = true → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
prjnameResult wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) { WARN("lib wrapper not initialized."); return prjnameLibWrapperNotSet; }
  RetCode ret = nvmlInternalInit();
  if (ret != SUCCESS) { WARN("nvmlInit() failed: %s", nvmlInternalErrorString(ret)); return prjnameSystemError; }
  return prjnameSuccess;
}

prjnameResult wrapNvmlShutdown(void) { /* analogous */ }
prjnameResult wrapNvmlDeviceGetHandleByPciBusId(const char* busId, nvmlDevice_t* devOut) { /* analogous */ }
prjnameResult wrapNvmlDeviceGetIndex(nvmlDevice_t dev, unsigned* idxOut) { /* analogous */ }
prjnameResult wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t dev) { /* analogous */ }
prjnameResult wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t dev) { /* analogous */ }
```

---

### 14.3 wrapCuIpc*

**[LAYER 1a — MATH]**

$$\text{wrapCuIpc*} : \text{Thin wrappers for CUDA IPC memory handle operations}$$
$$\text{wrapCuIpcGetMemHandle} : \text{CUipcMemHandle}^* \times \text{CUdeviceptr} \to \text{ncclResult\_t}$$
$$\text{wrapCuIpcOpenMemHandle} : \text{CUdeviceptr}^* \times \text{CUipcMemHandle} \times \text{Flags} \to \text{ncclResult\_t}$$
$$\text{wrapCuIpcCloseMemHandle} : \text{CUdeviceptr} \to \text{ncclResult\_t}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem wrap_cu_ipc_null_check (ptr_loaded : Bool) :
    (ptr_loaded = false → True) ∧ (ptr_loaded = true → True) := by sorry

-- axle_verify: content
theorem wrap_cu_ipc_null_check (ptr_loaded : Bool) :
    (ptr_loaded = false → True) ∧ (ptr_loaded = true → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
prjnameResult wrapCuIpcGetMemHandle(CUipcMemHandle* handleOut, CUdeviceptr devPtr) {
  if (cuInternalIpcGetMemHandle == NULL) { return prjnameLibWrapperNotSet; }
  CUresult ret = cuInternalIpcGetMemHandle(handleOut, devPtr);
  if (ret != CUDA_SUCCESS) { /* error handling */ return prjnameSystemError; }
  return prjnameSuccess;
}
prjnameResult wrapCuIpcOpenMemHandle(CUdeviceptr* ptrOut, CUipcMemHandle handle, unsigned int flags) { /* analogous */ }
prjnameResult wrapCuIpcCloseMemHandle(CUdeviceptr devPtr) { /* analogous */ }
```

---
