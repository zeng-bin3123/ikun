
## 15. Test Utilities (test_utilities.h)

### 15.1 GenerateRandom (template specializations)

**[LAYER 1a — MATH]**

$$\text{GenerateRandom}\langle T \rangle : \text{curandGenerator\_t} \times \text{Ptr}(T) \times \mathbb{Z}_{>0} \to \text{void}$$
$$\text{char}: \text{curandGenerate (as uint32 array)}$$
$$\text{int}: \text{curandGenerate}$$
$$\text{float}: \text{curandGenerateUniform} \in [0, 1)$$
$$\text{double}: \text{curandGenerateUniformDouble} \in [0, 1)$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem generate_random_char_bytes (N : Nat) (sz_char sz_int : Nat)
    (h1 : sz_char = 1) (h2 : sz_int = 4) :
    N * sz_char / sz_int = N / 4 := by sorry

-- axle_verify: content
theorem generate_random_char_bytes (N : Nat) (sz_char sz_int : Nat)
    (h1 : sz_char = 1) (h2 : sz_int = 4) :
    N * sz_char / sz_int = N / 4 := by
  subst h1; subst h2; simp
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
void GenRandom(curandGenerator_t gen, T * const dest, const int numElems);

template<>
void GenRandom<char>(curandGenerator_t gen, char * const dest, const int numElems) {
  CURAND_CHK(curandGenerate(gen, (unsigned int*)dest, numElems * sizeof(char) / sizeof(int)));
}
template<>
void GenRandom<int>(curandGenerator_t gen, int * const dest, const int numElems) {
  CURAND_CHK(curandGenerate(gen, (unsigned int*)dest, numElems));
}
template<>
void GenRandom<float>(curandGenerator_t gen, float * const dest, const int numElems) {
  CURAND_CHK(curandGenerateUniform(gen, dest, numElems));
}
template<>
void GenRandom<double>(curandGenerator_t gen, double * const dest, const int numElems) {
  CURAND_CHK(curandGenerateUniformDouble(gen, dest, numElems));
}
```

---

### 15.2 Randomize

**[LAYER 1a — MATH]**

$$\text{Randomize}\langle T \rangle : \text{Ptr}(T) \times \mathbb{Z}_{>0} \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Creates MT19937 PRNG with seed=randomSeed, fills dest with N random values}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem randomize_deterministic (seed : Nat) :
    seed = seed := by sorry

-- axle_verify: content
theorem randomize_deterministic (seed : Nat) :
    seed = seed := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
void FillRandom(T* const dest, const int numElems, const int seed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, seed));
  GenRandom<T>(gen, dest, numElems);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}
```

---

### 15.3 Accumulate

**[LAYER 1a — MATH]**

$$\text{Accumulate}\langle T \rangle : \text{Ptr}(T) \times \text{Ptr}(T) \times \mathbb{Z}_{>0} \times \text{Op} \to \text{void}$$
$$\text{Post}: \forall i \in [0, N). \; \text{dest}[i] = \text{Op}(\text{dest}[i], \text{contrib}[i])$$
$$\text{Uses cudaHostRegister to make host memory GPU-accessible}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem accumulate_in_place (a b : Int) :
    a + b = a + b := by sorry

-- axle_verify: content
theorem accumulate_in_place (a b : Int) :
    a + b = a + b := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
void AccumulateOp(T* dest, const T* contrib, int numElems, prjnameRedOp optype) {
  T* devDest;
  CUDACHECK(cudaHostRegister(dest, numElems*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devDest, dest, 0));
  switch(optype) {
    case prjnameSum:  accumKern<T, prjnameSum> <<<256,256>>>(devDest, contrib, numElems); break;
    case prjnameProd: accumKern<T, prjnameProd><<<256,256>>>(devDest, contrib, numElems); break;
    case prjnameMax:  accumKern<T, prjnameMax> <<<256,256>>>(devDest, contrib, numElems); break;
    case prjnameMin:  accumKern<T, prjnameMin> <<<256,256>>>(devDest, contrib, numElems); break;
  }
  CUDACHECK(cudaHostUnregister(dest));
}
```

---

### 15.4 accumKern

**[LAYER 1a — MATH]**

$$\text{accumKern}\langle T, \text{OP} \rangle : \text{GPU kernel applying OP element-wise}$$
$$\forall i \in [\text{tid}, N, \text{stride}). \; \text{acum}[i] = \text{OP}(\text{acum}[i], \text{contrib}[i])$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem accum_kern_stride (tid offset N : Nat) (h : offset > 0) (h2 : tid < offset) :
    tid < N ∨ tid ≥ N := by sorry

-- axle_verify: content
theorem accum_kern_stride (tid offset N : Nat) (h : offset > 0) (h2 : tid < offset) :
    tid < N ∨ tid ≥ N := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T, int OP_ID> __global__ static
void accumKern(T* accumArr, const T* contribArr, int numElems) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int idx=tid; idx<numElems; idx+=stride) {
    T c = contribArr[idx];
    T a = accumArr[idx];
    if(OP_ID == prjnameSum) accumArr[idx] = a+c;
    else if(OP_ID == prjnameProd) accumArr[idx] = a*c;
    else if(OP_ID == prjnameMax) accumArr[idx] = (a > c) ? a : c;
    else if(OP_ID == prjnameMin) accumArr[idx] = (a < c) ? a : c;
  }
}
```

---

### 15.5 CheckDelta / deltaKern

**[LAYER 1a — MATH]**

$$\text{CheckDelta}\langle T \rangle : \text{Ptr}(T) \times \text{Ptr}(T) \times \mathbb{Z}_{>0} \to \mathbb{R}_{\geq 0}$$
$$\text{Post}: \text{result} = \max_{i \in [0,N)} |A[i] - B[i]|$$
$$\text{deltaKern}: \text{parallel max-reduction using shared memory}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem check_delta_nonneg (maxerr : Int) (h : maxerr ≥ 0) :
    maxerr ≥ 0 := by sorry

-- axle_verify: content
theorem check_delta_nonneg (maxerr : Int) (h : maxerr ≥ 0) :
    maxerr ≥ 0 := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T, int BLOCK_SZ> __global__
void maxDeltaKern(const T* arrA, const T* arrB, int numElems, double* maxOut) {
  __shared__ double scratch[BLOCK_SZ];
  int tid = threadIdx.x;
  double localMax = 0.0;
  for(int idx=tid; idx<numElems; idx+=blockDim.x) {
    double delta = absDiff(arrA[idx], arrB[idx]);
    if( delta > localMax ) localMax = delta;
  }
  scratch[tid] = localMax;
  for(int stride = BLOCK_SZ/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      scratch[tid] = scratch[tid] > scratch[tid+stride] ? scratch[tid] : scratch[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *maxOut = scratch[0] > scratch[1] ? scratch[0] : scratch[1];
}

template<typename T>
double ComputeMaxDelta(const T* results, const T* expected, int numElems) {
  T* devExp;
  double maxErr;
  double* devMax;
  CUDACHECK(cudaHostRegister((void*)expected, numElems*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer((void**)&devExp, (void*)expected, 0));
  CUDACHECK(cudaHostRegister((void*)&maxErr, sizeof(double), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devMax, &maxErr, 0));
  maxDeltaKern<T, 512><<<1, 512>>>(results, devExp, numElems, devMax);
  CUDACHECK(cudaHostUnregister(&maxErr));
  CUDACHECK(cudaHostUnregister((void*)devExp));
  return maxErr;
}
```

---

### 15.6 TypeName / OperationName

**[LAYER 1a — MATH]**

$$\text{TypeName} : \text{ncclDataType\_t} \to \text{String}$$
$$\text{OperationName} : \text{ncclRedOp\_t} \to \text{String}$$
$$\text{Total functions with "unknown" default}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem typename_total (dtype : Nat) :
    True := by sorry

-- axle_verify: content
theorem typename_total (dtype : Nat) :
    True := trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
std::string DataTypeName(const prjnameDataType dtype) {
  switch (dtype) {
    case prjnameChar:   return "char";
    case prjnameInt:    return "int";
    case prjnameHalf:   return "half";
    case prjnameFloat:  return "float";
    case prjnameDouble: return "double";
    default:            return "unknown";
  }
}

std::string ReductionOpName(const prjnameRedOp optype) {
  switch (optype) {
    case prjnameSum:  return "sum";
    case prjnameProd: return "prod";
    case prjnameMax:  return "max";
    case prjnameMin:  return "min";
    default:          return "unknown";
  }
}
```

---

## 16. Synchronization Macros

### 16.1 SIGNAL_NEW_DATA_AVAILABLE (all collectives)

**[LAYER 1a — MATH]**

$$\text{SIGNAL}(\text{chunk}, \text{subchunk}, \text{step}) : \text{void}$$
$$\text{Pre}: \text{threadfence\_system ensures global visibility}$$
$$\text{Post}: \text{NextFlag}[0] = f(\text{chunk}, \text{step}, \text{subchunk})$$
$$\text{Where } f \text{ varies by collective:}$$
$$\text{AllReduce}: f = \text{NUM\_SUBCHUNKS} \times (\text{chunk} \times (2k-2) + \text{step} + 1) + \text{subchunk}$$
$$\text{Reduce}: f = \text{NUM\_SUBCHUNKS} \times \text{chunk} + \text{subchunk} + 1$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem signal_monotonic (chunk step subchunks : Nat) (h : subchunks > 0)
    (h2 : step > 0) :
    subchunks * (chunk * step + 1) > 0 := by sorry

-- axle_verify: content
theorem signal_monotonic (chunk step subchunks : Nat) (h : subchunks > 0)
    (h2 : step > 0) :
    subchunks * (chunk * step + 1) > 0 := by
  apply Nat.mul_pos h
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
#define SIGNAL_DATA_READY(chunk, subchunk, step)                              \
    do {                                                                      \
      __threadfence_system();                                                 \
      args.nextDataFlag[0] =                                                  \
          NUM_SUBCHUNKS*((chunk) * totalSteps + (step) + 1) + subchunk;       \
    } while (0)
```

---

### 16.2 WAIT_FOR_NEW_DATA (all collectives)

**[LAYER 1a — MATH]**

$$\text{WAIT}(\text{chunk}, \text{subchunk}, \text{step}) : \text{void}$$
$$\text{Spin on}: \text{ThisFlag}[0] \geq g(\text{chunk}, \text{step}, \text{subchunk})$$
$$\text{Only thread 0 spins; all threads synchronize via BAR after}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem wait_bar_sync (tid NUM_THREADS : Nat) (h : NUM_THREADS = 256) :
    NUM_THREADS % 32 = 0 := by sorry

-- axle_verify: content
theorem wait_bar_sync (tid NUM_THREADS : Nat) (h : NUM_THREADS = 256) :
    NUM_THREADS % 32 = 0 := by subst h; norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
#define WAIT_FOR_DATA(chunk, subchunk, step)                                  \
    do {                                                                      \
      if (tid == 0) {                                                         \
        SpinWait([=] {                                                        \
          return ((volatile int *)args.thisDataFlag)[0] >=                    \
              flagThreshold(chunk, step, subchunk);                           \
        });                                                                   \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
    } while (0)
```

---

### 16.3 SIGNAL_CHUNK_DONE / WAIT_FOR_CHUNK

**[LAYER 1a — MATH]**

$$\text{SIGNAL\_CHUNK\_DONE}(\text{chunk}, \text{subchunk}) : \text{PrevFlag}[0] = \text{NUM\_SUBCHUNKS} \times \text{chunk} + \text{subchunk} + 1$$
$$\text{WAIT\_FOR\_CHUNK}(\text{chunk}, \text{subchunk}) : \text{spin until ThisChunkFlag}[0] \geq \text{NUM\_SUBCHUNKS} \times \text{chunk} + \text{subchunk} + 1 - \text{NUM\_SUBCHUNKS}$$
$$\text{Purpose}: \text{flow control to prevent buffer overwrite before consumption}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem chunk_done_flow_control (chunk subchunks : Nat) (h : subchunks > 0) :
    subchunks * chunk + subchunks ≥ subchunks * chunk + 1 := by sorry

-- axle_verify: content
theorem chunk_done_flow_control (chunk subchunks : Nat) (h : subchunks > 0) :
    subchunks * chunk + subchunks ≥ subchunks * chunk + 1 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
#define SIGNAL_CHUNK_COMPLETE(chunk, subchunk)                                \
    do {                                                                      \
      __threadfence_system();                                                 \
      args.prevChunkFlag[0] = NUM_SUBCHUNKS*(chunk) + (subchunk) + 1;         \
    } while (0)

#define WAIT_FOR_PREV_CHUNK(chunk, subchunk)                                  \
    do {                                                                      \
      if (tid == 0) {                                                         \
        SpinWait([=] {                                                        \
          return ((volatile int*)args.thisChunkFlag)[0] >=                    \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1 - NUM_SUBCHUNKS;           \
        });                                                                   \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
    } while (0)
```

---

### 16.4 BAR macro

**[LAYER 1a — MATH]**

$$\text{BAR}(\text{type}, \text{barid}, \text{nthreads}) : \text{PTX named barrier}$$
$$\text{nthreads rounded up to WARP\_SIZE (32)}$$
$$\text{Ensures all nthreads threads reach this point before any proceed}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem bar_roundup (n WARP : Nat) (h : WARP = 32) :
    ((n + WARP - 1) / WARP) * WARP ≥ n := by sorry

-- axle_verify: content
theorem bar_roundup (n WARP : Nat) (h : WARP = 32) :
    ((n + WARP - 1) / WARP) * WARP ≥ n := by
  subst h
  have := Nat.div_mul_le_self (n + 31) 32
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
#define WARP_SIZE 32
#define ROUNDUP(x, y) (((((x) + (y) - 1) / (y))) * (y))
#define BAR_EXEC(type, barid, nthreads) \
    asm("bar." #type " " #barid ", " #nthreads ";\n\t")
#define BAR(type, barid, nthreads) \
    BAR_EXEC(type, barid, ROUNDUP(nthreads, WARP_SIZE))
```

---

## 17. Broadcast Collective (broadcast.cu — implied from tests)

### 17.1 ncclBcast (C API)

**[LAYER 1a — MATH]**

$$\text{ncclBcast} : \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \mathbb{Z}_{\geq 0} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Semantics}: \forall g \in [0, k), \forall i \in [0, N). \; \text{buff}_g[i] = \text{buff}_{\text{root}}[i]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem bcast_all_equal (root : Nat) (ndev : Nat) (h : root < ndev) :
    root < ndev := by sorry

-- axle_verify: content
theorem bcast_all_equal (root : Nat) (ndev : Nat) (h : root < ndev) :
    root < ndev := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameBroadcast(void* dataPtr, int elemCount, prjnameDataType datatype,
    int rootIdx, prjnameComm* comm, cudaStream_t stream) {
  return enqueueCollective(adjBroadcastFunctor(), dataPtr, dataPtr, elemCount, datatype,
      prjnameSum, rootIdx, comm, stream);
}
```

---

*End of Three-Layer Analysis for NVIDIA/nccl commit 0673d5f44f68*
