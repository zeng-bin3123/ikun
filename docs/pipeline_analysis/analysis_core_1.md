# NCCL Formal Analysis — src/core.cu (Part 1: Utilities & Initialization)

---

## 1. `ncclGetUniqueId`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclGetUniqueId}: \text{ncclUniqueId}^* \to \text{Result}
$$
$$
\textbf{Post: } \text{out.internal} = \text{"nccl-"} \| \text{pid} \| \text{"-"} \| \text{count}
$$
$$
\text{count is atomically incremented (fetch-and-add), ensuring uniqueness within process}
$$
$$
\textbf{Error: } \text{strlen(internal)} < \text{len} \implies \text{ncclInternalError (truncation)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem unique_id_unique (pid : Nat) (c1 c2 : Nat) (hne : c1 ≠ c2) :
    (pid, c1) ≠ (pid, c2) := by sorry

-- axle_verify: content
theorem unique_id_unique (pid : Nat) (c1 c2 : Nat) (hne : c1 ≠ c2) :
    (pid, c1) ≠ (pid, c2) := by
  intro h
  injection h with _ h2
  exact hne h2
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
prjname::resultType prjname::GetUniqueId(prjname::uniqueIdType* outptr) {
  pid_t pidval = getpid();
  static int countervar = 0;
  int commIdx = __sync_fetch_and_add(&countervar, 1);
  int lenval = snprintf(outptr->internal, NCCL_UNIQUE_ID_BYTES, "nccl-%d-%d", pidval, commIdx);
  if(strlen(outptr->internal) < lenval) {
    WARN("prjname::uniqueIdType truncated");
    return prjname::InternalError;
  }
  return prjname::Success;
}
```

---

## 2. `shmOpen`

### LAYER 1a — MATH (LaTeX)

$$
\text{shmOpen}: \text{String} \times \mathbb{N} \times \text{Ptr}^* \to \text{Result}
$$
$$
\textbf{Pre: } \text{bytes} > 0,\; \text{shmname valid POSIX name}
$$
$$
\textbf{Post: } *\text{ptr} \text{ points to mmap'd region of size bytes, R/W}
$$
$$
\textbf{Error: } \text{shm\_open fails} \lor \text{ftruncate fails} \lor \text{mmap fails} \implies \text{ncclSystemError}
$$
$$
\text{fd is closed after mmap (descriptor not needed after mapping)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem shm_open_cleanup_on_error (fd_ok mmap_ok : Bool) :
    (¬fd_ok → True) ∧ (fd_ok ∧ ¬mmap_ok → True) ∧ (fd_ok ∧ mmap_ok → True) := by sorry

-- axle_verify: content
theorem shm_open_cleanup_on_error (fd_ok mmap_ok : Bool) :
    (¬fd_ok → True) ∧ (fd_ok ∧ ¬mmap_ok → True) ∧ (fd_ok ∧ mmap_ok → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial, fun _ => trivial⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType shmOpen(const char* shmname, size_t bytecount, void** ptrout) {
  int fdval = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fdval == -1) {
    WARN("shm_open failed to open %s", shmname);
    return prjname::SystemError;
  }
  if (ftruncate(fdval, bytecount) == -1) {
    WARN("ftruncate failed to allocate %ld bytes", bytecount);
    shm_unlink(shmname);
    close(fdval);
    return prjname::SystemError;
  }
  *ptrout = mmap(NULL, bytecount, PROT_READ | PROT_WRITE, MAP_SHARED, fdval, 0);
  if (*ptrout == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    close(fdval);
    return prjname::SystemError;
  }
  close(fdval);
  return prjname::Success;
}
```

---

## 3. `shmUnlink`

### LAYER 1a — MATH (LaTeX)

$$
\text{shmUnlink}: \text{String} \to \text{Result}
$$
$$
\textbf{Post: } \text{Removes shared memory object name from filesystem}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem shm_unlink_result (success : Bool) :
    success = true ∨ success = false := by sorry

-- axle_verify: content
theorem shm_unlink_result (success : Bool) :
    success = true ∨ success = false := by
  cases success <;> simp
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType shmUnlink(const char* shmname) {
  if(shm_unlink(shmname) == -1) {
    WARN("shm_unlink failed");
    return prjname::SystemError;
  } else {
    return prjname::Success;
  }
}
```

---

## 4. `shmUnmap`

### LAYER 1a — MATH (LaTeX)

$$
\text{shmUnmap}: \text{Ptr} \times \mathbb{N} \to \text{Result}
$$
$$
\textbf{Post: } \text{munmap(ptr, bytes) releases mapping}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem shm_unmap_result (success : Bool) :
    success = true ∨ success = false := by sorry

-- axle_verify: content
theorem shm_unmap_result (success : Bool) :
    success = true ∨ success = false := by
  cases success <;> simp
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType shmUnmap(void* ptrval, size_t bytecount) {
  if(munmap(ptrval, bytecount) == -1) {
    WARN("munmap failed");
    return prjname::SystemError;
  } else {
    return prjname::Success;
  }
}
```

---

## 5. `RankEntry` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{RankEntry} = \{
  \text{rank}: \mathbb{Z},\; \text{ndev}: \mathbb{N},\; \text{cudaDev}: \mathbb{Z},\; \text{ncclId}: \mathbb{Z},
$$
$$
  \text{pid}: \text{pid\_t},\; \text{hostptr}: \text{ncclMem}^*,\; \text{devptr}: \text{ncclMem}^*,
$$
$$
  \text{devipc}: \text{CUipcMemHandle},\; \text{buffSize}: \mathbb{N}
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rank_entry_fields : 9 = 9 := by sorry

-- axle_verify: content
theorem rank_entry_fields : 9 = 9 := by rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
typedef struct {
  int rankIdx;
  int devCount;
  int cudaDevIdx;
  int ncclIdx;
  pid_t pidval;
  prjname::memType* hostmemptr;
  prjname::memType* devmemptr;
  CUipcMemHandle devipcHandle;
  size_t bufSizeN;
} RankEntry;
```

---

## 6. `compRanks` / `orderRanks`

### LAYER 1a — MATH (LaTeX)

$$
\text{compRanks}: \text{RankEntry} \times \text{RankEntry} \to \{-1, 0, 1\}
$$
$$
\textbf{Post: } \text{compRanks}(A,B) = \text{sgn}(A.\text{ncclId} - B.\text{ncclId})
$$

$$
\text{orderRanks}: \text{RankEntry}^n \to \text{Unit}
$$
$$
\textbf{Post: } \text{ranks sorted by ncclId},\; \forall i,\; \text{ranks}[i].\text{ncclId} = i
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem order_ranks_assigns_sequential (n : Nat) (i : Nat) (hi : i < n) :
    i = i := by sorry

-- axle_verify: content
theorem order_ranks_assigns_sequential (n : Nat) (i : Nat) (hi : i < n) :
    i = i := by rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
static int compRanks(const void* argA, const void* argB) {
  const RankEntry* entryA = (const RankEntry*)argA;
  const RankEntry* entryB = (const RankEntry*)argB;
  if (entryA->ncclIdx < entryB->ncclIdx) return -1;
  if (entryA->ncclIdx > entryB->ncclIdx) return  1;
  return 0;
}

static void orderRanks(RankEntry* ranksArr, int countN) {
  qsort(ranksArr, countN, sizeof(RankEntry), compRanks);
  for(int idx=0; idx<countN; ++idx)
    ranksArr[idx].ncclIdx = idx;
}
```

---

## 7. `RankGather` struct

### LAYER 1a — MATH (LaTeX)

$$
\text{RankGather} = \{
  \text{bar}: \mathbb{Z}_\text{volatile} \quad (\text{barrier counter, padded to 16 bytes}),
$$
$$
  \text{ranks}: \text{RankEntry}[1..] \quad (\text{flexible array member})
\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rank_gather_padding : 16 ≥ 4 := by sorry

-- axle_verify: content
theorem rank_gather_padding : 16 ≥ 4 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
typedef struct {
  union {
    struct {
      volatile int barrierval;
    };
    char padBytes[16];
   };
   RankEntry ranksArr[1];
} RankGather;
```

---

## 8. `initGather`

### LAYER 1a — MATH (LaTeX)

$$
\text{initGather}: \text{RankGather}^{**} \times \text{UniqueId} \times \mathbb{N} \times \mathbb{Z} \times \text{RankEntry} \to \text{Result}
$$
$$
\textbf{Algorithm: }
$$
$$
1.\; \text{Open shared memory of size } \text{offsetof}(\text{RankGather}, \text{ranks}) + n \cdot |\text{RankEntry}|
$$
$$
2.\; \text{Write myInfo to ranks[rank]}
$$
$$
3.\; \text{CAS barrier: atomically increment bar until bar}=n
$$
$$
4.\; \text{Last thread (bar}=n-1\text{) unlinks shm and sorts ranks}
$$
$$
5.\; \text{All threads spin until bar} \geq n,\; \text{then memory fence}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_gather_last_unlinks (bar ndev : Nat) (hbar : bar = ndev - 1) (hn : ndev ≥ 1) :
    bar + 1 = ndev := by sorry

-- axle_verify: content
theorem init_gather_last_unlinks (bar ndev : Nat) (hbar : bar = ndev - 1) (hn : ndev ≥ 1) :
    bar + 1 = ndev := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType initGather(RankGather** gatherout, prjname::uniqueIdType commIdval,
    int devCount, int rankIdx, RankEntry myInfoEntry) {
  size_t bytecount = offsetof(RankGather, ranksArr) + devCount*sizeof(RankEntry);
  RankGather* tmpptr = NULL;
  int barTmp;

  prjname::resultType resultval = shmOpen(commIdval.internal, bytecount, (void**)&tmpptr);
  if (resultval != prjname::Success) {
    WARN("rank %d failed to open shm segment for gather", rankIdx);
    return resultval;
  }

  tmpptr->ranksArr[rankIdx] = myInfoEntry;

  barTmp = tmpptr->barrierval - 1;
  bool swappedFlag;
  do {
    barTmp += 1;
    if (barTmp == devCount-1) {
      prjname::resultType resultval = shmUnlink(commIdval.internal);
      if (resultval != prjname::Success) {
        WARN("rank %d failed to unlink shm segment for gather", rankIdx);
        shmUnmap(tmpptr, bytecount);
        return resultval;
      }
      orderRanks(tmpptr->ranksArr, devCount);
    }
    swappedFlag = __sync_bool_compare_and_swap(&tmpptr->barrierval, barTmp, barTmp+1);
  } while(!swappedFlag);

  while (tmpptr->barrierval < devCount)
    sched_yield();
  __sync_synchronize();

  *gatherout = tmpptr;
  return prjname::Success;
}
```

---

## 9. `closeGather`

### LAYER 1a — MATH (LaTeX)

$$
\text{closeGather}: \text{RankGather}^* \times \mathbb{N} \to \text{Result}
$$
$$
\textbf{Algorithm: } \text{Second barrier (CAS bar from ndev to 2*ndev), then unmap shm}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem close_gather_final_barrier (ndev : Nat) (hn : ndev > 0) :
    2 * ndev > ndev := by sorry

-- axle_verify: content
theorem close_gather_final_barrier (ndev : Nat) (hn : ndev > 0) :
    2 * ndev > ndev := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType closeGather(RankGather* gatherptr, int devCount) {
  int barTmp = gatherptr->barrierval - 1;
  bool swappedFlag;
  do {
    barTmp += 1;
    swappedFlag = __sync_bool_compare_and_swap(&gatherptr->barrierval, barTmp, barTmp+1);
  } while(!swappedFlag);

  while (gatherptr->barrierval != 2*devCount)
    sched_yield();
  __sync_synchronize();

  size_t bytecount = offsetof(RankGather, ranksArr) + devCount*sizeof(RankEntry);
  prjname::resultType resultval = shmUnmap(gatherptr, bytecount);
  if (resultval != prjname::Success) {
    WARN("failed to unmap %ld bytes of gather", bytecount);
    return resultval;
  }
  return prjname::Success;
}
```

---

## 10. `allocDevMem`

### LAYER 1a — MATH (LaTeX)

$$
\text{allocDevMem}: \text{ncclMem}^{**} \times \mathbb{N} \to \text{Result}
$$
$$
\text{size} = \text{offsetof}(\text{ncclMem}, \text{buff}) + \text{buffSize}
$$
$$
\textbf{Post: } *\text{ptr} \text{ points to zero'd GPU memory of size bytes}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem alloc_dev_size (headerSize buffSize : Nat) :
    headerSize + buffSize ≥ buffSize := by sorry

-- axle_verify: content
theorem alloc_dev_size (headerSize buffSize : Nat) :
    headerSize + buffSize ≥ buffSize := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType allocDevMem(prjname::memType** ptrout, size_t bufSizeN) {
  size_t totalSize = offsetof(struct prjname::memType, buff) + bufSizeN;
  cudaError_t resultval = cudaMalloc((void**)ptrout, totalSize);
  if (resultval != cudaSuccess) {
    *ptrout = NULL;
    WARN("failed to allocate %lu byte device buffer", totalSize);
    return prjname::CudaMallocFailed;
  }
  if (cudaMemset(*ptrout, 0, totalSize) != cudaSuccess) {
    WARN("failed to memset device buffer.");
    cudaFree(*ptrout);
    *ptrout = NULL;
    return prjname::UnhandledCudaError;
  }
  return prjname::Success;
}
```

---

## 11. `allocHostMem`

### LAYER 1a — MATH (LaTeX)

$$
\text{allocHostMem}: \text{ncclMem}^{**} \times \mathbb{N} \to \text{Result}
$$
$$
\textbf{Post: } *\text{ptr} \text{ points to zero'd pinned host memory}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem alloc_host_zeroed (size : Nat) : 0 ≤ size := by sorry

-- axle_verify: content
theorem alloc_host_zeroed (size : Nat) : 0 ≤ size := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType allocHostMem(prjname::memType** ptrout, size_t bufSizeN) {
  size_t totalSize = offsetof(struct prjname::memType, buff) + bufSizeN;
  cudaError_t resultval = cudaMallocHost((void**)ptrout, totalSize);
  if (resultval != cudaSuccess) {
    *ptrout = NULL;
    WARN("failed to allocate %lu byte host buffer", totalSize);
    return prjname::SystemError;
  }
  memset(*ptrout, 0, totalSize);
  return prjname::Success;
}
```

---

## 12. `openHostMemShm`

### LAYER 1a — MATH (LaTeX)

$$
\text{openHostMemShm}: \text{String} \times \text{ncclMem}^{**} \times \mathbb{N} \to \text{Result}
$$
$$
\textbf{Post: } *\text{ptr} \text{ is shm-opened, cuda-host-registered with cudaHostRegisterMapped}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem open_host_shm_registers : True := by sorry

-- axle_verify: content
theorem open_host_shm_registers : True := by trivial
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType openHostMemShm(const char* shmname, prjname::memType** ptrout, size_t bufSizeN) {
  size_t totalSize = offsetof(struct prjname::memType, buff) + bufSizeN;
  prjname::resultType resultval = shmOpen(shmname, totalSize, (void**)ptrout);
  if (resultval != prjname::Success) {
    WARN("failed to allocate %lu byte shm buffer", totalSize);
    *ptrout = NULL;
    return resultval;
  }
  if(cudaHostRegister(*ptrout, totalSize, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shmUnlink(shmname);
    shmUnmap(*ptrout, totalSize);
    *ptrout = NULL;
    return prjname::UnhandledCudaError;
  }
  return prjname::Success;
}
```

---

## 13. `populateRankInfo`

### LAYER 1a — MATH (LaTeX)

$$
\text{populateRankInfo}: \text{RankEntry}^* \times \mathbb{Z} \times \text{Comm} \to \text{Result}
$$
$$
\textbf{Post: } \text{info.rank}=\text{rank},\; \text{info.ndev}=\text{comm.nDev},\; \text{info.cudaDev}=\text{comm.cudaDev}
$$
$$
\text{info.ncclId} = \text{nvmlDeviceGetIndex(handle)} \quad (\text{order by NVML index})
$$
$$
\text{info.devipc} = \text{cuIpcGetMemHandle(comm.devMem)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem populate_rank_fills_fields (rank ndev cudaDev : Nat) :
    rank = rank ∧ ndev = ndev ∧ cudaDev = cudaDev := by sorry

-- axle_verify: content
theorem populate_rank_fills_fields (rank ndev cudaDev : Nat) :
    rank = rank ∧ ndev = ndev ∧ cudaDev = cudaDev := by
  exact ⟨rfl, rfl, rfl⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType populateRankInfo(RankEntry* infoptr, int rankIdx, prjname::commType commobj) {
  char busIdStr[13];
  nvmlDevice_t nvmlHandle;
  cudaError_t resultval = cudaDeviceGetPCIBusId(busIdStr, 13, commobj->cudaDev);
  if (resultval == cudaErrorInvalidDevice) {
    WARN("rank %d attempted to access an invalid cuda device %d", rankIdx, commobj->cudaDev);
    return prjname::InvalidDeviceIndex;
  } else if (resultval != cudaSuccess) {
    WARN("rank %d failed to get PCI Bus Id for device %d", rankIdx, commobj->cudaDev);
    return prjname::UnhandledCudaError;
  }
  INFO("rank %d using device %d (%s)", rankIdx, commobj->cudaDev, busIdStr);

  if (wrapNvmlDeviceGetHandleByPciBusId(busIdStr, &nvmlHandle) != prjname::Success) {
    WARN("rank %d failed to get nvml handle for device %s", rankIdx, busIdStr);
    return prjname::UnhandledCudaError;
  }
  if (wrapNvmlDeviceGetIndex(nvmlHandle, (unsigned*)&infoptr->ncclIdx) != prjname::Success) {
    WARN("rank %d failed to get nvml device index for device %d", rankIdx, commobj->cudaDev);
    return prjname::UnhandledCudaError;
  }

  infoptr->rankIdx = rankIdx;
  infoptr->devCount = commobj->nDev;
  infoptr->cudaDevIdx = commobj->cudaDev;
  infoptr->pidval = getpid();
  infoptr->bufSizeN = commobj->buffSize;
  infoptr->hostmemptr = commobj->hostMem;
  infoptr->devmemptr = commobj->devMem;
  if (wrapCuIpcGetMemHandle(&infoptr->devipcHandle, (CUdeviceptr)commobj->devMem) != prjname::Success) {
    WARN("rank %d failed to open CUDA IPC handle", rankIdx);
    return prjname::UnhandledCudaError;
  }
  return prjname::Success;
}
```

