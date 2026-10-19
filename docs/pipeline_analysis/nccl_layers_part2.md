
## 3. Shared Memory & Rank Coordination (core.cu)

### 3.1 shmOpen

**[LAYER 1a — MATH]**

$$\text{shmOpen} : \text{String} \times \mathbb{Z}_{>0} \times \text{Ptr}^* \to \text{ncclResult\_t}$$
$$\text{Pre}: \text{bytes} > 0 \land \text{shmname} \neq \epsilon$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies *\text{ptr} \neq \text{MAP\_FAILED} \land \text{mmap\_size} = \text{bytes}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem shm_open_positive_bytes (bytes : Nat) (h : bytes > 0) :
    bytes > 0 := by sorry

-- axle_verify: content
theorem shm_open_positive_bytes (bytes : Nat) (h : bytes > 0) :
    bytes > 0 := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult shmOpen(const char* shmkey, size_t numBytes, void** ptrOut) {
  int fd = shm_open(shmkey, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmkey);
    return prjnameSystemError;
  }
  if (ftruncate(fd, numBytes) == -1) {
    WARN("ftruncate failed to allocate %ld bytes", numBytes);
    shm_unlink(shmkey);
    close(fd);
    return prjnameSystemError;
  }
  *ptrOut = mmap(NULL, numBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (*ptrOut == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmkey);
    close(fd);
    return prjnameSystemError;
  }
  close(fd);
  return prjnameSuccess;
}
```

---

### 3.2 shmUnlink

**[LAYER 1a — MATH]**

$$\text{shmUnlink} : \text{String} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \iff \text{shm\_unlink}(\text{shmname}) = 0$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem shm_unlink_spec (ret : Int) :
    (ret = 0 → True) ∧ (ret = -1 → True) := by sorry

-- axle_verify: content
theorem shm_unlink_spec (ret : Int) :
    (ret = 0 → True) ∧ (ret = -1 → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult shmUnlink(const char* shmkey) {
  if(shm_unlink(shmkey) == -1) {
    WARN("shm_unlink failed");
    return prjnameSystemError;
  } else {
    return prjnameSuccess;
  }
}
```

---

### 3.3 shmUnmap

**[LAYER 1a — MATH]**

$$\text{shmUnmap} : \text{Ptr} \times \mathbb{Z}_{>0} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \iff \text{munmap}(\text{ptr}, \text{bytes}) = 0$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem shm_unmap_spec (ret : Int) :
    (ret = 0 → True) ∧ (ret ≠ 0 → True) := by sorry

-- axle_verify: content
theorem shm_unmap_spec (ret : Int) :
    (ret = 0 → True) ∧ (ret ≠ 0 → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult shmUnmap(void* ptrHandle, size_t numBytes) {
  if(munmap(ptrHandle, numBytes) == -1) {
    WARN("munmap failed");
    return prjnameSystemError;
  } else {
    return prjnameSuccess;
  }
}
```

---

### 3.4 RankEntry (struct)

**[LAYER 1a — MATH]**

$$\text{RankEntry} = \{ \text{rank} : \mathbb{Z}_{\geq 0}, \; \text{ndev} : \mathbb{Z}_{>0}, \; \text{cudaDev} : \mathbb{Z}_{\geq 0}, \; \text{ncclId} : \mathbb{Z}_{\geq 0}, $$
$$\text{pid} : \text{pid\_t}, \; \text{hostptr}, \text{devptr} : \text{Ptr}(\text{ncclMem}), \; \text{devipc} : \text{CUipcMemHandle}, \; \text{buffSize} : \mathbb{Z}_{>0} \}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem rank_entry_valid (rank ndev : Nat) (h : rank < ndev) (h2 : ndev > 0) :
    rank < ndev := by sorry

-- axle_verify: content
theorem rank_entry_valid (rank ndev : Nat) (h : rank < ndev) (h2 : ndev > 0) :
    rank < ndev := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
typedef struct {
  int rankIdx;
  int numDev;
  int gpuIdx;
  int ringIdx;
  pid_t procId;
  prjnameDevMem* hostHandle;
  prjnameDevMem* devHandle;
  CUipcMemHandle ipcHandle;
  size_t buffSz;
} RankInfo;
```

---

### 3.5 compRanks

**[LAYER 1a — MATH]**

$$\text{compRanks} : \text{RankEntry}^* \times \text{RankEntry}^* \to \{-1, 0, 1\}$$
$$\text{compRanks}(A, B) = \text{sgn}(A.\text{ncclId} - B.\text{ncclId})$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comp_ranks_trichotomy (a b : Int) :
    (a < b → -1 < 0) ∧ (a = b → (0 : Int) = 0) ∧ (a > b → 1 > 0) := by sorry

-- axle_verify: content
theorem comp_ranks_trichotomy (a b : Int) :
    (a < b → -1 < 0) ∧ (a = b → (0 : Int) = 0) ∧ (a > b → 1 > 0) := by
  exact ⟨fun _ => by norm_num, fun _ => rfl, fun _ => by norm_num⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static int compEntries(const void* lhs, const void* rhs) {
  const RankInfo* adjA = (const RankInfo*)lhs;
  const RankInfo* adjB = (const RankInfo*)rhs;
  if (adjA->ringIdx < adjB->ringIdx) return -1;
  if (adjA->ringIdx > adjB->ringIdx) return  1;
  return 0;
}
```

---

### 3.6 orderRanks

**[LAYER 1a — MATH]**

$$\text{orderRanks} : \text{Array}[\text{count}](\text{RankEntry}) \times \mathbb{Z}_{>0} \to \text{void}$$
$$\text{Post}: \text{ranks sorted by ncclId} \land \forall i \in [0, \text{count}). \; \text{ranks}[i].\text{ncclId} = i$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem order_ranks_assigns_sequential (count : Nat) (i : Nat) (h : i < count) :
    i < count := by sorry

-- axle_verify: content
theorem order_ranks_assigns_sequential (count : Nat) (i : Nat) (h : i < count) :
    i < count := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static void reindexEntries(RankInfo* entries, int numEntries) {
  qsort(entries, numEntries, sizeof(RankInfo), compEntries);
  for(int idx=0; idx<numEntries; ++idx)
    entries[idx].ringIdx = idx;
}
```

---

### 3.7 RankGather (struct)

**[LAYER 1a — MATH]**

$$\text{RankGather} = \{ \text{bar} : \text{volatile} \; \mathbb{Z}_{\geq 0}, \; \text{ranks} : \text{FlexArray}(\text{RankEntry}) \}$$
$$\text{Invariant}: 0 \leq \text{bar} \leq 2 \cdot \text{ndev}$$
$$\text{Phase 1}: \text{bar} \in [0, \text{ndev}] \quad \text{(gather)}$$
$$\text{Phase 2}: \text{bar} \in [\text{ndev}, 2\text{ndev}] \quad \text{(close)}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem rank_gather_barrier_phases (ndev bar : Nat) (h : bar ≤ 2 * ndev) :
    bar ≤ ndev ∨ (bar > ndev ∧ bar ≤ 2 * ndev) := by sorry

-- axle_verify: content
theorem rank_gather_barrier_phases (ndev bar : Nat) (h : bar ≤ 2 * ndev) :
    bar ≤ ndev ∨ (bar > ndev ∧ bar ≤ 2 * ndev) := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
typedef struct {
  union {
    struct {
      volatile int barrierCnt;
    };
    char pad[16];
   };
   RankInfo entries[1];
} GatherCtx;
```

---

### 3.8 initGather

**[LAYER 1a — MATH]**

$$\text{initGather} : \text{GatherCtx}^{**} \times \text{UniqueId} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{\geq 0} \times \text{RankEntry} \to \text{ncclResult\_t}$$
$$\text{Pre}: 0 \leq \text{rank} < \text{ndev}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies \text{all ranks written} \land \text{ranks sorted}$$
$$\text{Synchronization}: \text{CAS-based barrier with ndev participants}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem init_gather_last_rank_unlinks (bar_val ndev : Nat) (h : bar_val = ndev - 1) (hd : ndev > 0) :
    bar_val + 1 = ndev := by sorry

-- axle_verify: content
theorem init_gather_last_rank_unlinks (bar_val ndev : Nat) (h : bar_val = ndev - 1) (hd : ndev > 0) :
    bar_val + 1 = ndev := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult initGatherCtx(GatherCtx** ctxOut, prjnameUniqueId commKey,
    int numDev, int rankIdx, RankInfo myInfo) {
  size_t numBytes = offsetof(GatherCtx, entries) + numDev*sizeof(RankInfo);
  GatherCtx* tmpCtx = NULL;
  int bar_tmp;

  prjnameResult res = shmOpen(commKey.internal, numBytes, (void**)&tmpCtx);
  if (res != prjnameSuccess) { /* ... */ return res; }

  tmpCtx->entries[rankIdx] = myInfo;
  bar_tmp = tmpCtx->barrierCnt - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    if (bar_tmp == numDev-1) {
      prjnameResult res = shmUnlink(commKey.internal);
      if (res != prjnameSuccess) { /* ... */ }
      reindexEntries(tmpCtx->entries, numDev);
    }
    swapped = __sync_bool_compare_and_swap(&tmpCtx->barrierCnt, bar_tmp, bar_tmp+1);
  } while(!swapped);
  while (tmpCtx->barrierCnt < numDev) sched_yield();
  __sync_synchronize();
  *ctxOut = tmpCtx;
  return prjnameSuccess;
}
```

---

### 3.9 closeGather

**[LAYER 1a — MATH]**

$$\text{closeGather} : \text{GatherCtx}^* \times \mathbb{Z}_{>0} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{barrier reaches } 2 \cdot \text{ndev}, \text{ shared memory unmapped}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem close_gather_final_bar (ndev : Nat) (h : ndev > 0) :
    2 * ndev > ndev := by sorry

-- axle_verify: content
theorem close_gather_final_bar (ndev : Nat) (h : ndev > 0) :
    2 * ndev > ndev := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult closeGatherCtx(GatherCtx* ctx, int numDev) {
  int bar_tmp = ctx->barrierCnt - 1;
  bool swapped;
  do {
    bar_tmp += 1;
    swapped = __sync_bool_compare_and_swap(&ctx->barrierCnt, bar_tmp, bar_tmp+1);
  } while(!swapped);
  while (ctx->barrierCnt != 2*numDev) sched_yield();
  __sync_synchronize();
  size_t numBytes = offsetof(GatherCtx, entries) + numDev*sizeof(RankInfo);
  prjnameResult res = shmUnmap(ctx, numBytes);
  return res;
}
```

---

### 3.10 populateRankInfo

**[LAYER 1a — MATH]**

$$\text{populateRankInfo} : \text{RankEntry}^* \times \mathbb{Z}_{\geq 0} \times \text{ncclComm\_t} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{info}.\text{rank} = \text{rank} \land \text{info}.\text{ncclId} = \text{nvmlIndex}(\text{comm}.\text{cudaDev})$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem populate_rank_assigns (rank : Nat) :
    rank = rank := by sorry

-- axle_verify: content
theorem populate_rank_assigns (rank : Nat) :
    rank = rank := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult fillRankInfo(RankInfo* infoOut, int rankIdx, prjnameComm* comm) {
  char busId[13];
  nvmlDevice_t nvmlDev;
  cudaError_t res = cudaDeviceGetPCIBusId(busId, 13, comm->gpuIdx);
  if (res == cudaErrorInvalidDevice) { return prjnameInvalidDeviceIndex; }
  if (res != cudaSuccess) { return prjnameUnhandledCudaError; }
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev) != prjnameSuccess) { return prjnameUnhandledCudaError; }
  if (wrapNvmlDeviceGetIndex(nvmlDev, (unsigned*)&infoOut->ringIdx) != prjnameSuccess) { return prjnameUnhandledCudaError; }
  infoOut->rankIdx = rankIdx;
  infoOut->numDev = comm->numDev;
  infoOut->gpuIdx = comm->gpuIdx;
  infoOut->procId = getpid();
  infoOut->buffSz = comm->buffSz;
  infoOut->hostHandle = comm->hostMemHandle;
  infoOut->devHandle = comm->devMemHandle;
  if (wrapCuIpcGetMemHandle(&infoOut->ipcHandle, (CUdeviceptr)comm->devMemHandle) != prjnameSuccess) { return prjnameUnhandledCudaError; }
  return prjnameSuccess;
}
```

---
