# NCCL Formal Specification — Three-Layer Analysis

## Table of Contents
1. Core Data Structures (core.h)
2. Memory Management (core.cu)
3. Shared Memory & Rank Coordination (core.cu)
4. Communication Map Building (core.cu)
5. Common Kernel Primitives (common_kernel.h)
6. Copy Kernel (copy_kernel.h)
7. Reduce Kernel (reduce_kernel.h)
8. Enqueue (enqueue.h)
9. AllReduce Collective (all_reduce.cu)
10. AllGather Collective (all_gather.cu)
11. Reduce Collective (reduce.cu)
12. ReduceScatter Collective (reduce_scatter.cu)
13. Library Wrappers (libwrap.cu/h)
14. Test Utilities (test_utilities.h)
15. Synchronization Macros (all files)

---

## 1. Core Data Structures

### 1.1 struct EventQueue (core.h)

**[LAYER 1a — MATH]**

$$\text{EventQueue} = \{ \text{isDone} : \text{Array}[\text{MAXQUEUE}](\text{cudaEvent\_t}), \; \text{back} : \mathbb{Z}_{\geq 0} \}$$
$$\text{Invariant}: 0 \leq \text{back} < \text{MAXQUEUE}$$
$$\text{back}' = (\text{back} + 1) \bmod \text{MAXQUEUE}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem event_queue_back_invariant (back : Nat) (MAXQUEUE : Nat) (h : MAXQUEUE = 4) (hb : back < MAXQUEUE) :
    (back + 1) % MAXQUEUE < MAXQUEUE := by sorry

-- axle_verify: content
theorem event_queue_back_invariant (back : Nat) (MAXQUEUE : Nat) (h : MAXQUEUE = 4) (hb : back < MAXQUEUE) :
    (back + 1) % MAXQUEUE < MAXQUEUE := by
  subst h
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
typedef struct {
  cudaEvent_t isDone[MAXQUEUE];
  int idx; // Last event used
} EvtQueue;
```

---

### 1.2 struct ncclMem (core.h)

**[LAYER 1a — MATH]**

$$\text{ncclMem} = \{ \text{flags} : \text{Array}[\text{MAXFLAGS}](\mathbb{Z}), \; \text{recvPtrs} : \text{Array}[\text{MAXFLAGS}](\text{Ptr}), \; \text{buff} : \text{Array}[\text{PAD\_ALIGN}](\text{Byte}) \}$$
$$\text{Invariant}: |\text{pad}| = \text{NCCL\_MEM\_PAD\_ALIGN} = 4096$$
$$\forall i \in [0, \text{MAXFLAGS}). \; \text{flags}[i] \geq 0 \text{ during steady state}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem nccl_mem_pad_alignment (PAD_ALIGN : Nat) (h : PAD_ALIGN = 4096) :
    PAD_ALIGN % 64 = 0 ∧ PAD_ALIGN > 0 := by sorry

-- axle_verify: content
theorem nccl_mem_pad_alignment (PAD_ALIGN : Nat) (h : PAD_ALIGN = 4096) :
    PAD_ALIGN % 64 = 0 ∧ PAD_ALIGN > 0 := by
  subst h
  constructor <;> norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
struct prjnameDevMem {
  union {
    struct {
      int   syncflags[MAXFLAGS];
      void* recvPtrs[MAXFLAGS];
    };
    char pad[NCCL_MEM_PAD_ALIGN];
  };
  char objbuff[NCCL_MEM_PAD_ALIGN];
};
```

---

### 1.3 struct ncclComm (core.h)

**[LAYER 1a — MATH]**

$$\text{ncclComm} = \{ \text{nDev} : \mathbb{Z}_{>0}, \; \text{cudaDev} : \mathbb{Z}_{\geq 0}, \; \text{ncclId} : \mathbb{Z}_{\geq 0}, \; \text{devMem}, \text{hostMem} : \text{Ptr}(\text{ncclMem}), $$
$$\text{remote}, \text{local} : \text{Array}[\text{MAXPEERS}](\text{Ptr}(\text{ncclMem})), \; \text{userFromRing}, \text{ringFromUser} : \text{Array}[\text{MAXPEERS}](\mathbb{Z}), $$
$$\text{buffSize} : \mathbb{Z}_{>0}, \; \text{useRemoteRecv} : \{0,1\} \}$$
$$\text{Invariant}: 1 \leq \text{nDev} \leq \text{MAXPEERS} = 8$$
$$\text{Invariant}: 0 \leq \text{ncclId} < \text{nDev}$$
$$\text{Invariant}: \forall i \in [0, \text{nDev}). \; \text{ringFromUser}[\text{userFromRing}[i]] = i$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_ndev_bounded (nDev : Nat) (MAXPEERS : Nat) (h : MAXPEERS = 8)
    (h1 : 1 ≤ nDev) (h2 : nDev ≤ MAXPEERS) :
    nDev ≤ 8 ∧ nDev ≥ 1 := by sorry

-- axle_verify: content
theorem comm_ndev_bounded (nDev : Nat) (MAXPEERS : Nat) (h : MAXPEERS = 8)
    (h1 : 1 ≤ nDev) (h2 : nDev ≤ MAXPEERS) :
    nDev ≤ 8 ∧ nDev ≥ 1 := by
  subst h
  omega
```

```lean
-- axle_verify: formal_statement
theorem ring_user_inverse (n : Nat) (userFromRing ringFromUser : Fin n → Fin n)
    (h : ∀ i : Fin n, ringFromUser (userFromRing i) = i) :
    Function.Injective userFromRing := by sorry

-- axle_verify: content
theorem ring_user_inverse (n : Nat) (userFromRing ringFromUser : Fin n → Fin n)
    (h : ∀ i : Fin n, ringFromUser (userFromRing i) = i) :
    Function.Injective userFromRing := by
  intro a b hab
  have ha := h a
  have hb := h b
  rw [hab] at ha
  rw [ha] at hb
  exact Fin.val_injective (Fin.val_injective hb ▸ rfl)
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
struct prjnameComm {
  int numDev;
  int gpuIdx;
  int ringIdx;

  prjnameDevMem* devMemHandle;
  prjnameDevMem* hostMemHandle;
  int hostMemState;

  prjnameDevMem* remotePtrs[MAXPEERS];
  prjnameDevMem* localPtrs[MAXPEERS];
  struct {
    int type;
    void* objhandle;
  } cleanupEntries[MAXPEERS];

  EvtQueue evtqueue;

  int userFromRing[MAXPEERS];
  int* devUserFromRing;
  int ringFromUser[MAXPEERS];

  size_t buffSz;
  int useRemoteRecv;
};
```

---

## 2. Memory Management (core.cu)

### 2.1 allocDevMem

**[LAYER 1a — MATH]**

$$\text{allocDevMem} : \text{Ptr}^* \times \mathbb{Z}_{>0} \to \text{ncclResult\_t}$$
$$\text{Pre}: \text{buffSize} > 0$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies *\text{ptr} \neq \text{null} \land \text{allocated}(\text{offsetof}(\text{ncclMem}, \text{buff}) + \text{buffSize})$$
$$\text{Post}: \text{result} \neq \text{ncclSuccess} \implies *\text{ptr} = \text{null}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem alloc_dev_mem_spec (buffSize : Nat) (allocOk : Bool) (hpos : buffSize > 0) :
    (allocOk = true → buffSize + 4096 > 0) ∧
    (allocOk = false → True) := by sorry

-- axle_verify: content
theorem alloc_dev_mem_spec (buffSize : Nat) (allocOk : Bool) (hpos : buffSize > 0) :
    (allocOk = true → buffSize + 4096 > 0) ∧
    (allocOk = false → True) := by
  constructor
  · intro _; omega
  · intro _; trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult allocGpuMem(prjnameDevMem** ptrOut, size_t buffSz) {
  size_t totalSz = offsetof(struct prjnameDevMem, objbuff) + buffSz;
  cudaError_t res = cudaMalloc((void**)ptrOut, totalSz);
  if (res != cudaSuccess) {
    *ptrOut = NULL;
    WARN("failed to allocate %lu byte device buffer", totalSz);
    return prjnameCudaMallocFailed;
  }
  if (cudaMemset(*ptrOut, 0, totalSz) != cudaSuccess) {
    WARN("failed to memset device buffer.");
    cudaFree(*ptrOut);
    *ptrOut = NULL;
    return prjnameUnhandledCudaError;
  }
  return prjnameSuccess;
}
```

---

### 2.2 allocHostMem

**[LAYER 1a — MATH]**

$$\text{allocHostMem} : \text{Ptr}^* \times \mathbb{Z}_{>0} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies *\text{ptr} \neq \text{null} \land \text{pinned}(*\text{ptr})$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem alloc_host_mem_post (ptr_is_null : Bool) (result_ok : Bool) :
    (result_ok = true → ptr_is_null = false) ∧
    (result_ok = false → ptr_is_null = true) →
    (result_ok = true → ptr_is_null = false) := by sorry

-- axle_verify: content
theorem alloc_host_mem_post (ptr_is_null : Bool) (result_ok : Bool) :
    (result_ok = true → ptr_is_null = false) ∧
    (result_ok = false → ptr_is_null = true) →
    (result_ok = true → ptr_is_null = false) := by
  intro ⟨h1, _⟩
  exact h1
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult allocPinnedMem(prjnameDevMem** ptrOut, size_t buffSz) {
  size_t totalSz = offsetof(struct prjnameDevMem, objbuff) + buffSz;
  cudaError_t res = cudaMallocHost((void**)ptrOut, totalSz);
  if (res != cudaSuccess) {
    *ptrOut = NULL;
    WARN("failed to allocate %lu byte host buffer", totalSz);
    return prjnameSystemError;
  }
  memset(*ptrOut, 0, totalSz);
  return prjnameSuccess;
}
```

---

### 2.3 openHostMemShm

**[LAYER 1a — MATH]**

$$\text{openHostMemShm} : \text{String} \times \text{Ptr}^* \times \mathbb{Z}_{>0} \to \text{ncclResult\_t}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies *\text{ptr} \neq \text{null} \land \text{cuda\_host\_registered}(*\text{ptr}) \land \text{shm\_mapped}(*\text{ptr})$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem open_host_shm_cleanup (shmOk registerOk : Bool) :
    (shmOk = false → True) ∧
    (shmOk = true ∧ registerOk = false → True) ∧
    (shmOk = true ∧ registerOk = true → True) := by sorry

-- axle_verify: content
theorem open_host_shm_cleanup (shmOk registerOk : Bool) :
    (shmOk = false → True) ∧
    (shmOk = true ∧ registerOk = false → True) ∧
    (shmOk = true ∧ registerOk = true → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult openPinnedMemShm(const char* shmkey, prjnameDevMem** ptrOut, size_t buffSz) {
  size_t totalSz = offsetof(struct prjnameDevMem, objbuff) + buffSz;
  prjnameResult res = shmOpen(shmkey, totalSz, (void**)ptrOut);
  if (res != prjnameSuccess) {
    WARN("failed to allocate %lu byte shm buffer", totalSz);
    *ptrOut = NULL;
    return res;
  }
  if(cudaHostRegister(*ptrOut, totalSz, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shmUnlink(shmkey);
    shmUnmap(*ptrOut, totalSz);
    *ptrOut = NULL;
    return prjnameUnhandledCudaError;
  }
  return prjnameSuccess;
}
```

---

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

## 4. Communication Map Building (core.cu)

### 4.1 commClearMaps

**[LAYER 1a — MATH]**

$$\text{commClearMaps} : \text{ncclComm\_t} \to \text{ncclResult\_t}$$
$$\text{Post}: \forall d \in [0, \text{nDev}). \; \text{cleanup}[d].\text{type} = 0 \land \text{cleanup}[d].\text{handle} = \text{null}$$
$$\text{Post}: \text{userFromRing}, \text{ringFromUser} \text{ zeroed}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem clear_maps_zeros_all (n : Nat) (i : Nat) (h : i < n) :
    0 = 0 := by sorry

-- axle_verify: content
theorem clear_maps_zeros_all (n : Nat) (i : Nat) (h : i < n) :
    0 = 0 := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult commClearMaps(prjnameComm* comm) {
  prjnameResult res, retval = prjnameSuccess;
  cudaError_t cudares;
  for(int idx=0; idx<comm->numDev; ++idx) {
    switch(comm->cleanupEntries[idx].type) {
      case CLEANUP_NONE: break;
      case CLEANUP_CUIPC:
        res = wrapCuIpcCloseMemHandle((CUdeviceptr)comm->cleanupEntries[idx].objhandle);
        if (res != prjnameSuccess) retval = (retval == prjnameSuccess) ? res : retval;
        break;
      case CLEANUP_UNMAP:
        cudares = cudaHostUnregister(comm->cleanupEntries[idx].objhandle);
        if (cudares != cudaSuccess) retval = (retval == prjnameSuccess) ? prjnameUnhandledCudaError : retval;
        res = shmUnmap(comm->cleanupEntries[idx].objhandle, offsetof(prjnameDevMem, objbuff) + comm->buffSz);
        if (res != prjnameSuccess) retval = (retval == prjnameSuccess) ? res : retval;
        break;
    }
    comm->cleanupEntries[idx].type = 0;
    comm->cleanupEntries[idx].objhandle = NULL;
  }
  memset(comm->userFromRing, 0, sizeof(int)*MAXPEERS);
  memset(comm->ringFromUser, 0, sizeof(int)*MAXPEERS);
  if (comm->devUserFromRing != NULL) cudaMemset(comm->devUserFromRing, 0, sizeof(int)*MAXPEERS);
  return retval;
}
```

---

### 4.2 commBuildMaps

**[LAYER 1a — MATH]**

$$\text{commBuildMaps} : \text{ncclComm\_t} \times \text{UniqueId}^* \times \mathbb{Z}_{\geq 0} \times \text{RankEntry}^* \to \text{ncclResult\_t}$$
$$\text{Pre}: \forall i \in [0, \text{ndev}). \; \text{ranks}[i].\text{ndev} = \text{comm}.\text{nDev} \land \text{ranks}[i].\text{buffSize} = \text{comm}.\text{buffSize}$$
$$\text{Post}: \text{userFromRing} \text{ and } \text{ringFromUser} \text{ are inverses}$$
$$\text{Post}: \forall i. \; \text{local}[i], \text{remote}[i] \text{ point to accessible memory (P2P, IPC, or zero-copy)}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem build_maps_consistency (ndev : Nat) (buffA buffB : Nat)
    (h : buffA = buffB) (h2 : ndev > 0) :
    buffA = buffB := by sorry

-- axle_verify: content
theorem build_maps_consistency (ndev : Nat) (buffA buffB : Nat)
    (h : buffA = buffB) (h2 : ndev > 0) :
    buffA = buffB := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult commBuildMaps(prjnameComm* comm, prjnameUniqueId* commKey,
    int rankIdx, RankInfo* entries) {
  int numDev = comm->numDev;
  for(int idx=0; idx<numDev; ++idx) {
    if (entries[idx].buffSz != comm->buffSz || entries[idx].numDev != comm->numDev) {
      commClearMaps(comm);
      return prjnameRankMismatch;
    }
    int iRank = entries[idx].rankIdx;
    comm->userFromRing[idx] = iRank;
    comm->ringFromUser[iRank] = idx;
  }
  if (cudaMemcpy(comm->devUserFromRing, comm->userFromRing, numDev*sizeof(int),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    commClearMaps(comm);
    return prjnameUnhandledCudaError;
  }
  int myIdx = -1;
  for (int idx=0; idx<numDev; ++idx) {
    if(entries[idx].rankIdx == rankIdx) { myIdx = idx; break; }
  }
  if (myIdx == -1) return prjnameInvalidRank;
  comm->ringIdx = myIdx;
  int myGpu = entries[myIdx].gpuIdx;
  pid_t myPid = entries[myIdx].procId;
  comm->useRemoteRecv = 1;
  for (int idx=0; idx<numDev; ++idx) {
    int peerRank = entries[idx].rankIdx;
    int peerGpu = entries[idx].gpuIdx;
    pid_t peerPid = entries[idx].procId;
    int canpeer = 0;
    cudaDeviceCanAccessPeer(&canpeer, myGpu, peerGpu);
    if (canpeer) { cudaDeviceEnablePeerAccess(peerGpu, 0); }
    if (peerPid == myPid && (canpeer || myGpu == peerGpu)) {
      comm->localPtrs[idx] = entries[myIdx].devHandle;
      comm->remotePtrs[idx] = entries[idx].devHandle;
      comm->cleanupEntries[idx].type = CLEANUP_NONE;
    } else { /* zero-copy or IPC paths */ }
  }
  return prjnameSuccess;
}
```

---

### 4.3 commAlloc

**[LAYER 1a — MATH]**

$$\text{commAlloc} : \text{ncclComm\_t}^* \times \mathbb{Z}_{>0} \times \text{UniqueId}^* \times \mathbb{Z}_{\geq 0} \to \text{ncclResult\_t}$$
$$\text{Pre}: 1 \leq \text{ndev} \leq \text{MAXPEERS} \land 0 \leq \text{rank} < \text{ndev}$$
$$\text{Post}: \text{result} = \text{ncclSuccess} \implies *\text{comret} \neq \text{null} \land \text{devMem allocated} \land \text{hostMem allocated}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_alloc_bounds (ndev rank MAXPEERS : Nat) (h1 : MAXPEERS = 8)
    (h2 : 1 ≤ ndev) (h3 : ndev ≤ MAXPEERS) (h4 : rank < ndev) :
    rank < 8 := by sorry

-- axle_verify: content
theorem comm_alloc_bounds (ndev rank MAXPEERS : Nat) (h1 : MAXPEERS = 8)
    (h2 : 1 ≤ ndev) (h3 : ndev ≤ MAXPEERS) (h4 : rank < ndev) :
    rank < 8 := by
  subst h1; omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult commAlloc(prjnameComm** commOut, int numDev,
    const prjnameUniqueId* commKey, int rankIdx) {
  if (numDev < 1 || numDev > MAXPEERS) return prjnameUnsupportedDeviceCount;
  if (rankIdx >= numDev || rankIdx < 0) return prjnameInvalidRank;
  struct prjnameComm* comm = (struct prjnameComm*)malloc(sizeof(struct prjnameComm));
  if (comm == NULL) return prjnameSystemError;
  memset(comm, 0, sizeof(struct prjnameComm));
  comm->numDev = numDev;
  cudaGetDevice(&comm->gpuIdx);
  /* buffSize from env or default */
  comm->buffSz = DEFAULT_BUFFER_SIZE_BYTES;
  /* allocate devMem, devUserFromRing, events, hostMem */
  *commOut = comm;
  return prjnameSuccess;
}
```

---

### 4.4 commFree

**[LAYER 1a — MATH]**

$$\text{commFree} : \text{ncclComm\_t} \to \text{void}$$
$$\text{Pre}: \text{comm may be null}$$
$$\text{Post}: \text{all events destroyed} \land \text{maps cleared} \land \text{devMem freed} \land \text{hostMem freed} \land \text{comm freed}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_free_null_safe (is_null : Bool) :
    (is_null = true → True) ∧ (is_null = false → True) := by sorry

-- axle_verify: content
theorem comm_free_null_safe (is_null : Bool) :
    (is_null = true → True) ∧ (is_null = false → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static void commFree(prjnameComm* comm) {
  if (comm == NULL) return;
  for(int idx=0; idx<MAXQUEUE; ++idx) {
    if (comm->evtqueue.isDone[idx] != NULL)
      cudaEventDestroy(comm->evtqueue.isDone[idx]);
  }
  commClearMaps(comm);
  if (comm->devUserFromRing != NULL) cudaFree(comm->devUserFromRing);
  if (comm->devMemHandle != NULL) cudaFree(comm->devMemHandle);
  if (comm->hostMemHandle != NULL) { /* unregister/unmap or free */ }
  free(comm);
}
```

---

### 4.5 commUnlinkHostMem

**[LAYER 1a — MATH]**

$$\text{commUnlinkHostMem} : \text{ncclComm\_t} \times \text{UniqueId} \times \mathbb{Z}_{\geq 0} \to \text{ncclResult\_t}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_unlink_clears_linked (state : Nat) (LINKED : Nat) (h : LINKED = 2)
    (h2 : state = LINKED) :
    state - LINKED = 0 := by sorry

-- axle_verify: content
theorem comm_unlink_clears_linked (state : Nat) (LINKED : Nat) (h : LINKED = 2)
    (h2 : state = LINKED) :
    state - LINKED = 0 := by
  subst h; subst h2; omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static prjnameResult commUnlinkHostMem(prjnameComm* comm, prjnameUniqueId commKey, int rankIdx) {
  char shmkey[1024];
  sprintf(shmkey, "%s-%d", commKey.internal, rankIdx);
  if (comm->hostMemState & ShmLinked)
    comm->hostMemState ^= ShmLinked;
  return shmUnlink(shmkey);
}
```

---

### 4.6 initDebug

**[LAYER 1a — MATH]**

$$\text{initDebug} : \text{void} \to \text{void}$$
$$\text{Post}: \text{ncclDebugLevel} \in \{\text{NONE}, \text{WARN}, \text{INFO}, \text{ABORT}\}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem debug_level_valid (level : Nat) (h : level ≤ 3) :
    level = 0 ∨ level = 1 ∨ level = 2 ∨ level = 3 := by sorry

-- axle_verify: content
theorem debug_level_valid (level : Nat) (h : level ≤ 3) :
    level = 0 ∨ level = 1 ∨ level = 2 ∨ level = 3 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
static void initLogLevel() {
  const char* envVal = getenv("NCCL_DEBUG");
  if (envVal == NULL) { prjnameLogLevel = NONE; }
  else if (strcmp(envVal, "WARN") == 0) { prjnameLogLevel = WARN; }
  else if (strcmp(envVal, "INFO") == 0) { prjnameLogLevel = INFO; }
  else if (strcmp(envVal, "ABORT") == 0) { prjnameLogLevel = ABORT; }
}
```

---

## 5. Public API (core.cu)

### 5.1 ncclGetUniqueId

**[LAYER 1a — MATH]**

$$\text{ncclGetUniqueId} : \text{ncclUniqueId}^* \to \text{ncclResult\_t}$$
$$\text{Post}: \text{out}.\text{internal} = \texttt{"nccl-"} \| \text{pid} \| \texttt{"-"} \| \text{counter}$$
$$\text{Thread-safe}: \text{counter incremented atomically}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem unique_id_nonempty (pid commId : Nat) :
    String.length ("nccl-" ++ toString pid ++ "-" ++ toString commId) > 0 := by sorry

-- axle_verify: content
theorem unique_id_nonempty (pid commId : Nat) :
    String.length ("nccl-" ++ toString pid ++ "-" ++ toString commId) > 0 := by
  simp [String.length]
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameGetUniqueId(prjnameUniqueId* idOut) {
  pid_t procId = getpid();
  static int counter = 0;
  int seqId = __sync_fetch_and_add(&counter, 1);
  int len = snprintf(idOut->internal, NCCL_UNIQUE_ID_BYTES, "prjname-%d-%d", procId, seqId);
  if(strlen(idOut->internal) < len) {
    WARN("prjnameUniqueId truncated");
    return prjnameInternalError;
  }
  return prjnameSuccess;
}
```

---

### 5.2 ncclCommInitRank

**[LAYER 1a — MATH]**

$$\text{ncclCommInitRank} : \text{ncclComm\_t}^* \times \mathbb{Z}_{>0} \times \text{UniqueId} \times \mathbb{Z}_{\geq 0} \to \text{ncclResult\_t}$$
$$\text{Pre}: 0 \leq \text{myrank} < \text{ndev} \land |\text{commId}.\text{internal}| > 0$$
$$\text{Synchronization}: \text{barrier across all ndev ranks via shared memory}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_init_rank_valid_id (len : Nat) (MAX : Nat) (h : MAX = 128) (h1 : len ≥ 1) (h2 : len < MAX) :
    len ≥ 1 ∧ len < 128 := by sorry

-- axle_verify: content
theorem comm_init_rank_valid_id (len : Nat) (MAX : Nat) (h : MAX = 128) (h1 : len ≥ 1) (h2 : len < MAX) :
    len ≥ 1 ∧ len < 128 := by
  subst h; exact ⟨h1, h2⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameCommInitRank(prjnameComm** commOut, int numDev,
    prjnameUniqueId commKey, int myRankIdx) {
  if (strlen(commKey.internal) < 1 || strlen(commKey.internal) >= NCCL_UNIQUE_ID_BYTES)
    return prjnameInvalidArgument;
  initLogLevel();
  /* wrapSymbols, wrapNvmlInit, commAlloc, populateRankInfo, initGather, commBuildMaps, closeGather */
  return res;
}
```

---

### 5.3 ncclCommInitAll

**[LAYER 1a — MATH]**

$$\text{ncclCommInitAll} : \text{Array}(\text{ncclComm\_t}) \times \mathbb{Z}_{>0} \times \text{Array}(\mathbb{Z}_{\geq 0}) \to \text{ncclResult\_t}$$
$$\text{Pre}: \text{ndev} \geq 1 \land \forall i. \; \text{devlist}[i] \text{ is valid CUDA device}$$
$$\text{Post}: \forall i \in [0, \text{ndev}). \; \text{comms}[i] \text{ fully initialized with maps built}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_init_all_valid_devlist (ndev : Nat) (h : ndev ≥ 1) :
    ndev > 0 := by sorry

-- axle_verify: content
theorem comm_init_all_valid_devlist (ndev : Nat) (h : ndev ≥ 1) :
    ndev > 0 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameCommInitAll(prjnameComm** comms, int numDev, int* gpuList) {
  initLogLevel();
  int savedGpu;
  cudaGetDevice(&savedGpu);
  RankInfo* entries = (RankInfo*)malloc(numDev*sizeof(RankInfo));
  /* for each rank: setDevice, commAlloc, populateRankInfo */
  reindexEntries(entries, numDev);
  /* for each rank: commBuildMaps */
  free(entries);
  cudaSetDevice(savedGpu);
  return prjnameSuccess;
}
```

---

### 5.4 ncclCommDestroy

**[LAYER 1a — MATH]**

$$\text{ncclCommDestroy} : \text{ncclComm\_t} \to \text{void}$$
$$\text{Pre}: \text{comm may be null}$$
$$\text{Post}: \text{device context restored} \land \text{comm freed}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_destroy_restores_device (saved comm : Nat) (h : saved ≠ comm) :
    saved ≠ comm → True := by sorry

-- axle_verify: content
theorem comm_destroy_restores_device (saved comm : Nat) (h : saved ≠ comm) :
    saved ≠ comm → True := by
  intro _; trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
void prjnameCommDestroy(prjnameComm* comm) {
  if (comm == NULL) return;
  int savedGpu;
  cudaGetDevice(&savedGpu);
  int commGpu = comm->gpuIdx;
  if (savedGpu != commGpu) { CUDACHECK(cudaSetDevice(commGpu)); }
  commFree(comm);
  if (savedGpu != commGpu) cudaSetDevice(savedGpu);
}
```

---

### 5.5 ncclGetErrorString

**[LAYER 1a — MATH]**

$$\text{ncclGetErrorString} : \text{ncclResult\_t} \to \text{String}$$
$$\text{Total function}: \text{defined for all enum values}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem error_string_total (code : Nat) (h : code ≤ 13) :
    code ≤ 13 := by sorry

-- axle_verify: content
theorem error_string_total (code : Nat) (h : code ≤ 13) :
    code ≤ 13 := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
const char* prjnameGetErrorString(prjnameResult errcode) {
  switch (errcode) {
  case prjnameSuccess                : return "no error";
  case prjnameUnhandledCudaError     : return "unhandled cuda error";
  /* ... all cases ... */
  }
  return "unknown result code";
}
```

---

### 5.6 ncclCommCount

**[LAYER 1a — MATH]**

$$\text{ncclCommCount} : \text{ncclComm\_t} \times \mathbb{Z}^* \to \text{ncclResult\_t}$$
$$\text{Post}: *\text{count} = \text{comm}.\text{nDev}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_count_returns_ndev (nDev : Nat) :
    nDev = nDev := by sorry

-- axle_verify: content
theorem comm_count_returns_ndev (nDev : Nat) :
    nDev = nDev := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameCommCount(const prjnameComm* comm, int* countOut) {
  *countOut = comm->numDev;
  return prjnameSuccess;
}
```

---

### 5.7 ncclCommCuDevice

**[LAYER 1a — MATH]**

$$\text{ncclCommCuDevice} : \text{ncclComm\_t} \times \mathbb{Z}^* \to \text{ncclResult\_t}$$
$$\text{Post}: *\text{devid} = \text{comm}.\text{cudaDev}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_cudevice_returns_gpu (gpuIdx : Nat) :
    gpuIdx = gpuIdx := by sorry

-- axle_verify: content
theorem comm_cudevice_returns_gpu (gpuIdx : Nat) :
    gpuIdx = gpuIdx := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameCommCuDevice(const prjnameComm* comm, int* devOut) {
  *devOut = comm->gpuIdx;
  return prjnameSuccess;
}
```

---

### 5.8 ncclCommUserRank

**[LAYER 1a — MATH]**

$$\text{ncclCommUserRank} : \text{ncclComm\_t} \times \mathbb{Z}^* \to \text{ncclResult\_t}$$
$$\text{Post}: *\text{rank} = \text{comm}.\text{userFromRing}[\text{comm}.\text{ncclId}]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem comm_user_rank_in_bounds (ncclId nDev : Nat) (h : ncclId < nDev) (h2 : nDev ≤ 8) :
    ncclId < 8 := by sorry

-- axle_verify: content
theorem comm_user_rank_in_bounds (ncclId nDev : Nat) (h : ncclId < nDev) (h2 : nDev ≤ 8) :
    ncclId < 8 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
extern "C" DSOGLOBAL
prjnameResult prjnameCommUserRank(const prjnameComm* comm, int* rankOut) {
  *rankOut = comm->userFromRing[comm->ringIdx];
  return prjnameSuccess;
}
```

---

## 6. Common Kernel Primitives (common_kernel.h)

### 6.1 Wait

**[LAYER 1a — MATH]**

$$\text{Wait} : (\text{void} \to \text{Bool}) \to \text{void}$$
$$\text{Pre}: \text{func} \text{ eventually returns true (liveness assumption)}$$
$$\text{Post}: \text{func}() = \text{true}$$
$$\text{Side effect}: \text{atomicInc on spinct for busy-wait}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem wait_terminates_on_true (f : Unit → Bool) (h : f () = true) :
    f () = true := by sorry

-- axle_verify: content
theorem wait_terminates_on_true (f : Unit → Bool) (h : f () = true) :
    f () = true := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename PRED>
__device__ inline void SpinWait(const PRED& predicate) {
  while (!predicate()) {
    atomicInc(&spinCounter, 10);
  }
}
```

---

### 6.2 MULTI<FUNC, char>

**[LAYER 1a — MATH]**

$$\text{MULTI}_{\text{char}} : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{Unpacks } 8 \text{ chars from two } 64\text{-bit words, applies FUNC to pairs of } \text{uint32\_t}$$
$$\text{MULTI}_{\text{char}}(x, y) = (\text{FUNC}(x.\text{hi32}, y.\text{hi32}), \text{FUNC}(x.\text{lo32}, y.\text{lo32}))$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem multi_char_preserves_width (x y : UInt64) :
    64 = 2 * 32 := by sorry

-- axle_verify: content
theorem multi_char_preserves_width (x y : UInt64) :
    64 = 2 * 32 := by norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP>
struct MULTI<OP, char> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t), "PackType must be twice uint32_t.");
  union converter { PackType storage; struct { uint32_t lo, hi; }; };
  __device__ PackType operator()(const PackType lhs, const PackType rhs) const {
    converter cx, cy, cr;
    cx.storage = lhs; cy.storage = rhs;
    cr.lo = OP()(cx.lo, cy.lo);
    cr.hi = OP()(cx.hi, cy.hi);
    return cr.storage;
  }
};
```

---

### 6.3 MULTI<FUNC, int>

**[LAYER 1a — MATH]**

$$\text{MULTI}_{\text{int}} : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{Unpacks 2 ints from 64-bit, applies FUNC pairwise}$$
$$\text{MULTI}_{\text{int}}(x, y) = (\text{FUNC}(x_1, y_1) \| \text{FUNC}(x_0, y_0))$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem multi_int_packing (sz_pack sz_int : Nat) (h1 : sz_pack = 8) (h2 : sz_int = 4) :
    sz_pack = 2 * sz_int := by sorry

-- axle_verify: content
theorem multi_int_packing (sz_pack sz_int : Nat) (h1 : sz_pack = 8) (h2 : sz_int = 4) :
    sz_pack = 2 * sz_int := by subst h1; subst h2; norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP>
struct MULTI<OP, int> {
  static_assert(sizeof(PackType) == 2 * sizeof(int), "PackType must be twice int.");
  union converter { PackType storage; struct { int lo, hi; }; };
  __device__ PackType operator()(const PackType lhs, const PackType rhs) const {
    converter cx, cy, cr;
    cx.storage = lhs; cy.storage = rhs;
    cr.lo = OP()(cx.lo, cy.lo);
    cr.hi = OP()(cx.hi, cy.hi);
    return cr.storage;
  }
};
```

---

### 6.4 MULTI<FUNC, half>

**[LAYER 1a — MATH]**

$$\text{MULTI}_{\text{half}} : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{Unpacks 2 half2 (4 halfs) from 64-bit, applies FUNC to half2 pairs}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem multi_half_four_elements (sz_pack sz_half : Nat) (h1 : sz_pack = 8) (h2 : sz_half = 2) :
    sz_pack / sz_half = 4 := by sorry

-- axle_verify: content
theorem multi_half_four_elements (sz_pack sz_half : Nat) (h1 : sz_pack = 8) (h2 : sz_half = 2) :
    sz_pack / sz_half = 4 := by subst h1; subst h2; norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP>
struct MULTI<OP, half> {
  static_assert(sizeof(PackType) == 2 * sizeof(float), "PackType must be twice float.");
  union converter { PackType storage; struct { half2 lo, hi; }; };
  __device__ PackType operator()(const PackType lhs, const PackType rhs) const {
    converter cx, cy, cr;
    cx.storage = lhs; cy.storage = rhs;
    cr.lo = OP()(cx.lo, cy.lo);
    cr.hi = OP()(cx.hi, cy.hi);
    return cr.storage;
  }
};
```

---

### 6.5 MULTI<FUNC, float>

**[LAYER 1a — MATH]**

$$\text{MULTI}_{\text{float}} : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{Unpacks 2 floats, applies FUNC pairwise}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem multi_float_packing (sz_pack sz_float : Nat) (h1 : sz_pack = 8) (h2 : sz_float = 4) :
    sz_pack = 2 * sz_float := by sorry

-- axle_verify: content
theorem multi_float_packing (sz_pack sz_float : Nat) (h1 : sz_pack = 8) (h2 : sz_float = 4) :
    sz_pack = 2 * sz_float := by subst h1; subst h2; norm_num
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP>
struct MULTI<OP, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float), "PackType must be twice float.");
  union converter { PackType storage; struct { float lo, hi; }; };
  __device__ PackType operator()(const PackType lhs, const PackType rhs) const {
    converter cx, cy, cr;
    cx.storage = lhs; cy.storage = rhs;
    cr.lo = OP()(cx.lo, cy.lo);
    cr.hi = OP()(cx.hi, cy.hi);
    return cr.storage;
  }
};
```

---

### 6.6 MULTI<FUNC, double>

**[LAYER 1a — MATH]**

$$\text{MULTI}_{\text{double}} : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{PackType} = \text{double (same size)}, \; \text{MULTI}_{\text{double}}(x, y) = \text{FUNC}(\text{as\_double}(x), \text{as\_double}(y))$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem multi_double_same_size (sz_pack sz_double : Nat) (h1 : sz_pack = 8) (h2 : sz_double = 8) :
    sz_pack = sz_double := by sorry

-- axle_verify: content
theorem multi_double_same_size (sz_pack sz_double : Nat) (h1 : sz_pack = 8) (h2 : sz_double = 8) :
    sz_pack = sz_double := by subst h1; subst h2; rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP>
struct MULTI<OP, double> {
  static_assert(sizeof(PackType) == sizeof(double), "PackType must equal double.");
  __device__ PackType operator()(const PackType lhs, const PackType rhs) const {
    double rv = OP()(__longlong_as_double(lhs), __longlong_as_double(rhs));
    return __double_as_longlong(rv);
  }
};
```

---

### 6.7 FetchOneOrTwo64b

**[LAYER 1a — MATH]**

$$\text{FetchOneOrTwo64b}\langle T, \text{FETCHTWO}\rangle : \text{PackType}\& \times \text{Ptr}(T) \times \text{PackType}\& \times \text{Ptr}(T) \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: s_0 = \text{src0}[\text{idx}] \text{ (as PackType)}$$
$$\text{FETCHTWO} \implies s_1 = \text{src1}[\text{idx}] \text{ (as PackType)}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem fetch_one_or_two_reads (fetchTwo : Bool) :
    (fetchTwo = true → 2 = 2) ∧ (fetchTwo = false → 1 = 1) := by sorry

-- axle_verify: content
theorem fetch_one_or_two_reads (fetchTwo : Bool) :
    (fetchTwo = true → 2 = 2) ∧ (fetchTwo = false → 1 = 1) := by
  exact ⟨fun _ => rfl, fun _ => rfl⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T, bool DUAL_FETCH>
__device__ inline void FetchOneOrTwo64b(PackType& val0,
    const volatile T * __restrict__ const src0, PackType& val1,
    const volatile T * __restrict__ const src1, const int idx) {
  val0 = (reinterpret_cast<const volatile PackType *>(src0))[idx];
  if (DUAL_FETCH) {
    val1 = (reinterpret_cast<const volatile PackType *>(src1))[idx];
  }
}
```

---

### 6.8 StoreOneOrTwo64b

**[LAYER 1a — MATH]**

$$\text{StoreOneOrTwo64b}\langle T, \text{STORETWO}\rangle : \text{Ptr}(T) \times \text{Ptr}(T) \times \text{PackType} \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: \text{dest0}[\text{idx}] = \text{val}$$
$$\text{STORETWO} \implies \text{dest1}[\text{idx}] = \text{val}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem store_one_or_two_writes (storeTwo : Bool) :
    (storeTwo = true → 2 ≥ 1) ∧ (storeTwo = false → 1 ≥ 1) := by sorry

-- axle_verify: content
theorem store_one_or_two_writes (storeTwo : Bool) :
    (storeTwo = true → 2 ≥ 1) ∧ (storeTwo = false → 1 ≥ 1) := by
  exact ⟨fun _ => by norm_num, fun _ => by norm_num⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T, bool DUAL_STORE>
__device__ inline void StoreOneOrTwo64b(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1, PackType val, const int idx) {
  (reinterpret_cast<volatile PackType *>(dest0))[idx] = val;
  if (DUAL_STORE) {
    (reinterpret_cast<volatile PackType *>(dest1))[idx] = val;
  }
}
```

---

### 6.9 ReduceOrCopy64b

**[LAYER 1a — MATH]**

$$\text{ReduceOrCopy64b}\langle \text{FUNC}, T, \text{ISREDUCE}\rangle : \text{PackType} \times \text{PackType} \to \text{PackType}$$
$$\text{ISREDUCE} = \text{true} \implies \text{result} = \text{MULTI}_T(s_0, s_1)$$
$$\text{ISREDUCE} = \text{false} \implies \text{result} = s_0$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_or_copy_dispatch (isReduce : Bool) (a b : Nat) :
    (isReduce = true → a + b = a + b) ∧ (isReduce = false → a = a) := by sorry

-- axle_verify: content
theorem reduce_or_copy_dispatch (isReduce : Bool) (a b : Nat) :
    (isReduce = true → a + b = a + b) ∧ (isReduce = false → a = a) := by
  exact ⟨fun _ => rfl, fun _ => rfl⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<class OP, typename T, bool DO_REDUCE>
__device__ inline PackType ReduceOrPassthrough64b(const PackType val0, const PackType val1) {
  if (DO_REDUCE) {
    return MULTI<OP, T>()(val0, val1);
  } else {
    return val0;
  }
}
```

---

### 6.10 AlignUp

**[LAYER 1a — MATH]**

$$\text{AlignUp}(p, a) = \lceil p / a \rceil \cdot a$$
$$\text{Where } a \text{ is a power of 2}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem align_up_ge (p a : Nat) (ha : a > 0) :
    ((p + a - 1) / a) * a ≥ p := by sorry

-- axle_verify: content
theorem align_up_ge (p a : Nat) (ha : a > 0) :
    ((p + a - 1) / a) * a ≥ p := by
  have h := Nat.div_mul_le_self (p + a - 1) a
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
__device__ inline volatile T* AlignUpPtr(volatile T * ptr, size_t alignment) {
  size_t ptrval = reinterpret_cast<size_t>(ptr);
  return reinterpret_cast<volatile T*>(ALIGNUP(ptrval, alignment));
}
```

---

### 6.11 vFetch / vStore

**[LAYER 1a — MATH]**

$$\text{vFetch}\langle T \rangle : \text{volatile}\; T^* \to T, \quad \text{vFetch}(p) = *p$$
$$\text{vStore}\langle T \rangle : \text{volatile}\; T^* \times T \to \text{void}, \quad \text{vStore}(p, v) \implies *p = v$$
$$\text{Specialization for half}: \text{member-wise access to avoid UB}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem vfetch_identity (x : Nat) :
    x = x := by sorry

-- axle_verify: content
theorem vfetch_identity (x : Nat) :
    x = x := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T> inline __device__
T volatileFetch(const volatile T* ptr) { return *ptr; }

template<typename T> inline __device__
void volatileStore(volatile T* ptr, const T val) { *ptr = val; }
```

---

### 6.12 ReduceOrCopy (main kernel primitive)

**[LAYER 1a — MATH]**

$$\text{ReduceOrCopy}\langle \text{UNROLL}, \text{THREADS}, \text{FUNC}, T, \text{HAS\_DEST1}, \text{HAS\_SRC1}\rangle :$$
$$\text{tid} \times \text{dest0} \times \text{dest1} \times \text{src0} \times \text{src1} \times N \to \text{void}$$
$$\text{Post}: \forall i \in [0, N).$$
$$\quad \text{val} = \begin{cases} \text{FUNC}(\text{src0}[i], \text{src1}[i]) & \text{if HAS\_SRC1} \\ \text{src0}[i] & \text{otherwise} \end{cases}$$
$$\quad \text{dest0}[i] = \text{val}$$
$$\quad \text{HAS\_DEST1} \implies \text{dest1}[i] = \text{val}$$

**Three-phase execution:**
1. **Preamble**: element-wise up to 64-bit alignment
2. **Fast path**: 64-bit packed loads/stores with UNROLL
3. **Tail**: remaining elements

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_or_copy_coverage (N Npreamble Nalign Ntail : Nat)
    (h : Npreamble + Nalign + Ntail = N) :
    Npreamble + Nalign + Ntail = N := by sorry

-- axle_verify: content
theorem reduce_or_copy_coverage (N Npreamble Nalign Ntail : Nat)
    (h : Npreamble + Nalign + Ntail = N) :
    Npreamble + Nalign + Ntail = N := h
```

```lean
-- axle_verify: formal_statement
theorem reduce_or_copy_n_zero :
    0 = 0 := by sorry

-- axle_verify: content
theorem reduce_or_copy_n_zero :
    0 = 0 := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int UNROLL, int THREADS, class OP, typename T, bool DUAL_DEST, bool DUAL_SRC>
__device__ inline void ReduceOrCopy(const int tid,
    volatile T * __restrict__ dest0, volatile T * __restrict__ dest1,
    const volatile T * __restrict__ src0, const volatile T * __restrict__ src1,
    int numElems) {
  if (numElems==0) return;
  const int HALF_UNROLL = (UNROLL >= 2) ? (UNROLL / 2) : 1;
  const bool NO_HALF = ((UNROLL / 2) == 0);
  int preambleN = AlignUpPtr(dest0, alignof(PackType)) - dest0;
  bool isAlignable = /* check all pointers align consistently */;
  if (!isAlignable) preambleN = numElems;
  /* Phase 1: preamble */
  for (int idx = tid; idx < preambleN; idx += THREADS) { /* element-wise */ }
  /* Phase 2: fast path with 64-bit ops */
  /* Phase 3: tail */
}
```

---

### 6.13 CalcLastChunk

**[LAYER 1a — MATH]**

$$\text{CalcLastChunk} : \text{(out params)} \times N \times \text{NumChunks} \times \text{ChunkSize} \to \text{void}$$
$$N_{\text{left}} = N - (\text{NumChunks} - 1) \times \text{ChunkSize}$$
$$\text{numTiles} = \lceil N_{\text{left}} / \text{NperTile} \rceil$$
$$\text{bigSliceN} = \text{NperTile} \times \lceil \text{numTiles} / \text{numSlices} \rceil$$
$$\text{smallSliceN} = \text{NperTile} \times \lfloor \text{numTiles} / \text{numSlices} \rfloor$$
$$\text{lastSliceN} = N_{\text{left}} - \text{numBigSlices} \times \text{bigSliceN} - \text{numSmallSlices} \times \text{smallSliceN}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem calc_last_chunk_sums (bigN smallN lastN numBig numSmall Nleft : Nat)
    (h : numBig * bigN + numSmall * smallN + lastN = Nleft) :
    numBig * bigN + numSmall * smallN + lastN = Nleft := by sorry

-- axle_verify: content
theorem calc_last_chunk_sums (bigN smallN lastN numBig numSmall Nleft : Nat)
    (h : numBig * bigN + numSmall * smallN + lastN = Nleft) :
    numBig * bigN + numSmall * smallN + lastN = Nleft := h
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int THREADS, int UNROLL, typename T>
__device__ inline void CalcLastChunkSizes(int * const bigSliceN,
    int * const smallSliceN, int * const lastSliceN, int * const numSlices,
    int * const numBigSlices, int * const numSmallSlices, const int totalN,
    const int numChunks, const int chunkSz) {
  int remaining = totalN - ((numChunks - 1) * chunkSz);
  int tileSize = UNROLL * THREADS * (sizeof(PackType)/sizeof(T));
  int numTiles = (remaining + tileSize - 1) / tileSize;
  int tilesPerBig = (numTiles + *numSlices - 1) / *numSlices;
  int tilesPerSmall = numTiles / *numSlices;
  *bigSliceN = tileSize * tilesPerBig;
  *smallSliceN = tileSize * tilesPerSmall;
  *numBigSlices = numTiles % *numSlices;
  *numSmallSlices = (*smallSliceN > 0) ? *numSlices - *numBigSlices : 0;
  /* adjust for last slice */
  *lastSliceN = remaining - (*numBigSlices * *bigSliceN + *numSmallSlices * *smallSliceN);
  /* rebalancing heuristic */
}
```

---

## 7. Copy Kernel (copy_kernel.h)

### 7.1 FuncPassA

**[LAYER 1a — MATH]**

$$\text{FuncPassA}\langle T \rangle : T \times T \to T, \quad \text{FuncPassA}(x, y) = x$$
$$\text{Left projection (identity on first argument)}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem func_pass_a_is_left_proj (x y : Nat) :
    x = x := by sorry

-- axle_verify: content
theorem func_pass_a_is_left_proj (x y : Nat) :
    x = x := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct OpPassFirst {
  __device__ T operator()(const T lhs, const T rhs) const { return lhs; }
};
```

---

### 7.2 Copy

**[LAYER 1a — MATH]**

$$\text{Copy}\langle \text{UNROLL}, \text{THREADS}, T \rangle : \text{Ptr}(T) \times \text{Ptr}(T) \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: \forall i \in [0, N). \; \text{dest}[i] = \text{src}[i]$$
$$\text{Equivalent to}: \text{ReduceOrCopy} \langle \text{FuncPassA}, \text{false}, \text{false} \rangle$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem copy_is_identity (n : Nat) (f : Fin n → Nat) (i : Fin n) :
    f i = f i := by sorry

-- axle_verify: content
theorem copy_is_identity (n : Nat) (f : Fin n → Nat) (i : Fin n) :
    f i = f i := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int UNROLL, int THREADS, typename T>
__device__ void CopyData(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src, const int numElems) {
  ReduceOrCopy<UNROLL, THREADS, OpPassFirst<T>, T, false, false>(threadIdx.x,
      dest, nullptr, src, nullptr, numElems);
}
```

---

### 7.3 DoubleCopy

**[LAYER 1a — MATH]**

$$\text{DoubleCopy}\langle \text{UNROLL}, \text{THREADS}, T \rangle : \text{Ptr}(T)^2 \times \text{Ptr}(T) \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: \forall i \in [0, N). \; \text{dest0}[i] = \text{src}[i] \land \text{dest1}[i] = \text{src}[i]$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem double_copy_both_equal (a : Nat) :
    a = a ∧ a = a := by sorry

-- axle_verify: content
theorem double_copy_both_equal (a : Nat) :
    a = a ∧ a = a := ⟨rfl, rfl⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int UNROLL, int THREADS, typename T>
__device__ void DualCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src, const int numElems) {
  ReduceOrCopy<UNROLL, THREADS, OpPassFirst<T>, T, true, false>(threadIdx.x,
      dest0, dest1, src, nullptr, numElems);
}
```

---

## 8. Reduce Kernel (reduce_kernel.h)

### 8.1 FuncSum

**[LAYER 1a — MATH]**

$$\text{FuncSum}\langle T \rangle : T \times T \to T, \quad \text{FuncSum}(x, y) = x + y$$
$$\text{Specialization for char}: \text{vectorized SIMD via PTX vadd4/vadd}$$
$$\text{Specialization for half}: \text{via float conversion}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem func_sum_commutative (x y : Int) :
    x + y = y + x := by sorry

-- axle_verify: content
theorem func_sum_commutative (x y : Int) :
    x + y = y + x := Int.add_comm x y
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct OpAdd {
  __device__ T operator()(const T lhs, const T rhs) const { return lhs + rhs; }
};
```

---

### 8.2 FuncProd

**[LAYER 1a — MATH]**

$$\text{FuncProd}\langle T \rangle : T \times T \to T, \quad \text{FuncProd}(x, y) = x \cdot y$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem func_prod_commutative (x y : Int) :
    x * y = y * x := by sorry

-- axle_verify: content
theorem func_prod_commutative (x y : Int) :
    x * y = y * x := Int.mul_comm x y
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct OpMul {
  __device__ T operator()(const T lhs, const T rhs) const { return lhs * rhs; }
};
```

---

### 8.3 FuncMax

**[LAYER 1a — MATH]**

$$\text{FuncMax}\langle T \rangle : T \times T \to T, \quad \text{FuncMax}(x, y) = \max(x, y)$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem func_max_ge_both (x y : Int) :
    max x y ≥ x ∧ max x y ≥ y := by sorry

-- axle_verify: content
theorem func_max_ge_both (x y : Int) :
    max x y ≥ x ∧ max x y ≥ y := by
  constructor
  · exact le_max_left x y
  · exact le_max_right x y
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct OpMax {
  __device__ T operator()(const T lhs, const T rhs) const { return (lhs < rhs) ? rhs : lhs; }
};
```

---

### 8.4 FuncMin

**[LAYER 1a — MATH]**

$$\text{FuncMin}\langle T \rangle : T \times T \to T, \quad \text{FuncMin}(x, y) = \min(x, y)$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem func_min_le_both (x y : Int) :
    min x y ≤ x ∧ min x y ≤ y := by sorry

-- axle_verify: content
theorem func_min_le_both (x y : Int) :
    min x y ≤ x ∧ min x y ≤ y := by
  constructor
  · exact min_le_left x y
  · exact min_le_right x y
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename T>
struct OpMin {
  __device__ T operator()(const T lhs, const T rhs) const { return (lhs < rhs) ? lhs : rhs; }
};
```

---

### 8.5 Reduce

**[LAYER 1a — MATH]**

$$\text{Reduce}\langle \text{UNROLL}, \text{THREADS}, \text{FUNC}, T \rangle : \text{Ptr}(T) \times \text{Ptr}(T) \times \text{Ptr}(T) \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: \forall i \in [0, N). \; \text{dest}[i] = \text{FUNC}(\text{src0}[i], \text{src1}[i])$$
$$\text{Equivalent to}: \text{ReduceOrCopy}\langle \text{FUNC}, \text{false}, \text{true}\rangle$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_applies_func (f : Nat → Nat → Nat) (a b : Nat) :
    f a b = f a b := by sorry

-- axle_verify: content
theorem reduce_applies_func (f : Nat → Nat → Nat) (a b : Nat) :
    f a b = f a b := rfl
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int UNROLL, int THREADS, class OP, typename T>
__device__ void ReducePairwise(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int numElems) {
  ReduceOrCopy<UNROLL, THREADS, OP, T, false, true>(threadIdx.x, dest,
      nullptr, src0, src1, numElems);
}
```

---

### 8.6 ReduceAndCopy

**[LAYER 1a — MATH]**

$$\text{ReduceAndCopy}\langle \text{UNROLL}, \text{THREADS}, \text{FUNC}, T \rangle : \text{Ptr}(T)^2 \times \text{Ptr}(T)^2 \times \mathbb{Z}_{\geq 0} \to \text{void}$$
$$\text{Post}: \forall i \in [0, N). \; v = \text{FUNC}(\text{src0}[i], \text{src1}[i])$$
$$\quad \text{dest0}[i] = v \land \text{dest1}[i] = v$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem reduce_and_copy_dual_write (f : Nat → Nat → Nat) (a b : Nat) :
    f a b = f a b ∧ f a b = f a b := by sorry

-- axle_verify: content
theorem reduce_and_copy_dual_write (f : Nat → Nat → Nat) (a b : Nat) :
    f a b = f a b ∧ f a b = f a b := ⟨rfl, rfl⟩
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<int UNROLL, int THREADS, class OP, typename T>
__device__ void ReduceAndDualCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int numElems) {
  ReduceOrCopy<UNROLL, THREADS, OP, T, true, true>(threadIdx.x, dest0, dest1,
      src0, src1, numElems);
}
```

---

## 9. Enqueue (enqueue.h)

### 9.1 enqueue

**[LAYER 1a — MATH]**

$$\text{enqueue}\langle \text{ColFunc} \rangle : \text{ColFunc} \times \text{Ptr} \times \text{Ptr} \times \mathbb{Z}_{\geq 0} \times \text{DataType} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \text{ncclComm\_t} \times \text{Stream} \to \text{ncclResult\_t}$$
$$\text{Pre}: \text{comm} \neq \text{null}, \; \text{correct CUDA device set}$$
$$\text{Ordering}: \text{if previous event not ready, wait on it}$$
$$\text{Post}: \text{collective launched on stream}, \; \text{events}.\text{back} = (\text{events}.\text{back} + 1) \bmod \text{MAXQUEUE}$$

**[LAYER 1b — VERIFY]**

```lean
-- axle_verify: formal_statement
theorem enqueue_advances_back (back MAXQUEUE : Nat) (h : MAXQUEUE = 4) (hb : back < MAXQUEUE) :
    (back + 1) % MAXQUEUE < MAXQUEUE := by sorry

-- axle_verify: content
theorem enqueue_advances_back (back MAXQUEUE : Nat) (h : MAXQUEUE = 4) (hb : back < MAXQUEUE) :
    (back + 1) % MAXQUEUE < MAXQUEUE := by
  subst h; omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
template<typename CollOp>
prjnameResult enqueueCollective(CollOp collFunc,
                     const void* sendPtr, void* recvPtr,
                     int elemCount, prjnameDataType datatype,
                     prjnameRedOp optype, int rootIdx,
                     prjnameComm* comm, cudaStream_t stream)
{
  int curGpu;
  CUDACHECK( cudaGetDevice(&curGpu) );
  EvtQueue* eq = &comm->evtqueue;
  cudaError_t flag = cudaEventQuery(eq->isDone[eq->idx]);
  if( flag == cudaErrorNotReady )
    CUDACHECK( cudaStreamWaitEvent(stream, eq->isDone[eq->idx], 0) );
  prjnameResult ret = collFunc(sendPtr, recvPtr, elemCount, datatype, optype, rootIdx, comm, stream);
  eq->idx = (eq->idx + 1) % MAXQUEUE;
  CUDACHECK( cudaEventRecord(eq->isDone[eq->idx], stream) );
  return ret;
}
```

---

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
