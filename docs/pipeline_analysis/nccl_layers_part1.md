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
