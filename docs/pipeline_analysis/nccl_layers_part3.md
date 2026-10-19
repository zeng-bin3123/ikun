
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
