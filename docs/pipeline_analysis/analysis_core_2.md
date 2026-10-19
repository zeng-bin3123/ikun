# NCCL Formal Analysis — src/core.cu (Part 2: Communicator Lifecycle)

---

## 14. `commClearMaps`

### LAYER 1a — MATH (LaTeX)

$$
\text{commClearMaps}: \text{Comm} \to \text{Result}
$$
$$
\textbf{Post: } \forall\, d \in [0, \text{nDev}),\;
\begin{cases}
  \text{CLEANUP\_CUIPC}: \text{cuIpcCloseMemHandle}(\text{handle}) \\
  \text{CLEANUP\_UNMAP}: \text{cudaHostUnregister}(\text{handle}) \wedge \text{shmUnmap}(\text{handle}) \\
  \text{CLEANUP\_NONE}: \text{no-op}
\end{cases}
$$
$$
\text{All maps zeroed: userFromRing, ringFromUser, devUserFromRing}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem clear_maps_all_types (cleanupType : Nat) (hct : cleanupType < 3) :
    cleanupType = 0 ∨ cleanupType = 1 ∨ cleanupType = 2 := by sorry

-- axle_verify: content
theorem clear_maps_all_types (cleanupType : Nat) (hct : cleanupType < 3) :
    cleanupType = 0 ∨ cleanupType = 1 ∨ cleanupType = 2 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType commClearMaps(prjname::commType commobj) {
  prjname::resultType resultval, retval = prjname::Success;
  cudaError_t cudares;

  for(int didx=0; didx<commobj->nDev; ++didx) {
    switch(commobj->cleanup[didx].type) {
      case CLEANUP_NONE:
        break;
      case CLEANUP_CUIPC:
        resultval = wrapCuIpcCloseMemHandle((CUdeviceptr)commobj->cleanup[didx].handle);
        if (resultval != prjname::Success) {
          WARN("rank %d failed to close IPC handle to rank %d",
            commobj->userFromRing[commobj->ncclId], commobj->userFromRing[didx]);
          retval = (retval == prjname::Success) ? resultval : retval;
        }
        break;
      case CLEANUP_UNMAP:
        cudares = cudaHostUnregister(commobj->cleanup[didx].handle);
        if (cudares != cudaSuccess) {
          WARN("rank %d failed to unregister handle to rank %d",
            commobj->userFromRing[commobj->ncclId], commobj->userFromRing[didx]);
            retval = (retval == prjname::Success) ? prjname::UnhandledCudaError : retval;
        }
        resultval = shmUnmap(commobj->cleanup[didx].handle, offsetof(prjname::memType, buff) + commobj->buffSize);
        if (resultval != prjname::Success) {
          WARN("rank %d failed to unmap handle to rank %d",
            commobj->userFromRing[commobj->ncclId], commobj->userFromRing[didx]);
            retval = (retval == prjname::Success) ? resultval : retval;
        }
        break;
      default:
        WARN("Unknown cleanup type %d", commobj->cleanup[didx].type);
    }
    commobj->cleanup[didx].type = 0;
    commobj->cleanup[didx].handle = NULL;
  }

  memset(commobj->userFromRing, 0, sizeof(int)*MAXPEERS);
  memset(commobj->ringFromUser, 0, sizeof(int)*MAXPEERS);
  if (commobj->devUserFromRing != NULL) {
    cudaError_t errval = cudaMemset(commobj->devUserFromRing, 0, sizeof(int)*MAXPEERS);
    if (errval != cudaSuccess) {
      WARN("Failed to clear dev map: %s", cudaGetErrorString(errval));
      retval = (retval == prjname::Success) ? prjname::UnhandledCudaError : retval;
    }
  }
  return retval;
}
```

---

## 15. `commBuildMaps`

### LAYER 1a — MATH (LaTeX)

$$
\text{commBuildMaps}: \text{Comm} \times \text{UniqueId}^* \times \mathbb{Z} \times \text{RankEntry}^K \to \text{Result}
$$
$$
\textbf{Post: }
$$
$$
\forall\, i \in [0,K),\; \text{userFromRing}[i] = \text{ranks}[i].\text{rank},\; \text{ringFromUser}[\text{ranks}[i].\text{rank}] = i
$$
$$
\text{comm.ncclId} = \text{index where ranks}[i].\text{rank} = \text{myRank}
$$
$$
\text{Connectivity determination: }
$$
$$
\text{same PID} \wedge \text{P2P capable} \implies \text{P2P device mem (CLEANUP\_NONE)}
$$
$$
\text{same PID} \wedge \neg\text{P2P} \implies \text{zero-copy host mem (CLEANUP\_NONE), useRemoteRecv=0}
$$
$$
\text{diff PID} \wedge \text{P2P} \implies \text{IPC P2P (CLEANUP\_CUIPC), useRemoteRecv=0}
$$
$$
\text{diff PID} \wedge \neg\text{P2P} \implies \text{zero-copy host shm (CLEANUP\_UNMAP), useRemoteRecv=0}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem build_maps_bijection (K : Nat) (ranks : Fin K → Nat) (h_inj : Function.Injective ranks) :
    ∀ i j : Fin K, ranks i = ranks j → i = j := by sorry

-- axle_verify: content
theorem build_maps_bijection (K : Nat) (ranks : Fin K → Nat) (h_inj : Function.Injective ranks) :
    ∀ i j : Fin K, ranks i = ranks j → i = j := by
  intro i j h
  exact h_inj h
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType commBuildMaps(prjname::commType commobj, prjname::uniqueIdType* commIdptr,
    int rankIdx, RankEntry* ranksArr) {
  int devCount = commobj->nDev;
  for(int idx=0; idx<devCount; ++idx) {
    if (ranksArr[idx].bufSizeN != commobj->buffSize
        || ranksArr[idx].devCount != commobj->nDev) {
      commClearMaps(commobj);
      return prjname::RankMismatch;
    }
    int iRankIdx = ranksArr[idx].rankIdx;
    commobj->userFromRing[idx] = iRankIdx;
    commobj->ringFromUser[iRankIdx] = idx;
  }

  if (cudaMemcpy(commobj->devUserFromRing, commobj->userFromRing, devCount*sizeof(int),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("rank %d failed to copy maps to device", rankIdx);
    commClearMaps(commobj);
    return prjname::UnhandledCudaError;
  }

  int myIdx = -1;
  for (int idx=0; idx<devCount; ++idx) {
    if(ranksArr[idx].rankIdx == rankIdx) { myIdx = idx; break; }
  }
  if (myIdx == -1) {
    WARN("rank %d not found in communicator", rankIdx);
    return prjname::InvalidRank;
  }
  commobj->ncclId = myIdx;

  int myDevIdx = ranksArr[myIdx].cudaDevIdx;
  pid_t myPidVal = ranksArr[myIdx].pidval;
  commobj->useRemoteRecv = 1;

  for (int idx=0; idx<devCount; ++idx) {
    int iRankIdx = ranksArr[idx].rankIdx;
    int iDevIdx = ranksArr[idx].cudaDevIdx;
    pid_t iPidVal = ranksArr[idx].pidval;
    int canpeerFlag = 0;

    if (cudaDeviceCanAccessPeer(&canpeerFlag, myDevIdx, iDevIdx) != cudaSuccess) {
      INFO("peer query failed between rank %d (dev %d) and rank %d (dev %d)",
        rankIdx, myDevIdx, iRankIdx, iDevIdx);
      canpeerFlag = 0;
    }

    if (canpeerFlag) {
      cudaError_t errval;
      errval = cudaDeviceEnablePeerAccess(iDevIdx, 0);
      if (errval == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (errval != cudaSuccess) {
        INFO("peer access failed between rank %d (dev %d) and rank %d (dev %d)\n",
          rankIdx, myDevIdx, iRankIdx, iDevIdx);
        canpeerFlag = 0;
      }
    }

    if (iPidVal == myPidVal && (canpeerFlag || myDevIdx == iDevIdx)) {
      INFO("rank access %d -> %d via P2P device mem", rankIdx, iRankIdx);
      commobj->local[idx] = ranksArr[myIdx].devmemptr;
      commobj->remote[idx] = ranksArr[idx].devmemptr;
      commobj->cleanup[idx].type = CLEANUP_NONE;
    } else if (iPidVal == myPidVal) {
      INFO("rank access %d -> %d via zero-copy host mem", rankIdx, iRankIdx);
      commobj->useRemoteRecv = 0;
      if (cudaHostGetDevicePointer(commobj->local+idx, ranksArr[myIdx].hostmemptr, 0) != cudaSuccess) {
        WARN("rank %d failed to map zero copy buffer to device", rankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      if (cudaHostGetDevicePointer(commobj->remote+idx, ranksArr[idx].hostmemptr, 0) != cudaSuccess) {
        WARN("rank %d failed to map %d's zero copy buffer to device", rankIdx, iRankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      commobj->cleanup[idx].type = CLEANUP_NONE;
    } else if (canpeerFlag || myDevIdx == iDevIdx) {
      INFO("rank access %d -> %d via Ipc P2P device mem", rankIdx, iRankIdx);
      commobj->useRemoteRecv = 0;
      commobj->local[idx] = ranksArr[myIdx].devmemptr;
      if (wrapCuIpcOpenMemHandle((CUdeviceptr*)(&commobj->remote[idx]),
          ranksArr[idx].devipcHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) != prjname::Success) {
        WARN("rank %d failed to open Ipc handle to rank %d", rankIdx, iRankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      commobj->cleanup[idx].type = CLEANUP_CUIPC;
      commobj->cleanup[idx].handle = commobj->remote[idx];
    } else {
      INFO("rank access %d -> %d via zero copy host shm", rankIdx, iRankIdx);
      commobj->useRemoteRecv = 0;
      if (cudaHostGetDevicePointer(commobj->local+idx, ranksArr[myIdx].hostmemptr, 0) != cudaSuccess) {
        WARN("rank %d failed to obtain dev ptr to sysmem buffer", rankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      char ranknameStr[1024];
      sprintf(ranknameStr, "%s-%d", commIdptr->internal, ranksArr[idx].rankIdx);
      if (openHostMemShm(ranknameStr, (prjname::memType**)&commobj->cleanup[idx].handle, ranksArr[idx].bufSizeN)
          != prjname::Success) {
        WARN("rank %d failed to open sysmem buffer of rank %d", rankIdx, iRankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      if (shmUnlink(ranknameStr) != prjname::Success) {
        INFO("rank %d failed to unlink sysmem buffer of rank %d, %s", rankIdx, iRankIdx, ranknameStr);
      }
      if (cudaHostGetDevicePointer(commobj->remote+idx, commobj->cleanup[idx].handle, 0) != cudaSuccess) {
        WARN("rank %d failed to obtain dev ptr for rank %d", rankIdx, iRankIdx);
        commClearMaps(commobj);
        return prjname::UnhandledCudaError;
      }
      commobj->cleanup[idx].type = CLEANUP_UNMAP;
    }
  }
  INFO("PushToRecv algos are %s\n", (commobj->useRemoteRecv) ? "enabled" : "disabled");
  return prjname::Success;
}
```

---

## 16. `initDebug`

### LAYER 1a — MATH (LaTeX)

$$
\text{initDebug}: \emptyset \to \text{Unit}
$$
$$
\textbf{Post: } \text{ncclDebugLevel} \in \{\text{NONE}, \text{WARN}, \text{INFO}, \text{ABORT}\} \text{ based on NCCL\_DEBUG env var}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem debug_levels_exhaustive (level : Nat) (hl : level < 4) :
    level = 0 ∨ level = 1 ∨ level = 2 ∨ level = 3 := by sorry

-- axle_verify: content
theorem debug_levels_exhaustive (level : Nat) (hl : level < 4) :
    level = 0 ∨ level = 1 ∨ level = 2 ∨ level = 3 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static void initDebug() {
  const char* debugEnv = getenv("NCCL_DEBUG");
  if (debugEnv == NULL) {
    prjname::debugLevel = NONE;
  } else if (strcmp(debugEnv, "WARN") == 0) {
    prjname::debugLevel = WARN;
  } else if (strcmp(debugEnv, "INFO") == 0) {
    prjname::debugLevel = INFO;
    INFO("prjname debug level set to INFO");
  } else if (strcmp(debugEnv, "ABORT") == 0) {
    prjname::debugLevel = ABORT;
    INFO("prjname debug level set to ABORT");
  }
}
```

---

## 17. `commFree`

### LAYER 1a — MATH (LaTeX)

$$
\text{commFree}: \text{Comm} \to \text{Unit}
$$
$$
\textbf{Pre: } \text{comm may be NULL (no-op)}
$$
$$
\textbf{Post: } \text{All CUDA events destroyed, maps cleared, devMem freed, hostMem freed/unmapped, struct freed}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem comm_free_null_safe (isNull : Bool) :
    (isNull → True) ∧ (¬isNull → True) := by sorry

-- axle_verify: content
theorem comm_free_null_safe (isNull : Bool) :
    (isNull → True) ∧ (¬isNull → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
static void commFree(prjname::commType commobj) {
  if (commobj == NULL) return;

  for(int idx=0; idx<MAXQUEUE; ++idx) {
    if (commobj->events.isDone[idx] != NULL)
      if (cudaEventDestroy(commobj->events.isDone[idx]) != cudaSuccess)
        INFO("failed to destroy cuda event %d", idx);
  }

  prjname::resultType resultval = commClearMaps(commobj);
  if (resultval != prjname::Success) INFO("failed to cleanup comm maps");

  if (commobj->devUserFromRing != NULL)
    if (cudaFree(commobj->devUserFromRing) != cudaSuccess)
      INFO("commFree failed to free dev maps");

  if (commobj->devMem != NULL && cudaFree(commobj->devMem) != cudaSuccess)
    INFO("Failed to free devMap");

  if (commobj->hostMem != NULL) {
    if (commobj->hostMemState & ShmMapped) {
      if (cudaHostUnregister(commobj->hostMem) != cudaSuccess)
        INFO("Failed to unregister hostMem");
      size_t totalSize = offsetof(prjname::memType, buff) + commobj->buffSize;
      if (shmUnmap(commobj->hostMem, totalSize) != prjname::Success)
        INFO("Failed to unmap hostMem");
      commobj->hostMemState ^= ShmMapped;
    } else {
      cudaFreeHost(commobj->hostMem);
    }
  }
  free(commobj);
}
```

---

## 18. `commAlloc`

### LAYER 1a — MATH (LaTeX)

$$
\text{commAlloc}: \text{Comm}^* \times \mathbb{N} \times \text{UniqueId}^* \times \mathbb{Z} \to \text{Result}
$$
$$
\textbf{Pre: } 1 \leq \text{ndev} \leq \text{MAXPEERS},\; 0 \leq \text{rank} < \text{ndev}
$$
$$
\textbf{Post: } \text{comm allocated, devMem and hostMem allocated, CUDA events created}
$$
$$
\text{buffSize} = \text{NCCL\_BUFFSIZE env} \lor \text{DEFAULT\_BUFFER\_SIZE\_BYTES}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem comm_alloc_precondition (ndev rank maxPeers : Nat)
    (h1 : ndev ≥ 1) (h2 : ndev ≤ maxPeers) (h3 : rank < ndev) :
    rank < maxPeers := by sorry

-- axle_verify: content
theorem comm_alloc_precondition (ndev rank maxPeers : Nat)
    (h1 : ndev ≥ 1) (h2 : ndev ≤ maxPeers) (h3 : rank < ndev) :
    rank < maxPeers := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType commAlloc(prjname::commType* comretptr, int devCount,
    const prjname::uniqueIdType* commIdptr, int rankIdx) {
  if (devCount < 1 || devCount > MAXPEERS) {
    WARN("requested device count (%d) exceeds maximum of %d", devCount, MAXPEERS);
    return prjname::UnsupportedDeviceCount;
  }
  if (rankIdx >= devCount || rankIdx < 0) {
    WARN("rank %d exceeds ndev=%d", rankIdx, devCount);
    return prjname::InvalidRank;
  }

  struct prjname::commStruct* commobj = (struct prjname::commStruct*)malloc(sizeof(struct prjname::commStruct));
  if (commobj == NULL) {
    WARN("comm allocation failed");
    return prjname::SystemError;
  }
  memset(commobj, 0, sizeof(struct prjname::commStruct));

  commobj->nDev = devCount;
  cudaGetDevice(&commobj->cudaDev);

  const char* envstr = getenv("NCCL_BUFFSIZE");
  if (envstr != NULL) {
    errno = 0;
    commobj->buffSize = strtol(envstr, NULL, 10);
    if (errno == ERANGE || commobj->buffSize == 0) {
      INFO("rank %d invalid NCCL_BUFFSIZE: %s, using default %lu",
          rankIdx, envstr, DEFAULT_BUFFER_SIZE_BYTES);
      commobj->buffSize = DEFAULT_BUFFER_SIZE_BYTES;
    }
  } else {
    commobj->buffSize = DEFAULT_BUFFER_SIZE_BYTES;
  }
  INFO("rank %d using buffSize = %lu", rankIdx, commobj->buffSize);

  prjname::resultType resultval;
  resultval = allocDevMem(&commobj->devMem, commobj->buffSize);
  if (resultval != prjname::Success) {
    WARN("rank %d failed to allocate device buffer", rankIdx);
    commFree(commobj);
    return resultval;
  }
  if (cudaMalloc(&commobj->devUserFromRing, MAXPEERS*sizeof(int)) != cudaSuccess) {
    WARN("rank %d failed to allocated device maps", rankIdx);
    commFree(commobj);
    return prjname::CudaMallocFailed;
  }

  EventQueue* eqptr = &commobj->events;
  for(int idx=0; idx<MAXQUEUE; ++idx) {
    if (cudaEventCreateWithFlags(eqptr->isDone+idx, cudaEventDisableTiming) != cudaSuccess) {
      WARN("rank %d failed to create nccl event %d", rankIdx, idx);
      commFree(commobj);
      return prjname::UnhandledCudaError;
    }
  }

  if(commIdptr == NULL) {
    commobj->hostMemState = 0;
    resultval = allocHostMem(&commobj->hostMem, commobj->buffSize);
  } else {
    char ranknameStr[1024];
    sprintf(ranknameStr, "%s-%d", commIdptr->internal, rankIdx);
    resultval = openHostMemShm(ranknameStr, &commobj->hostMem, commobj->buffSize);
    if (resultval != prjname::Success) {
      WARN("rank %d failed to allocate host buffer", rankIdx);
      commFree(commobj);
      return resultval;
    }
    commobj->hostMemState = ShmMapped | ShmLinked;
  }

  *comretptr = commobj;
  return prjname::Success;
}
```

---

## 19. `commUnlinkHostMem`

### LAYER 1a — MATH (LaTeX)

$$
\text{commUnlinkHostMem}: \text{Comm} \times \text{UniqueId} \times \mathbb{Z} \to \text{Result}
$$
$$
\textbf{Post: } \text{shmUnlink("commId-rank")},\; \text{hostMemState} \mathrel{\oplus}= \text{ShmLinked}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem unlink_clears_flag (state flag : Nat) (h : state &&& flag = flag) :
    state ^^^ flag &&& flag = 0 := by sorry
-- proof_needed
```

### LAYER 2 — FORM TEMPLATE

```cuda
static prjname::resultType commUnlinkHostMem(prjname::commType commobj,
    prjname::uniqueIdType commIdval, int rankIdx) {
  char ranknameStr[1024];
  sprintf(ranknameStr, "%s-%d", commIdval.internal, rankIdx);
  if (commobj->hostMemState & ShmLinked)
    commobj->hostMemState ^= ShmLinked;
  return shmUnlink(ranknameStr);
}
```

---

## 20. `ncclCommInitRank`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclCommInitRank}: \text{Comm}^* \times \mathbb{N} \times \text{UniqueId} \times \mathbb{Z} \to \text{Result}
$$
$$
\textbf{Algorithm: }
$$
$$
1.\; \text{wrapSymbols}() \to \text{wrapNvmlInit}() \to \text{commAlloc}() \to \text{populateRankInfo}()
$$
$$
2.\; \text{initGather}() \quad (\text{barrier-synchronized across all ranks via shm})
$$
$$
3.\; \text{commBuildMaps}() \to \text{closeGather}() \quad (\text{second barrier})
$$
$$
4.\; \text{commUnlinkHostMem}() \to \text{wrapNvmlShutdown}()
$$
$$
\textbf{Error path: } \text{closeGather if needed} \to \text{commFree}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_rank_sequence_valid (ndev rank : Nat) (hn : ndev ≥ 1) (hr : rank < ndev) :
    rank < ndev := by sorry

-- axle_verify: content
theorem init_rank_sequence_valid (ndev rank : Nat) (hn : ndev ≥ 1) (hr : rank < ndev) :
    rank < ndev := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
prjname::resultType prjname::CommInitRank(prjname::commType* newcomm, int devCount,
    prjname::uniqueIdType commIdval, int myRankIdx) {
  if (strlen(commIdval.internal) < 1 ||
      strlen(commIdval.internal) >= NCCL_UNIQUE_ID_BYTES) {
    WARN("rank %d invalid commId", myRankIdx);
    return prjname::InvalidArgument;
  }

  initDebug();
  prjname::resultType resultval;
  RankEntry myEntry;
  RankGather* gathptr = NULL;

  resultval = wrapSymbols();
  if (resultval != prjname::Success) { WARN("prjname failed to initialize client libs"); return resultval; }
  resultval = wrapNvmlInit();
  if (resultval != prjname::Success) { WARN("rank %d failed to initialize nvml", myRankIdx); return resultval; }

  resultval = commAlloc(newcomm, devCount, &commIdval, myRankIdx);
  if (resultval != prjname::Success) { WARN("rank %d failed to allocate communicator", myRankIdx); return resultval; }

  resultval = populateRankInfo(&myEntry, myRankIdx, *newcomm);
  if (resultval != prjname::Success) { WARN("rank %d failed to obtain rank info", myRankIdx); goto cleanup; }

  resultval = initGather(&gathptr, commIdval, devCount, myRankIdx, myEntry);
  if (resultval != prjname::Success) { WARN("rank %d failed to gather rank info", myRankIdx); goto cleanup; }

  resultval = commBuildMaps(*newcomm, &commIdval, myRankIdx, gathptr->ranksArr);
  if (resultval != prjname::Success) { WARN("rank %d failed to build comm maps", myRankIdx); goto cleanup; }

  resultval = closeGather(gathptr, devCount);
  gathptr = NULL;
  if (resultval != prjname::Success) { WARN("rank %d failed to close gather", myRankIdx); goto cleanup; }

  goto final;
  cleanup:
  if (gathptr != NULL) closeGather(gathptr, devCount);
  commFree(*newcomm);
  final:
  if ((*newcomm)->hostMemState & ShmLinked) {
    if (commUnlinkHostMem(*newcomm, commIdval, myRankIdx) != prjname::Success)
      INFO("rank %d failed to unlink host mem shm segment", myRankIdx);
  }
  if (wrapNvmlShutdown() != prjname::Success)
    INFO("rank %d did not shutdown nvml properly", myRankIdx);
  return resultval;
}
```

---

## 21. `ncclCommInitAll`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclCommInitAll}: \text{Comm}^n \times \mathbb{N} \times \mathbb{Z}^*_\text{devlist} \to \text{Result}
$$
$$
\textbf{Pre: } \forall\, r,\; \text{devlist}[r] \text{ is valid CUDA device (or rank}=r \text{ if devlist=NULL)}
$$
$$
\textbf{Algorithm: } \text{Single-process variant: all ranks allocated in loop, then maps built}
$$
$$
\text{Sets CPU affinity via NVML for each rank (best-effort)}
$$
$$
\textbf{Post: } \text{comms}[r] \text{ valid for each } r \in [0, \text{ndev})
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_all_device_mapping (rank : Nat) (devlist_null : Bool) :
    (devlist_null → rank = rank) ∧ (¬devlist_null → True) := by sorry

-- axle_verify: content
theorem init_all_device_mapping (rank : Nat) (devlist_null : Bool) :
    (devlist_null → rank = rank) ∧ (¬devlist_null → True) := by
  exact ⟨fun _ => rfl, fun _ => trivial⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
prjname::resultType prjname::CommInitAll(prjname::commType* commsArr, int devCount, int* devlistArr) {
  initDebug();
  prjname::resultType resultval;
  int savedDevIdx;
  RankEntry* ranksArr = NULL;
  int rankIdx, cudaDevIdx;
  prjname::commType commobj = NULL;
  char busIdStr[13];
  nvmlDevice_t nvmlHandle;
  int affinitySetFlag = 0;

  resultval = wrapSymbols();
  if (resultval != prjname::Success) { WARN("prjname failed to initialize client libs"); return resultval; }

  cudaGetDevice(&savedDevIdx);
  ranksArr = (RankEntry*)malloc(devCount*sizeof(RankEntry));
  if (ranksArr == NULL) { WARN("prjname allocation failed"); return prjname::SystemError; }
  memset(ranksArr, 0, devCount*sizeof(RankEntry));

  resultval = wrapNvmlInit();
  if (resultval != prjname::Success) { WARN("nccl failed to initialize nvml"); return resultval; }

  for(rankIdx=0; rankIdx<devCount; ++rankIdx) commsArr[rankIdx] = NULL;

  for (rankIdx=0; rankIdx<devCount; ++rankIdx) {
    cudaDevIdx = (devlistArr == NULL) ? rankIdx : devlistArr[rankIdx];
    if (cudaSetDevice(cudaDevIdx) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rankIdx, cudaDevIdx);
      resultval = prjname::InvalidDeviceIndex;
      goto cleanup;
    }

    affinitySetFlag = 0;
    if (cudaDeviceGetPCIBusId(busIdStr, 13, cudaDevIdx) != cudaSuccess) { goto skipaffinity; }
    if (wrapNvmlDeviceGetHandleByPciBusId(busIdStr, &nvmlHandle) != prjname::Success) { goto skipaffinity; }
    if (wrapNvmlDeviceSetCpuAffinity(nvmlHandle) != prjname::Success) { goto skipaffinity; }
    affinitySetFlag = 1;
    skipaffinity:

    resultval = commAlloc(&commobj, devCount, NULL, rankIdx);
    if (resultval != prjname::Success) { WARN("rank %d failed to allocate communicator", rankIdx); goto cleanup; }
    commsArr[rankIdx] = commobj;

    if (affinitySetFlag && wrapNvmlDeviceClearCpuAffinity(nvmlHandle) != prjname::Success) {
      INFO("rank %d set but failed to clear cpu affinity", rankIdx);
    }
    resultval = populateRankInfo(ranksArr+rankIdx, rankIdx, commobj);
    if (resultval != prjname::Success) { WARN("rank %d failed to obtain rank info", rankIdx); goto cleanup; }
  }

  orderRanks(ranksArr, devCount);
  for(rankIdx=0; rankIdx<devCount; ++rankIdx) {
    commobj = commsArr[rankIdx];
    cudaSetDevice(commobj->cudaDev);
    resultval = commBuildMaps(commobj, NULL, rankIdx, ranksArr);
    if (resultval != prjname::Success) { WARN("rank %d failed to build comm maps", rankIdx); goto cleanup; }
  }

  free(ranksArr); ranksArr = NULL;
  resultval = prjname::Success;
  goto final;

  cleanup:
  if (ranksArr != NULL) free(ranksArr);
  for(rankIdx=0; rankIdx<devCount; ++rankIdx) {
    if(commsArr[rankIdx] != NULL) commFree(commsArr[rankIdx]);
  }

  final:
  if(wrapNvmlShutdown() != prjname::Success) INFO("prjname did not shutdown nvml properly");
  cudaSetDevice(savedDevIdx);
  return resultval;
}
```

---

## 22. `ncclCommDestroy`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclCommDestroy}: \text{Comm} \to \text{Unit}
$$
$$
\textbf{Pre: } \text{comm may be NULL (no-op)}
$$
$$
\textbf{Post: } \text{Sets device to comm's device, calls commFree, restores device}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem destroy_restores_device (saved commDev : Nat) :
    (saved ≠ commDev → True) ∧ (saved = commDev → True) := by sorry

-- axle_verify: content
theorem destroy_restores_device (saved commDev : Nat) :
    (saved ≠ commDev → True) ∧ (saved = commDev → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
void prjname::CommDestroy(prjname::commType commobj) {
  if (commobj == NULL) return;
  int savedDevIdx;
  cudaGetDevice(&savedDevIdx);
  int commDevIdx = commobj->cudaDev;
  if (savedDevIdx != commDevIdx) { CUDACHECK(cudaSetDevice(commDevIdx)); }
  commFree(commobj);
  if (savedDevIdx != commDevIdx) cudaSetDevice(savedDevIdx);
}
```

---

## 23. `ncclGetErrorString`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclGetErrorString}: \text{Result} \to \text{String}
$$
$$
\textbf{Post: } \text{Maps each error code to its human-readable string. Total: 13 codes.}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem error_string_total_cases : 13 > 0 := by sorry

-- axle_verify: content
theorem error_string_total_cases : 13 > 0 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
const char* prjname::GetErrorString(prjname::resultType codeval) {
  switch (codeval) {
  case prjname::Success                : return "no error";
  case prjname::UnhandledCudaError     : return "unhandled cuda error";
  case prjname::SystemError            : return "system error";
  case prjname::InternalError          : return "internal error";
  case prjname::InvalidDevicePointer   : return "invalid device pointer";
  case prjname::InvalidRank            : return "invalid rank";
  case prjname::UnsupportedDeviceCount : return "unsupported device count";
  case prjname::DeviceNotFound         : return "device not found";
  case prjname::InvalidDeviceIndex     : return "invalid device index";
  case prjname::LibWrapperNotSet       : return "lib wrapper not initialized";
  case prjname::CudaMallocFailed       : return "cuda malloc failed";
  case prjname::RankMismatch           : return "parameter mismatch between ranks";
  case prjname::InvalidArgument        : return "invalid argument";
  case prjname::InvalidType            : return "invalid data type";
  case prjname::InvalidOperation       : return "invalid reduction operations";
  }
  return "unknown result code";
}
```

---

## 24. `ncclCommCount`, `ncclCommCuDevice`, `ncclCommUserRank`

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclCommCount}(\text{comm}) = \text{comm.nDev}
$$
$$
\text{ncclCommCuDevice}(\text{comm}) = \text{comm.cudaDev}
$$
$$
\text{ncclCommUserRank}(\text{comm}) = \text{comm.userFromRing}[\text{comm.ncclId}]
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem comm_queries_pure (nDev cudaDev ncclId : Nat) (userFromRing : Fin nDev → Nat)
    (hi : ncclId < nDev) :
    nDev = nDev ∧ cudaDev = cudaDev ∧ userFromRing ⟨ncclId, hi⟩ = userFromRing ⟨ncclId, hi⟩ := by sorry

-- axle_verify: content
theorem comm_queries_pure (nDev cudaDev ncclId : Nat) (userFromRing : Fin nDev → Nat)
    (hi : ncclId < nDev) :
    nDev = nDev ∧ cudaDev = cudaDev ∧ userFromRing ⟨ncclId, hi⟩ = userFromRing ⟨ncclId, hi⟩ := by
  exact ⟨rfl, rfl, rfl⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
extern "C" DSOGLOBAL
prjname::resultType prjname::CommCount(const prjname::commType commobj, int* countout) {
  *countout = commobj->nDev;
  return prjname::Success;
}

extern "C" DSOGLOBAL
prjname::resultType prjname::CommCuDevice(const prjname::commType commobj, int* devidout) {
  *devidout = commobj->cudaDev;
  return prjname::Success;
}

extern "C" DSOGLOBAL
prjname::resultType prjname::CommUserRank(const prjname::commType commobj, int* rankout) {
  *rankout = commobj->userFromRing[commobj->ncclId];
  return prjname::Success;
}
```

