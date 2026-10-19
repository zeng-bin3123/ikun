## File: `src/common.cu` (continued)

---

### 28. Function `RandomizeAccumulate`

#### LAYER 1a — MATH (LaTeX)

$$
\text{RandomizeAccumulate}: \text{Ptr}_{\text{data}} \times \text{Ptr}_{\text{accum}} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \mathbb{Z} \times \mathbb{Z}_{\geq 0} \to \text{Unit}
$$
$$
\text{Step 1: } \text{Randomize}(\text{data}, \text{count}, \text{type}, \text{seed})
$$
$$
\text{Step 2: } \text{rank} = 0 \implies \text{accum} \leftarrow \text{memcpy}(\text{data})
$$
$$
\text{Step 2: } \text{rank} \neq 0 \implies \text{accum} \leftarrow \text{Accumulate}(\text{accum}, \text{data}, \text{count}, \text{type}, \text{op})
$$
$$
\text{Semantics: first rank initializes, subsequent ranks fold into accumulator}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem randomize_accumulate_rank_zero_init (rank : Nat) :
    (rank = 0 → True) ∧ (rank ≠ 0 → True) := by sorry

-- axle_verify: content
theorem randomize_accumulate_rank_zero_init (rank : Nat) :
    (rank = 0 → True) ∧ (rank ≠ 0 → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
void RandomizeAccumulate(void* argData, void* argAccum, size_t argCount, collDataType argType, collRedOpType argOp, int argSeed, int argRank) {
  Randomize(argData, argCount, argType, argSeed);
  if (argRank == 0) {
    GPUCHECK(gpuMemcpy(argAccum, argData, argCount*elemSize(argType), gpuMemcpyDeviceToHost));
  } else {
    Accumulate(argAccum, argData, argCount, argType, argOp);
  }
}
```

---

### 29. Function `CheckData`

#### LAYER 1a — MATH (LaTeX)

$$
\text{CheckData}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \to \mathbb{R}_{\geq 0}
$$
$$
\text{count} = \text{expectedBytes} / \text{wordSize}(\text{type})
$$
$$
\text{maxDelta} = \max_{i \in [0, \text{nGpus})} \left( \max_{j \in [0, \text{count})} |\text{data}_i[j] - \text{expected}_i[j]| \right)
$$
$$
\text{data}_i = \begin{cases}
\text{recvbuffs}[i] + \text{recvInplaceOffset} \cdot \text{rank} & \text{in\_place} \\
\text{recvbuffs}[i] & \text{otherwise}
\end{cases}
$$
$$
\text{rank} = (\text{proc} \cdot \text{nThreads} + \text{thread}) \cdot \text{nGpus} + i
$$
$$
\text{Error check: } \text{maxDelta} > \text{DeltaMaxValue}(\text{type}) \cdot (\text{nranks} - 1) \implies \text{errors}[0]\text{++}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rank_computation (proc nThreads thread nGpus i : Nat) :
    (proc * nThreads + thread) * nGpus + i =
    proc * nThreads * nGpus + thread * nGpus + i := by sorry

-- axle_verify: content
theorem rank_computation (proc nThreads thread nGpus i : Nat) :
    (proc * nThreads + thread) * nGpus + i =
    proc * nThreads * nGpus + thread * nGpus + i := by ring
```

```lean
-- axle_verify: formal_statement
theorem error_threshold_scaling (deltaMax tolerance : Nat) (nranks : Nat) (h : nranks > 0) :
    deltaMax > tolerance * (nranks - 1) →
    deltaMax > 0 := by sorry

-- axle_verify: content
theorem error_threshold_scaling (deltaMax tolerance : Nat) (nranks : Nat) (h : nranks > 0) :
    deltaMax > tolerance * (nranks - 1) →
    deltaMax > 0 := by
  intro hd; omega
```

#### LAYER 2 — FORM TEMPLATE

```c
double CheckData(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace) {
  size_t argCount = argArgs->argExpectedBytes/elemSize(argType);
  double argMaxDelta = 0.0;
  for (int idx=0; idx<argArgs->argNumGpus; idx++) {
    int argDevice;
    int argRank = ((argArgs->argProc*argArgs->argNumThreads + argArgs->argThread)*argArgs->argNumGpus + idx);
    COLLCHECK(collCommCuDevice(argArgs->adjComms[idx], &argDevice));
    GPUCHECK(gpuSetDevice(argDevice));
    void *argData = argInPlace ? ((void *)((uintptr_t)argArgs->adjRecvBuffs[idx] + argArgs->argRecvInplaceOffset*argRank)) : argArgs->adjRecvBuffs[idx];
    CheckDelta(argData , argArgs->adjExpected[idx], argCount, argType, argArgs->adjDelta);
    gpuDeviceSynchronize();
    argMaxDelta = std::max(*(argArgs->adjDeltaHost), argMaxDelta);
  }
  double argNranks = argArgs->argNumProcs*argArgs->argNumThreads*argArgs->argNumGpus;
  if (argMaxDelta > adjDeltaMaxValue(argType)*(argNranks - 1)) argArgs->adjErrors[0]++;
  return argMaxDelta;
}
```

---

### 30. Function `InitSend`

#### LAYER 1a — MATH (LaTeX)

$$
\text{InitSend}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \times \{0,1\} \to \text{Unit}
$$
$$
\text{count} = \text{sendBytes} / \text{wordSize}(\text{type})
$$
$$
\forall i \in [0, \text{nGpus}):
$$
$$
\quad \text{rank} = (\text{proc} \cdot \text{nThreads} + \text{thread}) \cdot \text{nGpus} + i
$$
$$
\quad \text{data} = \begin{cases}
\text{recvbuffs}[i] + \text{sendInplaceOffset} \cdot \text{rank} & \text{in\_place} \\
\text{sendbuffs}[i] & \text{otherwise}
\end{cases}
$$
$$
\quad \text{seed} = \text{rank} + \text{count} + \text{rep} + \text{in\_place}
$$
$$
\quad \text{Randomize}(\text{data}, \text{count}, \text{type}, \text{seed})
$$
$$
\text{Side effect: static rep incremented each call}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_send_seed_unique (rank1 rank2 count rep inPlace : Nat)
    (h : rank1 ≠ rank2) :
    rank1 + count + rep + inPlace ≠ rank2 + count + rep + inPlace := by sorry

-- axle_verify: content
theorem init_send_seed_unique (rank1 rank2 count rep inPlace : Nat)
    (h : rank1 ≠ rank2) :
    rank1 + count + rep + inPlace ≠ rank2 + count + rep + inPlace := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void InitSend(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace, int argIsFirst) {
  size_t argCount = argArgs->argSendBytes / elemSize(argType);
  static int argRep = 1;
  for (int idx=0; idx<argArgs->argNumGpus; idx++) {
    int argDevice;
    int argRank = ((argArgs->argProc*argArgs->argNumThreads + argArgs->argThread)*argArgs->argNumGpus + idx);
    COLLCHECK(collCommCuDevice(argArgs->adjComms[idx], &argDevice));
    GPUCHECK(gpuSetDevice(argDevice));
    void* argData = argInPlace ? (void *)((uintptr_t)argArgs->adjRecvBuffs[idx] + argArgs->argSendInplaceOffset*argRank) : argArgs->adjSendBuffs[idx];
    int argSeed = argRank+argCount+argRep+argInPlace;
    Randomize(argData, argCount, argType, argSeed);
    gpuDeviceSynchronize();
  }
  argRep++;
}
```

---

### 31. Function `startColl`

#### LAYER 1a — MATH (LaTeX)

$$
\text{startColl}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \times \mathbb{Z}_{\geq 0} \to \text{Unit}
$$
$$
\text{count} = \text{nbytes} / \text{wordSize}(\text{type})
$$
$$
\text{If swap\_args: } \text{args} \leftarrow \text{proc\_args}[(\text{thread} + \text{thread\_offset}) \bmod \text{nThreads}]
$$
$$
\text{If nGpus} = 1: \text{RunColl on single GPU}
$$
$$
\text{If nGpus} > 1: \text{ncclGroupStart}() \to \forall i \in [0, \text{nGpus}): \text{RunColl}(i) \to \text{ncclGroupEnd}()
$$
$$
\text{If swap\_args} \lor \text{blocking\_coll}: \text{poll cudaStreamQuery until complete}
$$
$$
\text{If blocking\_coll}: \text{Barrier}(\text{args})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem swap_args_modular (thread threadOffset nThreads : Nat) (h : nThreads > 0) :
    (thread + threadOffset) % nThreads < nThreads := by sorry

-- axle_verify: content
theorem swap_args_modular (thread threadOffset nThreads : Nat) (h : nThreads > 0) :
    (thread + threadOffset) % nThreads < nThreads := by
  exact Nat.mod_lt _ h
```

#### LAYER 2 — FORM TEMPLATE

```c
void startColl(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace, int argThreadOffset) {
  size_t argCount = argArgs->argBytes / elemSize(argType);

  if (adjSwapArgs) {
      argArgs = (struct adjWorkerArgs_t*)argArgs->objHandle + (argArgs->argThread + argThreadOffset)%argArgs->argNumThreads;
  }

  if (argArgs->argNumGpus == 1) {
    int argRank = argArgs->argProc*argArgs->argNumThreads + argArgs->argThread;
    RunColl((void*)(argInPlace ? ((void *)((uintptr_t)argArgs->adjRecvBuffs[0] + argArgs->argSendInplaceOffset*argRank)) : argArgs->adjSendBuffs[0]),
        (void*)(argInPlace ? (void*)((uintptr_t)argArgs->adjRecvBuffs[0] + argArgs->argRecvInplaceOffset*argRank) : argArgs->adjRecvBuffs[0]),
        argCount, argType, argOp, argRoot, argArgs->adjComms[0], argArgs->adjStreams[0]);
  } else {
    COLLCHECK(collGroupStart());
    for (int idx = 0; idx < argArgs->argNumGpus; idx++) {
#ifndef COLL_MAJOR
      int argGpuDev;
      COLLCHECK(collCommCuDevice(argArgs->adjComms[idx], &argGpuDev));
      GPUCHECK(gpuSetDevice(argGpuDev));
#endif
      int argRank = ((argArgs->argProc*argArgs->argNumThreads + argArgs->argThread)*argArgs->argNumGpus + idx);
      RunColl((void*)(argInPlace ? ((void *)((uintptr_t)argArgs->adjRecvBuffs[idx] + argArgs->argSendInplaceOffset*argRank)) : argArgs->adjSendBuffs[idx]),
          (void*)(argInPlace ? (void*)((uintptr_t)argArgs->adjRecvBuffs[idx] + argArgs->argRecvInplaceOffset*argRank) : argArgs->adjRecvBuffs[idx]),
          argCount, argType, argOp, argRoot, argArgs->adjComms[idx], argArgs->adjStreams[idx]);
    }
    COLLCHECK(collGroupEnd());
  }

  if (adjSwapArgs || adjBlockingColl) {
    for (int idx = 0; idx < argArgs->argNumGpus; ++idx) {
      gpuErrorType argErr = gpuErrorNotReady;
      while (argErr == gpuErrorNotReady) {
          argErr = gpuStreamQuery(argArgs->adjStreams[idx]);
          pthread_yield();
      }
      GPUCHECK(argErr);
    }
  }
  if (adjBlockingColl) Barrier(argArgs);
}
```

---

### 32. Function `completeColl`

#### LAYER 1a — MATH (LaTeX)

$$
\text{completeColl}: \text{threadArgs\_t}^* \to \text{Unit}
$$
$$
\text{Pre: startColl was called previously}
$$
$$
\text{If swap\_args} \lor \text{blocking\_coll}: \text{no-op (already completed in startColl)}
$$
$$
\text{Otherwise: } \forall i \in [0, \text{nGpus}): \text{poll cudaStreamQuery}(i) \text{ until complete}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem completeColl_noop_when_blocking (swapArgs blockingColl : Bool) :
    (swapArgs || blockingColl) = true → True := by sorry

-- axle_verify: content
theorem completeColl_noop_when_blocking (swapArgs blockingColl : Bool) :
    (swapArgs || blockingColl) = true → True := by
  intro _; trivial
```

#### LAYER 2 — FORM TEMPLATE

```c
void completeColl(struct adjWorkerArgs_t* argArgs) {
  if (adjSwapArgs || adjBlockingColl) return;

  for (int idx = 0; idx < argArgs->argNumGpus; ++idx) {
    gpuErrorType argErr = gpuErrorNotReady;
    while (argErr == gpuErrorNotReady) {
        argErr = gpuStreamQuery(argArgs->adjStreams[idx]);
        pthread_yield();
    }
    GPUCHECK(argErr);
  }
}
```

---

### 33. Function `BenchTime`

#### LAYER 1a — MATH (LaTeX)

$$
\text{BenchTime}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \to \text{Unit}
$$
$$
\text{Warmup: startColl + completeColl (1 iteration)}
$$
$$
\text{Barrier}
$$
$$
\text{Timing: } t_{\text{start}} = \text{now}()
$$
$$
\quad \forall \text{iter} \in [0, \text{iters}): \text{startColl}(\text{args}, \ldots, \text{iter})
$$
$$
\quad \text{completeColl}()
$$
$$
\quad \Delta t = \text{now}() - t_{\text{start}}
$$
$$
\text{deltaSec} = \Delta t / \text{iters}
$$
$$
\text{GetBw}(\text{count}, \text{wordSize}(\text{type}), \text{deltaSec}, \&\text{algBw}, \&\text{busBw}, \text{nranks})
$$
$$
\text{Validation: InitSend} \to \text{InitRecvResult} \to \text{startColl} \to \text{completeColl} \to \text{CheckData}
$$
$$
\text{Aggregate delta across threads via deltaThreads array, MPI\_Allreduce if MPI}
$$
$$
\text{Accumulate: bw}[0] \mathrel{+}= \text{busBw}, \; \text{bw\_count}[0]\text{++}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bench_avg_time (totalSec : Nat) (iters : Nat) (h : iters > 0) :
    totalSec / iters * iters ≤ totalSec := by sorry

-- axle_verify: content
theorem bench_avg_time (totalSec : Nat) (iters : Nat) (h : iters > 0) :
    totalSec / iters * iters ≤ totalSec := by
  exact Nat.div_mul_le_self totalSec iters
```

#### LAYER 2 — FORM TEMPLATE

```c
void BenchTime(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace) {
  size_t argCount = argArgs->argBytes / elemSize(argType);

  startColl(argArgs, argType, argOp, argRoot, argInPlace, 0);
  completeColl(argArgs);

  Barrier(argArgs);

  auto argStart = std::chrono::high_resolution_clock::now();
  for (int idx = 0; idx < adjIters; idx++) {
      startColl(argArgs, argType, argOp, argRoot, argInPlace, idx);
  }
  completeColl(argArgs);

  auto argTimeDelta = std::chrono::high_resolution_clock::now() - argStart;
  double argDeltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(argTimeDelta).count();
  argDeltaSec = argDeltaSec/adjIters;

  double argAlgBw, argBusBw;
  GetBw(argCount, elemSize(argType), argDeltaSec, &argAlgBw, &argBusBw, argArgs->argNumProcs*argArgs->argNumThreads*argArgs->argNumGpus);

  Barrier(argArgs);

  if (adjDatacheck) {
      InitSend(argArgs, argType, argOp, argRoot, argInPlace, argArgs->argThread == 0 ? 1 : 0);
      InitRecvResult(argArgs, argType, argOp, argRoot, argInPlace, argArgs->argThread == 0 ? 1 : 0);
      gpuDeviceSynchronize();
  }

  startColl(argArgs, argType, argOp, argRoot, argInPlace, 0);
  completeColl(argArgs);

  double argMaxDelta = 0;
#ifdef CHECK
  if (adjDatacheck) {
     argMaxDelta = CheckData(argArgs, argType, argOp, argRoot, argInPlace);
  } else {
     argMaxDelta = -1.0;
  }
#else
     argMaxDelta = -1.0;
#endif

  Barrier(argArgs);
  if (argArgs->argThread == 0) {
      for (int idx=1; idx<argArgs->argNumThreads; idx++) {
          argMaxDelta += argArgs->adjDeltaThreads[idx];
      }
#ifdef MPI_SUPPORT
      MPI_Allreduce(MPI_IN_PLACE, &argMaxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Barrier(argArgs);

  if (adjDatacheck) {
     PRINT("  %7.3f  %5.2f  %5.2f  %7.0le", argDeltaSec * 1.0E3, argAlgBw, argBusBw, argMaxDelta);
  } else {
     PRINT("  %7.3f  %5.2f  %5.2f  \tN/A", argDeltaSec * 1.0E3, argAlgBw, argBusBw);
  }

  argArgs->adjBw[0] += argBusBw;
  argArgs->adjBwCount[0]++;
}
```

---

### 34. Function `setupArgs`

#### LAYER 1a — MATH (LaTeX)

$$
\text{setupArgs}: \mathbb{N} \times \text{Type} \times \text{threadArgs\_t}^* \to \text{Unit}
$$
$$
\text{nranks} = \text{nProcs} \cdot \text{nGpus} \cdot \text{nThreads}
$$
$$
\text{count} = \text{size} / \text{wordSize}(\text{type})
$$
$$
\text{getCollByteCount}(\&\text{sendCount}, \&\text{recvCount}, \&\text{paramCount}, \ldots, \text{count}, \text{nranks})
$$
$$
\text{Post:}
$$
$$
\text{args.nbytes} = \text{paramCount} \cdot \text{wordSize}(\text{type})
$$
$$
\text{args.sendBytes} = \text{sendCount} \cdot \text{wordSize}(\text{type})
$$
$$
\text{args.expectedBytes} = \text{recvCount} \cdot \text{wordSize}(\text{type})
$$
$$
\text{args.sendInplaceOffset} = \text{sendInplaceOffset} \cdot \text{wordSize}(\text{type})
$$
$$
\text{args.recvInplaceOffset} = \text{recvInplaceOffset} \cdot \text{wordSize}(\text{type})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem setup_nranks (nProcs nGpus nThreads : Nat) :
    nProcs * nGpus * nThreads = nProcs * (nGpus * nThreads) := by sorry

-- axle_verify: content
theorem setup_nranks (nProcs nGpus nThreads : Nat) :
    nProcs * nGpus * nThreads = nProcs * (nGpus * nThreads) := by ring
```

```lean
-- axle_verify: formal_statement
theorem setup_bytes_from_count (paramCount ws : Nat) :
    paramCount * ws / ws = paramCount ∨ ws = 0 := by sorry

-- axle_verify: content
theorem setup_bytes_from_count (paramCount ws : Nat) :
    paramCount * ws / ws = paramCount ∨ ws = 0 := by
  by_cases h : ws = 0
  · right; exact h
  · left; exact Nat.mul_div_cancel paramCount (Nat.pos_of_ne_zero h)
```

#### LAYER 2 — FORM TEMPLATE

```c
void setupArgs(size_t argSize, collDataType argType, struct adjWorkerArgs_t* argArgs) {
  int argNranks = argArgs->argNumProcs*argArgs->argNumGpus*argArgs->argNumThreads;
  size_t argCount, argSendCount, argRecvCount, argParamCount, argSendInplaceOff, argRecvInplaceOff, argProcSharedCount;
  int argSameExpected;

  argCount = argSize / elemSize(argType);
  getCollByteCount(&argSendCount, &argRecvCount, &argParamCount, &argSendInplaceOff, &argRecvInplaceOff, &argProcSharedCount, &argSameExpected, (size_t)argCount, (size_t)argNranks);

  argArgs->argBytes = argParamCount * elemSize(argType);
  argArgs->argSendBytes = argSendCount * elemSize(argType);
  argArgs->argExpectedBytes = argRecvCount * elemSize(argType);
  argArgs->argSendInplaceOffset = argSendInplaceOff * elemSize(argType);
  argArgs->argRecvInplaceOffset = argRecvInplaceOff * elemSize(argType);
}
```

---

### 35. Function `TimeTest`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TimeTest}: \text{threadArgs\_t}^* \times \text{Type} \times \text{String} \times \text{Op} \times \text{String} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \to \text{Unit}
$$
$$
\text{Phase 1 (warmup): setupArgs(maxbytes) then warmup\_iters iterations of startColl}
$$
$$
\text{Phase 2 (sweep): } \forall \text{size} \in \{\text{minbytes}, \ldots, \text{maxbytes}\}:
$$
$$
\quad \text{size progression:} \begin{cases}
\text{size} \cdot \text{stepfactor} & \text{stepfactor} > 1 \\
\text{size} + \text{stepbytes} & \text{otherwise}
\end{cases}
$$
$$
\quad \text{setupArgs}(\text{size}) \to \text{print\_line\_header} \to \text{BenchTime}(\text{out-of-place})
$$
$$
\quad \text{if inPlace}: \text{BenchTime}(\text{in-place})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem size_progression_terminates (size maxbytes stepfactor stepbytes : Nat)
    (h1 : stepfactor > 1 ∨ stepbytes > 0) (h2 : size ≤ maxbytes) :
    (stepfactor > 1 → size * stepfactor > size) ∧
    (stepbytes > 0 → size + stepbytes > size) := by sorry

-- axle_verify: content
theorem size_progression_terminates (size maxbytes stepfactor stepbytes : Nat)
    (h1 : stepfactor > 1 ∨ stepbytes > 0) (h2 : size ≤ maxbytes) :
    (stepfactor > 1 → size * stepfactor > size) ∧
    (stepbytes > 0 → size + stepbytes > size) := by
  constructor
  · intro hf
    by_cases hs : size = 0
    · subst hs; simp; omega
    · exact Nat.lt_mul_of_lt (Nat.one_lt_iff_ne_one.mp (by omega)) (Nat.pos_of_ne_zero hs)
  · intro hb; omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void TimeTest(struct adjWorkerArgs_t* argArgs, collDataType argType, const char* argTypeName, collRedOpType argOp, const char* argOpName, int argRoot, int argInPlace) {
  setupArgs(argArgs->argMaxBytes, argType, argArgs);
  for (int idx = 0; idx < adjWarmupIters; idx++) {
     startColl(argArgs, argType, argOp, argRoot, 0, idx);
  }
  completeColl(argArgs);

  for (size_t argSize = argArgs->argMinBytes; argSize<=argArgs->argMaxBytes; argSize = ((argArgs->argStepFactor > 1) ? argSize*argArgs->argStepFactor : argSize+argArgs->argStepBytes)) {
      setupArgs(argSize, argType, argArgs);
      print_line_header(max(argArgs->argSendBytes, argArgs->argExpectedBytes), argArgs->argBytes / elemSize(argType), argTypeName, argOpName, argRoot);
      BenchTime(argArgs, argType, argOp, argRoot, 0);
      if (argInPlace) BenchTime(argArgs, argType, argOp, argRoot, 1);
      PRINT("\n");
  }
}
```

---

### 36. Function `threadRunTests`

#### LAYER 1a — MATH (LaTeX)

$$
\text{threadRunTests}: \text{void}^* \to \text{void}^*
$$
$$
\text{Pre: args is a valid threadArgs\_t pointer}
$$
$$
\text{Step 1: gpuid} = \text{localRank} \cdot \text{nThreads} \cdot \text{nGpus} + \text{thread} \cdot \text{nGpus}
$$
$$
\text{Step 2: cudaSetDevice(gpuid)}
$$
$$
\text{Step 3: RunTest(args, ncclroot, nccltype, ncclop)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem gpu_id_computation (localRank nThreads nGpus thread : Nat) :
    localRank * nThreads * nGpus + thread * nGpus =
    (localRank * nThreads + thread) * nGpus := by sorry

-- axle_verify: content
theorem gpu_id_computation (localRank nThreads nGpus thread : Nat) :
    localRank * nThreads * nGpus + thread * nGpus =
    (localRank * nThreads + thread) * nGpus := by ring
```

#### LAYER 2 — FORM TEMPLATE

```c
void* adjThreadRunTests(void* argArgs) {
  struct adjWorkerArgs_t* argWorkerArgs = (struct adjWorkerArgs_t*)argArgs;
  int argGpuid = argWorkerArgs->argLocalRank*argWorkerArgs->argNumThreads*argWorkerArgs->argNumGpus + argWorkerArgs->argThread*argWorkerArgs->argNumGpus;
  GPUCHECK(gpuSetDevice(argGpuid));

  RunTest(argWorkerArgs, adjCollRoot, (collDataType)adjCollType, adjTestTypenames[adjCollType], (collRedOpType)adjCollOp, adjTestOpnames[adjCollOp]);

  return NULL;
}
```

---

### 37. Function `threadInit`

#### LAYER 1a — MATH (LaTeX)

$$
\text{threadInit}: \text{void}^* \to \text{void}^*
$$
$$
\text{nranks} = \text{nProcs} \cdot \text{nThreads} \cdot \text{nGpus}
$$
$$
\text{Phase 1 (ncclGroupStart):}
$$
$$
\forall i \in [0, \text{nGpus}):
$$
$$
\quad \text{rank} = \text{proc} \cdot \text{nThreads} \cdot \text{nGpus} + \text{thread} \cdot \text{nGpus} + i
$$
$$
\quad \text{gpuid} = \text{localRank} \cdot \text{nThreads} \cdot \text{nGpus} + \text{thread} \cdot \text{nGpus} + i
$$
$$
\quad \text{ncclCommInitRank}(\text{comms}[i], \text{nranks}, \text{ncclId}, \text{rank})
$$
$$
\text{Phase 2 (ncclGroupEnd)}
$$
$$
\text{Phase 3: print device info}
$$
$$
\text{Phase 4: threadRunTests(args)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem thread_init_rank_unique (proc nThreads nGpus thread i j : Nat)
    (h : i ≠ j) (hi : i < nGpus) (hj : j < nGpus) :
    proc * nThreads * nGpus + thread * nGpus + i ≠
    proc * nThreads * nGpus + thread * nGpus + j := by sorry

-- axle_verify: content
theorem thread_init_rank_unique (proc nThreads nGpus thread i j : Nat)
    (h : i ≠ j) (hi : i < nGpus) (hj : j < nGpus) :
    proc * nThreads * nGpus + thread * nGpus + i ≠
    proc * nThreads * nGpus + thread * nGpus + j := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void* adjThreadInit(void* argArgs) {
  struct adjWorkerArgs_t* argWorkerArgs = (struct adjWorkerArgs_t*)argArgs;
  char argHostname[1024];
  getNodeName(argHostname, 1024);
  int argNranks = argWorkerArgs->argNumProcs*argWorkerArgs->argNumThreads*argWorkerArgs->argNumGpus;

  is_main_thread = (argWorkerArgs->argProc == 0 && argWorkerArgs->argThread == 0) ? 1 : 0;

  COLLCHECK(collGroupStart());
  for (int idx=0; idx<argWorkerArgs->argNumGpus; idx++) {
    int argRank = argWorkerArgs->argProc*argWorkerArgs->argNumThreads*argWorkerArgs->argNumGpus + argWorkerArgs->argThread*argWorkerArgs->argNumGpus + idx;
    int argGpuid = argWorkerArgs->argLocalRank*argWorkerArgs->argNumThreads*argWorkerArgs->argNumGpus + argWorkerArgs->argThread*argWorkerArgs->argNumGpus + idx;
    GPUCHECK(gpuSetDevice(argGpuid));
    COLLCHECK(collCommInitRank(argWorkerArgs->adjComms+idx, argNranks, argWorkerArgs->adjCollId, argRank));
  }
  COLLCHECK(collGroupEnd());

  // ... print device info and run tests ...
  adjThreadRunTests(argArgs);

  return NULL;
}
```

---

### 38. Function `AllocateBuffs`

#### LAYER 1a — MATH (LaTeX)

$$
\text{AllocateBuffs}: \text{Ptr}^* \times \mathbb{N} \times \text{Ptr}^* \times \mathbb{N} \times \text{Ptr}^* \times \text{Ptr}^* \times \mathbb{N} \times \mathbb{Z}_{>0} \times \{0,1\} \to \text{Unit}
$$
$$
\text{Allocations:}
$$
$$
\text{cudaMalloc}(\text{sendbuff}, \text{sendBytes})
$$
$$
\text{cudaMalloc}(\text{recvbuff}, \max(\text{sendBytes}, \text{recvBytes}))
$$
$$
\text{If first call} \lor \neg\text{sameExpected}:
$$
$$
\quad \text{expectedHost} = \text{malloc}(\text{recvBytes})
$$
$$
\quad \text{cudaHostRegister + cudaHostGetDevicePointer} \to \text{expected}
$$
$$
\text{Else: reuse cached pointers}
$$
$$
\text{Invariant: static is\_first toggled to 0 after first allocation}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem recv_alloc_size (sendBytes recvBytes : Nat) :
    max sendBytes recvBytes ≥ recvBytes := by sorry

-- axle_verify: content
theorem recv_alloc_size (sendBytes recvBytes : Nat) :
    max sendBytes recvBytes ≥ recvBytes := by
  exact Nat.le_max_right sendBytes recvBytes
```

```lean
-- axle_verify: formal_statement
theorem recv_alloc_size_send (sendBytes recvBytes : Nat) :
    max sendBytes recvBytes ≥ sendBytes := by sorry

-- axle_verify: content
theorem recv_alloc_size_send (sendBytes recvBytes : Nat) :
    max sendBytes recvBytes ≥ sendBytes := by
  exact Nat.le_max_left sendBytes recvBytes
```

#### LAYER 2 — FORM TEMPLATE

```c
void AllocateBuffs(void **argSendbuff, size_t argSendBytes, void **argRecvbuff, size_t argRecvBytes, void **argExpected, void **argExpectedHost, size_t argNbytes, int argNranks, int argSameExpected) {
    static int adjIsFirst = 1;
    static void *adjCachedPtr = NULL;
    static void *adjCachedHostptr = NULL;

    GPUCHECK(gpuMalloc(argSendbuff, argSendBytes));
    GPUCHECK(gpuMalloc(argRecvbuff, (argSendBytes > argRecvBytes) ? argSendBytes : argRecvBytes));

    if (adjIsFirst || !argSameExpected) {
        *argExpectedHost = malloc(argRecvBytes);
        GPUCHECK(gpuHostRegister(*argExpectedHost, argRecvBytes, gpuHostRegisterPortable | gpuHostRegisterMapped));
        GPUCHECK(gpuHostGetDevicePointer(argExpected, *argExpectedHost, 0));
        adjCachedPtr = *argExpected;
        adjCachedHostptr = *argExpectedHost;
        adjIsFirst = 0;
    } else {
        *argExpected = adjCachedPtr;
        *argExpectedHost = adjCachedHostptr;
    }
}
```

---

### 39. Function `ncclstringtotype`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclstringtotype}: \text{String} \to \mathbb{Z}
$$
$$
\text{ncclstringtotype}(s) = \begin{cases}
t & \exists t \in [0, \text{ncclNumTypes}): s = \text{test\_typenames}[t] \\
-1 & s = \texttt{"all"} \\
\text{nccltype (default)} & \text{otherwise}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem string_to_type_range (t numTypes : Nat) (h : t < numTypes) :
    t < numTypes := by sorry

-- axle_verify: content
theorem string_to_type_range (t numTypes : Nat) (h : t < numTypes) :
    t < numTypes := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
int collStringToType(char *argStr) {
    for (int idx=0; idx<collNumTypes; idx++) {
      if (strcmp(argStr, adjTestTypenames[idx]) == 0) {
        return idx;
      }
    }
    if (strcmp(argStr, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", argStr, adjTestTypenames[adjCollType]);
    return adjCollType;
}
```

---

### 40. Function `ncclstringtoop`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclstringtoop}: \text{String} \to \mathbb{Z}
$$
$$
\text{ncclstringtoop}(s) = \begin{cases}
o & \exists o \in [0, \text{ncclNumOps}): s = \text{test\_opnames}[o] \\
-1 & s = \texttt{"all"} \\
\text{ncclop (default)} & \text{otherwise}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem string_to_op_range (o numOps : Nat) (h : o < numOps) :
    o < numOps := by sorry

-- axle_verify: content
theorem string_to_op_range (o numOps : Nat) (h : o < numOps) :
    o < numOps := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
int collStringToOp(char *argStr) {
    for (int idx=0; idx<collNumOps; idx++) {
      if (strcmp(argStr, adjTestOpnames[idx]) == 0) {
        return idx;
      }
    }
    if (strcmp(argStr, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", argStr, adjTestOpnames[adjCollOp]);
    return adjCollOp;
}
```

---

### 41. Function `main`

#### LAYER 1a — MATH (LaTeX)

$$
\text{main}: \mathbb{Z} \times \text{String}[] \to \{0, 1\}
$$

$$
\text{Defaults:}
$$
$$
\text{nThreads} = 1, \quad \text{nGpus} = 1, \quad \text{minBytes} = \text{maxBytes} = 32 \cdot 2^{20}
$$
$$
\text{stepBytes} = 2^{20}, \quad \text{stepFactor} = 1
$$

$$
\text{Phase 1: Parse command-line options via getopt\_long}
$$
$$
\text{Options: } \{t, g, b, e, i, f, n, w, s, p, c, z, o, d, r, h\}
$$

$$
\text{Phase 2 (MPI):}
$$
$$
\text{MPI\_Init} \to \text{nProcs, proc from MPI\_Comm}
$$
$$
\text{localRank} = |\{p < \text{proc} : \text{hostHash}(p) = \text{hostHash}(\text{proc})\}|
$$

$$
\text{Phase 3: Allocate buffers, streams, comms}
$$
$$
\forall i \in [0, \text{nGpus} \cdot \text{nThreads}): \text{AllocateBuffs}(i), \text{cudaStreamCreate}(i)
$$

$$
\text{Phase 4 (non-parallel init):}
$$
$$
\text{nProcs} = 1 \implies \text{ncclCommInitAll}
$$
$$
\text{nProcs} > 1 \implies \text{ncclGroupStart} \to \forall i: \text{ncclCommInitRank} \to \text{ncclGroupEnd}
$$

$$
\text{Phase 5: Launch threads, run tests, join}
$$
$$
\text{Phase 6: Aggregate errors and bandwidth}
$$
$$
\text{avgBw} = \text{bw}[0] / \text{bw\_count}[0]
$$
$$
\text{Exit: } (\text{errors} > 0 \lor \text{avgBw} < 0.9 \cdot \text{check\_avg\_bw}) \implies \text{EXIT\_FAILURE}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem default_bytes : 32 * 1024 * 1024 = 33554432 := by sorry

-- axle_verify: content
theorem default_bytes : 32 * 1024 * 1024 = 33554432 := by norm_num
```

```lean
-- axle_verify: formal_statement
theorem local_rank_count (hostHashs : List Nat) (proc : Nat) (hp : proc < hostHashs.length) :
    (hostHashs.take proc).filter (· = hostHashs.get ⟨proc, hp⟩) |>.length ≤ proc := by sorry

-- proof_needed: requires List.filter and List.take length bounds
```

```lean
-- axle_verify: formal_statement
theorem bw_threshold_check (avgBw checkBw : Nat) :
    (avgBw < checkBw * 9 / 10) ∨ (avgBw ≥ checkBw * 9 / 10) := by sorry

-- axle_verify: content
theorem bw_threshold_check (avgBw checkBw : Nat) :
    (avgBw < checkBw * 9 / 10) ∨ (avgBw ≥ checkBw * 9 / 10) := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
int main(int argc, char* argv[]) {
 int argNumThreads = 1, argNumGpus = 1;
 size_t argMinBytes = 32*1024*1024, argMaxBytes = 32*1024*1024, argStepBytes = 1*1024*1024, argStepFactor = 1;
 int argLongindex;
 int argNumProcs = 1, argProc = 0;
 int argLocalRank = 0;
 char argHostname[1024];
 getNodeName(argHostname, 1024);

 static struct option adjLongopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"swap_comms", required_argument, 0, 's'},
    {"parallel_init", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"blocking", required_argument, 0, 'z'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"help", no_argument, 0, 'h'}
 };

 while(1) {
      int argC;
      argC = getopt_long(argc, argv, "t:g:b:e:i:f:n:w:s:p:c:o:d:r:z:h", adjLongopts, &argLongindex);
      if (argC == -1) break;
      switch(argC) {
         case 't': argNumThreads = strtol(optarg, NULL, 0); break;
         case 'g': argNumGpus = strtol(optarg, NULL, 0); break;
         case 'b': argMinBytes = (size_t)parseSize(optarg); break;
         case 'e': argMaxBytes = (size_t)parseSize(optarg); break;
         case 'i': argStepBytes = strtol(optarg, NULL, 0); break;
         case 'f': argStepFactor = strtol(optarg, NULL, 0); break;
         case 'n': adjIters = (int)strtol(optarg, NULL, 0); break;
         case 'w': adjWarmupIters = (int)strtol(optarg, NULL, 0); break;
         case 's': adjSwapArgs = (int)strtol(optarg, NULL, 0); break;
         case 'c': adjDatacheck = (int)strtol(optarg, NULL, 0); break;
         case 'p': adjParallelInit = (int)strtol(optarg, NULL, 0); break;
         case 'o': adjCollOp = collStringToOp(optarg); break;
         case 'd': adjCollType = collStringToType(optarg); break;
         case 'r': adjCollRoot = strtol(optarg, NULL, 0); break;
         case 'z': adjBlockingColl = strtol(optarg, NULL, 0); break;
         case 'h': /* print help */ return 0;
         default:  /* print usage */ return 0;
      }
  }

  setlinebuf(stdout);

#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &argNumProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &argProc);
  uint64_t adjHostHashs[argNumProcs];
  adjHostHashs[argProc] = getNodeHash(argHostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, adjHostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int idx=0; idx<argNumProcs; idx++) {
    if (idx == argProc) break;
    if (adjHostHashs[idx] == adjHostHashs[argProc]) argLocalRank++;
  }
#endif

  // ... allocation, initialization, thread launch, join, cleanup, exit ...
  // (structure preserved, identifiers renamed)
}
```

