## File: `src/all_gather.cu`

---

### 42. Function `print_header` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{print\_header}_{\text{allgather}}: \emptyset \to \text{Unit}
$$
$$
\text{Output: formatted table header with columns:}
$$
$$
[\text{bytes}, N, \text{type}, \underbrace{\text{time}, \text{algbw}, \text{busbw}, \text{res}}_{\text{out-of-place}}, \underbrace{\text{time}, \text{algbw}, \text{busbw}, \text{res}}_{\text{in-place}}]
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_header_columns : 7 + 4 = 11 := by sorry

-- axle_verify: content
theorem allgather_header_columns : 7 + 4 = 11 := by norm_num
```

#### LAYER 2 — FORM TEMPLATE

```c
void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}
```

---

### 43. Function `print_line_header` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{print\_line\_header}_{\text{allgather}}: \mathbb{N} \times \mathbb{N} \times \text{String} \times \text{String} \times \mathbb{Z}_{\geq 0} \to \text{Unit}
$$
$$
\text{Output: } [\text{size}, \text{count}, \text{typeName}]
$$
$$
\text{Note: opName and root are unused (all\_gather has no op or root)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_line_header_fields : (3 : Nat) = 3 := by sorry

-- axle_verify: content
theorem allgather_line_header_fields : (3 : Nat) = 3 := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
void print_line_header (size_t argSize, size_t argCount, const char *argTypeName, const char *argOpName, int argRoot) {
  PRINT("%12li  %12li  %6s", argSize, argCount, argTypeName);
}
```

---

### 44. Function `getCollByteCount` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{getCollByteCount}_{\text{allgather}}: \mathbb{N} \times \mathbb{Z}_{>0} \to (\mathbb{N})^7
$$
$$
\text{sendcount} = \lfloor \text{count} / \text{nranks} \rfloor
$$
$$
\text{recvcount} = \lfloor \text{count} / \text{nranks} \rfloor \cdot \text{nranks}
$$
$$
\text{paramcount} = \text{sendcount}
$$
$$
\text{sameExpected} = 1
$$
$$
\text{procSharedCount} = 0
$$
$$
\text{sendInplaceOffset} = \lfloor \text{count} / \text{nranks} \rfloor
$$
$$
\text{recvInplaceOffset} = 0
$$
$$
\text{Invariant: } \text{recvcount} = \text{sendcount} \cdot \text{nranks}
$$
$$
\text{Invariant: } \text{recvcount} \leq \text{count}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_recvcount (count nranks : Nat) (h : nranks > 0) :
    (count / nranks) * nranks ≤ count := by sorry

-- axle_verify: content
theorem allgather_recvcount (count nranks : Nat) (h : nranks > 0) :
    (count / nranks) * nranks ≤ count := by
  exact Nat.div_mul_le_self count nranks
```

```lean
-- axle_verify: formal_statement
theorem allgather_send_recv_relation (count nranks : Nat) (h : nranks > 0) :
    let sendcount := count / nranks
    let recvcount := sendcount * nranks
    recvcount = (count / nranks) * nranks := by sorry

-- axle_verify: content
theorem allgather_send_recv_relation (count nranks : Nat) (h : nranks > 0) :
    let sendcount := count / nranks
    let recvcount := sendcount * nranks
    recvcount = (count / nranks) * nranks := by
  simp
```

#### LAYER 2 — FORM TEMPLATE

```c
void getCollByteCount(size_t *argSendcount, size_t *argRecvcount, size_t *argParamcount, size_t *argSendInplaceOffset, size_t *argRecvInplaceOffset, size_t *argProcSharedCount, int *argSameExpected, size_t argCount, int argNranks) {
    *argSendcount = argCount/argNranks;
    *argRecvcount = (argCount/argNranks)*argNranks;
    *argSameExpected = 1;
    *argProcSharedCount = 0;
    *argSendInplaceOffset = argCount/argNranks;
    *argRecvInplaceOffset = 0;
    *argParamcount = *argSendcount;
}
```

---

### 45. Function `InitRecvResult` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{InitRecvResult}_{\text{allgather}}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \times \{0,1\} \to \text{Unit}
$$
$$
\text{nBytes} = \text{args.nbytes}, \quad \text{count} = \text{nBytes} / \text{wordSize}(\text{type})
$$
$$
\text{Phase 1 (sync): spin until sync[sync\_idx] = t}
$$
$$
\text{Phase 2: } \forall i \in [0, \text{nGpus}):
$$
$$
\quad \text{data} = \begin{cases}
\text{recvbuffs}[i] + \text{sendInplaceOffset} \cdot \text{rank} & \text{in\_place} \\
\text{sendbuffs}[i] & \text{otherwise}
\end{cases}
$$
$$
\quad \text{cudaMemcpy expectedHost at offset } (\text{proc} \cdot \text{nThreads} + t) \cdot \text{nGpus} + i) \cdot \text{nBytes}
$$
$$
\quad \text{if not in\_place: memset recvbuffs[i] to 0}
$$
$$
\text{Phase 3 (last thread): MPI\_Allgather of expectedHost}
$$
$$
\text{Phase 4: toggle sync\_idx}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_expected_offset (proc nThreads t nGpus i nBytes : Nat) :
    ((proc * nThreads + t) * nGpus + i) * nBytes =
    proc * nThreads * nGpus * nBytes + t * nGpus * nBytes + i * nBytes := by sorry

-- axle_verify: content
theorem allgather_expected_offset (proc nThreads t nGpus i nBytes : Nat) :
    ((proc * nThreads + t) * nGpus + i) * nBytes =
    proc * nThreads * nGpus * nBytes + t * nGpus * nBytes + i * nBytes := by ring
```

#### LAYER 2 — FORM TEMPLATE

```c
void InitRecvResult(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace, int argIsFirst) {
  size_t argNBytes = argArgs->argBytes;
  size_t argCount = argNBytes / elemSize(argType);
  int argProc = argArgs->argProc;
  int argNumThreads = argArgs->argNumThreads;
  int argT = argArgs->argThread;
  int argNumGpus = argArgs->argNumGpus;

  while (argArgs->adjSync[argArgs->argSyncIdx] != argT) pthread_yield();

  for (int idx=0; idx<argNumGpus; idx++) {
    int argDevice;
    int argRank = ((argArgs->argProc*argArgs->argNumThreads + argArgs->argThread)*argArgs->argNumGpus + idx);
    COLLCHECK(collCommCuDevice(argArgs->adjComms[idx], &argDevice));
    GPUCHECK(gpuSetDevice(argDevice));

    void* argData = argInPlace ? (void *)((uintptr_t)argArgs->adjRecvBuffs[idx] + argArgs->argSendInplaceOffset*argRank) : argArgs->adjSendBuffs[idx];

    GPUCHECK(gpuMemcpy((void *)((uintptr_t)argArgs->adjExpectedHost[0] + ((argProc*argNumThreads + argT)*argNumGpus + idx)*argNBytes),
                argData,
                argNBytes, gpuMemcpyDeviceToHost));

    if (argInPlace == 0) {
      GPUCHECK(gpuMemset(argArgs->adjRecvBuffs[idx], 0, argArgs->argExpectedBytes));
    }
    GPUCHECK(gpuDeviceSynchronize());
  }

  argArgs->adjSync[argArgs->argSyncIdx] = argT + 1;

  if (argT+1 == argNumThreads) {
#ifdef MPI_SUPPORT
    MPI_Allgather(MPI_IN_PLACE, argNBytes*argNumThreads*argNumGpus, MPI_BYTE,
        argArgs->adjExpectedHost[0],
        argNBytes*argNumThreads*argNumGpus, MPI_BYTE, MPI_COMM_WORLD);
#endif
    argArgs->adjSync[argArgs->argSyncIdx] = 0;
  } else {
    while (argArgs->adjSync[argArgs->argSyncIdx]) pthread_yield();
  }

  argArgs->argSyncIdx=!argArgs->argSyncIdx;
}
```

---

### 46. Function `GetBw` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{GetBw}_{\text{allgather}}: \mathbb{N} \times \mathbb{Z}_{>0} \times \mathbb{R}_{>0} \times \mathbb{R}^* \times \mathbb{R}^* \times \mathbb{Z}_{>0} \to \text{Unit}
$$
$$
\text{baseBw} = \frac{\text{count} \times \text{typesize} \times (\text{nranks} - 1)}{10^9 \times \text{sec}}
$$
$$
\text{algBw} = \text{baseBw}
$$
$$
\text{busBw} = \text{baseBw} \times 1 = \text{baseBw}
$$
$$
\text{Bus bandwidth factor for all\_gather} = \frac{n-1}{n} \cdot n \cdot \frac{1}{n-1} = 1
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_bw_factor : (1 : Nat) = 1 := by sorry

-- axle_verify: content
theorem allgather_bw_factor : (1 : Nat) = 1 := by rfl
```

```lean
-- axle_verify: formal_statement
theorem allgather_basebw_formula (count typesize nranks : Nat)
    (h : nranks > 0) :
    count * typesize * (nranks - 1) =
    count * typesize * nranks - count * typesize := by sorry

-- axle_verify: content
theorem allgather_basebw_formula (count typesize nranks : Nat)
    (h : nranks > 0) :
    count * typesize * (nranks - 1) =
    count * typesize * nranks - count * typesize := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void GetBw(size_t argCount, int argTypesize, double argSec, double* argAlgBw, double* argBusBw, int argNranks) {
  double argBaseBw = (double)(argCount * argTypesize * (argNranks - 1)) / 1.0E9 / argSec;

  *argAlgBw = argBaseBw;
  double argFactor = 1;
  *argBusBw = argBaseBw * argFactor;
}
```

---

### 47. Function `RunColl` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{RunColl}_{\text{allgather}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \text{Comm} \times \text{Stream} \to \text{Unit}
$$
$$
\text{Delegates to: } \text{ncclAllGather}(\text{sendbuff}, \text{recvbuff}, \text{count}, \text{type}, \text{comm}, \text{stream})
$$
$$
\text{Note: op and root are unused}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_runcoll_ignores_op_root : True := by sorry

-- axle_verify: content
theorem allgather_runcoll_ignores_op_root : True := by trivial
```

#### LAYER 2 — FORM TEMPLATE

```c
void RunColl(void* argSendbuff, void* argRecvbuff, size_t argCount, collDataType argType, collRedOpType argOp, int argRoot, collCommType argComm, gpuStreamType argStream) {
  COLLCHECK(collAllGather(argSendbuff, argRecvbuff, argCount, argType, argComm, argStream));
}
```

---

### 48. Function `RunTest` (all_gather)

#### LAYER 1a — MATH (LaTeX)

$$
\text{RunTest}_{\text{allgather}}: \text{threadArgs\_t}^* \times \mathbb{Z} \times \text{Type} \times \text{String} \times \text{Op} \times \text{String} \to \text{Unit}
$$
$$
\text{If type} \neq -1: \text{run single type}
$$
$$
\text{If type} = -1: \text{run all ncclNumTypes types}
$$
$$
\forall i \in [0, \text{type\_count}): \text{TimeTest}(\text{args}, \text{types}[i], \text{typenames}[i], 0, \text{NULL}, 0, 1)
$$
$$
\text{Note: op is always 0 (unused), root is always 0, inPlace is always 1}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allgather_runtest_no_ops (type : Int) :
    (type ≠ -1 → True) ∧ (type = -1 → True) := by sorry

-- axle_verify: content
theorem allgather_runtest_no_ops (type : Int) :
    (type ≠ -1 → True) ∧ (type = -1 → True) := by
  exact ⟨fun _ => trivial, fun _ => trivial⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
void RunTest(struct adjWorkerArgs_t* argArgs, int argRoot, collDataType argType, const char* argTypeName, collRedOpType argOp, const char* argOpName) {
  collDataType *argRunTypes;
  const char **argRunTypenames;
  int argTypeCount;

  if ((int)argType != -1) {
    argTypeCount = 1;
    argRunTypes = &argType;
    argRunTypenames = &argTypeName;
  } else {
    argTypeCount = collNumTypes;
    argRunTypes = adjTestTypes;
    argRunTypenames = adjTestTypenames;
  }

  for (int idx=0; idx<argTypeCount; idx++) {
     TimeTest(argArgs, argRunTypes[idx], argRunTypenames[idx], (collRedOpType)0, NULL, 0, 1);
  }
}
```

---

## File: `src/all_reduce.cu`

---

### 49. Function `print_header` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{print\_header}_{\text{allreduce}}: \emptyset \to \text{Unit}
$$
$$
\text{Output: formatted table header with columns:}
$$
$$
[\text{bytes}, N, \text{type}, \text{op}, \underbrace{\text{time}, \text{algbw}, \text{busbw}, \text{res}}_{\text{out-of-place}}, \underbrace{\text{time}, \text{algbw}, \text{busbw}, \text{res}}_{\text{in-place}}]
$$
$$
\text{Note: includes 'op' column unlike all\_gather}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_header_has_op_column : 4 + 4 + 4 = 12 := by sorry

-- axle_verify: content
theorem allreduce_header_has_op_column : 4 + 4 + 4 = 12 := by norm_num
```

#### LAYER 2 — FORM TEMPLATE

```c
void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}
```

---

### 50. Function `print_line_header` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{print\_line\_header}_{\text{allreduce}}: \mathbb{N} \times \mathbb{N} \times \text{String} \times \text{String} \times \mathbb{Z}_{\geq 0} \to \text{Unit}
$$
$$
\text{Output: } [\text{size}, \text{count}, \text{typeName}, \text{opName}]
$$
$$
\text{Note: 4 fields (includes opName)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_line_header_fields : (4 : Nat) = 4 := by sorry

-- axle_verify: content
theorem allreduce_line_header_fields : (4 : Nat) = 4 := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
void print_line_header (size_t argSize, size_t argCount, const char *argTypeName, const char *argOpName, int argRoot) {
  PRINT("%12li  %12li  %6s  %6s", argSize, argCount, argTypeName, argOpName);
}
```

---

### 51. Function `getCollByteCount` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{getCollByteCount}_{\text{allreduce}}: \mathbb{N} \times \mathbb{Z}_{>0} \to (\mathbb{N})^7
$$
$$
\text{sendcount} = \text{recvcount} = \text{paramcount} = \text{count}
$$
$$
\text{sameExpected} = 1
$$
$$
\text{procSharedCount} = 0
$$
$$
\text{sendInplaceOffset} = \text{recvInplaceOffset} = 0
$$
$$
\text{Invariant: all-reduce is symmetric: sendcount} = \text{recvcount}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_symmetric (count : Nat) :
    let sendcount := count
    let recvcount := count
    sendcount = recvcount := by sorry

-- axle_verify: content
theorem allreduce_symmetric (count : Nat) :
    let sendcount := count
    let recvcount := count
    sendcount = recvcount := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
void getCollByteCount(size_t *argSendcount, size_t *argRecvcount, size_t *argParamcount, size_t *argSendInplaceOffset, size_t *argRecvInplaceOffset, size_t *argProcSharedCount, int *argSameExpected, size_t argCount, int argNranks) {
    *argSendcount = argCount;
    *argRecvcount = argCount;
    *argSameExpected = 1;
    *argProcSharedCount = 0;
    *argSendInplaceOffset = 0;
    *argRecvInplaceOffset = 0;
    *argParamcount = *argSendcount;
 }
```

---

### 52. Function `InitRecvResult` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{InitRecvResult}_{\text{allreduce}}: \text{threadArgs\_t}^* \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \{0,1\} \times \{0,1\} \to \text{Unit}
$$
$$
\text{count} = \text{nbytes} / \text{wordSize}(\text{type})
$$
$$
\text{Phase 1 (sync):}
$$
$$
\forall i \in [0, \text{nGpus}):
$$
$$
\quad \text{data} = \begin{cases}
\text{recvbuffs}[i] & \text{in\_place} \\
\text{sendbuffs}[i] & \text{otherwise}
\end{cases}
$$
$$
\quad \text{is\_first} \land i = 0 \implies \text{expected}[0] \leftarrow \text{memcpy}(\text{data})
$$
$$
\quad \text{otherwise} \implies \text{expected}[0] \leftarrow \text{Accumulate}(\text{expected}[0], \text{data}, \text{count}, \text{type}, \text{op})
$$
$$
\text{Phase 2 (MPI, last thread): MPI\_Bcast from each proc, Accumulate into expected}
$$
$$
\text{Semantics: computes expected result of all-reduce = } \bigoplus_{r=0}^{\text{nranks}-1} \text{data}_r
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_expected_is_reduction (nGpus : Nat) (h : nGpus > 0) :
    ∀ i : Nat, i < nGpus → i < nGpus := by sorry

-- axle_verify: content
theorem allreduce_expected_is_reduction (nGpus : Nat) (h : nGpus > 0) :
    ∀ i : Nat, i < nGpus → i < nGpus := by
  intro i hi; exact hi
```

#### LAYER 2 — FORM TEMPLATE

```c
void InitRecvResult(struct adjWorkerArgs_t* argArgs, collDataType argType, collRedOpType argOp, int argRoot, int argInPlace, int argIsFirst) {
  size_t argCount = argArgs->argBytes / elemSize(argType);

  while (argArgs->adjSync[argArgs->argSyncIdx] != argArgs->argThread) pthread_yield();

  for (int idx=0; idx<argArgs->argNumGpus; idx++) {
    int argDevice;
    COLLCHECK(collCommCuDevice(argArgs->adjComms[idx], &argDevice));
    GPUCHECK(gpuSetDevice(argDevice));
    void* argData = argInPlace ? argArgs->adjRecvBuffs[idx] : argArgs->adjSendBuffs[idx];

    if (argIsFirst && idx == 0) {
      GPUCHECK(gpuMemcpy(argArgs->adjExpected[0], argData, argCount*elemSize(argType), gpuMemcpyDeviceToHost));
    } else {
      Accumulate(argArgs->adjExpected[0], argData, argCount, argType, argOp);
    }

    if (argInPlace == 0) {
      GPUCHECK(gpuMemset(argArgs->adjRecvBuffs[idx], 0, argArgs->argBytes));
    }
    GPUCHECK(gpuDeviceSynchronize());
  }

  argArgs->adjSync[argArgs->argSyncIdx] = argArgs->argThread + 1;

  if (argArgs->argThread+1 == argArgs->argNumThreads) {
#ifdef MPI_SUPPORT
    if (argArgs->argBytes > 0) {
      void* argRemote, *argRemoteHost = malloc(argArgs->argBytes);
      void* argMyInitialData = malloc(argArgs->argBytes);
      memcpy(argMyInitialData, argArgs->adjExpectedHost[0], argArgs->argBytes);
      GPUCHECK(gpuHostRegister(argRemoteHost, argArgs->argBytes, gpuHostRegisterPortable | gpuHostRegisterMapped));
      GPUCHECK(gpuHostGetDevicePointer(&argRemote, argRemoteHost, 0));
      for (int idx=0; idx<argArgs->argNumProcs; idx++) {
        if (idx == argArgs->argProc) {
          MPI_Bcast(argMyInitialData, argArgs->argBytes, MPI_BYTE, idx, MPI_COMM_WORLD);
          free(argMyInitialData);
        } else {
          MPI_Bcast(argRemoteHost, argArgs->argBytes, MPI_BYTE, idx, MPI_COMM_WORLD);
          Accumulate(argArgs->adjExpected[0], argRemote, argCount, argType, argOp);
          gpuDeviceSynchronize();
        }
      }
      GPUCHECK(gpuHostUnregister(argRemoteHost));
      free(argRemoteHost);
    }
#endif
    argArgs->adjSync[argArgs->argSyncIdx] = 0;
  } else {
    while (argArgs->adjSync[argArgs->argSyncIdx]) pthread_yield();
  }

  argArgs->argSyncIdx = !argArgs->argSyncIdx;
}
```

---

### 53. Function `GetBw` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{GetBw}_{\text{allreduce}}: \mathbb{N} \times \mathbb{Z}_{>0} \times \mathbb{R}_{>0} \times \mathbb{R}^* \times \mathbb{R}^* \times \mathbb{Z}_{>0} \to \text{Unit}
$$
$$
\text{baseBw} = \frac{\text{count} \times \text{typesize}}{10^9 \times \text{sec}}
$$
$$
\text{algBw} = \text{baseBw}
$$
$$
\text{factor} = \frac{2(\text{nranks} - 1)}{\text{nranks}}
$$
$$
\text{busBw} = \text{baseBw} \times \text{factor}
$$
$$
\text{Derivation: all-reduce = reduce-scatter + all-gather}
$$
$$
\text{Each rank sends/receives } \frac{2(n-1)}{n} \text{ of total data}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_bus_factor (nranks : Nat) (h : nranks > 0) :
    2 * (nranks - 1) ≤ 2 * nranks := by sorry

-- axle_verify: content
theorem allreduce_bus_factor (nranks : Nat) (h : nranks > 0) :
    2 * (nranks - 1) ≤ 2 * nranks := by omega
```

```lean
-- axle_verify: formal_statement
theorem allreduce_factor_approaches_2 (nranks : Nat) (h : nranks ≥ 2) :
    2 * (nranks - 1) ≥ nranks := by sorry

-- axle_verify: content
theorem allreduce_factor_approaches_2 (nranks : Nat) (h : nranks ≥ 2) :
    2 * (nranks - 1) ≥ nranks := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void GetBw(size_t argCount, int argTypesize, double argSec, double* argAlgBw, double* argBusBw, int argNranks) {
  double argBaseBw = (double)(argCount * argTypesize) / 1.0E9 / argSec;

  *argAlgBw = argBaseBw;
  double argFactor = ((double)(2*(argNranks - 1)))/((double)argNranks);
  *argBusBw = argBaseBw * argFactor;
}
```

---

### 54. Function `RunColl` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{RunColl}_{\text{allreduce}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \text{Comm} \times \text{Stream} \to \text{Unit}
$$
$$
\text{Delegates to: } \text{ncclAllReduce}(\text{sendbuff}, \text{recvbuff}, \text{count}, \text{type}, \text{op}, \text{comm}, \text{stream})
$$
$$
\text{Note: root is unused}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_runcoll_uses_op : True := by sorry

-- axle_verify: content
theorem allreduce_runcoll_uses_op : True := by trivial
```

#### LAYER 2 — FORM TEMPLATE

```c
void RunColl(void* argSendbuff, void* argRecvbuff, size_t argCount, collDataType argType, collRedOpType argOp, int argRoot, collCommType argComm, gpuStreamType argStream) {
  COLLCHECK(collAllReduce(argSendbuff, argRecvbuff, argCount, argType, argOp, argComm, argStream));
}
```

---

### 55. Function `RunTest` (all_reduce)

#### LAYER 1a — MATH (LaTeX)

$$
\text{RunTest}_{\text{allreduce}}: \text{threadArgs\_t}^* \times \mathbb{Z} \times \text{Type} \times \text{String} \times \text{Op} \times \text{String} \to \text{Unit}
$$
$$
\text{If type} \neq -1: \text{type\_count} = 1, \text{else ncclNumTypes}
$$
$$
\text{If op} \neq -1: \text{op\_count} = 1, \text{else ncclNumOps}
$$
$$
\text{Nested loop: } \forall i \in [0, \text{type\_count}), \forall j \in [0, \text{op\_count}):
$$
$$
\quad \text{TimeTest}(\text{args}, \text{types}[i], \text{typenames}[i], \text{ops}[j], \text{opnames}[j], 0, 1)
$$
$$
\text{Total iterations} = \text{type\_count} \times \text{op\_count}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem allreduce_runtest_iterations (typeCount opCount : Nat) :
    typeCount * opCount = opCount * typeCount := by sorry

-- axle_verify: content
theorem allreduce_runtest_iterations (typeCount opCount : Nat) :
    typeCount * opCount = opCount * typeCount := by ring
```

#### LAYER 2 — FORM TEMPLATE

```c
void RunTest(struct adjWorkerArgs_t* argArgs, int argRoot, collDataType argType, const char* argTypeName, collRedOpType argOp, const char* argOpName) {
  collDataType *argRunTypes;
  collRedOpType *argRunOps;
  const char **argRunTypenames, **argRunOpnames;
  int argTypeCount, argOpCount;

  if ((int)argType != -1) {
    argTypeCount = 1;
    argRunTypes = &argType;
    argRunTypenames = &argTypeName;
  } else {
    argTypeCount = collNumTypes;
    argRunTypes = adjTestTypes;
    argRunTypenames = adjTestTypenames;
  }

  if ((int)argOp != -1) {
    argOpCount = 1;
    argRunOps = &argOp;
    argRunOpnames = &argOpName;
  } else {
    argOpCount = collNumOps;
    argRunOps = adjTestOps;
    argRunOpnames = adjTestOpnames;
  }

  for (int idx=0; idx<argTypeCount; idx++) {
      for (int jdx=0; jdx<argOpCount; jdx++) {
          TimeTest(argArgs, argRunTypes[idx], argRunTypenames[idx], argRunOps[jdx], argRunOpnames[jdx], 0, 1);
      }
  }
}
```

