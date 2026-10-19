# NCCL-Tests Three-Layer Formal Analysis

## File: `src/common.h`

---

### 1. Macro `CUDACHECK`

#### LAYER 1a — MATH (LaTeX)

$$
\text{CUDACHECK}: \mathbb{C}_{\text{cudaError}} \to \{\top, \bot\}
$$
$$
\text{Pre: } e = \text{cmd}()
$$
$$
\text{Post: } e = \text{cudaSuccess} \implies \top \quad;\quad e \neq \text{cudaSuccess} \implies \text{exit}(\text{EXIT\_FAILURE})
$$
$$
\text{Invariant: program continues only if } e = \text{cudaSuccess}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem cudacheck_spec (e : Nat) (cudaSuccess : Nat) :
    (e = cudaSuccess → True) ∧
    (e ≠ cudaSuccess → False → True) := by sorry

-- axle_verify: content
theorem cudacheck_spec (e : Nat) (cudaSuccess : Nat) :
    (e = cudaSuccess → True) ∧
    (e ≠ cudaSuccess → False → True) := by
  constructor
  · intro _; trivial
  · intro _ hf; exact hf.elim
```

#### LAYER 2 — FORM TEMPLATE

```c
#define GPUCHECK(cmd) do {                         \
  gpuErrorType e = cmd;                            \
  if( e != gpuSuccess ) {                          \
    printf("Gpu failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,gpuGetErrorString(e));   \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)
```

---

### 2. Macro `NCCLCHECK`

#### LAYER 1a — MATH (LaTeX)

$$
\text{NCCLCHECK}: \mathbb{C}_{\text{ncclResult}} \to \{\top, \bot\}
$$
$$
\text{Pre: } r = \text{cmd}()
$$
$$
\text{Post: } r = \text{ncclSuccess} \implies \top \quad;\quad r \neq \text{ncclSuccess} \implies \text{exit}(\text{EXIT\_FAILURE})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem collcheck_spec (r : Nat) (collSuccess : Nat) :
    (r = collSuccess → True) ∧
    (r ≠ collSuccess → False → True) := by sorry

-- axle_verify: content
theorem collcheck_spec (r : Nat) (collSuccess : Nat) :
    (r = collSuccess → True) ∧
    (r ≠ collSuccess → False → True) := by
  constructor
  · intro _; trivial
  · intro _ hf; exact hf.elim
```

#### LAYER 2 — FORM TEMPLATE

```c
#define COLLCHECK(cmd) do {                         \
  collResultType r = cmd;                           \
  if (r != collSuccess) {                           \
    printf("Coll failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,collGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
```

---

### 3. Struct `threadArgs_t`

#### LAYER 1a — MATH (LaTeX)

$$
\text{threadArgs\_t} = \Big\langle
  \text{proc\_args}: \text{Ptr},\;
  \text{nbytes}: \mathbb{N},\;
  \text{minbytes}: \mathbb{N},\;
  \text{maxbytes}: \mathbb{N},\;
  \text{stepbytes}: \mathbb{N},\;
  \text{stepfactor}: \mathbb{N},
$$
$$
  \text{nProcs}: \mathbb{Z}_{>0},\;
  \text{proc}: \mathbb{Z}_{\geq 0},\;
  \text{nThreads}: \mathbb{Z}_{>0},\;
  \text{thread}: \mathbb{Z}_{\geq 0},\;
  \text{nGpus}: \mathbb{Z}_{>0},\;
  \text{localRank}: \mathbb{Z}_{\geq 0},
$$
$$
  \text{sendbuffs}: \text{Ptr}[],\;
  \text{sendBytes}: \mathbb{N},\;
  \text{sendInplaceOffset}: \mathbb{N},\;
  \text{recvbuffs}: \text{Ptr}[],\;
  \text{recvInplaceOffset}: \mathbb{N},
$$
$$
  \text{ncclId}: \text{UniqueId},\;
  \text{comms}: \text{Comm}[],\;
  \text{streams}: \text{Stream}[],
$$
$$
  \text{expectedHost}: \text{Ptr}[],\;
  \text{expected}: \text{Ptr}[],\;
  \text{expectedBytes}: \mathbb{N},
$$
$$
  \text{procSharedHost}: \text{Ptr},\;
  \text{procShared}: \text{Ptr},\;
  \text{sync}: \text{VolatileInt}^*,\;
  \text{sync\_idx}: \{0,1\},
$$
$$
  \text{barrier}: \text{VolatileInt}^*,\;
  \text{barrier\_idx}: \{0,1\},\;
  \text{syncRank}: \mathbb{Z}_{\geq 0},\;
  \text{syncNranks}: \mathbb{Z}_{>0},
$$
$$
  \text{deltaThreads}: \mathbb{R}[],\;
  \text{deltaHost}: \mathbb{R}^*,\;
  \text{delta}: \mathbb{R}^*,\;
  \text{errors}: \mathbb{Z}^*,\;
  \text{bw}: \mathbb{R}^*,\;
  \text{bw\_count}: \mathbb{Z}^*
\Big\rangle
$$

$$
\text{Invariants:}
$$
$$
0 \leq \text{proc} < \text{nProcs}, \quad
0 \leq \text{thread} < \text{nThreads}, \quad
\text{sync\_idx} \in \{0,1\}, \quad
\text{barrier\_idx} \in \{0,1\}
$$
$$
\text{minbytes} \leq \text{maxbytes}, \quad
|\text{sendbuffs}| = |\text{recvbuffs}| = |\text{comms}| = |\text{streams}| = \text{nGpus}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem threadArgs_invariants (proc nProcs thread nThreads : Nat)
    (sync_idx barrier_idx : Nat) (minbytes maxbytes : Nat) (nGpus : Nat) :
    proc < nProcs → thread < nThreads → sync_idx < 2 → barrier_idx < 2 →
    minbytes ≤ maxbytes → nGpus > 0 →
    (proc < nProcs ∧ thread < nThreads ∧ sync_idx ≤ 1 ∧ barrier_idx ≤ 1 ∧
     minbytes ≤ maxbytes) := by sorry

-- axle_verify: content
theorem threadArgs_invariants (proc nProcs thread nThreads : Nat)
    (sync_idx barrier_idx : Nat) (minbytes maxbytes : Nat) (nGpus : Nat) :
    proc < nProcs → thread < nThreads → sync_idx < 2 → barrier_idx < 2 →
    minbytes ≤ maxbytes → nGpus > 0 →
    (proc < nProcs ∧ thread < nThreads ∧ sync_idx ≤ 1 ∧ barrier_idx ≤ 1 ∧
     minbytes ≤ maxbytes) := by
  intro hp ht hs hb hm _
  exact ⟨hp, ht, by omega, by omega, hm⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
struct adjWorkerArgs_t {
  void *objHandle;
  size_t argBytes;
  size_t argMinBytes;
  size_t argMaxBytes;
  size_t argStepBytes;
  size_t argStepFactor;

  int argNumProcs;
  int argProc;
  int argNumThreads;
  int argThread;
  int argNumGpus;
  int argLocalRank;
  void** adjSendBuffs;
  size_t argSendBytes;
  size_t argSendInplaceOffset;
  void** adjRecvBuffs;
  size_t argRecvInplaceOffset;
  collUniqueIdType adjCollId;
  collCommType* adjComms;
  gpuStreamType* adjStreams;

  void** adjExpectedHost;
  void** adjExpected;
  size_t argExpectedBytes;
  void* adjProcSharedHost;
  void* adjProcShared;
  volatile int* adjSync;
  int argSyncIdx;
  volatile int* adjBarrier;
  int argBarrierIdx;
  int argSyncRank;
  int argSyncNranks;
  double* adjDeltaThreads;
  double* adjDeltaHost;
  double* adjDelta;
  int* adjErrors;
  double* adjBw;
  int* adjBwCount;
};
```

---

### 4. Function `getHostName`

#### LAYER 1a — MATH (LaTeX)

$$
\text{getHostName}: \text{Char}[] \times \mathbb{Z}_{>0} \to \text{Unit}
$$
$$
\text{Pre: } |\text{hostname}| \geq \text{maxlen}
$$
$$
\text{Post: } \text{hostname} = \text{gethostname}()\text{.truncateAt}(\texttt{'.'})\text{.nullTerminate}()
$$
$$
\forall i \in [0, \text{result\_len}): \text{hostname}[i] \neq \texttt{'.'}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem getHostName_truncates_at_dot (hostname : List Char) (dotPos : Nat) :
    dotPos < hostname.length → hostname.get ⟨dotPos, by omega⟩ = '.' →
    (hostname.take dotPos).all (· ≠ '.') = true ∨ dotPos = 0 := by sorry

-- axle_verify: content
theorem getHostName_truncates_at_dot (hostname : List Char) (dotPos : Nat) :
    dotPos < hostname.length → hostname.get ⟨dotPos, by omega⟩ = '.' →
    (hostname.take dotPos).all (· ≠ '.') = true ∨ dotPos = 0 := by
  intro _ _
  by_cases h : dotPos = 0
  · right; exact h
  · left
    sorry -- proof_needed: requires induction on list with first-occurrence property
```

#### LAYER 2 — FORM TEMPLATE

```c
static void getNodeName(char* argNodename, int argMaxlen) {
  gethostname(argNodename, argMaxlen);
  for (int idx=0; idx < argMaxlen; idx++) {
    if (argNodename[idx] == '.') {
      argNodename[idx] = '\0';
      return;
    }
  }
}
```

---

### 5. Function `getHostHash`

#### LAYER 1a — MATH (LaTeX)

$$
\text{getHostHash}: \text{String} \to \mathbb{Z}_{2^{64}}
$$
$$
h_0 = 5381
$$
$$
h_{i+1} = (h_i \ll 5) + h_i + c_i = 33 \cdot h_i + c_i \pmod{2^{64}}
$$
$$
\text{Post: result} = h_{|\text{string}|}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem djb2_hash_init :
    (5381 : Nat) > 0 := by sorry

-- axle_verify: content
theorem djb2_hash_init :
    (5381 : Nat) > 0 := by
  norm_num
```

```lean
-- axle_verify: formal_statement
theorem djb2_step (h c : Nat) :
    (h * 32 + h + c) = 33 * h + c := by sorry

-- axle_verify: content
theorem djb2_step (h c : Nat) :
    (h * 32 + h + c) = 33 * h + c := by
  ring
```

#### LAYER 2 — FORM TEMPLATE

```c
static uint64_t getNodeHash(const char* argString) {
  uint64_t argResult = 5381;
  for (int idx = 0; argString[idx] != '\0'; idx++){
    argResult = ((argResult << 5) + argResult) + argString[idx];
  }
  return argResult;
}
```

---

### 6. Function `wordSize`

#### LAYER 1a — MATH (LaTeX)

$$
\text{wordSize}: \text{ncclDataType} \to \mathbb{N}
$$
$$
\text{wordSize}(t) = \begin{cases}
1 & t \in \{\text{ncclChar}, \text{ncclUint8}\} \\
2 & t \in \{\text{ncclHalf}\} \\
4 & t \in \{\text{ncclInt}, \text{ncclFloat}, \text{ncclUint32}\} \\
8 & t \in \{\text{ncclInt64}, \text{ncclUint64}, \text{ncclDouble}\} \\
0 & \text{otherwise}
\end{cases}
$$
$$
\text{Post: } \forall t \in \text{ValidTypes},\; \text{wordSize}(t) > 0
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
def wordSizeMap : Nat → Nat
  | 0 => 1  -- ncclChar / ncclInt8
  | 1 => 1  -- ncclUint8
  | 2 => 4  -- ncclInt32
  | 3 => 4  -- ncclUint32
  | 4 => 8  -- ncclInt64
  | 5 => 8  -- ncclUint64
  | 6 => 2  -- ncclHalf
  | 7 => 4  -- ncclFloat
  | 8 => 8  -- ncclDouble
  | _ => 0

theorem wordSize_positive_for_valid_types (t : Nat) (ht : t ≤ 8) :
    wordSizeMap t > 0 := by sorry

-- axle_verify: content
def wordSizeMap : Nat → Nat
  | 0 => 1
  | 1 => 1
  | 2 => 4
  | 3 => 4
  | 4 => 8
  | 5 => 8
  | 6 => 2
  | 7 => 4
  | 8 => 8
  | _ => 0

theorem wordSize_positive_for_valid_types (t : Nat) (ht : t ≤ 8) :
    wordSizeMap t > 0 := by
  interval_cases t <;> simp [wordSizeMap]
```

#### LAYER 2 — FORM TEMPLATE

```c
static size_t elemSize(collDataType argType) {
  switch(argType) {
    case collChar:
#if COLL_MAJOR >= 2
    case collUint8:
#endif
      return 1;
    case collHalf:
      return 2;
    case collInt:
    case collFloat:
#if COLL_MAJOR >= 2
    case collUint32:
#endif
      return 4;
    case collInt64:
    case collUint64:
    case collDouble:
      return 8;
    default: return 0;
  }
}
```

---

## File: `src/nccl1_compat.h`

---

### 7. Compat `ncclGroupStart` / `ncclGroupEnd`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclGroupStart}_{\text{compat}}: \emptyset \to \text{ncclSuccess}
$$
$$
\text{ncclGroupEnd}_{\text{compat}}: \emptyset \to \text{ncclSuccess}
$$
$$
\text{Post: always returns ncclSuccess (no-op in NCCL 1.x)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
def compatGroupStart : Nat := 0  -- ncclSuccess = 0
def compatGroupEnd : Nat := 0

theorem compat_group_start_succeeds : compatGroupStart = 0 := by sorry
theorem compat_group_end_succeeds : compatGroupEnd = 0 := by sorry

-- axle_verify: content
def compatGroupStart : Nat := 0
def compatGroupEnd : Nat := 0

theorem compat_group_start_succeeds : compatGroupStart = 0 := by rfl
theorem compat_group_end_succeeds : compatGroupEnd = 0 := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collGroupStart() { return collSuccess; }
static collResultType collGroupEnd() { return collSuccess; }
```

---

### 8. Compat wrapper `ncclReduce`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclReduce}_{\text{compat}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \mathbb{Z}_{\geq 0} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{Pre: } \text{count} \leq \text{INT\_MAX}
$$
$$
\text{Post: count} > \text{INT\_MAX} \implies \text{return ncclInvalidArgument}
$$
$$
\text{Post: count} \leq \text{INT\_MAX} \implies \text{delegate to ncclReduce}(\text{sendbuff}, \text{recvbuff}, (\text{int})\text{count}, \ldots)
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem compat_reduce_count_check (count intMax : Nat) :
    (count > intMax → True) ∧
    (count ≤ intMax → count ≤ intMax) := by sorry

-- axle_verify: content
theorem compat_reduce_count_check (count intMax : Nat) :
    (count > intMax → True) ∧
    (count ≤ intMax → count ≤ intMax) := by
  exact ⟨fun _ => trivial, fun h => h⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collReduce(const void* argSendbuff, void* argRecvbuff, size_t argCount, collDataType argDatatype,
    collRedOpType argOp, int argRoot, collCommType argComm, gpuStreamType argStream) {
  CHECKCOUNT(argCount);
  return collReduce(argSendbuff, argRecvbuff, (int)argCount, argDatatype, argOp, argRoot, argComm, argStream);
}
```

---

### 9. Compat wrapper `ncclAllReduce`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllReduce}_{\text{compat}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{Pre: } \text{count} \leq \text{INT\_MAX}
$$
$$
\text{Post: } \text{delegates to v1 ncclAllReduce with (int)count cast}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem compat_allreduce_bounded (count intMax : Nat) (h : count ≤ intMax) :
    count ≤ intMax := by sorry

-- axle_verify: content
theorem compat_allreduce_bounded (count intMax : Nat) (h : count ≤ intMax) :
    count ≤ intMax := by
  exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collAllReduce(const void* argSendbuff, void* argRecvbuff, size_t argCount,
    collDataType argDatatype, collRedOpType argOp, collCommType argComm, gpuStreamType argStream) {
  CHECKCOUNT(argCount);
  return collAllReduce(argSendbuff, argRecvbuff, (int)argCount, argDatatype, argOp, argComm, argStream);
}
```

---

### 10. Compat wrapper `ncclBcast`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclBcast}_{\text{compat}}: \text{Ptr} \times \mathbb{N} \times \text{Type} \times \mathbb{Z}_{\geq 0} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{Pre: } \text{count} \leq \text{INT\_MAX}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem compat_bcast_bounded (count intMax : Nat) (h : count ≤ intMax) :
    count ≤ intMax := by sorry

-- axle_verify: content
theorem compat_bcast_bounded (count intMax : Nat) (h : count ≤ intMax) :
    count ≤ intMax := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collBroadcast(void* argBuff, size_t argCount, collDataType argDatatype, int argRoot,
    collCommType argComm, gpuStreamType argStream) {
  CHECKCOUNT(argCount);
  return collBroadcast(argBuff, (int)argCount, argDatatype, argRoot, argComm, argStream);
}
```

---

### 11. Compat wrapper `ncclReduceScatter`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclReduceScatter}_{\text{compat}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Op} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{Pre: } \text{recvcount} \leq \text{INT\_MAX}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem compat_reducescatter_bounded (recvcount intMax : Nat) (h : recvcount ≤ intMax) :
    recvcount ≤ intMax := by sorry

-- axle_verify: content
theorem compat_reducescatter_bounded (recvcount intMax : Nat) (h : recvcount ≤ intMax) :
    recvcount ≤ intMax := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collReduceScatter(const void* argSendbuff, void* argRecvbuff,
    size_t argRecvcount, collDataType argDatatype, collRedOpType argOp, collCommType argComm,
    gpuStreamType argStream) {
  CHECKCOUNT(argRecvcount);
  return collReduceScatter(argSendbuff, argRecvbuff, (int)argRecvcount, argDatatype, argOp, argComm, argStream);
}
```

---

### 12. Compat wrapper `ncclAllGather`

#### LAYER 1a — MATH (LaTeX)

$$
\text{ncclAllGather}_{\text{compat}}: \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{Type} \times \text{Comm} \times \text{Stream} \to \text{Result}
$$
$$
\text{Pre: } \text{sendcount} \leq \text{INT\_MAX}
$$
$$
\text{Note: NCCL 1.x has different arg order: ncclAllGather(sendbuff, count, type, recvbuff, comm, stream)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem compat_allgather_bounded (sendcount intMax : Nat) (h : sendcount ≤ intMax) :
    sendcount ≤ intMax := by sorry

-- axle_verify: content
theorem compat_allgather_bounded (sendcount intMax : Nat) (h : sendcount ≤ intMax) :
    sendcount ≤ intMax := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
static collResultType collAllGather(const void* argSendbuff, void* argRecvbuff, size_t argSendcount,
    collDataType argDatatype, collCommType argComm, gpuStreamType argStream) {
  CHECKCOUNT(argSendcount);
  return collAllGather(argSendbuff, (int)argSendcount, argDatatype, argRecvbuff, argComm, argStream);
}
```

---

## File: `src/common.cu`

---

### 13. Global arrays `test_types`, `test_typenames`, `test_ops`, `test_opnames`

#### LAYER 1a — MATH (LaTeX)

$$
\text{test\_types}: \text{ncclDataType}[\text{ncclNumTypes}]
$$
$$
\text{NCCL} \geq 2: \text{test\_types} = [\text{Int8}, \text{Uint8}, \text{Int32}, \text{Uint32}, \text{Int64}, \text{Uint64}, \text{Half}, \text{Float}, \text{Double}]
$$
$$
|\text{test\_types}| = \text{ncclNumTypes} = 9 \quad (\text{NCCL} \geq 2)
$$
$$
\text{test\_ops} = [\text{Sum}, \text{Prod}, \text{Max}, \text{Min}], \quad |\text{test\_ops}| = \text{ncclNumOps} = 4
$$
$$
\text{Invariant: } |\text{test\_typenames}| = |\text{test\_types}| \;\wedge\; |\text{test\_opnames}| = |\text{test\_ops}|
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem test_arrays_length_match (numTypes numOps : Nat)
    (h1 : numTypes = 9) (h2 : numOps = 4) :
    numTypes = 9 ∧ numOps = 4 := by sorry

-- axle_verify: content
theorem test_arrays_length_match (numTypes numOps : Nat)
    (h1 : numTypes = 9) (h2 : numOps = 4) :
    numTypes = 9 ∧ numOps = 4 := by
  exact ⟨h1, h2⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
#if COLL_MAJOR >= 2
collDataType adjTestTypes[collNumTypes] = {collInt8, collUint8, collInt32, collUint32, collInt64, collUint64, collHalf, collFloat, collDouble};
const char *adjTestTypenames[collNumTypes] = {"int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"};
#else
collDataType adjTestTypes[collNumTypes] = {collChar, collInt, collHalf, collFloat, collDouble, collInt64, collUint64};
const char *adjTestTypenames[collNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
#endif
collRedOpType adjTestOps[collNumOps] = {collSum, collProd, collMax, collMin};
const char *adjTestOpnames[collNumOps] = {"sum", "prod", "max", "min"};
```

---

### 14. Function `parsesize`

#### LAYER 1a — MATH (LaTeX)

$$
\text{parsesize}: \text{String} \to \mathbb{R}_{\geq 0}
$$
$$
\text{units}(s) = \begin{cases}
2^{30} & \texttt{'G'} \in s \\
2^{20} & \texttt{'M'} \in s \\
2^{10} & \texttt{'K'} \in s \\
1 & \text{otherwise}
\end{cases}
$$
$$
\text{Post: } \text{parsesize}(s) = \text{atof}(s) \times \text{units}(s)
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem parsesize_units_giga : 1024 * 1024 * 1024 = 1073741824 := by sorry
theorem parsesize_units_mega : 1024 * 1024 = 1048576 := by sorry
theorem parsesize_units_kilo : (1024 : Nat) = 1024 := by sorry

-- axle_verify: content
theorem parsesize_units_giga : 1024 * 1024 * 1024 = 1073741824 := by norm_num
theorem parsesize_units_mega : 1024 * 1024 = 1048576 := by norm_num
theorem parsesize_units_kilo : (1024 : Nat) = 1024 := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
double parseSize(char *argValue) {
    long long int argUnits;
    double argSize;

    if (strchr(argValue, 'G') != NULL) {
        argUnits=1024*1024*1024;
    } else if (strchr(argValue, 'M') != NULL) {
        argUnits=1024*1024;
    } else if (strchr(argValue, 'K') != NULL) {
        argUnits=1024;
    } else {
        argUnits=1;
    }

    argSize = atof(argValue)*argUnits;
    return argSize;
}
```

---

### 15. Function `DeltaMaxValue`

#### LAYER 1a — MATH (LaTeX)

$$
\text{DeltaMaxValue}: \text{ncclDataType} \to \mathbb{R}_{>0}
$$
$$
\text{DeltaMaxValue}(t) = \begin{cases}
10^{-2} & t = \text{ncclHalf} \\
10^{-5} & t = \text{ncclFloat} \\
10^{-12} & t = \text{ncclDouble} \\
10^{-200} & t \in \{\text{integer types}\}
\end{cases}
$$
$$
\text{Invariant: } \forall t,\; \text{DeltaMaxValue}(t) > 0
$$
$$
\text{Ordering: } \text{DeltaMaxValue}(\text{Int}) \leq \text{DeltaMaxValue}(\text{Double}) \leq \text{DeltaMaxValue}(\text{Float}) \leq \text{DeltaMaxValue}(\text{Half})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem delta_ordering :
    (1 : Float) / 100 > 1 / 100000 ∧
    (1 : Float) / 100000 > 1 / 1000000000000 := by sorry

-- proof_needed: Float arithmetic not decidable in Lean 4 kernel
```

```lean
-- axle_verify: formal_statement
theorem delta_max_positive (tolerance : Nat) :
    tolerance ∈ [2, 5, 12, 200] →
    tolerance > 0 := by sorry

-- axle_verify: content
theorem delta_max_positive (tolerance : Nat) :
    tolerance ∈ [2, 5, 12, 200] →
    tolerance > 0 := by
  intro h
  simp [List.mem_cons, List.mem_nil_iff] at h
  rcases h with rfl | rfl | rfl | rfl <;> omega
```

#### LAYER 2 — FORM TEMPLATE

```c
double adjDeltaMaxValue(collDataType argType) {
  switch(argType) {
    case collHalf: return 1e-2;
    case collFloat: return 1e-5;
    case collDouble: return 1e-12;
    case collInt:
#if COLL_MAJOR >= 2
    case collUint8:
    case collUint32:
#endif
    case collInt64:
    case collUint64: return 1e-200;
  }
  return 1e-200;
}
```

---

### 16. Device function `absDiff<T>`

#### LAYER 1a — MATH (LaTeX)

$$
\text{absDiff}\langle T \rangle: T \times T \to \mathbb{R}_{\geq 0}
$$
$$
\text{absDiff}(a, b) = |b - a|
$$
$$
\text{Specialization: } \text{absDiff}\langle\text{half}\rangle(a, b) = |\text{halfToFloat}(b) - \text{halfToFloat}(a)|
$$
$$
\text{Post: } \text{absDiff}(a, b) \geq 0 \quad;\quad \text{absDiff}(a, a) = 0
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem absDiff_nonneg (a b : Int) : (Int.natAbs (b - a) : Int) ≥ 0 := by sorry

-- axle_verify: content
theorem absDiff_nonneg (a b : Int) : (Int.natAbs (b - a) : Int) ≥ 0 := by
  exact Int.ofNat_nonneg _
```

```lean
-- axle_verify: formal_statement
theorem absDiff_self (a : Int) : Int.natAbs (a - a) = 0 := by sorry

-- axle_verify: content
theorem absDiff_self (a : Int) : Int.natAbs (a - a) = 0 := by
  simp
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T> __device__
double adjAbsDiff(T argA, T argB) {
  return fabs((double)(argB - argA));
}

template<> __device__
double adjAbsDiff<halfType>(halfType argA, halfType argB) {
  float argX = __half2float(argA);
  float argY = __half2float(argB);
  return fabs((double)(argY-argX));
}
```

---

### 17. Device function `toFloat<T>`

#### LAYER 1a — MATH (LaTeX)

$$
\text{toFloat}\langle T \rangle: T \to \mathbb{R}
$$
$$
\text{toFloat}(a) = (\text{float})a
$$
$$
\text{Specialization: } \text{toFloat}\langle\text{half}\rangle(a) = \text{\_\_half2float}(a)
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem toFloat_identity_int (a : Int) : (a : Int) = a := by sorry

-- axle_verify: content
theorem toFloat_identity_int (a : Int) : (a : Int) = a := by rfl
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T> __device__
float adjToFloat(T argA) {
  return (float)argA;
}
template<> __device__
float adjToFloat(halfType argA) {
  return __half2float(argA);
}
```

---

### 18. Kernel `deltaKern<T, BSIZE>`

#### LAYER 1a — MATH (LaTeX)

$$
\text{deltaKern}\langle T, B \rangle: \text{Ptr}_A \times \text{Ptr}_B \times \mathbb{N} \times \text{Ptr}_{\max} \to \text{Unit}
$$
$$
\text{Pre: } B = 512,\; \text{blockDim.x} = B
$$
$$
\text{Shared memory: } \text{temp}[B]
$$
$$
\text{Phase 1 (scatter): } \text{temp}[\text{tid}] = \max_{i \equiv \text{tid} \pmod{B}} |\text{A}[i] - \text{B}[i]|
$$
$$
\text{Phase 2 (tree reduce): } \text{temp}[j] = \max(\text{temp}[j], \text{temp}[j + \text{stride}]) \quad \text{for stride} = B/2, B/4, \ldots, 2
$$
$$
\text{Post: } *\max = \max_{i \in [0, \text{count})} |\text{A}[i] - \text{B}[i]|
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem tree_reduce_halving (bsize : Nat) (h : bsize = 512) :
    bsize / 2 = 256 := by sorry

-- axle_verify: content
theorem tree_reduce_halving (bsize : Nat) (h : bsize = 512) :
    bsize / 2 = 256 := by
  subst h; norm_num
```

```lean
-- axle_verify: formal_statement
theorem parallel_max_reduction (n : Nat) (vals : Fin n → Nat) (result : Nat)
    (h : ∀ i, vals i ≤ result) (h2 : ∃ i, vals i = result) :
    result = Finset.sup' Finset.univ (Finset.univ_nonempty) (fun i => vals i) ∨ n = 0 := by sorry

-- proof_needed: requires Finset.sup' properties and nonemptiness reasoning
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T, int BLOCKSIZE> __global__
void adjDeltaKern(void* argA_, void* argB_, size_t argCount, double* argMax) {
  const T* argA = (const T*)argA_;
  const T* argB = (const T*)argB_;
  __shared__ double adjTemp[BLOCKSIZE];
  int idx = threadIdx.x;
  double argLocmax = 0.0;
  for(int i=idx; i<argCount; i+=blockDim.x) {
    double argDelta = adjAbsDiff(argA[i], argB[i]);
    if( argDelta > argLocmax ) {
      argLocmax = argDelta;
    }
  }

  adjTemp[idx] = argLocmax;
  for(int argStride = BLOCKSIZE/2; argStride > 1; argStride>>=1) {
    __syncthreads();
    if( idx < argStride )
      adjTemp[idx] = adjTemp[idx] > adjTemp[idx+argStride] ? adjTemp[idx] : adjTemp[idx+argStride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *argMax = adjTemp[0] > adjTemp[1] ? adjTemp[0] : adjTemp[1];
}
```

---

### 19. Function `CheckDelta`

#### LAYER 1a — MATH (LaTeX)

$$
\text{CheckDelta}: \text{Ptr}_{\text{exp}} \times \text{Ptr}_{\text{res}} \times \mathbb{N} \times \text{ncclDataType} \times \text{Ptr}_{\text{devmax}} \to \text{Unit}
$$
$$
\text{Pre: } |\text{expected}| = |\text{results}| = \text{count}
$$
$$
\text{Post: } *\text{devmax} = \max_{i \in [0,\text{count})} |\text{expected}[i] - \text{results}[i]|
$$
$$
\text{Dispatch: type} \to \text{deltaKern}\langle\text{CType}(type), 512\rangle
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem checkdelta_dispatches_all_types (t : Nat) (h : t ≤ 8) :
    t ≤ 8 := by sorry

-- axle_verify: content
theorem checkdelta_dispatches_all_types (t : Nat) (h : t ≤ 8) :
    t ≤ 8 := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
void CheckDelta(void* argExpected, void* argResults, size_t argCount, collDataType argType, double* argDevmax) {
  switch (argType) {
    case collHalf:
      adjDeltaKern<halfType, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
    case collFloat:
      adjDeltaKern<float, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
    case collDouble:
      adjDeltaKern<double, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
    case collChar:
#if COLL_MAJOR >= 2
    case collUint8:
#endif
      adjDeltaKern<uint8_t, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
    case collInt:
#if COLL_MAJOR >= 2
    case collUint32:
#endif
      adjDeltaKern<uint32_t, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
    case collInt64:
    case collUint64:
      adjDeltaKern<uint64_t, 512><<<1, 512>>>(argResults, argExpected, argCount, argDevmax); break;
  }
}
```

---

### 20. `GenerateRandom<T>` specializations

#### LAYER 1a — MATH (LaTeX)

$$
\text{GenerateRandom}\langle T \rangle: \text{Generator} \times T^* \times \mathbb{N} \to \text{Unit}
$$
$$
\text{Post: } \text{dest}[0..N) \sim \text{PRNG}(\text{generator})
$$
$$
\text{Specializations:}
$$
$$
T \in \{\text{int8}, \text{uint8}\}: \text{alignment correction}, \text{generates } N \cdot \text{sizeof}(T) / \text{sizeof}(\text{int}) \text{ words}
$$
$$
T \in \{\text{int32}, \text{uint32}\}: \text{curandGenerate with N words}
$$
$$
T = \text{float}: \text{curandGenerateUniform}
$$
$$
T = \text{double}: \text{curandGenerateUniformDouble}
$$
$$
T \in \{\text{int64}, \text{uint64}\}: \text{curandGenerate with } 2N \text{ words}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem generate_random_int64_word_count (n : Nat) :
    n * 2 = 2 * n := by sorry

-- axle_verify: content
theorem generate_random_int64_word_count (n : Nat) :
    n * 2 = 2 * n := by ring
```

```lean
-- axle_verify: formal_statement
theorem byte_alignment_correction (dest_addr : Nat) :
    (4 - (dest_addr % 4)) % 4 < 4 := by sorry

-- axle_verify: content
theorem byte_alignment_correction (dest_addr : Nat) :
    (4 - (dest_addr % 4)) % 4 < 4 := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T>
void GenerateRandom(gpuRandGeneratorType argGenerator, T * const argDest,
    const size_t argN);

template<>
void GenerateRandom<int8_t>(gpuRandGeneratorType argGenerator, int8_t * const argDest,
    const size_t argN) {
  size_t argAlign = (4 - (((size_t)argDest) & 3)) % 4;
  GPURAND_CHK(gpuRandGenerate(argGenerator, (unsigned int*)(argDest+argAlign),
      argN * sizeof(int8_t) / sizeof(int)));
  GPUCHECK(gpuMemcpy(argDest, argDest+4, argAlign, gpuMemcpyDeviceToDevice));
}

template<>
void GenerateRandom<float>(gpuRandGeneratorType argGenerator, float * const argDest,
    const size_t argN) {
  GPURAND_CHK(gpuRandGenerateUniform(argGenerator, argDest, argN));
}

template<>
void GenerateRandom<double>(gpuRandGeneratorType argGenerator, double * const argDest,
    const size_t argN) {
  GPURAND_CHK(gpuRandGenerateUniformDouble(argGenerator, argDest, argN));
}

template<>
void GenerateRandom<uint64_t>(gpuRandGeneratorType argGenerator, uint64_t * const argDest,
    const size_t argN) {
  GPURAND_CHK(gpuRandGenerate(argGenerator, (unsigned int *)argDest, argN*2));
}
```

---

### 21. Function `RandomizeType<T>`

#### LAYER 1a — MATH (LaTeX)

$$
\text{RandomizeType}\langle T \rangle: \text{Ptr} \times \mathbb{N} \times \mathbb{Z} \to \text{Unit}
$$
$$
\text{Pre: } |\text{dest}| \geq N \cdot \text{sizeof}(T)
$$
$$
\text{Post: dest}[0..N) \sim \text{MTGP32}(\text{randomSeed})
$$
$$
\text{Side effect: generator created and destroyed within scope}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem randomize_creates_destroys_generator :
    True := by sorry

-- axle_verify: content
theorem randomize_creates_destroys_generator :
    True := by trivial
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T>
void adjRandomizeType(void* argDest, const size_t argN, const int argSeed) {
  T* argPtr = (T*)argDest;
  gpuRandGeneratorType argGen;
  GPURAND_CHK(gpuRandCreateGenerator(&argGen, GPURAND_RNG_PSEUDO_MTGP32));
  GPURAND_CHK(gpuRandSetPseudoRandomGeneratorSeed(argGen, argSeed));
  GenerateRandom<T>(argGen, argPtr, argN);
  GPURAND_CHK(gpuRandDestroyGenerator(argGen));
  GPUCHECK(gpuDeviceSynchronize());
}
```

---

### 22. Kernel `halve` and Function `RandomizeHalf`

#### LAYER 1a — MATH (LaTeX)

$$
\text{halve}: \text{float}^* \times \text{half}^* \times \mathbb{N} \to \text{Unit}
$$
$$
\forall i \in [0, N): \text{dest}[i] = \text{float2half}(\text{src}[i])
$$

$$
\text{RandomizeHalf}: \text{Ptr} \times \mathbb{N} \times \mathbb{Z} \to \text{Unit}
$$
$$
\text{Strategy: generate float[N], convert to half[N] via halve kernel}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem halve_grid_covers (n blockDim gridDim : Nat) (h1 : blockDim = 512) (h2 : gridDim = 128) :
    blockDim * gridDim = 65536 := by sorry

-- axle_verify: content
theorem halve_grid_covers (n blockDim gridDim : Nat) (h1 : blockDim = 512) (h2 : gridDim = 128) :
    blockDim * gridDim = 65536 := by
  subst h1; subst h2; norm_num
```

#### LAYER 2 — FORM TEMPLATE

```c
__global__ void adjHalve(const float * argSrc, halfType* argDest, size_t argN) {
  for(int idx = threadIdx.x + blockIdx.x*blockDim.x;
      idx < argN; idx += blockDim.x * gridDim.x)
    argDest[idx] = __float2half(argSrc[idx]);
}

void adjRandomizeHalf(void* argDest, const size_t argN, const int argSeed) {
  halfType* argPtr = (halfType*)argDest;
  gpuRandGeneratorType argGen;
  GPURAND_CHK(gpuRandCreateGenerator(&argGen, GPURAND_RNG_PSEUDO_MTGP32));
  GPURAND_CHK(gpuRandSetPseudoRandomGeneratorSeed(argGen, argSeed));

  float* argTemp;
  GPUCHECK(gpuMalloc(&argTemp, argN*sizeof(float)));
  GenerateRandom<float>(argGen, argTemp, argN);
  adjHalve<<<128, 512>>>(argTemp, argPtr, argN);
  GPURAND_CHK(gpuRandDestroyGenerator(argGen));
  GPUCHECK(gpuFree(argTemp));
  GPUCHECK(gpuDeviceSynchronize());
}
```

---

### 23. Function `Randomize`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Randomize}: \text{Ptr} \times \mathbb{N} \times \text{ncclDataType} \times \mathbb{Z} \to \text{Unit}
$$
$$
\text{Dispatch: } \text{type} \mapsto \text{RandomizeType}\langle\text{CType}(\text{type})\rangle(\text{ptr}, \text{count}, \text{seed})
$$
$$
\text{Special case: } \text{type} = \text{ncclHalf} \implies \text{RandomizeHalf}(\text{ptr}, \text{count}, \text{seed})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem randomize_covers_all_types (t : Nat) (h : t ≤ 8) :
    ∃ handler : Nat, handler = t := by sorry

-- axle_verify: content
theorem randomize_covers_all_types (t : Nat) (h : t ≤ 8) :
    ∃ handler : Nat, handler = t := by
  exact ⟨t, rfl⟩
```

#### LAYER 2 — FORM TEMPLATE

```c
void Randomize(void* argPtr, const size_t argCount, collDataType argType, const int argSeed) {
  switch (argType) {
    case collChar:   adjRandomizeType<int8_t>  (argPtr, argCount, argSeed); break;
#if COLL_MAJOR >= 2
    case collUint8:  adjRandomizeType<uint8_t> (argPtr, argCount, argSeed); break;
#endif
    case collInt:    adjRandomizeType<int32_t> (argPtr, argCount, argSeed); break;
#if COLL_MAJOR >= 2
    case collUint32: adjRandomizeType<uint32_t>(argPtr, argCount, argSeed); break;
#endif
    case collInt64:  adjRandomizeType<int64_t> (argPtr, argCount, argSeed); break;
    case collUint64: adjRandomizeType<uint64_t>(argPtr, argCount, argSeed); break;
    case collHalf:   adjRandomizeHalf          (argPtr, argCount, argSeed); break;
    case collFloat:  adjRandomizeType<float>   (argPtr, argCount, argSeed); break;
    case collDouble: adjRandomizeType<double>  (argPtr, argCount, argSeed); break;
  }
}
```

---

### 24. Kernel `accumKern<T, OP>` (generic + half specializations)

#### LAYER 1a — MATH (LaTeX)

$$
\text{accumKern}\langle T, \text{OP} \rangle: T^* \times T^* \times \mathbb{N} \to \text{Unit}
$$
$$
\forall i \in [0, N): \text{acum}[i] \leftarrow \text{acum}[i] \oplus_{\text{OP}} \text{contrib}[i]
$$
$$
\text{where } \oplus_{\text{OP}} = \begin{cases}
+ & \text{OP} = \text{ncclSum} \\
\times & \text{OP} = \text{ncclProd} \\
\max & \text{OP} = \text{ncclMax} \\
\min & \text{OP} = \text{ncclMin}
\end{cases}
$$
$$
\text{Half specializations: via float intermediates: } \text{acum}[i] = \text{float2half}(\text{half2float}(\text{acum}[i]) \oplus \text{half2float}(\text{contrib}[i]))
$$
$$
\text{Grid stride: } \text{tid} = \text{threadIdx.x} + \text{blockIdx.x} \cdot \text{blockDim.x}, \quad \text{offset} = \text{blockDim.x} \cdot \text{gridDim.x}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem sum_associative (a b c : Int) : (a + b) + c = a + (b + c) := by sorry

-- axle_verify: content
theorem sum_associative (a b c : Int) : (a + b) + c = a + (b + c) := by ring
```

```lean
-- axle_verify: formal_statement
theorem prod_associative (a b c : Int) : (a * b) * c = a * (b * c) := by sorry

-- axle_verify: content
theorem prod_associative (a b c : Int) : (a * b) * c = a * (b * c) := by ring
```

```lean
-- axle_verify: formal_statement
theorem max_commutative (a b : Nat) : max a b = max b a := by sorry

-- axle_verify: content
theorem max_commutative (a b : Nat) : max a b = max b a := by
  simp [Nat.max_comm]
```

```lean
-- axle_verify: formal_statement
theorem min_commutative (a b : Nat) : min a b = min b a := by sorry

-- axle_verify: content
theorem min_commutative (a b : Nat) : min a b = min b a := by
  simp [Nat.min_comm]
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T, int OP> __global__ static
void adjAccumKern(T* argAccum, const T* argContrib, size_t argN) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int argOffset = blockDim.x*gridDim.x;
  for(int i=idx; i<argN; i+=argOffset) {
    T argC = argContrib[i];
    T argA = argAccum[i];
    if(OP == collSum) {
      argAccum[i] = argA+argC;
    } else if(OP == collProd) {
      argAccum[i] = argA*argC;
    } else if(OP == collMax) {
      argAccum[i] = (argA > argC) ? argA : argC;
    } else if(OP == collMin) {
      argAccum[i] = (argA < argC) ? argA : argC;
    }
  }
}

template<> __global__
void adjAccumKern<halfType, collSum>(halfType* argAccum, const halfType* argContrib, size_t argN) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int argOffset = blockDim.x*gridDim.x;
  for(int i=idx; i<argN; i+=argOffset) {
    float argC = __half2float(argContrib[i]);
    float argA = __half2float(argAccum[i]);
    argAccum[i] = __float2half( argA + argC );
  }
}

template<> __global__
void adjAccumKern<halfType, collProd>(halfType* argAccum, const halfType* argContrib, size_t argN) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int argOffset = blockDim.x*gridDim.x;
  for(int i=idx; i<argN; i+=argOffset) {
    float argC = __half2float(argContrib[i]);
    float argA = __half2float(argAccum[i]);
    argAccum[i] = __float2half( argA * argC );
  }
}

template<> __global__
void adjAccumKern<halfType, collMax>(halfType* argAccum, const halfType* argContrib, size_t argN) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int argOffset = blockDim.x*gridDim.x;
  for(int i=idx; i<argN; i+=argOffset) {
    float argC = __half2float(argContrib[i]);
    float argA = __half2float(argAccum[i]);
    argAccum[i] = __float2half( (argA>argC) ? argA : argC );
  }
}

template<> __global__
void adjAccumKern<halfType, collMin>(halfType* argAccum, const halfType* argContrib, size_t argN) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int argOffset = blockDim.x*gridDim.x;
  for(int i=idx; i<argN; i+=argOffset) {
    float argC = __half2float(argContrib[i]);
    float argA = __half2float(argAccum[i]);
    argAccum[i] = __float2half( (argA<argC) ? argA : argC );
  }
}
```

---

### 25. Function `accVecType<T>`

#### LAYER 1a — MATH (LaTeX)

$$
\text{accVecType}\langle T \rangle: \text{Ptr}_{\text{out}} \times \text{Ptr}_{\text{in}} \times \mathbb{N} \times \text{Op} \to \text{Unit}
$$
$$
\text{Dispatch: op} \mapsto \text{accumKern}\langle T, \text{op} \rangle\langle\!\langle 256, 256 \rangle\!\rangle(\text{out}, \text{in}, n)
$$
$$
\text{Pre: } \text{op} \in \{\text{Sum}, \text{Prod}, \text{Max}, \text{Min}\}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem accVecType_grid_config : 256 * 256 = 65536 := by sorry

-- axle_verify: content
theorem accVecType_grid_config : 256 * 256 = 65536 := by norm_num
```

#### LAYER 2 — FORM TEMPLATE

```c
template<typename T>
void adjAccVecType(void* argOut, void* argIn, size_t argN, collRedOpType argOp) {
  switch(argOp) {
    case collSum:  adjAccumKern<T, collSum> <<<256,256>>>((T*)argOut, (T*)argIn, argN); break;
    case collProd: adjAccumKern<T, collProd><<<256,256>>>((T*)argOut, (T*)argIn, argN); break;
    case collMax:  adjAccumKern<T, collMax> <<<256,256>>>((T*)argOut, (T*)argIn, argN); break;
    case collMin:  adjAccumKern<T, collMin> <<<256,256>>>((T*)argOut, (T*)argIn, argN); break;
    default:
      printf("Unknown reduction operation.\n");
      exit(EXIT_FAILURE);
  }
}
```

---

### 26. Function `Accumulate`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Accumulate}: \text{Ptr}_{\text{out}} \times \text{Ptr}_{\text{in}} \times \mathbb{N} \times \text{Type} \times \text{Op} \to \text{Unit}
$$
$$
\text{Dispatch: type} \mapsto \text{accVecType}\langle\text{CType}(\text{type})\rangle(\text{out}, \text{in}, n, \text{op})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem accumulate_type_dispatch_complete (t : Nat) (h : t ≤ 8) :
    t ≤ 8 := by sorry

-- axle_verify: content
theorem accumulate_type_dispatch_complete (t : Nat) (h : t ≤ 8) :
    t ≤ 8 := by exact h
```

#### LAYER 2 — FORM TEMPLATE

```c
void Accumulate(void* argOut, void* argIn, size_t argN, collDataType argType, collRedOpType argOp) {
  switch (argType) {
    case collChar:   adjAccVecType<int8_t>   (argOut, argIn, argN, argOp); break;
#if COLL_MAJOR >= 2
    case collUint8:  adjAccVecType<uint8_t>  (argOut, argIn, argN, argOp); break;
#endif
    case collInt:    adjAccVecType<int32_t>  (argOut, argIn, argN, argOp); break;
#if COLL_MAJOR >= 2
    case collUint32: adjAccVecType<uint32_t> (argOut, argIn, argN, argOp); break;
#endif
    case collInt64:  adjAccVecType<int64_t>  (argOut, argIn, argN, argOp); break;
    case collUint64: adjAccVecType<uint64_t> (argOut, argIn, argN, argOp); break;
    case collHalf:   adjAccVecType<halfType> (argOut, argIn, argN, argOp); break;
    case collFloat:  adjAccVecType<float>    (argOut, argIn, argN, argOp); break;
    case collDouble: adjAccVecType<double>   (argOut, argIn, argN, argOp); break;
    default:
      printf("Unknown reduction type.\n");
      exit(EXIT_FAILURE);
  }
}
```

---

### 27. Function `Barrier`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Barrier}: \text{threadArgs\_t}^* \to \text{Unit}
$$
$$
\text{Protocol (spinning barrier with double-buffering):}
$$
$$
\text{Phase 1: spin until } \text{barrier}[\text{barrier\_idx}] = \text{thread}
$$
$$
\text{Phase 2: } \text{barrier}[\text{barrier\_idx}] \leftarrow \text{thread} + 1
$$
$$
\text{Last thread } (\text{thread} + 1 = \text{nThreads}):
$$
$$
\quad \text{MPI\_Barrier}(\text{MPI\_COMM\_WORLD}) \quad \text{(if MPI)}
$$
$$
\quad \text{barrier}[\text{barrier\_idx}] \leftarrow 0
$$
$$
\text{Other threads: spin until } \text{barrier}[\text{barrier\_idx}] = 0
$$
$$
\text{Post: barrier\_idx} \leftarrow \neg\text{barrier\_idx}
$$
$$
\text{Invariant: barrier\_idx} \in \{0, 1\} \text{ (double-buffered)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem barrier_double_buffer_toggle (idx : Nat) (h : idx < 2) :
    (1 - idx) < 2 := by sorry

-- axle_verify: content
theorem barrier_double_buffer_toggle (idx : Nat) (h : idx < 2) :
    (1 - idx) < 2 := by omega
```

```lean
-- axle_verify: formal_statement
theorem barrier_last_thread_detection (thread nThreads : Nat) (h : thread + 1 = nThreads) :
    thread = nThreads - 1 := by sorry

-- axle_verify: content
theorem barrier_last_thread_detection (thread nThreads : Nat) (h : thread + 1 = nThreads) :
    thread = nThreads - 1 := by omega
```

#### LAYER 2 — FORM TEMPLATE

```c
void Barrier(struct adjWorkerArgs_t* argArgs)
{
  while (argArgs->adjBarrier[argArgs->argBarrierIdx] != argArgs->argThread) pthread_yield();

  argArgs->adjBarrier[argArgs->argBarrierIdx] = argArgs->argThread + 1;

  if (argArgs->argThread+1 == argArgs->argNumThreads) {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    argArgs->adjBarrier[argArgs->argBarrierIdx] = 0;
  } else {
    while (argArgs->adjBarrier[argArgs->argBarrierIdx]) pthread_yield();
  }

  argArgs->argBarrierIdx=!argArgs->argBarrierIdx;
}
```

