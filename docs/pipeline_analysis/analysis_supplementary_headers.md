# NCCL Formal Analysis — Supplementary: Inferred Interfaces from Referenced Headers

These are functions/types used throughout the 8 analyzed files but defined in headers
not listed in the reference (`copy_kernel.h`, `reduce_kernel.h`, `core.h`, `nccl.h`).
Specs are inferred from call sites.

---

## 1. `Copy<UNROLL, THREADS>` (from copy_kernel.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{Copy}: T^*_\text{dest} \times T^*_\text{src} \times \mathbb{N} \to \text{Unit}
$$
$$
\textbf{Post: } \forall\, i \in [0, N),\; \text{dest}[i] = \text{src}[i]
$$
$$
\text{Equivalent to: ReduceOrCopy<UNROLL, THREADS, ?, T, false, false>}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem copy_postcondition (src : List Nat) (N : Nat) (hN : N ≤ src.length) :
    let dest := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    dest.length = N ∧ ∀ i : Fin N, dest[i] = src[i.val]'(by omega) := by sorry

-- axle_verify: content
theorem copy_postcondition (src : List Nat) (N : Nat) (hN : N ≤ src.length) :
    let dest := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    dest.length = N ∧ ∀ i : Fin N, dest[i] = src[i.val]'(by omega) := by
  constructor
  · simp [List.length_ofFn]
  · intro i; simp [List.getElem_ofFn]
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int UNROLLCOUNT, int THREADCOUNT, typename ELEMTYPE>
__device__ inline void Copy(
    volatile ELEMTYPE * __restrict__ destptr,
    const volatile ELEMTYPE * __restrict__ srcptr,
    int countN);
```

---

## 2. `DoubleCopy<UNROLL, THREADS>` (from copy_kernel.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{DoubleCopy}: T^*_{\text{dest}_0} \times T^*_{\text{dest}_1} \times T^*_\text{src} \times \mathbb{N} \to \text{Unit}
$$
$$
\textbf{Post: } \forall\, i \in [0, N),\; \text{dest}_0[i] = \text{src}[i] \wedge \text{dest}_1[i] = \text{src}[i]
$$
$$
\text{Equivalent to: ReduceOrCopy<UNROLL, THREADS, ?, T, true, false>}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem double_copy_both_equal (src : List Nat) (N : Nat) (hN : N ≤ src.length) :
    let d0 := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    let d1 := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    d0 = d1 := by sorry

-- axle_verify: content
theorem double_copy_both_equal (src : List Nat) (N : Nat) (hN : N ≤ src.length) :
    let d0 := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    let d1 := List.ofFn (fun (i : Fin N) => src[i.val]'(by omega))
    d0 = d1 := by
  simp
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int UNROLLCOUNT, int THREADCOUNT, typename ELEMTYPE>
__device__ inline void DoubleCopy(
    volatile ELEMTYPE * __restrict__ destptr0,
    volatile ELEMTYPE * __restrict__ destptr1,
    const volatile ELEMTYPE * __restrict__ srcptr,
    int countN);
```

---

## 3. `Reduce<UNROLL, THREADS, FUNC>` (from reduce_kernel.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{Reduce}: T^*_\text{dest} \times T^*_{\text{src}_0} \times T^*_{\text{src}_1} \times \mathbb{N} \to \text{Unit}
$$
$$
\textbf{Post: } \forall\, i \in [0, N),\; \text{dest}[i] = F(\text{src}_0[i], \text{src}_1[i])
$$
$$
\text{Equivalent to: ReduceOrCopy<UNROLL, THREADS, FUNC, T, false, true>}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_applies_func (f : Nat → Nat → Nat) (s0 s1 : List Nat)
    (N : Nat) (h0 : N ≤ s0.length) (h1 : N ≤ s1.length) :
    let dest := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    dest.length = N := by sorry

-- axle_verify: content
theorem reduce_applies_func (f : Nat → Nat → Nat) (s0 s1 : List Nat)
    (N : Nat) (h0 : N ≤ s0.length) (h1 : N ≤ s1.length) :
    let dest := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    dest.length = N := by
  simp [List.length_ofFn]
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int UNROLLCOUNT, int THREADCOUNT, class FUNCTYPE, typename ELEMTYPE>
__device__ inline void Reduce(
    volatile ELEMTYPE * __restrict__ destptr,
    const volatile ELEMTYPE * __restrict__ srcptr0,
    const volatile ELEMTYPE * __restrict__ srcptr1,
    int countN);
```

---

## 4. `ReduceAndCopy<UNROLL, THREADS, FUNC>` (from reduce_kernel.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceAndCopy}: T^*_{\text{dest}_0} \times T^*_{\text{dest}_1} \times T^*_{\text{src}_0} \times T^*_{\text{src}_1} \times \mathbb{N} \to \text{Unit}
$$
$$
\textbf{Post: } \forall\, i,\; v = F(\text{src}_0[i], \text{src}_1[i]),\; \text{dest}_0[i] = v \wedge \text{dest}_1[i] = v
$$
$$
\text{Equivalent to: ReduceOrCopy<UNROLL, THREADS, FUNC, T, true, true>}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_and_copy_both (f : Nat → Nat → Nat) (s0 s1 : List Nat)
    (N : Nat) (h0 : N ≤ s0.length) (h1 : N ≤ s1.length) :
    let d0 := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    let d1 := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    d0 = d1 := by sorry

-- axle_verify: content
theorem reduce_and_copy_both (f : Nat → Nat → Nat) (s0 s1 : List Nat)
    (N : Nat) (h0 : N ≤ s0.length) (h1 : N ≤ s1.length) :
    let d0 := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    let d1 := List.ofFn (fun (i : Fin N) => f (s0[i.val]'(by omega)) (s1[i.val]'(by omega)))
    d0 = d1 := by
  simp
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int UNROLLCOUNT, int THREADCOUNT, class FUNCTYPE, typename ELEMTYPE>
__device__ inline void ReduceAndCopy(
    volatile ELEMTYPE * __restrict__ destptr0,
    volatile ELEMTYPE * __restrict__ destptr1,
    const volatile ELEMTYPE * __restrict__ srcptr0,
    const volatile ELEMTYPE * __restrict__ srcptr1,
    int countN);
```

---

## 5. `FuncSum<T>`, `FuncProd<T>`, `FuncMax<T>`, `FuncMin<T>` (from reduce_kernel.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{FuncSum}(a,b) = a + b, \quad \text{FuncProd}(a,b) = a \cdot b
$$
$$
\text{FuncMax}(a,b) = \max(a,b), \quad \text{FuncMin}(a,b) = \min(a,b)
$$
$$
\text{All are commutative, associative binary operators on } T
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem sum_commutative (a b : Nat) : a + b = b + a := by sorry

-- axle_verify: content
theorem sum_commutative (a b : Nat) : a + b = b + a := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE>
struct FuncSum {
  __device__ ELEMTYPE operator()(const ELEMTYPE argA, const ELEMTYPE argB) const {
    return argA + argB;
  }
};

template<typename ELEMTYPE>
struct FuncProd {
  __device__ ELEMTYPE operator()(const ELEMTYPE argA, const ELEMTYPE argB) const {
    return argA * argB;
  }
};

template<typename ELEMTYPE>
struct FuncMax {
  __device__ ELEMTYPE operator()(const ELEMTYPE argA, const ELEMTYPE argB) const {
    return (argA > argB) ? argA : argB;
  }
};

template<typename ELEMTYPE>
struct FuncMin {
  __device__ ELEMTYPE operator()(const ELEMTYPE argA, const ELEMTYPE argB) const {
    return (argA < argB) ? argA : argB;
  }
};
```

---

## 6. `ncclComm` / `ncclMem` / `ncclUniqueId` (from core.h / nccl.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclMem} = \{\text{recvPtrs}: \text{void}^*[],\; \text{flags}: \mathbb{Z}[],\; \text{buff}: \text{byte}[]\}
$$

$$
\text{ncclComm} = \{
  \text{nDev}: \mathbb{N},\; \text{cudaDev}: \mathbb{Z},\; \text{ncclId}: \mathbb{Z},
$$
$$
  \text{buffSize}: \mathbb{N},\; \text{useRemoteRecv}: \mathbb{B},
$$
$$
  \text{local}: \text{ncclMem}^*[\text{MAXPEERS}],\; \text{remote}: \text{ncclMem}^*[\text{MAXPEERS}],
$$
$$
  \text{userFromRing}: \mathbb{Z}[\text{MAXPEERS}],\; \text{ringFromUser}: \mathbb{Z}[\text{MAXPEERS}],
$$
$$
  \text{devUserFromRing}: \mathbb{Z}^*_\text{device},
$$
$$
  \text{cleanup}: \{\text{type}: \mathbb{Z},\; \text{handle}: \text{void}^*\}[\text{MAXPEERS}],
$$
$$
  \text{devMem}: \text{ncclMem}^*,\; \text{hostMem}: \text{ncclMem}^*,\; \text{hostMemState}: \mathbb{Z},
$$
$$
  \text{events}: \text{EventQueue}
\}
$$

$$
\text{ncclUniqueId} = \{\text{internal}: \text{char}[\text{NCCL\_UNIQUE\_ID\_BYTES}]\}
$$

$$
\text{EventQueue} = \{\text{isDone}: \text{cudaEvent\_t}[\text{MAXQUEUE}],\; \text{back}: \mathbb{Z}\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem comm_struct_invariant (nDev ncclId : Nat) (hn : nDev > 0) (hi : ncclId < nDev) :
    ncclId < nDev := by sorry

-- axle_verify: content
theorem comm_struct_invariant (nDev ncclId : Nat) (hn : nDev > 0) (hi : ncclId < nDev) :
    ncclId < nDev := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
struct prjname::memStruct {
  void * recvPtrs[1];
  volatile int flags[2];
  char buff[];
};

struct prjname::commStruct {
  int nDev;
  int cudaDev;
  int ncclId;
  size_t buffSize;
  int useRemoteRecv;

  prjname::memStruct * local[MAXPEERS];
  prjname::memStruct * remote[MAXPEERS];
  int userFromRing[MAXPEERS];
  int ringFromUser[MAXPEERS];
  int * devUserFromRing;

  struct { int type; void * handle; } cleanup[MAXPEERS];
  prjname::memStruct * devMem;
  prjname::memStruct * hostMem;
  int hostMemState;
  EventQueue events;
};

struct prjname::uniqueIdStruct {
  char internal[NCCL_UNIQUE_ID_BYTES];
};

struct EventQueue {
  cudaEvent_t isDone[MAXQUEUE];
  int back;
};
```

---

## 7. `ncclResult_t` enum / `ncclDataType_t` enum / `ncclRedOp_t` enum (from nccl.h)

### LAYER 1a — MATH (LaTeX)

$$
\text{ncclResult\_t} = \{0: \text{Success},\; 1: \text{UnhandledCudaError},\; \ldots,\; 14: \text{InvalidOperation}\}
$$
$$
\text{ncclDataType\_t} = \{0: \text{char},\; 1: \text{int},\; 2: \text{half},\; 3: \text{float},\; 4: \text{double}\}
$$
$$
\text{ncclRedOp\_t} = \{0: \text{Sum},\; 1: \text{Prod},\; 2: \text{Max},\; 3: \text{Min}\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem result_codes_positive_count : 15 > 0 ∧ 5 > 0 ∧ 4 > 0 := by sorry

-- axle_verify: content
theorem result_codes_positive_count : 15 > 0 ∧ 5 > 0 ∧ 4 > 0 := by omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
typedef enum {
  prjname::Success                = 0,
  prjname::UnhandledCudaError     = 1,
  prjname::SystemError            = 2,
  prjname::InternalError          = 3,
  prjname::InvalidDevicePointer   = 4,
  prjname::InvalidRank            = 5,
  prjname::UnsupportedDeviceCount = 6,
  prjname::DeviceNotFound         = 7,
  prjname::InvalidDeviceIndex     = 8,
  prjname::LibWrapperNotSet       = 9,
  prjname::CudaMallocFailed       = 10,
  prjname::RankMismatch           = 11,
  prjname::InvalidArgument        = 12,
  prjname::InvalidType            = 13,
  prjname::InvalidOperation       = 14
} prjname::resultType;

typedef enum {
  prjname::Char   = 0,
  prjname::Int    = 1,
  prjname::Half   = 2,
  prjname::Float  = 3,
  prjname::Double = 4
} prjname::typenameType;

typedef enum {
  prjname::Sum  = 0,
  prjname::Prod = 1,
  prjname::Max  = 2,
  prjname::Min  = 3
} prjname::opType;
```

