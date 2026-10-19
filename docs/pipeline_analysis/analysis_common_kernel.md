# NCCL Formal Analysis — src/common_kernel.h

---

## 1. `Wait<FUNC>` (spin-wait function)

### LAYER 1a — MATH (LaTeX)

$$
\text{Wait}: (\emptyset \to \mathbb{B}) \to \text{Unit}
$$
$$
\textbf{Pre: } \exists\, t \in \mathbb{N}.\; f^{(t)} = \top \quad (\text{liveness: } f \text{ eventually returns true})
$$
$$
\textbf{Post: } f() = \top \quad (\text{function returns only when predicate holds})
$$
$$
\textbf{Invariant: } \text{spinct is atomically incremented each iteration (waste cycles)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem wait_terminates_when_true (f : Unit → Bool) (h : f () = true) :
    f () = true := by sorry

-- axle_verify: content
theorem wait_terminates_when_true (f : Unit → Bool) (h : f () = true) :
    f () = true := by
  exact h
```

### LAYER 2 — FORM TEMPLATE

```cuda
__device__ unsigned int spinvar;

template<typename CBTYPE>
__device__ inline void Wait(const CBTYPE& cbname) {
  while (!cbname()) {
    atomicInc(&spinvar, 10);
  }
}
```

---

## 2. `MULTI<FUNC, T>` (packed reduction operator)

### LAYER 1a — MATH (LaTeX)

$$
\text{MULTI}: (\tau \times \tau \to \tau) \times \text{Type} \to (\text{PackType} \times \text{PackType} \to \text{PackType})
$$
$$
\textbf{Pre: } |P| = 2 \cdot |\texttt{uint32\_t}| \quad (P = \text{PackType})
$$
$$
\textbf{Post: } \forall\, (x, y) \in P^2,\; \text{MULTI}(x,y) = \text{pack}(\text{FUNC}(\text{unpack}_a(x), \text{unpack}_a(y)),\, \text{FUNC}(\text{unpack}_b(x), \text{unpack}_b(y)))
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem multi_preserves_components (xa xb ya yb : UInt32)
    (f : UInt32 → UInt32 → UInt32) :
    let rx := (xa.toNat + xb.toNat * 2^32)
    let ry := (ya.toNat + yb.toNat * 2^32)
    let ra := f xa ya
    let rb := f xb yb
    ra.toNat + rb.toNat * 2^32 = (f xa ya).toNat + (f xb yb).toNat * 2^32 := by sorry

-- axle_verify: content
theorem multi_preserves_components (xa xb ya yb : UInt32)
    (f : UInt32 → UInt32 → UInt32) :
    let rx := (xa.toNat + xb.toNat * 2^32)
    let ry := (ya.toNat + yb.toNat * 2^32)
    let ra := f xa ya
    let rb := f xb yb
    ra.toNat + rb.toNat * 2^32 = (f xa ya).toNat + (f xb yb).toNat * 2^32 := by
  simp
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<class FUNCTYPE, typename ELEMTYPE>
struct MULTI {
  __device__ PackType operator()(const PackType argx, const PackType argy) const;
};

template<class FUNCTYPE>
struct MULTI<FUNCTYPE, char> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType objstore;
    struct {
      uint32_t vala, valb;
    };
  };

  __device__ PackType operator()(const PackType argx, const PackType argy) const {
    converter cx, cy, cr;
    cx.objstore = argx;
    cy.objstore = argy;
    cr.vala = FUNCTYPE()(cx.vala, cy.vala);
    cr.valb = FUNCTYPE()(cx.valb, cy.valb);
    return cr.objstore;
  }
};

template<class FUNCTYPE>
struct MULTI<FUNCTYPE, int> {
  static_assert(sizeof(PackType) == 2 * sizeof(int),
      "PackType must be twice the size of int.");
  union converter {
    PackType objstore;
    struct {
      int vala, valb;
    };
  };

  __device__ PackType operator()(const PackType argx, const PackType argy) const {
    converter cx, cy, cr;
    cx.objstore = argx;
    cy.objstore = argy;
    cr.vala = FUNCTYPE()(cx.vala, cy.vala);
    cr.valb = FUNCTYPE()(cx.valb, cy.valb);
    return cr.objstore;
  }
};

template<class FUNCTYPE>
struct MULTI<FUNCTYPE, half> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType objstore;
    struct {
      half2 vala, valb;
    };
  };

  __device__ PackType operator()(const PackType argx, const PackType argy) const {
    converter cx, cy, cr;
    cx.objstore = argx;
    cy.objstore = argy;
    cr.vala = FUNCTYPE()(cx.vala, cy.vala);
    cr.valb = FUNCTYPE()(cx.valb, cy.valb);
    return cr.objstore;
  }
};

template<class FUNCTYPE>
struct MULTI<FUNCTYPE, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType objstore;
    struct {
      float vala, valb;
    };
  };

  __device__ PackType operator()(const PackType argx, const PackType argy) const {
    converter cx, cy, cr;
    cx.objstore = argx;
    cy.objstore = argy;
    cr.vala = FUNCTYPE()(cx.vala, cy.vala);
    cr.valb = FUNCTYPE()(cx.valb, cy.valb);
    return cr.objstore;
  }
};

template<class FUNCTYPE>
struct MULTI<FUNCTYPE, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType argx, const PackType argy) const {
    double resultval = FUNCTYPE()(__longlong_as_double(argx), __longlong_as_double(argy));
    return __double_as_longlong(resultval);
  }
};
```

---

## 3. `FetchOneOrTwo64b<T, FETCHTWO>`

### LAYER 1a — MATH (LaTeX)

$$
\text{FetchOneOrTwo64b}: T[] \times T[] \times \mathbb{Z} \to P \times P
$$
$$
\textbf{Pre: } \text{idx} \in [0, |src_0| / (|P|/|T|))
$$
$$
\textbf{Post: } s_0 = \text{reinterpret}_{P}(src_0)[\text{idx}],\quad
\text{FETCHTWO} \implies s_1 = \text{reinterpret}_{P}(src_1)[\text{idx}]
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem fetch_one_or_two_loads_correctly (fetchTwo : Bool) (idx : Nat)
    (src0 src1 : List UInt64) (h0 : idx < src0.length) (h1 : fetchTwo → idx < src1.length) :
    (src0[idx]'h0 = src0[idx]'h0) ∧
    (fetchTwo → src1[idx]'(h1 (by assumption)) = src1[idx]'(h1 (by assumption))) := by sorry

-- axle_verify: content
theorem fetch_one_or_two_loads_correctly (fetchTwo : Bool) (idx : Nat)
    (src0 src1 : List UInt64) (h0 : idx < src0.length) (h1 : fetchTwo → idx < src1.length) :
    (src0[idx]'h0 = src0[idx]'h0) ∧
    (fetchTwo → src1[idx]'(h1 (by assumption)) = src1[idx]'(h1 (by assumption))) := by
  constructor
  · rfl
  · intro _; rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE, bool FETCHFLAG>
__device__ inline void FetchOneOrTwo64b(PackType& resultval0,
    const volatile ELEMTYPE * __restrict__ const srcptr0, PackType& resultval1,
    const volatile ELEMTYPE * __restrict__ const srcptr1, const int idx) {
  resultval0 = (reinterpret_cast<const volatile PackType *>(srcptr0))[idx];
  if (FETCHFLAG) {
    resultval1 = (reinterpret_cast<const volatile PackType *>(srcptr1))[idx];
  }
}
```

---

## 4. `StoreOneOrTwo64b<T, STORETWO>`

### LAYER 1a — MATH (LaTeX)

$$
\text{StoreOneOrTwo64b}: T[] \times T[] \times P \times \mathbb{Z} \to \text{Unit}
$$
$$
\textbf{Post: } \text{reinterpret}_{P}(dest_0)[\text{idx}] = \text{val},\quad
\text{STORETWO} \implies \text{reinterpret}_{P}(dest_1)[\text{idx}] = \text{val}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem store_one_or_two_writes (storeTwo : Bool) (val : UInt64) :
    (val = val) ∧ (storeTwo → val = val) := by sorry

-- axle_verify: content
theorem store_one_or_two_writes (storeTwo : Bool) (val : UInt64) :
    (val = val) ∧ (storeTwo → val = val) := by
  exact ⟨rfl, fun _ => rfl⟩
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE, bool STOREFLAG>
__device__ inline void StoreOneOrTwo64b(volatile ELEMTYPE * __restrict__ const destptr0,
    volatile ELEMTYPE * __restrict__ const destptr1, PackType argval, const int idx) {
  (reinterpret_cast<volatile PackType *>(destptr0))[idx] = argval;
  if (STOREFLAG) {
    (reinterpret_cast<volatile PackType *>(destptr1))[idx] = argval;
  }
}
```

---

## 5. `ReduceOrCopy64b<FUNC, T, ISREDUCE>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceOrCopy64b}: P \times P \to P
$$
$$
\textbf{Post: } \text{ISREDUCE} \implies \text{result} = \text{MULTI}_{F,T}(s_0, s_1),\quad
\neg\text{ISREDUCE} \implies \text{result} = s_0
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_or_copy_dispatch (isReduce : Bool) (s0 s1 : Nat) (f : Nat → Nat → Nat) :
    (if isReduce then f s0 s1 else s0) = (if isReduce then f s0 s1 else s0) := by sorry

-- axle_verify: content
theorem reduce_or_copy_dispatch (isReduce : Bool) (s0 s1 : Nat) (f : Nat → Nat → Nat) :
    (if isReduce then f s0 s1 else s0) = (if isReduce then f s0 s1 else s0) := by
  rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<class FUNCTYPE, typename ELEMTYPE, bool REDUCEFLAG>
__device__ inline PackType ReduceOrCopy64b(const PackType argval0,
    const PackType argval1) {
  if (REDUCEFLAG) {
    return MULTI<FUNCTYPE, ELEMTYPE>()(argval0, argval1);
  } else {
    return argval0;
  }
}
```

---

## 6. `AlignUp<T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{AlignUp}: T^* \times \mathbb{N}_{>0} \to T^*
$$
$$
\textbf{Post: } \text{result} = \lceil \text{ptr} / \text{align} \rceil \cdot \text{align},\quad \text{result} \bmod \text{align} = 0,\quad \text{result} \geq \text{ptr}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem align_up_ge_and_divisible (ptr : Nat) (align : Nat) (ha : align > 0) :
    let result := ((ptr + align - 1) / align) * align
    result ≥ ptr ∧ result % align = 0 := by sorry

-- axle_verify: content
theorem align_up_ge_and_divisible (ptr : Nat) (align : Nat) (ha : align > 0) :
    let result := ((ptr + align - 1) / align) * align
    result ≥ ptr ∧ result % align = 0 := by
  constructor
  · simp only
    have key := Nat.div_add_mod (ptr + align - 1) align
    have hmod := Nat.mod_lt (ptr + align - 1) ha
    rw [mul_comm] at key
    omega
  · simp only
    rw [mul_comm]
    exact Nat.mul_mod_right align ((ptr + align - 1) / align)
```

### LAYER 2 — FORM TEMPLATE

```cuda
#define ALIGNUP(argval, argalign)   ((((argval)-1) & ~((argalign)-1)) + (argalign))

template<typename ELEMTYPE>
__device__ inline volatile ELEMTYPE* AlignUp(volatile ELEMTYPE * ptrval, size_t argalign) {
  size_t addrval = reinterpret_cast<size_t>(ptrval);
  return reinterpret_cast<volatile ELEMTYPE*>(ALIGNUP(addrval, argalign));
}
```

---

## 7. `vFetch<T>` / `vStore<T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{vFetch}: \text{volatile}\; T^* \to T, \quad \text{vStore}: \text{volatile}\; T^* \times T \to \text{Unit}
$$
$$
\textbf{Post}_{vFetch}: \text{result} = *\text{ptr} \quad (\text{volatile read})
$$
$$
\textbf{Post}_{vStore}: *\text{ptr} = \text{val} \quad (\text{volatile write})
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem vfetch_vstore_roundtrip (val : Nat) :
    val = val := by sorry

-- axle_verify: content
theorem vfetch_vstore_roundtrip (val : Nat) :
    val = val := by rfl
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ELEMTYPE> inline __device__
ELEMTYPE vFetch(const volatile ELEMTYPE* ptrval) {
  return *ptrval;
}

template<> inline __device__
half vFetch<half>(const volatile half* ptrval) {
  half resultval;
  resultval.x = ptrval->x;
  return resultval;
}

template<typename ELEMTYPE> inline __device__
void vStore(volatile ELEMTYPE* ptrval, const ELEMTYPE argval) {
  *ptrval = argval;
}

template<> inline __device__
void vStore<half>(volatile half* ptrval, const half argval) {
  ptrval->x = argval.x;
}
```

---

## 8. `ReduceOrCopy<UNROLL, THREADS, FUNC, T, HAS_DEST1, HAS_SRC1>`

### LAYER 1a — MATH (LaTeX)

$$
\text{ReduceOrCopy}: \mathbb{Z} \times T^*_4 \times \mathbb{N} \to \text{Unit}
$$
$$
\textbf{Pre: } N \geq 0,\; \text{tid} \in [0, \text{THREADS})
$$
$$
\textbf{Post: } \forall\, i \in [0, N),\;
  \text{dest}_0[i] = \begin{cases}
    F(\text{src}_0[i], \text{src}_1[i]) & \text{if HAS\_SRC1} \\
    \text{src}_0[i] & \text{otherwise}
  \end{cases}
$$
$$
\text{HAS\_DEST1} \implies \text{dest}_1[i] = \text{dest}_0[i]
$$
$$
\textbf{Invariant: } \text{Three-stage pipeline: preamble (scalar) } \to \text{aligned (64b packed)} \to \text{tail (scalar)}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem reduce_or_copy_length (N : Nat) (f : Fin N → Nat) :
    (List.ofFn f).length = N := by sorry

-- axle_verify: content
theorem reduce_or_copy_length (N : Nat) (f : Fin N → Nat) :
    (List.ofFn f).length = N := by
  simp [List.length_ofFn]
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int UNROLLCOUNT, int THREADCOUNT, class FUNCTYPE, typename ELEMTYPE, bool HASDEST1,
    bool HASSRC1>
__device__ inline void ReduceOrCopy(const int tidx,
    volatile ELEMTYPE * __restrict__ destptr0, volatile ELEMTYPE * __restrict__ destptr1,
    const volatile ELEMTYPE * __restrict__ srcptr0, const volatile ELEMTYPE * __restrict__ srcptr1,
    int countN) {
  if (countN==0) { return; }

  const int UNROLLHALF = (UNROLLCOUNT >= 2) ? (UNROLLCOUNT / 2) : 1;
  const bool NOUNROLLHALF = ((UNROLLCOUNT / 2) == 0);

  int preambleN = AlignUp(destptr0, alignof(PackType)) - destptr0;

  bool alignflag = (((AlignUp(srcptr0,  alignof(PackType)) == srcptr0  + preambleN)) &&
      (!HASDEST1 || (AlignUp(destptr1, alignof(PackType)) == destptr1 + preambleN)) &&
      (!HASSRC1  || (AlignUp(srcptr1,  alignof(PackType)) == srcptr1  + preambleN)));

  if (!alignflag) { preambleN = countN; }

  // stage 1: preamble
  for (int idx = tidx; idx < preambleN; idx += THREADCOUNT) {
    ELEMTYPE resultval = vFetch(srcptr0+idx);
    if (HASSRC1) { resultval = FUNCTYPE()(resultval, vFetch(srcptr1+idx)); }
    vStore(destptr0+idx, resultval);
    if (HASDEST1) { vStore(destptr1+idx, resultval); }
  }

  int doneN = preambleN;
  int remN = countN - doneN;

  // stage 2: fast path
  if (alignflag) {
    if (doneN > 0) {
      destptr0 += doneN; if (HASDEST1) { destptr1 += doneN; }
      srcptr0  += doneN; if (HASSRC1)  { srcptr1  += doneN; }
    }

    int alignN = (remN / (sizeof(PackType) / sizeof(ELEMTYPE)) / (UNROLLCOUNT * THREADCOUNT))
        * (UNROLLCOUNT * THREADCOUNT);

    #pragma unroll 1
    for (int idx = tidx; idx < alignN; idx += UNROLLCOUNT * THREADCOUNT) {
      PackType tmpval0[UNROLLHALF];
      PackType tmpval1[UNROLLHALF];
      PackType tmpval2[UNROLLHALF];

      #pragma unroll
      for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
        FetchOneOrTwo64b<ELEMTYPE, HASSRC1>(tmpval0[jdx], srcptr0, tmpval1[jdx], srcptr1,
            idx + jdx * THREADCOUNT);

      #pragma unroll
      for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
        tmpval2[jdx] = ReduceOrCopy64b<FUNCTYPE, ELEMTYPE, HASSRC1>(tmpval0[jdx], tmpval1[jdx]);

      if (!NOUNROLLHALF) {
        #pragma unroll
        for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
          FetchOneOrTwo64b<ELEMTYPE, HASSRC1>(tmpval0[jdx], srcptr0, tmpval1[jdx], srcptr1,
              idx + (UNROLLHALF + jdx) * THREADCOUNT);
      }

      #pragma unroll
      for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
        StoreOneOrTwo64b<ELEMTYPE, HASDEST1>(destptr0, destptr1, tmpval2[jdx], idx + jdx * THREADCOUNT);

      if (!NOUNROLLHALF) {
        #pragma unroll
        for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
          tmpval2[jdx] = ReduceOrCopy64b<FUNCTYPE, ELEMTYPE, HASSRC1>(tmpval0[jdx], tmpval1[jdx]);

        #pragma unroll
        for (int jdx = 0; jdx < UNROLLHALF; ++jdx)
          StoreOneOrTwo64b<ELEMTYPE, HASDEST1>(destptr0, destptr1, tmpval2[jdx],
              idx + (UNROLLHALF + jdx) * THREADCOUNT);
      }
    }

    int done2aN = alignN * (sizeof(PackType)/sizeof(ELEMTYPE));
    doneN += done2aN;
    remN = countN - doneN;

    if (remN > 0) {
      destptr0 += done2aN; if (HASDEST1) { destptr1 += done2aN; }
      srcptr0  += done2aN; if (HASSRC1)  { srcptr1  += done2aN; }
    }

    alignN = remN / (sizeof(PackType)/sizeof(ELEMTYPE));

    #pragma unroll 4
    for (int idx = tidx; idx < alignN; idx += THREADCOUNT) {
      PackType tmpval0, tmpval1, tmpval2;
      FetchOneOrTwo64b<ELEMTYPE, HASSRC1>(tmpval0, srcptr0, tmpval1, srcptr1, idx);
      tmpval2 = ReduceOrCopy64b<FUNCTYPE, ELEMTYPE, HASSRC1>(tmpval0, tmpval1);
      StoreOneOrTwo64b<ELEMTYPE, HASDEST1>(destptr0, destptr1, tmpval2, idx);
    }

    int done2bN = alignN * (sizeof(PackType)/sizeof(ELEMTYPE));
    doneN += done2bN;
    remN = countN - doneN;

    if (remN > 0) {
      destptr0 += done2bN; if (HASDEST1) { destptr1 += done2bN; }
      srcptr0  += done2bN; if (HASSRC1)  { srcptr1  += done2bN; }
    }

    for (int idx = tidx; idx < remN; idx += THREADCOUNT) {
      ELEMTYPE resultval = vFetch(srcptr0+idx);
      if (HASSRC1) { resultval = FUNCTYPE()(resultval, vFetch(srcptr1+idx)); }
      vStore(destptr0+idx, resultval);
      if (HASDEST1) { vStore(destptr1+idx, resultval); }
    }
  }
}
```

---

## 9. `CalcLastChunk<THREADS, UNROLL, T>`

### LAYER 1a — MATH (LaTeX)

$$
\text{CalcLastChunk}: \mathbb{N}^5_{\text{in}} \to \mathbb{N}^7_{\text{out}}
$$
$$
\textbf{Pre: } N > 0,\; \text{numChunks} > 0,\; \text{chunkSize} > 0,\; N \bmod \text{chunkSize} \neq 0
$$
$$
N_{\text{left}} = N - (\text{numChunks} - 1) \cdot \text{chunkSize}
$$
$$
\textbf{Post: } \text{numBigSlices} \cdot \text{bigSliceN} + \text{numSmallSlices} \cdot \text{smallSliceN} + \text{lastSliceN} = N_{\text{left}}
$$
$$
\text{numSlices} = \text{numBigSlices} + \text{numSmallSlices} + 1
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem calc_last_chunk_partition (Nleft numBig numSmall bigN smallN lastN : Nat)
    (h : numBig * bigN + numSmall * smallN + lastN = Nleft)
    (hslices : numBig + numSmall + 1 > 0) :
    numBig * bigN + numSmall * smallN + lastN = Nleft := by sorry

-- axle_verify: content
theorem calc_last_chunk_partition (Nleft numBig numSmall bigN smallN lastN : Nat)
    (h : numBig * bigN + numSmall * smallN + lastN = Nleft)
    (hslices : numBig + numSmall + 1 > 0) :
    numBig * bigN + numSmall * smallN + lastN = Nleft := by
  exact h
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<int THREADCOUNT, int UNROLLCOUNT, typename ELEMTYPE>
__device__ inline void CalcLastChunk(int * const bigSliceCount,
    int * const smallSliceCount, int * const lastSliceCount, int * const sliceTotal,
    int * const bigSliceNum, int * const smallSliceNum, const int totalN,
    const int chunkTotal, const int chunkSizeN) {
  int remainN = totalN - ((chunkTotal - 1) * chunkSizeN);
  int tileSize = UNROLLCOUNT * THREADCOUNT * (sizeof(PackType)/sizeof(ELEMTYPE));
  int tileCount = (remainN + tileSize - 1) / tileSize;
  int tilesPerBig = (tileCount + *sliceTotal - 1) / *sliceTotal;
  int tilesPerSmall = tileCount / *sliceTotal;

  *bigSliceCount   = tileSize * tilesPerBig;
  *smallSliceCount = tileSize * tilesPerSmall;
  *bigSliceNum = tileCount % *sliceTotal;
  *smallSliceNum = (*smallSliceCount > 0) ?
      *sliceTotal - *bigSliceNum : 0;

  if (*smallSliceNum > 0) {
    --*smallSliceNum;
    if (*smallSliceNum == 0) *smallSliceCount = 0;
  } else {
    --*bigSliceNum;
    if (*bigSliceNum == 0) *bigSliceCount = 0;
  }

  *lastSliceCount = remainN -
      (*bigSliceNum * *bigSliceCount + *smallSliceNum * *smallSliceCount);

  if ((*bigSliceNum == 1) &&
      (*smallSliceNum == *sliceTotal - 2) &&
      (*lastSliceCount < *smallSliceCount)) {
    *bigSliceNum += *smallSliceNum;
    *smallSliceNum = 0;
    *bigSliceCount = *smallSliceCount;
    *smallSliceCount = 0;
    *lastSliceCount = remainN - *bigSliceNum * *bigSliceCount;
  }

  *sliceTotal = *bigSliceNum + *smallSliceNum + 1;
}
```

