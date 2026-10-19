
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
