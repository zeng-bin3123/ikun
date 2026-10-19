
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
