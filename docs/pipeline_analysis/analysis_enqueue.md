# NCCL Formal Analysis — src/enqueue.h

---

## 1. `enqueue<ColFunc>`

### LAYER 1a — MATH (LaTeX)

$$
\text{enqueue}: \mathcal{F} \times \text{Ptr} \times \text{Ptr} \times \mathbb{N} \times \text{DType} \times \text{Op} \times \mathbb{Z} \times \text{Comm} \times \text{Stream} \to \text{ncclResult}
$$
$$
\textbf{Pre: } \text{comm} \neq \text{null},\; \text{stream is valid CUDA stream}
$$
$$
\textbf{Post: } \text{result} = \text{colfunc}(\text{sendbuff}, \text{recvbuff}, \text{count}, \text{type}, \text{op}, \text{root}, \text{comm}, \text{stream})
$$
$$
\textbf{Invariant (ordering): } eq.\text{back}' = (eq.\text{back} + 1) \bmod \text{MAXQUEUE}
$$
$$
\textbf{Invariant (serialization): } \text{cudaEventQuery}(eq.\text{isDone}[eq.\text{back}]) = \text{NotReady} \implies \text{cudaStreamWaitEvent inserted}
$$
$$
\textbf{Post (event): } \text{cudaEventRecord}(eq.\text{isDone}[eq.\text{back}'], \text{stream}) \text{ called}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem enqueue_advances_back (back : Nat) (maxQueue : Nat) (hm : maxQueue > 0) :
    (back + 1) % maxQueue < maxQueue := by sorry

-- axle_verify: content
theorem enqueue_advances_back (back : Nat) (maxQueue : Nat) (hm : maxQueue > 0) :
    (back + 1) % maxQueue < maxQueue := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cuda
template<typename ColFuncType>
prjname::resultType enqueue(ColFuncType cbname,
                     const void* sendbufptr,
                     void* recvbufptr,
                     int countN,
                     prjname::typenameType dtypearg,
                     prjname::opType oparg,
                     int rootidx,
                     prjname::commType commobj,
                     cudaStream_t streamobj)
{
  int curDevIdx;
  CUDACHECK( cudaGetDevice(&curDevIdx) );

  EventQueue* eqptr = &commobj->events;

  cudaError_t flagval = cudaEventQuery(eqptr->isDone[eqptr->back]);
  if( flagval == cudaErrorNotReady )
    CUDACHECK( cudaStreamWaitEvent(streamobj, eqptr->isDone[eqptr->back], 0) );

  prjname::resultType resultval = cbname(sendbufptr, recvbufptr, countN, dtypearg, oparg, rootidx, commobj, streamobj);

  eqptr->back = (eqptr->back + 1) % MAXQUEUE;
  CUDACHECK( cudaEventRecord(eqptr->isDone[eqptr->back], streamobj) );
  return resultval;
}
```

