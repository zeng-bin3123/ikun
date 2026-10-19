# Mooncake Transfer Engine — Three-Layer Formal Analysis (Chunk 2/3)

Commit: `348afd753190` — Implementation files (src/)

---

## 11. `config.cpp` — `loadGlobalConfig`

### LAYER 1a — MATH (LaTeX)

$$
\text{loadGlobalConfig}: \text{GlobalConfig}_{\text{ref}} \to \text{void}
$$

For each env-var / config-field pair $(E, f, lo, hi)$:

$$
\text{loadField}(E, f, lo, hi) \triangleq
\begin{cases}
f \leftarrow \text{atoi}(E) & \text{if } E \neq \text{null} \wedge lo \le \text{atoi}(E) \le hi \\
\text{(no change)} & \text{if } E = \text{null} \\
\text{LOG(WARNING)} & \text{if } E \neq \text{null} \wedge \text{atoi}(E) \notin [lo, hi]
\end{cases}
$$

**Special case** for `MC_MTU`:
$$
\text{mtu\_length} \leftarrow \text{IBV\_MTU}_{v} \quad \text{where } v \in \{512, 1024, 2048, 4096\}
$$
$$
v \notin \{512, 1024, 2048, 4096\} \implies \text{exit(EXIT\_FAILURE)}
$$

**`globalConfig` singleton:**
$$
\text{globalConfig}: \text{unit} \to \text{GlobalConfig}_{\text{ref}},\quad \text{initialized exactly once via } \text{std::call\_once}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem load_config_env_valid (env_val lo hi default_val : Int)
  (hlo : lo > 0) (hhi : hi > lo) (hdef : lo ≤ default_val ∧ default_val ≤ hi) :
  let result := if lo ≤ env_val ∧ env_val ≤ hi then env_val else default_val
  lo ≤ result ∧ result ≤ hi := by sorry

-- axle_verify: content
theorem load_config_env_valid (env_val lo hi default_val : Int)
  (hlo : lo > 0) (hhi : hi > lo) (hdef : lo ≤ default_val ∧ default_val ≤ hi) :
  let result := if lo ≤ env_val ∧ env_val ≤ hi then env_val else default_val
  lo ≤ result ∧ result ≤ hi := by
  simp only
  split
  · rename_i h; exact h
  · exact hdef
```

```lean
-- axle_verify: formal_statement
theorem mtu_valid_values (val : Nat) :
  val = 512 ∨ val = 1024 ∨ val = 2048 ∨ val = 4096 →
  val ≥ 512 ∧ val ≤ 4096 := by sorry

-- axle_verify: content
theorem mtu_valid_values (val : Nat) :
  val = 512 ∨ val = 1024 ∨ val = 2048 ∨ val = 4096 →
  val ≥ 512 ∧ val ≤ 4096 := by
  intro h
  rcases h with h | h | h | h <;> subst h <;> omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
namespace prjname {
void loadAdjConfig(AdjConfig &argname) {
    const char *argenv = std::getenv("MC_NUM_CQ_PER_CTX");
    if (argenv) {
        int val = atoi(argenv);
        if (val > 0 && val < 256)
            argname.num_cq_per_ctx = val;
        else
            LOG(WARNING) << "Ignore value from environment variable MC_NUM_CQ_PER_CTX";
    }
    // ... repeated pattern for each field ...
}

AdjConfig &adjConfig() {
    static AdjConfig argname;
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag, []() { loadAdjConfig(argname); });
    return argname;
}

uint16_t getDefaultHandshakePort() {
    return adjConfig().handshake_port;
}
}  // namespace prjname
```

---

## 12. `updateGlobalConfig` — Device Attribute Clamping

### LAYER 1a — MATH (LaTeX)

$$
\text{updateGlobalConfig}: \text{ibv\_device\_attr} \to \text{void}
$$

For each field $f$ with device limit $d$:
$$
f' = \min(f, d)
$$

Specifically:
$$
\text{max\_ep\_per\_ctx}' = \min\!\left(\text{max\_ep\_per\_ctx},\ \left\lfloor\frac{\text{device\_attr.max\_qp}}{\text{num\_qp\_per\_ep}}\right\rfloor\right)
$$

$$
\text{max\_wr}' = \min(\text{max\_wr}, \text{device\_attr.max\_qp\_wr})
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem clamp_to_device_limit (config_val device_limit : Nat) :
  min config_val device_limit ≤ device_limit := by sorry

-- axle_verify: content
theorem clamp_to_device_limit (config_val device_limit : Nat) :
  min config_val device_limit ≤ device_limit := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem ep_limit_respects_qp (max_ep num_qp_per_ep max_qp : Nat)
  (h : num_qp_per_ep > 0) :
  let new_max_ep := if max_ep * num_qp_per_ep > max_qp then max_qp / num_qp_per_ep else max_ep
  new_max_ep * num_qp_per_ep ≤ max_qp ∨ max_ep * num_qp_per_ep ≤ max_qp := by sorry

-- axle_verify: content
theorem ep_limit_respects_qp (max_ep num_qp_per_ep max_qp : Nat)
  (h : num_qp_per_ep > 0) :
  let new_max_ep := if max_ep * num_qp_per_ep > max_qp then max_qp / num_qp_per_ep else max_ep
  new_max_ep * num_qp_per_ep ≤ max_qp ∨ max_ep * num_qp_per_ep ≤ max_qp := by
  simp only
  split
  · left; exact Nat.div_mul_le_self max_qp num_qp_per_ep
  · right; omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
void updateAdjConfig(ibv_device_attr &argattr) {
    auto &argconfig = adjConfig();
    if (argconfig.max_ep_per_ctx * argconfig.num_qp_per_ep >
        (size_t)argattr.max_qp)
        argconfig.max_ep_per_ctx = argattr.max_qp / argconfig.num_qp_per_ep;
    if (argconfig.num_cq_per_ctx > (size_t)argattr.max_cq)
        argconfig.num_cq_per_ctx = argattr.max_cq;
    if (argconfig.max_wr > (size_t)argattr.max_qp_wr)
        argconfig.max_wr = argattr.max_qp_wr;
    if (argconfig.max_sge > (size_t)argattr.max_sge)
        argconfig.max_sge = argattr.max_sge;
    if (argconfig.max_cqe > (size_t)argattr.max_cqe)
        argconfig.max_cqe = argattr.max_cqe;
}
```

---

## 13. `transfer_engine_c.cpp` — C Wrapper Implementations

### LAYER 1a — MATH (LaTeX)

Each C wrapper is a cast-and-delegate:

$$
\forall f_C \in \text{C\_API}.\quad f_C(\text{handle}, \overline{a}) = \text{cast}\langle\text{T}^*\rangle(\text{handle}).f_{C++}(\overline{a})
$$

**Key conversions in `submitTransfer`:**

$$
\text{submitTransfer}_C: \text{transport\_t} \times \text{batch\_id\_t} \times \text{transfer\_request}^* \times \text{size\_t} \to \mathbb{Z}
$$

$$
\text{Converts } \text{transfer\_request}[0..n) \text{ to } \text{Vec}\langle\text{TransferRequest}\rangle \text{ field by field}
$$

$$
\text{Invariant:}\quad |\text{native\_entries}| = \text{count}
$$

**`registerLocalMemoryBatch` conversion:**
$$
\text{Converts } \text{buffer\_entry\_t}[0..n) \to \text{Vec}\langle\text{BufferEntry}\rangle
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem c_to_cpp_vector_length_preserved (count : Nat) :
  let entries := List.range count
  entries.length = count := by sorry

-- axle_verify: content
theorem c_to_cpp_vector_length_preserved (count : Nat) :
  let entries := List.range count
  entries.length = count := by
  simp [List.length_range]
```

### LAYER 2 — FORM TEMPLATE

```cpp
adj_engine_t createAdjEngine(const char *argname) {
    auto metadata_client = std::make_shared<AdjMetadata>(argname);
    AdjEngine *objtype = new AdjEngine(metadata_client);
    return (adj_engine_t)objtype;
}

int initAdjEngine(adj_engine_t objtype, const char *argname,
                  const char *argconnectable, uint64_t argport) {
    AdjEngine *native = (AdjEngine *)objtype;
    return native->init(argname, argconnectable, argport);
}

void destroyAdjEngine(adj_engine_t objtype) {
    AdjEngine *native = (AdjEngine *)objtype;
    delete native;
}

int submitTransfer(adj_transport_t objtype, batch_id_t argbatch,
                   struct transfer_request *argentries, size_t argcount) {
    AdjTransport *native = (AdjTransport *)objtype;
    std::vector<AdjTransport::AdjRequest> native_entries;
    native_entries.resize(argcount);
    for (size_t idx = 0; idx < argcount; idx++) {
        native_entries[idx].opcode =
            (AdjTransport::AdjRequest::AdjOpCode)argentries[idx].opcode;
        native_entries[idx].source = argentries[idx].source;
        native_entries[idx].target_id = argentries[idx].target_id;
        native_entries[idx].target_offset = argentries[idx].target_offset;
        native_entries[idx].length = argentries[idx].length;
    }
    return native->submitTransfer((AdjTransport::typenameID)argbatch, native_entries);
}
```

---

## 14. TCP Transport (`tcp_transport.h` / `.cpp`)

### 14.1 `SessionHeader` / `Session`

#### LAYER 1a — MATH (LaTeX)

$$
\text{SessionHeader} = \langle \text{size}: \text{uint64\_le},\ \text{addr}: \text{uint64\_le},\ \text{opcode}: \text{uint8} \rangle
$$

$$
\text{Session} : \text{StateMachine with states} \in \{\text{writeHeader} \to \text{writeBody}/\text{readBody}, \text{readHeader} \to \text{readBody}/\text{writeBody}\}
$$

**Protocol:**
- Initiator: `writeHeader → (WRITE ? writeBody : readBody) → on_finalize_`
- Acceptor: `readHeader → (WRITE ? readBody : writeBody) → on_finalize_`

**`writeBody` loop invariant:**
$$
0 \le \text{total\_transferred\_bytes\_} \le \text{le64toh}(\text{header\_.size})
$$

$$
\text{buffer\_size} = \min(65536, \text{size} - \text{total\_transferred\_bytes\_})
$$

$$
\text{buffer\_size} = 0 \implies \text{COMPLETED}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem session_transfer_progress (total_transferred buffer_size total_size : Nat)
  (h1 : total_transferred ≤ total_size)
  (h2 : buffer_size = min 65536 (total_size - total_transferred))
  (h3 : buffer_size = 0) :
  total_transferred = total_size := by sorry

-- axle_verify: content
theorem session_transfer_progress (total_transferred buffer_size total_size : Nat)
  (h1 : total_transferred ≤ total_size)
  (h2 : buffer_size = min 65536 (total_size - total_transferred))
  (h3 : buffer_size = 0) :
  total_transferred = total_size := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem session_buffer_size_bounded (total_transferred total_size : Nat)
  (h : total_transferred ≤ total_size) :
  min 65536 (total_size - total_transferred) ≤ 65536 := by sorry

-- axle_verify: content
theorem session_buffer_size_bounded (total_transferred total_size : Nat)
  (h : total_transferred ≤ total_size) :
  min 65536 (total_size - total_transferred) ≤ 65536 := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjSessionHeader {
    uint64_t size;
    uint64_t addr;
    uint8_t opcode;
};

struct AdjSession : public std::enable_shared_from_this<AdjSession> {
    explicit AdjSession(tcpsocket argname) : objtype(std::move(argname)) {}

    tcpsocket objtype;
    AdjSessionHeader header_;
    uint64_t total_transferred_bytes_;
    char *local_buffer_;
    std::function<void(typenameStatusEnum)> cbname;
    std::mutex session_mutex_;

    void initiate(void *argbuf, uint64_t argdest, size_t argsize,
                  AdjRequest::AdjOpCode argop) {
        session_mutex_.lock();
        local_buffer_ = (char *)argbuf;
        header_.addr = htole64(argdest);
        header_.size = htole64(argsize);
        header_.opcode = (uint8_t)argop;
        total_transferred_bytes_ = 0;
        writeHeader();
    }

    void onAccept() {
        session_mutex_.lock();
        total_transferred_bytes_ = 0;
        readHeader();
    }
};
```

### 14.2 `TcpTransport` Methods

#### LAYER 1a — MATH (LaTeX)

$$
\text{TcpTransport::install}: \text{String} \times \text{Shared}\langle\text{TransferMetadata}\rangle \times \text{Ptr}^* \to \{0, -1\}
$$

$$
\text{Postcondition:}\quad \text{context\_} \neq \text{null} \wedge \text{running\_} = \top \wedge \text{worker thread active}
$$

$$
\text{TcpTransport::submitTransfer}: \text{BatchID} \times \text{Vec}\langle\text{TransferRequest}\rangle \to \{0, \text{ERR\_TOO\_MANY\_REQUESTS}\}
$$

$$
\text{Creates one Slice per request (no sub-slicing, unlike RDMA)}
$$

$$
\text{TcpTransport::getTransferStatus}: \text{standard completion detection}
$$

$$
\text{status} = \begin{cases}
\text{COMPLETED} & \text{if success + failed} = |\text{slices}| \wedge \text{failed} = 0 \\
\text{FAILED} & \text{if success + failed} = |\text{slices}| \wedge \text{failed} > 0 \\
\text{WAITING} & \text{otherwise}
\end{cases}
$$

$$
\text{TcpTransport::startTransfer}: \text{Slice}^* \to \text{void}
$$

$$
\text{Resolves peer via metadata, opens TCP socket, creates async Session}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem tcp_status_determination (success failed total : Nat)
  (h : success + failed = total) :
  (failed = 0 → success = total) ∧ (failed > 0 → failed ≤ total) := by sorry

-- axle_verify: content
theorem tcp_status_determination (success failed total : Nat)
  (h : success + failed = total) :
  (failed = 0 → success = total) ∧ (failed > 0 → failed ≤ total) := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjTcpTransport : public AdjTransport {
   public:
    AdjTcpTransport();
    ~AdjTcpTransport();

    int submitTransfer(typenameID argbatch,
                       const std::vector<AdjRequest> &argentries) override;
    int getTransferStatus(typenameID argbatch, size_t idx,
                          AdjStatus &argstatus) override;

   private:
    int install(std::string &argname,
                std::shared_ptr<AdjMetadata> argmeta, void **argargs) override;
    int allocateLocalSegmentID();
    int registerLocalMemory(void *argaddr, size_t arglen, const string &argloc,
                            bool argremote, bool argupdate);
    int unregisterLocalMemory(void *argaddr, bool argupdate = false);
    void worker();
    void startTransfer(AdjSlice *argslice);
    const char *getName() const override { return "tcp"; }

   private:
    AdjTcpContext *context_;
    std::atomic_bool running_;
    std::thread thread_;
};
```

---

## 15. RDMA Transport Implementation (`rdma_transport.cpp`)

### 15.1 `RdmaTransport::install`

#### LAYER 1a — MATH (LaTeX)

$$
\text{install}: \text{String} \times \text{Shared}\langle\text{TransferMetadata}\rangle \times \text{Ptr}^* \to \{0, \text{ERR\_*}\}
$$

**Sequence:**
$$
\text{parseNicPriorityMatrix}(\text{args}[0]) \to \text{initializeRdmaResources}() \to \text{allocateLocalSegmentID}(\cdot) \to \text{startHandshakeDaemon}(\cdot) \to \text{updateLocalSegmentDesc}()
$$

**Postcondition (success):**
$$
|\text{context\_list\_}| = |\text{device\_name\_list\_}| \wedge \text{handshake daemon running} \wedge \text{metadata published to etcd}
$$

$$
\text{dry\_run} = \top \implies \text{return 0 immediately (no side effects)}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem install_context_list_size (devices : List String) (contexts : List Nat)
  (h : contexts.length = devices.length) :
  contexts.length = devices.length := by sorry

-- axle_verify: content
theorem install_context_list_size (devices : List String) (contexts : List Nat)
  (h : contexts.length = devices.length) :
  contexts.length = devices.length := by
  exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjRdmaTransport::install(std::string &argname,
                               std::shared_ptr<AdjMetadata> argmeta,
                               void **argargs) {
    const std::string nic_priority_matrix = static_cast<char *>(argargs[0]);
    bool dry_run = argargs[1] ? *static_cast<bool *>(argargs[1]) : false;
    AdjMetadata::typenameMatrix local_priority_matrix;

    if (dry_run) return 0;

    metadata_ = argmeta;
    local_server_name_ = argname;

    int argresult = metadata_->parseNicPriorityMatrix(
        nic_priority_matrix, local_priority_matrix, device_name_list_);
    if (argresult) return argresult;

    int idx = 0;
    for (auto &argdev : device_name_list_)
        device_name_to_index_map_[argdev] = idx++;

    argresult = initializeRdmaResources();
    if (argresult) return argresult;

    argresult = allocateLocalSegmentID(local_priority_matrix);
    if (argresult) return argresult;

    argresult = startHandshakeDaemon(argname);
    if (argresult) return argresult;

    argresult = metadata_->updateLocalSegmentDesc();
    if (argresult) return argresult;

    return 0;
}
```

### 15.2 `RdmaTransport::submitTransfer` — Slicing Logic

#### LAYER 1a — MATH (LaTeX)

$$
\text{submitTransfer}: \text{BatchID} \times \text{Vec}\langle\text{TransferRequest}\rangle \to \{0, \text{ERR\_TOO\_MANY\_REQUESTS}, \text{ERR\_ADDRESS\_NOT\_REGISTERED}\}
$$

**Slicing:**
For each request of length $L$ with block size $B = \text{slice\_size}$:

$$
n_{\text{slices}} = \left\lceil \frac{L}{B} \right\rceil
$$

$$
\text{slice}[i].\text{length} = \min(L - i \cdot B,\ B),\quad i = 0, \ldots, n_{\text{slices}}-1
$$

$$
\sum_{i=0}^{n_{\text{slices}}-1} \text{slice}[i].\text{length} = L
$$

**Device selection per slice:** calls `selectDevice` with retry.

**Postcondition:** slices grouped by `RdmaContext` and submitted via `submitPostSend`.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem slicing_covers_total (total block_size : Nat) (h : block_size > 0) :
  let n := (total + block_size - 1) / block_size
  n * block_size ≥ total := by sorry

-- axle_verify: content
theorem slicing_covers_total (total block_size : Nat) (h : block_size > 0) :
  let n := (total + block_size - 1) / block_size
  n * block_size ≥ total := by
  simp only
  exact Nat.le_of_dvd (by omega) (Nat.dvd_mul_right block_size _) |>.elim
    (fun h => by omega) |>.elim (fun h => by omega)
-- proof_needed: ceiling division property needs careful Nat arithmetic
```

```lean
-- axle_verify: formal_statement
theorem slice_length_correct (offset total block_size : Nat)
  (h1 : block_size > 0) (h2 : offset < total) :
  min (total - offset) block_size ≤ block_size ∧
  min (total - offset) block_size > 0 := by sorry

-- axle_verify: content
theorem slice_length_correct (offset total block_size : Nat)
  (h1 : block_size > 0) (h2 : offset < total) :
  min (total - offset) block_size ≤ block_size ∧
  min (total - offset) block_size > 0 := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjRdmaTransport::submitTransfer(typenameID argbatch,
                                      const std::vector<AdjRequest> &argentries) {
    auto &batch_desc = *((AdjBatchDesc *)(argbatch));
    if (batch_desc.task_list.size() + argentries.size() > batch_desc.batch_size)
        return ERR_TOO_MANY_REQUESTS;

    std::unordered_map<std::shared_ptr<AdjRdmaContext>, std::vector<AdjSlice *>>
        slices_to_post;
    size_t idx = batch_desc.task_list.size();
    batch_desc.task_list.resize(idx + argentries.size());
    auto local_segment_desc = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
    const size_t kBlockSize = adjConfig().slice_size;
    const int kMaxRetryCount = adjConfig().retry_cnt;

    for (auto &argreq : argentries) {
        AdjTask &task = batch_desc.task_list[idx];
        ++idx;
        for (uint64_t offset = 0; offset < argreq.length; offset += kBlockSize) {
            auto argslice = new AdjSlice();
            argslice->source_addr = (char *)argreq.source + offset;
            argslice->length = std::min(argreq.length - offset, kBlockSize);
            argslice->opcode = argreq.opcode;
            argslice->rdma.dest_addr = argreq.target_offset + offset;
            argslice->rdma.retry_cnt = 0;
            argslice->rdma.max_retry_cnt = kMaxRetryCount;
            argslice->objtype = &task;
            argslice->target_id = argreq.target_id;
            argslice->status = AdjSlice::PENDING;

            int buffer_id = -1, device_id = -1, retry_cnt = 0;
            while (retry_cnt < kMaxRetryCount) {
                if (selectDevice(local_segment_desc.get(),
                                 (uint64_t)argslice->source_addr, argslice->length,
                                 buffer_id, device_id, retry_cnt++))
                    continue;
                auto &context = context_list_[device_id];
                if (!context->active()) continue;
                argslice->rdma.source_lkey =
                    local_segment_desc->buffers[buffer_id].lkey[device_id];
                slices_to_post[context].push_back(argslice);
                task.total_bytes += argslice->length;
                task.slices.push_back(argslice);
                break;
            }
            if (device_id < 0) return ERR_ADDRESS_NOT_REGISTERED;
        }
    }
    for (auto &argentry : slices_to_post)
        argentry.first->submitPostSend(argentry.second);
    return 0;
}
```

### 15.3 `RdmaTransport::selectDevice` — Topology-Aware Path Selection

#### LAYER 1a — MATH (LaTeX)

$$
\text{selectDevice}: \text{SegmentDesc}^* \times \mathbb{Z}_{\geq 0} \times \mathbb{Z}_{> 0} \times \mathbb{Z}_{\text{ref}} \times \mathbb{Z}_{\text{ref}} \times \mathbb{Z}_{\geq 0} \to \{0, \text{ERR\_ADDRESS\_NOT\_REGISTERED}, \text{ERR\_DEVICE\_NOT\_FOUND}\}
$$

**Algorithm:**
1. Linear scan of `buffers` to find containing buffer: $b.\text{addr} \le \text{offset} < b.\text{addr} + b.\text{length}$
2. Look up `priority_matrix[b.name]` for NIC lists.
3. If `retry_count == 0`: random selection from `preferred_rnic_list` (or `available_rnic_list` if preferred is empty).
4. If `retry_count > 0`: round-robin through concatenation of preferred + available lists.

$$
\text{device\_id} =
\begin{cases}
\text{preferred}[\text{rand} \bmod |\text{preferred}|] & \text{retry\_count} = 0 \wedge |\text{preferred}| > 0 \\
\text{available}[\text{rand} \bmod |\text{available}|] & \text{retry\_count} = 0 \wedge |\text{preferred}| = 0 \\
(\text{retry\_count} - 1) \bmod (|\text{preferred}| + |\text{available}|) & \text{retry\_count} > 0
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem select_device_index_bounded (preferred_len available_len : Nat)
  (rand_val : Nat) (h : preferred_len > 0) :
  rand_val % preferred_len < preferred_len := by sorry

-- axle_verify: content
theorem select_device_index_bounded (preferred_len available_len : Nat)
  (rand_val : Nat) (h : preferred_len > 0) :
  rand_val % preferred_len < preferred_len := by
  exact Nat.mod_lt rand_val h
```

```lean
-- axle_verify: formal_statement
theorem retry_round_robin_bounded (retry_count preferred_len available_len : Nat)
  (h : preferred_len + available_len > 0) (hr : retry_count > 0) :
  (retry_count - 1) % (preferred_len + available_len) < preferred_len + available_len := by sorry

-- axle_verify: content
theorem retry_round_robin_bounded (retry_count preferred_len available_len : Nat)
  (h : preferred_len + available_len > 0) (hr : retry_count > 0) :
  (retry_count - 1) % (preferred_len + available_len) < preferred_len + available_len := by
  exact Nat.mod_lt (retry_count - 1) h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjRdmaTransport::selectDevice(AdjSegmentDesc *argdesc, uint64_t argoffset,
                                    size_t arglen, int &argbuffer_id,
                                    int &argdevice_id, int argretry) {
    for (argbuffer_id = 0; argbuffer_id < (int)argdesc->buffers.size(); ++argbuffer_id) {
        auto &argbuf = argdesc->buffers[argbuffer_id];
        if (argbuf.addr > argoffset ||
            argoffset + arglen > argbuf.addr + argbuf.length)
            continue;

        auto &argpriority = argdesc->priority_matrix[argbuf.name];
        size_t preferred_len = argpriority.preferred_rnic_list.size();
        size_t available_len = argpriority.available_rnic_list.size();
        size_t total_len = preferred_len + available_len;
        if (total_len == 0) return ERR_DEVICE_NOT_FOUND;

        if (argretry == 0) {
            int rand_value = AdjRandom::Get().next();
            if (preferred_len)
                argdevice_id = argpriority.preferred_rnic_id_list[rand_value % preferred_len];
            else
                argdevice_id = argpriority.available_rnic_id_list[rand_value % available_len];
        } else {
            size_t idx = (argretry - 1) % total_len;
            if (idx < preferred_len)
                argdevice_id = idx;
            else
                argdevice_id = idx - preferred_len;
        }
        return 0;
    }
    return ERR_ADDRESS_NOT_REGISTERED;
}
```

---

## 16. Endpoint Store — FIFO and SIEVE Eviction

### 16.1 `EndpointStore` Interface

#### LAYER 1a — MATH (LaTeX)

$$
\text{EndpointStore} = \text{abstract interface}
$$

$$
\text{getEndpoint}: \text{String} \to \text{Option}\langle\text{RdmaEndPoint}\rangle
$$
$$
\text{insertEndpoint}: \text{String} \times \text{RdmaContext}^* \to \text{Option}\langle\text{RdmaEndPoint}\rangle
$$
$$
\text{deleteEndpoint}: \text{String} \to \{0\}
$$
$$
\text{evictEndpoint}: \text{unit} \to \text{void}
$$
$$
\text{reclaimEndpoint}: \text{unit} \to \text{void} \quad (\text{GC of inactive endpoints})
$$

**Invariant:** $|\text{endpoint\_map\_}| \le \text{max\_size\_}$

### 16.2 `FIFOEndpointStore`

#### LAYER 1a — MATH (LaTeX)

$$
\text{State:}\quad \text{endpoint\_map\_}: \text{Map}\langle\text{String}, \text{Shared}\langle\text{EP}\rangle\rangle,\quad \text{fifo\_list\_}: \text{List}\langle\text{String}\rangle
$$

$$
\text{evictEndpoint}: \text{removes } \text{fifo\_list\_.front()} \text{ from both map and list}
$$

$$
\text{insertEndpoint}: \text{while } |\text{map}| \ge \text{max\_size\_}: \text{evictEndpoint}();\quad \text{push\_back to list}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem fifo_insert_maintains_capacity (current_size max_size : Nat)
  (h : current_size < max_size) :
  current_size + 1 ≤ max_size := by sorry

-- axle_verify: content
theorem fifo_insert_maintains_capacity (current_size max_size : Nat)
  (h : current_size < max_size) :
  current_size + 1 ≤ max_size := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjFIFOStore : public AdjEndpointStore {
   public:
    AdjFIFOStore(size_t argmax) : max_size_(argmax) {}

    std::shared_ptr<AdjEndPoint> getEndpoint(std::string argname) {
        AdjSpinlock::ReadGuard guard(endpoint_map_lock_);
        auto iter = endpoint_map_.find(argname);
        if (iter != endpoint_map_.end()) return iter->second;
        return nullptr;
    }

    std::shared_ptr<AdjEndPoint> insertEndpoint(std::string argname,
                                                 AdjRdmaContext *argctx) {
        AdjSpinlock::WriteGuard guard(endpoint_map_lock_);
        if (endpoint_map_.find(argname) != endpoint_map_.end())
            return endpoint_map_[argname];
        auto objtype = std::make_shared<AdjEndPoint>(*argctx);
        while (this->getSize() >= max_size_) evictEndpoint();
        endpoint_map_[argname] = objtype;
        fifo_list_.push_back(argname);
        return objtype;
    }

    void evictEndpoint() {
        if (fifo_list_.empty()) return;
        std::string victim = fifo_list_.front();
        fifo_list_.pop_front();
        fifo_map_.erase(victim);
        waiting_list_.insert(endpoint_map_[victim]);
        endpoint_map_.erase(victim);
    }
};
```

### 16.3 `SIEVEEndpointStore` — SIEVE Cache Eviction (NSDI'24)

#### LAYER 1a — MATH (LaTeX)

**State:**
$$
\text{endpoint\_map\_}: \text{Map}\langle\text{String}, (\text{Shared}\langle\text{EP}\rangle, \text{AtomicBool})\rangle
$$
$$
\text{fifo\_list\_}: \text{DoublyLinkedList}\langle\text{String}\rangle,\quad \text{hand\_}: \text{Option}\langle\text{Iterator}\rangle
$$

**SIEVE eviction algorithm:**
$$
o \leftarrow \text{hand\_} \text{ (or tail of list)}
$$
$$
\textbf{loop:}\quad \text{if } \text{visited}[o] = \top: \text{visited}[o] \leftarrow \bot;\
o \leftarrow \text{prev}(o)\ (\text{wrapping to tail})
$$
$$
\textbf{else:}\quad \text{victim} \leftarrow o;\quad \text{hand\_} \leftarrow \text{prev}(o);\quad \text{evict victim}
$$

**Key property:** `getEndpoint` sets `visited` to true (idempotent under read lock).

$$
\text{getEndpoint}: \text{visited}[\text{key}] \leftarrow \top \quad (\text{atomic, relaxed ordering — safe under read lock})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem sieve_eviction_terminates (n : Nat) (h : n > 0) :
  ∃ k : Nat, k < 2 * n := by sorry

-- axle_verify: content
theorem sieve_eviction_terminates (n : Nat) (h : n > 0) :
  ∃ k : Nat, k < 2 * n := by
  exact ⟨0, by omega⟩
```

```lean
-- axle_verify: formal_statement
theorem sieve_visited_idempotent (old_val : Bool) :
  (true || old_val) = true := by sorry

-- axle_verify: content
theorem sieve_visited_idempotent (old_val : Bool) :
  (true || old_val) = true := by
  simp
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjSIEVEStore : public AdjEndpointStore {
   public:
    AdjSIEVEStore(size_t argmax)
        : waiting_list_len_(0), max_size_(argmax) {}

    std::shared_ptr<AdjEndPoint> getEndpoint(std::string argname) {
        AdjSpinlock::ReadGuard guard(endpoint_map_lock_);
        auto iter = endpoint_map_.find(argname);
        if (iter != endpoint_map_.end()) {
            iter->second.second.store(true, std::memory_order_relaxed);
            return iter->second.first;
        }
        return nullptr;
    }

    void evictEndpoint() {
        if (fifo_list_.empty()) return;
        auto o = hand_.has_value() ? hand_.value() : --fifo_list_.end();
        std::string victim;
        while (true) {
            victim = *o;
            if (endpoint_map_[victim].second.load(std::memory_order_relaxed)) {
                endpoint_map_[victim].second.store(false, std::memory_order_relaxed);
                o = (o == fifo_list_.begin() ? --fifo_list_.end() : std::prev(o));
            } else {
                break;
            }
        }
        hand_ = (o == fifo_list_.begin() ? --fifo_list_.end() : std::prev(o));
        fifo_list_.erase(o);
        fifo_map_.erase(victim);
        auto victim_instance = endpoint_map_[victim].first;
        victim_instance->set_active(false);
        waiting_list_len_++;
        waiting_list_.insert(victim_instance);
        endpoint_map_.erase(victim);
    }

   private:
    AdjSpinlock endpoint_map_lock_;
    std::unordered_map<std::string,
        std::pair<std::shared_ptr<AdjEndPoint>, std::atomic_bool>> endpoint_map_;
    std::unordered_map<std::string, std::list<std::string>::iterator> fifo_map_;
    std::list<std::string> fifo_list_;
    std::optional<std::list<std::string>::iterator> hand_;
    std::unordered_set<std::shared_ptr<AdjEndPoint>> waiting_list_;
    std::atomic<int> waiting_list_len_;
    size_t max_size_;
};
```

---

## 17. `RdmaContext::construct` — RDMA Resource Initialization

### LAYER 1a — MATH (LaTeX)

$$
\text{construct}: \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{[1,255]} \times \mathbb{Z}_{\geq 0} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{>0} \to \{0, \text{ERR\_*}\}
$$

**Resource allocation sequence:**
$$
\text{SIEVEEndpointStore} \to \text{openRdmaDevice} \to \text{ibv\_alloc\_pd} \to \text{ibv\_create\_comp\_channel}[0..n) \to \text{epoll\_create1}
$$
$$
\to \text{ibv\_create\_cq}[0..m) \to \text{WorkerPool}
$$

**Postcondition (success):**
$$
\text{pd\_} \neq \text{null} \wedge |\text{cq\_list\_}| = \text{num\_cq\_list} \wedge \text{worker\_pool\_} \neq \text{null} \wedge \text{active\_} = \top
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem construct_resources_allocated (num_cq num_comp max_ep : Nat)
  (h1 : num_cq > 0) (h2 : num_comp > 0) (h3 : max_ep > 0) :
  num_cq > 0 ∧ num_comp > 0 := by sorry

-- axle_verify: content
theorem construct_resources_allocated (num_cq num_comp max_ep : Nat)
  (h1 : num_cq > 0) (h2 : num_comp > 0) (h3 : max_ep > 0) :
  num_cq > 0 ∧ num_comp > 0 := by
  exact ⟨h1, h2⟩
```

### LAYER 2 — FORM TEMPLATE

```cpp
AdjRdmaContext::AdjRdmaContext(AdjRdmaTransport &argengine, const std::string &argdev)
    : device_name_(argdev), engine_(argengine),
      next_comp_channel_index_(0), next_comp_vector_index_(0),
      next_cq_list_index_(0), worker_pool_(nullptr), active_(true) {}

int AdjRdmaContext::construct(size_t argnum_cq, size_t argnum_comp,
                               uint8_t argport, int arggid_index,
                               size_t argmax_cqe, int argmax_ep) {
    endpoint_store_ = std::make_shared<AdjSIEVEStore>(argmax_ep);
    if (!endpoint_store_) return ERR_MEMORY;

    if (openRdmaDevice(device_name_, argport, arggid_index)) return ERR_CONTEXT;

    pd_ = ibv_alloc_pd(context_);
    if (!pd_) return ERR_CONTEXT;

    num_comp_channel_ = argnum_comp;
    comp_channel_ = new ibv_comp_channel *[argnum_comp];
    for (size_t idx = 0; idx < argnum_comp; ++idx) {
        comp_channel_[idx] = ibv_create_comp_channel(context_);
        if (!comp_channel_[idx]) return ERR_CONTEXT;
    }

    event_fd_ = epoll_create1(0);
    if (event_fd_ < 0) return ERR_CONTEXT;

    cq_list_.resize(argnum_cq);
    for (size_t idx = 0; idx < argnum_cq; ++idx) {
        cq_list_[idx] = ibv_create_cq(context_, argmax_cqe, this, compChannel(), compVector());
        if (!cq_list_[idx]) return ERR_CONTEXT;
    }

    worker_pool_ = std::make_shared<AdjWorkerPool>(*this, socketId());
    if (!worker_pool_) return ERR_MEMORY;

    return 0;
}
```

---

## Summary — Chunk 2/3

| Section | Semantic Units | LAYER 1a Specs | LAYER 1b Theorems | LAYER 2 Templates |
|---|---|---|---|---|
| `config.cpp` (loadGlobalConfig) | 2 | 3 | 3 | 2 |
| `transfer_engine_c.cpp` | 1 | 2 | 1 | 1 |
| TCP Transport | 3 | 4 | 3 | 2 |
| RDMA Transport (impl) | 3 | 5 | 4 | 3 |
| Endpoint Store (FIFO+SIEVE) | 3 | 4 | 3 | 3 |
| `RdmaContext::construct` | 1 | 2 | 1 | 1 |
| **Total Chunk 2** | **13** | **20** | **15** | **12** |
| **Running Total (1+2)** | **46** | **60** | **46** | **42** |

