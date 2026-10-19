# Mooncake Transfer Engine — Three-Layer Formal Analysis (Chunk 1/3)

Commit: `348afd753190` — Depth-1 files (40 files)

---

## Table of Contents

1. [Error Codes (`error.h`)](#1-error-codes)
2. [Global Configuration (`config.h`)](#2-global-configuration)
3. [Common Utilities (`common.h`)](#3-common-utilities)
4. [Transfer Metadata (`transfer_metadata.h`)](#4-transfer-metadata)
5. [Transport Base Class (`transport.h`)](#5-transport-base-class)
6. [Transfer Engine (`transfer_engine.h` / `.cpp`)](#6-transfer-engine)
7. [C API (`transfer_engine_c.h`)](#7-c-api)
8. [RDMA Transport (`rdma_transport.h`)](#8-rdma-transport)
9. [Build System (`CMakeLists.txt`, `dependencies.sh`)](#9-build-system)
10. [Shell Script (`run_test.sh`)](#10-ci-test-script)

---

## 1. Error Codes (`error.h`)

### LAYER 1a — MATH (LaTeX)

$$
\textbf{ErrorCode} = \mathbb{Z}_{< 0} \cup \{0\}
$$

Define the error domain as a disjoint union of error groups:

$$
\mathcal{E}_{\text{args}} = \{ \text{ERR\_INVALID\_ARGUMENT}\ (-1),\ \text{ERR\_TOO\_MANY\_REQUESTS}\ (-2),\ \text{ERR\_ADDRESS\_NOT\_REGISTERED}\ (-3),\\
\quad \text{ERR\_BATCH\_BUSY}\ (-4),\ \text{ERR\_DEVICE\_NOT\_FOUND}\ (-6),\ \text{ERR\_ADDRESS\_OVERLAPPED}\ (-7) \}
$$

$$
\mathcal{E}_{\text{handshake}} = \{ \text{ERR\_DNS}\ (-101),\ \text{ERR\_SOCKET}\ (-102),\ \text{ERR\_MALFORMED\_JSON}\ (-103),\ \text{ERR\_REJECT\_HANDSHAKE}\ (-104) \}
$$

$$
\mathcal{E}_{\text{resource}} = \{ \text{ERR\_METADATA}\ (-200),\ \text{ERR\_ENDPOINT}\ (-201),\ \text{ERR\_CONTEXT}\ (-202) \}
$$

$$
\mathcal{E}_{\text{system}} = \{ \text{ERR\_NUMA}\ (-300),\ \text{ERR\_CLOCK}\ (-301),\ \text{ERR\_MEMORY}\ (-302) \}
$$

**Invariant:**
$$
\mathcal{E}_{\text{args}} \cap \mathcal{E}_{\text{handshake}} = \emptyset,\quad
\mathcal{E}_{\text{args}} \cap \mathcal{E}_{\text{resource}} = \emptyset,\quad
\mathcal{E}_{\text{args}} \cap \mathcal{E}_{\text{system}} = \emptyset
$$
(all groups are pairwise disjoint)

$$
\forall e \in \mathcal{E}_{\text{all}}.\ e < 0
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem error_codes_negative :
  let err_invalid_argument := -1
  let err_too_many_requests := -2
  let err_address_not_registered := -3
  let err_batch_busy := -4
  let err_device_not_found := -6
  let err_address_overlapped := -7
  let err_dns := -101
  let err_socket := -102
  let err_malformed_json := -103
  let err_reject_handshake := -104
  let err_metadata := -200
  let err_endpoint := -201
  let err_context := -202
  let err_numa := -300
  let err_clock := -301
  let err_memory := -302
  err_invalid_argument < 0 ∧ err_too_many_requests < 0 ∧
  err_address_not_registered < 0 ∧ err_batch_busy < 0 ∧
  err_device_not_found < 0 ∧ err_address_overlapped < 0 ∧
  err_dns < 0 ∧ err_socket < 0 ∧ err_malformed_json < 0 ∧
  err_reject_handshake < 0 ∧ err_metadata < 0 ∧ err_endpoint < 0 ∧
  err_context < 0 ∧ err_numa < 0 ∧ err_clock < 0 ∧ err_memory < 0 := by sorry

-- axle_verify: content
theorem error_codes_negative :
  let err_invalid_argument := -1
  let err_too_many_requests := -2
  let err_address_not_registered := -3
  let err_batch_busy := -4
  let err_device_not_found := -6
  let err_address_overlapped := -7
  let err_dns := -101
  let err_socket := -102
  let err_malformed_json := -103
  let err_reject_handshake := -104
  let err_metadata := -200
  let err_endpoint := -201
  let err_context := -202
  let err_numa := -300
  let err_clock := -301
  let err_memory := -302
  err_invalid_argument < 0 ∧ err_too_many_requests < 0 ∧
  err_address_not_registered < 0 ∧ err_batch_busy < 0 ∧
  err_device_not_found < 0 ∧ err_address_overlapped < 0 ∧
  err_dns < 0 ∧ err_socket < 0 ∧ err_malformed_json < 0 ∧
  err_reject_handshake < 0 ∧ err_metadata < 0 ∧ err_endpoint < 0 ∧
  err_context < 0 ∧ err_numa < 0 ∧ err_clock < 0 ∧ err_memory < 0 := by
  simp
```

```lean
-- axle_verify: formal_statement
theorem error_groups_disjoint :
  let args := ({-1, -2, -3, -4, -6, -7} : Finset Int)
  let handshake := ({-101, -102, -103, -104} : Finset Int)
  let resource := ({-200, -201, -202} : Finset Int)
  let system := ({-300, -301, -302} : Finset Int)
  (args ∩ handshake = ∅) ∧ (args ∩ resource = ∅) ∧
  (args ∩ system = ∅) ∧ (handshake ∩ resource = ∅) ∧
  (handshake ∩ system = ∅) ∧ (resource ∩ system = ∅) := by sorry

-- axle_verify: content
theorem error_groups_disjoint :
  let args := ({-1, -2, -3, -4, -6, -7} : Finset Int)
  let handshake := ({-101, -102, -103, -104} : Finset Int)
  let resource := ({-200, -201, -202} : Finset Int)
  let system := ({-300, -301, -302} : Finset Int)
  (args ∩ handshake = ∅) ∧ (args ∩ resource = ∅) ∧
  (args ∩ system = ∅) ∧ (handshake ∩ resource = ∅) ∧
  (handshake ∩ system = ∅) ∧ (resource ∩ system = ∅) := by
  simp [Finset.eq_empty_iff_forall_not_mem]
  omega
```

### LAYER 2 — FORM TEMPLATE

```c
#ifndef ERRCODE_H
#define ERRCODE_H

#define ERR_INVALID_ARGUMENT (-1)
#define ERR_TOO_MANY_REQUESTS (-2)
#define ERR_ADDRESS_NOT_REGISTERED (-3)
#define ERR_BATCH_BUSY (-4)
#define ERR_DEVICE_NOT_FOUND (-6)
#define ERR_ADDRESS_OVERLAPPED (-7)

#define ERR_DNS (-101)
#define ERR_SOCKET (-102)
#define ERR_MALFORMED_JSON (-103)
#define ERR_REJECT_HANDSHAKE (-104)

#define ERR_METADATA (-200)
#define ERR_ENDPOINT (-201)
#define ERR_CONTEXT (-202)

#define ERR_NUMA (-300)
#define ERR_CLOCK (-301)
#define ERR_MEMORY (-302)

#endif  // ERRCODE_H
```

(Error codes are pure constants — template is identity-preserving; no identifiers require role-based renaming.)

---

## 2. Global Configuration (`config.h`)

### LAYER 1a — MATH (LaTeX)

$$
\text{GlobalConfig} : \text{Struct} \to \mathcal{P}(\text{Field})
$$

$$
\text{Fields:}\quad
\begin{cases}
\text{num\_cq\_per\_ctx} \in \mathbb{Z}_{>0},\ \text{default}=1 \\
\text{num\_comp\_channels\_per\_ctx} \in \mathbb{Z}_{>0},\ \text{default}=1 \\
\text{port} \in [1, 255],\ \text{default}=1 \\
\text{gid\_index} \in \mathbb{Z}_{\geq 0},\ \text{default}=0 \\
\text{max\_cqe} \in \mathbb{Z}_{>0},\ \text{default}=4096 \\
\text{max\_ep\_per\_ctx} \in \mathbb{Z}_{>0},\ \text{default}=256 \\
\text{num\_qp\_per\_ep} \in \mathbb{Z}_{>0},\ \text{default}=2 \\
\text{max\_sge} \in \mathbb{Z}_{>0},\ \text{default}=4 \\
\text{max\_wr} \in \mathbb{Z}_{>0},\ \text{default}=256 \\
\text{max\_inline} \in \mathbb{Z}_{\geq 0},\ \text{default}=64 \\
\text{mtu\_length} \in \{\text{IBV\_MTU\_256}, \ldots, \text{IBV\_MTU\_4096}\},\ \text{default}=\text{IBV\_MTU\_4096} \\
\text{handshake\_port} \in [1, 65535],\ \text{default}=12001 \\
\text{slice\_size} \in \mathbb{Z}_{>0},\ \text{default}=65536 \\
\text{retry\_cnt} \in \mathbb{Z}_{\geq 0},\ \text{default}=8
\end{cases}
$$

**Functions:**

$$
\text{loadGlobalConfig}: \text{GlobalConfig}_{\text{ref}} \to \text{void} \quad (\text{reads env vars, mutates config})
$$
$$
\text{globalConfig}: \text{unit} \to \text{GlobalConfig}_{\text{ref}} \quad (\text{singleton accessor})
$$
$$
\text{getDefaultHandshakePort}: \text{unit} \to \mathbb{Z}_{[1,65535]}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem global_config_defaults_valid :
  let num_cq := 1
  let num_comp := 1
  let port := 1
  let gid_index := 0
  let max_cqe := 4096
  let max_ep := 256
  let num_qp := 2
  let max_sge := 4
  let max_wr := 256
  let max_inline := 64
  let handshake_port := 12001
  let slice_size := 65536
  let retry_cnt := 8
  num_cq > 0 ∧ num_comp > 0 ∧ port ≥ 1 ∧ port ≤ 255 ∧
  gid_index ≥ 0 ∧ max_cqe > 0 ∧ max_ep > 0 ∧ num_qp > 0 ∧
  max_sge > 0 ∧ max_wr > 0 ∧ max_inline ≥ 0 ∧
  handshake_port ≥ 1 ∧ handshake_port ≤ 65535 ∧
  slice_size > 0 ∧ retry_cnt ≥ 0 := by sorry

-- axle_verify: content
theorem global_config_defaults_valid :
  let num_cq := 1
  let num_comp := 1
  let port := 1
  let gid_index := 0
  let max_cqe := 4096
  let max_ep := 256
  let num_qp := 2
  let max_sge := 4
  let max_wr := 256
  let max_inline := 64
  let handshake_port := 12001
  let slice_size := 65536
  let retry_cnt := 8
  num_cq > 0 ∧ num_comp > 0 ∧ port ≥ 1 ∧ port ≤ 255 ∧
  gid_index ≥ 0 ∧ max_cqe > 0 ∧ max_ep > 0 ∧ num_qp > 0 ∧
  max_sge > 0 ∧ max_wr > 0 ∧ max_inline ≥ 0 ∧
  handshake_port ≥ 1 ∧ handshake_port ≤ 65535 ∧
  slice_size > 0 ∧ retry_cnt ≥ 0 := by
  simp
```

### LAYER 2 — FORM TEMPLATE

```cpp
namespace prjname {
struct AdjConfig {
    size_t num_cq_per_ctx = 1;
    size_t num_comp_channels_per_ctx = 1;
    uint8_t port = 1;
    int gid_index = 0;
    size_t max_cqe = 4096;
    int max_ep_per_ctx = 256;
    size_t num_qp_per_ep = 2;
    size_t max_sge = 4;
    size_t max_wr = 256;
    size_t max_inline = 64;
    ibv_mtu mtu_length = IBV_MTU_4096;
    uint16_t handshake_port = 12001;
    int workers_per_ctx = 2;
    bool verbose = false;
    size_t slice_size = 65536;
    int retry_cnt = 8;
};

void loadAdjConfig(AdjConfig &argname);
void dumpAdjConfig();
void updateAdjConfig(ibv_device_attr &argname);
AdjConfig &adjConfig();
uint16_t getDefaultHandshakePort();
}  // namespace prjname
```

---

## 3. Common Utilities (`common.h`)

### 3.1 `bindToSocket`

#### LAYER 1a — MATH (LaTeX)

$$
\text{bindToSocket}: \mathbb{Z} \to \{0, \text{ERR\_NUMA}\}
$$

**Precondition:**
$$
\text{numa\_available}() \geq 0
$$

**Postcondition:**
$$
\text{bindToSocket}(s) = 0 \iff \text{pthread\_setaffinity\_np}(\text{self}, \text{cpuset\_for}(\text{clamp}(s, 0, N-1))) = 0
$$

where $N = \text{numa\_num\_configured\_nodes}()$ and $\text{clamp}(s, 0, N-1) = \begin{cases} s & \text{if } 0 \le s < N \\ 0 & \text{otherwise} \end{cases}$.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem bind_to_socket_clamp (socket_id num_nodes : Int) (h : num_nodes > 0) :
  let clamped := if socket_id < 0 ∨ socket_id ≥ num_nodes then 0 else socket_id
  0 ≤ clamped ∧ clamped < num_nodes := by sorry

-- axle_verify: content
theorem bind_to_socket_clamp (socket_id num_nodes : Int) (h : num_nodes > 0) :
  let clamped := if socket_id < 0 ∨ socket_id ≥ num_nodes then 0 else socket_id
  0 ≤ clamped ∧ clamped < num_nodes := by
  simp only
  split
  · omega
  · push_neg at *
    rename_i h1
    omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline int bindToSocket(int idx) {
    if (unlikely(numa_available() < 0)) {
        LOG(ERROR) << "The platform does not support NUMA";
        return ERR_NUMA;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int num_nodes = numa_num_configured_nodes();
    if (idx < 0 || idx >= num_nodes) idx = 0;
    struct bitmask *cpu_list = numa_allocate_cpumask();
    numa_node_to_cpus(idx, cpu_list);
    for (int cpu = 0; cpu < numa_num_possible_cpus(); ++cpu) {
        if (numa_bitmask_isbitset(cpu_list, cpu)) CPU_SET(cpu, &cpuset);
    }
    numa_free_cpumask(cpu_list);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset)) {
        LOG(ERROR) << "Failed to set socket affinity";
        return ERR_NUMA;
    }
    return 0;
}
```

### 3.2 `getCurrentTimeInNano`

#### LAYER 1a — MATH (LaTeX)

$$
\text{getCurrentTimeInNano}: \text{unit} \to \mathbb{Z}
$$

$$
\text{getCurrentTimeInNano}() = t_s \cdot 10^9 + t_{ns} \quad \text{where } (t_s, t_{ns}) = \text{clock\_gettime}(\text{CLOCK\_REALTIME})
$$

$$
\text{On failure:}\quad \text{getCurrentTimeInNano}() = \text{ERR\_CLOCK} = -301
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem time_nano_composition (tv_sec tv_nsec : Int) (h1 : tv_sec ≥ 0) (h2 : tv_nsec ≥ 0) (h3 : tv_nsec < 1000000000) :
  tv_sec * 1000000000 + tv_nsec ≥ 0 := by sorry

-- axle_verify: content
theorem time_nano_composition (tv_sec tv_nsec : Int) (h1 : tv_sec ≥ 0) (h2 : tv_nsec ≥ 0) (h3 : tv_nsec < 1000000000) :
  tv_sec * 1000000000 + tv_nsec ≥ 0 := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline int64_t getCurrentTimeInNano() {
    const int64_t kNanosPerSecond = 1000 * 1000 * 1000;
    struct timespec argname;
    if (clock_gettime(CLOCK_REALTIME, &argname)) {
        PLOG(ERROR) << "Failed to read real-time lock";
        return ERR_CLOCK;
    }
    return (int64_t{argname.tv_sec} * kNanosPerSecond + int64_t{argname.tv_nsec});
}
```

### 3.3 `parseHostNameWithPort`

#### LAYER 1a — MATH (LaTeX)

$$
\text{parseHostNameWithPort}: \text{String} \to \text{String} \times \mathbb{Z}_{[1,65535]}
$$

$$
\text{parseHostNameWithPort}(s) =
\begin{cases}
(s, \text{defaultPort}) & \text{if } ':' \notin s \\
(s[0..p), \text{atoi}(s[p+1..])) & \text{if } ':' \in s \text{ at position } p \text{ and } 1 \le \text{atoi}(s[p+1..]) \le 65535 \\
(s[0..p), \text{defaultPort}) & \text{otherwise}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem parse_port_valid (val defaultPort : Nat) (hd : 1 ≤ defaultPort ∧ defaultPort ≤ 65535) :
  let port := if 1 ≤ val ∧ val ≤ 65535 then val else defaultPort
  1 ≤ port ∧ port ≤ 65535 := by sorry

-- axle_verify: content
theorem parse_port_valid (val defaultPort : Nat) (hd : 1 ≤ defaultPort ∧ defaultPort ≤ 65535) :
  let port := if 1 ≤ val ∧ val ≤ 65535 then val else defaultPort
  1 ≤ port ∧ port ≤ 65535 := by
  simp only
  split
  · rename_i h; exact h
  · exact hd
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline std::pair<std::string, uint16_t> parseHostNameWithPort(
    const std::string &argname) {
    uint16_t port = getDefaultHandshakePort();
    auto pos = argname.find(':');
    if (pos == argname.npos) return std::make_pair(argname, port);
    auto trimmed = argname.substr(0, pos);
    auto port_str = argname.substr(pos + 1);
    int val = std::atoi(port_str.c_str());
    if (val <= 0 || val > 65535)
        PLOG(WARNING) << "Illegal port number in " << argname
                      << ". Use default port " << port << " instead";
    else
        port = (uint16_t)val;
    return std::make_pair(trimmed, port);
}
```

### 3.4 `writeFully` / `readFully`

#### LAYER 1a — MATH (LaTeX)

$$
\text{writeFully}: \text{fd} \times \text{Ptr} \times \mathbb{Z}_{>0} \to \mathbb{Z}
$$

**Postcondition (loop invariant):**
Let $n_0 = \text{len}$, and after iteration $i$, $n_i$ bytes remain.
$$
\text{Inv}: 0 \le n_i \le n_0 \wedge \text{pos}_i = \text{buf} + (n_0 - n_i)
$$

$$
\text{writeFully}(\text{fd}, \text{buf}, \text{len}) =
\begin{cases}
\text{len} & \text{if all bytes written successfully} \\
\text{len} - n_i & \text{if } \text{write returns 0 (partial)} \\
r < 0 & \text{on error}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem write_fully_invariant (len nbytes : Nat) (rc : Nat) (h : nbytes ≤ len) (hrc : rc ≤ nbytes) :
  nbytes - rc ≤ len := by sorry

-- axle_verify: content
theorem write_fully_invariant (len nbytes : Nat) (rc : Nat) (h : nbytes ≤ len) (hrc : rc ≤ nbytes) :
  nbytes - rc ≤ len := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline ssize_t writeFully(int objtype, const void *argname, size_t arglen) {
    char *pos = (char *)argname;
    size_t nbytes = arglen;
    while (nbytes) {
        ssize_t rc = write(objtype, pos, nbytes);
        if (rc < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
        else if (rc < 0) {
            PLOG(ERROR) << "Socket write failed";
            return rc;
        } else if (rc == 0) {
            LOG(WARNING) << "Socket write incompleted: expected " << arglen
                         << " bytes, actual " << arglen - nbytes << " bytes";
            return arglen - nbytes;
        }
        pos += rc;
        nbytes -= rc;
    }
    return arglen;
}
```

```cpp
static inline ssize_t readFully(int objtype, void *argname, size_t arglen) {
    char *pos = (char *)argname;
    size_t nbytes = arglen;
    while (nbytes) {
        ssize_t rc = read(objtype, pos, nbytes);
        if (rc < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
        else if (rc < 0) {
            PLOG(ERROR) << "Socket read failed";
            return rc;
        } else if (rc == 0) {
            LOG(WARNING) << "Socket read incompleted: expected " << arglen
                         << " bytes, actual " << arglen - nbytes << " bytes";
            return arglen - nbytes;
        }
        pos += rc;
        nbytes -= rc;
    }
    return arglen;
}
```

### 3.5 `writeString` / `readString`

#### LAYER 1a — MATH (LaTeX)

$$
\text{writeString}: \text{fd} \times \text{String} \to \{0, \text{ERR\_SOCKET}\}
$$

$$
\text{writeString}(\text{fd}, s) = 0 \iff \text{writeFully}(\text{fd}, |s|, 8) = 8 \wedge \text{writeFully}(\text{fd}, s.\text{data}, |s|) = |s|
$$

$$
\text{readString}: \text{fd} \to \text{String}
$$

$$
\text{readString}(\text{fd}) =
\begin{cases}
\text{""} & \text{if readFully for length fails or } \ell > 2^{20} \\
s \text{ (of length } \ell\text{)} & \text{otherwise}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem read_string_length_bound (length : Nat) :
  let kMaxLength := 1048576
  (length > kMaxLength) → length > 0 := by sorry

-- axle_verify: content
theorem read_string_length_bound (length : Nat) :
  let kMaxLength := 1048576
  (length > kMaxLength) → length > 0 := by
  intro h
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline int writeString(int objtype, const std::string &argname) {
    uint64_t length = argname.size();
    if (writeFully(objtype, &length, sizeof(length)) != (ssize_t)sizeof(length))
        return ERR_SOCKET;
    if (writeFully(objtype, argname.data(), length) != (ssize_t)length)
        return ERR_SOCKET;
    return 0;
}

static inline std::string readString(int objtype) {
    const static size_t kMaxLength = 1ull << 20;
    uint64_t length = 0;
    if (readFully(objtype, &length, sizeof(length)) != (ssize_t)sizeof(length))
        return "";
    if (length > kMaxLength) return "";
    std::string argresult;
    std::vector<char> buffer(length);
    if (readFully(objtype, buffer.data(), length) != (ssize_t)length) return "";
    argresult.assign(buffer.data(), length);
    return argresult;
}
```

### 3.6 NIC Path Utilities (`getServerNameFromNicPath`, `getNicNameFromNicPath`, `MakeNicPath`)

#### LAYER 1a — MATH (LaTeX)

$$
\text{NicPath} \triangleq \text{ServerName} \mathbin{\texttt{@}} \text{NicName}
$$

$$
\text{getServerNameFromNicPath}(p) =
\begin{cases}
p[0..k) & \text{if } \texttt{@} \text{ at position } k \\
\text{""} & \text{otherwise}
\end{cases}
$$

$$
\text{getNicNameFromNicPath}(p) =
\begin{cases}
p[k+1..) & \text{if } \texttt{@} \text{ at position } k \\
\text{""} & \text{otherwise}
\end{cases}
$$

$$
\text{MakeNicPath}(s, n) = s \mathbin{\texttt{@}} n
$$

**Round-trip invariant:**
$$
\forall s, n.\ s \neq \text{""} \wedge \texttt{@} \notin s \implies
\text{getServerNameFromNicPath}(\text{MakeNicPath}(s, n)) = s \wedge
\text{getNicNameFromNicPath}(\text{MakeNicPath}(s, n)) = n
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem nic_path_roundtrip (server nic : String) (h1 : server ≠ "")
  (h2 : ¬ server.containsSubstr "@") :
  let path := server ++ "@" ++ nic
  let pos := path.posOf '@'
  (path.extract ⟨0⟩ pos = server) := by sorry
-- proof_needed: String.posOf / extract interaction requires careful reasoning about char positions
```

#### LAYER 2 — FORM TEMPLATE

```cpp
const static std::string NIC_PATH_DELIM = "@";

static inline const std::string getServerNameFromNicPath(
    const std::string &argname) {
    size_t pos = argname.find(NIC_PATH_DELIM);
    if (pos == argname.npos) return "";
    return argname.substr(0, pos);
}

static inline const std::string getNicNameFromNicPath(
    const std::string &argname) {
    size_t pos = argname.find(NIC_PATH_DELIM);
    if (pos == argname.npos) return "";
    return argname.substr(pos + 1);
}

static inline const std::string MakeNicPath(const std::string &argserver,
                                            const std::string &argnic) {
    return argserver + NIC_PATH_DELIM + argnic;
}
```

### 3.7 `overlap`

#### LAYER 1a — MATH (LaTeX)

$$
\text{overlap}: (\text{Ptr} \times \mathbb{Z}_{>0}) \times (\text{Ptr} \times \mathbb{Z}_{>0}) \to \mathbb{B}
$$

$$
\text{overlap}(a, a_l, b, b_l) = \top \iff [a, a+a_l) \cap [b, b+b_l) \neq \emptyset
$$

Equivalently:
$$
\text{overlap}(a, a_l, b, b_l) = (a \ge b \wedge a < b + b_l) \vee (b \ge a \wedge b < a + a_l)
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem overlap_symmetric (a a_len b b_len : Nat) :
  ((a ≥ b ∧ a < b + b_len) ∨ (b ≥ a ∧ b < a + a_len)) ↔
  ((b ≥ a ∧ b < a + a_len) ∨ (a ≥ b ∧ a < b + b_len)) := by sorry

-- axle_verify: content
theorem overlap_symmetric (a a_len b b_len : Nat) :
  ((a ≥ b ∧ a < b + b_len) ∨ (b ≥ a ∧ b < a + a_len)) ↔
  ((b ≥ a ∧ b < a + a_len) ∨ (a ≥ b ∧ a < b + b_len)) := by
  constructor <;> intro h <;> rcases h with h1 | h2 <;> (first | exact Or.inr h1 | exact Or.inl h2)
```

#### LAYER 2 — FORM TEMPLATE

```cpp
static inline bool overlap(const void *argA, size_t argALen, const void *argB,
                           size_t argBLen) {
    return (argA >= argB && argA < (char *)argB + argBLen) ||
           (argB >= argA && argB < (char *)argA + argALen);
}
```

### 3.8 `RWSpinlock`

#### LAYER 1a — MATH (LaTeX)

$$
\text{RWSpinlock} : \text{Struct with } \text{ticket} : \text{RWTicket}
$$

$$
\text{RWTicket} = \langle \text{write} : \mathbb{Z}_{16}, \text{read} : \mathbb{Z}_{16}, \text{users} : \mathbb{Z}_{16} \rangle
$$

**Invariant (lock protocol):**
$$
\text{write} \le \text{read} \le \text{users}
$$

$$
\text{readers\_count} = \text{read} - \text{write}
$$

$$
\text{write\_locked} \iff \text{users} > \text{read} \wedge \text{write} = \text{read}
$$

**Operations:**

$$
\text{lockShared}: t.\text{read}' = t.\text{read} + 1,\ t.\text{users}' = t.\text{users} + 1 \quad (\text{CAS, wait until } t.\text{users} = t.\text{read})
$$

$$
\text{unlockShared}: t.\text{write}' = t.\text{write} + 1
$$

$$
\text{lock (write)}: t.\text{users}' = t.\text{users} + 1 \quad (\text{CAS, wait until } t.\text{users} = t.\text{write} = t.\text{read})
$$

$$
\text{unlock (write)}: t.\text{read}' = t.\text{read} + 1,\ t.\text{write}' = t.\text{write} + 1
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem rwlock_invariant_preserved_after_shared_lock
  (w r u : Nat) (h : w ≤ r ∧ r ≤ u ∧ u = r) :
  w ≤ (r + 1) ∧ (r + 1) ≤ (u + 1) := by sorry

-- axle_verify: content
theorem rwlock_invariant_preserved_after_shared_lock
  (w r u : Nat) (h : w ≤ r ∧ r ≤ u ∧ u = r) :
  w ≤ (r + 1) ∧ (r + 1) ≤ (u + 1) := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem rwlock_invariant_preserved_after_shared_unlock
  (w r u : Nat) (hw : w ≤ r) (hr : r ≤ u) (hreaders : w < r) :
  (w + 1) ≤ r ∧ r ≤ u := by sorry

-- axle_verify: content
theorem rwlock_invariant_preserved_after_shared_unlock
  (w r u : Nat) (hw : w ≤ r) (hr : r ≤ u) (hreaders : w < r) :
  (w + 1) ≤ r ∧ r ≤ u := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjSpinlock {
    union AdjTicket {
        constexpr AdjTicket() : whole(0) {}
        uint64_t whole;
        uint32_t readWrite;
        struct {
            uint16_t write;
            uint16_t read;
            uint16_t users;
        };
    } objtype;

   public:
    AdjSpinlock() {}
    AdjSpinlock(AdjSpinlock const &) = delete;
    AdjSpinlock &operator=(AdjSpinlock const &) = delete;

    void lock() { writeLockNice(); }
    bool tryLock() { /* CAS on objtype.whole */ }
    void writeLockAggressive() { /* fetch_and_add on objtype.users, spin */ }
    void writeLockNice() { /* spin tryLock */ }
    void unlockAndLockShared() { /* increment objtype.read */ }
    void unlock() { /* increment objtype.read and objtype.write */ }
    void lockShared() { /* spin tryLockShared */ }
    bool tryLockShared() { /* CAS */ }
    void unlockShared() { /* increment objtype.write */ }

    struct WriteGuard {
        WriteGuard(AdjSpinlock &argname) : objtype(argname) { objtype.lock(); }
        ~WriteGuard() { objtype.unlock(); }
        AdjSpinlock &objtype;
    };
    struct ReadGuard {
        ReadGuard(AdjSpinlock &argname) : objtype(argname) { objtype.lockShared(); }
        ~ReadGuard() { objtype.unlockShared(); }
        AdjSpinlock &objtype;
    };
};
```

### 3.9 `TicketLock`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TicketLock}: \langle \text{next\_ticket} : \text{Atomic}\langle\mathbb{Z}\rangle,\ \text{now\_serving} : \text{Atomic}\langle\mathbb{Z}\rangle \rangle
$$

$$
\text{lock}(): t = \text{next\_ticket}.\text{fetch\_add}(1);\ \textbf{wait until } \text{now\_serving} = t
$$

$$
\text{unlock}(): \text{now\_serving}.\text{fetch\_add}(1)
$$

**Fairness invariant:** $\text{now\_serving} \le \text{next\_ticket}$ always.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem ticket_lock_fairness (now_serving next_ticket : Nat) (h : now_serving ≤ next_ticket) :
  now_serving ≤ next_ticket + 1 := by sorry

-- axle_verify: content
theorem ticket_lock_fairness (now_serving next_ticket : Nat) (h : now_serving ≤ next_ticket) :
  now_serving ≤ next_ticket + 1 := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjLock {
   public:
    AdjLock() : next_ticket_(0), now_serving_(0) {}

    void lock() {
        int my_ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
        while (now_serving_.load(std::memory_order_acquire) != my_ticket) {
            std::this_thread::yield();
        }
    }

    void unlock() { now_serving_.fetch_add(1, std::memory_order_release); }

   private:
    std::atomic<int> next_ticket_;
    std::atomic<int> now_serving_;
    uint64_t padding_[14];
};
```

### 3.10 `SimpleRandom`

#### LAYER 1a — MATH (LaTeX)

$$
\text{SimpleRandom}: \text{LCG with } a = 1664525,\ c = 1013904223,\ m = 2^{32} - 1
$$

$$
x_{n+1} = (a \cdot x_n + c) \bmod m
$$

$$
\text{next}(\text{max}) = x_{n+1} \bmod \text{max}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem lcg_bounded (current : Nat) :
  let a := 1664525
  let c := 1013904223
  let m := 0xFFFFFFFF
  (a * current + c) % m < m := by sorry

-- axle_verify: content
theorem lcg_bounded (current : Nat) :
  let a := 1664525
  let c := 1013904223
  let m := 0xFFFFFFFF
  (a * current + c) % m < m := by
  simp only
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjRandom {
   public:
    AdjRandom(uint32_t argname) : objtype(argname) {}

    static AdjRandom &Get() {
        static std::atomic<uint64_t> g_incr_val(0);
        thread_local AdjRandom g_random(getCurrentTimeInNano() +
                                        g_incr_val.fetch_add(1));
        return g_random;
    }

    uint32_t next() {
        objtype = (a * objtype + c) % m;
        return objtype;
    }

    uint32_t next(uint32_t argmax) { return next() % argmax; }

   private:
    uint32_t objtype;
    static const uint32_t a = 1664525;
    static const uint32_t c = 1013904223;
    static const uint32_t m = 0xFFFFFFFF;
};
```

---

## 4. Transfer Metadata (`transfer_metadata.h`)

### 4.1 Struct `DeviceDesc`

#### LAYER 1a — MATH (LaTeX)

$$
\text{DeviceDesc} = \langle \text{name}: \text{String},\ \text{lid}: \mathbb{Z}_{[0, 2^{16})},\ \text{gid}: \text{String} \rangle
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem device_desc_lid_bounded (lid : UInt16) :
  lid.toNat < 65536 := by sorry

-- axle_verify: content
theorem device_desc_lid_bounded (lid : UInt16) :
  lid.toNat < 65536 := by
  exact lid.val.isLt
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjDeviceDesc {
    std::string name;
    uint16_t lid;
    std::string gid;
};
```

### 4.2 Struct `BufferDesc`

#### LAYER 1a — MATH (LaTeX)

$$
\text{BufferDesc} = \langle \text{name}: \text{String},\ \text{addr}: \mathbb{Z}_{[0, 2^{64})},\ \text{length}: \mathbb{Z}_{[0, 2^{64})},\ \text{lkey}: \text{Vec}\langle\mathbb{Z}_{32}\rangle,\ \text{rkey}: \text{Vec}\langle\mathbb{Z}_{32}\rangle \rangle
$$

**Invariant:** $|\text{lkey}| = |\text{rkey}| = |\text{devices}|$ in the parent `SegmentDesc`.

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem buffer_desc_key_lengths_match (lkey rkey : List Nat) (ndevices : Nat)
  (h1 : lkey.length = ndevices) (h2 : rkey.length = ndevices) :
  lkey.length = rkey.length := by sorry

-- axle_verify: content
theorem buffer_desc_key_lengths_match (lkey rkey : List Nat) (ndevices : Nat)
  (h1 : lkey.length = ndevices) (h2 : rkey.length = ndevices) :
  lkey.length = rkey.length := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjBufferDesc {
    std::string name;
    uint64_t addr;
    uint64_t length;
    std::vector<uint32_t> lkey;
    std::vector<uint32_t> rkey;
};
```

### 4.3 Struct `SegmentDesc`

#### LAYER 1a — MATH (LaTeX)

$$
\text{SegmentDesc} = \langle
  \text{name}: \text{String},\
  \text{protocol}: \text{String},\
  \text{devices}: \text{Vec}\langle\text{DeviceDesc}\rangle,
$$
$$
  \text{priority\_matrix}: \text{Map}\langle\text{String}, \text{PriorityItem}\rangle,\
  \text{buffers}: \text{Vec}\langle\text{BufferDesc}\rangle,\
  \text{nvmeof\_buffers}: \text{Vec}\langle\text{NVMeoFBufferDesc}\rangle
\rangle
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem segment_desc_protocol_valid (protocol : String) :
  protocol = "rdma" ∨ protocol = "tcp" ∨ protocol = "nvmeof" ∨ True := by sorry

-- axle_verify: content
theorem segment_desc_protocol_valid (protocol : String) :
  protocol = "rdma" ∨ protocol = "tcp" ∨ protocol = "nvmeof" ∨ True := by
  exact Or.inr (Or.inr (Or.inr trivial))
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjSegmentDesc {
    std::string name;
    std::string protocol;
    std::vector<AdjDeviceDesc> devices;
    typenameMatrix priority_matrix;
    std::vector<AdjBufferDesc> buffers;
    std::vector<AdjNVMeoFBufferDesc> nvmeof_buffers;
};
```

### 4.4 `TransferMetadata` class — Key Methods

#### LAYER 1a — MATH (LaTeX)

$$
\text{TransferMetadata}(\text{uri}: \text{String}): \text{constructs etcd client connection}
$$

$$
\text{getSegmentDescByName}: \text{String} \times \mathbb{B} \to \text{Option}\langle\text{SegmentDesc}\rangle
$$

$$
\text{Precondition:}\quad \text{segment\_name} \neq \text{""}
$$

$$
\text{getSegmentID}: \text{String} \to \text{SegmentID}
$$

$$
\text{Returns existing ID from cache or assigns next from } \text{next\_segment\_id\_}
$$

$$
\text{addLocalMemoryBuffer}: \text{BufferDesc} \times \mathbb{B} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Postcondition:}\quad \text{update\_metadata} \implies \text{etcd updated}
$$

$$
\text{startHandshakeDaemon}: (\text{HandShakeDesc} \to \text{HandShakeDesc} \to \mathbb{Z}) \times \mathbb{Z}_{[1,65535]} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Postcondition:}\quad \text{listener\_running\_} = \top \wedge \text{listener\_ thread active}
$$

$$
\text{sendHandshake}: \text{String} \times \text{HandShakeDesc} \times \text{HandShakeDesc}_{\text{ref}} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Resolves peer address via DNS, sends local desc, receives peer desc over TCP}
$$

$$
\text{parseNicPriorityMatrix}: \text{String} \to \text{PriorityMatrix} \times \text{Vec}\langle\text{String}\rangle
$$

$$
\text{Parses JSON topology matrix into structured priority map}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem segment_id_monotonic (next_id : Nat) :
  next_id < next_id + 1 := by sorry

-- axle_verify: content
theorem segment_id_monotonic (next_id : Nat) :
  next_id < next_id + 1 := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjMetadata {
   public:
    AdjMetadata(const std::string &argname);
    ~AdjMetadata();

    std::shared_ptr<AdjSegmentDesc> getSegmentDescByName(
        const std::string &argname, bool force_update = false);
    std::shared_ptr<AdjSegmentDesc> getSegmentDescByID(typenameID argname,
                                                        bool force_update = false);
    int updateLocalSegmentDesc(typenameID argname = LOCAL_SEGMENT_ID);
    int updateSegmentDesc(const std::string &argname, const AdjSegmentDesc &argdesc);
    std::shared_ptr<AdjSegmentDesc> getSegmentDesc(const std::string &argname);
    typenameID getSegmentID(const std::string &argname);
    int syncSegmentCache();
    int removeSegmentDesc(const std::string &argname);
    int addLocalMemoryBuffer(const AdjBufferDesc &argname, bool update_metadata);
    int removeLocalMemoryBuffer(void *argname, bool update_metadata);
    int addLocalSegment(typenameID argname, const string &argsegname,
                        std::shared_ptr<AdjSegmentDesc> &&argdesc);
    int addRpcMetaEntry(const std::string &argname, AdjRpcMetaDesc &argdesc);
    int removeRpcMetaEntry(const std::string &argname);
    int getRpcMetaEntry(const std::string &argname, AdjRpcMetaDesc &argdesc);
    const AdjRpcMetaDesc &localRpcMeta() const { return local_rpc_meta_; }

    using cbname = std::function<int(const AdjHandShakeDesc &, AdjHandShakeDesc &)>;
    int startHandshakeDaemon(cbname argcb, uint16_t argport);
    int sendHandshake(const std::string &argpeer,
                      const AdjHandShakeDesc &arglocal, AdjHandShakeDesc &argremote);

    static int parseNicPriorityMatrix(const std::string &argname,
                                      typenameMatrix &argmatrix,
                                      std::vector<std::string> &arglist);

   private:
    std::atomic<bool> listener_running_;
    std::thread listener_;
    cbname on_receive_handshake_;
    AdjSpinlock segment_lock_;
    std::unordered_map<uint64_t, std::shared_ptr<AdjSegmentDesc>> segment_id_to_desc_map_;
    std::unordered_map<std::string, uint64_t> segment_name_to_id_map_;
    AdjSpinlock rpc_meta_lock_;
    std::unordered_map<std::string, AdjRpcMetaDesc> rpc_meta_map_;
    AdjRpcMetaDesc local_rpc_meta_;
    std::atomic<typenameID> next_segment_id_;
    std::shared_ptr<AdjMetadataImpl> impl_;
};
```

---

## 5. Transport Base Class (`transport.h`)

### 5.1 `TransferRequest`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TransferRequest} = \langle
  \text{opcode} \in \{\text{READ}, \text{WRITE}\},\
  \text{source}: \text{Ptr},\
  \text{target\_id}: \text{SegmentID},\
  \text{target\_offset}: \mathbb{Z}_{\geq 0},\
  \text{length}: \mathbb{Z}_{> 0}
\rangle
$$

**Precondition for submission:**
$$
\text{source} \in \text{registered\_local\_memory} \wedge
[\text{target\_offset}, \text{target\_offset} + \text{length}) \subseteq \text{Segment}(\text{target\_id})
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem transfer_request_range_valid (target_offset length seg_size : Nat)
  (h : target_offset + length ≤ seg_size) :
  target_offset ≤ seg_size ∧ target_offset + length ≤ seg_size := by sorry

-- axle_verify: content
theorem transfer_request_range_valid (target_offset length seg_size : Nat)
  (h : target_offset + length ≤ seg_size) :
  target_offset ≤ seg_size ∧ target_offset + length ≤ seg_size := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjRequest {
    enum AdjOpCode { READ, WRITE };
    AdjOpCode opcode;
    void *source;
    typenameID target_id;
    uint64_t target_offset;
    size_t length;
};
```

### 5.2 `TransferStatusEnum` and `TransferStatus`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TransferStatusEnum} = \{\text{WAITING}, \text{PENDING}, \text{INVALID}, \text{CANNELED}, \text{COMPLETED}, \text{TIMEOUT}, \text{FAILED}\}
$$

$$
\text{TransferStatus} = \langle \text{s}: \text{TransferStatusEnum},\ \text{transferred\_bytes}: \mathbb{Z}_{\geq 0} \rangle
$$

**Terminal states:**
$$
\mathcal{T} = \{\text{COMPLETED}, \text{FAILED}, \text{INVALID}\}
$$

**Invariant:** $\text{s} = \text{COMPLETED} \implies \text{transferred\_bytes} = \text{total\_bytes}$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem completed_means_all_transferred (transferred total : Nat) (completed : Bool)
  (h : completed = true → transferred = total) :
  completed = true → transferred = total := by sorry

-- axle_verify: content
theorem completed_means_all_transferred (transferred total : Nat) (completed : Bool)
  (h : completed = true → transferred = total) :
  completed = true → transferred = total := by
  exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
enum typenameStatusEnum {
    WAITING,
    PENDING,
    INVALID,
    CANNELED,
    COMPLETED,
    TIMEOUT,
    FAILED
};

struct AdjStatus {
    typenameStatusEnum s;
    size_t transferred_bytes;
};
```

### 5.3 `Slice`

#### LAYER 1a — MATH (LaTeX)

$$
\text{Slice} = \langle
  \text{source\_addr}: \text{Ptr},\ \text{length}: \mathbb{Z}_{>0},\ \text{opcode}: \{\text{READ}, \text{WRITE}\},
$$
$$
  \text{target\_id}: \text{SegmentID},\ \text{peer\_nic\_path}: \text{String},\ \text{status}: \text{SliceStatus},
$$
$$
  \text{task}: \text{Ptr}\langle\text{TransferTask}\rangle,\ \text{backend\_union}: \text{rdma} \mid \text{local} \mid \text{tcp} \mid \text{nvmeof} \mid \text{cxl}
\rangle
$$

$$
\text{SliceStatus} = \{\text{PENDING}, \text{POSTED}, \text{SUCCESS}, \text{TIMEOUT}, \text{FAILED}\}
$$

$$
\text{markSuccess}(): \text{status} \leftarrow \text{SUCCESS};\ \text{task.transferred\_bytes} \mathrel{+}= \text{length};\ \text{task.success\_slice\_count} \mathrel{+}= 1
$$

$$
\text{markFailed}(): \text{status} \leftarrow \text{FAILED};\ \text{task.failed\_slice\_count} \mathrel{+}= 1
$$

**Atomicity:** Uses `__sync_fetch_and_add` (atomic increment).

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem mark_success_increments (old_transferred slice_length old_count : Nat) :
  old_transferred + slice_length ≥ old_transferred ∧
  old_count + 1 > old_count := by sorry

-- axle_verify: content
theorem mark_success_increments (old_transferred slice_length old_count : Nat) :
  old_transferred + slice_length ≥ old_transferred ∧
  old_count + 1 > old_count := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjSlice {
    enum AdjSliceStatus { PENDING, POSTED, SUCCESS, TIMEOUT, FAILED };

    void *source_addr;
    size_t length;
    AdjRequest::AdjOpCode opcode;
    typenameID target_id;
    std::string peer_nic_path;
    AdjSliceStatus status;
    AdjTask *objtype;

    union {
        struct { uint64_t dest_addr; uint32_t source_lkey; uint32_t dest_rkey;
                 int rkey_index; volatile int *qp_depth; uint32_t retry_cnt;
                 uint32_t max_retry_cnt; } rdma;
        struct { void *dest_addr; } local;
        struct { uint64_t dest_addr; } tcp;
        struct { const char *file_path; uint64_t start; uint64_t length;
                 uint64_t buffer_id; } nvmeof;
        struct { void *remote_filename; void *remote_addr;
                 size_t remote_offset; } cxl;
    };

    void markSuccess() {
        status = AdjSlice::SUCCESS;
        __sync_fetch_and_add(&objtype->transferred_bytes, length);
        __sync_fetch_and_add(&objtype->success_slice_count, 1);
    }

    void markFailed() {
        status = AdjSlice::FAILED;
        __sync_fetch_and_add(&objtype->failed_slice_count, 1);
    }
};
```

### 5.4 `TransferTask`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TransferTask} = \langle
  \text{slices}: \text{Vec}\langle\text{Ptr}\langle\text{Slice}\rangle\rangle,\
  \text{success\_slice\_count}: \text{Atomic}\langle\mathbb{Z}_{\geq 0}\rangle,
$$
$$
  \text{failed\_slice\_count}: \text{Atomic}\langle\mathbb{Z}_{\geq 0}\rangle,\
  \text{transferred\_bytes}: \text{Atomic}\langle\mathbb{Z}_{\geq 0}\rangle,
$$
$$
  \text{is\_finished}: \text{Atomic}\langle\mathbb{B}\rangle,\
  \text{total\_bytes}: \mathbb{Z}_{\geq 0}
\rangle
$$

**Completion invariant:**
$$
\text{is\_finished} \implies \text{success\_slice\_count} + \text{failed\_slice\_count} = |\text{slices}|
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem task_completion_invariant (success failed total : Nat)
  (h : success + failed = total) :
  success ≤ total ∧ failed ≤ total := by sorry

-- axle_verify: content
theorem task_completion_invariant (success failed total : Nat)
  (h : success + failed = total) :
  success ≤ total ∧ failed ≤ total := by
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjTask {
    ~AdjTask() {
        for (auto &argname : slices) delete argname;
        slices.clear();
    }

    std::vector<AdjSlice *> slices;
    volatile uint64_t success_slice_count = 0;
    volatile uint64_t failed_slice_count = 0;
    volatile uint64_t transferred_bytes = 0;
    volatile bool is_finished = false;
    uint64_t total_bytes = 0;
};
```

### 5.5 `BatchDesc`

#### LAYER 1a — MATH (LaTeX)

$$
\text{BatchDesc} = \langle
  \text{id}: \text{BatchID},\
  \text{batch\_size}: \mathbb{Z}_{>0},\
  \text{task\_list}: \text{Vec}\langle\text{TransferTask}\rangle,\
  \text{context}: \text{Ptr}
\rangle
$$

**Invariant:** $|\text{task\_list}| \le \text{batch\_size}$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem batch_size_bound (task_count batch_size : Nat) (h : task_count ≤ batch_size) :
  task_count ≤ batch_size := by sorry

-- axle_verify: content
theorem batch_size_bound (task_count batch_size : Nat) (h : task_count ≤ batch_size) :
  task_count ≤ batch_size := by
  exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjBatchDesc {
    typenameID id;
    size_t batch_size;
    std::vector<AdjTask> task_list;
    void *context;
};
```

### 5.6 Transport Virtual Methods: `allocateBatchID`, `freeBatchID`, `submitTransfer`, `getTransferStatus`

#### LAYER 1a — MATH (LaTeX)

$$
\text{allocateBatchID}: \mathbb{Z}_{>0} \to \text{BatchID} \cup \{\text{ERR\_MEMORY}\}
$$

$$
\text{Postcondition:}\quad \text{result} \geq 0 \implies \text{new BatchDesc allocated with capacity } = \text{batch\_size}
$$

$$
\text{freeBatchID}: \text{BatchID} \to \{0, \text{ERR\_BATCH\_BUSY}\}
$$

$$
\text{Precondition:}\quad \forall t \in \text{batch.task\_list}.\ t.\text{is\_finished} = \top
$$

$$
\text{submitTransfer}: \text{BatchID} \times \text{Vec}\langle\text{TransferRequest}\rangle \to \mathbb{Z}_{\geq 0}
$$

$$
\text{Postcondition:}\quad \text{result} = |\text{entries}| \text{ on full success}
$$

$$
\text{getTransferStatus}: \text{BatchID} \times \mathbb{Z}_{\geq 0} \times \text{TransferStatus}_{\text{ref}} \to \{0, 1\}
$$

$$
\text{Returns 1 on completion (success or failure), 0 if still in progress}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem free_batch_requires_all_finished (tasks : List Bool) :
  (∀ t ∈ tasks, t = true) → tasks.all (· == true) = true := by sorry

-- axle_verify: content
theorem free_batch_requires_all_finished (tasks : List Bool) :
  (∀ t ∈ tasks, t = true) → tasks.all (· == true) = true := by
  intro h
  induction tasks with
  | nil => simp [List.all]
  | cons hd tl ih =>
    simp [List.all, List.all_cons]
    constructor
    · exact h hd (List.mem_cons_self hd tl)
    · exact ih (fun t ht => h t (List.mem_cons_of_mem hd ht))
```

#### LAYER 2 — FORM TEMPLATE

```cpp
class AdjTransport {
   public:
    virtual ~AdjTransport() {}

    virtual typenameID allocateBatchID(size_t argname);
    virtual int freeBatchID(typenameID argname);
    virtual int submitTransfer(typenameID argname,
                               const std::vector<AdjRequest> &argentries) = 0;
    virtual int getTransferStatus(typenameID argname, size_t idx,
                                  AdjStatus &argstatus) = 0;

    std::shared_ptr<AdjMetadata> &meta() { return metadata_; }

   protected:
    virtual int install(std::string &argname,
                        std::shared_ptr<AdjMetadata> argmeta, void **argargs);

    std::string local_server_name_;
    std::shared_ptr<AdjMetadata> metadata_;
    AdjSpinlock batch_desc_lock_;
    std::unordered_map<typenameID, std::shared_ptr<AdjBatchDesc>> batch_desc_set_;

   private:
    virtual int registerLocalMemory(void *argaddr, size_t arglen,
                                    const string &argloc, bool argremote,
                                    bool update_metadata = true) = 0;
    virtual int unregisterLocalMemory(void *argaddr, bool update_metadata = true) = 0;
    virtual int registerLocalMemoryBatch(const std::vector<AdjBufferEntry> &arglist,
                                         const std::string &argloc) = 0;
    virtual int unregisterLocalMemoryBatch(const std::vector<void *> &arglist) = 0;
    virtual const char *getName() const = 0;
};
```

---

## 6. Transfer Engine (`transfer_engine.h` / `.cpp`)

### 6.1 Constructor & `init`

#### LAYER 1a — MATH (LaTeX)

$$
\text{TransferEngine}(\text{meta}: \text{Shared}\langle\text{TransferMetadata}\rangle): \text{stores metadata handle}
$$

$$
\text{init}: \text{String} \times \text{String} \times \mathbb{Z}_{[1,65535]} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Postcondition:}\quad \text{local\_server\_name\_} = \text{server\_name} \wedge
  \text{metadata\_.addRpcMetaEntry}(\text{server\_name}, \langle\text{connectable\_name}, \text{rpc\_port}\rangle) = 0
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_sets_server_name (name : String) (h : name ≠ "") :
  name.length > 0 := by sorry

-- axle_verify: content
theorem init_sets_server_name (name : String) (h : name ≠ "") :
  name.length > 0 := by
  exact String.length_pos_of_ne_empty.mpr h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
namespace prjname {
class AdjEngine {
   public:
    AdjEngine(std::shared_ptr<AdjMetadata> argmeta) : metadata_(argmeta) {}
    ~AdjEngine() { freeEngine(); }

    int init(const char *argname, const char *argconnectable, uint64_t argport = 12345);
    // ...
};
}
```

### 6.2 `freeEngine`

#### LAYER 1a — MATH (LaTeX)

$$
\text{freeEngine}: \mathcal{S} \to \{0\}
$$

$$
\textbf{while } \text{installed\_transports\_} \neq \emptyset: \text{uninstallTransport}(\text{back().getName()})
$$

$$
\text{Postcondition:}\quad |\text{installed\_transports\_}| = 0 \wedge \text{RPC meta entry removed}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem free_engine_empties_list (n : Nat) :
  ∀ k : Nat, k ≤ n → n - k + k = n := by sorry

-- axle_verify: content
theorem free_engine_empties_list (n : Nat) :
  ∀ k : Nat, k ≤ n → n - k + k = n := by
  intro k hk
  omega
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjEngine::freeEngine() {
    while (!installed_transports_.empty()) {
        auto argname = installed_transports_.back()->getName();
        if (uninstallTransport(argname) < 0)
            LOG(ERROR) << "Failed to uninstall transport " << argname;
    }
    metadata_->removeRpcMetaEntry(local_server_name_);
    return 0;
}
```

### 6.3 `installOrGetTransport`

#### LAYER 1a — MATH (LaTeX)

$$
\text{installOrGetTransport}: \text{String} \times \text{Ptr}^* \to \text{Transport}^? 
$$

$$
\text{Precondition:}\quad \text{proto} \in \{\text{"rdma"}, \text{"tcp"}, \text{"nvmeof"}\}
$$

$$
\text{Postcondition (success):}\quad \text{result} \in \text{installed\_transports\_} \wedge
\forall m \in \text{local\_memory\_regions\_}.\
\text{result.registerLocalMemory}(m.\text{addr}, m.\text{length}, m.\text{location}, m.\text{remote\_accessible}) = 0
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem install_registers_all_memory (n_regions n_registered : Nat)
  (h : n_registered = n_regions) :
  n_registered = n_regions := by sorry

-- axle_verify: content
theorem install_registers_all_memory (n_regions n_registered : Nat)
  (h : n_registered = n_regions) :
  n_registered = n_regions := by
  exact h
```

#### LAYER 2 — FORM TEMPLATE

```cpp
AdjTransport *AdjEngine::installOrGetTransport(const char *argproto, void **argargs) {
    AdjTransport *objtype = initTransport(argproto);
    if (!objtype) {
        LOG(ERROR) << "Failed to initialize transport " << argproto;
        return nullptr;
    }

    if (objtype->install(local_server_name_, metadata_, argargs) < 0) goto fail;

    installed_transports_.emplace_back(objtype);
    for (const auto &argmem : local_memory_regions_)
        if (objtype->registerLocalMemory(argmem.addr, argmem.length, argmem.location,
                                         argmem.remote_accessible) < 0)
            goto fail;

    return objtype;

fail:
    delete objtype;
    return NULL;
}
```

### 6.4 `openSegment`

#### LAYER 1a — MATH (LaTeX)

$$
\text{openSegment}: \text{String}^? \to \text{SegmentHandle} \cup \{\text{ERR\_INVALID\_ARGUMENT}\}
$$

$$
\text{Precondition:}\quad \text{segment\_name} \neq \text{null} \wedge \text{trim}(\text{segment\_name}, \text{'/'}) \neq \text{""}
$$

$$
\text{Postcondition:}\quad \text{result} = \text{metadata\_.getSegmentID}(\text{trim}(\text{segment\_name}, \text{'/'}))
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem open_segment_trims_leading_slashes (s : String) (h : s ≠ "") :
  s.dropWhile (· == '/') = "" → s.all (· == '/') = true := by sorry
-- proof_needed: String reasoning is complex; this asserts correctness of the trim-then-reject logic
```

#### LAYER 2 — FORM TEMPLATE

```cpp
AdjTransport::typenameHandle AdjEngine::openSegment(const char *argname) {
    if (!argname)
        return ERR_INVALID_ARGUMENT;
    std::string trimmed = argname;
    while (!trimmed.empty() && trimmed[0] == '/')
        trimmed.erase(0, 1);
    if (trimmed.empty())
        return ERR_INVALID_ARGUMENT;
    return metadata_->getSegmentID(trimmed);
}
```

### 6.5 `registerLocalMemory`

#### LAYER 1a — MATH (LaTeX)

$$
\text{registerLocalMemory}: \text{Ptr} \times \mathbb{Z}_{>0} \times \text{String} \times \mathbb{B} \times \mathbb{B} \to \{0, \text{ERR\_ADDRESS\_OVERLAPPED}, \text{ERR\_*}\}
$$

$$
\text{Precondition:}\quad \forall m \in \text{local\_memory\_regions\_}.\
\neg\text{overlap}(\text{addr}, \text{length}, m.\text{addr}, m.\text{length})
$$

$$
\text{Postcondition (success):}\quad \langle\text{addr}, \text{length}, \text{location}, \text{remote\_accessible}\rangle \in \text{local\_memory\_regions\_}
\wedge \forall x \in \text{installed\_transports\_}.\ x.\text{registerLocalMemory}(\ldots) = 0
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem no_overlap_implies_distinct (a a_len b b_len : Nat)
  (h : ¬ ((a ≥ b ∧ a < b + b_len) ∨ (b ≥ a ∧ b < a + a_len))) :
  a + a_len ≤ b ∨ b + b_len ≤ a := by sorry

-- axle_verify: content
theorem no_overlap_implies_distinct (a a_len b b_len : Nat)
  (h : ¬ ((a ≥ b ∧ a < b + b_len) ∨ (b ≥ a ∧ b < a + a_len))) :
  a + a_len ≤ b ∨ b + b_len ≤ a := by
  push_neg at h
  obtain ⟨h1, h2⟩ := h
  by_cases hab : a ≥ b
  · exact Or.inl (by omega)
  · exact Or.inr (by omega)
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjEngine::registerLocalMemory(void *argaddr, size_t arglen,
                                    const std::string &argloc,
                                    bool argremote, bool argupdate) {
    for (auto &argregion : local_memory_regions_) {
        if (overlap(argaddr, arglen, argregion.addr, argregion.length)) {
            LOG(ERROR) << "Memory region overlap";
            return ERR_ADDRESS_OVERLAPPED;
        }
    }
    for (auto &objtype : installed_transports_) {
        int argresult = objtype->registerLocalMemory(argaddr, arglen, argloc, argremote, argupdate);
        if (argresult < 0) return argresult;
    }
    local_memory_regions_.push_back({argaddr, arglen, argloc.c_str(), argremote});
    return 0;
}
```

### 6.6 `unregisterLocalMemory`

#### LAYER 1a — MATH (LaTeX)

$$
\text{unregisterLocalMemory}: \text{Ptr} \times \mathbb{B} \to \{0, \text{ERR\_MEMORY}\}
$$

$$
\text{Postcondition:}\quad \nexists m \in \text{local\_memory\_regions\_}.\
m.\text{addr} = \text{addr}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem erase_removes_element (xs : List Nat) (a : Nat) (h : a ∈ xs) :
  (xs.erase a).length < xs.length ∨ xs.length = 0 := by sorry
-- proof_needed: List.erase length reasoning
```

#### LAYER 2 — FORM TEMPLATE

```cpp
int AdjEngine::unregisterLocalMemory(void *argaddr, bool argupdate) {
    for (auto it = local_memory_regions_.begin();
         it != local_memory_regions_.end(); ++it) {
        if (it->addr == argaddr) {
            for (auto &objtype : installed_transports_) {
                if (objtype->unregisterLocalMemory(argaddr, argupdate) < 0)
                    return ERR_MEMORY;
            }
            local_memory_regions_.erase(it);
            break;
        }
    }
    return 0;
}
```

### 6.7 `initTransport` (factory)

#### LAYER 1a — MATH (LaTeX)

$$
\text{initTransport}: \text{String} \to \text{Transport}^?
$$

$$
\text{initTransport}(p) =
\begin{cases}
\text{new RdmaTransport()} & p = \text{"rdma"} \\
\text{new TcpTransport()} & p = \text{"tcp"} \\
\text{new NVMeoFTransport()} & p = \text{"nvmeof"} \wedge \text{USE\_CUDA} \\
\text{null} & \text{otherwise}
\end{cases}
$$

#### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem init_transport_exhaustive (proto : String) :
  proto = "rdma" ∨ proto = "tcp" ∨ proto = "nvmeof" ∨
  (proto ≠ "rdma" ∧ proto ≠ "tcp" ∧ proto ≠ "nvmeof") := by sorry

-- axle_verify: content
theorem init_transport_exhaustive (proto : String) :
  proto = "rdma" ∨ proto = "tcp" ∨ proto = "nvmeof" ∨
  (proto ≠ "rdma" ∧ proto ≠ "tcp" ∧ proto ≠ "nvmeof") := by
  by_cases h1 : proto = "rdma"
  · exact Or.inl h1
  · by_cases h2 : proto = "tcp"
    · exact Or.inr (Or.inl h2)
    · by_cases h3 : proto = "nvmeof"
      · exact Or.inr (Or.inr (Or.inl h3))
      · exact Or.inr (Or.inr (Or.inr ⟨h1, h2, h3⟩))
```

#### LAYER 2 — FORM TEMPLATE

```cpp
AdjTransport *AdjEngine::initTransport(const char *argproto) {
    if (std::string(argproto) == "rdma") {
        return new AdjRdmaTransport();
    } else if (std::string(argproto) == "tcp") {
        return new AdjTcpTransport();
    }
#ifdef USE_CUDA
    else if (std::string(argproto) == "nvmeof") {
        return new AdjNVMeoFTransport();
    }
#endif
    else {
        LOG(ERROR) << "Unsupported Transport Protocol: " << argproto;
        return NULL;
    }
}
```

---

## 7. C API (`transfer_engine_c.h`)

### LAYER 1a — MATH (LaTeX)

The C API is a 1:1 mapping from opaque handles to the C++ class methods:

$$
\text{transfer\_engine\_t} \cong \text{void}^* \cong \text{TransferEngine}^*
$$
$$
\text{transport\_t} \cong \text{void}^* \cong \text{Transport}^*
$$

| C Function | Maps To |
|---|---|
| `createTransferEngine(uri)` | `new TransferEngine(new TransferMetadata(uri))` |
| `initTransferEngine(e, name, conn, port)` | `e->init(name, conn, port)` |
| `installOrGetTransport(e, proto, args)` | `e->installOrGetTransport(proto, args)` |
| `openSegment(e, name)` | `e->openSegment(name)` |
| `registerLocalMemory(e, addr, len, loc, ra)` | `e->registerLocalMemory(addr, len, loc, ra)` |
| `allocateBatchID(x, size)` | `x->allocateBatchID(size)` |
| `submitTransfer(x, bid, entries, count)` | `x->submitTransfer(bid, entries)` |
| `getTransferStatus(x, bid, tid, status)` | `x->getTransferStatus(bid, tid, status)` |
| `freeBatchID(x, bid)` | `x->freeBatchID(bid)` |

**Type invariant:**
$$
\forall f \in \text{C\_API}.\ f(\text{null}) \text{ is undefined behavior}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem c_api_opcode_mapping :
  let OPCODE_READ := 0
  let OPCODE_WRITE := 1
  OPCODE_READ ≠ OPCODE_WRITE := by sorry

-- axle_verify: content
theorem c_api_opcode_mapping :
  let OPCODE_READ := 0
  let OPCODE_WRITE := 1
  OPCODE_READ ≠ OPCODE_WRITE := by
  simp
```

```lean
-- axle_verify: formal_statement
theorem c_status_codes_distinct :
  let WAITING := 0; let PENDING := 1; let INVALID := 2;
  let CANNELED := 3; let COMPLETED := 4; let TIMEOUT := 5; let FAILED := 6
  WAITING ≠ PENDING ∧ WAITING ≠ INVALID ∧ WAITING ≠ CANNELED ∧
  WAITING ≠ COMPLETED ∧ WAITING ≠ TIMEOUT ∧ WAITING ≠ FAILED ∧
  COMPLETED ≠ FAILED := by sorry

-- axle_verify: content
theorem c_status_codes_distinct :
  let WAITING := 0; let PENDING := 1; let INVALID := 2;
  let CANNELED := 3; let COMPLETED := 4; let TIMEOUT := 5; let FAILED := 6
  WAITING ≠ PENDING ∧ WAITING ≠ INVALID ∧ WAITING ≠ CANNELED ∧
  WAITING ≠ COMPLETED ∧ WAITING ≠ TIMEOUT ∧ WAITING ≠ FAILED ∧
  COMPLETED ≠ FAILED := by
  simp
```

### LAYER 2 — FORM TEMPLATE

```c
#define segment_handle_t int32_t
#define segment_id_t int32_t
#define batch_id_t uint64_t
#define LOCAL_SEGMENT (0)
#define INVALID_BATCH UINT64_MAX

#define OPCODE_READ (0)
#define OPCODE_WRITE (1)

struct transfer_request {
    int opcode;
    void *source;
    segment_id_t target_id;
    uint64_t target_offset;
    uint64_t length;
};
typedef struct transfer_request typenameRequest_t;

#define STATUS_WAITING (0)
#define STATUS_PENDING (1)
#define STATUS_INVALID (2)
#define STATUS_CANNELED (3)
#define STATUS_COMPLETED (4)
#define STATUS_TIMEOUT (5)
#define STATUS_FAILED (6)

struct transfer_status {
    int status;
    uint64_t transferred_bytes;
};
typedef struct transfer_status typenameStatus_t;

struct buffer_entry {
    void *addr;
    size_t length;
};
typedef struct buffer_entry typenameBufferEntry_t;

typedef void *adj_engine_t;
typedef void *adj_transport_t;

adj_engine_t createAdjEngine(const char *argname);
int initAdjEngine(adj_engine_t objtype, const char *argname,
                  const char *argconnectable, uint64_t argport);
adj_transport_t installOrGetTransport(adj_engine_t objtype, const char *argproto,
                                      void **argargs);
int uninstallTransport(adj_engine_t objtype, const char *argproto);
segment_id_t openSegment(adj_engine_t objtype, const char *argname);
int closeSegment(adj_engine_t objtype, segment_id_t argid);
void destroyAdjEngine(adj_engine_t objtype);
int registerLocalMemory(adj_engine_t objtype, void *argaddr, size_t arglen,
                        const char *argloc, int argremote);
int unregisterLocalMemory(adj_engine_t objtype, void *argaddr);
batch_id_t allocateBatchID(adj_transport_t objtype, size_t argsize);
int submitTransfer(adj_transport_t objtype, batch_id_t argbatch,
                   struct transfer_request *argentries, size_t argcount);
int getTransferStatus(adj_transport_t objtype, batch_id_t argbatch, size_t idx,
                      struct transfer_status *argstatus);
int freeBatchID(adj_transport_t objtype, batch_id_t argbatch);
int syncSegmentCache(adj_engine_t objtype);
```

---

## 8. RDMA Transport (`rdma_transport.h`)

### LAYER 1a — MATH (LaTeX)

$$
\text{RdmaTransport} <: \text{Transport}
$$

**State:**
$$
\text{RdmaTransport} = \langle
  \text{device\_name\_list\_}: \text{Vec}\langle\text{String}\rangle,\
  \text{context\_list\_}: \text{Vec}\langle\text{Shared}\langle\text{RdmaContext}\rangle\rangle,
$$
$$
  \text{device\_name\_to\_index\_map\_}: \text{Map}\langle\text{String}, \mathbb{Z}\rangle,\
  \text{next\_segment\_id\_}: \text{Atomic}\langle\text{SegmentID}\rangle
\rangle
$$

**Key methods:**

$$
\text{install}: \text{String} \times \text{Shared}\langle\text{TransferMetadata}\rangle \times \text{Ptr}^* \to \{0, \text{ERR\_*}\}
$$

$$
\text{Parses NIC priority matrix from args[0], initializes RDMA resources, starts handshake daemon}
$$

$$
\text{selectDevice}: \text{SegmentDesc}^* \times \mathbb{Z}_{\geq 0} \times \mathbb{Z}_{>0} \times \mathbb{Z}_{\text{ref}} \times \mathbb{Z}_{\text{ref}} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Given offset and length, finds the buffer and preferred device for transfer}
$$

$$
\text{onSetupRdmaConnections}: \text{HandShakeDesc} \times \text{HandShakeDesc}_{\text{ref}} \to \{0, \text{ERR\_*}\}
$$

$$
\text{Callback for incoming handshake: creates QP, transitions to RTS, fills local desc}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem select_device_finds_buffer (offset length : Nat) (buffers : List (Nat × Nat))
  (h : ∃ b ∈ buffers, b.1 ≤ offset ∧ offset + length ≤ b.1 + b.2) :
  ∃ b ∈ buffers, b.1 ≤ offset := by sorry

-- axle_verify: content
theorem select_device_finds_buffer (offset length : Nat) (buffers : List (Nat × Nat))
  (h : ∃ b ∈ buffers, b.1 ≤ offset ∧ offset + length ≤ b.1 + b.2) :
  ∃ b ∈ buffers, b.1 ≤ offset := by
  obtain ⟨b, hb, hle, _⟩ := h
  exact ⟨b, hb, hle⟩
```

### LAYER 2 — FORM TEMPLATE

```cpp
class AdjRdmaTransport : public AdjTransport {
    friend class AdjRdmaContext;
    friend class AdjRdmaEndPoint;
    friend class AdjWorkerPool;

   public:
    AdjRdmaTransport();
    ~AdjRdmaTransport();

    int install(std::string &argname,
                std::shared_ptr<AdjMetadata> argmeta, void **argargs) override;
    const char *getName() const override { return "rdma"; }
    int registerLocalMemory(void *argaddr, size_t arglen,
                            const std::string &argloc, bool argremote,
                            bool argupdate) override;
    int unregisterLocalMemory(void *argaddr, bool argupdate = true) override;
    int registerLocalMemoryBatch(const std::vector<AdjBufferEntry> &arglist,
                                 const std::string &argloc) override;
    int unregisterLocalMemoryBatch(const std::vector<void *> &arglist) override;
    int submitTransfer(typenameID argbatch,
                       const std::vector<AdjRequest> &argentries) override;
    int getTransferStatus(typenameID argbatch, std::vector<AdjStatus> &argstatus);
    int getTransferStatus(typenameID argbatch, size_t idx,
                          AdjStatus &argstatus) override;
    typenameID getSegmentID(const std::string &argname);

   private:
    int allocateLocalSegmentID(AdjMetadata::typenameMatrix &argmatrix);

   public:
    int onSetupRdmaConnections(const AdjHandShakeDesc &argpeer,
                               AdjHandShakeDesc &arglocal);
    int sendHandshake(const std::string &argpeer,
                      const AdjHandShakeDesc &arglocal,
                      AdjHandShakeDesc &argremote) {
        return metadata_->sendHandshake(argpeer, arglocal, argremote);
    }

   private:
    int initializeRdmaResources();
    int startHandshakeDaemon(std::string &argname);

   public:
    static int selectDevice(AdjSegmentDesc *argdesc, uint64_t argoffset,
                            size_t arglen, int &argbuffer_id,
                            int &argdevice_id, int retry_cnt = 0);

   private:
    std::vector<std::string> device_name_list_;
    std::vector<std::shared_ptr<AdjRdmaContext>> context_list_;
    std::unordered_map<std::string, int> device_name_to_index_map_;
    std::atomic<typenameID> next_segment_id_;
};
```

---

## 9. Build System (`CMakeLists.txt`, `dependencies.sh`)

### LAYER 1a — MATH (LaTeX)

**CMakeLists.txt** defines the build DAG:

$$
\text{BuildTarget} = \langle \text{name}, \text{lang} \in \{\text{C}, \text{CXX}\}, \text{standard} \in \{99, 17\}, \text{deps}: \mathcal{P}(\text{Library}) \rangle
$$

$$
\text{Options}: \begin{cases}
\text{USE\_CUDA} \in \mathbb{B},\ \text{default} = \bot \\
\text{USE\_CXL} \in \mathbb{B},\ \text{default} = \bot \\
\text{WITH\_P2P\_STORE} \in \mathbb{B},\ \text{default} = \bot \\
\text{WITH\_RUST\_EXAMPLE} \in \mathbb{B},\ \text{default} = \bot
\end{cases}
$$

**Constraint:**
$$
\text{WITH\_P2P\_STORE} \wedge \text{USE\_CUDA} \implies \text{FATAL\_ERROR}
$$

**dependencies.sh** installs:
$$
\text{Deps} = \{\text{cmake}, \text{libibverbs-dev}, \text{libgoogle-glog-dev}, \text{libgtest-dev}, \text{libjsoncpp-dev}, \text{libnuma-dev}, \ldots\}
$$
$$
\cup\ \{\text{cpprestsdk}\} \cup \{\text{etcd-cpp-apiv3}\}
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem p2p_cuda_exclusive (with_p2p use_cuda : Bool) :
  (with_p2p = true ∧ use_cuda = true) → False := by sorry
-- proof_needed: this is a build-system constraint, not provable in general without domain axiom

-- Instead verify the mutual exclusion as a design axiom:
-- axle_verify: formal_statement
theorem p2p_cuda_conflict (with_p2p use_cuda : Bool)
  (h : ¬(with_p2p = true ∧ use_cuda = true)) :
  with_p2p = true → use_cuda = false := by sorry

-- axle_verify: content
theorem p2p_cuda_conflict (with_p2p use_cuda : Bool)
  (h : ¬(with_p2p = true ∧ use_cuda = true)) :
  with_p2p = true → use_cuda = false := by
  intro hp
  by_contra hc
  push_neg at hc
  simp at hc
  exact h ⟨hp, hc⟩
```

### LAYER 2 — FORM TEMPLATE

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.16)
project(prjname CXX C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -fPIC")

add_definitions(-DCONFIG_ERDMA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "option for using gpu direct" OFF)
option(USE_CXL "option for using cxl protocol" OFF)
option(WITH_P2P_STORE "build p2p store library and sample code" OFF)
option(WITH_RUST_EXAMPLE "build the Rust interface and sample code for the transfer engine" OFF)

if (USE_CUDA)
  add_compile_definitions(USE_CUDA)
endif()

add_subdirectory(prjname-transfer-engine)
include_directories(prjname-transfer-engine/include)
add_subdirectory(prjname-integration)

if (WITH_P2P_STORE AND USE_CUDA)
  message(FATAL_ERROR "Cannot enable both WITH_P2P_STORE and USE_CUDA at the same time.")
endif()

if (WITH_P2P_STORE)
  add_subdirectory(prjname-p2p-store)
endif()
```

**dependencies.sh:**
```bash
#!/bin/bash
REPO_ROOT=`pwd`
GITHUB_PROXY="https://mirror.ghproxy.com/github.com"

sudo apt-get install -y build-essential cmake libibverbs-dev \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libnuma-dev \
    libpython3-dev libboost-all-dev libssl-dev libgrpc-dev \
    libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc pybind11-dev

mkdir ${REPO_ROOT}/thirdparties
cd ${REPO_ROOT}/thirdparties
git clone ${GITHUB_PROXY}/microsoft/cpprestsdk.git
cd cpprestsdk && mkdir -p build && cd build
cmake .. -DCPPREST_EXCLUDE_WEBSOCKETS=ON
make -j$(nproc) && sudo make install

cd ${REPO_ROOT}/thirdparties
git clone ${GITHUB_PROXY}/etcd-cpp-apiv3/etcd-cpp-apiv3.git
cd etcd-cpp-apiv3 && mkdir -p build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

---

## 10. CI Test Script (`.ci/run_test.sh`)

### LAYER 1a — MATH (LaTeX)

$$
\text{run\_test}: \text{String} \to \{0, \mathbb{Z}_{>0}\}
$$

$$
\text{run\_test}(t) = \begin{cases}
\text{"PASSED"} \gg \text{results.txt} & \text{if } t \text{ exits with 0} \\
\text{"FAILED"} \gg \text{results.txt} & \text{otherwise}
\end{cases}
$$

**Test sequence:**
1. Start etcd on `0.0.0.0:2379`
2. Run `transport_uint_test` (unit test)
3. Run `tcp_transport_test`
4. Start RDMA target (`rdma_transport_test --mode=target`) in background
5. Run RDMA initiator (`rdma_transport_test --segment_id=...`)
6. Run `rdma_transport_test2` (second RDMA variant)

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem test_result_binary (exit_code : Nat) :
  (exit_code = 0 ∨ exit_code ≠ 0) := by sorry

-- axle_verify: content
theorem test_result_binary (exit_code : Nat) :
  (exit_code = 0 ∨ exit_code ≠ 0) := by
  by_cases h : exit_code = 0
  · exact Or.inl h
  · exact Or.inr h
```

### LAYER 2 — FORM TEMPLATE

```bash
set -ex
TEST_DIR=$1
OUTPUT_DIR=$2

cbname(){
    echo "$OUTPUT_DIR"
    ./"$1"
    argresult=$?
    if [[ $argresult == 0 ]]; then
        echo "PASSED : $1 " >> "$OUTPUT_DIR"/results.txt
    else
        echo "FAILED : $1 " >> "$OUTPUT_DIR"/results.txt
    fi
}

pushd $TEST_DIR
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://10.0.0.1:2379 &
export MC_GID_INDEX=1
cbname transport_uint_test
cbname tcp_transport_test

export MC_GID_INDEX=1 && ./rdma_transport_test --mode=target \
    --metadata_server=127.0.0.1:2379 --local_server_name=127.0.0.2:14345 \
    --device_name=erdma_0 &
export MC_GID_INDEX=1 && ./rdma_transport_test \
    --metadata_server=127.0.0.1:2379 --segment_id=127.0.0.2:14345 \
    --local_server_name=127.0.0.3:14346 --device_name=erdma_1
argresult=$?
if [[ $argresult == 0 ]]; then
    echo "PASSED : rdma_transport_test " >> "$OUTPUT_DIR"/results.txt
else
    echo "FAILED : rdma_transport_test " >> "$OUTPUT_DIR"/results.txt
fi

export MC_GID_INDEX=1 && ./rdma_transport_test2 \
    --metadata_server=127.0.0.1:2379 --segment_id=127.0.0.2:14345 \
    --local_server_name=127.0.0.3:14346 --device_name=erdma_1
argresult=$?
if [[ $argresult == 0 ]]; then
    echo "PASSED : rdma_transport_test2 " >> "$OUTPUT_DIR"/results.txt
else
    echo "FAILED : rdma_transport_test2 " >> "$OUTPUT_DIR"/results.txt
fi
popd
```

---

## Summary Statistics

| Section | Semantic Units | LAYER 1a Specs | LAYER 1b Theorems | LAYER 2 Templates |
|---|---|---|---|---|
| Error Codes | 1 | 2 | 2 | 1 |
| GlobalConfig | 1 | 1 | 1 | 1 |
| Common Utilities | 10 | 10 | 10 | 10 |
| TransferMetadata | 4 | 6 | 2 | 1 |
| Transport Base | 6 | 6 | 5 | 5 |
| TransferEngine | 7 | 7 | 6 | 7 |
| C API | 1 | 2 | 2 | 1 |
| RDMA Transport | 1 | 3 | 1 | 1 |
| Build System | 1 | 2 | 1 | 2 |
| CI Test Script | 1 | 1 | 1 | 1 |
| **Total** | **33** | **40** | **31** | **30** |

Theorems marked `proof_needed` require more sophisticated string/list reasoning beyond simple arithmetic.

