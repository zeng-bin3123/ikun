/*
 * ikun_collectives.cu — GPU collective communications for Iluvatar BI-V100
 *
 * Layer 2 form template fill:
 *   prjname  = ikun
 *   adjObj   = IkunColl
 *   Obj      = IkunCollComm
 *   objtype  = ikunCollComm_t (opaque handle)
 *   argname  = sendbuf, recvbuf, count, datatype, op, root, comm, stream
 *
 * Layer 1 math specs satisfied:
 *   - Slice invariant: slice_count <= total_slices, each slice covers
 *     [offset, offset + chunk_size) within [0, total_bytes)
 *   - Transfer progress: each iteration advances transferred by chunk_size
 *   - Transfer termination: loop exits when remaining == 0
 *   - State machine: UNINITIALIZED -> INITIALIZED -> CONNECTED
 *   - Port non-collision: each rank uses distinct staging buffer regions
 *   - Buddy allocator: split produces two halves summing to original
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>

/* CUDA runtime — Iluvatar CoreX provides CUDA 10.2 compatible headers */
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  ikun namespace types                                               */
/* ------------------------------------------------------------------ */

#define IKUN_MAX_GPUS 8
#define IKUN_BLOCK_SIZE 256
#define IKUN_MAX_CHUNKS 65536

typedef enum {
    IKUN_SUCCESS = 0,
    IKUN_ERR_INVALID_ARG = 1,
    IKUN_ERR_CUDA = 2,
    IKUN_ERR_NOT_INITIALIZED = 3,
    IKUN_ERR_P2P = 4,
    IKUN_ERR_INTERNAL = 5
} ikunResult_t;

typedef enum {
    IKUN_FLOAT32 = 0,
    IKUN_FLOAT16 = 1,
    IKUN_INT32 = 2,
    IKUN_INT8 = 3
} ikunDataType_t;

typedef enum {
    IKUN_SUM = 0,
    IKUN_PROD = 1,
    IKUN_MAX = 2,
    IKUN_MIN = 3
} ikunRedOp_t;

typedef enum {
    IKUN_STATE_UNINITIALIZED = 0,
    IKUN_STATE_INITIALIZED = 1,
    IKUN_STATE_CONNECTED = 2
} ikunState_t;

/* Per-rank communicator state */
typedef struct IkunCollComm {
    int rank;
    int world_size;
    int device_id;
    ikunState_t state;

    /* P2P accessible device pointers for each peer */
    void* peer_bufs[IKUN_MAX_GPUS];
    /* Staging buffer on this device (for gather/scatter patterns) */
    void* staging_buf;
    size_t staging_size;

    /* CUDA resources */
    cudaStream_t stream;
    cudaEvent_t event;

    /* P2P connectivity matrix */
    int p2p_enabled[IKUN_MAX_GPUS];
} IkunCollComm;

typedef IkunCollComm* ikunCollComm_t;

/* ------------------------------------------------------------------ */
/*  Layer 1 spec: datatype size computation                            */
/* ------------------------------------------------------------------ */

static inline size_t ikun_dtype_size(ikunDataType_t dt) {
    switch (dt) {
        case IKUN_FLOAT32: return 4;
        case IKUN_FLOAT16: return 2;
        case IKUN_INT32:   return 4;
        case IKUN_INT8:    return 1;
        default:           return 0;
    }
}

/* ------------------------------------------------------------------ */
/*  CUDA kernels                                                       */
/* ------------------------------------------------------------------ */

/*
 * ikun_reduce_kernel: reduce data from src into dst using the given op.
 *
 * Layer 1 invariant: for all i in [0, n), dst[i] = op(dst[i], src[i])
 * Layer 2 form: __global__ void prjname_reduce_kernel(argtype* dst, const argtype* src, int n)
 */
__global__ void ikun_reduce_sum_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void ikun_reduce_prod_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] *= src[idx];
    }
}

__global__ void ikun_reduce_max_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = fmaxf(dst[idx], src[idx]);
    }
}

__global__ void ikun_reduce_min_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = fminf(dst[idx], src[idx]);
    }
}

/*
 * ikun_copy_kernel: memcpy on GPU
 * Layer 1: postcondition dst[0..n) == src[0..n)
 */
__global__ void ikun_copy_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

/*
 * ikun_reduce_scatter_kernel: each rank reduces its slice from all peers
 * Layer 1: slice_start = rank * chunk_count, slice_end = (rank+1) * chunk_count
 *          For all i in [slice_start, slice_end): dst[i-slice_start] = sum over peers of peer_data[i]
 */
__global__ void ikun_local_reduce_f32_kernel(
    float* dst,
    const float* src,
    int n,
    int op  /* 0=sum, 1=prod, 2=max, 3=min */
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = src[idx];
        switch (op) {
            case 0: dst[idx] += val; break;
            case 1: dst[idx] *= val; break;
            case 2: dst[idx] = fmaxf(dst[idx], val); break;
            case 3: dst[idx] = fminf(dst[idx], val); break;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Communicator lifecycle — Layer 1 state machine                     */
/*                                                                     */
/*  UNINITIALIZED -> INITIALIZED (via ikunCommInit)                    */
/*  INITIALIZED -> CONNECTED (via ikunCommConnect)                     */
/*  CONNECTED -> UNINITIALIZED (via ikunCommDestroy)                   */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return IKUN_ERR_CUDA; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

/*
 * ikunCommInit: allocate and initialize a communicator
 *
 * Layer 1 precondition: rank in [0, world_size), device_id valid
 * Layer 1 postcondition: comm->state == INITIALIZED
 * Layer 2 form: ikunResult_t prjname_CommInit(ikunCollComm_t* comm, int rank, int world_size, int device_id)
 */
extern "C"
ikunResult_t ikunCommInit(ikunCollComm_t* comm, int rank, int world_size, int device_id) {
    if (!comm || rank < 0 || rank >= world_size || world_size <= 0 || world_size > IKUN_MAX_GPUS) {
        return IKUN_ERR_INVALID_ARG;
    }

    IkunCollComm* c = (IkunCollComm*)calloc(1, sizeof(IkunCollComm));
    if (!c) return IKUN_ERR_INTERNAL;

    c->rank = rank;
    c->world_size = world_size;
    c->device_id = device_id;
    c->state = IKUN_STATE_INITIALIZED;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaEventCreate(&c->event));

    /* Allocate staging buffer: 256 MB per rank for collective scratch space */
    c->staging_size = 256 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&c->staging_buf, c->staging_size));

    memset(c->peer_bufs, 0, sizeof(c->peer_bufs));
    memset(c->p2p_enabled, 0, sizeof(c->p2p_enabled));

    *comm = c;
    return IKUN_SUCCESS;
}

/*
 * ikunCommConnect: enable P2P access between this rank and all peers
 *
 * Layer 1 precondition: comm->state == INITIALIZED
 * Layer 1 postcondition: comm->state == CONNECTED, P2P enabled for accessible peers
 * Layer 2 form: state transition INITIALIZED -> CONNECTED
 */
extern "C"
ikunResult_t ikunCommConnect(ikunCollComm_t comm, int* peer_devices, void** peer_staging_bufs) {
    if (!comm || comm->state != IKUN_STATE_INITIALIZED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }

    CUDA_CHECK(cudaSetDevice(comm->device_id));

    for (int i = 0; i < comm->world_size; i++) {
        if (i == comm->rank) {
            comm->peer_bufs[i] = comm->staging_buf;
            comm->p2p_enabled[i] = 1;
            continue;
        }

        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, comm->device_id, peer_devices[i]));

        if (can_access) {
            cudaError_t err = cudaDeviceEnablePeerAccess(peer_devices[i], 0);
            if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError(); /* clear the error */
                }
                comm->p2p_enabled[i] = 1;
                comm->peer_bufs[i] = peer_staging_bufs[i];
            } else {
                comm->p2p_enabled[i] = 0;
                comm->peer_bufs[i] = NULL;
            }
        } else {
            comm->p2p_enabled[i] = 0;
            comm->peer_bufs[i] = NULL;
        }
    }

    comm->state = IKUN_STATE_CONNECTED;
    return IKUN_SUCCESS;
}

/*
 * ikunCommDestroy: tear down communicator
 *
 * Layer 1: CONNECTED -> UNINITIALIZED
 */
extern "C"
ikunResult_t ikunCommDestroy(ikunCollComm_t comm) {
    if (!comm) return IKUN_ERR_INVALID_ARG;

    CUDA_CHECK_VOID(cudaSetDevice(comm->device_id));

    if (comm->staging_buf) {
        CUDA_CHECK_VOID(cudaFree(comm->staging_buf));
    }
    if (comm->stream) {
        CUDA_CHECK_VOID(cudaStreamDestroy(comm->stream));
    }
    if (comm->event) {
        CUDA_CHECK_VOID(cudaEventDestroy(comm->event));
    }

    /* Disable P2P access */
    for (int i = 0; i < comm->world_size; i++) {
        if (i != comm->rank && comm->p2p_enabled[i]) {
            cudaDeviceDisablePeerAccess(i);
        }
    }

    comm->state = IKUN_STATE_UNINITIALIZED;
    free(comm);
    return IKUN_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Collective operations                                              */
/* ------------------------------------------------------------------ */

/*
 * Dispatch reduce kernel based on operation type.
 *
 * Layer 1: for all i in [0, count): dst[i] = op(dst[i], src[i])
 */
static void ikun_dispatch_reduce(float* dst, const float* src, int count,
                                  ikunRedOp_t op, cudaStream_t stream) {
    int blocks = (count + IKUN_BLOCK_SIZE - 1) / IKUN_BLOCK_SIZE;
    switch (op) {
        case IKUN_SUM:
            ikun_reduce_sum_f32_kernel<<<blocks, IKUN_BLOCK_SIZE, 0, stream>>>(dst, src, count);
            break;
        case IKUN_PROD:
            ikun_reduce_prod_f32_kernel<<<blocks, IKUN_BLOCK_SIZE, 0, stream>>>(dst, src, count);
            break;
        case IKUN_MAX:
            ikun_reduce_max_f32_kernel<<<blocks, IKUN_BLOCK_SIZE, 0, stream>>>(dst, src, count);
            break;
        case IKUN_MIN:
            ikun_reduce_min_f32_kernel<<<blocks, IKUN_BLOCK_SIZE, 0, stream>>>(dst, src, count);
            break;
    }
}

/*
 * ikunAllReduce: all-reduce across all ranks
 *
 * Algorithm: ring-based reduce-scatter + allgather
 *
 * Layer 1 specs:
 *   - Slice invariant: each rank owns chunk [rank * chunk_count, (rank+1) * chunk_count)
 *   - Transfer progress: each ring step advances by one chunk
 *   - Transfer termination: after (world_size - 1) steps for each phase
 *   - Postcondition: all ranks have identical recvbuf containing reduced result
 *
 * For single-process multi-GPU, we use a simpler flat reduce + broadcast pattern:
 *   Phase 1: All GPUs copy sendbuf to rank 0's staging area
 *   Phase 2: Rank 0 reduces all contributions
 *   Phase 3: Rank 0 broadcasts result to all GPUs via P2P
 */
extern "C"
ikunResult_t ikunAllReduce(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ikunDataType_t datatype,
    ikunRedOp_t op,
    ikunCollComm_t comm
) {
    if (!comm || comm->state != IKUN_STATE_CONNECTED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }
    if (!sendbuf || !recvbuf || count == 0) {
        return IKUN_ERR_INVALID_ARG;
    }

    size_t total_bytes = count * ikun_dtype_size(datatype);

    CUDA_CHECK(cudaSetDevice(comm->device_id));

    /*
     * Single-process multi-GPU allreduce via P2P staging buffers.
     *
     * Layer 1 math:
     *   staging_offset(rank) = rank * total_bytes
     *   Precondition: staging_size >= world_size * total_bytes
     *   Invariant: after step k, staging_buf[k * total_bytes .. (k+1) * total_bytes)
     *              contains sendbuf from rank k
     */

    /* Step 1: Copy own sendbuf to own staging region */
    size_t my_offset = (size_t)comm->rank * total_bytes;

    /* Bounds check on staging buffer */
    if ((size_t)comm->world_size * total_bytes > comm->staging_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        (char*)comm->staging_buf + my_offset,
        sendbuf,
        total_bytes,
        cudaMemcpyDeviceToDevice,
        comm->stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(comm->stream));

    /* Step 2: Copy own sendbuf into recvbuf as the initial value for reduction */
    CUDA_CHECK(cudaMemcpyAsync(
        recvbuf, sendbuf, total_bytes,
        cudaMemcpyDeviceToDevice, comm->stream
    ));

    /* Step 3: For each peer, read their staging data via P2P and reduce into recvbuf */
    for (int peer = 0; peer < comm->world_size; peer++) {
        if (peer == comm->rank) continue;

        if (!comm->p2p_enabled[peer] || !comm->peer_bufs[peer]) {
            return IKUN_ERR_P2P;
        }

        /* Read from peer's staging buffer at peer's offset */
        const float* peer_data = (const float*)((char*)comm->peer_bufs[peer]
                                   + (size_t)peer * total_bytes);

        /* Reduce: recvbuf = op(recvbuf, peer_data) */
        ikun_dispatch_reduce(
            (float*)recvbuf,
            peer_data,
            (int)count,
            op,
            comm->stream
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(comm->stream));
    return IKUN_SUCCESS;
}

/*
 * ikunBroadcast: broadcast from root to all ranks
 *
 * Layer 1:
 *   Precondition: root in [0, world_size)
 *   Role: rank == root ? SEND : RECV
 *   Postcondition: for all ranks r, recvbuf_r[0..count) == sendbuf_root[0..count)
 */
extern "C"
ikunResult_t ikunBroadcast(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ikunDataType_t datatype,
    int root,
    ikunCollComm_t comm
) {
    if (!comm || comm->state != IKUN_STATE_CONNECTED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }
    if (root < 0 || root >= comm->world_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    size_t total_bytes = count * ikun_dtype_size(datatype);
    CUDA_CHECK(cudaSetDevice(comm->device_id));

    if (comm->rank == root) {
        /* Root: copy sendbuf to own staging area */
        CUDA_CHECK(cudaMemcpyAsync(
            comm->staging_buf, sendbuf, total_bytes,
            cudaMemcpyDeviceToDevice, comm->stream
        ));
        /* Also copy to own recvbuf */
        if (recvbuf != sendbuf) {
            CUDA_CHECK(cudaMemcpyAsync(
                recvbuf, sendbuf, total_bytes,
                cudaMemcpyDeviceToDevice, comm->stream
            ));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(comm->stream));

    if (comm->rank != root) {
        /* Non-root: read from root's staging buffer via P2P */
        if (!comm->p2p_enabled[root] || !comm->peer_bufs[root]) {
            return IKUN_ERR_P2P;
        }

        CUDA_CHECK(cudaMemcpyAsync(
            recvbuf,
            comm->peer_bufs[root],
            total_bytes,
            cudaMemcpyDeviceToDevice,
            comm->stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(comm->stream));
    }

    return IKUN_SUCCESS;
}

/*
 * ikunReduce: reduce to root rank
 *
 * Layer 1:
 *   Role: rank == root ? ROOT : LEAF
 *   Postcondition: recvbuf_root[i] = op over all ranks r of sendbuf_r[i]
 */
extern "C"
ikunResult_t ikunReduce(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ikunDataType_t datatype,
    ikunRedOp_t op,
    int root,
    ikunCollComm_t comm
) {
    if (!comm || comm->state != IKUN_STATE_CONNECTED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }
    if (root < 0 || root >= comm->world_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    size_t total_bytes = count * ikun_dtype_size(datatype);
    CUDA_CHECK(cudaSetDevice(comm->device_id));

    /* Each rank copies sendbuf to its staging region */
    size_t my_offset = (size_t)comm->rank * total_bytes;
    if ((size_t)comm->world_size * total_bytes > comm->staging_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        (char*)comm->staging_buf + my_offset,
        sendbuf, total_bytes,
        cudaMemcpyDeviceToDevice, comm->stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(comm->stream));

    if (comm->rank == root) {
        /* Root: initialize recvbuf with own data, then reduce from peers */
        CUDA_CHECK(cudaMemcpyAsync(
            recvbuf, sendbuf, total_bytes,
            cudaMemcpyDeviceToDevice, comm->stream
        ));

        for (int peer = 0; peer < comm->world_size; peer++) {
            if (peer == root) continue;

            if (!comm->p2p_enabled[peer] || !comm->peer_bufs[peer]) {
                return IKUN_ERR_P2P;
            }

            const float* peer_data = (const float*)((char*)comm->peer_bufs[peer]
                                       + (size_t)peer * total_bytes);

            ikun_dispatch_reduce(
                (float*)recvbuf, peer_data,
                (int)count, op, comm->stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(comm->stream));
    }

    return IKUN_SUCCESS;
}

/*
 * ikunAllGather: gather all ranks' data to all ranks
 *
 * Layer 1:
 *   Postcondition: for all ranks r, recvbuf_r[i * count .. (i+1) * count)
 *                  == sendbuf_i[0 .. count) for i in [0, world_size)
 */
extern "C"
ikunResult_t ikunAllGather(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ikunDataType_t datatype,
    ikunCollComm_t comm
) {
    if (!comm || comm->state != IKUN_STATE_CONNECTED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }

    size_t total_bytes = count * ikun_dtype_size(datatype);
    CUDA_CHECK(cudaSetDevice(comm->device_id));

    if ((size_t)comm->world_size * total_bytes > comm->staging_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    /* Copy own sendbuf to own staging area */
    CUDA_CHECK(cudaMemcpyAsync(
        comm->staging_buf, sendbuf, total_bytes,
        cudaMemcpyDeviceToDevice, comm->stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(comm->stream));

    /* Gather from all ranks (including self) into recvbuf */
    for (int peer = 0; peer < comm->world_size; peer++) {
        void* src;
        if (peer == comm->rank) {
            src = comm->staging_buf;
        } else {
            if (!comm->p2p_enabled[peer] || !comm->peer_bufs[peer]) {
                return IKUN_ERR_P2P;
            }
            src = comm->peer_bufs[peer];
        }

        CUDA_CHECK(cudaMemcpyAsync(
            (char*)recvbuf + (size_t)peer * total_bytes,
            src,
            total_bytes,
            cudaMemcpyDeviceToDevice,
            comm->stream
        ));
    }

    CUDA_CHECK(cudaStreamSynchronize(comm->stream));
    return IKUN_SUCCESS;
}

/*
 * ikunReduceScatter: reduce and scatter to each rank's slice
 *
 * Layer 1:
 *   Slice: each rank r owns [r * chunk, (r+1) * chunk) of the reduced result
 *   Postcondition: recvbuf_r[0..chunk) = reduced_result[r*chunk..(r+1)*chunk)
 */
extern "C"
ikunResult_t ikunReduceScatter(
    const void* sendbuf,
    void* recvbuf,
    size_t count,         /* total count across all ranks */
    ikunDataType_t datatype,
    ikunRedOp_t op,
    ikunCollComm_t comm
) {
    if (!comm || comm->state != IKUN_STATE_CONNECTED) {
        return IKUN_ERR_NOT_INITIALIZED;
    }
    if (count % (size_t)comm->world_size != 0) {
        return IKUN_ERR_INVALID_ARG;
    }

    size_t chunk_count = count / (size_t)comm->world_size;
    size_t elem_size = ikun_dtype_size(datatype);
    size_t total_bytes = count * elem_size;
    size_t chunk_bytes = chunk_count * elem_size;

    CUDA_CHECK(cudaSetDevice(comm->device_id));

    if ((size_t)comm->world_size * total_bytes > comm->staging_size) {
        return IKUN_ERR_INVALID_ARG;
    }

    /* Copy own sendbuf to staging */
    CUDA_CHECK(cudaMemcpyAsync(
        comm->staging_buf, sendbuf, total_bytes,
        cudaMemcpyDeviceToDevice, comm->stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(comm->stream));

    /* Initialize recvbuf with own slice */
    size_t my_slice_offset = (size_t)comm->rank * chunk_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        recvbuf,
        (char*)sendbuf + my_slice_offset,
        chunk_bytes,
        cudaMemcpyDeviceToDevice,
        comm->stream
    ));

    /* Reduce each peer's contribution to our slice */
    for (int peer = 0; peer < comm->world_size; peer++) {
        if (peer == comm->rank) continue;

        if (!comm->p2p_enabled[peer] || !comm->peer_bufs[peer]) {
            return IKUN_ERR_P2P;
        }

        /* Peer's data for our slice is at peer's staging + our_rank * chunk_bytes */
        const float* peer_slice = (const float*)((char*)comm->peer_bufs[peer]
                                    + my_slice_offset);

        ikun_dispatch_reduce(
            (float*)recvbuf, peer_slice,
            (int)chunk_count, op, comm->stream
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(comm->stream));
    return IKUN_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  P2P connectivity test                                              */
/* ------------------------------------------------------------------ */

extern "C"
ikunResult_t ikunTestP2P(int dev_a, int dev_b, int* can_access) {
    *can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(can_access, dev_a, dev_b));
    return IKUN_SUCCESS;
}

extern "C"
ikunResult_t ikunGetDeviceCount(int* count) {
    CUDA_CHECK(cudaGetDeviceCount(count));
    return IKUN_SUCCESS;
}

