"""
ikun_collectives.py — Python bindings for ikun GPU collective communications.

Layer 2 form template fill:
    prjname = ikun
    Obj = IkunCommunicator
    objtype = ctypes.c_void_p (maps to ikunCollComm_t)
    argname = sendbuf, recvbuf, count, dtype, op, root, comm

Layer 1 math specs:
    - State machine: UNINITIALIZED -> INITIALIZED -> CONNECTED
    - AllReduce postcondition: all ranks have identical reduced result
    - Broadcast postcondition: all ranks have root's data
    - P2P connectivity: bidirectional peer access required
"""

import ctypes
import ctypes.util
import os
import sys
import struct

# -------------------------------------------------------------------
# Constants (mirror C enum values)
# -------------------------------------------------------------------

IKUN_SUCCESS = 0
IKUN_ERR_INVALID_ARG = 1
IKUN_ERR_CUDA = 2
IKUN_ERR_NOT_INITIALIZED = 3
IKUN_ERR_P2P = 4
IKUN_ERR_INTERNAL = 5

IKUN_FLOAT32 = 0
IKUN_FLOAT16 = 1
IKUN_INT32 = 2
IKUN_INT8 = 3

IKUN_SUM = 0
IKUN_PROD = 1
IKUN_MAX = 2
IKUN_MIN = 3

# -------------------------------------------------------------------
# Library loading
# -------------------------------------------------------------------

def _find_library():
    """Find the ikun shared library."""
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'build'),
        os.path.join(os.path.dirname(__file__), '..'),
        os.path.join(os.path.dirname(__file__)),
        '/home/claude/ikun/build',
        '/home/claude/ikun',
    ]
    for path in search_paths:
        lib_path = os.path.join(path, 'libikun_collectives.so')
        if os.path.exists(lib_path):
            return lib_path
    raise RuntimeError(
        "Could not find libikun_collectives.so. "
        "Searched: " + ", ".join(search_paths)
    )


def _load_library():
    """Load and configure the ikun library."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)

    # ikunCommInit
    lib.ikunCommInit.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # comm
        ctypes.c_int,                      # rank
        ctypes.c_int,                      # world_size
        ctypes.c_int,                      # device_id
    ]
    lib.ikunCommInit.restype = ctypes.c_int

    # ikunCommConnect
    lib.ikunCommConnect.argtypes = [
        ctypes.c_void_p,                   # comm
        ctypes.POINTER(ctypes.c_int),      # peer_devices
        ctypes.POINTER(ctypes.c_void_p),   # peer_staging_bufs
    ]
    lib.ikunCommConnect.restype = ctypes.c_int

    # ikunCommDestroy
    lib.ikunCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ikunCommDestroy.restype = ctypes.c_int

    # ikunAllReduce
    lib.ikunAllReduce.argtypes = [
        ctypes.c_void_p,  # sendbuf
        ctypes.c_void_p,  # recvbuf
        ctypes.c_size_t,  # count
        ctypes.c_int,     # datatype
        ctypes.c_int,     # op
        ctypes.c_void_p,  # comm
    ]
    lib.ikunAllReduce.restype = ctypes.c_int

    # ikunBroadcast
    lib.ikunBroadcast.argtypes = [
        ctypes.c_void_p,  # sendbuf
        ctypes.c_void_p,  # recvbuf
        ctypes.c_size_t,  # count
        ctypes.c_int,     # datatype
        ctypes.c_int,     # root
        ctypes.c_void_p,  # comm
    ]
    lib.ikunBroadcast.restype = ctypes.c_int

    # ikunReduce
    lib.ikunReduce.argtypes = [
        ctypes.c_void_p,  # sendbuf
        ctypes.c_void_p,  # recvbuf
        ctypes.c_size_t,  # count
        ctypes.c_int,     # datatype
        ctypes.c_int,     # op
        ctypes.c_int,     # root
        ctypes.c_void_p,  # comm
    ]
    lib.ikunReduce.restype = ctypes.c_int

    # ikunAllGather
    lib.ikunAllGather.argtypes = [
        ctypes.c_void_p,  # sendbuf
        ctypes.c_void_p,  # recvbuf
        ctypes.c_size_t,  # count
        ctypes.c_int,     # datatype
        ctypes.c_void_p,  # comm
    ]
    lib.ikunAllGather.restype = ctypes.c_int

    # ikunReduceScatter
    lib.ikunReduceScatter.argtypes = [
        ctypes.c_void_p,  # sendbuf
        ctypes.c_void_p,  # recvbuf
        ctypes.c_size_t,  # count
        ctypes.c_int,     # datatype
        ctypes.c_int,     # op
        ctypes.c_void_p,  # comm
    ]
    lib.ikunReduceScatter.restype = ctypes.c_int

    # ikunTestP2P
    lib.ikunTestP2P.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ikunTestP2P.restype = ctypes.c_int

    # ikunGetDeviceCount
    lib.ikunGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ikunGetDeviceCount.restype = ctypes.c_int

    return lib


_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# -------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------

_ERROR_STRINGS = {
    IKUN_SUCCESS: "success",
    IKUN_ERR_INVALID_ARG: "invalid argument",
    IKUN_ERR_CUDA: "CUDA error",
    IKUN_ERR_NOT_INITIALIZED: "communicator not initialized/connected",
    IKUN_ERR_P2P: "P2P access not available",
    IKUN_ERR_INTERNAL: "internal error",
}

class IkunError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f"ikun error {code}: {_ERROR_STRINGS.get(code, 'unknown')}")

def _check(code):
    if code != IKUN_SUCCESS:
        raise IkunError(code)


# -------------------------------------------------------------------
# High-level API
# -------------------------------------------------------------------

def get_device_count():
    """Return the number of available GPUs."""
    lib = _get_lib()
    count = ctypes.c_int(0)
    _check(lib.ikunGetDeviceCount(ctypes.byref(count)))
    return count.value


def test_p2p(dev_a, dev_b):
    """Test if P2P access is possible between two GPUs."""
    lib = _get_lib()
    can_access = ctypes.c_int(0)
    _check(lib.ikunTestP2P(dev_a, dev_b, ctypes.byref(can_access)))
    return bool(can_access.value)


class IkunCommunicator:
    """
    A communicator for GPU collective operations.

    Layer 1 state machine:
        __init__  → INITIALIZED
        connect() → CONNECTED
        close()   → UNINITIALIZED

    Usage:
        comms = IkunCommunicator.create_group(device_ids=[0,1,2,3])
        for c in comms:
            c.connect(comms)
        # ... run collectives ...
        for c in comms:
            c.close()
    """

    def __init__(self, rank, world_size, device_id):
        self.lib = _get_lib()
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self._handle = ctypes.c_void_p(0)
        _check(self.lib.ikunCommInit(
            ctypes.byref(self._handle), rank, world_size, device_id
        ))
        self._connected = False

    def connect(self, all_comms):
        """
        Enable P2P access to all peers.

        all_comms: list of IkunCommunicator, one per rank
        """
        peer_devices = (ctypes.c_int * self.world_size)()
        peer_staging = (ctypes.c_void_p * self.world_size)()

        for i, c in enumerate(all_comms):
            peer_devices[i] = c.device_id
            # Read the staging_buf pointer from the C struct
            # The struct layout: rank(4) + world_size(4) + device_id(4) + state(4)
            # + peer_bufs(8*MAX_GPUS) + staging_buf(8) ...
            # Offset to staging_buf = 16 + 8*8 = 80
            handle_addr = c._handle.value
            # Use ctypes to read the staging pointer
            staging_offset = 16 + 8 * 8  # 4+4+4+4 + 8*IKUN_MAX_GPUS(=8)
            staging_ptr_bytes = (ctypes.c_char * 8).from_address(
                handle_addr + staging_offset
            )
            peer_staging[i] = ctypes.c_void_p(
                struct.unpack('P', bytes(staging_ptr_bytes))[0]
            )

        _check(self.lib.ikunCommConnect(
            self._handle, peer_devices, peer_staging
        ))
        self._connected = True

    def allreduce(self, sendbuf, recvbuf, count, dtype=IKUN_FLOAT32, op=IKUN_SUM):
        """
        All-reduce: reduce across all ranks, result available on all.

        sendbuf, recvbuf: device pointers (int or ctypes.c_void_p)
        count: number of elements
        """
        _check(self.lib.ikunAllReduce(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            count, dtype, op,
            self._handle
        ))

    def broadcast(self, sendbuf, recvbuf, count, root=0, dtype=IKUN_FLOAT32):
        """Broadcast from root to all ranks."""
        _check(self.lib.ikunBroadcast(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            count, dtype, root,
            self._handle
        ))

    def reduce(self, sendbuf, recvbuf, count, root=0,
               dtype=IKUN_FLOAT32, op=IKUN_SUM):
        """Reduce to root rank."""
        _check(self.lib.ikunReduce(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            count, dtype, op, root,
            self._handle
        ))

    def allgather(self, sendbuf, recvbuf, count, dtype=IKUN_FLOAT32):
        """All-gather: gather data from all ranks to all ranks."""
        _check(self.lib.ikunAllGather(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            count, dtype,
            self._handle
        ))

    def reduce_scatter(self, sendbuf, recvbuf, count,
                       dtype=IKUN_FLOAT32, op=IKUN_SUM):
        """Reduce-scatter: reduce, then scatter slices to each rank."""
        _check(self.lib.ikunReduceScatter(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            count, dtype, op,
            self._handle
        ))

    def close(self):
        """Destroy the communicator."""
        if self._handle.value:
            self.lib.ikunCommDestroy(self._handle)
            self._handle = ctypes.c_void_p(0)
            self._connected = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def create_group(device_ids):
        """
        Create a communicator group for the given GPU device IDs.

        Returns a list of IkunCommunicator, one per rank.
        """
        world_size = len(device_ids)
        comms = []
        for rank, dev in enumerate(device_ids):
            comms.append(IkunCommunicator(rank, world_size, dev))
        return comms


# -------------------------------------------------------------------
# Convenience: full P2P test matrix
# -------------------------------------------------------------------

def test_all_p2p_pairs(device_ids=None):
    """
    Test P2P connectivity between all GPU pairs.

    Returns:
        ok_pairs: list of (dev_a, dev_b) pairs with P2P access
        fail_pairs: list of (dev_a, dev_b) pairs without P2P access
    """
    if device_ids is None:
        n = get_device_count()
        device_ids = list(range(n))

    ok_pairs = []
    fail_pairs = []
    for a in device_ids:
        for b in device_ids:
            if a == b:
                continue
            if test_p2p(a, b):
                ok_pairs.append((a, b))
            else:
                fail_pairs.append((a, b))
    return ok_pairs, fail_pairs


if __name__ == "__main__":
    print(f"ikun collectives library loaded from: {_find_library()}")
    n = get_device_count()
    print(f"GPU count: {n}")
    ok, fail = test_all_p2p_pairs()
    print(f"P2P OK pairs: {ok}")
    print(f"P2P fail pairs: {fail}")

