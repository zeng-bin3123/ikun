use std::ffi::c_void;

pub const INVALID_BATCH: u64 = u64::MAX;

pub type SegmentId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BatchId(pub u64);

impl BatchId {
    #[inline]
    pub fn is_invalid(self) -> bool { self.0 == INVALID_BATCH }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Opcode {
    Read = 0,
    Write = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StatusCode {
    Waiting = 0,
    Pending = 1,
    Completed = 4,
    Timeout = 5,
    Failed = 6,
}

impl StatusCode {
    pub fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::Waiting,
            1 => Self::Pending,
            4 => Self::Completed,
            5 => Self::Timeout,
            6 => Self::Failed,
            _ => Self::Failed,
        }
    }
    pub fn is_done(self) -> bool { matches!(self, Self::Completed | Self::Failed | Self::Timeout) }
    pub fn is_ok(self) -> bool { matches!(self, Self::Completed) }
}

#[derive(Debug, Clone)]
pub struct TransferStatus {
    pub code: StatusCode,
    pub transferred_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct PeerTransferRequest {
    pub src: *mut c_void,
    pub src_gpu: i32,
    pub dst: *mut c_void,
    pub dst_gpu: i32,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct BiV100Caps {
    pub warp_size: i32,
    pub sm_count: i32,
    pub fp64_works: bool,
    pub hbm_bw_gbps: f32,
    pub peer_bw_gbps: f32,
    pub peer_latency_us: f32,
    pub peer_memcpy_needs_src_device: bool,
    pub peer_memcpy_needs_blocking_stream: bool,
}
