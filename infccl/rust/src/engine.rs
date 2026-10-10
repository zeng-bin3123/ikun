use std::ffi::{c_void, CString};
use std::ptr;

use crate::error::{check, InfcclError, Result};
use crate::ffi;
use crate::types::*;

pub struct TransferEngine {
    engine: *mut c_void,
    transport: *mut c_void,
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new(ndev: i32, devlist: &[i32]) -> Result<Self> {
        let engine = unsafe { ffi::infccl_create_engine(ndev, devlist.as_ptr()) };
        if engine.is_null() {
            return Err(InfcclError::InitFailed("create_engine returned null".into()));
        }
        let proto = CString::new("peer").unwrap();
        let transport = unsafe { ffi::infccl_install_transport(engine, proto.as_ptr()) };
        if transport.is_null() {
            unsafe { ffi::infccl_destroy_engine(engine) };
            return Err(InfcclError::InitFailed("install_transport returned null".into()));
        }
        Ok(Self { engine, transport })
    }

    pub fn open_segment(&self, name: &str) -> Result<SegmentId> {
        let cname = CString::new(name).map_err(|_| InfcclError::InvalidArg)?;
        let id = unsafe { ffi::infccl_open_segment(self.engine, cname.as_ptr()) };
        Ok(id)
    }

    pub fn register_memory(&self, addr: *mut c_void, length: usize, location: &str) -> Result<()> {
        let cloc = CString::new(location).map_err(|_| InfcclError::InvalidArg)?;
        let rc = unsafe { ffi::infccl_register_memory(self.engine, addr, length, cloc.as_ptr()) };
        check(rc)
    }

    pub fn unregister_memory(&self, addr: *mut c_void) -> Result<()> {
        let rc = unsafe { ffi::infccl_unregister_memory(self.engine, addr) };
        check(rc)
    }

    pub fn allocate_batch(&self, size: usize) -> Result<BatchId> {
        let id = unsafe { ffi::infccl_alloc_batch(self.transport, size) };
        if id == INVALID_BATCH {
            return Err(InfcclError::BatchInvalid);
        }
        Ok(BatchId(id))
    }

    pub fn submit_peer_transfer(
        &self, batch: BatchId,
        src: *mut c_void, src_gpu: i32,
        dst: *mut c_void, dst_gpu: i32,
        length: usize,
    ) -> Result<()> {
        let rc = unsafe {
            ffi::infccl_submit_peer_transfer(
                self.transport, batch.0, src, src_gpu, dst, dst_gpu, length)
        };
        check(rc)
    }

    pub fn submit_peer_transfers(&self, batch: BatchId, reqs: &[PeerTransferRequest]) -> Result<()> {
        for req in reqs {
            self.submit_peer_transfer(
                batch, req.src, req.src_gpu, req.dst, req.dst_gpu, req.length)?;
        }
        Ok(())
    }

    pub fn wait_batch(&self, batch: BatchId, timeout_ms: i32) -> Result<()> {
        let rc = unsafe { ffi::infccl_wait_batch(self.transport, batch.0, timeout_ms) };
        check(rc)
    }

    pub fn get_transfer_status(&self, batch: BatchId, task_id: usize) -> Result<TransferStatus> {
        let mut raw = ffi::infccl_transfer_status {
            status: 0,
            transferred_bytes: 0,
        };
        let rc = unsafe {
            ffi::infccl_get_transfer_status(self.transport, batch.0, task_id, &mut raw)
        };
        check(rc)?;
        Ok(TransferStatus {
            code: StatusCode::from_raw(raw.status),
            transferred_bytes: raw.transferred_bytes,
        })
    }

    pub fn free_batch(&self, batch: BatchId) -> Result<()> {
        let rc = unsafe { ffi::infccl_free_batch(self.transport, batch.0) };
        check(rc)
    }

    pub fn transfer_sync(
        &self,
        src: *mut c_void, src_gpu: i32,
        dst: *mut c_void, dst_gpu: i32,
        length: usize,
    ) -> Result<()> {
        let batch = self.allocate_batch(1)?;
        self.submit_peer_transfer(batch, src, src_gpu, dst, dst_gpu, length)?;
        self.wait_batch(batch, 10000)?;
        self.free_batch(batch)?;
        Ok(())
    }

    pub fn raw_engine(&self) -> *mut c_void { self.engine }
    pub fn raw_transport(&self) -> *mut c_void { self.transport }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        if !self.engine.is_null() {
            unsafe { ffi::infccl_destroy_engine(self.engine) };
            self.engine = ptr::null_mut();
            self.transport = ptr::null_mut();
        }
    }
}

pub fn probe_caps(gpu: i32) -> Result<BiV100Caps> {
    let mut raw = ffi::infccl_bi_v100_caps {
        warp_size: 0,
        sm_count: 0,
        fp64_works: 0,
        hbm_bw_gbps: 0.0,
        peer_bw_gbps: 0.0,
        peer_latency_us: 0.0,
        peer_memcpy_needs_src_device: 0,
        peer_memcpy_needs_blocking_stream: 0,
    };
    let rc = unsafe { ffi::infccl_probe_caps(gpu, &mut raw) };
    check(rc)?;
    Ok(BiV100Caps {
        warp_size: raw.warp_size,
        sm_count: raw.sm_count,
        fp64_works: raw.fp64_works != 0,
        hbm_bw_gbps: raw.hbm_bw_gbps,
        peer_bw_gbps: raw.peer_bw_gbps,
        peer_latency_us: raw.peer_latency_us,
        peer_memcpy_needs_src_device: raw.peer_memcpy_needs_src_device != 0,
        peer_memcpy_needs_blocking_stream: raw.peer_memcpy_needs_blocking_stream != 0,
    })
}

pub fn print_caps(gpu: i32) {
    unsafe { ffi::infccl_print_caps(gpu) };
}
