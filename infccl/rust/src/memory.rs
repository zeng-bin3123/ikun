use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{check, Result, InfcclError};
use crate::ffi;

#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub addr: *mut c_void,
    pub length: usize,
    pub gpu: i32,
    pub location: String,
}

unsafe impl Send for MemoryRegion {}
unsafe impl Sync for MemoryRegion {}

pub struct MemoryPool {
    engine: *mut c_void,
    regions: Mutex<HashMap<usize, MemoryRegion>>,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    pub fn new(engine: *mut c_void) -> Self {
        Self { engine, regions: Mutex::new(HashMap::new()) }
    }

    pub fn register(&self, addr: *mut c_void, length: usize, gpu: i32, location: &str) -> Result<()> {
        let cloc = std::ffi::CString::new(location).map_err(|_| InfcclError::InvalidArg)?;
        let rc = unsafe { ffi::infccl_register_memory(self.engine, addr, length, cloc.as_ptr()) };
        check(rc)?;
        let key = addr as usize;
        let mut regions = self.regions.lock().unwrap();
        regions.insert(key, MemoryRegion {
            addr, length, gpu, location: location.to_string(),
        });
        Ok(())
    }

    pub fn unregister(&self, addr: *mut c_void) -> Result<()> {
        let rc = unsafe { ffi::infccl_unregister_memory(self.engine, addr) };
        check(rc)?;
        let key = addr as usize;
        let mut regions = self.regions.lock().unwrap();
        regions.remove(&key);
        Ok(())
    }

    pub fn find(&self, addr: *mut c_void) -> Option<MemoryRegion> {
        let a = addr as usize;
        let regions = self.regions.lock().unwrap();
        for region in regions.values() {
            let start = region.addr as usize;
            let end = start + region.length;
            if a >= start && a < end {
                return Some(region.clone());
            }
        }
        None
    }

    pub fn find_gpu(&self, addr: *mut c_void) -> Option<i32> {
        self.find(addr).map(|r| r.gpu)
    }

    pub fn registered_count(&self) -> usize {
        self.regions.lock().unwrap().len()
    }

    pub fn total_bytes(&self) -> usize {
        self.regions.lock().unwrap().values().map(|r| r.length).sum()
    }

    pub fn regions_on_gpu(&self, gpu: i32) -> Vec<MemoryRegion> {
        self.regions.lock().unwrap().values()
            .filter(|r| r.gpu == gpu)
            .cloned()
            .collect()
    }

    pub fn unregister_all(&self) -> Result<()> {
        let addrs: Vec<*mut c_void> = {
            self.regions.lock().unwrap().values().map(|r| r.addr).collect()
        };
        for addr in addrs {
            self.unregister(addr)?;
        }
        Ok(())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let _ = self.unregister_all();
    }
}

pub struct GpuBuffer {
    ptr: *mut c_void,
    len: usize,
    gpu: i32,
}

unsafe impl Send for GpuBuffer {}

impl GpuBuffer {
    pub fn alloc(gpu: i32, bytes: usize) -> Result<Self> {
        extern "C" {
            fn cudaSetDevice(dev: i32) -> i32;
            fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        }
        unsafe { cudaSetDevice(gpu); }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        if rc != 0 || ptr.is_null() {
            return Err(InfcclError::Cuda(rc));
        }
        Ok(Self { ptr, len: bytes, gpu })
    }

    pub fn ptr(&self) -> *mut c_void { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn gpu(&self) -> i32 { self.gpu }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            extern "C" {
                fn cudaSetDevice(dev: i32) -> i32;
                fn cudaFree(ptr: *mut c_void) -> i32;
            }
            unsafe {
                cudaSetDevice(self.gpu);
                cudaFree(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}
