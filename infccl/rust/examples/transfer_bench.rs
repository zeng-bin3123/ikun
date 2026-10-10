use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

fn main() {
    let ndev = 4i32;
    let devs = [0i32, 1, 2, 3];

    let engine = infccl::TransferEngine::new(ndev, &devs)
        .expect("engine init failed");

    println!("=== infccl Rust transfer bench ===");

    let bytes: usize = 16 * 1024 * 1024;
    let iters = 50;

    extern "C" {
        fn cudaSetDevice(dev: i32) -> i32;
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    }

    for src in 0..ndev {
        for dst in 0..ndev {
            if src == dst { continue; }
            let mut s_buf: *mut c_void = ptr::null_mut();
            let mut d_buf: *mut c_void = ptr::null_mut();

            unsafe {
                cudaSetDevice(src); cudaMalloc(&mut s_buf, bytes);
                cudaMemset(s_buf, 1, bytes);
                cudaSetDevice(dst); cudaMalloc(&mut d_buf, bytes);
            }

            let _ = engine.transfer_sync(s_buf, src, d_buf, dst, bytes);

            let start = Instant::now();
            for _ in 0..iters {
                engine.transfer_sync(s_buf, src, d_buf, dst, bytes).unwrap();
            }
            let elapsed = start.elapsed();
            let bw = (bytes * iters) as f64 / elapsed.as_secs_f64() / 1e9;
            println!("  {src}->{dst}: {bw:.1} GB/s");

            unsafe {
                cudaSetDevice(src); cudaFree(s_buf);
                cudaSetDevice(dst); cudaFree(d_buf);
            }
        }
    }
}
