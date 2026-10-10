fn main() {
    let ndev = 4;
    for gpu in 0..ndev {
        println!("GPU {gpu}:");
        match infccl::engine::probe_caps(gpu) {
            Ok(caps) => {
                println!("  warp_size: {}", caps.warp_size);
                println!("  SMs: {}", caps.sm_count);
                println!("  fp64: {}", if caps.fp64_works { "OK" } else { "BROKEN" });
                println!("  HBM BW: {:.1} GB/s", caps.hbm_bw_gbps);
                println!("  peer BW: {:.1} GB/s", caps.peer_bw_gbps);
                println!("  peer latency: {:.1} us", caps.peer_latency_us);
                println!("  needs cudaSetDevice(src): {}", caps.peer_memcpy_needs_src_device);
                println!("  needs blocking stream: {}", caps.peer_memcpy_needs_blocking_stream);
            }
            Err(e) => println!("  probe failed: {e}"),
        }
    }
}
