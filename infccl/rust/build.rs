use std::env;
use std::path::PathBuf;

fn main() {
    let corex = env::var("COREX_PATH").unwrap_or_else(|_| "/usr/local/corex".to_string());
    let infccl_src = env::var("INFCCL_SRC").unwrap_or_else(|_| {
        let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
        format!("{manifest}/../src")
    });

    println!("cargo:rerun-if-changed={infccl_src}/transfer_engine_c.h");
    println!("cargo:rerun-if-changed={infccl_src}/peer_transport.cu");
    println!("cargo:rerun-if-changed={infccl_src}/transfer_engine_c.cu");
    println!("cargo:rerun-if-changed={infccl_src}/collectives.cu");

    let cuda_include = format!("{corex}/include");
    let cuda_lib = format!("{corex}/lib64");

    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let status = std::process::Command::new(format!("{corex}/bin/clang++"))
        .args([
            "-x", "cuda",
            "--cuda-gpu-arch=ivcore10",
            &format!("--cuda-path={corex}"),
            &format!("-I{cuda_include}"),
            &format!("-I{infccl_src}"),
            "-O2", "-fPIC", "-c",
            &format!("{infccl_src}/peer_transport.cu"),
            "-o",
        ])
        .arg(out_dir.join("peer_transport.o").to_str().unwrap())
        .status()
        .expect("clang++ failed");
    assert!(status.success(), "peer_transport.cu compilation failed");

    let status = std::process::Command::new(format!("{corex}/bin/clang++"))
        .args([
            "-x", "cuda",
            "--cuda-gpu-arch=ivcore10",
            &format!("--cuda-path={corex}"),
            &format!("-I{cuda_include}"),
            &format!("-I{infccl_src}"),
            "-O2", "-fPIC", "-c",
            &format!("{infccl_src}/transfer_engine_c.cu"),
            "-o",
        ])
        .arg(out_dir.join("transfer_engine_c.o").to_str().unwrap())
        .status()
        .expect("clang++ failed");
    assert!(status.success(), "transfer_engine_c.cu compilation failed");

    let status = std::process::Command::new(format!("{corex}/bin/clang++"))
        .args([
            "-x", "cuda",
            "--cuda-gpu-arch=ivcore10",
            &format!("--cuda-path={corex}"),
            &format!("-I{cuda_include}"),
            &format!("-I{infccl_src}"),
            "-O2", "-fPIC", "-c",
            &format!("{infccl_src}/collectives.cu"),
            "-o",
        ])
        .arg(out_dir.join("collectives.o").to_str().unwrap())
        .status()
        .expect("clang++ failed");
    assert!(status.success(), "collectives.cu compilation failed");

    let status = std::process::Command::new("ar")
        .args(["rcs"])
        .arg(out_dir.join("libinfccl.a").to_str().unwrap())
        .arg(out_dir.join("peer_transport.o").to_str().unwrap())
        .arg(out_dir.join("transfer_engine_c.o").to_str().unwrap())
        .arg(out_dir.join("collectives.o").to_str().unwrap())
        .status()
        .expect("ar failed");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=infccl");

    let bindings = bindgen::Builder::default()
        .header(format!("{infccl_src}/transfer_engine_c.h"))
        .clang_arg(format!("-I{cuda_include}"))
        .clang_arg(format!("-I{infccl_src}"))
        .allowlist_function("infccl_.*")
        .allowlist_type("infccl_.*")
        .allowlist_var("INFCCL_.*")
        .generate()
        .expect("bindgen failed");

    bindings.write_to_file(out_dir.join("bindings.rs")).expect("write bindings");
}
