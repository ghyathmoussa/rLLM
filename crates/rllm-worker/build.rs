use std::env;
use std::path::PathBuf;

fn main() {
    // Declare the has_cuda cfg key so cargo check doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    if env::var("CARGO_FEATURE_CUDA").is_ok() && which("nvcc").is_some() && find_cuda_home().is_some() {
        println!("cargo:rustc-cfg=has_cuda");
    }
}

fn which(name: &str) -> Option<PathBuf> {
    let path_var = env::var("PATH").ok()?;
    for dir in path_var.split(':') {
        let candidate = PathBuf::from(dir).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn find_cuda_home() -> Option<PathBuf> {
    for var in &["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(home) = env::var(var) {
            let p = PathBuf::from(&home);
            if p.exists() {
                return Some(p);
            }
        }
    }
    for path in &["/usr/local/cuda", "/opt/cuda"] {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}
