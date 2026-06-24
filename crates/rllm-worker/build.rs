use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    // Declare the has_cuda cfg key so cargo check doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    let nvcc = which("nvcc");
    if env::var("CARGO_FEATURE_CUDA").is_ok()
        && nvcc.as_ref().is_some_and(|path| find_cuda_home(path).is_some())
    {
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

fn find_cuda_home(nvcc: &Path) -> Option<PathBuf> {
    for var in &["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(home) = env::var(var) {
            let p = PathBuf::from(&home);
            if has_cuda_headers(&p) {
                return Some(p);
            }
        }
    }
    for path in &["/usr/local/cuda", "/opt/cuda"] {
        let p = PathBuf::from(path);
        if has_cuda_headers(&p) {
            return Some(p);
        }
    }
    if let Some(root) = nvcc.parent().and_then(Path::parent) {
        if has_cuda_headers(root) {
            return Some(root.to_path_buf());
        }
    }
    None
}

fn has_cuda_headers(path: &Path) -> bool {
    path.join("include").join("cuda_runtime.h").exists()
}
