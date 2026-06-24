use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    // Declare the has_cuda cfg key so cargo check doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // Only compile CUDA kernels when the "cuda" feature is enabled.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:warning=cuda feature not enabled, skipping CUDA kernel compilation");
        return;
    }

    // Emit rerun triggers up front so env changes invalidate the cache even
    // when detection currently fails (e.g. a missing CUDA_HOME that is later set).
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-changed=src/cuda/");

    // Probe for nvcc on PATH.
    let nvcc = which_nvcc();
    if nvcc.is_none() {
        println!("cargo:warning=nvcc not found on PATH; CUDA kernels will not be compiled");
        return;
    }
    let nvcc = nvcc.unwrap();

    // Locate CUDA toolkit root.
    let cuda_home = find_cuda_home(nvcc.as_path());
    if cuda_home.is_none() {
        println!("cargo:warning=CUDA toolkit not found; CUDA kernels will not be compiled");
        return;
    }
    let cuda_home = cuda_home.unwrap();

    // CUDA is available — set the cfg flag.
    println!("cargo:rustc-cfg=has_cuda");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=src/cuda/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Determine target GPU architectures.
    let arch_flags = cuda_arch_flags();

    // Collect .cu source files.
    let cuda_src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/cuda");
    let cu_files = collect_cu_files(&cuda_src_dir);

    if cu_files.is_empty() {
        println!("cargo:warning=no .cu files found in {}", cuda_src_dir.display());
        return;
    }

    // Compile each .cu file into an object file.
    let mut obj_files = Vec::new();
    for cu in &cu_files {
        let file_stem = cu.file_stem().unwrap().to_string_lossy();
        let obj = out_dir.join(format!("{file_stem}.o"));

        let mut cmd = std::process::Command::new(&nvcc);
        cmd.arg("-c")
            .arg(cu)
            .arg("-o")
            .arg(&obj)
            .arg("-O2")
            .args(&arch_flags)
            .arg(format!("-I{}/include", cuda_home.display()));

        if cfg!(target_os = "linux") {
            cmd.arg("-Xcompiler").arg("-fPIC");
        }

        println!("cargo:warning=compiling CUDA kernel: {}", cu.display());
        let status = cmd.status().expect("failed to run nvcc");
        if !status.success() {
            panic!("nvcc compilation failed for {}", cu.display());
        }

        obj_files.push(obj);
    }

    // Create a static library from all object files.
    let lib_path = out_dir.join("librllm_cuda_kernels.a");
    let mut ar = std::process::Command::new("ar");
    ar.arg("rcs").arg(&lib_path).args(&obj_files);

    let status = ar.status().expect("failed to run ar");
    if !status.success() {
        panic!("ar archiving failed");
    }

    // Tell cargo to link the static library.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rllm_cuda_kernels");

    // Link against CUDA runtime. Locate the lib dir robustly: some distros
    // (Debian/Ubuntu `nvidia-cuda-toolkit`) install libcudart under a multiarch
    // path like /usr/lib/x86_64-linux-gnu rather than $CUDA_HOME/lib64.
    let lib_dir = find_cuda_lib_dir(&cuda_home)
        .unwrap_or_else(|| cuda_home.join("lib64"));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn which_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("NVCC") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }
    which("nvcc")
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
    // 1. Explicit env vars (highest priority).
    for var in &["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(home) = env::var(var) {
            let p = PathBuf::from(&home);
            if toolkit_has_headers(&p) {
                return Some(p);
            }
        }
    }
    // 2. Standard CUDA install roots.
    for path in &["/usr/local/cuda", "/opt/cuda"] {
        let p = PathBuf::from(path);
        if toolkit_has_headers(&p) {
            return Some(p);
        }
    }
    // 3. Derive from the nvcc location. Distros that ship CUDA via the package
    //    manager (e.g. Debian/Ubuntu `nvidia-cuda-toolkit`) put nvcc in
    //    /usr/bin, so the toolkit root is the parent of `bin` (i.e. /usr).
    if let Some(nvcc) = which_nvcc() {
        if let Some(bin) = nvcc.parent() {
            if let Some(root) = bin.parent() {
                if toolkit_has_headers(root) {
                    return Some(root.to_path_buf());
                }
            }
        }
    }
    None
}

/// A directory counts as a CUDA toolkit root if it exposes the CUDA headers
/// under `include/`. Debian's apt layout ships them directly under `/usr/include`.
fn toolkit_has_headers(root: &std::path::Path) -> bool {
    root.join("include/cuda_runtime.h").exists()
}

/// Locate the directory containing `libcudart.so`. Covers the canonical
/// `<root>/lib64` layout as well as Debian/Ubuntu multiarch (`<root>/lib` and
/// `<root>/lib/x86_64-linux-gnu`).
fn find_cuda_lib_dir(cuda_home: &std::path::Path) -> Option<PathBuf> {
    let candidates = [
        cuda_home.join("lib64"),
        cuda_home.join("lib/x86_64-linux-gnu"),
        cuda_home.join("lib"),
    ];
    for c in &candidates {
        if c.join("libcudart.so").exists() {
            return Some(c.clone());
        }
    }
    None
}

fn has_cuda_headers(path: &Path) -> bool {
    path.join("include").join("cuda_runtime.h").exists()
}

fn cuda_lib_dirs(cuda_home: &Path) -> Vec<PathBuf> {
    ["lib64", "lib", "lib/x86_64-linux-gnu"]
        .iter()
        .map(|dir| cuda_home.join(dir))
        .filter(|dir| dir.exists())
        .collect()
}

fn cuda_arch_flags() -> Vec<String> {
    if let Ok(archs) = env::var("CUDA_ARCH") {
        return archs
            .split(';')
            .map(|a| {
                let v = a.replace('.', "");
                format!("-gencode=arch=compute_{v},code=[sm_{v},compute_{v}]")
            })
            .collect();
    }

    // Turing (7.5) through Hopper (9.0). `sm_70` (Volta) is intentionally
    // omitted: CUDA 13 removed Volta support, so a `compute_70` gencode line
    // makes nvcc from a CUDA-13 toolkit fail. Override with CUDA_ARCH if needed.
    let default_archs = ["7.5", "8.0", "8.6", "8.9", "9.0"];
    default_archs
        .iter()
        .map(|a| {
            let v = a.replace('.', "");
            format!("-gencode=arch=compute_{v},code=[sm_{v},compute_{v}]")
        })
        .collect()
}

fn collect_cu_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    files.sort();
    files
}
