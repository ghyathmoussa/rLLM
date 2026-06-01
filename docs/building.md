# Building rLLM

This guide covers building rLLM for two distinct targets:

1. **CPU release binary** — a fully static, portable Linux binary (what CI publishes).
2. **CUDA GPU build** — built from source on the GPU host with `--features cuda`.

These are different artifacts with different portability rules. Read the section that
matches what you need.

---

## Prerequisites: Rust 1.85+ (edition 2024)

rLLM uses **Rust edition 2024**, which requires a **Rust toolchain ≥ 1.85** (the project MSRV).

> ⚠️ The `rust`/`cargo` package shipped by AlmaLinux/RHEL/Rocky (`dnf install rust cargo`,
> or `rust-toolset`) is often **too old** — e.g. **Cargo 1.75.0** — and fails with:
>
> ```
> error: failed to parse manifest at `.../Cargo.toml`
>   feature `edition2024` is required
>   The package requires the Cargo feature called `edition2024`, but that feature is
>   not stabilized in this version of Cargo (1.75.0).
> ```
>
> **Use `rustup`, not the distro RPM.** Install/update it:
>
> ```bash
> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
>   | sh -s -- -y --default-toolchain stable
> source "$HOME/.cargo/env"
> # If rustup is already installed but old:  rustup update stable && rustup default stable
> rustc --version       # must be >= 1.85
> which cargo           # must be ~/.cargo/bin/cargo, NOT /usr/bin/cargo
> ```
>
> If `which cargo` still points at `/usr/bin/cargo`, the distro version is shadowing rustup —
> ensure `~/.cargo/bin` is first on `PATH` (or `dnf remove rust cargo`).

---

## 1. CPU release binary (static, portable)

The published release artifact is a **statically linked musl binary**
(`x86_64-unknown-linux-musl`). It has **zero glibc dependency**, so it runs on any x86-64
Linux — AlmaLinux/Rocky/RHEL 8/9/10, Ubuntu 20.04+, Debian, Alpine — with no "GLIBC_2.xx
not found" errors. This is what `.github/workflows/release.yml` produces.

> ℹ️ The static CPU binary is **CPU-only**. It runs anywhere but does **not** use the GPU.
> For GPU inference, build from source with CUDA — see section 2.

### Why static/musl?

A dynamically linked binary records the *newest* glibc symbol versions it was built against
and demands them at runtime. glibc is backward- but **not** forward-compatible, so a binary
built on Ubuntu 24.04 (glibc 2.39) fails on AlmaLinux 9.x (glibc 2.34) with
`GLIBC_2.38 / GLIBC_2.39 not found`. A static musl build sidesteps this entirely — there is
no glibc dependency to mismatch.

### Building on Ubuntu (what CI does)

```bash
rustup target add x86_64-unknown-linux-musl
# musl C cross-toolchain (for the C/C++ deps: aws-lc-sys, ring, onig)
cargo install --locked cargo-zigbuild   # or: taiki-e/setup-cross-toolchain-action in CI
cargo build --release --bin rllm --target x86_64-unknown-linux-musl
```

CI uses `taiki-e/setup-cross-toolchain-action@v1` to provide the musl C toolchain, then runs
`cargo build --release --bin rllm --target x86_64-unknown-linux-musl`.

### Building inside an AlmaLinux container (verified)

These are the exact steps used to reproduce the release build in an AlmaLinux 10 container
(works on AlmaLinux 8/9/10). They are the RPM-based equivalent of the CI musl steps.

```bash
# 1. Base build tooling. cmake + perl are required by aws-lc-sys.
microdnf install -y gcc gcc-c++ make cmake perl tar gzip xz which findutils

# 2. Rust toolchain + musl target
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --profile minimal
source "$HOME/.cargo/env"
rustup target add x86_64-unknown-linux-musl

# 3. musl C cross-compiler (AlmaLinux equivalent of the CI cross-toolchain action).
#    Provided by EPEL.
microdnf install -y epel-release
microdnf install -y musl-gcc musl-devel musl-libc-static

# 4. Build. CC_<target> tells the `cc` crate to compile C deps with musl-gcc.
export CC_x86_64_unknown_linux_musl=musl-gcc
cargo build --release --bin rllm --target x86_64-unknown-linux-musl --locked
```

### Verifying the static binary

```bash
BIN=target/x86_64-unknown-linux-musl/release/rllm
file "$BIN"                       # => "... static-pie linked ..."
ldd  "$BIN"                       # => "statically linked"
objdump -T "$BIN" | grep -c GLIBC # => 0  (no glibc symbols required)
"$BIN" --version                  # => "rllm 0.1.0", exit 0
"$BIN" serve --help               # => usage, exit 0
```

If `objdump -T ... | grep -c GLIBC` prints `0`, the binary will run on **any** glibc version,
including AlmaLinux 9.8.

### Note on tokenizer features

The workspace pins `tokenizers` with `default-features = false, features = ["onig"]`
(`Cargo.toml`). The dropped defaults — `esaxx_fast` (a **C++** suffix-automaton used only for
BPE *training*) and `progressbar` — are never used for inference, and `esaxx_fast` breaks
musl-static builds because it needs a musl **C++** toolchain (EPEL's `musl-gcc` is C-only).
`onig` (C) is kept and builds fine against musl.

---

## 2. CUDA GPU build (from source on the GPU host)

To actually use an NVIDIA GPU you must build from source **on the GPU host** with
`--features cuda`. Do **not** expect the static CPU release binary to use the GPU.

### Requirements

- NVIDIA GPU, compute capability **7.5+** (Turing or newer): RTX 20xx, Titan RTX,
  RTX 6000 Ada (sm_89), A100 (sm_80), H100 (sm_90), etc.
- **CUDA Toolkit 12.x** (e.g. 12.8). **CUDA 13.x is not yet supported** — candle's transitive
  `cudarc 0.17.8` only ships bindings up to CUDA 13.0, so a 13.x toolkit can fail with
  `Unsupported cuda toolkit version`. Install CUDA 12.x; your GPU **driver** can stay newer
  (drivers are backward compatible with older toolkits).
- `nvcc` on `PATH`.

### Why build on the GPU host?

A CUDA build is dynamically linked (glibc + CUDA runtime libraries), so musl/static is not an
option. The simplest way to avoid any glibc mismatch is to **build on the same machine you run
on** — it links against that host's own glibc. On an AlmaLinux 9.8 box, the binary links
glibc 2.34 and runs there without issue.

### Build

```bash
# Point at the CUDA 12.x toolkit (NOT 13.x)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compile kernels only for your GPU (faster). Examples:
#   RTX 6000 Ada / RTX 40xx  -> 8.9
#   Titan RTX / RTX 20xx     -> 7.5
#   A100                     -> 8.0
#   H100                     -> 9.0
export CUDA_ARCH=8.9

cargo build --release --features cuda --bin rllm
```

Default kernel architectures (if `CUDA_ARCH` is unset) are `sm_75`–`sm_90`. `sm_70` (Volta)
is intentionally omitted because CUDA 13 removed it.

### Smoke test

Use `scripts/verify_cuda_build.sh` (below) to automate toolkit checks, build, and a serve
smoke test:

```bash
./scripts/verify_cuda_build.sh                 # auto-detects CUDA_ARCH from the GPU
CUDA_ARCH=8.9 ./scripts/verify_cuda_build.sh   # or pin it
```

---

## Build / test / lint reference

```bash
cargo build --workspace                  # CPU
cargo build --workspace --features cuda  # CUDA
cargo test  --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```
