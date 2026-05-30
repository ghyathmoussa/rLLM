#!/usr/bin/env bash
#
# verify_cuda_build.sh — build rLLM with CUDA on a GPU host and smoke-test it.
#
# Run this ON the GPU machine (e.g. the AlmaLinux 9.8 + RTX 6000 Ada box).
# It checks the CUDA toolkit, auto-detects the GPU arch, builds with --features cuda,
# and runs a smoke test. Optionally serves a small model and hits the API.
#
# Usage:
#   ./scripts/verify_cuda_build.sh                 # build + smoke test
#   CUDA_ARCH=8.9 ./scripts/verify_cuda_build.sh   # pin the arch
#   SERVE=1 MODEL=Qwen/Qwen2.5-0.5B-Instruct ./scripts/verify_cuda_build.sh  # live serve test
#
# Env:
#   CUDA_HOME   CUDA 12.x toolkit root (auto-detected if unset)
#   CUDA_ARCH   GPU compute capability, e.g. 8.9 (auto-detected from nvidia-smi if unset)
#   SERVE       if "1", start the server with MODEL and curl it
#   MODEL       HF model id for the serve test (default: Qwen/Qwen2.5-0.5B-Instruct)
#   PORT        serve port (default: 8000)

set -euo pipefail

RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'
info() { echo "${BLU}==>${RST} $*"; }
ok()   { echo "${GRN}OK ${RST} $*"; }
warn() { echo "${YLW}WARN${RST} $*"; }
fail() { echo "${RED}FAIL${RST} $*" >&2; exit 1; }

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 0. Rust toolchain must be >= 1.85 (edition 2024).
#    AlmaLinux/RHEL's distro cargo (e.g. 1.75) is too old — use rustup.
# ---------------------------------------------------------------------------
command -v cargo >/dev/null 2>&1 || fail "cargo not found. Install rustup: https://rustup.rs"
RUST_VER=$(cargo --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
RUST_MAJOR=${RUST_VER%%.*}; RUST_MINOR=$(echo "$RUST_VER" | cut -d. -f2)
if (( RUST_MAJOR < 1 || (RUST_MAJOR == 1 && RUST_MINOR < 85) )); then
  warn "cargo $RUST_VER is too old; rLLM needs Rust >= 1.85 (edition 2024)."
  warn "The distro rust RPM (often 1.75) won't work. Install/refresh rustup:"
  warn "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
  warn "  source \$HOME/.cargo/env && rustup default stable"
  fail "upgrade Rust to >= 1.85 and re-run ($(command -v cargo) is $RUST_VER)."
fi
ok "Rust $RUST_VER (>= 1.85)"

# ---------------------------------------------------------------------------
# 1. Locate the CUDA toolkit
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 \
           /usr/local/cuda /opt/cuda; do
    [[ -d "$c" ]] && { CUDA_HOME="$c"; break; }
  done
fi
[[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]] \
  || fail "CUDA toolkit not found. Install CUDA 12.x and set CUDA_HOME."
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
info "CUDA_HOME = $CUDA_HOME"

# ---------------------------------------------------------------------------
# 2. Check nvcc and toolkit version (must be 12.x for the pinned deps)
# ---------------------------------------------------------------------------
command -v nvcc >/dev/null 2>&1 || fail "nvcc not on PATH (looked in $CUDA_HOME/bin)."
NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
NVCC_MAJOR=${NVCC_VER%%.*}
info "nvcc version = $NVCC_VER"
if [[ "$NVCC_MAJOR" -ge 13 ]]; then
  warn "CUDA $NVCC_VER detected. The pinned cudarc/candle only ship bindings up to ~13.2,"
  warn "and candle's transitive cudarc 0.17.8 tops out at 13.0 — the build may fail with"
  warn "'Unsupported cuda toolkit version'. Strongly recommend installing CUDA 12.x and"
  warn "pointing CUDA_HOME at it. Continuing anyway (fallback-latest may help the direct dep)..."
elif [[ "$NVCC_MAJOR" -ne 12 ]]; then
  warn "CUDA $NVCC_VER is untested; 12.x is the supported line."
else
  ok "CUDA 12.x — supported toolkit line."
fi

# ---------------------------------------------------------------------------
# 3. Detect GPU + compute capability (CUDA_ARCH)
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  info "GPUs:"
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | sed 's/^/    /'
  if [[ -z "${CUDA_ARCH:-}" ]]; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  fi
else
  warn "nvidia-smi not found; cannot auto-detect GPU. Set CUDA_ARCH manually if needed."
fi
if [[ -n "${CUDA_ARCH:-}" ]]; then
  export CUDA_ARCH
  info "CUDA_ARCH = $CUDA_ARCH (kernels compiled for this arch only)"
  case "$CUDA_ARCH" in
    7.0) warn "sm_70 (Volta) is dropped by CUDA 13 and not targeted by rLLM defaults." ;;
  esac
fi

# ---------------------------------------------------------------------------
# 4. Build with CUDA
# ---------------------------------------------------------------------------
info "Building: cargo build --release --features cuda --bin rllm"
cargo build --release --features cuda --bin rllm
BIN="target/release/rllm"
[[ -x "$BIN" ]] || fail "binary not produced at $BIN"
ok "CUDA build succeeded: $BIN"

# ---------------------------------------------------------------------------
# 5. Smoke test the binary
# ---------------------------------------------------------------------------
info "Binary smoke test"
"$BIN" --version
"$BIN" serve --help >/dev/null
ok "binary runs (--version, serve --help)"
# Confirm it actually linked against the CUDA runtime.
if ldd "$BIN" | grep -qiE 'libcudart|libcuda'; then
  ok "linked against CUDA runtime (libcudart/libcuda present)"
else
  warn "no CUDA runtime in ldd output — check that the 'cuda' feature compiled kernels."
fi

# ---------------------------------------------------------------------------
# 6. Optional: live serve + API call (exercises the GPU)
# ---------------------------------------------------------------------------
if [[ "${SERVE:-0}" == "1" ]]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  PORT="${PORT:-8000}"
  info "Live serve test: $MODEL on port $PORT (downloads weights on first run)"
  "$BIN" serve "$MODEL" --dtype bf16 --port "$PORT" >/tmp/rllm_serve.log 2>&1 &
  SRV_PID=$!
  trap 'kill $SRV_PID 2>/dev/null || true' EXIT
  info "waiting for server (pid $SRV_PID) to come up..."
  for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
    kill -0 "$SRV_PID" 2>/dev/null || { tail -30 /tmp/rllm_serve.log; fail "server exited early"; }
  done
  info "GPU state while serving:"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'
  info "completion request:"
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16}" \
    && echo
  ok "served a request on the GPU"
fi

echo
ok "All checks passed."
