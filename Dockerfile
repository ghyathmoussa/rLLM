# Stage 1: Build the Rust application
FROM rust:1.85-slim-bookworm AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/rllm

# Copy the entire workspace source
COPY . .

# Build the release binary (default CPU-based backend)
RUN cargo build --release --bin rllm

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies (OpenSSL and CA certificates for model downloads)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the builder
COPY --from=builder /usr/src/rllm/target/release/rllm /usr/local/bin/rllm

# Expose HTTP API and Prometheus metrics port
EXPOSE 8000

# Set entrypoint to rllm binary
ENTRYPOINT ["rllm"]
CMD ["serve", "--model", "meta-llama/Llama-3.2-1B-Instruct"]
