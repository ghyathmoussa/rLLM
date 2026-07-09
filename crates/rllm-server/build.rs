fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // SAFETY: build scripts run as a single-purpose process before code generation.
    // Setting PROTOC here only affects this build script and its child prost-build work.
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/rllm/v1/inference.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/rllm/v1/inference.proto");
    Ok(())
}
