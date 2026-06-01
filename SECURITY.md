# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in rLLM, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities by:

1. **GitHub Security Advisories** — Use [GitHub's private vulnerability reporting](../../security/advisories/new) for this repository.
2. **Email** — Send a detailed report to the repository maintainer.

Please include the following in your report:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected versions
- Any suggested mitigations or fixes

We aim to acknowledge reports within **48 hours** and provide an initial assessment within **5 business days**.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | Active development |
| Latest release | Yes        |
| Older releases | No         |

## Security Features

rLLM includes several built-in security mechanisms:

- **API key authentication** — Enable via `--api-key` or the `RLLM_API_KEY` environment variable. Requests without a valid key are rejected.
- **Request size limits** — Configurable limits on input characters (`--max-input-chars`), message count (`--max-input-messages`), and concurrent requests (`--max-concurrent-requests`).
- **Request timeouts** — Enforced via `--request-timeout-secs` to prevent resource exhaustion.
- **CORS controls** — Restrict allowed origins with `--cors-allowed-origins`.
- **TLS** — Uses `rustls` (not OpenSSL) for all outbound HTTPS connections (model downloads, telemetry).
- **Constant-time comparison** — API key validation uses constant-time comparison to prevent timing attacks.
- **Dependency auditing** — `cargo-deny` enforces vulnerability advisory checking and license compliance in CI.

## Deployment Security Recommendations

When deploying rLLM in production:

- Always set `RLLM_API_KEY` and do not expose the server unauthenticated to public networks
- Run behind a reverse proxy (e.g., nginx, Caddy) that handles TLS termination
- Use `--cors-allowed-origins` to restrict cross-origin access
- Set appropriate resource limits (`--max-concurrent-requests`, `--max-input-chars`) to mitigate denial-of-service risk
- Avoid running as root; use a dedicated non-privileged user
- Keep GPU drivers and CUDA toolkit up to date
- Regularly update dependencies (`cargo audit`)

## Scope

This security policy covers the rLLM project code in this repository, including all workspace crates under `crates/`, the HTTP server, and build/distribution artifacts (Docker images).

The following are **out of scope**:

- Vulnerabilities in third-party models loaded at runtime
- Issues in downstream deployments not caused by rLLM code
- Attacks requiring physical access to the host machine
