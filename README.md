# SIMD Algorithms in Rust

When using Rust Analyzer it's recommended that you configure the `RUSTFLAGS`.
For example, for VSCode add this to the project config (`.vscode/settings.json`)

```json
{
    "rust-analyzer.server.extraEnv": {
        "RUSTFLAGS": "-C target-cpu=native"
    }
}
```
