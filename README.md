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

# Unpack Bitmap

Turns a bitmap into a `Vec<u32>` of the indexes of the one bits.

Benchmark for 1024 bit bitmap. Density is the ratio of 1-bits relative to the
total number of bits.

The x86 version uses a 2KiB lookup table (`256 * 8B`). The aarch64 version only 
processes looks up 4 bits at a time and thus only needs a 128 Byte lookup table.
The scalar version is the same for both.

| Density  | Hardware                     |  Scalar   |  SIMD   |  Ratio  |
|----------|------------------------------|----------:|--------:|--------:|
| 0.0625   | AMD Ryzen 9 3900X            |     68 ns |   64 ns |   1.06x |
| 0.125    | AMD Ryzen 9 3900X            |    100 ns |   64 ns |   1.56x |
| 0.25     | AMD Ryzen 9 3900X            |    178 ns |   65 ns |    2.7x |
| 0.5      | AMD Ryzen 9 3900X            |    407 ns |   65 ns |    6.3x |
| 0.75     | AMD Ryzen 9 3900X            |    615 ns |   65 ns |    9.5x |
| 0.9      | AMD Ryzen 9 3900X            |    739 ns |   65 ns |   11.4x |

| Density  | Hardware                     |  Scalar   |  SIMD   |  Ratio  |
|----------|------------------------------|----------:|--------:|--------:|
| 0.0625   | Apple M1 (Macbook Air 2020)  |    132 ns |  164 ns |    0.8x |
| 0.125    | Apple M1 (Macbook Air 2020)  |    264 ns |  164 ns |    1.6x |
| 0.25     | Apple M1 (Macbook Air 2020)  |    522 ns |  164 ns |    3.2x |
| 0.5      | Apple M1 (Macbook Air 2020)  |   1053 ns |  164 ns |    6.4x |
| 0.75     | Apple M1 (Macbook Air 2020)  |   1588 ns |  165 ns |    9.7x |
| 0.9      | Apple M1 (Macbook Air 2020)  |   1910 ns |  167 ns |   11.6x |

So, it looks like the M1 isn't very fast when doing these bit-wise operations.
But in both cases, the SIMD version is extremely consistent and the relative
performance compared to the scalar version is very similar.