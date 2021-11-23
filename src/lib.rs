// SIMD on AArch64 requires nightly as of Rust 1.58
#![cfg_attr(feature = "aarch64-simd", feature(stdsimd))]
#![cfg_attr(feature = "aarch64-simd", feature(aarch64_target_feature))]
// #![feature(stdsimd)]
// #![feature(aarch64_target_feature)]

pub mod unpack_bitmap;
