use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(all(target_arch = "aarch64", feature = "aarch64-simd"))]
use simd_stuff::unpack_bitmap::aarch64;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
use simd_stuff::unpack_bitmap::x86_64::{bitmap_ones_avx2, bitmap_ones_avx2_small_lut};
use simd_stuff::unpack_bitmap::{
    bitmap_ones, bitmap_ones_a, bitmap_ones_b, bitmap_ones_naive, random_bitmap,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let size = 1024;
    let mut group = c.benchmark_group("bitmap_ones");
    for density in [0.0625, 0.125, 0.25, 0.5, 0.75, 0.9] {
        let bitmap = random_bitmap(size, density);
        let mut output: Vec<u32> = Vec::with_capacity(size);
        group.throughput(Throughput::Elements(bitmap_ones(&bitmap) as u64));

        group.bench_with_input(
            format!("naive/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    bitmap_ones_naive(&bitmap, &mut output);
                    output.clear();
                });
            },
        );

        group.bench_with_input(
            format!("count_trailing_shift/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    bitmap_ones_a(&bitmap, &mut output);
                    output.clear();
                });
            },
        );

        group.bench_with_input(
            format!("count_trailing_toggle/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    bitmap_ones_b(&bitmap, &mut output);
                    output.clear();
                });
            },
        );

        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ))]
        group.bench_with_input(
            format!("avx2/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    unsafe { bitmap_ones_avx2(&bitmap, &mut output) };
                    output.clear();
                });
            },
        );
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ))]
        group.bench_with_input(
            format!("avx2_small_lut/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    bitmap_ones_avx2_small_lut(&bitmap, &mut output);
                    output.clear();
                });
            },
        );

        #[cfg(all(target_arch = "aarch64", feature = "aarch64-simd"))]
        group.bench_with_input(
            format!("neon_4bit/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    unsafe { aarch64::bitmap_ones_simd(&bitmap, &mut output) };
                    output.clear();
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
