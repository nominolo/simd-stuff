use criterion::{Criterion, Throughput, criterion_group, criterion_main, measurement::{Measurement, ValueFormatter}};
use rand::Rng;
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

fn flush_cache_random(indexes: &[usize], buf: &mut [u64], bytes: usize) {
    for i in 0..bytes / 8 {
        let index = indexes[i & 0x3ff];
        buf[index] = buf[index].wrapping_add(1);
    }
}

fn flush_cache(buf: &mut [u64], bytes: usize) {
    for i in 0..bytes/8 {
        buf[i] = buf[i].wrapping_add(1);
    }
}

fn random_buffer(bytes: usize) -> Vec<u64> {
    let words = (bytes + 7) / 8;
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(words);
    for _ in 0..words {
        result.push(rng.gen());
    }
    result
}

fn index_buffer(indexes: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(indexes);
    for _ in 0..indexes {
        result.push(rng.gen_range(0..1024 * (1024 / 64)));
    }
    result
}

pub fn cache_tests(c: &mut Criterion) {
    let mut flush = random_buffer(1024 * 1024);
    let indexes = index_buffer(1024);
    let mut group = c.benchmark_group("bitmap_ones");
    group.throughput(Throughput::Bytes(32 * 1024));
    group.bench_function("flush_32K", |b| {
        b.iter(|| flush_cache_random(&indexes, &mut flush, 32 * 1024))
    });
    group.throughput(Throughput::Bytes(64 * 1024));
    group.bench_function("flush_64K", |b| {
        b.iter(|| flush_cache_random(&indexes, &mut flush, 64 * 1024))
    });
    group.throughput(Throughput::Bytes(128 * 1024));
    group.bench_function("flush_128K", |b| {
        b.iter(|| flush_cache_random(&indexes, &mut flush, 128 * 1024))
    });
    group.throughput(Throughput::Bytes(256 * 1024));
    group.bench_function("flush_256K", |b| {
        b.iter(|| flush_cache_random(&indexes, &mut flush, 256 * 1024))
    });

    let size = 1024;
    let density = 0.5;
    let bitmap = random_bitmap(size, density);
    let mut output: Vec<u32> = Vec::with_capacity(size);

    group.bench_function("flush_128K", |b| {
        b.iter(|| flush_cache_random(&indexes, &mut flush, 128 * 1024))
    });
    group.throughput(Throughput::Elements(bitmap_ones(&bitmap) as u64));
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    group.bench_with_input(
        format!("flush_128K_avx2/{}/{:.4}", size, density),
        &size,
        |b, &_size| {
            b.iter(|| {
                let x = flush_cache_random(&indexes, &mut flush, 128 * 1024);
                unsafe { bitmap_ones_avx2(&bitmap, &mut output) };
                output.clear();
                x
            });
        },
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    group.bench_with_input(
        format!("flush_128K_avx2_small_lut/{}/{:.4}", size, density),
        &size,
        |b, &_size| {
            b.iter(|| {
                let x = flush_cache_random(&indexes, &mut flush, 128 * 1024);
                bitmap_ones_avx2_small_lut(&bitmap, &mut output);
                output.clear();
                x
            });
        },
    );
}

pub fn criterion_benchmark_cycles(c: &mut Criterion<Cycles>) {
    let size = 1024;
    let mut flush = random_buffer(1024 * 1024);

    let mut group = c.benchmark_group("bitmap_ones_cycles");
    
    {
        let density = 0.5;
        let bitmap = random_bitmap(size, density);
        let mut output: Vec<u32> = Vec::with_capacity(size);
        group.throughput(Throughput::Elements(bitmap_ones(&bitmap) as u64));
        group.bench_function("flush_64K", |b| {
            b.iter(|| flush_cache(&mut flush, 128 * 1024))
        });
        group.bench_with_input(
            format!("flush_count/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    flush_cache(&mut flush, 128 * 1024);
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
            format!("flush_avx2_big_lut/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    flush_cache(&mut flush, 128 * 1024);
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
            format!("flush_avx2_small_lut/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    flush_cache(&mut flush, 128 * 1024);
                    bitmap_ones_avx2_small_lut(&bitmap, &mut output);
                    output.clear();
                });
            },
        );
    }

    for density in [0.0625, 0.125, 0.25, 0.5, 0.75, 0.9] {
        let bitmap = random_bitmap(size, density);
        let mut output: Vec<u32> = Vec::with_capacity(size);
        group.throughput(Throughput::Elements(bitmap_ones(&bitmap) as u64));

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
            format!("avx2_small_lut/{}/{:.4}", size, density),
            &size,
            |b, &_size| {
                b.iter(|| {
                    bitmap_ones_avx2_small_lut(&bitmap, &mut output);
                    output.clear();
                });
            },
        );
    }
}

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

criterion_group!(benches, criterion_benchmark, cache_tests);

criterion_group!(
    name = cycle_benches;
    config = Criterion::default().with_measurement(Cycles);
    targets = criterion_benchmark_cycles
);
criterion_main!(benches, cycle_benches);


pub struct Cycles;

fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::x86::_rdtsc()
    }
}


impl Measurement for Cycles {
    type Intermediate = u64;
    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        rdtsc()
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        rdtsc() - i
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CyclesFormatter
    }
}

struct CyclesFormatter;

impl ValueFormatter for CyclesFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.4} cycles", value)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        match throughput {
            Throughput::Bytes(b) => format!("{:.4} cpb", value / *b as f64),
            Throughput::Elements(b) => format!("{:.4} cycles/elem", value / *b as f64),
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "cycles"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Bytes(n) => {
                for val in values {
                    *val /= *n as f64;
                }
                "cpb"
            }
            Throughput::Elements(n) => {
                for val in values {
                    *val /= *n as f64;
                }
                "c/e"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "cycles"
    }
}