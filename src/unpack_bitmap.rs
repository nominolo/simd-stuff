//! Unpacking a bitmap. Convert a bitmap to a list of
//!

use rand::Rng;

pub fn bitmap_ones_naive(bitmap: &[u64], out: &mut Vec<u32>) {
    for (word_index, word) in bitmap.iter().enumerate() {
        for bit in 0..64 {
            if *word & (1 << bit) != 0 {
                out.push((word_index * 64 + bit) as u32);
            }
        }
    }
}

pub fn bitmap_ones_a(bitmap: &[u64], out: &mut Vec<u32>) {
    for (word_index, word) in bitmap.iter().enumerate() {
        let mut temp = *word;
        while temp != 0 {
            let bit = temp.trailing_zeros();
            out.push((word_index * 64) as u32 + bit);
            temp ^= 1 << bit;
        }
    }
}

// https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/
pub fn bitmap_ones_b(bitmap: &[u64], out: &mut Vec<u32>) {
    for (word_index, word) in bitmap.iter().enumerate() {
        let mut temp = *word;
        while temp != 0 {
            // Creates a mask with only the lowest 1-bit. Can run in parallel
            // with the other ops. E.g., see
            // <https://catonmat.net/low-level-bit-hacks> and search for
            // "isolate lowest 1-bit"
            let toggle = temp & ((temp as i64).wrapping_neg() as u64);
            let bit = temp.trailing_zeros();
            out.push((word_index * 64) as u32 + bit);
            temp ^= toggle;
        }
    }
}

pub fn random_bitmap_accurate(total_bits: usize, one_bits: usize) -> Vec<u64> {
    let mut bits = vec![0_u64; (total_bits + 63) / 64];
    let mut rng = rand::thread_rng();
    let mut available_indexes: Vec<u32> = (0..total_bits as u32).collect();
    for _ in 0..one_bits {
        let i = available_indexes.swap_remove(rng.gen_range(0..available_indexes.len())) as usize;
        let word_idx = i >> 6;
        let bit_idx = i & 0x3f;
        bits[word_idx] |= 1 << bit_idx;
    }
    bits
}

pub fn random_bitmap(num_bits: usize, percent_ones: f64) -> Vec<u64> {
    let num_ones = (num_bits as f64 * percent_ones).round() as usize;
    random_bitmap_accurate(num_bits, num_ones)
}

pub fn bitmap_ones(bitmap: &[u64]) -> u32 {
    bitmap.iter().map(|b| b.count_ones()).sum()
}

#[test]
fn basic() {
    let num_bits = 64;
    let bitmap = random_bitmap(num_bits, 0.5);
    println!(
        "{:#b} {:.2}",
        bitmap[0],
        bitmap_ones(&bitmap) as f64 / num_bits as f64
    );

    let mut expected = Vec::with_capacity(num_bits);
    let mut faster = Vec::with_capacity(num_bits);
    let mut faster2 = Vec::with_capacity(num_bits);
    bitmap_ones_naive(&bitmap, &mut expected);
    bitmap_ones_a(&bitmap, &mut faster);
    bitmap_ones_b(&bitmap, &mut faster2);
    // println!("{:?}", expected);
    // println!("{:?}", faster);
    assert_eq!(expected, faster);
    assert_eq!(expected, faster2);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "ssse3"
))]
pub mod x86_64 {

    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_cvtepu8_epi32, _mm256_load_si256, _mm256_set1_epi32,
        _mm256_storeu_si256, _mm_cvtsi64_si128,
    };

    // https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/
    #[allow(unused, clippy::missing_safety_doc)]
    pub unsafe fn bitmap_ones_avx2(bitmap: &[u64], output: &mut Vec<u32>) {
        output.reserve(bitmap.len() * 64);
        let mut base_vec: __m256i = _mm256_set1_epi32(-1);
        let add_8: __m256i = _mm256_set1_epi32(8);
        let add_64: __m256i = _mm256_set1_epi32(64);
        let mut out = output.as_mut_ptr();
        for word in bitmap {
            let mut w = *word;
            if w == 0 {
                base_vec = _mm256_add_epi32(base_vec, add_64);
                continue;
            }
            for i in 0..4 {
                let byte_a = w as u8;
                let byte_b = (w >> 8) as u8;
                w >>= 16;

                let mut indexes_a: __m256i = _mm256_load_si256(
                    &(INDEX_LOOKUP_TABLE[byte_a as usize].0) as *const [u32; 8] as *const __m256i,
                );
                let mut indexes_b: __m256i = _mm256_load_si256(
                    &(INDEX_LOOKUP_TABLE[byte_b as usize].0) as *const [u32; 8] as *const __m256i,
                );

                let advance_a = byte_a.count_ones();
                let advance_b = byte_b.count_ones();

                indexes_a = _mm256_add_epi32(base_vec, indexes_a);
                base_vec = _mm256_add_epi32(base_vec, add_8);
                indexes_b = _mm256_add_epi32(base_vec, indexes_b);
                base_vec = _mm256_add_epi32(base_vec, add_8);

                _mm256_storeu_si256(out as *mut __m256i, indexes_a);
                out = out.add(advance_a as usize);
                _mm256_storeu_si256(out as *mut __m256i, indexes_b);
                out = out.add(advance_b as usize);
            }
        }
        let len = out.offset_from(output.as_ptr());
        output.set_len(len as usize);
    }

    /// This one seems to be the best. It uses a 2KiB lookup table.
    ///
    /// Based on <https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/>
    ///
    ///
    pub fn bitmap_ones_avx2_small_lut(bitmap: &[u64], output: &mut Vec<u32>) {
        // Reserve the maximum number of outputs that we will produce
        output.reserve(bitmap.len() * 64);
        // Safety: We write at most bitmap.len() * 64 u32s, so this is safe.
        unsafe {
            let mut base_vec: __m256i = _mm256_set1_epi32(-1);
            let add_8: __m256i = _mm256_set1_epi32(8);
            let add_64: __m256i = _mm256_set1_epi32(64);
            let mut out = output.as_mut_ptr();
            for word in bitmap {
                let mut w = *word;
                if w == 0 {
                    base_vec = _mm256_add_epi32(base_vec, add_64);
                    continue;
                }
                // this loop gets unrolled by the compiler
                for _i in 0..4 {
                    let byte_a = w as u8;
                    let byte_b = (w >> 8) as u8;
                    w >>= 16;

                    // Compiles to a single: vpmovzxbd
                    let mut indexes_a: __m256i = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                        INDEX_LOOKUP_TABLE_U8[byte_a as usize] as i64,
                    ));
                    let mut indexes_b: __m256i = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                        INDEX_LOOKUP_TABLE_U8[byte_b as usize] as i64,
                    ));

                    let advance_a = byte_a.count_ones();
                    let advance_b = byte_b.count_ones();

                    indexes_a = _mm256_add_epi32(base_vec, indexes_a);
                    base_vec = _mm256_add_epi32(base_vec, add_8);
                    indexes_b = _mm256_add_epi32(base_vec, indexes_b);
                    base_vec = _mm256_add_epi32(base_vec, add_8);

                    _mm256_storeu_si256(out as *mut __m256i, indexes_a);
                    out = out.add(advance_a as usize);
                    _mm256_storeu_si256(out as *mut __m256i, indexes_b);
                    out = out.add(advance_b as usize);
                }
            }
            let len = out.offset_from(output.as_ptr());
            output.set_len(len as usize);
        }
    }

    // Custom type to ensure 32 byte alignment.
    #[repr(C, align(32))]
    struct IndexLookup([u32; 8]);

    #[test]
    fn gen_index_lookup_table() {
        println!("#[allow(unused)]");
        println!("#[rustfmt::skip]");
        println!("static INDEX_LOOKUP_TABLE: [IndexLookup; 256] = [");
        for b in 0..256 {
            let mut indexes = Vec::with_capacity(8);
            for i in 0..8 {
                if b & (1 << i) != 0 {
                    indexes.push(i + 1);
                }
            }
            indexes.resize(8, 0);
            println!("    IndexLookup({:?}), // {:#08b}", indexes, b);
        }
        println!("];");
    }

    #[test]
    fn gen_index_lookup_table_u8() {
        println!("#[allow(unused)]");
        println!("#[rustfmt::skip]");
        println!("static INDEX_LOOKUP_TABLE_U8: [u64; 256] = [");
        for b in 0..256 {
            let mut indexes = Vec::with_capacity(8);
            for i in 0..8 {
                if b & (1 << i) != 0 {
                    indexes.push(i + 1);
                }
            }
            indexes.resize(8, 0);
            println!("    lookup_row_u8({:?}), // {:#08b}", indexes, b);
        }
        println!("];");
    }

    #[cfg(test)]
    mod tests {
        use super::{
            bitmap_ones, bitmap_ones_a, bitmap_ones_avx2, bitmap_ones_avx2_small_lut,
            bitmap_ones_b, bitmap_ones_naive, random_bitmap,
        };

        #[test]
        fn avx2() {
            let num_bits = 64;
            let bitmap = random_bitmap(num_bits, 0.5);
            println!(
                "{:#b} {:.2}",
                bitmap[0],
                bitmap_ones(&bitmap) as f64 / num_bits as f64
            );

            let mut expected = Vec::with_capacity(num_bits);
            let mut faster = Vec::with_capacity(num_bits);
            let mut faster2 = Vec::with_capacity(num_bits);
            bitmap_ones_naive(&bitmap, &mut expected);
            unsafe {
                bitmap_ones_avx2(&bitmap, &mut faster);
            }
            bitmap_ones_avx2_small_lut(&bitmap, &mut faster2);
            // println!("{:?}", expected);
            // println!("{:?}", faster);
            assert_eq!(expected, faster);
            assert_eq!(expected, faster2);
        }
    }

    const fn lookup_row_u8(bytes: [u8; 8]) -> u64 {
        bytes[0] as u64
            | (bytes[1] as u64) << 8
            | (bytes[2] as u64) << 16
            | (bytes[3] as u64) << 24
            | (bytes[4] as u64) << 32
            | (bytes[5] as u64) << 40
            | (bytes[6] as u64) << 48
            | (bytes[7] as u64) << 56
    }

    // 256 * 8B = 2KiB
    #[allow(unused)]
#[rustfmt::skip]
static INDEX_LOOKUP_TABLE_U8: [u64; 256] = [
    lookup_row_u8([0, 0, 0, 0, 0, 0, 0, 0]), // 0b000000
    lookup_row_u8([1, 0, 0, 0, 0, 0, 0, 0]), // 0b000001
    lookup_row_u8([2, 0, 0, 0, 0, 0, 0, 0]), // 0b000010
    lookup_row_u8([1, 2, 0, 0, 0, 0, 0, 0]), // 0b000011
    lookup_row_u8([3, 0, 0, 0, 0, 0, 0, 0]), // 0b000100
    lookup_row_u8([1, 3, 0, 0, 0, 0, 0, 0]), // 0b000101
    lookup_row_u8([2, 3, 0, 0, 0, 0, 0, 0]), // 0b000110
    lookup_row_u8([1, 2, 3, 0, 0, 0, 0, 0]), // 0b000111
    lookup_row_u8([4, 0, 0, 0, 0, 0, 0, 0]), // 0b001000
    lookup_row_u8([1, 4, 0, 0, 0, 0, 0, 0]), // 0b001001
    lookup_row_u8([2, 4, 0, 0, 0, 0, 0, 0]), // 0b001010
    lookup_row_u8([1, 2, 4, 0, 0, 0, 0, 0]), // 0b001011
    lookup_row_u8([3, 4, 0, 0, 0, 0, 0, 0]), // 0b001100
    lookup_row_u8([1, 3, 4, 0, 0, 0, 0, 0]), // 0b001101
    lookup_row_u8([2, 3, 4, 0, 0, 0, 0, 0]), // 0b001110
    lookup_row_u8([1, 2, 3, 4, 0, 0, 0, 0]), // 0b001111
    lookup_row_u8([5, 0, 0, 0, 0, 0, 0, 0]), // 0b010000
    lookup_row_u8([1, 5, 0, 0, 0, 0, 0, 0]), // 0b010001
    lookup_row_u8([2, 5, 0, 0, 0, 0, 0, 0]), // 0b010010
    lookup_row_u8([1, 2, 5, 0, 0, 0, 0, 0]), // 0b010011
    lookup_row_u8([3, 5, 0, 0, 0, 0, 0, 0]), // 0b010100
    lookup_row_u8([1, 3, 5, 0, 0, 0, 0, 0]), // 0b010101
    lookup_row_u8([2, 3, 5, 0, 0, 0, 0, 0]), // 0b010110
    lookup_row_u8([1, 2, 3, 5, 0, 0, 0, 0]), // 0b010111
    lookup_row_u8([4, 5, 0, 0, 0, 0, 0, 0]), // 0b011000
    lookup_row_u8([1, 4, 5, 0, 0, 0, 0, 0]), // 0b011001
    lookup_row_u8([2, 4, 5, 0, 0, 0, 0, 0]), // 0b011010
    lookup_row_u8([1, 2, 4, 5, 0, 0, 0, 0]), // 0b011011
    lookup_row_u8([3, 4, 5, 0, 0, 0, 0, 0]), // 0b011100
    lookup_row_u8([1, 3, 4, 5, 0, 0, 0, 0]), // 0b011101
    lookup_row_u8([2, 3, 4, 5, 0, 0, 0, 0]), // 0b011110
    lookup_row_u8([1, 2, 3, 4, 5, 0, 0, 0]), // 0b011111
    lookup_row_u8([6, 0, 0, 0, 0, 0, 0, 0]), // 0b100000
    lookup_row_u8([1, 6, 0, 0, 0, 0, 0, 0]), // 0b100001
    lookup_row_u8([2, 6, 0, 0, 0, 0, 0, 0]), // 0b100010
    lookup_row_u8([1, 2, 6, 0, 0, 0, 0, 0]), // 0b100011
    lookup_row_u8([3, 6, 0, 0, 0, 0, 0, 0]), // 0b100100
    lookup_row_u8([1, 3, 6, 0, 0, 0, 0, 0]), // 0b100101
    lookup_row_u8([2, 3, 6, 0, 0, 0, 0, 0]), // 0b100110
    lookup_row_u8([1, 2, 3, 6, 0, 0, 0, 0]), // 0b100111
    lookup_row_u8([4, 6, 0, 0, 0, 0, 0, 0]), // 0b101000
    lookup_row_u8([1, 4, 6, 0, 0, 0, 0, 0]), // 0b101001
    lookup_row_u8([2, 4, 6, 0, 0, 0, 0, 0]), // 0b101010
    lookup_row_u8([1, 2, 4, 6, 0, 0, 0, 0]), // 0b101011
    lookup_row_u8([3, 4, 6, 0, 0, 0, 0, 0]), // 0b101100
    lookup_row_u8([1, 3, 4, 6, 0, 0, 0, 0]), // 0b101101
    lookup_row_u8([2, 3, 4, 6, 0, 0, 0, 0]), // 0b101110
    lookup_row_u8([1, 2, 3, 4, 6, 0, 0, 0]), // 0b101111
    lookup_row_u8([5, 6, 0, 0, 0, 0, 0, 0]), // 0b110000
    lookup_row_u8([1, 5, 6, 0, 0, 0, 0, 0]), // 0b110001
    lookup_row_u8([2, 5, 6, 0, 0, 0, 0, 0]), // 0b110010
    lookup_row_u8([1, 2, 5, 6, 0, 0, 0, 0]), // 0b110011
    lookup_row_u8([3, 5, 6, 0, 0, 0, 0, 0]), // 0b110100
    lookup_row_u8([1, 3, 5, 6, 0, 0, 0, 0]), // 0b110101
    lookup_row_u8([2, 3, 5, 6, 0, 0, 0, 0]), // 0b110110
    lookup_row_u8([1, 2, 3, 5, 6, 0, 0, 0]), // 0b110111
    lookup_row_u8([4, 5, 6, 0, 0, 0, 0, 0]), // 0b111000
    lookup_row_u8([1, 4, 5, 6, 0, 0, 0, 0]), // 0b111001
    lookup_row_u8([2, 4, 5, 6, 0, 0, 0, 0]), // 0b111010
    lookup_row_u8([1, 2, 4, 5, 6, 0, 0, 0]), // 0b111011
    lookup_row_u8([3, 4, 5, 6, 0, 0, 0, 0]), // 0b111100
    lookup_row_u8([1, 3, 4, 5, 6, 0, 0, 0]), // 0b111101
    lookup_row_u8([2, 3, 4, 5, 6, 0, 0, 0]), // 0b111110
    lookup_row_u8([1, 2, 3, 4, 5, 6, 0, 0]), // 0b111111
    lookup_row_u8([7, 0, 0, 0, 0, 0, 0, 0]), // 0b1000000
    lookup_row_u8([1, 7, 0, 0, 0, 0, 0, 0]), // 0b1000001
    lookup_row_u8([2, 7, 0, 0, 0, 0, 0, 0]), // 0b1000010
    lookup_row_u8([1, 2, 7, 0, 0, 0, 0, 0]), // 0b1000011
    lookup_row_u8([3, 7, 0, 0, 0, 0, 0, 0]), // 0b1000100
    lookup_row_u8([1, 3, 7, 0, 0, 0, 0, 0]), // 0b1000101
    lookup_row_u8([2, 3, 7, 0, 0, 0, 0, 0]), // 0b1000110
    lookup_row_u8([1, 2, 3, 7, 0, 0, 0, 0]), // 0b1000111
    lookup_row_u8([4, 7, 0, 0, 0, 0, 0, 0]), // 0b1001000
    lookup_row_u8([1, 4, 7, 0, 0, 0, 0, 0]), // 0b1001001
    lookup_row_u8([2, 4, 7, 0, 0, 0, 0, 0]), // 0b1001010
    lookup_row_u8([1, 2, 4, 7, 0, 0, 0, 0]), // 0b1001011
    lookup_row_u8([3, 4, 7, 0, 0, 0, 0, 0]), // 0b1001100
    lookup_row_u8([1, 3, 4, 7, 0, 0, 0, 0]), // 0b1001101
    lookup_row_u8([2, 3, 4, 7, 0, 0, 0, 0]), // 0b1001110
    lookup_row_u8([1, 2, 3, 4, 7, 0, 0, 0]), // 0b1001111
    lookup_row_u8([5, 7, 0, 0, 0, 0, 0, 0]), // 0b1010000
    lookup_row_u8([1, 5, 7, 0, 0, 0, 0, 0]), // 0b1010001
    lookup_row_u8([2, 5, 7, 0, 0, 0, 0, 0]), // 0b1010010
    lookup_row_u8([1, 2, 5, 7, 0, 0, 0, 0]), // 0b1010011
    lookup_row_u8([3, 5, 7, 0, 0, 0, 0, 0]), // 0b1010100
    lookup_row_u8([1, 3, 5, 7, 0, 0, 0, 0]), // 0b1010101
    lookup_row_u8([2, 3, 5, 7, 0, 0, 0, 0]), // 0b1010110
    lookup_row_u8([1, 2, 3, 5, 7, 0, 0, 0]), // 0b1010111
    lookup_row_u8([4, 5, 7, 0, 0, 0, 0, 0]), // 0b1011000
    lookup_row_u8([1, 4, 5, 7, 0, 0, 0, 0]), // 0b1011001
    lookup_row_u8([2, 4, 5, 7, 0, 0, 0, 0]), // 0b1011010
    lookup_row_u8([1, 2, 4, 5, 7, 0, 0, 0]), // 0b1011011
    lookup_row_u8([3, 4, 5, 7, 0, 0, 0, 0]), // 0b1011100
    lookup_row_u8([1, 3, 4, 5, 7, 0, 0, 0]), // 0b1011101
    lookup_row_u8([2, 3, 4, 5, 7, 0, 0, 0]), // 0b1011110
    lookup_row_u8([1, 2, 3, 4, 5, 7, 0, 0]), // 0b1011111
    lookup_row_u8([6, 7, 0, 0, 0, 0, 0, 0]), // 0b1100000
    lookup_row_u8([1, 6, 7, 0, 0, 0, 0, 0]), // 0b1100001
    lookup_row_u8([2, 6, 7, 0, 0, 0, 0, 0]), // 0b1100010
    lookup_row_u8([1, 2, 6, 7, 0, 0, 0, 0]), // 0b1100011
    lookup_row_u8([3, 6, 7, 0, 0, 0, 0, 0]), // 0b1100100
    lookup_row_u8([1, 3, 6, 7, 0, 0, 0, 0]), // 0b1100101
    lookup_row_u8([2, 3, 6, 7, 0, 0, 0, 0]), // 0b1100110
    lookup_row_u8([1, 2, 3, 6, 7, 0, 0, 0]), // 0b1100111
    lookup_row_u8([4, 6, 7, 0, 0, 0, 0, 0]), // 0b1101000
    lookup_row_u8([1, 4, 6, 7, 0, 0, 0, 0]), // 0b1101001
    lookup_row_u8([2, 4, 6, 7, 0, 0, 0, 0]), // 0b1101010
    lookup_row_u8([1, 2, 4, 6, 7, 0, 0, 0]), // 0b1101011
    lookup_row_u8([3, 4, 6, 7, 0, 0, 0, 0]), // 0b1101100
    lookup_row_u8([1, 3, 4, 6, 7, 0, 0, 0]), // 0b1101101
    lookup_row_u8([2, 3, 4, 6, 7, 0, 0, 0]), // 0b1101110
    lookup_row_u8([1, 2, 3, 4, 6, 7, 0, 0]), // 0b1101111
    lookup_row_u8([5, 6, 7, 0, 0, 0, 0, 0]), // 0b1110000
    lookup_row_u8([1, 5, 6, 7, 0, 0, 0, 0]), // 0b1110001
    lookup_row_u8([2, 5, 6, 7, 0, 0, 0, 0]), // 0b1110010
    lookup_row_u8([1, 2, 5, 6, 7, 0, 0, 0]), // 0b1110011
    lookup_row_u8([3, 5, 6, 7, 0, 0, 0, 0]), // 0b1110100
    lookup_row_u8([1, 3, 5, 6, 7, 0, 0, 0]), // 0b1110101
    lookup_row_u8([2, 3, 5, 6, 7, 0, 0, 0]), // 0b1110110
    lookup_row_u8([1, 2, 3, 5, 6, 7, 0, 0]), // 0b1110111
    lookup_row_u8([4, 5, 6, 7, 0, 0, 0, 0]), // 0b1111000
    lookup_row_u8([1, 4, 5, 6, 7, 0, 0, 0]), // 0b1111001
    lookup_row_u8([2, 4, 5, 6, 7, 0, 0, 0]), // 0b1111010
    lookup_row_u8([1, 2, 4, 5, 6, 7, 0, 0]), // 0b1111011
    lookup_row_u8([3, 4, 5, 6, 7, 0, 0, 0]), // 0b1111100
    lookup_row_u8([1, 3, 4, 5, 6, 7, 0, 0]), // 0b1111101
    lookup_row_u8([2, 3, 4, 5, 6, 7, 0, 0]), // 0b1111110
    lookup_row_u8([1, 2, 3, 4, 5, 6, 7, 0]), // 0b1111111
    lookup_row_u8([8, 0, 0, 0, 0, 0, 0, 0]), // 0b10000000
    lookup_row_u8([1, 8, 0, 0, 0, 0, 0, 0]), // 0b10000001
    lookup_row_u8([2, 8, 0, 0, 0, 0, 0, 0]), // 0b10000010
    lookup_row_u8([1, 2, 8, 0, 0, 0, 0, 0]), // 0b10000011
    lookup_row_u8([3, 8, 0, 0, 0, 0, 0, 0]), // 0b10000100
    lookup_row_u8([1, 3, 8, 0, 0, 0, 0, 0]), // 0b10000101
    lookup_row_u8([2, 3, 8, 0, 0, 0, 0, 0]), // 0b10000110
    lookup_row_u8([1, 2, 3, 8, 0, 0, 0, 0]), // 0b10000111
    lookup_row_u8([4, 8, 0, 0, 0, 0, 0, 0]), // 0b10001000
    lookup_row_u8([1, 4, 8, 0, 0, 0, 0, 0]), // 0b10001001
    lookup_row_u8([2, 4, 8, 0, 0, 0, 0, 0]), // 0b10001010
    lookup_row_u8([1, 2, 4, 8, 0, 0, 0, 0]), // 0b10001011
    lookup_row_u8([3, 4, 8, 0, 0, 0, 0, 0]), // 0b10001100
    lookup_row_u8([1, 3, 4, 8, 0, 0, 0, 0]), // 0b10001101
    lookup_row_u8([2, 3, 4, 8, 0, 0, 0, 0]), // 0b10001110
    lookup_row_u8([1, 2, 3, 4, 8, 0, 0, 0]), // 0b10001111
    lookup_row_u8([5, 8, 0, 0, 0, 0, 0, 0]), // 0b10010000
    lookup_row_u8([1, 5, 8, 0, 0, 0, 0, 0]), // 0b10010001
    lookup_row_u8([2, 5, 8, 0, 0, 0, 0, 0]), // 0b10010010
    lookup_row_u8([1, 2, 5, 8, 0, 0, 0, 0]), // 0b10010011
    lookup_row_u8([3, 5, 8, 0, 0, 0, 0, 0]), // 0b10010100
    lookup_row_u8([1, 3, 5, 8, 0, 0, 0, 0]), // 0b10010101
    lookup_row_u8([2, 3, 5, 8, 0, 0, 0, 0]), // 0b10010110
    lookup_row_u8([1, 2, 3, 5, 8, 0, 0, 0]), // 0b10010111
    lookup_row_u8([4, 5, 8, 0, 0, 0, 0, 0]), // 0b10011000
    lookup_row_u8([1, 4, 5, 8, 0, 0, 0, 0]), // 0b10011001
    lookup_row_u8([2, 4, 5, 8, 0, 0, 0, 0]), // 0b10011010
    lookup_row_u8([1, 2, 4, 5, 8, 0, 0, 0]), // 0b10011011
    lookup_row_u8([3, 4, 5, 8, 0, 0, 0, 0]), // 0b10011100
    lookup_row_u8([1, 3, 4, 5, 8, 0, 0, 0]), // 0b10011101
    lookup_row_u8([2, 3, 4, 5, 8, 0, 0, 0]), // 0b10011110
    lookup_row_u8([1, 2, 3, 4, 5, 8, 0, 0]), // 0b10011111
    lookup_row_u8([6, 8, 0, 0, 0, 0, 0, 0]), // 0b10100000
    lookup_row_u8([1, 6, 8, 0, 0, 0, 0, 0]), // 0b10100001
    lookup_row_u8([2, 6, 8, 0, 0, 0, 0, 0]), // 0b10100010
    lookup_row_u8([1, 2, 6, 8, 0, 0, 0, 0]), // 0b10100011
    lookup_row_u8([3, 6, 8, 0, 0, 0, 0, 0]), // 0b10100100
    lookup_row_u8([1, 3, 6, 8, 0, 0, 0, 0]), // 0b10100101
    lookup_row_u8([2, 3, 6, 8, 0, 0, 0, 0]), // 0b10100110
    lookup_row_u8([1, 2, 3, 6, 8, 0, 0, 0]), // 0b10100111
    lookup_row_u8([4, 6, 8, 0, 0, 0, 0, 0]), // 0b10101000
    lookup_row_u8([1, 4, 6, 8, 0, 0, 0, 0]), // 0b10101001
    lookup_row_u8([2, 4, 6, 8, 0, 0, 0, 0]), // 0b10101010
    lookup_row_u8([1, 2, 4, 6, 8, 0, 0, 0]), // 0b10101011
    lookup_row_u8([3, 4, 6, 8, 0, 0, 0, 0]), // 0b10101100
    lookup_row_u8([1, 3, 4, 6, 8, 0, 0, 0]), // 0b10101101
    lookup_row_u8([2, 3, 4, 6, 8, 0, 0, 0]), // 0b10101110
    lookup_row_u8([1, 2, 3, 4, 6, 8, 0, 0]), // 0b10101111
    lookup_row_u8([5, 6, 8, 0, 0, 0, 0, 0]), // 0b10110000
    lookup_row_u8([1, 5, 6, 8, 0, 0, 0, 0]), // 0b10110001
    lookup_row_u8([2, 5, 6, 8, 0, 0, 0, 0]), // 0b10110010
    lookup_row_u8([1, 2, 5, 6, 8, 0, 0, 0]), // 0b10110011
    lookup_row_u8([3, 5, 6, 8, 0, 0, 0, 0]), // 0b10110100
    lookup_row_u8([1, 3, 5, 6, 8, 0, 0, 0]), // 0b10110101
    lookup_row_u8([2, 3, 5, 6, 8, 0, 0, 0]), // 0b10110110
    lookup_row_u8([1, 2, 3, 5, 6, 8, 0, 0]), // 0b10110111
    lookup_row_u8([4, 5, 6, 8, 0, 0, 0, 0]), // 0b10111000
    lookup_row_u8([1, 4, 5, 6, 8, 0, 0, 0]), // 0b10111001
    lookup_row_u8([2, 4, 5, 6, 8, 0, 0, 0]), // 0b10111010
    lookup_row_u8([1, 2, 4, 5, 6, 8, 0, 0]), // 0b10111011
    lookup_row_u8([3, 4, 5, 6, 8, 0, 0, 0]), // 0b10111100
    lookup_row_u8([1, 3, 4, 5, 6, 8, 0, 0]), // 0b10111101
    lookup_row_u8([2, 3, 4, 5, 6, 8, 0, 0]), // 0b10111110
    lookup_row_u8([1, 2, 3, 4, 5, 6, 8, 0]), // 0b10111111
    lookup_row_u8([7, 8, 0, 0, 0, 0, 0, 0]), // 0b11000000
    lookup_row_u8([1, 7, 8, 0, 0, 0, 0, 0]), // 0b11000001
    lookup_row_u8([2, 7, 8, 0, 0, 0, 0, 0]), // 0b11000010
    lookup_row_u8([1, 2, 7, 8, 0, 0, 0, 0]), // 0b11000011
    lookup_row_u8([3, 7, 8, 0, 0, 0, 0, 0]), // 0b11000100
    lookup_row_u8([1, 3, 7, 8, 0, 0, 0, 0]), // 0b11000101
    lookup_row_u8([2, 3, 7, 8, 0, 0, 0, 0]), // 0b11000110
    lookup_row_u8([1, 2, 3, 7, 8, 0, 0, 0]), // 0b11000111
    lookup_row_u8([4, 7, 8, 0, 0, 0, 0, 0]), // 0b11001000
    lookup_row_u8([1, 4, 7, 8, 0, 0, 0, 0]), // 0b11001001
    lookup_row_u8([2, 4, 7, 8, 0, 0, 0, 0]), // 0b11001010
    lookup_row_u8([1, 2, 4, 7, 8, 0, 0, 0]), // 0b11001011
    lookup_row_u8([3, 4, 7, 8, 0, 0, 0, 0]), // 0b11001100
    lookup_row_u8([1, 3, 4, 7, 8, 0, 0, 0]), // 0b11001101
    lookup_row_u8([2, 3, 4, 7, 8, 0, 0, 0]), // 0b11001110
    lookup_row_u8([1, 2, 3, 4, 7, 8, 0, 0]), // 0b11001111
    lookup_row_u8([5, 7, 8, 0, 0, 0, 0, 0]), // 0b11010000
    lookup_row_u8([1, 5, 7, 8, 0, 0, 0, 0]), // 0b11010001
    lookup_row_u8([2, 5, 7, 8, 0, 0, 0, 0]), // 0b11010010
    lookup_row_u8([1, 2, 5, 7, 8, 0, 0, 0]), // 0b11010011
    lookup_row_u8([3, 5, 7, 8, 0, 0, 0, 0]), // 0b11010100
    lookup_row_u8([1, 3, 5, 7, 8, 0, 0, 0]), // 0b11010101
    lookup_row_u8([2, 3, 5, 7, 8, 0, 0, 0]), // 0b11010110
    lookup_row_u8([1, 2, 3, 5, 7, 8, 0, 0]), // 0b11010111
    lookup_row_u8([4, 5, 7, 8, 0, 0, 0, 0]), // 0b11011000
    lookup_row_u8([1, 4, 5, 7, 8, 0, 0, 0]), // 0b11011001
    lookup_row_u8([2, 4, 5, 7, 8, 0, 0, 0]), // 0b11011010
    lookup_row_u8([1, 2, 4, 5, 7, 8, 0, 0]), // 0b11011011
    lookup_row_u8([3, 4, 5, 7, 8, 0, 0, 0]), // 0b11011100
    lookup_row_u8([1, 3, 4, 5, 7, 8, 0, 0]), // 0b11011101
    lookup_row_u8([2, 3, 4, 5, 7, 8, 0, 0]), // 0b11011110
    lookup_row_u8([1, 2, 3, 4, 5, 7, 8, 0]), // 0b11011111
    lookup_row_u8([6, 7, 8, 0, 0, 0, 0, 0]), // 0b11100000
    lookup_row_u8([1, 6, 7, 8, 0, 0, 0, 0]), // 0b11100001
    lookup_row_u8([2, 6, 7, 8, 0, 0, 0, 0]), // 0b11100010
    lookup_row_u8([1, 2, 6, 7, 8, 0, 0, 0]), // 0b11100011
    lookup_row_u8([3, 6, 7, 8, 0, 0, 0, 0]), // 0b11100100
    lookup_row_u8([1, 3, 6, 7, 8, 0, 0, 0]), // 0b11100101
    lookup_row_u8([2, 3, 6, 7, 8, 0, 0, 0]), // 0b11100110
    lookup_row_u8([1, 2, 3, 6, 7, 8, 0, 0]), // 0b11100111
    lookup_row_u8([4, 6, 7, 8, 0, 0, 0, 0]), // 0b11101000
    lookup_row_u8([1, 4, 6, 7, 8, 0, 0, 0]), // 0b11101001
    lookup_row_u8([2, 4, 6, 7, 8, 0, 0, 0]), // 0b11101010
    lookup_row_u8([1, 2, 4, 6, 7, 8, 0, 0]), // 0b11101011
    lookup_row_u8([3, 4, 6, 7, 8, 0, 0, 0]), // 0b11101100
    lookup_row_u8([1, 3, 4, 6, 7, 8, 0, 0]), // 0b11101101
    lookup_row_u8([2, 3, 4, 6, 7, 8, 0, 0]), // 0b11101110
    lookup_row_u8([1, 2, 3, 4, 6, 7, 8, 0]), // 0b11101111
    lookup_row_u8([5, 6, 7, 8, 0, 0, 0, 0]), // 0b11110000
    lookup_row_u8([1, 5, 6, 7, 8, 0, 0, 0]), // 0b11110001
    lookup_row_u8([2, 5, 6, 7, 8, 0, 0, 0]), // 0b11110010
    lookup_row_u8([1, 2, 5, 6, 7, 8, 0, 0]), // 0b11110011
    lookup_row_u8([3, 5, 6, 7, 8, 0, 0, 0]), // 0b11110100
    lookup_row_u8([1, 3, 5, 6, 7, 8, 0, 0]), // 0b11110101
    lookup_row_u8([2, 3, 5, 6, 7, 8, 0, 0]), // 0b11110110
    lookup_row_u8([1, 2, 3, 5, 6, 7, 8, 0]), // 0b11110111
    lookup_row_u8([4, 5, 6, 7, 8, 0, 0, 0]), // 0b11111000
    lookup_row_u8([1, 4, 5, 6, 7, 8, 0, 0]), // 0b11111001
    lookup_row_u8([2, 4, 5, 6, 7, 8, 0, 0]), // 0b11111010
    lookup_row_u8([1, 2, 4, 5, 6, 7, 8, 0]), // 0b11111011
    lookup_row_u8([3, 4, 5, 6, 7, 8, 0, 0]), // 0b11111100
    lookup_row_u8([1, 3, 4, 5, 6, 7, 8, 0]), // 0b11111101
    lookup_row_u8([2, 3, 4, 5, 6, 7, 8, 0]), // 0b11111110
    lookup_row_u8([1, 2, 3, 4, 5, 6, 7, 8]), // 0b11111111
];

    // size: 256 * 32B = 8KiB
    #[allow(unused)]
#[rustfmt::skip]
static INDEX_LOOKUP_TABLE: [IndexLookup; 256] = [
    IndexLookup([0, 0, 0, 0, 0, 0, 0, 0]), // 0b000000
    IndexLookup([1, 0, 0, 0, 0, 0, 0, 0]), // 0b000001
    IndexLookup([2, 0, 0, 0, 0, 0, 0, 0]), // 0b000010
    IndexLookup([1, 2, 0, 0, 0, 0, 0, 0]), // 0b000011
    IndexLookup([3, 0, 0, 0, 0, 0, 0, 0]), // 0b000100
    IndexLookup([1, 3, 0, 0, 0, 0, 0, 0]), // 0b000101
    IndexLookup([2, 3, 0, 0, 0, 0, 0, 0]), // 0b000110
    IndexLookup([1, 2, 3, 0, 0, 0, 0, 0]), // 0b000111
    IndexLookup([4, 0, 0, 0, 0, 0, 0, 0]), // 0b001000
    IndexLookup([1, 4, 0, 0, 0, 0, 0, 0]), // 0b001001
    IndexLookup([2, 4, 0, 0, 0, 0, 0, 0]), // 0b001010
    IndexLookup([1, 2, 4, 0, 0, 0, 0, 0]), // 0b001011
    IndexLookup([3, 4, 0, 0, 0, 0, 0, 0]), // 0b001100
    IndexLookup([1, 3, 4, 0, 0, 0, 0, 0]), // 0b001101
    IndexLookup([2, 3, 4, 0, 0, 0, 0, 0]), // 0b001110
    IndexLookup([1, 2, 3, 4, 0, 0, 0, 0]), // 0b001111
    IndexLookup([5, 0, 0, 0, 0, 0, 0, 0]), // 0b010000
    IndexLookup([1, 5, 0, 0, 0, 0, 0, 0]), // 0b010001
    IndexLookup([2, 5, 0, 0, 0, 0, 0, 0]), // 0b010010
    IndexLookup([1, 2, 5, 0, 0, 0, 0, 0]), // 0b010011
    IndexLookup([3, 5, 0, 0, 0, 0, 0, 0]), // 0b010100
    IndexLookup([1, 3, 5, 0, 0, 0, 0, 0]), // 0b010101
    IndexLookup([2, 3, 5, 0, 0, 0, 0, 0]), // 0b010110
    IndexLookup([1, 2, 3, 5, 0, 0, 0, 0]), // 0b010111
    IndexLookup([4, 5, 0, 0, 0, 0, 0, 0]), // 0b011000
    IndexLookup([1, 4, 5, 0, 0, 0, 0, 0]), // 0b011001
    IndexLookup([2, 4, 5, 0, 0, 0, 0, 0]), // 0b011010
    IndexLookup([1, 2, 4, 5, 0, 0, 0, 0]), // 0b011011
    IndexLookup([3, 4, 5, 0, 0, 0, 0, 0]), // 0b011100
    IndexLookup([1, 3, 4, 5, 0, 0, 0, 0]), // 0b011101
    IndexLookup([2, 3, 4, 5, 0, 0, 0, 0]), // 0b011110
    IndexLookup([1, 2, 3, 4, 5, 0, 0, 0]), // 0b011111
    IndexLookup([6, 0, 0, 0, 0, 0, 0, 0]), // 0b100000
    IndexLookup([1, 6, 0, 0, 0, 0, 0, 0]), // 0b100001
    IndexLookup([2, 6, 0, 0, 0, 0, 0, 0]), // 0b100010
    IndexLookup([1, 2, 6, 0, 0, 0, 0, 0]), // 0b100011
    IndexLookup([3, 6, 0, 0, 0, 0, 0, 0]), // 0b100100
    IndexLookup([1, 3, 6, 0, 0, 0, 0, 0]), // 0b100101
    IndexLookup([2, 3, 6, 0, 0, 0, 0, 0]), // 0b100110
    IndexLookup([1, 2, 3, 6, 0, 0, 0, 0]), // 0b100111
    IndexLookup([4, 6, 0, 0, 0, 0, 0, 0]), // 0b101000
    IndexLookup([1, 4, 6, 0, 0, 0, 0, 0]), // 0b101001
    IndexLookup([2, 4, 6, 0, 0, 0, 0, 0]), // 0b101010
    IndexLookup([1, 2, 4, 6, 0, 0, 0, 0]), // 0b101011
    IndexLookup([3, 4, 6, 0, 0, 0, 0, 0]), // 0b101100
    IndexLookup([1, 3, 4, 6, 0, 0, 0, 0]), // 0b101101
    IndexLookup([2, 3, 4, 6, 0, 0, 0, 0]), // 0b101110
    IndexLookup([1, 2, 3, 4, 6, 0, 0, 0]), // 0b101111
    IndexLookup([5, 6, 0, 0, 0, 0, 0, 0]), // 0b110000
    IndexLookup([1, 5, 6, 0, 0, 0, 0, 0]), // 0b110001
    IndexLookup([2, 5, 6, 0, 0, 0, 0, 0]), // 0b110010
    IndexLookup([1, 2, 5, 6, 0, 0, 0, 0]), // 0b110011
    IndexLookup([3, 5, 6, 0, 0, 0, 0, 0]), // 0b110100
    IndexLookup([1, 3, 5, 6, 0, 0, 0, 0]), // 0b110101
    IndexLookup([2, 3, 5, 6, 0, 0, 0, 0]), // 0b110110
    IndexLookup([1, 2, 3, 5, 6, 0, 0, 0]), // 0b110111
    IndexLookup([4, 5, 6, 0, 0, 0, 0, 0]), // 0b111000
    IndexLookup([1, 4, 5, 6, 0, 0, 0, 0]), // 0b111001
    IndexLookup([2, 4, 5, 6, 0, 0, 0, 0]), // 0b111010
    IndexLookup([1, 2, 4, 5, 6, 0, 0, 0]), // 0b111011
    IndexLookup([3, 4, 5, 6, 0, 0, 0, 0]), // 0b111100
    IndexLookup([1, 3, 4, 5, 6, 0, 0, 0]), // 0b111101
    IndexLookup([2, 3, 4, 5, 6, 0, 0, 0]), // 0b111110
    IndexLookup([1, 2, 3, 4, 5, 6, 0, 0]), // 0b111111
    IndexLookup([7, 0, 0, 0, 0, 0, 0, 0]), // 0b1000000
    IndexLookup([1, 7, 0, 0, 0, 0, 0, 0]), // 0b1000001
    IndexLookup([2, 7, 0, 0, 0, 0, 0, 0]), // 0b1000010
    IndexLookup([1, 2, 7, 0, 0, 0, 0, 0]), // 0b1000011
    IndexLookup([3, 7, 0, 0, 0, 0, 0, 0]), // 0b1000100
    IndexLookup([1, 3, 7, 0, 0, 0, 0, 0]), // 0b1000101
    IndexLookup([2, 3, 7, 0, 0, 0, 0, 0]), // 0b1000110
    IndexLookup([1, 2, 3, 7, 0, 0, 0, 0]), // 0b1000111
    IndexLookup([4, 7, 0, 0, 0, 0, 0, 0]), // 0b1001000
    IndexLookup([1, 4, 7, 0, 0, 0, 0, 0]), // 0b1001001
    IndexLookup([2, 4, 7, 0, 0, 0, 0, 0]), // 0b1001010
    IndexLookup([1, 2, 4, 7, 0, 0, 0, 0]), // 0b1001011
    IndexLookup([3, 4, 7, 0, 0, 0, 0, 0]), // 0b1001100
    IndexLookup([1, 3, 4, 7, 0, 0, 0, 0]), // 0b1001101
    IndexLookup([2, 3, 4, 7, 0, 0, 0, 0]), // 0b1001110
    IndexLookup([1, 2, 3, 4, 7, 0, 0, 0]), // 0b1001111
    IndexLookup([5, 7, 0, 0, 0, 0, 0, 0]), // 0b1010000
    IndexLookup([1, 5, 7, 0, 0, 0, 0, 0]), // 0b1010001
    IndexLookup([2, 5, 7, 0, 0, 0, 0, 0]), // 0b1010010
    IndexLookup([1, 2, 5, 7, 0, 0, 0, 0]), // 0b1010011
    IndexLookup([3, 5, 7, 0, 0, 0, 0, 0]), // 0b1010100
    IndexLookup([1, 3, 5, 7, 0, 0, 0, 0]), // 0b1010101
    IndexLookup([2, 3, 5, 7, 0, 0, 0, 0]), // 0b1010110
    IndexLookup([1, 2, 3, 5, 7, 0, 0, 0]), // 0b1010111
    IndexLookup([4, 5, 7, 0, 0, 0, 0, 0]), // 0b1011000
    IndexLookup([1, 4, 5, 7, 0, 0, 0, 0]), // 0b1011001
    IndexLookup([2, 4, 5, 7, 0, 0, 0, 0]), // 0b1011010
    IndexLookup([1, 2, 4, 5, 7, 0, 0, 0]), // 0b1011011
    IndexLookup([3, 4, 5, 7, 0, 0, 0, 0]), // 0b1011100
    IndexLookup([1, 3, 4, 5, 7, 0, 0, 0]), // 0b1011101
    IndexLookup([2, 3, 4, 5, 7, 0, 0, 0]), // 0b1011110
    IndexLookup([1, 2, 3, 4, 5, 7, 0, 0]), // 0b1011111
    IndexLookup([6, 7, 0, 0, 0, 0, 0, 0]), // 0b1100000
    IndexLookup([1, 6, 7, 0, 0, 0, 0, 0]), // 0b1100001
    IndexLookup([2, 6, 7, 0, 0, 0, 0, 0]), // 0b1100010
    IndexLookup([1, 2, 6, 7, 0, 0, 0, 0]), // 0b1100011
    IndexLookup([3, 6, 7, 0, 0, 0, 0, 0]), // 0b1100100
    IndexLookup([1, 3, 6, 7, 0, 0, 0, 0]), // 0b1100101
    IndexLookup([2, 3, 6, 7, 0, 0, 0, 0]), // 0b1100110
    IndexLookup([1, 2, 3, 6, 7, 0, 0, 0]), // 0b1100111
    IndexLookup([4, 6, 7, 0, 0, 0, 0, 0]), // 0b1101000
    IndexLookup([1, 4, 6, 7, 0, 0, 0, 0]), // 0b1101001
    IndexLookup([2, 4, 6, 7, 0, 0, 0, 0]), // 0b1101010
    IndexLookup([1, 2, 4, 6, 7, 0, 0, 0]), // 0b1101011
    IndexLookup([3, 4, 6, 7, 0, 0, 0, 0]), // 0b1101100
    IndexLookup([1, 3, 4, 6, 7, 0, 0, 0]), // 0b1101101
    IndexLookup([2, 3, 4, 6, 7, 0, 0, 0]), // 0b1101110
    IndexLookup([1, 2, 3, 4, 6, 7, 0, 0]), // 0b1101111
    IndexLookup([5, 6, 7, 0, 0, 0, 0, 0]), // 0b1110000
    IndexLookup([1, 5, 6, 7, 0, 0, 0, 0]), // 0b1110001
    IndexLookup([2, 5, 6, 7, 0, 0, 0, 0]), // 0b1110010
    IndexLookup([1, 2, 5, 6, 7, 0, 0, 0]), // 0b1110011
    IndexLookup([3, 5, 6, 7, 0, 0, 0, 0]), // 0b1110100
    IndexLookup([1, 3, 5, 6, 7, 0, 0, 0]), // 0b1110101
    IndexLookup([2, 3, 5, 6, 7, 0, 0, 0]), // 0b1110110
    IndexLookup([1, 2, 3, 5, 6, 7, 0, 0]), // 0b1110111
    IndexLookup([4, 5, 6, 7, 0, 0, 0, 0]), // 0b1111000
    IndexLookup([1, 4, 5, 6, 7, 0, 0, 0]), // 0b1111001
    IndexLookup([2, 4, 5, 6, 7, 0, 0, 0]), // 0b1111010
    IndexLookup([1, 2, 4, 5, 6, 7, 0, 0]), // 0b1111011
    IndexLookup([3, 4, 5, 6, 7, 0, 0, 0]), // 0b1111100
    IndexLookup([1, 3, 4, 5, 6, 7, 0, 0]), // 0b1111101
    IndexLookup([2, 3, 4, 5, 6, 7, 0, 0]), // 0b1111110
    IndexLookup([1, 2, 3, 4, 5, 6, 7, 0]), // 0b1111111
    IndexLookup([8, 0, 0, 0, 0, 0, 0, 0]), // 0b10000000
    IndexLookup([1, 8, 0, 0, 0, 0, 0, 0]), // 0b10000001
    IndexLookup([2, 8, 0, 0, 0, 0, 0, 0]), // 0b10000010
    IndexLookup([1, 2, 8, 0, 0, 0, 0, 0]), // 0b10000011
    IndexLookup([3, 8, 0, 0, 0, 0, 0, 0]), // 0b10000100
    IndexLookup([1, 3, 8, 0, 0, 0, 0, 0]), // 0b10000101
    IndexLookup([2, 3, 8, 0, 0, 0, 0, 0]), // 0b10000110
    IndexLookup([1, 2, 3, 8, 0, 0, 0, 0]), // 0b10000111
    IndexLookup([4, 8, 0, 0, 0, 0, 0, 0]), // 0b10001000
    IndexLookup([1, 4, 8, 0, 0, 0, 0, 0]), // 0b10001001
    IndexLookup([2, 4, 8, 0, 0, 0, 0, 0]), // 0b10001010
    IndexLookup([1, 2, 4, 8, 0, 0, 0, 0]), // 0b10001011
    IndexLookup([3, 4, 8, 0, 0, 0, 0, 0]), // 0b10001100
    IndexLookup([1, 3, 4, 8, 0, 0, 0, 0]), // 0b10001101
    IndexLookup([2, 3, 4, 8, 0, 0, 0, 0]), // 0b10001110
    IndexLookup([1, 2, 3, 4, 8, 0, 0, 0]), // 0b10001111
    IndexLookup([5, 8, 0, 0, 0, 0, 0, 0]), // 0b10010000
    IndexLookup([1, 5, 8, 0, 0, 0, 0, 0]), // 0b10010001
    IndexLookup([2, 5, 8, 0, 0, 0, 0, 0]), // 0b10010010
    IndexLookup([1, 2, 5, 8, 0, 0, 0, 0]), // 0b10010011
    IndexLookup([3, 5, 8, 0, 0, 0, 0, 0]), // 0b10010100
    IndexLookup([1, 3, 5, 8, 0, 0, 0, 0]), // 0b10010101
    IndexLookup([2, 3, 5, 8, 0, 0, 0, 0]), // 0b10010110
    IndexLookup([1, 2, 3, 5, 8, 0, 0, 0]), // 0b10010111
    IndexLookup([4, 5, 8, 0, 0, 0, 0, 0]), // 0b10011000
    IndexLookup([1, 4, 5, 8, 0, 0, 0, 0]), // 0b10011001
    IndexLookup([2, 4, 5, 8, 0, 0, 0, 0]), // 0b10011010
    IndexLookup([1, 2, 4, 5, 8, 0, 0, 0]), // 0b10011011
    IndexLookup([3, 4, 5, 8, 0, 0, 0, 0]), // 0b10011100
    IndexLookup([1, 3, 4, 5, 8, 0, 0, 0]), // 0b10011101
    IndexLookup([2, 3, 4, 5, 8, 0, 0, 0]), // 0b10011110
    IndexLookup([1, 2, 3, 4, 5, 8, 0, 0]), // 0b10011111
    IndexLookup([6, 8, 0, 0, 0, 0, 0, 0]), // 0b10100000
    IndexLookup([1, 6, 8, 0, 0, 0, 0, 0]), // 0b10100001
    IndexLookup([2, 6, 8, 0, 0, 0, 0, 0]), // 0b10100010
    IndexLookup([1, 2, 6, 8, 0, 0, 0, 0]), // 0b10100011
    IndexLookup([3, 6, 8, 0, 0, 0, 0, 0]), // 0b10100100
    IndexLookup([1, 3, 6, 8, 0, 0, 0, 0]), // 0b10100101
    IndexLookup([2, 3, 6, 8, 0, 0, 0, 0]), // 0b10100110
    IndexLookup([1, 2, 3, 6, 8, 0, 0, 0]), // 0b10100111
    IndexLookup([4, 6, 8, 0, 0, 0, 0, 0]), // 0b10101000
    IndexLookup([1, 4, 6, 8, 0, 0, 0, 0]), // 0b10101001
    IndexLookup([2, 4, 6, 8, 0, 0, 0, 0]), // 0b10101010
    IndexLookup([1, 2, 4, 6, 8, 0, 0, 0]), // 0b10101011
    IndexLookup([3, 4, 6, 8, 0, 0, 0, 0]), // 0b10101100
    IndexLookup([1, 3, 4, 6, 8, 0, 0, 0]), // 0b10101101
    IndexLookup([2, 3, 4, 6, 8, 0, 0, 0]), // 0b10101110
    IndexLookup([1, 2, 3, 4, 6, 8, 0, 0]), // 0b10101111
    IndexLookup([5, 6, 8, 0, 0, 0, 0, 0]), // 0b10110000
    IndexLookup([1, 5, 6, 8, 0, 0, 0, 0]), // 0b10110001
    IndexLookup([2, 5, 6, 8, 0, 0, 0, 0]), // 0b10110010
    IndexLookup([1, 2, 5, 6, 8, 0, 0, 0]), // 0b10110011
    IndexLookup([3, 5, 6, 8, 0, 0, 0, 0]), // 0b10110100
    IndexLookup([1, 3, 5, 6, 8, 0, 0, 0]), // 0b10110101
    IndexLookup([2, 3, 5, 6, 8, 0, 0, 0]), // 0b10110110
    IndexLookup([1, 2, 3, 5, 6, 8, 0, 0]), // 0b10110111
    IndexLookup([4, 5, 6, 8, 0, 0, 0, 0]), // 0b10111000
    IndexLookup([1, 4, 5, 6, 8, 0, 0, 0]), // 0b10111001
    IndexLookup([2, 4, 5, 6, 8, 0, 0, 0]), // 0b10111010
    IndexLookup([1, 2, 4, 5, 6, 8, 0, 0]), // 0b10111011
    IndexLookup([3, 4, 5, 6, 8, 0, 0, 0]), // 0b10111100
    IndexLookup([1, 3, 4, 5, 6, 8, 0, 0]), // 0b10111101
    IndexLookup([2, 3, 4, 5, 6, 8, 0, 0]), // 0b10111110
    IndexLookup([1, 2, 3, 4, 5, 6, 8, 0]), // 0b10111111
    IndexLookup([7, 8, 0, 0, 0, 0, 0, 0]), // 0b11000000
    IndexLookup([1, 7, 8, 0, 0, 0, 0, 0]), // 0b11000001
    IndexLookup([2, 7, 8, 0, 0, 0, 0, 0]), // 0b11000010
    IndexLookup([1, 2, 7, 8, 0, 0, 0, 0]), // 0b11000011
    IndexLookup([3, 7, 8, 0, 0, 0, 0, 0]), // 0b11000100
    IndexLookup([1, 3, 7, 8, 0, 0, 0, 0]), // 0b11000101
    IndexLookup([2, 3, 7, 8, 0, 0, 0, 0]), // 0b11000110
    IndexLookup([1, 2, 3, 7, 8, 0, 0, 0]), // 0b11000111
    IndexLookup([4, 7, 8, 0, 0, 0, 0, 0]), // 0b11001000
    IndexLookup([1, 4, 7, 8, 0, 0, 0, 0]), // 0b11001001
    IndexLookup([2, 4, 7, 8, 0, 0, 0, 0]), // 0b11001010
    IndexLookup([1, 2, 4, 7, 8, 0, 0, 0]), // 0b11001011
    IndexLookup([3, 4, 7, 8, 0, 0, 0, 0]), // 0b11001100
    IndexLookup([1, 3, 4, 7, 8, 0, 0, 0]), // 0b11001101
    IndexLookup([2, 3, 4, 7, 8, 0, 0, 0]), // 0b11001110
    IndexLookup([1, 2, 3, 4, 7, 8, 0, 0]), // 0b11001111
    IndexLookup([5, 7, 8, 0, 0, 0, 0, 0]), // 0b11010000
    IndexLookup([1, 5, 7, 8, 0, 0, 0, 0]), // 0b11010001
    IndexLookup([2, 5, 7, 8, 0, 0, 0, 0]), // 0b11010010
    IndexLookup([1, 2, 5, 7, 8, 0, 0, 0]), // 0b11010011
    IndexLookup([3, 5, 7, 8, 0, 0, 0, 0]), // 0b11010100
    IndexLookup([1, 3, 5, 7, 8, 0, 0, 0]), // 0b11010101
    IndexLookup([2, 3, 5, 7, 8, 0, 0, 0]), // 0b11010110
    IndexLookup([1, 2, 3, 5, 7, 8, 0, 0]), // 0b11010111
    IndexLookup([4, 5, 7, 8, 0, 0, 0, 0]), // 0b11011000
    IndexLookup([1, 4, 5, 7, 8, 0, 0, 0]), // 0b11011001
    IndexLookup([2, 4, 5, 7, 8, 0, 0, 0]), // 0b11011010
    IndexLookup([1, 2, 4, 5, 7, 8, 0, 0]), // 0b11011011
    IndexLookup([3, 4, 5, 7, 8, 0, 0, 0]), // 0b11011100
    IndexLookup([1, 3, 4, 5, 7, 8, 0, 0]), // 0b11011101
    IndexLookup([2, 3, 4, 5, 7, 8, 0, 0]), // 0b11011110
    IndexLookup([1, 2, 3, 4, 5, 7, 8, 0]), // 0b11011111
    IndexLookup([6, 7, 8, 0, 0, 0, 0, 0]), // 0b11100000
    IndexLookup([1, 6, 7, 8, 0, 0, 0, 0]), // 0b11100001
    IndexLookup([2, 6, 7, 8, 0, 0, 0, 0]), // 0b11100010
    IndexLookup([1, 2, 6, 7, 8, 0, 0, 0]), // 0b11100011
    IndexLookup([3, 6, 7, 8, 0, 0, 0, 0]), // 0b11100100
    IndexLookup([1, 3, 6, 7, 8, 0, 0, 0]), // 0b11100101
    IndexLookup([2, 3, 6, 7, 8, 0, 0, 0]), // 0b11100110
    IndexLookup([1, 2, 3, 6, 7, 8, 0, 0]), // 0b11100111
    IndexLookup([4, 6, 7, 8, 0, 0, 0, 0]), // 0b11101000
    IndexLookup([1, 4, 6, 7, 8, 0, 0, 0]), // 0b11101001
    IndexLookup([2, 4, 6, 7, 8, 0, 0, 0]), // 0b11101010
    IndexLookup([1, 2, 4, 6, 7, 8, 0, 0]), // 0b11101011
    IndexLookup([3, 4, 6, 7, 8, 0, 0, 0]), // 0b11101100
    IndexLookup([1, 3, 4, 6, 7, 8, 0, 0]), // 0b11101101
    IndexLookup([2, 3, 4, 6, 7, 8, 0, 0]), // 0b11101110
    IndexLookup([1, 2, 3, 4, 6, 7, 8, 0]), // 0b11101111
    IndexLookup([5, 6, 7, 8, 0, 0, 0, 0]), // 0b11110000
    IndexLookup([1, 5, 6, 7, 8, 0, 0, 0]), // 0b11110001
    IndexLookup([2, 5, 6, 7, 8, 0, 0, 0]), // 0b11110010
    IndexLookup([1, 2, 5, 6, 7, 8, 0, 0]), // 0b11110011
    IndexLookup([3, 5, 6, 7, 8, 0, 0, 0]), // 0b11110100
    IndexLookup([1, 3, 5, 6, 7, 8, 0, 0]), // 0b11110101
    IndexLookup([2, 3, 5, 6, 7, 8, 0, 0]), // 0b11110110
    IndexLookup([1, 2, 3, 5, 6, 7, 8, 0]), // 0b11110111
    IndexLookup([4, 5, 6, 7, 8, 0, 0, 0]), // 0b11111000
    IndexLookup([1, 4, 5, 6, 7, 8, 0, 0]), // 0b11111001
    IndexLookup([2, 4, 5, 6, 7, 8, 0, 0]), // 0b11111010
    IndexLookup([1, 2, 4, 5, 6, 7, 8, 0]), // 0b11111011
    IndexLookup([3, 4, 5, 6, 7, 8, 0, 0]), // 0b11111100
    IndexLookup([1, 3, 4, 5, 6, 7, 8, 0]), // 0b11111101
    IndexLookup([2, 3, 4, 5, 6, 7, 8, 0]), // 0b11111110
    IndexLookup([1, 2, 3, 4, 5, 6, 7, 8]), // 0b11111111
];
}

#[cfg(all(target_arch = "aarch64", feature = "aarch64-simd"))]
pub mod aarch64 {
    use std::arch::aarch64::{
        self, int32x4_t, vaddq_s32, vdupq_n_s32, vld1_dup_f64, vld1_s16, vld1q_s32, vmovl_s16,
        vst1_s32_x2, vst1q_s32,
    };

    pub unsafe fn bitmap_ones_simd(bitmap: &[u64], output: &mut Vec<u32>) {
        output.reserve(bitmap.len() * 64);
        let mut base_lo: int32x4_t = vdupq_n_s32(-1);
        let mut base_hi: int32x4_t = vdupq_n_s32(3);
        let add_8 = vdupq_n_s32(8);
        let add_64 = vdupq_n_s32(64);
        // let mut base_vec: __m256i = _mm256_set1_epi32(-1);
        // let add_8: __m256i = _mm256_set1_epi32(8);
        // let add_64: __m256i = _mm256_set1_epi32(64);
        let mut out = output.as_mut_ptr();
        for word in bitmap {
            let mut w = *word;
            if w == 0 {
                base_lo = vaddq_s32(base_lo, add_64);
                base_hi = vaddq_s32(base_hi, add_64);
                continue;
            }
            for _i in 0..8 {
                let byte = w as u8;
                w >>= 8;
                let lo = byte & 0xf;
                let hi = byte >> 4;
                let indexes_lo = vmovl_s16(vld1_s16(&LOOKUP[lo as usize] as *const i16));
                let indexes_hi = vmovl_s16(vld1_s16(&LOOKUP[hi as usize] as *const i16));
                let advance_a = lo.count_ones();
                let advance_b = hi.count_ones();

                let indexes_a = vaddq_s32(base_lo, indexes_lo);
                base_lo = vaddq_s32(base_lo, add_8);
                let indexes_b = vaddq_s32(base_hi, indexes_hi);
                base_hi = vaddq_s32(base_hi, add_8);

                vst1q_s32(out as *mut i32, indexes_a);
                out = out.add(advance_a as usize);
                vst1q_s32(out as *mut i32, indexes_b);
                out = out.add(advance_b as usize);
            }
        }
        let len = out.offset_from(output.as_ptr());
        output.set_len(len as usize);
    }

    #[test]
    fn extract_aarch64() {
        let num_bits = 64;
        let bitmap = super::random_bitmap(num_bits, 0.5);
        println!(
            "{:#b} {:.2}",
            bitmap[0],
            super::bitmap_ones(&bitmap) as f64 / num_bits as f64
        );

        let mut expected = Vec::with_capacity(num_bits);
        let mut faster = Vec::with_capacity(num_bits);
        super::bitmap_ones_naive(&bitmap, &mut expected);
        unsafe {
            bitmap_ones_simd(&bitmap, &mut faster);
        }
        assert_eq!(faster, expected);
        // let mut result = Vec::new();
        // unsafe { bitmap_ones_simd(&[0x1f30, 0x400], &mut result) };
        // println!("{:?}", result);
    }

    #[test]
    fn simd_test() {
        unsafe {
            let x = vld1q_s32(&DATA[0] as *const u8 as *const i32);
            println!("vld1q_s32 => {:x?}", x);
            let x = aarch64::vld1_s16(&DATA[0] as *const u8 as *const i16);
            println!("vld1_s16 => {:x?}", x);
            let y = aarch64::vmovl_s16(x);
            println!("vmovl_s16 => {:x?}", y);
        }
    }

    #[rustfmt::skip]
    pub const DATA: [u8; 128] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
        0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
        0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
        0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
    ];

    #[rustfmt::skip]
    pub const LOOKUP: [[i16; 4]; 16] = [
        [0, 0, 0, 0],  // 0000
        [1, 0, 0, 0],  // 0001
        [2, 0, 0, 0],  // 0010
        [1, 2, 0, 0],  // 0011
        [3, 0, 0, 0],  // 0100
        [1, 3, 0, 0],  // 0101
        [2, 3, 0, 0],  // 0110
        [1, 2, 3, 0],  // 0111
        [4, 0, 0, 0],  // 1000
        [1, 4, 0, 0],  // 1001
        [2, 4, 0, 0],  // 1010
        [1, 2, 4, 0],  // 1011
        [3, 4, 0, 0],  // 1100
        [1, 3, 4, 0],  // 1101
        [2, 3, 4, 0],  // 1110
        [1, 2, 3, 4],  // 1111
    ];
}
