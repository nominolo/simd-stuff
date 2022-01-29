/*!

```text
   index     0123456789abcdef
   bitmap    0001101011110101

   data         AB C DEFG H I

   mask      1010100110101110

   result        B   D F  H
   result_index  4   8 a  d
   delta         4   4 2  3

```

Constructing just the indexes is easy:

```text
   bitmap    0001101011110101  
   mask    & 1010100110101110  
      =>     0000100010100100
   unpack => 4, 8, 10, 13
```

The left-pack mask might be more complicated?

```ignore
    bitmap: 0101
    mask:   1110
    length: 2, mask: 1,0

    bitmap: 1111
    mask:   1010
    length: 4, mask: 1,0,1,0
```

Look at `pdep`/`pext`?  (Slow on Zen2 or older)





# Left-Packing

```text
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
Input    │  1 │ -1 │  5 │  3 │ -2 │  7 │ -1 │  3 │
         └────┴────┴────┴────┴────┴────┴────┴────┘
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
Mask     │  Y │  n │  Y │  Y │  n │  Y │  n │  Y │
         └────┴────┴────┴────┴────┴────┴────┴────┘
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
LeftPack │  1 │  5 │  3 │ XX │  7 │  3 │ XX │ XX │
         └────┴────┴────┴────┴────┴────┴────┴────┘
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
Output   │  1 │  5 │  3 │  7 │  3 │ XX │ XX │ XX │
         └────┴────┴────┴────┴────┴────┴────┴────┘
```

*/

use std::arch::x86_64::{__m128, __m128i, _mm_castps_si128, _mm_load_si128, _mm_movemask_ps, _mm_shuffle_epi8, _mm_shuffle_ps, _pdep_u32, _pext_u32};

/// # Safety
///
/// Uses SIMD
pub unsafe fn left_pack(mask: __m128, values: __m128) -> __m128i {
    let mask: i32 = _mm_movemask_ps(mask);
    let shuffle =
        _mm_load_si128(&LEFT_PACK_SHUFFLE[mask as usize].0 as *const u8 as *const __m128i);
    let vals = _mm_castps_si128(values);
    let packed: __m128i = _mm_shuffle_epi8(vals, shuffle);
    packed
}

#[test]
fn gen_left_pack_shuffle_table() {
    println!("#[rustfmt::skip]");
    println!("static LEFT_PACK_SHUFFLE: [Align16<[u8; 16]>; 16] = [");
    for mask in 0..16 {
        let mut row = vec![];
        for bit in 0..4 {
            if mask & (1 << bit) != 0 {
                for byte in bit * 4..bit * 4 + 4 {
                    row.push(byte);
                }
            }
        }
        row.resize(16, 0x80);
        print!("    Align16([");
        for v in row {
            if v < 0x80 {
                print!("{:4}, ", v);
            } else {
                print!("0x80, ");
            }
        }
        println!("]), // {:#06b}", mask);
    }
    println!("];")
}

#[rustfmt::skip]
static LEFT_PACK_SHUFFLE: [Align16<[u8; 16]>; 16] = [
    Align16([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0000       
    Align16([   0,    1,    2,    3, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0001       
    Align16([   4,    5,    6,    7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0010       
    Align16([   0,    1,    2,    3,    4,    5,    6,    7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0011       
    Align16([   8,    9,   10,   11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0100       
    Align16([   0,    1,    2,    3,    8,    9,   10,   11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0101       
    Align16([   4,    5,    6,    7,    8,    9,   10,   11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b0110       
    Align16([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11, 0x80, 0x80, 0x80, 0x80, ]), // 0b0111       
    Align16([  12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b1000       
    Align16([   0,    1,    2,    3,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b1001       
    Align16([   4,    5,    6,    7,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b1010       
    Align16([   0,    1,    2,    3,    4,    5,    6,    7,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, ]), // 0b1011       
    Align16([   8,    9,   10,   11,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, ]), // 0b1100       
    Align16([   0,    1,    2,    3,    8,    9,   10,   11,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, ]), // 0b1101       
    Align16([   4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15, 0x80, 0x80, 0x80, 0x80, ]), // 0b1110       
    Align16([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15, ]), // 0b1111       
];

#[repr(C, align(16))]
struct Align16<T>(pub T);



#[test]
fn test_pdep() {
    let a: u32 = 0b0001_1010_1111_0101;
    let mask: u32 = 0b1010_1001_1010_1110;
    let b = unsafe { _pdep_u32(a, mask) };
    println!("a     {:#018b}", a);
    println!("mask  {:#018b}", mask);
    println!("b     {:#018b}", b);
    let c = a & mask;
    let d = unsafe { _pdep_u32(c, a) };
    println!("a     {:#018b}", a);
    println!("mask  {:#018b}", mask);
    println!("c     {:#018b}", c);
    println!("d     {:#018b}", d);
}

#[test]
fn test_pext() {
    let inp: u32 = 0b0001_1010_1111_0101;
    let msk: u32 = 0b1010_1001_1010_1110;
    let res = unsafe { _pext_u32(msk, inp) };
    println!("inp  {:#018b}", inp);
    println!("msk  {:#018b}", msk);
    println!("res  {:#018b}", res);

    let inp: u32 = 0b0001_1011_1111_0110;
    let msk: u32 = 0b1010_1001_1010_1110;
    let res = unsafe { _pext_u32(msk, inp) };
    println!("inp  {:#018b}", inp);
    println!("msk  {:#018b}", msk);
    println!("res  {:#018b}", res);
}