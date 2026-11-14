/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Routines for IDCT
//!
//! Essentially we provide 2 routines for IDCT, a scalar implementation and a not super optimized
//! AVX2 one, i'll talk about them here.
//!
//! There are 2 reasons why we have the avx one
//! 1. No one compiles with -C target-features=avx2 hence binaries won't probably take advantage(even
//! if it exists).
//! 2. AVX employs zero short circuit in a way the scalar code cannot employ it.
//!     - AVX does this by checking for MCU's whose 63 AC coefficients are zero and if true, it writes
//!        values directly, if false, it goes the long way of calculating.
//!     -   Although this can be trivially implemented in the scalar version, it  generates code
//!         I'm not happy width(scalar version that basically loops and that is too many branches for me)
//!         The avx one does a better job of using bitwise or's with (`_mm256_or_si256`) which is magnitudes of faster
//!         than anything I could come up with
//!
//! The AVX code also has some cool transpose_u16 instructions which look so complicated to be cool
//! (spoiler alert, i barely understand how it works, that's why I credited the owner).
//!
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::module_name_repetitions,
    unused_parens,
    clippy::wildcard_imports
)]

use zune_core::log::debug;
use zune_core::options::DecoderOptions;

use crate::decoder::IDCTPtr;
use crate::idct::scalar::{idct_int, idct_int_1x1};

#[cfg(feature = "x86")]
pub mod avx2;
#[cfg(feature = "neon")]
pub mod neon;

pub mod scalar;
pub mod scaled;

/// The block an inverse DCT writes from the start of `out`: 8 × `v_scale` rows `stride` apart,
/// 8 × `h_scale` samples each (`h_scale` and `v_scale` are 1 or 2), or `None` if `out` is too
/// short for it. Every inverse DCT indexes its output with checks, so a caller that takes its
/// output from here returns an error for a block outside the buffer instead of panicking.
pub(crate) fn idct_block(
    out: &mut [i16], stride: usize, h_scale: usize, v_scale: usize
) -> Option<&mut [i16]> {
    let rows = 8 * v_scale;
    let width = 8 * h_scale;
    if stride < width {
        return None;
    }
    let len = (rows - 1) * stride + width;
    out.get_mut(..len)
}

/// Choose an appropriate IDCT function
#[allow(unused_variables)]
pub fn choose_idct_func(options: &DecoderOptions) -> IDCTPtr {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "x86")]
    {
        if options.use_avx2() {
            debug!("Using vector integer IDCT");
            return |a: &mut [i32; 64], b: &mut [i16], c: usize| {
                // SAFETY: `options.use_avx2()` only returns true if avx2 is supported.
                unsafe { avx2::idct_avx2(a,b,c) }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[cfg(feature = "neon")]
    {
        if options.use_neon() {
            debug!("Using vector integer IDCT");
            return |a: &mut [i32; 64], b: &mut [i16], c: usize| {
                // SAFETY: `options.use_neon()` only returns true if neon is supported.
                unsafe { neon::idct_neon(a,b,c) }
            };
        }
    }
    debug!("Using scalar integer IDCT");
    // use generic one
    return idct_int;
}

/// Choose a function to implement 4x4 IDCT.
///
/// These functions get the same input but have an extra contract: Only the first 4x4 block of
/// coefficients are non-zero. All other entries are zeroed.
///
/// **The callee must uphold that contract on return**
pub fn choose_idct_4x4_func(_options: &DecoderOptions) -> IDCTPtr {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "x86")]
    {
        if _options.use_avx2() {
            debug!("Using vector integer IDCT");
            return |a: &mut [i32; 64], b: &mut [i16], c: usize| {
                // SAFETY: `options.use_avx2()` only returns true if avx2 is supported.
                unsafe { avx2::idct_avx2_4x4(a,b,c) }
            };
        }
    }

    scalar::idct4x4
}

pub fn choose_idct_1x1_func(_: &DecoderOptions) -> IDCTPtr {
    // These are simple stores, no alternative implementation for now
    idct_int_1x1
}

#[cfg(test)]
#[allow(unreachable_code)]
#[allow(dead_code)]
mod tests {
    use super::*;

    #[test]
    fn idct_test0() {
        let stride = 8;
        let mut coeff = [10; 64];
        let mut coeff2 = [10; 64];
        let mut output_scalar = [0; 64];
        let mut output_vector = [0; 64];
        let idct_func = choose_idct_func(&DecoderOptions::new_fast());
        idct_func(&mut coeff, &mut output_vector, stride);
        idct_int(&mut coeff2, &mut output_scalar, stride);
        assert_eq!(output_scalar, output_vector, "IDCT and scalar do not match");
    }

    #[test]
    fn do_idct_test1() {
        let stride = 8;
        let mut coeff = [14; 64];
        let mut coeff2 = [14; 64];
        let mut output_scalar = [0; 64];
        let mut output_vector = [0; 64];
        let idct_func = choose_idct_func(&DecoderOptions::new_fast());
        idct_func(&mut coeff, &mut output_vector, stride);
        idct_int(&mut coeff2, &mut output_scalar, stride);
        assert_eq!(output_scalar, output_vector, "IDCT and scalar do not match");
    }

    #[test]
    fn do_idct_test2() {
        let stride = 8;
        let mut coeff = [0; 64];
        coeff[0] = 255;
        coeff[63] = -256;
        let mut coeff2 = coeff;
        let mut output_scalar = [0; 64];
        let mut output_vector = [0; 64];
        let idct_func = choose_idct_func(&DecoderOptions::new_fast());
        idct_func(&mut coeff, &mut output_vector, stride);
        idct_int(&mut coeff2, &mut output_scalar, stride);
        assert_eq!(output_scalar, output_vector, "IDCT and scalar do not match");
    }

    #[test]
    fn do_idct_zeros() {
        let stride = 8;
        let mut coeff = [0; 64];
        let mut coeff2 = [0; 64];
        let mut output_scalar = [0; 64];
        let mut output_vector = [0; 64];
        let idct_func = choose_idct_func(&DecoderOptions::new_fast());
        idct_func(&mut coeff, &mut output_vector, stride);
        idct_int(&mut coeff2, &mut output_scalar, stride);
        assert_eq!(output_scalar, output_vector, "IDCT and scalar do not match");
    }

    #[test]
    fn idct_4x4() {
        #[rustfmt::skip]
        const A: [i32; 32] = [
            -254, -7, 0, 0, 0, 0, 0, 0,
            7, 0, -30, 32,  0, 0, 0, 0,
            7, 0, -30, 32,  0, 0, 0, 0,
            7, 0, -30, 32,  0, 0, 0, 0,
        ];

        let v: Vec<IDCTPtr> = vec![
            choose_idct_func(&DecoderOptions::new_safe()),
            choose_idct_4x4_func(&DecoderOptions::new_safe()),
            choose_idct_func(&DecoderOptions::new_fast()),
            choose_idct_4x4_func(&DecoderOptions::new_fast()),
        ];
        let dct_names = vec![
            "safe idct",
            "safe idct 4x4",
            "fast idct",
            "fast idct 4x4",
        ];

        let mut color = vec![];

        for idct in v {
            let mut a = [0i32; 64];
            a[..32].copy_from_slice(&A);
            let mut b = [0i16; 64];

            idct(&mut a, &mut b, 8);

            color.push(b);
        }

        for (wnd, name) in color.windows(2).zip(&dct_names) {
            let [a, b] = wnd else { unreachable!() };
            assert_eq!(a, b, "{name}");
        }
    }

    #[test]
    fn idct_block_is_the_whole_block_or_none() {
        // 8 rows of 8 samples, 16 apart: the last row starts at 7 * 16.
        let mut out = vec![0i16; 7 * 16 + 8];
        assert_eq!(idct_block(&mut out, 16, 1, 1).map(|b| b.len()), Some(7 * 16 + 8));
        assert!(idct_block(&mut out[1..], 16, 1, 1).is_none());
        // Rows narrower than the block would overlap.
        assert!(idct_block(&mut out, 4, 1, 1).is_none());
        // An upsampling inverse DCT writes 16 rows of 16 samples.
        let mut out = vec![0i16; 15 * 32 + 16];
        assert!(idct_block(&mut out, 32, 2, 2).is_some());
        assert!(idct_block(&mut out[..15 * 32 + 15], 32, 2, 2).is_none());
        assert!(idct_block(&mut out, 8, 2, 1).is_none());
        // Every inverse DCT stays inside the block it is given.
        for (h, v) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            let stride = 8 * h + 3;
            let mut out = vec![0i16; (8 * v - 1) * stride + 8 * h];
            let block = idct_block(&mut out, stride, h, v).unwrap();
            // Only the first 4 x 4 coefficients are non-zero, as the 4 x 4 inverse DCT needs.
            let mut coefficients = [0i32; 64];
            for i in (0..32).filter(|i| i % 8 < 4) {
                coefficients[i] = 3;
            }
            if h * v > 1 {
                scaled::idct_scaled(&coefficients, block, stride, h, v);
            } else {
                for idct in [
                    choose_idct_func(&DecoderOptions::new_safe()),
                    choose_idct_func(&DecoderOptions::new_fast()),
                    choose_idct_4x4_func(&DecoderOptions::new_fast()),
                    choose_idct_1x1_func(&DecoderOptions::new_fast())
                ] {
                    idct(&mut coefficients, block, stride);
                }
            }
        }
    }
}
