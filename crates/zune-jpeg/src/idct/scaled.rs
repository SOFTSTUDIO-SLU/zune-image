/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! An inverse DCT that also upsamples: an 8×8 block of coefficients to 16×16, 16×8 or 8×16
//! samples (see `JpegDecoder::set_idct_upsampling`).
//!
//! The block's coefficients are taken as the lowest frequencies of a 16-point transform (the
//! others zero) in each doubled direction, so the result is the same cosine sum as the 8×8
//! inverse DCT, evaluated at twice the sample rate: every output sample sits halfway between
//! the 8×8 sample positions and their neighbours, and a flat block stays flat. This is the
//! upsampling libjpeg 7 and later do for 2× subsampled chroma (DCT scaling, `jidctint.c`,
//! `jpeg_idct_16x16` and its 16×8 and 8×16 relatives).
//!
//! The sums are computed in `f32` from cosine tables and rounded once, so the result is the
//! transform's value to within rounding (a value within about 0.01 of a half may round either
//! way); libjpeg's fixed-point butterflies can differ by one level.

/// `DOWN_8[y][v]`: the weight of frequency `v` at row `y` of eight.
const DOWN_8: [[f32; 8]; 8] = [
    [0.35355339059327373, 0.4903926402016152, 0.46193976625564337, 0.4157348061512726, 0.3535533905932738, 0.27778511650980114, 0.19134171618254492, 0.09754516100806417],
    [0.35355339059327373, 0.4157348061512726, 0.19134171618254492, -0.0975451610080641, -0.35355339059327373, -0.4903926402016152, -0.4619397662556434, -0.2777851165098011],
    [0.35355339059327373, 0.27778511650980114, -0.19134171618254486, -0.4903926402016152, -0.35355339059327384, 0.09754516100806415, 0.46193976625564326, 0.41573480615127273],
    [0.35355339059327373, 0.09754516100806417, -0.46193976625564337, -0.2777851165098011, 0.3535533905932737, 0.41573480615127273, -0.19134171618254495, -0.4903926402016153],
    [0.35355339059327373, -0.0975451610080641, -0.4619397662556434, 0.2777851165098009, 0.35355339059327384, -0.41573480615127256, -0.19134171618254528, 0.4903926402016152],
    [0.35355339059327373, -0.277785116509801, -0.19134171618254517, 0.4903926402016152, -0.35355339059327334, -0.09754516100806401, 0.46193976625564337, -0.4157348061512725],
    [0.35355339059327373, -0.4157348061512727, 0.191341716182545, 0.09754516100806439, -0.35355339059327356, 0.4903926402016153, -0.4619397662556432, 0.27778511650980076],
    [0.35355339059327373, -0.4903926402016152, 0.46193976625564326, -0.41573480615127256, 0.3535533905932733, -0.27778511650980076, 0.19134171618254478, -0.09754516100806429]
];
/// `DOWN_16[y][v]`: the weight of frequency `v` at row `y` of sixteen.
const DOWN_16: [[f32; 8]; 16] = [
    [0.35355339059327373, 0.49759236333609846, 0.4903926402016152, 0.4784701678661044, 0.46193976625564337, 0.4409606321741775, 0.4157348061512726, 0.3865052266813685],
    [0.35355339059327373, 0.4784701678661044, 0.4157348061512726, 0.31719664208182274, 0.19134171618254492, 0.049008570164780385, -0.0975451610080641, -0.23569836841299885],
    [0.35355339059327373, 0.4409606321741775, 0.27778511650980114, 0.049008570164780385, -0.19134171618254486, -0.3865052266813685, -0.4903926402016152, -0.47847016786610447],
    [0.35355339059327373, 0.3865052266813685, 0.09754516100806417, -0.23569836841299885, -0.46193976625564337, -0.47847016786610447, -0.2777851165098011, 0.049008570164780045],
    [0.35355339059327373, 0.31719664208182274, -0.0975451610080641, -0.44096063217417747, -0.4619397662556434, -0.14514233862723122, 0.2777851165098009, 0.49759236333609846],
    [0.35355339059327373, 0.2356983684129989, -0.277785116509801, -0.49759236333609846, -0.19134171618254517, 0.3171966420818228, 0.4903926402016152, 0.14514233862723128],
    [0.35355339059327373, 0.14514233862723117, -0.4157348061512727, -0.38650522668136855, 0.191341716182545, 0.49759236333609846, 0.09754516100806439, -0.4409606321741774],
    [0.35355339059327373, 0.049008570164780385, -0.4903926402016152, -0.14514233862723122, 0.46193976625564326, 0.235698368412999, -0.41573480615127256, -0.3171966420818227],
    [0.35355339059327373, -0.04900857016478032, -0.4903926402016152, 0.14514233862723103, 0.46193976625564337, -0.23569836841299874, -0.4157348061512728, 0.31719664208182236],
    [0.35355339059327373, -0.14514233862723108, -0.41573480615127273, 0.38650522668136833, 0.19134171618254522, -0.49759236333609846, 0.0975451610080636, 0.44096063217417764],
    [0.35355339059327373, -0.23569836841299885, -0.2777851165098011, 0.49759236333609846, -0.19134171618254495, -0.3171966420818227, 0.4903926402016152, -0.14514233862723128],
    [0.35355339059327373, -0.3171966420818227, -0.09754516100806433, 0.4409606321741775, -0.4619397662556432, 0.14514233862723133, 0.27778511650980126, -0.49759236333609846],
    [0.35355339059327373, -0.3865052266813685, 0.09754516100806415, 0.235698368412999, -0.46193976625564337, 0.47847016786610425, -0.27778511650980076, -0.04900857016478135],
    [0.35355339059327373, -0.44096063217417747, 0.2777851165098009, -0.04900857016477998, -0.19134171618254528, 0.38650522668136883, -0.4903926402016153, 0.47847016786610447],
    [0.35355339059327373, -0.4784701678661044, 0.4157348061512726, -0.3171966420818224, 0.1913417161825449, -0.04900857016477986, -0.097545161008065, 0.2356983684129992],
    [0.35355339059327373, -0.4975923633360984, 0.49039264020161516, -0.47847016786610425, 0.4619397662556432, -0.44096063217417736, 0.415734806151272, -0.38650522668136783]
];
/// `ACROSS_8[u][x]`: the weight of frequency `u` at column `x` of eight.
const ACROSS_8: [[f32; 8]; 8] = [
    [0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373],
    [0.4903926402016152, 0.4157348061512726, 0.27778511650980114, 0.09754516100806417, -0.0975451610080641, -0.277785116509801, -0.4157348061512727, -0.4903926402016152],
    [0.46193976625564337, 0.19134171618254492, -0.19134171618254486, -0.46193976625564337, -0.4619397662556434, -0.19134171618254517, 0.191341716182545, 0.46193976625564326],
    [0.4157348061512726, -0.0975451610080641, -0.4903926402016152, -0.2777851165098011, 0.2777851165098009, 0.4903926402016152, 0.09754516100806439, -0.41573480615127256],
    [0.3535533905932738, -0.35355339059327373, -0.35355339059327384, 0.3535533905932737, 0.35355339059327384, -0.35355339059327334, -0.35355339059327356, 0.3535533905932733],
    [0.27778511650980114, -0.4903926402016152, 0.09754516100806415, 0.41573480615127273, -0.41573480615127256, -0.09754516100806401, 0.4903926402016153, -0.27778511650980076],
    [0.19134171618254492, -0.4619397662556434, 0.46193976625564326, -0.19134171618254495, -0.19134171618254528, 0.46193976625564337, -0.4619397662556432, 0.19134171618254478],
    [0.09754516100806417, -0.2777851165098011, 0.41573480615127273, -0.4903926402016153, 0.4903926402016152, -0.4157348061512725, 0.27778511650980076, -0.09754516100806429]
];
/// `ACROSS_16[u][x]`: the weight of frequency `u` at column `x` of sixteen.
const ACROSS_16: [[f32; 16]; 8] = [
    [0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373, 0.35355339059327373],
    [0.49759236333609846, 0.4784701678661044, 0.4409606321741775, 0.3865052266813685, 0.31719664208182274, 0.2356983684129989, 0.14514233862723117, 0.049008570164780385, -0.04900857016478032, -0.14514233862723108, -0.23569836841299885, -0.3171966420818227, -0.3865052266813685, -0.44096063217417747, -0.4784701678661044, -0.4975923633360984],
    [0.4903926402016152, 0.4157348061512726, 0.27778511650980114, 0.09754516100806417, -0.0975451610080641, -0.277785116509801, -0.4157348061512727, -0.4903926402016152, -0.4903926402016152, -0.41573480615127273, -0.2777851165098011, -0.09754516100806433, 0.09754516100806415, 0.2777851165098009, 0.4157348061512726, 0.49039264020161516],
    [0.4784701678661044, 0.31719664208182274, 0.049008570164780385, -0.23569836841299885, -0.44096063217417747, -0.49759236333609846, -0.38650522668136855, -0.14514233862723122, 0.14514233862723103, 0.38650522668136833, 0.49759236333609846, 0.4409606321741775, 0.235698368412999, -0.04900857016477998, -0.3171966420818224, -0.47847016786610425],
    [0.46193976625564337, 0.19134171618254492, -0.19134171618254486, -0.46193976625564337, -0.4619397662556434, -0.19134171618254517, 0.191341716182545, 0.46193976625564326, 0.46193976625564337, 0.19134171618254522, -0.19134171618254495, -0.4619397662556432, -0.46193976625564337, -0.19134171618254528, 0.1913417161825449, 0.4619397662556432],
    [0.4409606321741775, 0.049008570164780385, -0.3865052266813685, -0.47847016786610447, -0.14514233862723122, 0.3171966420818228, 0.49759236333609846, 0.235698368412999, -0.23569836841299874, -0.49759236333609846, -0.3171966420818227, 0.14514233862723133, 0.47847016786610425, 0.38650522668136883, -0.04900857016477986, -0.44096063217417736],
    [0.4157348061512726, -0.0975451610080641, -0.4903926402016152, -0.2777851165098011, 0.2777851165098009, 0.4903926402016152, 0.09754516100806439, -0.41573480615127256, -0.4157348061512728, 0.0975451610080636, 0.4903926402016152, 0.27778511650980126, -0.27778511650980076, -0.4903926402016153, -0.097545161008065, 0.415734806151272],
    [0.3865052266813685, -0.23569836841299885, -0.47847016786610447, 0.049008570164780045, 0.49759236333609846, 0.14514233862723128, -0.4409606321741774, -0.3171966420818227, 0.31719664208182236, 0.44096063217417764, -0.14514233862723128, -0.49759236333609846, -0.04900857016478135, 0.47847016786610447, 0.2356983684129992, -0.38650522668136783]
];

/// The inverse DCT of `coefficients` (dequantized, natural order: row `v` holds the vertical
/// frequency `v`) to `8 * h_scale` × `8 * v_scale` samples, level-shifted and clamped to
/// 0..=255, written to `out` at rows `stride` apart. `h_scale` and `v_scale` are 1 or 2.
#[allow(clippy::cast_possible_truncation)]
pub fn idct_scaled(
    coefficients: &[i32; 64], out: &mut [i16], stride: usize, h_scale: usize, v_scale: usize
) {
    let height = 8 * v_scale;
    if coefficients[1..].iter().all(|&c| c == 0) {
        // A flat block: the DC term alone, F / 8, rounded half up. The add wraps, as in the
        // other inverse DCTs, since a damaged stream can bring the DC term near `i32::MAX`.
        let value = (128 + (coefficients[0].wrapping_add(4) >> 3)).clamp(0, 255) as i16;
        for row in out.chunks_mut(stride).take(height) {
            row[..8 * h_scale].fill(value);
        }
        return;
    }
    let down: &[[f32; 8]] = if v_scale == 2 { &DOWN_16 } else { &DOWN_8 };
    if h_scale == 2 {
        transform(coefficients, out, stride, &ACROSS_16, down);
    } else {
        transform(coefficients, out, stride, &ACROSS_8, down);
    }
}

/// [`idct_scaled`] for `W` output columns and `down.len()` output rows.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn transform<const W: usize>(
    coefficients: &[i32; 64], out: &mut [i16], stride: usize, across: &[[f32; W]; 8],
    down: &[[f32; 8]]
) {
    // The horizontal pass: each row of coefficients that is not all zero, summed at every
    // output column; rows of even and of odd vertical frequency apart, with their frequency.
    let mut even_rows = [[0.0f32; W]; 4];
    let mut odd_rows = [[0.0f32; W]; 4];
    let mut even_v = [0usize; 4];
    let mut odd_v = [0usize; 4];
    let (mut evens, mut odds) = (0, 0);
    for (v, coeffs) in coefficients.chunks_exact(8).enumerate() {
        if coeffs.iter().all(|&c| c == 0) {
            continue;
        }
        let sums = if v % 2 == 0 {
            even_v[evens] = v;
            evens += 1;
            &mut even_rows[evens - 1]
        } else {
            odd_v[odds] = v;
            odds += 1;
            &mut odd_rows[odds - 1]
        };
        for (&c, weights) in coeffs.iter().zip(across) {
            if c != 0 {
                let c = c as f32;
                for (sum, &w) in sums.iter_mut().zip(weights) {
                    *sum += c * w;
                }
            }
        }
    }

    // The vertical pass. The weight of frequency `v` at row `height - 1 - y` is that at row `y`,
    // negated for odd `v`, so the even and the odd frequencies are summed once for both rows.
    // Then the level shift, rounded to nearest.
    let height = down.len();
    let round = |sum: f32| {
        // Adding 1.5 * 2^23 leaves the value rounded to an integer in the low mantissa bits
        // (exact for |sum| < 2^22; larger sums are clamped first).
        let shifted = sum.clamp(-1024.0, 1024.0) + (128.0 + 12_582_912.0);
        (i64::from(shifted.to_bits()) - 0x4B40_0000).clamp(0, 255) as i16
    };
    for y in 0..height / 2 {
        let weights = &down[y];
        let mut even = [0.0f32; W];
        for (sums, &v) in even_rows.iter().zip(&even_v).take(evens) {
            let w = weights[v];
            for (acc, &s) in even.iter_mut().zip(sums) {
                *acc += w * s;
            }
        }
        let mut odd = [0.0f32; W];
        for (sums, &v) in odd_rows.iter().zip(&odd_v).take(odds) {
            let w = weights[v];
            for (acc, &s) in odd.iter_mut().zip(sums) {
                *acc += w * s;
            }
        }
        let top = &mut out[y * stride..][..W];
        for ((sample, &e), &o) in top.iter_mut().zip(&even).zip(&odd) {
            *sample = round(e + o);
        }
        let bottom = &mut out[(height - 1 - y) * stride..][..W];
        for ((sample, &e), &o) in bottom.iter_mut().zip(&even).zip(&odd) {
            *sample = round(e - o);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    /// The transform's value at output sample (`x`, `y`), straight from the definition.
    fn reference(coefficients: &[i32; 64], x: usize, y: usize, h_scale: usize, v_scale: usize) -> f64 {
        let c = |u: usize| if u == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
        let mut sum = 0.0;
        for v in 0..8 {
            for u in 0..8 {
                let cx = (((2 * x + 1) * u) as f64 * core::f64::consts::PI / (16 * h_scale) as f64).cos();
                let cy = (((2 * y + 1) * v) as f64 * core::f64::consts::PI / (16 * v_scale) as f64).cos();
                sum += c(u) * c(v) / 4.0 * f64::from(coefficients[v * 8 + u]) * cx * cy;
            }
        }
        sum + 128.0
    }

    /// A deterministic pseudo-random block, larger at low frequencies.
    fn block(seed: u32) -> [i32; 64] {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut out = [0; 64];
        for (i, c) in out.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let span = 1024 / (1 + (i / 8 + i % 8) as i32);
            *c = (state % (2 * span as u32 + 1)) as i32 - span;
        }
        out
    }

    #[test]
    fn matches_the_definition() {
        for (h_scale, v_scale) in [(1, 1), (2, 2), (2, 1), (1, 2)] {
            for seed in 0..50 {
                let coefficients = block(seed);
                let (width, height) = (8 * h_scale, 8 * v_scale);
                let mut out = vec![0i16; width * height];
                idct_scaled(&coefficients, &mut out, width, h_scale, v_scale);
                for y in 0..height {
                    for x in 0..width {
                        let exact = reference(&coefficients, x, y, h_scale, v_scale).clamp(0.0, 255.0);
                        let got = f64::from(out[y * width + x]);
                        assert!((got - exact).abs() <= 0.51, "({x},{y}) {got} vs {exact}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_8x8_case_agrees_with_the_decoders_idct() {
        // Within one level of the integer IDCT used for blocks that are not upsampled.
        for seed in 0..50 {
            let coefficients = block(seed);
            let mut scaled = [0i16; 64];
            idct_scaled(&coefficients, &mut scaled, 8, 1, 1);
            let mut integer = [0i16; 64];
            let mut input = coefficients;
            super::super::scalar::idct_int(&mut input, &mut integer, 8);
            for (a, b) in scaled.iter().zip(&integer) {
                assert!((a - b).abs() <= 1, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn a_flat_block_stays_flat() {
        let mut coefficients = [0; 64];
        coefficients[0] = 8 * (200 - 128);
        let mut out = [0i16; 256];
        idct_scaled(&coefficients, &mut out, 16, 2, 2);
        assert!(out.iter().all(|&s| s == 200));
    }

    #[test]
    fn a_dc_term_near_the_limit_does_not_overflow() {
        // A damaged stream can bring the DC term anywhere in `i32`; the flat path must not
        // overflow (a panic in debug builds) and the block stays flat and in range.
        for (dc, expected) in [(i32::MAX, 0), (i32::MAX - 3, 0), (i32::MAX - 4, 255), (i32::MIN, 0)] {
            let mut coefficients = [0; 64];
            coefficients[0] = dc;
            for (h_scale, v_scale) in [(1, 1), (2, 2), (2, 1), (1, 2)] {
                let (width, height) = (8 * h_scale, 8 * v_scale);
                let mut out = vec![-1i16; width * height];
                idct_scaled(&coefficients, &mut out, width, h_scale, v_scale);
                assert!(out.iter().all(|&s| s == expected), "{dc}: {}", out[0]);
            }
        }
    }

    #[test]
    fn writes_only_its_block() {
        let coefficients = block(7);
        let mut out = vec![-1i16; 40 * 20];
        idct_scaled(&coefficients, &mut out[3..], 40, 2, 1);
        for (i, &s) in out.iter().enumerate() {
            let (x, y) = (i % 40, i / 40);
            let inside = (3..19).contains(&x) && y < 8;
            assert_eq!(s != -1, inside, "({x},{y})");
        }
    }
}
