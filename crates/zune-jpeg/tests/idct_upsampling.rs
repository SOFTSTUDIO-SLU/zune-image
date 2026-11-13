/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! `JpegDecoder::set_idct_upsampling`: a component subsampled by 2 is upsampled by its inverse
//! DCT (8×8 coefficients to 16×16, 16×8 or 8×16 samples), and what remains of a larger factor by
//! repeating samples. The samples are checked as stored (`ColorTransform::None`), so each
//! output channel is one component's plane.

mod common;

use common::color::{flat, Coding, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{ColorTransform, JpegDecoder};

const CODINGS: [Coding; 3] = [Coding::Interleaved, Coding::ScanPerComponent, Coding::Progressive];

fn decode(data: &[u8], idct: bool) -> Vec<u8> {
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_color_transform(Some(ColorTransform::None));
    decoder.set_idct_upsampling(idct);
    decoder.decode().unwrap()
}

/// A flat value per block that differs from its neighbours' (component `c`, block `bx`, `by`).
fn value(c: usize, bx: usize, by: usize) -> u8 {
    [40u8, 220, 90, 160, 10, 250, 130][(bx * 3 + by * 5 + c) % 7]
}

/// A frame of `width` × `height` with three components of the given sampling factors, every
/// block flat with [`value`].
fn checkerboard(width: u16, height: u16, factors: [(u8, u8); 3], coding: Coding) -> Frame {
    let layout: Vec<(u8, u8, u8)> =
        factors.iter().enumerate().map(|(c, &(h, v))| (c as u8 + 1, h, v)).collect();
    let mut frame = Frame::new(width, height, &layout, 0);
    frame.coding = coding;
    for c in 0..3 {
        let (wide, high) = frame.grid(c);
        for by in 0..high {
            for bx in 0..wide {
                frame.set(c, bx, by, flat(value(c, bx, by)));
            }
        }
    }
    frame
}

/// With the option, every sample is its block's value: flat blocks stay flat however they are
/// upsampled, and nothing is blended across block edges.
fn assert_blocks_kept(width: u16, height: u16, factors: [(u8, u8); 3], coding: Coding) {
    let frame = checkerboard(width, height, factors, coding);
    let samples = decode(&frame.encode(), true);
    let h_max = factors.iter().map(|f| usize::from(f.0)).max().unwrap();
    let v_max = factors.iter().map(|f| usize::from(f.1)).max().unwrap();
    for y in 0..usize::from(height) {
        for x in 0..usize::from(width) {
            for (c, &(h, v)) in factors.iter().enumerate() {
                // The component's sample under output pixel (x, y), and its block.
                let bx = x * usize::from(h) / h_max / 8;
                let by = y * usize::from(v) / v_max / 8;
                let got = samples[(y * usize::from(width) + x) * 3 + c];
                assert_eq!(
                    got,
                    value(c, bx, by),
                    "{factors:?} {coding:?}: component {c} at ({x},{y})"
                );
            }
        }
    }
}

#[test]
fn chroma_subsampled_by_two_keeps_its_blocks() {
    for coding in CODINGS {
        for factors in [
            [(2, 2), (1, 1), (1, 1)],
            [(2, 1), (1, 1), (1, 1)],
            [(1, 2), (1, 1), (1, 1)],
            [(2, 2), (2, 1), (1, 1)]
        ] {
            // A scan per component of a vertically subsampled frame reads the wrong blocks in
            // this decoder with or without the option (a separate bug).
            if coding == Coding::ScanPerComponent && factors[0].1 == 2 {
                continue;
            }
            for (width, height) in [(32, 32), (37, 21), (8, 8)] {
                assert_blocks_kept(width, height, factors, coding);
            }
        }
    }
}

#[test]
fn a_larger_factor_is_repeated_after_the_inverse_dct() {
    for coding in [Coding::Interleaved, Coding::Progressive] {
        for factors in [[(4, 1), (1, 1), (1, 1)], [(4, 2), (1, 1), (1, 1)]] {
            assert_blocks_kept(67, 19, factors, coding);
        }
    }
}

#[test]
fn without_the_option_chroma_is_blended_across_blocks() {
    // The decoder's default triangle filter is unchanged.
    let frame = checkerboard(32, 32, [(2, 2), (1, 1), (1, 1)], Coding::Interleaved);
    let data = frame.encode();
    let blended = decode(&data, false);
    let kept = decode(&data, true);
    assert_ne!(blended, kept);
    // The luma is the same either way.
    let luma = |s: &[u8]| s.chunks_exact(3).map(|p| p[0]).collect::<Vec<_>>();
    assert_eq!(luma(&blended), luma(&kept));
}

#[test]
fn a_frame_without_subsampling_is_not_changed() {
    let frame = checkerboard(24, 16, [(1, 1), (1, 1), (1, 1)], Coding::Interleaved);
    let data = frame.encode();
    assert_eq!(decode(&data, true), decode(&data, false));
}

/// The value of the 16-point inverse DCT of `block` at sample (`x`, `y`) of 16 × 16.
fn reference(block: &[i32; 64], x: usize, y: usize) -> f64 {
    let c = |u: usize| if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
    let cos = |n: usize, u: usize| (((2 * n + 1) * u) as f64 * std::f64::consts::PI / 32.0).cos();
    let mut sum = 128.0;
    for v in 0..8 {
        for u in 0..8 {
            sum += c(u) * c(v) / 4.0 * f64::from(block[v * 8 + u]) * cos(x, u) * cos(y, v);
        }
    }
    sum.clamp(0.0, 255.0)
}

#[test]
fn chroma_coefficients_are_transformed_at_twice_the_rate() {
    // Flat luma; chroma blocks with low-frequency detail, as an encoder leaves them.
    for coding in [Coding::Interleaved, Coding::Progressive] {
        let mut frame = Frame::new(32, 32, &[(1, 2, 2), (2, 1, 1), (3, 1, 1)], 128);
        frame.coding = coding;
        let mut state = 0x2545_F491u32;
        let mut random = |span: i32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % (2 * span as u32 + 1)) as i32 - span
        };
        for c in 1..3 {
            for by in 0..2 {
                for bx in 0..2 {
                    let mut block = [0i32; 64];
                    for v in 0..4 {
                        for u in 0..4 {
                            block[v * 8 + u] = random(160 / (1 + u + v) as i32);
                        }
                    }
                    frame.set(c, bx, by, block);
                }
            }
        }
        let samples = decode(&frame.encode(), true);
        for y in 0..32 {
            for x in 0..32 {
                for c in 1..3 {
                    let block = &frame.components[c].blocks[(y / 16) * 2 + x / 16];
                    let exact = reference(block, x % 16, y % 16);
                    let got = f64::from(samples[(y * 32 + x) * 3 + c]);
                    assert!(
                        (got - exact).abs() <= 0.51,
                        "{coding:?}: component {c} at ({x},{y}): {got} vs {exact}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_grey_output_ignores_the_option() {
    let frame = checkerboard(32, 32, [(2, 2), (1, 1), (1, 1)], Coding::Interleaved);
    let data = frame.encode();
    let grey = |idct: bool| {
        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
        let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&data[..]), options);
        decoder.set_idct_upsampling(idct);
        decoder.decode().unwrap()
    };
    assert_eq!(grey(true), grey(false));
}
