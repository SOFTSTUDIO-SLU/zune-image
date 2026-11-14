/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Frames whose first component does not have the largest sampling factors, for example
//! 1×2, 1×1, 2×1 or 1×1, 2×2, 1×1. Each component may have any factor from 1 to 4 in each
//! direction (ITU-T T.81 B.2.2), and the MCU of an interleaved scan is sized by the largest
//! factors over all components, whichever component has them (A.1.1). A scan of one component
//! covers that component's own block grid (A.2.2).
//!
//! Every block is flat, with a value that differs from its neighbours'. With
//! `set_idct_upsampling(true)` and no colour conversion (`ColorTransform::None` to RGB, or
//! YCbCr to YCbCr) each output channel is one component's plane, and every sample is its
//! block's value. The other outputs are checked to decode, and the grey one to be the luma of
//! the YCbCr output. No case may panic.

mod common;

use std::panic::{catch_unwind, AssertUnwindSafe};

use common::color::{flat, Coding, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{ColorTransform, JpegDecoder};

const CODINGS: [Coding; 3] = [Coding::Interleaved, Coding::ScanPerComponent, Coding::Progressive];

const LAYOUTS: [[(u8, u8); 3]; 4] = [
    [(1, 2), (1, 1), (2, 1)],
    [(1, 1), (2, 2), (1, 1)],
    [(1, 1), (1, 1), (2, 2)],
    [(2, 1), (1, 2), (1, 1)]
];

/// A flat value per block that differs from its neighbours' (component `c`, block `bx`, `by`).
fn value(c: usize, bx: usize, by: usize) -> u8 {
    [40u8, 220, 90, 160, 10, 250, 130][(bx * 3 + by * 5 + c) % 7]
}

fn frame(width: u16, height: u16, factors: [(u8, u8); 3], coding: Coding) -> Frame {
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

/// Decodes `data` to `out`, catching a panic: `Err` holds the error or the panic message.
/// `stored` asks for `ColorTransform::None` (the samples as stored; RGB output only), otherwise
/// the decoder's default transform applies.
fn decode(
    data: &[u8], out: ColorSpace, strict: bool, stored: bool, idct: bool
) -> Result<Vec<u8>, String> {
    let run = || {
        let options = DecoderOptions::default()
            .set_strict_mode(strict)
            .jpeg_set_out_colorspace(out);
        let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
        if stored {
            decoder.set_color_transform(Some(ColorTransform::None));
        }
        decoder.set_idct_upsampling(idct);
        decoder.decode_headers().map_err(|e| format!("{e:?}"))?;
        let size = decoder.output_buffer_size().unwrap();
        let pixels = decoder.decode().map_err(|e| format!("{e:?}"))?;
        assert_eq!(pixels.len(), size, "output size");
        Ok(pixels)
    };
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Err(format!("PANIC: {message}"))
        }
    }
}

/// The first sample of `samples` (with `channels` per pixel) that is not its block's value.
fn first_wrong(
    samples: &[u8], channels: usize, width: u16, height: u16, factors: [(u8, u8); 3]
) -> Option<String> {
    let h_max = factors.iter().map(|f| usize::from(f.0)).max().unwrap();
    let v_max = factors.iter().map(|f| usize::from(f.1)).max().unwrap();
    for y in 0..usize::from(height) {
        for x in 0..usize::from(width) {
            for (c, &(h, v)) in factors.iter().enumerate().take(channels) {
                let bx = x * usize::from(h) / h_max / 8;
                let by = y * usize::from(v) / v_max / 8;
                let got = samples[(y * usize::from(width) + x) * channels + c];
                let want = value(c, bx, by);
                if got != want {
                    return Some(format!("component {c} at ({x},{y}): {got}, expected {want}"));
                }
            }
        }
    }
    None
}

/// Every way of decoding `factors` × `coding` that fails, as a line each.
fn failures(factors: [(u8, u8); 3], coding: Coding, width: u16, height: u16) -> Vec<String> {
    let data = frame(width, height, factors, coding).encode();
    let mut failures = Vec::new();
    for strict in [true, false] {
        let what = |how: &str| format!("{factors:?} {coding:?} {width}x{height} {how} strict {strict}");
        let mut check = |how: &str, out, stored, idct, exact: Option<usize>| {
            match decode(&data, out, strict, stored, idct) {
                Ok(samples) => {
                    if let Some(channels) = exact {
                        if let Some(wrong) = first_wrong(&samples, channels, width, height, factors)
                        {
                            failures.push(format!("{}: {wrong}", what(how)));
                        }
                    }
                    Some(samples)
                }
                Err(e) => {
                    failures.push(format!("{}: {e}", what(how)));
                    None
                }
            }
        };
        // The planes as stored, and as YCbCr (no conversion): every sample is its block's.
        check("RGB stored", ColorSpace::RGB, true, true, Some(3));
        check("YCbCr", ColorSpace::YCbCr, false, true, Some(3));
        // Converted to RGB, and without the IDCT upsampling: decodes.
        check("RGB", ColorSpace::RGB, false, true, None);
        let ycbcr = check("YCbCr triangle", ColorSpace::YCbCr, false, false, None);
        // Grey is the luma plane of the YCbCr output.
        let grey = check("Luma", ColorSpace::Luma, false, false, None);
        if let (Some(ycbcr), Some(grey)) = (ycbcr, grey) {
            let luma: Vec<u8> = ycbcr.chunks_exact(3).map(|p| p[0]).collect();
            if luma != grey {
                failures.push(format!("{}: grey is not the YCbCr luma", what("Luma")));
            }
        }
    }
    failures
}

#[test]
fn a_later_component_with_larger_factors_decodes() {
    let mut all = Vec::new();
    for factors in LAYOUTS {
        for coding in CODINGS {
            for (width, height) in [(37, 23), (16, 16), (8, 8), (8, 40)] {
                all.extend(failures(factors, coding, width, height));
            }
        }
    }
    assert!(all.is_empty(), "{} failures:\n{}", all.len(), all.join("\n"));
}
