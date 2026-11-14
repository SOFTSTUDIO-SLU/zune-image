/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Sequential frames coded one scan per component. Such a scan is not interleaved: its MCU is
//! one block, and it covers the component's own block grid (ITU-T T.81 A.2.2), which has more
//! block rows than the frame has MCU rows when the component's vertical factor is above 1.
//!
//! Every block is flat, with a value that differs from its neighbours', and the samples are
//! checked as stored (`ColorTransform::None` and `set_idct_upsampling(true)`), so each output
//! channel is one component's plane and every sample is its block's value.

mod common;

use common::color::{flat, Coding, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{ColorTransform, JpegDecoder};

/// A flat value per block that differs from its neighbours' (component `c`, block `bx`, `by`).
fn value(c: usize, bx: usize, by: usize) -> u8 {
    [40u8, 220, 90, 160, 10, 250, 130][(bx * 3 + by * 5 + c) % 7]
}

fn frame(width: u16, height: u16, factors: [(u8, u8); 3]) -> Frame {
    let layout: Vec<(u8, u8, u8)> =
        factors.iter().enumerate().map(|(c, &(h, v))| (c as u8 + 1, h, v)).collect();
    let mut frame = Frame::new(width, height, &layout, 0);
    frame.coding = Coding::ScanPerComponent;
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

/// Every sample of `width` × `height` with `factors` is its block's value, in strict and
/// lenient mode.
fn assert_blocks_kept(width: u16, height: u16, factors: [(u8, u8); 3]) {
    let data = frame(width, height, factors).encode();
    let h_max = factors.iter().map(|f| usize::from(f.0)).max().unwrap();
    let v_max = factors.iter().map(|f| usize::from(f.1)).max().unwrap();
    for strict in [true, false] {
        let options = DecoderOptions::default()
            .set_strict_mode(strict)
            .jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&data[..]), options);
        decoder.set_color_transform(Some(ColorTransform::None));
        decoder.set_idct_upsampling(true);
        let samples = decoder.decode().unwrap();
        for y in 0..usize::from(height) {
            for x in 0..usize::from(width) {
                for (c, &(h, v)) in factors.iter().enumerate() {
                    let bx = x * usize::from(h) / h_max / 8;
                    let by = y * usize::from(v) / v_max / 8;
                    assert_eq!(
                        samples[(y * usize::from(width) + x) * 3 + c],
                        value(c, bx, by),
                        "{factors:?} {width}x{height} strict {strict}: component {c} at ({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn every_block_row_of_a_vertically_sampled_component_is_decoded() {
    for factors in [[(2, 2), (1, 1), (1, 1)], [(1, 2), (1, 1), (1, 1)], [(2, 2), (1, 2), (1, 1)]] {
        for (width, height) in [(37, 23), (64, 64), (40, 8)] {
            assert_blocks_kept(width, height, factors);
        }
    }
}

/// A scan's data can be so short that the bit reader reaches the next scan's marker while
/// blocks of the scan are still buffered: they are decoded before the next scan starts.
#[test]
fn a_short_scan_keeps_its_last_rows() {
    for factors in [[(1, 1), (1, 1), (1, 1)], [(2, 2), (1, 1), (1, 1)]] {
        for (width, height) in [(8, 16), (8, 24), (16, 32)] {
            assert_blocks_kept(width, height, factors);
        }
    }
}
