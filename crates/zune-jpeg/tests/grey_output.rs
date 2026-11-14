/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A grey output (`ColorSpace::Luma`) of a colour frame whose chroma is subsampled vertically
//! only (sampling factors 1×2, 1×1, 1×1). For such an output the decoder outputs two rows of
//! blocks per MCU row, the MCU's two luma block rows; the frame's components must still be read
//! one MCU row of the frame at a time, as for a colour output, whether the frame is coded in one
//! interleaved scan, in more than one scan or progressively.

mod common;

use common::color::{flat, Coding, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// A value per block that differs from its neighbours' (component `c`, block `bx`, `by`).
fn value(c: usize, bx: usize, by: usize) -> u8 {
    [40u8, 220, 90, 160, 10, 250, 130][(bx * 3 + by * 5 + c) % 7]
}

/// A frame of `width` × `height`, sampling 1×2, 1×1, 1×1, every block flat with [`value`].
fn frame(width: u16, height: u16, coding: Coding) -> Frame {
    let mut frame = grey_frame(width, height, coding);
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

/// The same frame with every sample 128: every coefficient is 0.
fn grey_frame(width: u16, height: u16, coding: Coding) -> Frame {
    let mut frame = Frame::new(width, height, &[(1, 1, 2), (2, 1, 1), (3, 1, 1)], 128);
    frame.coding = coding;
    frame
}

fn decode(data: &[u8], out: ColorSpace, strict: bool) -> Result<Vec<u8>, String> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.decode_headers().map_err(|e| format!("{e:?}"))?;
    let size = decoder.output_buffer_size().unwrap();
    let pixels = decoder.decode().map_err(|e| format!("{e:?}"))?;
    assert_eq!(pixels.len(), size);
    Ok(pixels)
}

/// The index of the first sample where `a` and `b` differ, if any (or the shorter length).
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len())))
}

/// The grey output of `frame` is its luma plane: the first channel of the YCbCr output.
fn assert_grey_is_luma(frame: &Frame, strict: bool) -> Vec<u8> {
    let what = format!("{:?} {}x{} strict {strict}", frame.coding, frame.width, frame.height);
    let data = frame.encode();
    let grey = decode(&data, ColorSpace::Luma, strict).unwrap_or_else(|e| panic!("{what}: {e}"));
    let ycbcr = decode(&data, ColorSpace::YCbCr, strict).unwrap_or_else(|e| panic!("{what}: {e}"));
    let luma: Vec<u8> = ycbcr.chunks_exact(3).map(|p| p[0]).collect();
    if let Some(i) = first_difference(&grey, &luma) {
        panic!("{what}: grey differs from the YCbCr luma at sample {i}");
    }
    grey
}

/// The luma samples the frame holds: each block's value.
fn expected_luma(width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..usize::from(height) {
        for x in 0..usize::from(width) {
            out.push(value(0, x / 8, y / 8));
        }
    }
    out
}

const SIZES: [(u16, u16); 3] = [(37, 23), (32, 32), (8, 16)];

#[test]
fn an_interleaved_frame_decodes_to_grey() {
    for (width, height) in SIZES {
        for strict in [true, false] {
            for restart in [0, 1, 3] {
                // The strict mode failed here (premature end of data, or a Huffman error when
                // more bytes follow the scan): the scan was read for twice the frame's MCU rows.
                let mut grey = grey_frame(width, height, Coding::Interleaved);
                grey.restart = restart;
                assert_grey_is_luma(&grey, strict);
                let mut frame = frame(width, height, Coding::Interleaved);
                frame.restart = restart;
                let grey = assert_grey_is_luma(&frame, strict);
                let expected = expected_luma(width, height);
                if let Some(i) = first_difference(&grey, &expected) {
                    let what = format!("{width}x{height} restart {restart} strict {strict}");
                    panic!("{what}: wrong luma at sample {i}");
                }
            }
        }
    }
}

#[test]
fn a_scan_per_component_decodes_to_grey() {
    for (width, height) in SIZES {
        for strict in [true, false] {
            // This panicked (a slice of the coefficient buffer out of range) in the lenient
            // mode, and failed in the strict one: the luma scan was read for twice its rows.
            assert_grey_is_luma(&grey_frame(width, height, Coding::ScanPerComponent), strict);
            // In the lenient mode this gave an image of 128 for the same reason.
            assert_grey_is_luma(&frame(width, height, Coding::ScanPerComponent), strict);
        }
    }
}

#[test]
fn a_progressive_frame_decodes_to_grey_with_the_right_rows() {
    for (width, height) in SIZES {
        for strict in [true, false] {
            // This took the rows of MCU row 2i for MCU row i.
            let grey = assert_grey_is_luma(&frame(width, height, Coding::Progressive), strict);
            let expected = expected_luma(width, height);
            if let Some(i) = first_difference(&grey, &expected) {
                panic!("{width}x{height} strict {strict}: wrong luma at sample {i}");
            }
        }
    }
}
