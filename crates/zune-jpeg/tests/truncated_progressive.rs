/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A progressive scan that runs out of data before its last block: strict mode fails, lenient
//! mode keeps the block that ran out, leaves the blocks after it (up to the next restart marker)
//! as the earlier scans left them, and reports it with `JpegDecoder::ran_out_of_data`.
//!
//! The frames come from the writer in tests/common/color.rs: a DC scan of all components, then
//! an AC scan per component; every code is eight bits long and the symbol with code 0 is
//! category 0 (DC) or end of block (AC), so zero bits decode to "no change". A block of value
//! 129 has the DC coefficient 8: the first block of a scan or restart interval takes 12 bits
//! (code and four extra bits), each later one 8 bits (difference 0).

mod common;

use common::color::{Coding, Cut, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::{ColorTransform, JpegDecoder};

/// Decodes `data` with no colour transform; the samples and whether a scan ran out of data.
fn decode(data: &[u8], strict: bool, out: ColorSpace) -> Result<(Vec<u8>, bool), DecodeErrors> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_color_transform(Some(ColorTransform::None));
    let samples = decoder.decode()?;
    Ok((samples, decoder.ran_out_of_data()))
}

fn assert_premature_end(result: Result<(Vec<u8>, bool), DecodeErrors>) {
    match result {
        Err(DecodeErrors::FormatStatic("Premature end of buffer")) => {}
        Err(e) => panic!("expected a premature end, got {e:?}"),
        Ok(_) => panic!("expected a premature end, got an image")
    }
}

/// A progressive grey frame of 8 × 2 blocks, every block 129.
fn grey() -> Frame {
    let mut frame = Frame::new(64, 16, &[(1, 1, 1)], 129);
    frame.coding = Coding::Progressive;
    frame
}

/// Grey samples of a 64 × 16 image whose block `n` (row order) has value `value(n)`.
fn grey_samples(value: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut samples = Vec::new();
    for y in 0..16 {
        for x in 0..64 {
            samples.push(value((y / 8) * 8 + x / 8));
        }
    }
    samples
}

#[test]
fn whole_progressive_files_decode_in_strict_mode() {
    let mut frames = vec![grey()];
    let mut restarts = grey();
    restarts.restart = 3;
    frames.push(restarts);
    // Every block 128: each scan's data ends on a byte boundary right before the next marker.
    let mut aligned = Frame::new(64, 16, &[(1, 1, 1)], 128);
    aligned.coding = Coding::Progressive;
    frames.push(aligned);
    for frame in frames {
        let expected = frame.components[0].blocks[0][0] / 8 + 128;
        let (samples, ran_out) = decode(&frame.encode(), true, ColorSpace::Luma).unwrap();
        assert!(samples.iter().all(|&s| i32::from(s) == expected));
        assert!(!ran_out);
    }
}

#[test]
fn a_dc_scan_cut_short_fails_in_strict_mode() {
    let data = grey().encode_cut(Cut::End { scan: 0, keep: 6, eoi: true });
    assert_premature_end(decode(&data, true, ColorSpace::Luma));
}

#[test]
fn a_dc_scan_cut_short_keeps_the_blocks_before_the_cut_in_lenient_mode() {
    // 6 bytes are 48 bits: blocks 0-4 end by bit 44, block 5 (bits 44-52) runs out and keeps
    // what it decoded, blocks 6-15 get no DC coefficient: 128. Decoding the padding instead
    // would give them all 129.
    let data = grey().encode_cut(Cut::End { scan: 0, keep: 6, eoi: true });
    let (samples, ran_out) = decode(&data, false, ColorSpace::Luma).unwrap();
    assert_eq!(samples, grey_samples(|n| if n <= 5 { 129 } else { 128 }));
    assert!(ran_out);
}

#[test]
fn data_that_ends_in_a_dc_scan_without_eoi() {
    let data = grey().encode_cut(Cut::End { scan: 0, keep: 6, eoi: false });
    assert_premature_end(decode(&data, true, ColorSpace::Luma));
    let (samples, ran_out) = decode(&data, false, ColorSpace::Luma).unwrap();
    assert_eq!(samples, grey_samples(|n| if n <= 5 { 129 } else { 128 }));
    assert!(ran_out);
}

#[test]
fn an_ac_scan_cut_short() {
    // Every block also has the first AC coefficient 16: 21 bits a block in the AC scan (a
    // run/size code, five extra bits, end of block). 8 bytes are 64 bits: blocks 0-2 are whole,
    // block 3 runs out on the first bit of its code and decodes an end of block, and blocks
    // 4-15 keep the DC scan's coefficients only.
    let mut frame = grey();
    for block in &mut frame.components[0].blocks {
        block[1] = 16;
    }
    let data = frame.encode_cut(Cut::End { scan: 1, keep: 8, eoi: true });
    assert_premature_end(decode(&data, true, ColorSpace::Luma));

    let mut expected = grey();
    for block in &mut expected.components[0].blocks[..3] {
        block[1] = 16;
    }
    let (expected, _) = decode(&expected.encode(), true, ColorSpace::Luma).unwrap();
    let (samples, ran_out) = decode(&data, false, ColorSpace::Luma).unwrap();
    assert_eq!(samples, expected);
    assert!(ran_out);
}

#[test]
fn a_cut_restart_interval_is_skipped_to_the_next_restart_marker() {
    // Four intervals of four blocks; interval 1 (blocks 4-7) keeps one byte. Block 4 runs out
    // in its extra bits (its value is not checked), blocks 5-7 are skipped (128), and the
    // intervals after the restart marker decode as before.
    let mut frame = grey();
    frame.restart = 4;
    let data = frame.encode_cut(Cut::Interval { scan: 0, interval: 1, keep: 1 });
    assert_premature_end(decode(&data, true, ColorSpace::Luma));

    let (samples, ran_out) = decode(&data, false, ColorSpace::Luma).unwrap();
    let expected = grey_samples(|n| match n {
        5..=7 => 128,
        _ => 129
    });
    // Block 4 is (4, 0): masked out of both.
    let mask = |mut s: Vec<u8>| {
        for y in 0..8 {
            s[y * 64 + 32..y * 64 + 40].fill(0);
        }
        s
    };
    assert_eq!(mask(samples), mask(expected));
    assert!(ran_out);
}

#[test]
fn an_interleaved_dc_scan_cut_short() {
    // Three components, one block each per MCU, four MCUs. The DC scan takes 36 bits for MCU 0
    // and 24 for each later one; 6 bytes (48 bits) end in MCU 1, whose codes are all zero bits
    // anyway, so it keeps 129; MCUs 2 and 3 get no DC coefficients: 128.
    let mut frame = Frame::new(32, 8, &[(1, 1, 1), (2, 1, 1), (3, 1, 1)], 129);
    frame.coding = Coding::Progressive;
    let data = frame.encode_cut(Cut::End { scan: 0, keep: 6, eoi: true });
    assert_premature_end(decode(&data, true, ColorSpace::RGB));

    let (samples, ran_out) = decode(&data, false, ColorSpace::RGB).unwrap();
    let mut expected = Vec::new();
    for _ in 0..8 {
        for x in 0..32 {
            expected.extend([if x < 16 { 129 } else { 128 }; 3]);
        }
    }
    assert_eq!(samples, expected);
    assert!(ran_out);
}
