/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A baseline scan whose entropy-coded data ends before its last MCU, with the EOI marker
//! right after the cut: strict mode reports it. Lenient mode does what libjpeg does: the MCUs
//! decoded from real data stay, the MCU that ran out is kept as decoded (from zero bits), and
//! every MCU after it has samples of 128. Scans that are whole stay valid, including one whose
//! last MCU ends exactly on a byte boundary before EOI, and ones with restart markers.

mod common;

use std::ops::Range;

use common::Gray;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::JpegDecoder;

fn decode(data: &[u8], strict: bool) -> Result<Vec<u8>, DecodeErrors> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(ColorSpace::Luma);
    JpegDecoder::new_with_options(ZCursor::new(data), options).decode()
}

/// 16 × 32 samples: two blocks per row of blocks, four rows of blocks; one block per MCU.
fn image() -> Gray {
    Gray::new(16, 32, &[200, 60, 220, 40, 180, 90, 30, 240])
}

/// The bytes that hold the first `blocks` blocks of `gray`'s scan (no restart markers).
fn bytes_for(gray: &Gray, blocks: usize) -> usize {
    gray.scan_bits(&gray.blocks[..blocks]).len().div_ceil(8)
}

/// The samples of a 16-sample-wide image in the columns and rows of `area`.
fn area(image: &[u8], columns: Range<usize>, rows: Range<usize>) -> Vec<u8> {
    rows.flat_map(|y| columns.clone().map(move |x| image[y * 16 + x]))
        .collect()
}

fn assert_128(image: &[u8], columns: Range<usize>, rows: Range<usize>) {
    let samples = area(image, columns.clone(), rows.clone());
    assert!(
        samples.iter().all(|&s| s == 128),
        "columns {columns:?}, rows {rows:?}: {samples:?}"
    );
}

#[test]
fn whole_scan_decodes_in_strict_mode() {
    let gray = image();
    assert_eq!(decode(&gray.encode(), true).unwrap(), gray.samples());
}

#[test]
fn scan_cut_after_half_of_its_mcus_fails_in_strict_mode() {
    let gray = image();
    let cut = gray.encode_cut(bytes_for(&gray, 4));
    assert!(decode(&cut, true).is_err());
}

#[test]
fn scan_cut_after_half_of_its_mcus_keeps_that_half_and_fills_128_in_lenient_mode() {
    // The cut keeps the bytes of blocks 0-3 (the first two rows of blocks); block 4 runs out
    // and stays as decoded; blocks 5-7 are 128.
    let gray = image();
    let cut = gray.encode_cut(bytes_for(&gray, 4));
    let out = decode(&cut, false).unwrap();
    assert_eq!(out.len(), 16 * 32);
    assert_eq!(out[..16 * 16], gray.samples()[..16 * 16]);
    assert_128(&out, 8..16, 16..24);
    assert_128(&out, 0..16, 24..32);
}

#[test]
fn scan_cut_in_its_last_row_fails_in_strict_mode_and_fills_after_it_in_lenient_mode() {
    // Block 6, the first of the last row, runs out; block 7 is 128.
    let gray = image();
    let cut = gray.encode_cut(bytes_for(&gray, 7) - 1);
    assert!(decode(&cut, true).is_err());

    let out = decode(&cut, false).unwrap();
    assert_eq!(out[..16 * 24], gray.samples()[..16 * 24]);
    assert_128(&out, 8..16, 24..32);
}

#[test]
fn scan_with_restart_markers_cut_in_the_last_block_of_an_interval() {
    // One block per restart interval: block 3, which ends its interval, runs out.
    let mut gray = image();
    gray.restart = 1;
    let keep: usize = (0..3)
        .map(|i| gray.scan_bits(&gray.blocks[i..=i]).len().div_ceil(8))
        .sum();
    let cut = gray.encode_cut(keep + 1);
    assert!(decode(&cut, true).is_err());

    let out = decode(&cut, false).unwrap();
    let samples = gray.samples();
    assert_eq!(out[..16 * 8], samples[..16 * 8]);
    assert_eq!(area(&out, 0..8, 8..16), area(&samples, 0..8, 8..16));
    assert_128(&out, 0..16, 16..32);
}

#[test]
fn last_mcu_ending_on_a_byte_boundary_before_eoi_is_valid() {
    let gray = Gray::new(16, 32, &[200, 60, 220, 40, 180, 90, 30, 20]);
    assert_eq!(gray.scan_bits(&gray.blocks).len() % 8, 0);
    assert_eq!(decode(&gray.encode(), true).unwrap(), gray.samples());
}

#[test]
fn restart_intervals_ending_on_byte_boundaries_or_not_are_valid() {
    for restart in [1, 2, 3] {
        let mut gray = image();
        gray.restart = restart;
        assert_eq!(
            decode(&gray.encode(), true).unwrap(),
            gray.samples(),
            "restart interval {restart}"
        );
    }
}
