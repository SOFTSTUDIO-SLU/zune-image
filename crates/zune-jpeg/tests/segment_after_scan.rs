/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A segment between the scan and EOI (a comment, an application segment, a table) when the
//! last rows of the scan take only a few bytes: the bit reader reaches the segment's marker
//! while bits of those rows are still buffered, and the rows are decoded before the marker is
//! handled, in strict and in lenient mode. A scan whose data really ends before such a
//! segment is reported as cut: strict mode fails, lenient mode keeps the MCUs it decoded and
//! fills the rest with 128.

mod common;

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

/// `file` with a segment `marker` `body` inserted before its EOI marker.
fn insert_before_eoi(file: &[u8], marker: u8, body: &[u8]) -> Vec<u8> {
    let (head, eoi) = file.split_at(file.len() - 2);
    assert_eq!(eoi, [0xFF, 0xD9]);
    let mut out = head.to_vec();
    out.extend([0xFF, marker]);
    out.extend(((body.len() + 2) as u16).to_be_bytes());
    out.extend(body);
    out.extend(eoi);
    out
}

/// A comment, an APP0 segment and a DQT segment (table 1).
fn segments() -> Vec<(u8, Vec<u8>)> {
    let mut dqt = vec![0x01];
    dqt.extend([1; 64]);
    vec![
        (0xFE, b"comment".to_vec()),
        (0xE0, b"app0 segment".to_vec()),
        (0xDB, dqt),
    ]
}

#[test]
fn segment_after_a_short_scan_waits_for_its_last_rows() {
    // 16 × 16 samples, two rows of two blocks: the whole scan takes a few bytes.
    let gray = Gray::new(16, 16, &[200, 60, 220, 40]);
    for (marker, body) in segments() {
        let file = insert_before_eoi(&gray.encode(), marker, &body);
        for strict in [true, false] {
            assert_eq!(
                decode(&file, strict).unwrap(),
                gray.samples(),
                "marker 0xFF{marker:X}, strict mode {strict}"
            );
        }
    }
}

#[test]
fn segment_after_a_scan_with_restart_markers_waits_for_its_last_rows() {
    let mut gray = Gray::new(16, 16, &[200, 60, 220, 40]);
    gray.restart = 2;
    for (marker, body) in segments() {
        let file = insert_before_eoi(&gray.encode(), marker, &body);
        for strict in [true, false] {
            assert_eq!(
                decode(&file, strict).unwrap(),
                gray.samples(),
                "marker 0xFF{marker:X}, strict mode {strict}"
            );
        }
    }
}

#[test]
fn scan_cut_before_a_segment_fails_in_strict_mode_and_fills_128_in_lenient_mode() {
    // 16 × 32 samples, four rows of two blocks, the data cut after the first two rows.
    let gray = Gray::new(16, 32, &[200, 60, 220, 40, 180, 90, 30, 240]);
    let keep = gray.scan_bits(&gray.blocks[..4]).len().div_ceil(8);
    let file = insert_before_eoi(&gray.encode_cut(keep), 0xFE, b"comment");
    assert!(decode(&file, true).is_err());

    let out = decode(&file, false).unwrap();
    assert_eq!(out.len(), 16 * 32);
    assert_eq!(out[..16 * 16], gray.samples()[..16 * 16]);
    assert!(out[16 * 24..].iter().all(|&s| s == 128), "{out:?}");
}
