/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A DNL (define number of lines) segment after the scan of a frame whose header gives the
//! line count (ITU-T T.81 B.2.2, B.2.5) adds nothing the decode needs: it is skipped, as
//! libjpeg does, in strict and in lenient mode, and the image decodes at the frame header's
//! size. A malformed DNL segment fails in strict mode and is skipped by its length in lenient
//! mode. A DNL segment before the frame header has no frame to complete and still fails, and so
//! does a frame header of 0 lines (its count only in DNL), which needs the count before the
//! output is sized.

mod common;

use common::color::{flat, Coding, Frame};
use common::Gray;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::{ColorTransform, JpegDecoder};

fn decode(data: &[u8], strict: bool) -> Result<Vec<u8>, DecodeErrors> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(ColorSpace::Luma);
    JpegDecoder::new_with_options(ZCursor::new(data), options).decode()
}

/// 16 × 16 samples: two rows of two flat blocks.
fn image() -> Gray {
    Gray::new(16, 16, &[200, 60, 220, 40])
}

/// `file` with the segment `marker` `body` inserted before its EOI marker.
fn insert_before_eoi(file: &[u8], marker: u8, length: u16, body: &[u8]) -> Vec<u8> {
    let (head, eoi) = file.split_at(file.len() - 2);
    assert_eq!(eoi, [0xFF, 0xD9]);
    let mut out = head.to_vec();
    out.extend([0xFF, marker]);
    out.extend(length.to_be_bytes());
    out.extend(body);
    out.extend(eoi);
    out
}

fn with_dnl(file: &[u8], lines: u16) -> Vec<u8> {
    insert_before_eoi(file, 0xDC, 4, &lines.to_be_bytes())
}

/// Decodes `data` with no colour transform.
fn decode_as(data: &[u8], strict: bool, out: ColorSpace) -> Result<Vec<u8>, DecodeErrors> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_color_transform(Some(ColorTransform::None));
    decoder.decode()
}

/// `file` with a DNL segment of `lines` right after the entropy-coded data of its first scan,
/// where ITU-T T.81 B.2.5 places it.
fn with_dnl_after_first_scan(file: &[u8], lines: u16) -> Vec<u8> {
    let sos = marker_at(file, 0xDA);
    let length = usize::from(u16::from_be_bytes([file[sos + 2], file[sos + 3]]));
    let mut end = sos + 2 + length;
    // The first marker after the scan's data: not a stuffed byte and not a restart marker.
    while !(file[end] == 0xFF && !matches!(file[end + 1], 0x00 | 0xD0..=0xD7)) {
        end += 1;
    }
    let mut out = file[..end].to_vec();
    out.extend([0xFF, 0xDC, 0, 4]);
    out.extend(lines.to_be_bytes());
    out.extend(&file[end..]);
    out
}

/// A 32 × 16 frame of `layout`, its blocks of different values, coded as `coding`.
fn frame(layout: &[(u8, u8, u8)], coding: Coding) -> Frame {
    let mut frame = Frame::new(32, 16, layout, 128);
    frame.coding = coding;
    for c in 0..layout.len() {
        let (wide, high) = frame.grid(c);
        for by in 0..high {
            for bx in 0..wide {
                let value = (40 + 25 * (c + 2 * by + 3 * bx) % 180) as u8;
                frame.set(c, bx, by, flat(value));
            }
        }
    }
    frame
}

/// The offset of the first `0xFF code` marker in `file`.
fn marker_at(file: &[u8], code: u8) -> usize {
    file.windows(2).position(|w| w == [0xFF, code]).unwrap()
}

#[test]
fn dnl_after_the_scan_is_skipped_in_strict_and_lenient_mode() {
    let gray = image();
    let file = with_dnl(&gray.encode(), 16);
    for strict in [true, false] {
        assert_eq!(
            decode(&file, strict).unwrap(),
            gray.samples(),
            "strict mode {strict}"
        );
    }
}

#[test]
fn dnl_count_other_than_the_frame_header_decodes_at_the_frame_header_size() {
    // A placeholder count in the frame header and another in DNL: the frame header sized the
    // output and the scan, so the decode keeps it.
    let gray = image();
    for lines in [1, 9, 0xFFFF] {
        let file = with_dnl(&gray.encode(), lines);
        for strict in [true, false] {
            assert_eq!(
                decode(&file, strict).unwrap(),
                gray.samples(),
                "DNL {lines} lines, strict mode {strict}"
            );
        }
    }
}

#[test]
fn dnl_with_restart_markers_in_the_scan_is_skipped() {
    let mut gray = image();
    gray.restart = 1;
    let file = with_dnl(&gray.encode(), 16);
    for strict in [true, false] {
        assert_eq!(
            decode(&file, strict).unwrap(),
            gray.samples(),
            "strict mode {strict}"
        );
    }
}

#[test]
fn dnl_of_a_bad_length_fails_in_strict_mode_and_is_skipped_in_lenient_mode() {
    let gray = image();
    let file = insert_before_eoi(&gray.encode(), 0xDC, 6, &[0, 16, 0, 0]);
    assert!(decode(&file, true).is_err());
    assert_eq!(decode(&file, false).unwrap(), gray.samples());
}

#[test]
fn dnl_before_the_frame_header_fails() {
    let file = image().encode();
    let sof = marker_at(&file, 0xC0);
    let mut out = file[..sof].to_vec();
    out.extend([0xFF, 0xDC, 0, 4, 0, 16]);
    out.extend(&file[sof..]);
    for strict in [true, false] {
        assert!(decode(&out, strict).is_err(), "strict mode {strict}");
    }
}

#[test]
fn frame_header_of_zero_lines_with_dnl_still_fails() {
    let mut file = with_dnl(&image().encode(), 16);
    // SOF0: length (2), precision (1), then the number of lines.
    let lines = marker_at(&file, 0xC0) + 5;
    assert_eq!(file[lines..lines + 2], [0, 16]);
    file[lines..lines + 2].copy_from_slice(&[0, 0]);
    for strict in [true, false] {
        assert!(decode(&file, strict).is_err(), "strict mode {strict}");
    }
}

#[test]
fn dnl_after_the_first_scan_of_a_multi_scan_frame_is_skipped() {
    let layout = [(1, 1, 1), (2, 1, 1), (3, 1, 1)];
    for coding in [Coding::ScanPerComponent, Coding::Progressive] {
        for restart in [0, 2] {
            let mut frame = frame(&layout, coding);
            frame.restart = restart;
            let file = frame.encode();
            let dnl = with_dnl_after_first_scan(&file, 16);
            assert_ne!(dnl, file);
            for strict in [true, false] {
                assert_eq!(
                    decode_as(&dnl, strict, ColorSpace::RGB).unwrap(),
                    decode_as(&file, strict, ColorSpace::RGB).unwrap(),
                    "{coding:?}, restart {restart}, strict mode {strict}"
                );
            }
        }
    }
}

#[test]
fn dnl_after_the_first_scan_of_a_grey_progressive_frame_is_skipped() {
    let frame = frame(&[(1, 1, 1)], Coding::Progressive);
    let file = frame.encode();
    let dnl = with_dnl_after_first_scan(&file, 16);
    for strict in [true, false] {
        assert_eq!(
            decode_as(&dnl, strict, ColorSpace::Luma).unwrap(),
            decode_as(&file, strict, ColorSpace::Luma).unwrap(),
            "strict mode {strict}"
        );
    }
}
