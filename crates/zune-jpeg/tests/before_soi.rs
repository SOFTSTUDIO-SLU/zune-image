/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Bytes before the SOI marker: `JpegDecoder::set_skip_before_soi` skips them in lenient mode
//! and `JpegDecoder::bytes_before_soi` counts them; strict mode, and the default, refuse them.

mod common;

use common::Gray;
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::JpegDecoder;

/// Decodes `data` to grey; the samples and the bytes skipped before SOI.
fn decode(data: &[u8], strict: bool, skip: bool) -> Result<(Vec<u8>, usize), DecodeErrors> {
    let options = DecoderOptions::default()
        .set_strict_mode(strict)
        .jpeg_set_out_colorspace(ColorSpace::Luma);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_skip_before_soi(skip);
    let samples = decoder.decode()?;
    Ok((samples, decoder.bytes_before_soi()))
}

fn image() -> Gray {
    Gray::new(16, 16, &[200, 60, 30, 240])
}

fn prefixed(junk: &[u8], jpeg: &[u8]) -> Vec<u8> {
    let mut data = junk.to_vec();
    data.extend_from_slice(jpeg);
    data
}

fn assert_illegal_magic(result: Result<(Vec<u8>, usize), DecodeErrors>) {
    match result {
        Err(DecodeErrors::IllegalMagicBytes(_)) => {}
        Err(e) => panic!("expected IllegalMagicBytes, got {e:?}"),
        Ok(_) => panic!("expected IllegalMagicBytes, got an image")
    }
}

#[test]
fn leading_bytes_are_skipped_in_lenient_mode() {
    let gray = image();
    let junk = b"\r\n%junk ";
    let (samples, skipped) = decode(&prefixed(junk, &gray.encode()), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
}

#[test]
fn a_stray_soi_not_followed_by_a_marker_is_skipped_too() {
    let gray = image();
    let junk = [0x00, 0xFF, 0xD8, 0x00, 0x41, 0xFF, 0xD8, 0x12, 0xFF];
    let (samples, skipped) = decode(&prefixed(&junk, &gray.encode()), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
}

#[test]
fn data_that_starts_with_soi_skips_nothing() {
    let gray = image();
    let (samples, skipped) = decode(&gray.encode(), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, 0);
}

#[test]
fn leading_bytes_fail_in_strict_mode() {
    let data = prefixed(b"\r\n%junk ", &image().encode());
    assert_illegal_magic(decode(&data, true, true));
}

#[test]
fn leading_bytes_fail_without_the_option() {
    let data = prefixed(b"\r\n%junk ", &image().encode());
    assert_illegal_magic(decode(&data, false, false));
    assert_illegal_magic(decode(&data, true, false));
}

#[test]
fn data_without_an_soi_fails() {
    assert_illegal_magic(decode(b"no image here", false, true));
    // An SOI at the very end has no marker after it.
    assert_illegal_magic(decode(&[0x41, 0x42, 0xFF, 0xD8], false, true));
    assert_illegal_magic(decode(&[0x41, 0x42, 0xFF, 0xD8, 0xFF], false, true));
}
