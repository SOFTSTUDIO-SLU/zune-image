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

/// `jpeg` with an APPn segment (`app`) right after its SOI that carries `header` and then a
/// whole JPEG file, `thumbnail`, as Exif and JFXX segments carry a thumbnail.
fn with_thumbnail(jpeg: &[u8], app: u8, header: &[u8], thumbnail: &[u8]) -> Vec<u8> {
    let mut body = header.to_vec();
    body.extend_from_slice(thumbnail);
    let mut data = vec![0xFF, 0xD8, 0xFF, app];
    data.extend(u16::try_from(body.len() + 2).unwrap().to_be_bytes());
    data.extend(body);
    data.extend_from_slice(&jpeg[2..]);
    data
}

/// A 16 × 16 image with an Exif segment that carries an 8 × 8 thumbnail of another value.
fn exif_image() -> (Gray, Vec<u8>) {
    let gray = image();
    let thumbnail = Gray::new(8, 8, &[90]);
    let data = with_thumbnail(&gray.encode(), 0xE1, b"Exif\0\0", &thumbnail.encode());
    (gray, data)
}

#[test]
fn an_image_with_a_thumbnail_decodes_the_image() {
    let (gray, data) = exif_image();
    for strict in [false, true] {
        let (samples, skipped) = decode(&data, strict, true).unwrap();
        assert_eq!(samples, gray.samples());
        assert_eq!(skipped, 0);
    }
}

#[test]
fn bytes_before_an_image_with_a_thumbnail_skip_to_the_image() {
    let (gray, data) = exif_image();
    let junk = b"\r\n%junk ";
    let (samples, skipped) = decode(&prefixed(junk, &data), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
}

#[test]
fn a_damaged_soi_does_not_decode_an_exif_thumbnail() {
    let (_, data) = exif_image();
    for damaged in [[0x00, 0x00], [0x00, 0xD8], [0xFF, 0x00], [0xD8, 0xFF]] {
        let mut data = data.clone();
        data[..2].copy_from_slice(&damaged);
        assert_illegal_magic(decode(&data, false, true));
    }
}

#[test]
fn a_damaged_soi_does_not_decode_a_jfxx_thumbnail() {
    let thumbnail = Gray::new(8, 8, &[90]);
    // JFXX extension code 0x10: a thumbnail coded with JPEG.
    let mut data = with_thumbnail(&image().encode(), 0xE0, b"JFXX\0\x10", &thumbnail.encode());
    data[0] = 0x00;
    assert_illegal_magic(decode(&data, false, true));
}

#[test]
fn a_damaged_soi_with_a_segment_before_the_thumbnail_segment() {
    // A COM segment, then the Exif segment: both are passed over.
    let (_, exif) = exif_image();
    let mut data = vec![0x00, 0x00, 0xFF, 0xFE, 0x00, 0x05, b'a', b'b', b'c'];
    data.extend_from_slice(&exif[2..]);
    assert_illegal_magic(decode(&data, false, true));
}

#[test]
fn a_segment_in_the_leading_bytes_that_ends_on_the_soi_is_passed_over() {
    let gray = image();
    // FF E1 with length 3 ends right before the image's SOI.
    let junk = [0xFF, 0xE1, 0x00, 0x03, 0x41];
    let (samples, skipped) = decode(&prefixed(&junk, &gray.encode()), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
}

#[test]
fn a_segment_length_that_does_not_end_on_a_marker_is_taken_for_stray_bytes() {
    let gray = image();
    // Length 4 would end on the D8 of the SOI, not on a marker: the bytes are scanned.
    let junk = [0xFF, 0xE1, 0x00, 0x04, 0x41];
    let (samples, skipped) = decode(&prefixed(&junk, &gray.encode()), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
    // A length past the end of the data, too.
    let junk = [0xFF, 0xE1, 0xFF, 0xF0, 0x41];
    let (samples, skipped) = decode(&prefixed(&junk, &gray.encode()), false, true).unwrap();
    assert_eq!(samples, gray.samples());
    assert_eq!(skipped, junk.len());
}

#[test]
fn a_segment_like_run_that_leaps_over_the_real_soi_is_refused() {
    // The rule cannot tell such junk from a real segment; it refuses rather than guesses. The
    // length (73, counted from the length field) ends right before the image's SOF marker: the
    // DQT segment after the image's SOI is 2 + 67 bytes. The search then passes over the
    // image's own segments, finds no SOI followed by a marker, and fails.
    let jpeg = image().encode();
    assert_eq!(&jpeg[2..4], &[0xFF, 0xDB]);
    assert_eq!(&jpeg[71..73], &[0xFF, 0xC0]);
    let junk = [0xFF, 0xE1, 0x00, 73];
    assert_illegal_magic(decode(&prefixed(&junk, &jpeg), false, true));
}
