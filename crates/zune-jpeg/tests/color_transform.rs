/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! The colour space after the header pass is the one the decode uses,
//! `JpegDecoder::adobe_transform` reports the Adobe APP14 flag, and
//! `JpegDecoder::set_color_transform` replaces the decoder's colour rules.

mod common;

use common::color::{flat, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::{ColorTransform, JpegDecoder};

/// Three components, 16×16, no subsampling; component `c` is flat with `values[c]`.
fn three(ids: [u8; 3], values: [u8; 3], adobe: Option<u8>) -> Vec<u8> {
    let mut frame = Frame::new(
        16,
        16,
        &[(ids[0], 1, 1), (ids[1], 1, 1), (ids[2], 1, 1)],
        0
    );
    for (c, &value) in values.iter().enumerate() {
        for b in &mut frame.components[c].blocks {
            *b = flat(value);
        }
    }
    frame.adobe = adobe;
    frame.encode()
}

/// Four components, 16×16, no subsampling, each flat.
fn four(values: [u8; 4], adobe: Option<u8>) -> Vec<u8> {
    let mut frame = Frame::new(16, 16, &[(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)], 0);
    for (c, &value) in values.iter().enumerate() {
        for b in &mut frame.components[c].blocks {
            *b = flat(value);
        }
    }
    frame.adobe = adobe;
    frame.encode()
}

fn headers(data: &[u8], transform: Option<ColorTransform>) -> JpegDecoder<ZCursor<&[u8]>> {
    let mut decoder = JpegDecoder::new(ZCursor::new(data));
    decoder.set_color_transform(transform);
    decoder.decode_headers().unwrap();
    decoder
}

fn decode(
    data: &[u8], transform: Option<ColorTransform>, out: ColorSpace
) -> Result<Vec<u8>, DecodeErrors> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_color_transform(transform);
    decoder.decode()
}

/// Every pixel of `samples` is `pixel`.
fn assert_all(samples: &[u8], pixel: &[u8]) {
    assert!(!samples.is_empty());
    for px in samples.chunks_exact(pixel.len()) {
        assert_eq!(px, pixel);
    }
}

#[test]
fn the_adobe_flag_is_reported() {
    for flag in [0, 1, 2] {
        let data = four([10, 20, 30, 40], Some(flag));
        assert_eq!(headers(&data, None).adobe_transform(), Some(flag));
    }
    let data = three([1, 2, 3], [10, 20, 30], None);
    assert_eq!(headers(&data, None).adobe_transform(), None);
    // Not before the headers are decoded.
    let data = three([1, 2, 3], [10, 20, 30], Some(1));
    assert_eq!(JpegDecoder::new(ZCursor::new(&data[..])).adobe_transform(), None);
}

#[test]
fn three_components_with_adobe_flag_0_are_rgb_after_the_header_pass() {
    // The flag says "no transform"; for four components that is CMYK, for three RGB. The
    // header pass used to report CMYK here and correct it only when the decode started.
    let data = three([1, 2, 3], [10, 20, 30], Some(0));
    let decoder = headers(&data, None);
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::RGB));
    assert_all(&decode(&data, None, ColorSpace::RGB).unwrap(), &[10, 20, 30]);
}

#[test]
fn ycck_flag_with_three_components_is_ycbcr_after_the_header_pass() {
    let data = three([1, 2, 3], [128, 128, 128], Some(2));
    assert_eq!(headers(&data, None).input_colorspace(), Some(ColorSpace::YCbCr));
}

#[test]
fn the_decoders_rules_are_unchanged_without_a_transform() {
    let data = three([1, 2, 3], [10, 20, 30], None);
    assert_eq!(headers(&data, None).input_colorspace(), Some(ColorSpace::YCbCr));
    let data = three(*b"RGB", [10, 20, 30], None);
    assert_eq!(headers(&data, None).input_colorspace(), Some(ColorSpace::RGB));
    let data = four([10, 20, 30, 40], None);
    assert_eq!(headers(&data, None).input_colorspace(), Some(ColorSpace::CMYK));
    let data = four([10, 20, 30, 40], Some(2));
    assert_eq!(headers(&data, None).input_colorspace(), Some(ColorSpace::YCCK));
}

#[test]
fn no_transform_gives_the_stored_samples() {
    // The Adobe marker says YCbCr; the caller says the samples are the colours.
    let data = three([1, 2, 3], [10, 20, 30], Some(1));
    let transform = Some(ColorTransform::None);
    assert_eq!(headers(&data, transform).input_colorspace(), Some(ColorSpace::RGB));
    assert_all(&decode(&data, transform, ColorSpace::RGB).unwrap(), &[10, 20, 30]);

    let data = four([10, 20, 30, 40], Some(2));
    assert_eq!(headers(&data, transform).input_colorspace(), Some(ColorSpace::CMYK));
    assert_all(&decode(&data, transform, ColorSpace::CMYK).unwrap(), &[10, 20, 30, 40]);

    let data = Frame::new(16, 16, &[(1, 1, 1)], 77).encode();
    assert_eq!(headers(&data, transform).input_colorspace(), Some(ColorSpace::Luma));
    assert_all(&decode(&data, transform, ColorSpace::Luma).unwrap(), &[77]);
}

#[test]
fn ycbcr_transform_overrides_the_marker_and_the_component_ids() {
    // What the decoder makes of YCbCr samples by its own rules (no marker, ordinary ids).
    let reference = decode(&three([1, 2, 3], [90, 60, 200], None), None, ColorSpace::RGB).unwrap();
    assert_ne!(&reference[..3], &[90, 60, 200]);

    let transform = Some(ColorTransform::YCbCrToRgb);
    for data in [
        three([1, 2, 3], [90, 60, 200], Some(0)),
        three(*b"RGB", [90, 60, 200], None)
    ] {
        assert_eq!(headers(&data, transform).input_colorspace(), Some(ColorSpace::YCbCr));
        assert_eq!(decode(&data, transform, ColorSpace::RGB).unwrap(), reference);
    }
}

#[test]
fn a_transform_that_does_not_fit_the_component_count_fails_the_header_pass() {
    let cases = [
        (four([1, 2, 3, 4], None), ColorTransform::YCbCrToRgb),
        (three([1, 2, 3], [1, 2, 3], None), ColorTransform::YcckToCmyk),
        (Frame::new(8, 8, &[(1, 1, 1)], 9).encode(), ColorTransform::YCbCrToRgb)
    ];
    for (data, transform) in cases {
        let mut decoder = JpegDecoder::new(ZCursor::new(&data[..]));
        decoder.set_color_transform(Some(transform));
        match decoder.decode_headers() {
            Err(DecodeErrors::Format(_)) => {}
            other => panic!("expected a format error for {transform:?}, got {other:?}")
        }
    }
}
