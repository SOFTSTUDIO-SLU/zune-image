/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! YCCK decodes to CMYK: Y, Cb and Cr go to RGB as they do for a YCbCr image and are inverted
//! to C, M and Y; K is kept as stored.

mod common;

use common::color::{flat, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{ColorTransform, JpegDecoder};

fn decode(data: &[u8], transform: Option<ColorTransform>, out: ColorSpace) -> Vec<u8> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.set_color_transform(transform);
    decoder.decode().unwrap()
}

/// A 32×16 frame with the given layout; each component's blocks take turns over `values`.
fn frame(layout: &[(u8, u8, u8)], values: &[[u8; 4]], adobe: Option<u8>) -> Vec<u8> {
    let mut frame = Frame::new(32, 16, layout, 0);
    for (c, component) in frame.components.iter_mut().enumerate() {
        for (n, b) in component.blocks.iter_mut().enumerate() {
            *b = flat(values[n % values.len()][c]);
        }
    }
    frame.adobe = adobe;
    frame.encode()
}

const VALUES: [[u8; 4]; 4] = [
    [128, 128, 128, 0],
    [76, 85, 255, 40],
    [200, 30, 180, 255],
    [10, 240, 20, 128]
];

/// CMYK from a YCCK frame against RGB from the YCbCr frame with the same first three
/// components.
fn check(layout: &[(u8, u8, u8)], adobe: Option<u8>, transform: Option<ColorTransform>) {
    let cmyk = decode(&frame(layout, &VALUES, adobe), transform, ColorSpace::CMYK);
    let rgb = decode(&frame(&layout[..3], &VALUES, None), None, ColorSpace::RGB);
    // The stored K plane, from the same frame decoded without a transform.
    let stored = decode(
        &frame(layout, &VALUES, adobe),
        Some(ColorTransform::None),
        ColorSpace::CMYK
    );
    assert_eq!(cmyk.len(), 32 * 16 * 4);
    for ((c, r), s) in cmyk
        .chunks_exact(4)
        .zip(rgb.chunks_exact(3))
        .zip(stored.chunks_exact(4))
    {
        assert_eq!(c, &[255 - r[0], 255 - r[1], 255 - r[2], s[3]]);
    }
}

#[test]
fn adobe_ycck_decodes_to_cmyk() {
    check(&[(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)], Some(2), None);
}

#[test]
fn subsampled_ycck_decodes_to_cmyk() {
    check(&[(1, 2, 2), (2, 1, 1), (3, 1, 1), (4, 2, 2)], Some(2), None);
}

#[test]
fn the_callers_ycck_transform_decodes_to_cmyk_without_a_marker() {
    let layout = [(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)];
    check(&layout, None, Some(ColorTransform::YcckToCmyk));
    check(&layout, Some(0), Some(ColorTransform::YcckToCmyk));
}

#[test]
fn neutral_chroma_inverts_the_luma() {
    let data = frame(&[(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)], &[[77, 128, 128, 33]], Some(2));
    for px in decode(&data, None, ColorSpace::CMYK).chunks_exact(4) {
        assert_eq!(px, [255 - 77, 255 - 77, 255 - 77, 33]);
    }
}
