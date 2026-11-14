/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! The colour space is derived from what the headers say each time it is settled: settling
//! again changes nothing, the headers count as decoded only once it settles, and a change of
//! the transform or the options after the header pass derives it again from the file.

mod common;

use common::color::{flat, Frame};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{ColorTransform, JpegDecoder};

/// A 16×16 frame without subsampling whose component `c` is flat with `60 + 50 * c`.
fn frame(components: usize, adobe: Option<u8>) -> Frame {
    let layout: Vec<(u8, u8, u8)> = (1..=components as u8).map(|id| (id, 1, 1)).collect();
    let mut frame = Frame::new(16, 16, &layout, 0);
    for (c, component) in frame.components.iter_mut().enumerate() {
        for b in &mut component.blocks {
            *b = flat(60 + 50 * c as u8);
        }
    }
    frame.adobe = adobe;
    frame
}

/// `data` (from [`Frame::encode`], whose Adobe APP14 segment comes first) with the APP14
/// segment moved after the frame header, so that it is read after the frame header sets the
/// colour space.
fn app14_after_sof(data: &[u8]) -> Vec<u8> {
    assert_eq!(&data[2..4], &[0xFF, 0xEE]);
    let length = |at: usize| 2 + usize::from(u16::from_be_bytes([data[at + 2], data[at + 3]]));
    let app14 = 2..2 + length(2);
    let mut at = app14.end;
    while !matches!(data[at + 1], 0xC0 | 0xC2) {
        at += length(at);
    }
    let sof_end = at + length(at);
    let mut out = data[..2].to_vec();
    out.extend(&data[app14.end..sof_end]);
    out.extend(&data[app14.clone()]);
    out.extend(&data[sof_end..]);
    out
}

fn options(strict: bool, out: ColorSpace) -> DecoderOptions {
    DecoderOptions::default().set_strict_mode(strict).jpeg_set_out_colorspace(out)
}

#[test]
fn an_adobe_ycck_flag_on_one_or_two_components_decodes_as_before_the_colour_changes() {
    // The frame has fewer components than YCCK, so the decoder's rules take it as YCbCr, as
    // stb_image and libjpeg-turbo do. Settling a second time once took that YCbCr, again with
    // more components than the frame, to MultiBand, or failed in strict mode. The expected
    // samples are those of the decoder before the colour changes.
    let cases: [(usize, bool, &[u8], &[u8]); 4] = [
        // components, APP14 after SOF, RGB output pixel, Luma output pixel
        (1, false, &[60, 60, 60], &[60]),
        (1, true, &[0, 0, 0], &[60]),
        (2, false, &[0, 0, 0], &[60]),
        (2, true, &[0, 0, 0], &[60]),
    ];
    for (components, after_sof, rgb, luma) in cases {
        let mut data = frame(components, Some(2)).encode();
        if after_sof {
            data = app14_after_sof(&data);
        }
        for strict in [false, true] {
            for (out, pixel) in [(ColorSpace::RGB, rgb), (ColorSpace::Luma, luma)] {
                let what = format!("{components} components, after SOF {after_sof}, strict {strict}, {out:?}");
                let mut decoder =
                    JpegDecoder::new_with_options(ZCursor::new(&data[..]), options(strict, out));
                decoder.decode_headers().unwrap();
                let settled = decoder.input_colorspace();
                let samples = decoder.decode().unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(decoder.input_colorspace(), settled, "{what}");
                assert_eq!(samples.len(), 16 * 16 * pixel.len(), "{what}");
                for px in samples.chunks_exact(pixel.len()) {
                    assert_eq!(px, pixel, "{what}");
                }
            }
        }
    }
}

#[test]
fn a_header_pass_whose_colour_space_fails_to_settle_leaves_no_headers() {
    // Two components with an Adobe flag of 0 (CMYK) fail in strict mode; three components do
    // not fit the YCCK transform.
    let strict_cmyk = frame(2, Some(0)).encode();
    let three = frame(3, None).encode();
    let cases: [(&[u8], Option<ColorTransform>); 2] =
        [(&strict_cmyk, None), (&three, Some(ColorTransform::YcckToCmyk))];
    for (data, transform) in cases {
        let mut decoder =
            JpegDecoder::new_with_options(ZCursor::new(data), options(true, ColorSpace::RGB));
        decoder.set_color_transform(transform);
        assert!(decoder.decode_headers().is_err());
        assert_eq!(decoder.input_colorspace(), None);
        assert_eq!(decoder.estimated_memory(), None);
        assert_eq!(decoder.adobe_transform(), None);
        assert!(decoder.info().is_none());
        assert_eq!(decoder.output_buffer_size(), None);
        // A second header pass fails the same way, as does the decode.
        assert!(decoder.decode_headers().is_err());
        assert_eq!(decoder.input_colorspace(), None);
        assert!(decoder.decode().is_err());
    }
}

#[test]
fn a_transform_that_fits_after_a_failed_header_pass_decodes() {
    let data = frame(3, None).encode();
    let mut decoder = JpegDecoder::new(ZCursor::new(&data[..]));
    decoder.set_color_transform(Some(ColorTransform::YcckToCmyk));
    assert!(decoder.decode_headers().is_err());
    decoder.set_color_transform(Some(ColorTransform::None));
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::RGB));
    let samples = decoder.decode().unwrap();
    for px in samples.chunks_exact(3) {
        assert_eq!(px, [60, 110, 160]);
    }
}

#[test]
fn a_transform_reset_after_the_header_pass_gives_the_files_colour_space() {
    let data = frame(3, None).encode();
    let expected = JpegDecoder::new(ZCursor::new(&data[..])).decode().unwrap();

    let mut decoder = JpegDecoder::new(ZCursor::new(&data[..]));
    decoder.set_color_transform(Some(ColorTransform::None));
    decoder.decode_headers().unwrap();
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::RGB));
    decoder.set_color_transform(None);
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::YCbCr));
    assert_eq!(decoder.decode().unwrap(), expected);

    // And the other way round: the transform set after the header pass applies.
    let mut decoder = JpegDecoder::new(ZCursor::new(&data[..]));
    decoder.decode_headers().unwrap();
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::YCbCr));
    decoder.set_color_transform(Some(ColorTransform::None));
    assert_eq!(decoder.input_colorspace(), Some(ColorSpace::RGB));
    for px in decoder.decode().unwrap().chunks_exact(3) {
        assert_eq!(px, [60, 110, 160]);
    }
}

#[test]
fn strict_mode_set_after_the_header_pass_applies_to_the_colour_space() {
    // Two components with an Adobe flag of 0: MultiBand when lenient, an error when strict.
    let data = frame(2, Some(0)).encode();
    let mut decoder =
        JpegDecoder::new_with_options(ZCursor::new(&data[..]), options(false, ColorSpace::RGB));
    decoder.decode_headers().unwrap();
    assert!(matches!(decoder.input_colorspace(), Some(ColorSpace::MultiBand(n)) if n.get() == 2));
    decoder.set_options(options(true, ColorSpace::RGB));
    assert_eq!(decoder.input_colorspace(), None);
    assert!(decoder.decode().is_err());
    // Lenient again: it settles as before.
    decoder.set_options(options(false, ColorSpace::RGB));
    assert!(matches!(decoder.input_colorspace(), Some(ColorSpace::MultiBand(n)) if n.get() == 2));
}
