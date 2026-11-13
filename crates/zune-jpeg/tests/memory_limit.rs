/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! The memory limit: `JpegDecoder::set_max_memory` and `JpegDecoder::estimated_memory`.
//!
//! A counting global allocator measures what a decode really allocates, so this file holds a
//! single test that runs its cases one after the other: tests running in parallel threads
//! would count each other's allocations.
//!
//! The JPEG files are written by the test itself: every block is flat (a DC difference of 0 and
//! an immediate end of block), and both Huffman tables have a single one-bit code for the
//! symbol 0, so the entropy-coded data is one zero bit per coded symbol. The marker layout
//! follows ITU-T T.81 Annex B; progressive scans (a first DC scan for all components, then one
//! first AC scan per component, where the symbol 0 ends one block) follow its Annex G.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::JpegDecoder;

// --- A counting allocator ------------------------------------------------------------------------

/// Counts the bytes in use and their peak, and refuses any allocation that would take the bytes
/// in use above `CAP` (as an address-space limit would), so that an aborting allocation fails
/// the test instead of exhausting the machine.
struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CAP: AtomicUsize = AtomicUsize::new(usize::MAX);

impl Counting {
    fn admit(size: usize) -> bool {
        CURRENT.load(Ordering::SeqCst).saturating_add(size) <= CAP.load(Ordering::SeqCst)
    }
    fn added(size: usize) {
        let now = CURRENT.fetch_add(size, Ordering::SeqCst) + size;
        PEAK.fetch_max(now, Ordering::SeqCst);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !Self::admit(layout.size()) {
            return core::ptr::null_mut();
        }
        let p = System.alloc(layout);
        if !p.is_null() {
            Self::added(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !Self::admit(layout.size()) {
            return core::ptr::null_mut();
        }
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            Self::added(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() && !Self::admit(new_size - layout.size()) {
            return core::ptr::null_mut();
        }
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            if new_size >= layout.size() {
                Self::added(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f` and returns its result with the most bytes it had allocated at once.
fn peak_of<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let r = f();
    (r, PEAK.load(Ordering::SeqCst) - base)
}

// --- A writer of flat JPEG images ----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coding {
    /// Sequential, all components in one scan.
    Interleaved,
    /// Sequential, one scan per component.
    ScanPerComponent,
    /// Progressive: a DC scan of all components, then an AC scan per component.
    Progressive
}

#[derive(Clone, Debug)]
struct Frame {
    width:   u16,
    height:  u16,
    /// Horizontal and vertical sampling factors per component.
    factors: Vec<(u8, u8)>,
    coding:  Coding
}

impl Frame {
    fn new(width: u16, height: u16, factors: &[(u8, u8)], coding: Coding) -> Frame {
        Frame { width, height, factors: factors.to_vec(), coding }
    }

    fn max_factors(&self) -> (usize, usize) {
        let h = self.factors.iter().map(|f| usize::from(f.0)).max().unwrap();
        let v = self.factors.iter().map(|f| usize::from(f.1)).max().unwrap();
        (h, v)
    }

    /// The blocks of component `c` in a scan of that component alone.
    fn component_blocks(&self, c: usize) -> usize {
        let (h_max, v_max) = self.max_factors();
        let (h, v) = (usize::from(self.factors[c].0), usize::from(self.factors[c].1));
        let w = (usize::from(self.width) * h).div_ceil(h_max);
        let ht = (usize::from(self.height) * v).div_ceil(v_max);
        w.div_ceil(8) * ht.div_ceil(8)
    }

    /// The blocks of a scan of all components (interleaved when there is more than one).
    fn all_blocks(&self) -> usize {
        if self.factors.len() == 1 {
            return self.component_blocks(0);
        }
        let (h_max, v_max) = self.max_factors();
        let mcus = usize::from(self.width).div_ceil(8 * h_max)
            * usize::from(self.height).div_ceil(8 * v_max);
        let per_mcu: usize = self.factors.iter().map(|&(h, v)| usize::from(h * v)).sum();
        mcus * per_mcu
    }

    fn encode(&self) -> Vec<u8> {
        self.encode_with(true)
    }

    /// The image, with its entropy-coded data or, without `data`, with empty scans.
    fn encode_with(&self, data: bool) -> Vec<u8> {
        let bits = |n: usize| if data { n } else { 0 };
        let n = self.factors.len();
        let mut out = vec![0xFF, 0xD8];
        // One quantization table of ones.
        segment(&mut out, 0xDB, &[&[0x00][..], &[1; 64][..]].concat());
        // The frame header.
        let sof = if self.coding == Coding::Progressive { 0xC2 } else { 0xC0 };
        let mut frame = vec![8];
        frame.extend_from_slice(&self.height.to_be_bytes());
        frame.extend_from_slice(&self.width.to_be_bytes());
        frame.push(n as u8);
        for (i, &(h, v)) in self.factors.iter().enumerate() {
            frame.extend_from_slice(&[i as u8 + 1, (h << 4) | v, 0]);
        }
        segment(&mut out, sof, &frame);
        // A DC and an AC table, each with one code of length 1 for the symbol 0.
        let table = |class: u8| {
            let mut t = vec![class, 1];
            t.extend_from_slice(&[0; 15]);
            t.push(0);
            t
        };
        let tables = [table(0x00), table(0x10)].concat();
        segment(&mut out, 0xC4, &tables);

        let all: Vec<usize> = (0..n).collect();
        match self.coding {
            Coding::Interleaved => scan(&mut out, &all, 0, 63, bits(2 * self.all_blocks())),
            Coding::ScanPerComponent => {
                for c in 0..n {
                    scan(&mut out, &[c], 0, 63, bits(2 * self.component_blocks(c)));
                }
            }
            Coding::Progressive => {
                scan(&mut out, &all, 0, 0, bits(self.all_blocks()));
                for c in 0..n {
                    scan(&mut out, &[c], 1, 63, bits(self.component_blocks(c)));
                }
            }
        }
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }
}

/// A marker segment: the marker, its length (which counts itself) and `body`.
fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    out.extend_from_slice(body);
}

/// A scan of `components` over the spectral band `ss..=se` and its data: `bits` zero bits,
/// the last byte padded with ones.
fn scan(out: &mut Vec<u8>, components: &[usize], ss: u8, se: u8, bits: usize) {
    let mut body = vec![components.len() as u8];
    for &c in components {
        body.extend_from_slice(&[c as u8 + 1, 0x00]);
    }
    body.extend_from_slice(&[ss, se, 0x00]);
    segment(out, 0xDA, &body);
    out.resize(out.len() + bits / 8, 0);
    if bits % 8 != 0 {
        out.push(0xFF >> (bits % 8));
    }
}

// --- The cases -----------------------------------------------------------------------------------

fn options(out: ColorSpace) -> DecoderOptions {
    lenient(out).set_strict_mode(true)
}

fn lenient(out: ColorSpace) -> DecoderOptions {
    DecoderOptions::default()
        .set_strict_mode(false)
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX))
        .jpeg_set_out_colorspace(out)
}

/// The estimate is an upper bound of what a decode allocates, for every coding, sampling and
/// output colour space, in the strict mode and in the lenient one, with the upsampling inverse
/// DCT and without it, and a close one for progressive frames. (A few combinations of
/// vertical subsampling and a grey output fail the strict mode on these images: the decoder
/// reads more blocks than the scan has. The lenient mode decodes them.)
fn estimate_bounds_the_peak() {
    let samplings: &[&[(u8, u8)]] = &[
        &[(1, 1)],
        &[(2, 2)],
        &[(1, 1), (1, 1), (1, 1)],
        &[(2, 2), (1, 1), (1, 1)],
        &[(2, 1), (1, 1), (1, 1)],
        &[(1, 2), (1, 1), (1, 1)],
        &[(4, 1), (1, 1), (1, 1)],
        &[(4, 2), (1, 1), (1, 1)],
        &[(2, 2), (2, 1), (1, 1)],
        &[(1, 1), (1, 1), (1, 1), (1, 1)],
        &[(2, 2), (1, 1), (1, 1), (2, 2)]
    ];
    let codings = [Coding::Interleaved, Coding::ScanPerComponent, Coding::Progressive];
    let sizes = [(37, 23), (301, 157)];
    for factors in samplings {
        for coding in codings {
            for (width, height) in sizes {
                let frame = Frame::new(width, height, factors, coding);
                let jpeg = frame.encode();
                let outs: &[ColorSpace] = match factors.len() {
                    1 => &[ColorSpace::Luma],
                    3 => &[ColorSpace::RGB, ColorSpace::Luma, ColorSpace::YCbCr],
                    _ => &[ColorSpace::CMYK, ColorSpace::RGB]
                };
                for &out in outs {
                    for (strict, idct) in [(true, false), (false, false), (true, true), (false, true)] {
                        let options = if strict { options(out) } else { lenient(out) };
                        let mut decoder =
                            JpegDecoder::new_with_options(ZCursor::new(&jpeg), options);
                        decoder.set_idct_upsampling(idct);
                        decoder.decode_headers().unwrap();
                        let estimate = decoder.estimated_memory().unwrap();
                        let (pixels, peak) = peak_of(|| decoder.decode());
                        // An error message is allocated while the buffers are still there.
                        let message = if pixels.is_err() { 256 } else { 0 };
                        assert!(
                            peak <= estimate + message,
                            "{frame:?} to {out:?} (idct {idct}): peak {peak} > estimate {estimate} ({pixels:?})"
                        );
                        // Tight enough to charge against a budget: where the coefficients
                        // dominate, within a third of the peak.
                        if coding == Coding::Progressive && width > 300 && pixels.is_ok() {
                            assert!(
                                3 * estimate <= 4 * peak,
                                "{frame:?} to {out:?}: estimate {estimate} for a peak of {peak}"
                            );
                        }
                        match pixels {
                            Ok(pixels) => {
                                assert_eq!(pixels.len(), decoder.output_buffer_size().unwrap());
                            }
                            Err(e) if strict => eprintln!("strict (idct {idct}): {frame:?} to {out:?}: {e:?}"),
                            Err(e) => panic!("{frame:?} to {out:?} (idct {idct}): {e:?}")
                        }
                    }
                }
            }
        }
    }
}

/// The estimate counts whole MCUs: a progressive 17 × 9 grey image keeps 3 × 2 blocks of 64
/// coefficients, 2 bytes each, besides its 153 output bytes.
fn estimate_counts_padded_coefficients() {
    let jpeg = Frame::new(17, 9, &[(1, 1)], Coding::Progressive).encode();
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::Luma));
    assert_eq!(decoder.estimated_memory(), None);
    decoder.decode_headers().unwrap();
    let estimate = decoder.estimated_memory().unwrap();
    assert!(estimate >= 17 * 9 + 2 * 64 * 3 * 2, "{estimate}");

    // The same image, sequential and in one scan, keeps no coefficient plane.
    let jpeg = Frame::new(17, 9, &[(1, 1)], Coding::Interleaved).encode();
    let mut sequential =
        JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::Luma));
    sequential.decode_headers().unwrap();
    assert!(sequential.estimated_memory().unwrap() < 17 * 9 + 2 * 64 * 3 * 2);
}

/// A header of a few hundred bytes declaring a 65,535 × 65,535 four-component progressive frame
/// (some 34 GB of coefficients): over a 512 MiB limit, the decode fails before allocating.
fn huge_progressive_frame_is_refused_before_allocating() {
    let jpeg = Frame::new(u16::MAX, u16::MAX, &[(1, 1); 4], Coding::Progressive).encode_header();
    assert!(jpeg.len() < 400, "{}", jpeg.len());
    let limit = 512 << 20;

    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::CMYK));
    decoder.set_max_memory(Some(limit));
    assert_eq!(decoder.max_memory(), Some(limit));
    let (result, peak) = peak_of(|| decoder.decode());
    match result {
        Err(DecodeErrors::MemoryLimitExceeded(needed, l)) => {
            assert_eq!(l, limit);
            assert!(needed as u64 >= 4 * 2 * 65_536 * 65_536, "{needed}");
        }
        other => panic!("{:?}", other.map(|p| p.len()))
    }
    assert!(peak < 1 << 20, "{peak}");

    // decode_into checks the limit too, whatever the buffer.
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::CMYK));
    decoder.set_max_memory(Some(limit));
    let (result, peak) = peak_of(|| decoder.decode_into(&mut []));
    assert!(matches!(result, Err(DecodeErrors::MemoryLimitExceeded(_, _))), "{result:?}");
    assert!(peak < 1 << 20, "{peak}");
}

/// Without a limit, a coefficient plane the allocator refuses is an error, not an abort.
fn refused_coefficient_plane_is_an_error() {
    // 4,096 × 4,096 × 4 progressive, decoded to one grey component: the output takes 16 MiB and
    // each coefficient plane 32 MiB, so a 64 MiB cap refuses the second plane.
    let jpeg = Frame::new(4096, 4096, &[(1, 1); 4], Coding::Progressive).encode_header();
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::Luma));
    decoder.decode_headers().unwrap();
    CAP.store(CURRENT.load(Ordering::SeqCst) + (64 << 20), Ordering::SeqCst);
    let result = decoder.decode();
    CAP.store(usize::MAX, Ordering::SeqCst);
    match result {
        Err(DecodeErrors::OutOfMemory(bytes)) => assert_eq!(bytes, 2 * 4096 * 4096),
        other => panic!("{:?}", other.map(|p| p.len()))
    }

    // The same for the output buffer of decode().
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&jpeg), options(ColorSpace::CMYK));
    decoder.decode_headers().unwrap();
    CAP.store(CURRENT.load(Ordering::SeqCst) + (32 << 20), Ordering::SeqCst);
    let result = decoder.decode();
    CAP.store(usize::MAX, Ordering::SeqCst);
    match result {
        Err(DecodeErrors::OutOfMemory(bytes)) => assert_eq!(bytes, 4 * 4096 * 4096),
        other => panic!("{:?}", other.map(|p| p.len()))
    }
}

impl Frame {
    /// The headers and the first scan's header, without data: what a decoder sees before it
    /// allocates anything the size of the frame.
    fn encode_header(&self) -> Vec<u8> {
        let mut jpeg = self.encode_with(false);
        let sos = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        let len = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
        jpeg.truncate(sos + 2 + len);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        jpeg
    }
}

#[test]
fn memory_limit() {
    estimate_bounds_the_peak();
    estimate_counts_padded_coefficients();
    huge_progressive_frame_is_refused_before_allocating();
    refused_coefficient_plane_is_an_error();
}
