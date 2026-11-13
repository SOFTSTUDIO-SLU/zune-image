/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A writer of baseline JPEG files with one to four components, any sampling factors, an
//! optional Adobe APP14 marker and any quantized coefficients, for the tests.
//!
//! The quantization table has step 1 for every coefficient, so a block's coefficients are the
//! dequantized ones the decoder works with: a flat block of value `v` has the DC coefficient
//! `8 * (v - 128)`. Both Huffman tables give every symbol an eight-bit code, the symbols in
//! increasing order, so that any coefficient can be coded. A frame of more than one component
//! is one interleaved scan. The marker layout follows ITU-T T.81 Annex B; the APP14 segment of
//! Adobe follows Adobe Technical Note #5116.

#![allow(dead_code)]

use super::{pack, push, segment};

/// The natural (row-major) index of each coefficient, in zig-zag order.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
];

/// One component: its identifier, sampling factors and blocks.
pub struct Component {
    pub id:     u8,
    pub h:      u8,
    pub v:      u8,
    /// Coefficients in natural order, one block after another in row order over the
    /// component's block grid (whole MCUs).
    pub blocks: Vec<[i32; 64]>
}

/// A baseline frame.
pub struct Frame {
    pub width:      u16,
    pub height:     u16,
    pub components: Vec<Component>,
    /// The transform byte of an Adobe APP14 segment, if the file has one.
    pub adobe:      Option<u8>
}

/// The coefficients of a flat block of `value`.
pub fn flat(value: u8) -> [i32; 64] {
    let mut block = [0; 64];
    block[0] = 8 * (i32::from(value) - 128);
    block
}

impl Frame {
    /// A frame whose components have the given identifiers and sampling factors, every block
    /// flat with value `fill`.
    pub fn new(width: u16, height: u16, layout: &[(u8, u8, u8)], fill: u8) -> Frame {
        let mut frame = Frame {
            width,
            height,
            components: Vec::new(),
            adobe: None
        };
        for &(id, h, v) in layout {
            frame.components.push(Component {
                id,
                h,
                v,
                blocks: Vec::new()
            });
        }
        for c in 0..layout.len() {
            let (wide, high) = frame.grid(c);
            frame.components[c].blocks = vec![flat(fill); wide * high];
        }
        frame
    }

    fn max_factors(&self) -> (usize, usize) {
        let h = self.components.iter().map(|c| usize::from(c.h)).max().unwrap();
        let v = self.components.iter().map(|c| usize::from(c.v)).max().unwrap();
        (h, v)
    }

    /// MCUs across and down.
    pub fn mcus(&self) -> (usize, usize) {
        let (h, v) = self.max_factors();
        (
            usize::from(self.width).div_ceil(8 * h),
            usize::from(self.height).div_ceil(8 * v)
        )
    }

    /// The block grid of component `c`: blocks across and down (whole MCUs).
    pub fn grid(&self, c: usize) -> (usize, usize) {
        let (x, y) = self.mcus();
        let comp = &self.components[c];
        (x * usize::from(comp.h), y * usize::from(comp.v))
    }

    /// Sets block (`bx`, `by`) of component `c`.
    pub fn set(&mut self, c: usize, bx: usize, by: usize, block: [i32; 64]) {
        let (wide, _) = self.grid(c);
        self.components[c].blocks[by * wide + bx] = block;
    }

    /// The whole file.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];

        if let Some(transform) = self.adobe {
            // Identifier, version 100, two flag words, the transform.
            let mut app14 = b"Adobe".to_vec();
            app14.extend([0, 100, 0, 0, 0, 0, transform]);
            segment(&mut out, 0xEE, &app14);
        }

        // DQT: table 0, 8-bit steps, all 1.
        let mut dqt = vec![0x00];
        dqt.extend([1; 64]);
        segment(&mut out, 0xDB, &dqt);

        // SOF0: 8-bit samples; each component uses table 0.
        let [h0, h1] = self.height.to_be_bytes();
        let [w0, w1] = self.width.to_be_bytes();
        let mut sof = vec![8, h0, h1, w0, w1, self.components.len() as u8];
        for c in &self.components {
            sof.extend([c.id, (c.h << 4) | c.v, 0]);
        }
        segment(&mut out, 0xC0, &sof);

        // DHT: DC table 0 with the categories 0-11, AC table 0 with every run/size symbol;
        // all codes eight bits long.
        let mut dc = vec![0x00];
        dc.extend(lengths(12));
        dc.extend(0..12u8);
        segment(&mut out, 0xC4, &dc);
        let symbols = ac_symbols();
        let mut ac = vec![0x10];
        ac.extend(lengths(symbols.len()));
        ac.extend(&symbols);
        segment(&mut out, 0xC4, &ac);

        // SOS: every component with tables 0/0, spectral selection 0-63.
        let mut sos = vec![self.components.len() as u8];
        for c in &self.components {
            sos.extend([c.id, 0x00]);
        }
        sos.extend([0, 63, 0]);
        segment(&mut out, 0xDA, &sos);

        for b in pack(&self.scan_bits(&symbols)) {
            out.push(b);
            if b == 0xFF {
                out.push(0x00);
            }
        }
        out.extend([0xFF, 0xD9]);
        out
    }

    fn scan_bits(&self, ac_symbols: &[u8]) -> Vec<bool> {
        let mut bits = Vec::new();
        let mut predictions = vec![0i32; self.components.len()];
        let (mcus_x, mcus_y) = self.mcus();
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                for (c, comp) in self.components.iter().enumerate() {
                    let (wide, _) = self.grid(c);
                    for v in 0..usize::from(comp.v) {
                        for h in 0..usize::from(comp.h) {
                            let bx = mx * usize::from(comp.h) + h;
                            let by = my * usize::from(comp.v) + v;
                            let block = &comp.blocks[by * wide + bx];
                            block_bits(&mut bits, block, &mut predictions[c], ac_symbols);
                        }
                    }
                }
            }
        }
        bits
    }
}

/// The 16 code-length counts of a table whose `count` codes are all eight bits long.
fn lengths(count: usize) -> [u8; 16] {
    let mut bits = [0; 16];
    bits[7] = count as u8;
    bits
}

/// Every AC symbol: end of block, sixteen zeros, and each run 0-15 with each size 1-10.
fn ac_symbols() -> Vec<u8> {
    let mut symbols = vec![0x00, 0xF0];
    for run in 0..16u8 {
        for size in 1..=10u8 {
            symbols.push(run << 4 | size);
        }
    }
    symbols.sort_unstable();
    symbols
}

/// The size category of `value` and its extra bits (a negative value is coded as value - 1).
fn magnitude(value: i32) -> (u32, u32) {
    let size = 32 - value.unsigned_abs().leading_zeros();
    let extra = if value < 0 { value - 1 } else { value };
    (size, extra as u32 & ((1u32 << size) - 1))
}

fn block_bits(bits: &mut Vec<bool>, block: &[i32; 64], prediction: &mut i32, ac_symbols: &[u8]) {
    // With codes all eight bits long in increasing symbol order, a symbol's code is its index.
    let code = |symbol: u8| ac_symbols.iter().position(|&s| s == symbol).unwrap() as u32;

    let diff = block[0] - *prediction;
    *prediction = block[0];
    let (size, extra) = magnitude(diff);
    push(bits, size, 8);
    push(bits, extra, size);

    let mut run = 0u8;
    for &natural in &ZIGZAG[1..] {
        let value = block[natural];
        if value == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            push(bits, code(0xF0), 8);
            run -= 16;
        }
        let (size, extra) = magnitude(value);
        push(bits, code(run << 4 | size as u8), 8);
        push(bits, extra, size);
        run = 0;
    }
    if run > 0 {
        push(bits, code(0x00), 8);
    }
}
