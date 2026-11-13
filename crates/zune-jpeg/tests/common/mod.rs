/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A tiny writer of grey baseline JPEG files for the tests: every 8×8 block is flat, so it has a
//! DC coefficient and nothing else.
//!
//! The quantization step of the DC coefficient is 8, so a block of value `v` has the quantized
//! DC coefficient `v - 128`. The DC Huffman table gives each difference category 0 to 11 a
//! four-bit code (the category number), and the AC table has one one-bit code `0`, for end of
//! block. A block therefore takes 4 bits, the category's extra bits, and 1 bit. The marker
//! layout follows ITU-T T.81 Annex B.

#![allow(dead_code)]

pub mod color;

/// A grey image of `width` × `height` samples with one value per block, blocks in row order.
pub struct Gray {
    pub width:   u16,
    pub height:  u16,
    pub blocks:  Vec<u8>,
    /// Blocks per restart interval, or 0 for none.
    pub restart: u16
}

impl Gray {
    pub fn new(width: u16, height: u16, blocks: &[u8]) -> Gray {
        let gray = Gray {
            width,
            height,
            blocks: blocks.to_vec(),
            restart: 0
        };
        assert_eq!(gray.blocks.len(), gray.blocks_wide() * gray.blocks_high());
        gray
    }

    pub fn blocks_wide(&self) -> usize {
        usize::from(self.width).div_ceil(8)
    }

    pub fn blocks_high(&self) -> usize {
        usize::from(self.height).div_ceil(8)
    }

    /// The bits of the entropy-coded data of `blocks`, one restart interval or the whole scan
    /// (the DC prediction starts at 0).
    pub fn scan_bits(&self, blocks: &[u8]) -> Vec<bool> {
        let mut bits = Vec::new();
        let mut prediction = 0i32;
        for &value in blocks {
            let dc = i32::from(value) - 128;
            let diff = dc - prediction;
            prediction = dc;
            let category = 32 - diff.unsigned_abs().leading_zeros();
            push(&mut bits, category, 4);
            // Negative differences are coded as diff - 1 in `category` bits.
            let extra = if diff < 0 { diff - 1 } else { diff };
            push(&mut bits, extra as u32 & ((1 << category) - 1), category);
            // End of block.
            bits.push(false);
        }
        bits
    }

    /// The whole file.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_cut(usize::MAX)
    }

    /// The file with its entropy-coded data cut after `keep` bytes (before byte stuffing), the
    /// EOI marker kept.
    pub fn encode_cut(&self, keep: usize) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];

        // DQT: table 0, 8-bit steps; 8 for the DC coefficient, 1 for the rest.
        let mut dqt = vec![0x00, 8];
        dqt.extend([1; 63]);
        segment(&mut out, 0xDB, &dqt);

        // SOF0: 8-bit samples, one component (id 1, sampling 1×1, table 0).
        let [h0, h1] = self.height.to_be_bytes();
        let [w0, w1] = self.width.to_be_bytes();
        segment(&mut out, 0xC0, &[8, h0, h1, w0, w1, 1, 1, 0x11, 0]);

        // DHT: DC table 0, twelve four-bit codes for the categories 0-11.
        let mut dc = vec![0x00, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        dc.extend(0..12u8);
        segment(&mut out, 0xC4, &dc);
        // DHT: AC table 0, a single one-bit code for end of block.
        segment(
            &mut out,
            0xC4,
            &[0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00]
        );

        if self.restart > 0 {
            let [r0, r1] = self.restart.to_be_bytes();
            segment(&mut out, 0xDD, &[r0, r1]);
        }

        // SOS: component 1 with tables 0/0, spectral selection 0-63, no approximation.
        segment(&mut out, 0xDA, &[1, 1, 0x00, 0, 63, 0]);

        let interval = match self.restart {
            0 => self.blocks.len(),
            n => usize::from(n)
        };
        let mut data = Vec::new();
        for (n, blocks) in self.blocks.chunks(interval).enumerate() {
            if n > 0 {
                data.push(Byte::Marker(0xD0 + (n as u8 - 1) % 8));
            }
            data.extend(pack(&self.scan_bits(blocks)).into_iter().map(Byte::Data));
        }
        let mut data_bytes = 0;
        for byte in data {
            match byte {
                Byte::Data(b) => {
                    if data_bytes == keep {
                        break;
                    }
                    data_bytes += 1;
                    out.push(b);
                    if b == 0xFF {
                        out.push(0x00);
                    }
                }
                Byte::Marker(m) => out.extend([0xFF, m])
            }
        }

        out.extend([0xFF, 0xD9]);
        out
    }

    /// The samples a decoder should give: each block's value.
    pub fn samples(&self) -> Vec<u8> {
        let wide = self.blocks_wide();
        let mut out = Vec::new();
        for y in 0..usize::from(self.height) {
            for x in 0..usize::from(self.width) {
                out.push(self.blocks[(y / 8) * wide + x / 8]);
            }
        }
        out
    }
}

enum Byte {
    Data(u8),
    Marker(u8)
}

fn push(bits: &mut Vec<bool>, value: u32, count: u32) {
    for i in (0..count).rev() {
        bits.push(value >> i & 1 == 1);
    }
}

/// Packs bits into bytes, the last byte filled with one bits.
fn pack(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0xFFu8;
            for (i, &bit) in chunk.iter().enumerate() {
                if !bit {
                    byte &= !(0x80 >> i);
                }
            }
            byte
        })
        .collect()
}

fn segment(out: &mut Vec<u8>, code: u8, body: &[u8]) {
    out.extend([0xFF, code]);
    out.extend(((body.len() + 2) as u16).to_be_bytes());
    out.extend(body);
}
