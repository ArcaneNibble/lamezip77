use bitvec::prelude::*;
#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box, vec::Vec};

use crate::{util::*, LZEngine, LZOutput, LZSettings};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecompressState {
    Opcode0Start,
    Opcode0Lv1,
    OpcodeLv1MoreLen,
    OpcodeLv1MoreDisp,
    LiteralRunLv1,
    Opcode0Lv2,
    OpcodeLv2MoreLen,
    OpcodeLv2MoreDisp0,
    OpcodeLv2MoreDisp1,
    OpcodeLv2MoreDisp2,
    LiteralRunLv2,
}

impl DecompressState {
    const fn is_at_boundary(&self) -> bool {
        match self {
            DecompressState::Opcode0Start
            | DecompressState::Opcode0Lv1
            | DecompressState::Opcode0Lv2 => true,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecompressError {
    BadCompressionLevel(u8),
    BadLookback { disp: u32, avail: u32 },
    Truncated,
}

impl core::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self {
            DecompressError::BadCompressionLevel(actual) => {
                write!(f, "Bad compression level {}", actual)
            }
            DecompressError::BadLookback { disp, avail } => {
                write!(
                    f,
                    "Bad lookback (attempted -{} bytes out of {})",
                    disp + 1,
                    avail
                )
            }
            DecompressError::Truncated => {
                write!(f, "Input is truncated")
            }
        }
    }
}

#[cfg(feature = "std")]
impl Error for DecompressError {}

const LV1_LOOKBACK_SZ: usize = 0x1FFF + 1;
const LV2_LOOKBACK_SZ: usize = 0xFFFF + 0x1FFF + 1;

// technically copy-able, but don't want to make it easy to accidentally do so
#[derive(Clone)]
pub struct DecompressStreaming {
    lookback: [u8; LV2_LOOKBACK_SZ],
    lookback_wptr: usize,
    lookback_avail: usize,
    state: DecompressState,
    nlit: u8,
    matchlen: usize,
    matchdisp: usize,
}

impl DecompressStreaming {
    pub fn new() -> Self {
        Self {
            lookback: [0; LV2_LOOKBACK_SZ],
            lookback_wptr: 0,
            lookback_avail: 0,
            state: DecompressState::Opcode0Start,
            nlit: 0,
            matchlen: 0,
            matchdisp: 0,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            let p_lookback = core::ptr::addr_of_mut!((*p).lookback);
            for i in 0..LV2_LOOKBACK_SZ {
                // my understanding is that this is safe because u64 doesn't have Drop
                // and we don't construct a & anywhere
                (*p_lookback)[i] = 0;
            }
            (*p).lookback_wptr = 0;
            (*p).lookback_avail = 0;
            (*p).state = DecompressState::Opcode0Start;
            (*p).nlit = 0;
            (*p).matchlen = 0;
            (*p).matchdisp = 0;
            Box::from_raw(p)
        }
    }

    fn do_match<O>(&mut self, mut outp: O) -> Result<(), DecompressError>
    where
        O: FnMut(u8),
    {
        if self.matchdisp + 1 > self.lookback_avail {
            return Err(DecompressError::BadLookback {
                disp: self.matchdisp as u32,
                avail: self.lookback_avail as u32,
            });
        }

        for _ in 0..self.matchlen {
            let idx = (self.lookback_wptr + LV2_LOOKBACK_SZ - self.matchdisp - 1) % LV2_LOOKBACK_SZ;
            let copy_b = self.lookback[idx];
            outp(copy_b);
            self.lookback[self.lookback_wptr] = copy_b;
            self.lookback_wptr = (self.lookback_wptr + 1) % LV2_LOOKBACK_SZ;
            if self.lookback_avail < LV2_LOOKBACK_SZ {
                self.lookback_avail += 1;
            }
        }

        Ok(())
    }

    pub fn decompress<O>(&mut self, mut inp: &[u8], mut outp: O) -> Result<bool, DecompressError>
    where
        O: FnMut(u8),
    {
        while inp.len() > 0 {
            let b = get_inp::<1>(&mut inp).unwrap()[0];
            match self.state {
                DecompressState::Opcode0Start => {
                    let b = b.view_bits::<Msb0>();
                    let level = b[..3].load::<u8>();
                    let nlit = b[3..].load::<u8>() + 1;

                    self.nlit = nlit;
                    if level == 0 {
                        self.state = DecompressState::LiteralRunLv1;
                    } else if level == 1 {
                        self.state = DecompressState::LiteralRunLv2;
                    } else {
                        return Err(DecompressError::BadCompressionLevel(level));
                    }
                }
                DecompressState::Opcode0Lv1 => {
                    let b = b.view_bits::<Msb0>();
                    let matchlen = b[..3].load::<usize>() + 2;
                    let matchdisp = b[3..].load::<usize>() << 8;

                    self.matchlen = matchlen;
                    self.matchdisp = matchdisp;

                    if matchlen == 0b000 + 2 {
                        self.nlit = b[3..].load::<u8>() + 1;
                        self.state = DecompressState::LiteralRunLv1;
                    } else if matchlen == 0b111 + 2 {
                        self.state = DecompressState::OpcodeLv1MoreLen;
                    } else {
                        self.state = DecompressState::OpcodeLv1MoreDisp;
                    }
                }
                DecompressState::OpcodeLv1MoreLen => {
                    self.matchlen += b as usize;
                    self.state = DecompressState::OpcodeLv1MoreDisp;
                }
                DecompressState::OpcodeLv1MoreDisp => {
                    self.matchdisp |= b as usize;
                    self.do_match(&mut outp)?;
                    self.state = DecompressState::Opcode0Lv1;
                }
                DecompressState::Opcode0Lv2 => {
                    let b = b.view_bits::<Msb0>();
                    let matchlen = b[..3].load::<usize>() + 2;
                    let matchdisp = b[3..].load::<usize>() << 8;

                    self.matchlen = matchlen;
                    self.matchdisp = matchdisp;

                    if matchlen == 0b000 + 2 {
                        self.nlit = b[3..].load::<u8>() + 1;
                        self.state = DecompressState::LiteralRunLv2;
                    } else if matchlen == 0b111 + 2 {
                        self.state = DecompressState::OpcodeLv2MoreLen;
                    } else {
                        self.state = DecompressState::OpcodeLv2MoreDisp0;
                    }
                }
                DecompressState::OpcodeLv2MoreLen => {
                    self.matchlen += b as usize;

                    if b != 0xff {
                        self.state = DecompressState::OpcodeLv2MoreDisp0;
                    }
                }
                DecompressState::OpcodeLv2MoreDisp0 => {
                    self.matchdisp |= b as usize;

                    if self.matchdisp == 0b11111_11111111 {
                        self.state = DecompressState::OpcodeLv2MoreDisp1;
                    } else {
                        self.do_match(&mut outp)?;
                        self.state = DecompressState::Opcode0Lv2;
                    }
                }
                DecompressState::OpcodeLv2MoreDisp1 => {
                    self.matchdisp += (b as usize) << 8;
                    self.state = DecompressState::OpcodeLv2MoreDisp2;
                }
                DecompressState::OpcodeLv2MoreDisp2 => {
                    self.matchdisp += b as usize;
                    self.do_match(&mut outp)?;
                    self.state = DecompressState::Opcode0Lv2;
                }
                DecompressState::LiteralRunLv1 | DecompressState::LiteralRunLv2 => {
                    outp(b);
                    self.lookback[self.lookback_wptr] = b;
                    self.lookback_wptr = (self.lookback_wptr + 1) % LV2_LOOKBACK_SZ;
                    if self.lookback_avail < LV2_LOOKBACK_SZ {
                        self.lookback_avail += 1;
                    }

                    self.nlit -= 1;
                    if self.nlit == 0 {
                        match self.state {
                            DecompressState::LiteralRunLv1 => {
                                self.state = DecompressState::Opcode0Lv1;
                            }
                            DecompressState::LiteralRunLv2 => {
                                self.state = DecompressState::Opcode0Lv2;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }

        Ok(self.state.is_at_boundary())
    }
}

#[derive(Copy, Clone)]
pub struct DecompressBuffered {}

impl DecompressBuffered {
    pub fn new() -> Self {
        Self {}
    }

    fn decompress_lv1<B>(&self, mut inp: &[u8], outp: &mut B) -> Result<(), DecompressError>
    where
        B: MaybeGrowableBuf,
    {
        let mut first_opc = true;

        while inp.len() > 0 {
            let opc0 = get_inp::<1>(&mut inp).unwrap()[0];
            let opc0 = opc0.view_bits::<Msb0>();
            let opc_len = opc0[..3].load::<u8>();

            if first_opc || opc_len == 0 {
                first_opc = false;
                // literal run
                let nlit = opc0[3..].load::<u8>() + 1;
                for _ in 0..nlit {
                    // XXX efficiency?
                    if outp.cur_pos() < outp.limit() {
                        outp.add_lit(
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0],
                        );
                    }
                }
            } else {
                let matchlen = if opc_len != 0b111 {
                    (opc_len + 2) as usize
                } else {
                    let opc1 = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                    (opc1 as usize) + 9
                };

                let opc12 = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                let matchdisp = (opc0[3..].load::<usize>() << 8) | (opc12 as usize);

                outp.add_match(matchdisp + 1, matchlen).map_err(|_| {
                    DecompressError::BadLookback {
                        disp: matchdisp as u32,
                        avail: outp.cur_pos() as u32,
                    }
                })?;
            }
        }

        Ok(())
    }

    fn decompress_lv2<B>(&self, mut inp: &[u8], outp: &mut B) -> Result<(), DecompressError>
    where
        B: MaybeGrowableBuf,
    {
        let mut first_opc = true;

        while inp.len() > 0 {
            let opc0 = get_inp::<1>(&mut inp).unwrap()[0];
            let opc0 = opc0.view_bits::<Msb0>();
            let opc_len = opc0[..3].load::<u8>();

            if first_opc || opc_len == 0 {
                first_opc = false;
                // literal run
                let nlit = opc0[3..].load::<u8>() + 1;
                for _ in 0..nlit {
                    // XXX efficiency?
                    if outp.cur_pos() < outp.limit() {
                        outp.add_lit(
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0],
                        );
                    }
                }
            } else {
                let mut matchlen = (opc_len + 2) as usize;
                if matchlen == 0b111 + 2 {
                    loop {
                        let morelen =
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                        matchlen += morelen as usize;
                        if morelen != 0xff {
                            break;
                        }
                    }
                }

                let opc_dispnext =
                    get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                let mut matchdisp = (opc0[3..].load::<usize>() << 8) | (opc_dispnext as usize);
                if matchdisp == 0b11111_11111111 {
                    let moredisp =
                        get_inp::<2>(&mut inp).map_err(|_| DecompressError::Truncated)?;
                    let moredisp = ((moredisp[0] as usize) << 8) | (moredisp[1] as usize);
                    matchdisp += moredisp;
                }

                outp.add_match(matchdisp + 1, matchlen).map_err(|_| {
                    DecompressError::BadLookback {
                        disp: matchdisp as u32,
                        avail: outp.cur_pos() as u32,
                    }
                })?;
            }
        }
        Ok(())
    }

    fn decompress<B>(&self, inp: &[u8], outp: &mut B) -> Result<(), DecompressError>
    where
        B: MaybeGrowableBuf,
    {
        if inp.len() == 0 {
            return Err(DecompressError::Truncated);
        }
        let level = inp[0] >> 5;
        if level == 0 {
            self.decompress_lv1(inp, outp)
        } else if level == 1 {
            self.decompress_lv2(inp, outp)
        } else {
            return Err(DecompressError::BadCompressionLevel(level));
        }
    }

    pub fn decompress_into(&self, inp: &[u8], outp: &mut [u8]) -> Result<(), DecompressError> {
        self.decompress(inp, &mut FixedBuf::from(outp))
    }

    #[cfg(feature = "alloc")]
    pub fn decompress_new(&self, inp: &[u8], max_sz: usize) -> Result<Vec<u8>, DecompressError> {
        let mut buf = VecBuf::new(0, max_sz);
        self.decompress(inp, &mut buf)?;
        Ok(buf.into())
    }
}

pub struct CompressLevel1 {
    engine: LZEngine<
        LV1_LOOKBACK_SZ,
        264,
        { LV1_LOOKBACK_SZ + 264 },
        3,
        264,
        13,
        { 1 << 13 },
        13,
        { 1 << 13 },
    >,
    buffered_lits: [u8; 32],
    num_buffered_lits: u8,
}

impl CompressLevel1 {
    pub fn new() -> Self {
        Self {
            engine: LZEngine::new(),
            buffered_lits: [0; 32],
            num_buffered_lits: 0,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            LZEngine::initialize_at(core::ptr::addr_of_mut!((*p).engine));
            let p_buffered_lits = core::ptr::addr_of_mut!((*p).buffered_lits);
            for i in 0..32 {
                (*p_buffered_lits)[i] = 0;
            }
            (*p).num_buffered_lits = 0;
            Box::from_raw(p)
        }
    }

    pub fn compress<O>(&mut self, inp: &[u8], end_of_stream: bool, mut outp: O)
    where
        O: FnMut(u8),
    {
        let settings = LZSettings {
            good_enough_search_len: 264,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: 1 << 13,
            defer_output_match: true,
            good_enough_defer_len: 64,
            search_faster_defer_len: 128,
            min_disp: 1,
            eos_holdout_bytes: 0,
        };

        macro_rules! dump_lits {
            () => {
                if self.num_buffered_lits > 0 {
                    debug_assert!(self.num_buffered_lits <= 32);
                    outp(self.num_buffered_lits - 1);
                    for i in 0..self.num_buffered_lits {
                        outp(self.buffered_lits[i as usize]);
                    }
                    self.num_buffered_lits = 0;
                }
            };
        }

        self.engine
            .compress::<_, ()>(&settings, inp, end_of_stream, |x| {
                match x {
                    LZOutput::Lit(lit) => {
                        self.buffered_lits[self.num_buffered_lits as usize] = lit;
                        self.num_buffered_lits += 1;
                        if self.num_buffered_lits == 32 {
                            dump_lits!();
                        }
                    }
                    LZOutput::Ref { disp, len } => {
                        dump_lits!();

                        let disp = disp - 1;
                        debug_assert!(disp <= 0x1FFF);
                        debug_assert!(len >= 3);

                        let len_opc0 = if len >= 0b111 + 2 {
                            0b111
                        } else {
                            (len - 2) as u8
                        };

                        let mut opc0 = 0u8;
                        let opc0_v = opc0.view_bits_mut::<Msb0>();
                        opc0_v[..3].store(len_opc0);
                        opc0_v[3..].store(disp >> 8);
                        outp(opc0);

                        if len >= 0b111 + 2 {
                            let len_opc1 = len - (0b111 + 2);
                            debug_assert!(len_opc1 <= 0xff);
                            outp(len_opc1 as u8);
                        }

                        outp((disp & 0xff) as u8);
                    }
                }
                Ok(())
            })
            .unwrap();

        if end_of_stream {
            dump_lits!();
        }
    }
}

pub struct CompressLevel2 {
    engine: LZEngine<
        LV2_LOOKBACK_SZ,
        512,
        { LV2_LOOKBACK_SZ + 512 },
        3,
        { usize::MAX / 2 }, // XXX there are overflow issues
        17,
        { 1 << 17 },
        17,
        { 1 << 17 },
    >,
    buffered_lits: [u8; 32],
    num_buffered_lits: u8,
    tag_emitted: bool,
}

impl CompressLevel2 {
    pub fn new() -> Self {
        Self {
            engine: LZEngine::new(),
            buffered_lits: [0; 32],
            num_buffered_lits: 0,
            tag_emitted: false,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            LZEngine::initialize_at(core::ptr::addr_of_mut!((*p).engine));
            let p_buffered_lits = core::ptr::addr_of_mut!((*p).buffered_lits);
            for i in 0..32 {
                (*p_buffered_lits)[i] = 0;
            }
            (*p).num_buffered_lits = 0;
            (*p).tag_emitted = false;
            Box::from_raw(p)
        }
    }

    pub fn compress<O>(&mut self, inp: &[u8], end_of_stream: bool, mut outp: O)
    where
        O: FnMut(u8),
    {
        // XXX tweak?
        let good_enough_search_len = core::cmp::max(512, inp.len() as u64);

        let settings = LZSettings {
            good_enough_search_len,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: 1 << 17,
            defer_output_match: true,
            good_enough_defer_len: 8190, // XXX tweak?
            search_faster_defer_len: good_enough_search_len / 2,
            min_disp: 1,
            // for some reason, long matches cannot be any closer than -2 bytes from the end
            // this guarantees that condition is met
            eos_holdout_bytes: 1,
        };

        macro_rules! dump_lits {
            () => {
                if self.num_buffered_lits > 0 {
                    debug_assert!(self.num_buffered_lits <= 32);
                    let tag = if !self.tag_emitted {
                        self.tag_emitted = true;
                        0b001_00000
                    } else {
                        0
                    };
                    outp(tag | (self.num_buffered_lits - 1));
                    for i in 0..self.num_buffered_lits {
                        outp(self.buffered_lits[i as usize]);
                    }
                    self.num_buffered_lits = 0;
                }
            };
        }

        self.engine
            .compress::<_, ()>(&settings, inp, end_of_stream, |x| {
                match x {
                    LZOutput::Lit(lit) => {
                        self.buffered_lits[self.num_buffered_lits as usize] = lit;
                        self.num_buffered_lits += 1;
                        if self.num_buffered_lits == 32 {
                            dump_lits!();
                        }
                    }
                    LZOutput::Ref { disp, len } => {
                        dump_lits!();

                        let disp = disp - 1;
                        debug_assert!(disp <= 0x1FFF + 0xFFFF);
                        debug_assert!(len >= 3);

                        let disp_non_extra = core::cmp::min(0x1FFF, disp);
                        let len_opc0 = if len >= 0b111 + 2 {
                            0b111
                        } else {
                            (len - 2) as u8
                        };

                        let mut opc0 = 0u8;
                        let opc0_v = opc0.view_bits_mut::<Msb0>();
                        opc0_v[..3].store(len_opc0);
                        opc0_v[3..].store(disp_non_extra >> 8);
                        outp(opc0);

                        // match len
                        if len >= 0b111 + 2 {
                            let mut remaining_len = len - (0b111 + 2);
                            while remaining_len >= 0xFF {
                                outp(0xFF);
                                remaining_len -= 0xFF;
                            }
                            outp(remaining_len as u8);
                        }

                        // displacement
                        outp((disp_non_extra & 0xff) as u8);

                        // extra long displacement
                        if disp >= 0x1FFF {
                            let disp_extra = disp - disp_non_extra;
                            debug_assert!(disp_extra <= 0xFFFF);
                            outp(((disp_extra >> 8) & 0xff) as u8);
                            outp((disp_extra & 0xff) as u8);
                        }
                    }
                }
                Ok(())
            })
            .unwrap();

        if end_of_stream {
            dump_lits!();
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    extern crate std;
    use std::{
        fs::File,
        io::{BufWriter, Write},
        process::Command,
        vec,
        vec::Vec,
    };

    use super::*;

    #[test]
    fn flz_buffered_ref_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv1.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn flz_buffered_zeros_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv1.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_buffered_ref_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv2.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn flz_buffered_zeros_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_streaming_ref_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv1.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
        }

        let mut outp_f = BufWriter::new(File::create(d.join("dump.bin")).unwrap());
        outp_f.write(&out).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn flz_streaming_zeros_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv1.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
        }

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_streaming_ref_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv2.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
        }

        let mut outp_f = BufWriter::new(File::create(d.join("dump.bin")).unwrap());
        outp_f.write(&out).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn flz_streaming_zeros_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
        }

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_streaming_long_disp_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz2-longdisp.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();
        let mut ref_ = vec![0; 16384];
        ref_[0] = b'a';
        ref_[1] = b'b';
        ref_[2] = b'c';
        ref_[3] = b'd';
        ref_[4] = b'e';
        ref_[5] = b'f';
        ref_[6] = b'g';
        ref_[0x3ff0 + 0] = b'a';
        ref_[0x3ff0 + 1] = b'b';
        ref_[0x3ff0 + 2] = b'c';
        ref_[0x3ff0 + 3] = b'd';
        ref_[0x3ff0 + 4] = b'e';
        ref_[0x3ff0 + 5] = b'f';
        ref_[0x3ff0 + 6] = b'g';

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
        }

        assert_eq!(out, ref_);
    }

    #[test]
    fn flz_lv1_roundtrip_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/fastlz.rs");
        let outp_fn = d.join("fastlz-lv1-roundtrip-bytewise.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = CompressLevel1::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            comp.compress(&[inp[i]], false, |x| compressed_out.push(x));
        }
        comp.compress(&[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv1-roundtrip-bytewise.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv1_roundtrip_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/fastlz.rs");
        let outp_fn = d.join("fastlz-lv1-roundtrip-wholefile.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = CompressLevel1::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv1-roundtrip-wholefile.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/fastlz.rs");
        let outp_fn = d.join("fastlz-lv2-roundtrip-bytewise.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            comp.compress(&[inp[i]], false, |x| compressed_out.push(x));
        }
        comp.compress(&[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-bytewise.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/fastlz.rs");
        let outp_fn = d.join("fastlz-lv2-roundtrip-wholefile.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-wholefile.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_longlen_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let outp_fn = d.join("fastlz-lv2-roundtrip-longlen-bytewise.bin");

        let inp = vec![0; 1024 * 32];

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            comp.compress(&[inp[i]], false, |x| compressed_out.push(x));
        }
        comp.compress(&[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-longlen-bytewise.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_longlen_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let outp_fn = d.join("fastlz-lv2-roundtrip-longlen-wholefile.bin");

        let inp = vec![0; 1024 * 32];

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-longlen-wholefile.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_longdisp_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let outp_fn = d.join("fastlz-lv2-roundtrip-longdisp-bytewise.bin");
        let mut inp = vec![0; 16384];
        inp[0] = b'a';
        inp[1] = b'b';
        inp[2] = b'c';
        inp[3] = b'd';
        inp[4] = b'e';
        inp[5] = b'f';
        inp[6] = b'g';
        inp[0x3ff0 + 0] = b'a';
        inp[0x3ff0 + 1] = b'b';
        inp[0x3ff0 + 2] = b'c';
        inp[0x3ff0 + 3] = b'd';
        inp[0x3ff0 + 4] = b'e';
        inp[0x3ff0 + 5] = b'f';
        inp[0x3ff0 + 6] = b'g';

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            comp.compress(&[inp[i]], false, |x| compressed_out.push(x));
        }
        comp.compress(&[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-longdisp-bytewise.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn flz_lv2_roundtrip_longdisp_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let outp_fn = d.join("fastlz-lv2-roundtrip-longdisp-wholefile.bin");
        let mut inp = vec![0; 16384];
        inp[0] = b'a';
        inp[1] = b'b';
        inp[2] = b'c';
        inp[3] = b'd';
        inp[4] = b'e';
        inp[5] = b'f';
        inp[6] = b'g';
        inp[0x3ff0 + 0] = b'a';
        inp[0x3ff0 + 1] = b'b';
        inp[0x3ff0 + 2] = b'c';
        inp[0x3ff0 + 3] = b'd';
        inp[0x3ff0 + 4] = b'e';
        inp[0x3ff0 + 5] = b'f';
        inp[0x3ff0 + 6] = b'g';

        let mut comp = CompressLevel2::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();
        drop(outp_f);

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
        assert_eq!(inp, decompress_ourselves);

        let ref_fn = d.join("fastlz-lv2-roundtrip-longdisp-wholefile.out");
        Command::new("./fastlztest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }
}
