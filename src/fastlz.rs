use bitvec::prelude::*;
#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box};

use crate::{
    decompress::{InputPeeker, LZOutputBuf, StreamingDecompressState, StreamingOutputBuf},
    LZEngine, LZOutput, LZSettings,
};

#[derive(Debug, PartialEq, Eq)]
pub enum DecompressError {
    BadCompressionLevel(u8),
    BadLookback { disp: u32, avail: u32 },
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
        }
    }
}

#[cfg(feature = "std")]
impl Error for DecompressError {}

const LV1_LOOKBACK_SZ: usize = 0x1FFF + 1;
const LV2_LOOKBACK_SZ: usize = 0xFFFF + 0x1FFF + 1;

pub use lamezip77_macros::fastlz_decompress_make as decompress_make;

pub async fn decompress_impl<O>(
    outp: &mut O,
    peek1: InputPeeker<'_, '_, 2, 1>,
    peek2: InputPeeker<'_, '_, 2, 2>,
) -> Result<(), DecompressError>
where
    O: LZOutputBuf,
{
    let mut is_first_opc = true;
    let first_opc = (&peek1).await[0];

    let level = first_opc >> 5;
    if level == 0 {
        while !outp.is_at_limit() {
            let opc0 = if is_first_opc {
                first_opc
            } else {
                (&peek1).await[0]
            };
            let opc0 = opc0.view_bits::<Msb0>();
            let opc_len = opc0[..3].load::<u8>();

            if is_first_opc || opc_len == 0 {
                is_first_opc = false;
                // literal run
                let nlit = opc0[3..].load::<u8>() + 1;
                for _ in 0..nlit {
                    outp.add_lits(&(&peek1).await);
                }
            } else {
                let matchlen = if opc_len != 0b111 {
                    (opc_len + 2) as usize
                } else {
                    let opc1 = (&peek1).await[0];
                    (opc1 as usize) + 9
                };

                let opc12 = (&peek1).await[0];
                let matchdisp = (opc0[3..].load::<usize>() << 8) | (opc12 as usize);

                outp.add_match(matchdisp, matchlen)
                    .map_err(|_| DecompressError::BadLookback {
                        disp: matchdisp as u32,
                        avail: outp.cur_pos() as u32,
                    })?;
            }
        }
    } else if level == 1 {
        while !outp.is_at_limit() {
            let opc0 = if is_first_opc {
                first_opc
            } else {
                (&peek1).await[0]
            };
            let opc0 = opc0.view_bits::<Msb0>();
            let opc_len = opc0[..3].load::<u8>();

            if is_first_opc || opc_len == 0 {
                is_first_opc = false;
                // literal run
                let nlit = opc0[3..].load::<u8>() + 1;
                for _ in 0..nlit {
                    outp.add_lits(&(&peek1).await);
                }
            } else {
                let mut matchlen = (opc_len + 2) as usize;
                if matchlen == 0b111 + 2 {
                    loop {
                        let morelen = (&peek1).await[0];
                        matchlen += morelen as usize;
                        if morelen != 0xff {
                            break;
                        }
                    }
                }

                let opc_dispnext = (&peek1).await[0];
                let mut matchdisp = (opc0[3..].load::<usize>() << 8) | (opc_dispnext as usize);
                if matchdisp == 0b11111_11111111 {
                    let moredisp = (&peek2).await;
                    let moredisp = ((moredisp[0] as usize) << 8) | (moredisp[1] as usize);
                    matchdisp += moredisp;
                }

                outp.add_match(matchdisp, matchlen)
                    .map_err(|_| DecompressError::BadLookback {
                        disp: matchdisp as u32,
                        avail: outp.cur_pos() as u32,
                    })?;
            }
        }
    } else {
        return Err(DecompressError::BadCompressionLevel(level));
    }

    Ok(())
}

pub type Decompress<'a, F> = StreamingDecompressState<'a, F, DecompressError, 2>;
pub type DecompressBuffer<O> = StreamingOutputBuf<O, LV2_LOOKBACK_SZ>;

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

        let mut outp = crate::decompress::VecBuf::new(0, ref_.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&inp);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, ref_);
    }

    #[test]
    fn flz_buffered_zeros_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv1.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut outp = crate::decompress::VecBuf::new(0, 256 * 1024);
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&inp);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_buffered_ref_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv2.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut outp = crate::decompress::VecBuf::new(0, ref_.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&inp);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, ref_);
    }

    #[test]
    fn flz_buffered_zeros_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut outp = crate::decompress::VecBuf::new(0, 256 * 1024);
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&inp);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_streaming_ref_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz.lv1.bin");
        let ref_fn = d.join("fastlztest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut out = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| out.extend_from_slice(x), ref_.len());
            decompress_make!(dec, &mut outp, crate);

            let mut ret = Ok(usize::MAX);
            for b in inp {
                ret = dec.add_inp(&[b]);
                assert!(ret.is_ok());
            }
            assert!(ret.unwrap() == 0);
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

        let mut out = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| out.extend_from_slice(x), 256 * 1024);
            decompress_make!(dec, &mut outp, crate);

            let mut ret = Ok(usize::MAX);
            for b in inp {
                ret = dec.add_inp(&[b]);
                assert!(ret.is_ok());
            }
            assert!(ret.unwrap() == 0);
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

        let mut out = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| out.extend_from_slice(x), ref_.len());
            decompress_make!(dec, &mut outp, crate);

            let mut ret = Ok(usize::MAX);
            for b in inp {
                ret = dec.add_inp(&[b]);
                assert!(ret.is_ok());
            }
            assert!(ret.unwrap() == 0);
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

        let mut out = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| out.extend_from_slice(x), 256 * 1024);
            decompress_make!(dec, &mut outp, crate);

            let mut ret = Ok(usize::MAX);
            for b in inp {
                ret = dec.add_inp(&[b]);
                assert!(ret.is_ok());
            }
            assert!(ret.unwrap() == 0);
        }

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_streaming_long_disp_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz2-longdisp.bin");

        let inp = std::fs::read(inp_fn).unwrap();

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

        let mut out = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| out.extend_from_slice(x), ref_.len());
            decompress_make!(dec, &mut outp, crate);

            let mut ret = Ok(usize::MAX);
            for b in inp {
                ret = dec.add_inp(&[b]);
                assert!(ret.is_ok());
            }
            assert!(ret.unwrap() == 0);
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
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
