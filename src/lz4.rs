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
    BadLookback { disp: u16, avail: u16 },
    Truncated,
}

impl core::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self {
            DecompressError::BadLookback { disp, avail } => {
                write!(
                    f,
                    "Bad lookback (attempted -{} bytes out of {})",
                    disp, avail
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

const LOOKBACK_SZ: usize = 65535;

pub use lamezip77_macros::lz4_decompress_make as decompress_make;

pub async fn decompress_impl<O>(
    outp: &mut O,
    peek1: InputPeeker<'_, '_, 2, 1>,
    peek2: InputPeeker<'_, '_, 2, 2>,
) -> Result<(), DecompressError>
where
    O: LZOutputBuf,
{
    while !outp.is_at_limit() {
        let token = (&peek1).await[0];

        let mut nlits = (token >> 4) as usize;
        if nlits == 15 {
            loop {
                let b = (&peek1).await[0];
                nlits += b as usize;
                if b != 0xff {
                    break;
                }
            }
        }

        for _ in 0..nlits {
            outp.add_lits(&(&peek1).await);
        }

        if outp.is_at_limit() {
            // last block terminates here without offset
            break;
        }

        let offset = u16::from_le_bytes((&peek2).await);

        let mut matchlen = ((token & 0b1111) as usize) + 4;
        if matchlen == 19 {
            loop {
                let b = (&peek1).await[0];
                matchlen += b as usize;
                if b != 0xff {
                    break;
                }
            }
        }

        if offset == 0 {
            return Err(DecompressError::BadLookback {
                disp: offset as u16,
                avail: 0,
            });
        }

        outp.add_match(offset as usize - 1, matchlen).map_err(|_| {
            DecompressError::BadLookback {
                disp: offset as u16,
                avail: outp.cur_pos() as u16,
            }
        })?;
    }

    Ok(())
}

pub type Decompress<'a, F> = StreamingDecompressState<'a, F, DecompressError, 2>;
pub type DecompressBuffer<O> = StreamingOutputBuf<O, LOOKBACK_SZ>;

pub struct Compress<const MAX_LIT_BUF: usize> {
    engine: LZEngine<
        LOOKBACK_SZ,
        513,
        { LOOKBACK_SZ + 513 },
        4,
        { usize::MAX / 2 }, // XXX there are overflow issues
        16,
        { 1 << 16 },
        16,
        { 1 << 16 },
    >,
    // hard limit -- we FAIL if we encounter more than this number of incompressible literals
    buffered_lits: [u8; MAX_LIT_BUF],
    num_buffered_lits: usize,
}

impl<const MAX_LIT_BUF: usize> Compress<MAX_LIT_BUF> {
    pub fn new() -> Self {
        Self {
            engine: LZEngine::new(),
            buffered_lits: [0; MAX_LIT_BUF],
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
            for i in 0..MAX_LIT_BUF {
                (*p_buffered_lits)[i] = 0;
            }
            (*p).num_buffered_lits = 0;
            Box::from_raw(p)
        }
    }

    pub fn compress<O>(&mut self, inp: &[u8], end_of_stream: bool, mut outp: O) -> Result<(), ()>
    where
        O: FnMut(u8),
    {
        // XXX tweak?
        let good_enough_search_len = core::cmp::max(512, inp.len() as u64);

        let settings = LZSettings {
            good_enough_search_len,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: 1 << 16,
            defer_output_match: true,
            good_enough_defer_len: 525, // XXX tweak?
            search_faster_defer_len: good_enough_search_len / 2,
            min_disp: 1,
            // last 5 *must* be literals (always)
            // also, last match cannot be closer than 12 bytes from the end
            // just hold out 12 bytes. not optimal, but simple
            eos_holdout_bytes: 12,
        };

        macro_rules! dump_lits {
            ($match_len:expr) => {
                let match_len_lsb = if $match_len < 15 {
                    $match_len as u8
                } else {
                    15
                };
                let nlit_lsb = if self.num_buffered_lits < 15 {
                    self.num_buffered_lits as u8
                } else {
                    15
                };

                let token = nlit_lsb << 4 | match_len_lsb;
                outp(token);

                if self.num_buffered_lits >= 15 {
                    let mut nlit_left = self.num_buffered_lits - 15;
                    while nlit_left >= 0xFF {
                        outp(0xFF);
                        nlit_left -= 0xFF;
                    }
                    outp(nlit_left as u8);
                }

                if self.num_buffered_lits > 0 {
                    for i in 0..self.num_buffered_lits {
                        outp(self.buffered_lits[i]);
                    }
                    self.num_buffered_lits = 0;
                }
            };
        }

        self.engine.compress(&settings, inp, end_of_stream, |x| {
            match x {
                LZOutput::Lit(lit) => {
                    self.buffered_lits[self.num_buffered_lits as usize] = lit;
                    self.num_buffered_lits += 1;
                    if self.num_buffered_lits == MAX_LIT_BUF {
                        return Err(());
                    }
                }
                LZOutput::Ref { disp, len } => {
                    debug_assert!(len >= 4);
                    let len = len - 4;
                    dump_lits!(len);

                    debug_assert!(disp >= 1);
                    debug_assert!(disp <= 0xFFFF);
                    let disp = (disp as u16).to_le_bytes();
                    outp(disp[0]);
                    outp(disp[1]);

                    if len >= 15 {
                        let mut len_left = len - 15;
                        while len_left >= 0xFF {
                            outp(0xFF);
                            len_left -= 255;
                        }
                        outp(len_left as u8);
                    }
                }
            }
            Ok(())
        })?;

        if end_of_stream {
            dump_lits!(0);
        }

        Ok(())
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
    fn lz4_buffered_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/lz4.bin");
        let ref_fn = d.join("lz4test/tool.c");

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
    fn lz4_streaming_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/lz4.bin");
        let ref_fn = d.join("lz4test/tool.c");

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

        assert_eq!(out, ref_);
    }

    #[test]
    fn lz4_roundtrip_zerobytes() {
        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&[], true, |x| compressed_out.push(x))
            .unwrap();

        assert_eq!(compressed_out, [0x00]);

        let mut outp = crate::decompress::VecBuf::new(0, 0);
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
        assert_eq!(decompress_ourselves, []);
    }

    #[test]
    fn lz4_roundtrip_toosmall() {
        let inp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x))
            .unwrap();

        assert_eq!(compressed_out, [0xC0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
        assert_eq!(decompress_ourselves, inp);
    }

    #[test]
    fn lz4_roundtrip_toosmall2() {
        let inp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12];

        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x))
            .unwrap();

        assert_eq!(
            compressed_out,
            [0xD0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12]
        );

        let mut outp = crate::decompress::VecBuf::new(0, inp.len());
        {
            decompress_make!(dec, &mut outp, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress_ourselves: Vec<_> = outp.into();
        assert_eq!(decompress_ourselves, inp);
    }

    #[test]
    fn lz4_min_compressible() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let outp_fn = d.join("lz4-min.bin");
        let inp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x))
            .unwrap();

        assert_eq!(
            compressed_out,
            [0x10, 0, 1, 0, 0xC0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );

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
        assert_eq!(decompress_ourselves, inp);

        let ref_fn = d.join("lz4-min.out");
        Command::new("./lz4test/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn lz4_roundtrip_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/lz4.rs");
        let outp_fn = d.join("lz4-roundtrip-bytewise.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            comp.compress(&[inp[i]], false, |x| compressed_out.push(x))
                .unwrap();
        }
        comp.compress(&[], true, |x| compressed_out.push(x))
            .unwrap();

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

        let ref_fn = d.join("lz4-roundtrip-bytewise.out");
        Command::new("./lz4test/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn lz4_roundtrip_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/lz4.rs");
        let outp_fn = d.join("lz4-roundtrip-wholefile.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = Compress::<65536>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&inp, true, |x| compressed_out.push(x))
            .unwrap();

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

        let ref_fn = d.join("lz4-roundtrip-wholefile.out");
        Command::new("./lz4test/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }
}
