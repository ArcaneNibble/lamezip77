use bitvec::prelude::*;
use std::error::Error;

use crate::{util::*, LZEngine, LZOutput, LZSettings};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecompressState {
    Token,
    MoreLitLen,
    LiteralRun,
    Offset0,
    Offset1,
    MoreMatchLen,
}

impl DecompressState {
    const fn is_at_boundary(&self) -> bool {
        match self {
            DecompressState::Token | DecompressState::Offset0 => true,
            _ => false,
        }
    }
}

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

impl Error for DecompressError {}

const LOOKBACK_SZ: usize = 65535;

// technically copy-able, but don't want to make it easy to accidentally do so
#[derive(Clone)]
pub struct DecompressStreaming {
    lookback: [u8; LOOKBACK_SZ],
    lookback_wptr: usize,
    lookback_avail: usize,
    state: DecompressState,
    nlit: usize,
    offset: u16,
    matchlen: usize,
}

impl DecompressStreaming {
    pub fn new() -> Self {
        Self {
            lookback: [0; LOOKBACK_SZ],
            lookback_wptr: 0,
            lookback_avail: 0,
            state: DecompressState::Token,
            nlit: 0,
            offset: 0,
            matchlen: 0,
        }
    }
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = std::alloc::alloc(layout) as *mut Self;
            let p_lookback = core::ptr::addr_of_mut!((*p).lookback);
            for i in 0..LOOKBACK_SZ {
                // my understanding is that this is safe because u64 doesn't have Drop
                // and we don't construct a & anywhere
                (*p_lookback)[i] = 0;
            }
            (*p).lookback_wptr = 0;
            (*p).lookback_avail = 0;
            (*p).state = DecompressState::Token;
            (*p).nlit = 0;
            (*p).offset = 0;
            (*p).matchlen = 0;
            Box::from_raw(p)
        }
    }

    fn do_match<O>(&mut self, mut outp: O) -> Result<(), DecompressError>
    where
        O: FnMut(u8),
    {
        if self.offset == 0 {
            return Err(DecompressError::BadLookback {
                disp: self.offset,
                avail: 0,
            });
        }
        if self.offset as usize > self.lookback_avail {
            return Err(DecompressError::BadLookback {
                disp: self.offset,
                avail: self.lookback_avail as u16,
            });
        }

        for _ in 0..self.matchlen {
            let idx = (self.lookback_wptr + LOOKBACK_SZ - self.offset as usize) % LOOKBACK_SZ;
            let copy_b = self.lookback[idx];
            outp(copy_b);
            self.lookback[self.lookback_wptr] = copy_b;
            self.lookback_wptr = (self.lookback_wptr + 1) % LOOKBACK_SZ;
            if self.lookback_avail < LOOKBACK_SZ {
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
                DecompressState::Token => {
                    let token = b.view_bits::<Msb0>();
                    let nlit = token[..4].load::<usize>();
                    let matchlen = token[4..].load::<usize>();

                    self.nlit = nlit;
                    self.matchlen = matchlen + 4;

                    if nlit == 15 {
                        self.state = DecompressState::MoreLitLen;
                    } else if nlit != 0 {
                        self.state = DecompressState::LiteralRun;
                    } else {
                        self.state = DecompressState::Offset0;
                    }
                }
                DecompressState::MoreLitLen => {
                    self.nlit += b as usize;
                    if b != 0xFF {
                        self.state = DecompressState::LiteralRun;
                    }
                }
                DecompressState::LiteralRun => {
                    outp(b);
                    self.lookback[self.lookback_wptr] = b;
                    self.lookback_wptr = (self.lookback_wptr + 1) % LOOKBACK_SZ;
                    if self.lookback_avail < LOOKBACK_SZ {
                        self.lookback_avail += 1;
                    }

                    self.nlit -= 1;
                    if self.nlit == 0 {
                        self.state = DecompressState::Offset0;
                    }
                }
                DecompressState::Offset0 => {
                    self.offset = b as u16;
                    self.state = DecompressState::Offset1;
                }
                DecompressState::Offset1 => {
                    self.offset |= (b as u16) << 8;
                    if self.matchlen == 19 {
                        self.state = DecompressState::MoreMatchLen
                    } else {
                        self.do_match(&mut outp)?;
                        self.state = DecompressState::Token;
                    }
                }
                DecompressState::MoreMatchLen => {
                    self.matchlen += b as usize;
                    if b != 0xFF {
                        self.do_match(&mut outp)?;
                        self.state = DecompressState::Token;
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

    fn decompress<B>(&self, mut inp: &[u8], outp: &mut B) -> Result<(), DecompressError>
    where
        B: MaybeGrowableBuf,
    {
        while inp.len() > 0 {
            let token = get_inp::<1>(&mut inp).unwrap()[0];
            let token = token.view_bits::<Msb0>();

            let mut nlits = token[..4].load::<usize>();
            if nlits == 15 {
                loop {
                    let b = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                    nlits += b as usize;
                    if b != 0xff {
                        break;
                    }
                }
            }

            for _ in 0..nlits {
                // XXX efficiency?
                if outp.cur_pos() < outp.limit() {
                    outp.add_lit(
                        get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0],
                    );
                }
            }

            if inp.len() == 0 {
                // last block terminates here without offset
                break;
            }

            let offset =
                u16::from_le_bytes(get_inp::<2>(&mut inp).map_err(|_| DecompressError::Truncated)?);

            let mut matchlen = token[4..].load::<usize>() + 4;
            if matchlen == 19 {
                loop {
                    let b = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
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

            outp.add_match(offset as usize, matchlen).map_err(|_| {
                DecompressError::BadLookback {
                    disp: offset as u16,
                    avail: outp.cur_pos() as u16,
                }
            })?;
        }
        Ok(())
    }

    pub fn decompress_into(&self, inp: &[u8], outp: &mut [u8]) -> Result<(), DecompressError> {
        self.decompress(inp, &mut FixedBuf::from(outp))
    }

    pub fn decompress_new(&self, inp: &[u8], max_sz: usize) -> Result<Vec<u8>, DecompressError> {
        let mut buf = VecBuf::new(0, max_sz);
        self.decompress(inp, &mut buf)?;
        Ok(buf.into())
    }
}

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
    // we unfortunately will expand incompressible data more than the reference code
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
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = std::alloc::alloc(layout) as *mut Self;
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
                let match_len_lsb = if $match_len < 15 { $match_len } else { 15 };
                let nlit_lsb = if self.num_buffered_lits < 15 {
                    self.num_buffered_lits as u8
                } else {
                    15
                };

                let mut token = 0u8;
                let token_v = token.view_bits_mut::<Msb0>();
                token_v[..4].store(nlit_lsb);
                token_v[4..].store(match_len_lsb);
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

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufWriter, Write},
        process::Command,
    };

    use super::*;

    #[test]
    fn lz4_buffered_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/lz4.bin");
        let ref_fn = d.join("lz4test/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn lz4_streaming_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/lz4.bin");
        let ref_fn = d.join("lz4test/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressStreaming::new_boxed();
        let mut out = Vec::new();

        for b in inp {
            dec.decompress(&[b], |x| out.push(x)).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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

        let dec = DecompressBuffered::new();
        let decompress_ourselves = dec.decompress_new(&compressed_out, usize::MAX).unwrap();
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
