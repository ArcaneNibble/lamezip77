use bitvec::prelude::*;
#[cfg(feature = "std")]
extern crate std;
use core::future::Future;
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box, vec::Vec};

use crate::util::*;
use crate::{
    decompress::{InputPeeker, LZOutputBuf, StreamingDecompressState},
    LZEngine, LZOutput, LZSettings,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecompressState {
    HdrMagic,
    HdrLen0,
    HdrLen1,
    HdrLen2,
    BlockFlags,
    BlockData0,
    BlockData1,
}

impl DecompressState {
    const fn is_header_state(&self) -> bool {
        match self {
            DecompressState::HdrMagic
            | DecompressState::HdrLen0
            | DecompressState::HdrLen1
            | DecompressState::HdrLen2 => true,
            DecompressState::BlockFlags
            | DecompressState::BlockData0
            | DecompressState::BlockData1 => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecompressError {
    BadMagic(u8),
    BadLookback { disp: u16, avail: u16 },
    TooShort,
}

impl core::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self {
            DecompressError::BadMagic(actual) => {
                write!(f, "Bad header magic byte {:02X}", actual)
            }
            DecompressError::BadLookback { disp, avail } => {
                write!(
                    f,
                    "Bad lookback (attempted -{} bytes out of {})",
                    disp + 1,
                    avail
                )
            }
            DecompressError::TooShort => {
                write!(f, "Too little data available")
            }
        }
    }
}

#[cfg(feature = "std")]
impl Error for DecompressError {}

const LOOKBACK_SZ: usize = 0x1000;

pub struct Decompress<'a, F>
where
    F: Future<Output = Result<(), DecompressError>>,
{
    pub state: StreamingDecompressState<'a, F, DecompressError, 4>,
}

pub async fn decompress<O>(
    mut outp: O,
    peek1: InputPeeker<'_, '_, 4, 1>,
    peek2: InputPeeker<'_, '_, 4, 2>,
    peek4: InputPeeker<'_, '_, 4, 4>,
) -> Result<(), DecompressError>
where
    O: LZOutputBuf,
{
    let magic = (&peek4).await;
    if magic[0] != 0x10 {
        return Err(DecompressError::BadMagic(magic[0]));
    }
    let encoded_len =
        (magic[1] as usize) | ((magic[2] as usize) << 8) | ((magic[3] as usize) << 16);

    while outp.cur_pos() < encoded_len && !outp.is_at_limit() {
        let flags = (&peek1).await[0];
        let flags = flags.view_bits::<Msb0>();

        for i in 0..8 {
            if outp.cur_pos() == encoded_len || outp.is_at_limit() {
                break;
            }

            if flags[i] == false {
                let b = (&peek1).await[0];
                outp.add_lits(&[b]);
            } else {
                let matchb = (&peek2).await;
                let matchb = matchb.view_bits::<Msb0>();
                let disp = matchb[4..].load_be::<usize>();
                let len = matchb[..4].load_be::<usize>();

                outp.add_match(disp + 1, len + 3)
                    .map_err(|_| DecompressError::BadLookback {
                        disp: disp as u16,
                        avail: outp.cur_pos() as u16,
                    })?;
            }
        }
    }

    Ok(())
}

impl<'a, F> Decompress<'a, F>
where
    F: Future<Output = Result<(), DecompressError>>,
{
    pub fn add_inp(&mut self, inp: &[u8]) -> Result<usize, DecompressError> {
        self.state.add_inp(inp)
    }
}

// technically copy-able, but don't want to make it easy to accidentally do so
#[derive(Clone)]
pub struct DecompressStreaming {
    lookback: [u8; LOOKBACK_SZ],
    lookback_wptr: usize,
    lookback_avail: usize,
    state: DecompressState,
    remaining_bytes: u32,
    block_flags: u8,
    block_i: u8,
    match_b0: u8,
}

impl DecompressStreaming {
    pub fn new() -> Self {
        Self {
            lookback: [0; LOOKBACK_SZ],
            lookback_wptr: 0,
            lookback_avail: 0,
            state: DecompressState::HdrMagic,
            remaining_bytes: 0,
            block_flags: 0,
            block_i: 0,
            match_b0: 0,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            let p_lookback = core::ptr::addr_of_mut!((*p).lookback);
            for i in 0..LOOKBACK_SZ {
                // my understanding is that this is safe because u64 doesn't have Drop
                // and we don't construct a & anywhere
                (*p_lookback)[i] = 0;
            }
            (*p).lookback_wptr = 0;
            (*p).lookback_avail = 0;
            (*p).state = DecompressState::HdrMagic;
            (*p).remaining_bytes = 0;
            (*p).block_flags = 0;
            (*p).block_i = 0;
            (*p).match_b0 = 0;
            Box::from_raw(p)
        }
    }

    pub fn manually_set_len(&mut self, len: u32) {
        assert!(self.state == DecompressState::HdrMagic);
        self.state = DecompressState::BlockFlags;
        self.remaining_bytes = len;
    }

    const fn need_more_bytes(&self) -> bool {
        self.state.is_header_state() || self.remaining_bytes > 0
    }

    pub fn decompress<O>(&mut self, mut inp: &[u8], mut outp: O) -> Result<bool, DecompressError>
    where
        O: FnMut(u8),
    {
        while inp.len() > 0 && self.need_more_bytes() {
            let b = get_inp::<1>(&mut inp).unwrap()[0];

            match self.state {
                DecompressState::HdrMagic => {
                    if b != 0x10 {
                        return Err(DecompressError::BadMagic(b));
                    }
                    self.state = DecompressState::HdrLen0;
                }
                DecompressState::HdrLen0 => {
                    self.remaining_bytes = b as u32;
                    self.state = DecompressState::HdrLen1;
                }
                DecompressState::HdrLen1 => {
                    self.remaining_bytes = self.remaining_bytes | ((b as u32) << 8);
                    self.state = DecompressState::HdrLen2;
                }
                DecompressState::HdrLen2 => {
                    self.remaining_bytes = self.remaining_bytes | ((b as u32) << 16);
                    self.state = DecompressState::BlockFlags;
                }
                DecompressState::BlockFlags => {
                    self.block_flags = b;
                    self.block_i = 0;
                    self.state = DecompressState::BlockData0;
                }
                DecompressState::BlockData0 => {
                    let flags = self.block_flags.view_bits::<Msb0>();
                    if flags[self.block_i as usize] == false {
                        let lit = b;
                        outp(lit);
                        self.remaining_bytes -= 1;
                        self.lookback[self.lookback_wptr] = lit;
                        self.lookback_wptr = (self.lookback_wptr + 1) % LOOKBACK_SZ;
                        if self.lookback_avail < LOOKBACK_SZ {
                            self.lookback_avail += 1;
                        }

                        self.block_i += 1;
                        if self.block_i == 8 {
                            self.state = DecompressState::BlockFlags;
                        }
                    } else {
                        self.match_b0 = b;
                        self.state = DecompressState::BlockData1;
                    }
                }
                DecompressState::BlockData1 => {
                    let matchb = [self.match_b0, b];
                    let matchb = matchb.view_bits::<Msb0>();

                    let disp = matchb[4..].load_be::<usize>();
                    let len = matchb[..4].load_be::<usize>();

                    if disp + 1 > self.lookback_avail {
                        return Err(DecompressError::BadLookback {
                            disp: disp as u16,
                            avail: self.lookback_avail as u16,
                        });
                    }

                    for _ in 0..(len + 3) {
                        if self.remaining_bytes == 0 {
                            break;
                        }

                        let idx = (self.lookback_wptr + LOOKBACK_SZ - disp - 1) % LOOKBACK_SZ;
                        let copy_b = self.lookback[idx];
                        outp(copy_b);
                        self.remaining_bytes -= 1;
                        self.lookback[self.lookback_wptr] = copy_b;
                        self.lookback_wptr = (self.lookback_wptr + 1) % LOOKBACK_SZ;
                        if self.lookback_avail < LOOKBACK_SZ {
                            self.lookback_avail += 1;
                        }
                    }

                    self.block_i += 1;
                    if self.block_i == 8 {
                        self.state = DecompressState::BlockFlags;
                    } else {
                        self.state = DecompressState::BlockData0;
                    }
                }
            }
        }

        Ok(!self.need_more_bytes())
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
        let magic = get_inp::<4>(&mut inp).map_err(|_| DecompressError::TooShort)?;
        if magic[0] != 0x10 {
            return Err(DecompressError::BadMagic(magic[0]));
        }
        let encoded_len =
            (magic[1] as usize) | ((magic[2] as usize) << 8) | ((magic[3] as usize) << 16);
        let wanted_len = core::cmp::min(encoded_len, outp.limit());

        while outp.cur_pos() < wanted_len {
            let flags = get_inp::<1>(&mut inp).map_err(|_| DecompressError::TooShort)?[0];
            let flags = flags.view_bits::<Msb0>();

            for i in 0..8 {
                if outp.cur_pos() == wanted_len {
                    break;
                }

                if flags[i] == false {
                    if inp.len() == 0 {
                        return Err(DecompressError::TooShort);
                    }
                    let b = get_inp::<1>(&mut inp).map_err(|_| DecompressError::TooShort)?[0];
                    outp.add_lit(b);
                } else {
                    let matchb = get_inp::<2>(&mut inp).map_err(|_| DecompressError::TooShort)?;
                    let matchb = matchb.view_bits::<Msb0>();
                    let disp = matchb[4..].load_be::<usize>();
                    let len = matchb[..4].load_be::<usize>();

                    let len = core::cmp::min(len + 3, wanted_len - outp.cur_pos());

                    outp.add_match(disp + 1, len)
                        .map_err(|_| DecompressError::BadLookback {
                            disp: disp as u16,
                            avail: outp.cur_pos() as u16,
                        })?;
                }
            }
        }

        Ok(())
    }

    pub fn decompress_into(&self, inp: &[u8], outp: &mut [u8]) -> Result<(), DecompressError> {
        self.decompress(inp, &mut FixedBuf::from(outp))
    }

    #[cfg(feature = "alloc")]
    pub fn decompress_new(&self, inp: &[u8]) -> Result<Vec<u8>, DecompressError> {
        if inp.len() < 4 {
            return Err(DecompressError::TooShort);
        }
        let encoded_len = (inp[1] as usize) | ((inp[2] as usize) << 8) | ((inp[3] as usize) << 16);
        let mut buf = VecBuf::new(encoded_len, encoded_len);
        self.decompress(inp, &mut buf)?;
        Ok(buf.into())
    }
}

pub struct Compress {
    engine:
        LZEngine<LOOKBACK_SZ, 18, { LOOKBACK_SZ + 18 }, 3, 18, 12, { 1 << 12 }, 12, { 1 << 12 }>,
    buffered_out: [LZOutput; 8],
    num_buffered_out: u8,
}

impl Compress {
    pub fn new() -> Self {
        Self {
            engine: LZEngine::new(),
            buffered_out: [LZOutput::default(); 8],
            num_buffered_out: 0,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            LZEngine::initialize_at(core::ptr::addr_of_mut!((*p).engine));
            let p_buffered_out = core::ptr::addr_of_mut!((*p).buffered_out);
            for i in 0..8 {
                // my understanding is that this is safe because u64 doesn't have Drop
                // and we don't construct a & anywhere
                (*p_buffered_out)[i] = LZOutput::default();
            }
            (*p).num_buffered_out = 0;
            Box::from_raw(p)
        }
    }

    pub fn encode_header(&self, len: u32) -> [u8; 4] {
        [
            0x10,
            (len & 0xFF) as u8,
            ((len >> 8) & 0xFF) as u8,
            ((len >> 16) & 0xFF) as u8,
        ]
    }

    pub fn compress<O>(&mut self, vram_mode: bool, inp: &[u8], end_of_stream: bool, mut outp: O)
    where
        O: FnMut(u8),
    {
        let settings = LZSettings {
            good_enough_search_len: 18,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: 1 << 12,
            defer_output_match: true,
            good_enough_defer_len: 18,
            search_faster_defer_len: 10,
            min_disp: if vram_mode { 2 } else { 1 },
            eos_holdout_bytes: 0,
        };

        macro_rules! dump_buffered_out {
            () => {
                let mut flags = 0;
                for i in 0..self.num_buffered_out {
                    match self.buffered_out[i as usize] {
                        LZOutput::Lit(_) => {}
                        LZOutput::Ref { .. } => {
                            flags.view_bits_mut::<Msb0>().set(i as usize, true);
                        }
                    }
                }
                outp(flags);

                for i in 0..self.num_buffered_out {
                    match self.buffered_out[i as usize] {
                        LZOutput::Lit(lit) => {
                            outp(lit);
                        }
                        LZOutput::Ref { disp, len } => {
                            debug_assert!(disp >= 1);
                            debug_assert!(disp <= 0x1000);
                            debug_assert!(len >= 3);
                            debug_assert!(len <= 18);

                            let disp = disp - 1;
                            let len = len - 3;

                            let mut matchb = [0u8; 2];
                            let matchb_v = matchb.view_bits_mut::<Msb0>();
                            matchb_v[4..].store_be(disp);
                            matchb_v[..4].store_be(len);
                            outp(matchb[0]);
                            outp(matchb[1]);
                        }
                    }
                }
            };
        }

        self.engine
            .compress::<_, ()>(&settings, inp, end_of_stream, |x| {
                self.buffered_out[self.num_buffered_out as usize] = x;
                self.num_buffered_out += 1;
                if self.num_buffered_out == 8 {
                    dump_buffered_out!();
                    self.num_buffered_out = 0;
                }
                Ok(())
            })
            .unwrap();

        if end_of_stream && self.num_buffered_out > 0 {
            dump_buffered_out!();
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
        vec,
        vec::Vec,
    };

    use crate::decompress::StreamingOutputBuf;

    use super::*;

    #[test]
    fn nin_wip_new_decomp() {
        let mut outvec = Vec::new();
        {
            let outp = StreamingOutputBuf::<_, LOOKBACK_SZ>::new(|x| outvec.extend_from_slice(x));

            let innerstate = crate::decompress::StreamingDecompressExecState::<4>::new();
            let peek1 = innerstate.get_peeker::<1>();
            let peek2 = innerstate.get_peeker::<2>();
            let peek4 = innerstate.get_peeker::<4>();
            let x = core::pin::pin!(decompress(outp, peek1, peek2, peek4));
            let state = crate::decompress::StreamingDecompressState::new(&innerstate, x);
            let mut decomp = Decompress { state };

            let ret = decomp.add_inp(&[0x10]);
            assert_eq!(ret, Ok(3));

            let ret = decomp.add_inp(&[10, 0, 0]);
            assert_eq!(ret, Ok(1));

            let ret = decomp.add_inp(&[0]);
            assert_eq!(ret, Ok(1));

            for i in 0..8 {
                let ret = decomp.add_inp(&[i]);
                assert_eq!(ret, Ok(1));
            }

            let ret = decomp.add_inp(&[0]);
            assert_eq!(ret, Ok(1));
            let ret = decomp.add_inp(&[8]);
            assert_eq!(ret, Ok(1));
            let ret = decomp.add_inp(&[9]);
            assert_eq!(ret, Ok(0));
        }

        assert_eq!(outvec, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn nin_all_lits() {
        let mut outp = Vec::new();
        let mut dec = DecompressStreaming::new_boxed();

        let ret = dec.decompress(&[0x10], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        assert_eq!(dec.state, DecompressState::HdrLen0);

        let ret = dec.decompress(&[10], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        let ret = dec.decompress(&[0], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        let ret = dec.decompress(&[0], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        assert_eq!(dec.state, DecompressState::BlockFlags);
        assert_eq!(dec.remaining_bytes, 10);

        let ret = dec.decompress(&[0], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        assert_eq!(dec.state, DecompressState::BlockData0);

        for i in 0..8 {
            let ret = dec.decompress(&[i], |x| outp.push(x));
            assert_eq!(ret, Ok(false));
        }
        assert_eq!(dec.state, DecompressState::BlockFlags);

        let ret = dec.decompress(&[0], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        let ret = dec.decompress(&[8], |x| outp.push(x));
        assert_eq!(ret, Ok(false));
        let ret = dec.decompress(&[9], |x| outp.push(x));
        assert_eq!(ret, Ok(true));

        assert_eq!(outp, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn nin_disp_repeating() {
        let mut outp = Vec::new();
        let mut dec = DecompressStreaming::new_boxed();

        let ret = dec.decompress(
            &[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b0110_0000, 1],
            |x| outp.push(x),
        );
        assert_eq!(ret, Ok(true));
        assert_eq!(
            outp,
            &[0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
        );
    }

    #[test]
    fn nin_disp_repeating_overlong() {
        let mut outp = Vec::new();
        let mut dec = DecompressStreaming::new_boxed();

        let ret = dec.decompress(
            &[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b1111_0000, 1],
            |x| outp.push(x),
        );
        assert_eq!(ret, Ok(true));
        assert_eq!(
            outp,
            &[0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
        );
    }

    #[test]
    fn nin_disp_non_repeating() {
        let mut outp = Vec::new();
        let mut dec = DecompressStreaming::new_boxed();

        let ret = dec.decompress(
            &[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 2, 5, 6],
            |x| outp.push(x),
        );
        assert_eq!(ret, Ok(true));
        assert_eq!(outp, &[1, 2, 3, 4, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn nin_disp_invalid() {
        let mut outp = Vec::new();
        let mut dec = DecompressStreaming::new_boxed();

        let ret = dec.decompress(
            &[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 4, 5, 6],
            |x| outp.push(x),
        );
        assert_eq!(ret, Err(DecompressError::BadLookback { disp: 4, avail: 4 }));
    }

    #[test]
    fn nin_buffered_short() {
        let dec = DecompressBuffered::new();

        let ret = dec.decompress_new(&[0x10, 10]);
        assert_eq!(ret, Err(DecompressError::TooShort));

        let ret = dec.decompress_new(&[0x10, 10, 0, 0]);
        assert_eq!(ret, Err(DecompressError::TooShort));

        let ret = dec.decompress_new(&[0x10, 10, 0, 0, 0]);
        assert_eq!(ret, Err(DecompressError::TooShort));
    }

    #[test]
    fn nin_buffered_lits() {
        let dec = DecompressBuffered::new();

        let ret = dec.decompress_new(&[0x10, 10, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11]);
        assert_eq!(ret, Ok(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    }

    #[test]
    fn nin_buffered_disp_repeating() {
        let dec = DecompressBuffered::new();

        let ret = dec.decompress_new(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b0110_0000, 1]);
        assert_eq!(
            ret,
            Ok(vec![
                0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa
            ])
        );

        let ret = dec.decompress_new(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b1111_0000, 1]);
        assert_eq!(
            ret,
            Ok(vec![
                0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa
            ])
        );
    }

    #[test]
    fn nin_buffered_disp_invalid() {
        let dec = DecompressBuffered::new();

        let ret =
            dec.decompress_new(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 4, 5, 6]);
        assert_eq!(ret, Err(DecompressError::BadLookback { disp: 4, avail: 4 }));
    }

    #[test]
    fn nin_buffered_disp_truncated() {
        let dec = DecompressBuffered::new();

        let ret = dec.decompress_new(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000]);
        assert_eq!(ret, Err(DecompressError::TooShort));
    }

    #[test]
    fn nin_basic_compress() {
        let mut comp = Compress::new_boxed();
        let mut out = Vec::new();
        comp.compress(false, &[1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6], true, |x| {
            out.push(x)
        });

        assert_eq!(out, &[0b00001000, 1, 2, 3, 4, 0b0011_0000, 2, 5, 6]);
    }

    #[test]
    fn nin_roundtrip_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/nintendo_lz.rs");
        let outp_fn = d.join("nintendo_lz_test.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = Compress::new_boxed();
        let mut compressed_out = Vec::new();
        compressed_out.extend_from_slice(&comp.encode_header(inp.len() as u32));
        for i in 0..inp.len() {
            comp.compress(false, &[inp[i]], false, |x| compressed_out.push(x));
        }
        comp.compress(false, &[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(outp_fn).unwrap());
        outp_f.write(&compressed_out).unwrap();

        let dec = DecompressBuffered::new();
        let decompress = dec.decompress_new(&compressed_out).unwrap();

        assert_eq!(inp, decompress);
    }
}
