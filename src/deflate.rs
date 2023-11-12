use core::fmt::Debug;
#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box, vec::Vec};

use bitvec::prelude::*;

use crate::{
    decompress::{InputPeeker, LZOutputBuf, StreamingDecompressState, StreamingOutputBuf},
    util::*,
    LZEngine, LZOutput, LZSettings,
};

const CODE_LEN_ALPHABET_SIZE: usize = 19;
const CODE_LEN_ORDER: [u8; CODE_LEN_ALPHABET_SIZE] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
const CODE_LEN_ORDER_REVERSE: [u8; CODE_LEN_ALPHABET_SIZE] = [
    3, 17, 15, 13, 11, 9, 7, 5, 4, 6, 8, 10, 12, 14, 16, 18, 0, 1, 2,
];
const LEN_FOR_SYM: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_FOR_SYM: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA_BITS: [u16; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

#[derive(Debug, PartialEq, Eq)]
pub enum DecompressError {
    BadLookback { disp: u16, avail: u16 },
    InvalidBlockType,
    InvalidHLit,
    InvalidHDist,
    InvalidHuffNoSyms,
    InvalidCodeLenRep,
    BadNLen { len: u16, nlen: u16 },
    BadHuffSym,
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
            DecompressError::InvalidBlockType => {
                write!(f, "Block type is not valid")
            }
            DecompressError::InvalidHLit => {
                write!(f, "Invalid HLIT count of literal/length codes")
            }
            DecompressError::InvalidHDist => {
                write!(f, "Invalid HDIST count of distance codes")
            }
            DecompressError::InvalidHuffNoSyms => {
                write!(f, "Invalid Huffman tree without symbols")
            }
            DecompressError::InvalidCodeLenRep => {
                write!(f, "Invalid repeat in Huffman tree code lengths")
            }
            DecompressError::BadNLen { len, nlen } => {
                write!(f, "Bad LEN/NLEN ({:04X}/{:04X})", len, nlen)
            }
            DecompressError::BadHuffSym => {
                write!(f, "An invalid symbol was encoded")
            }
            DecompressError::Truncated => {
                write!(f, "Input is truncated")
            }
        }
    }
}

#[cfg(feature = "std")]
impl Error for DecompressError {}

const LOOKBACK_SZ: usize = 32768;

pub use lamezip77_macros::deflate_decompress_make as decompress_make;

struct BitBuf {
    bits: u32,
    nbits: u8,
}

impl BitBuf {
    fn new() -> Self {
        Self { bits: 0, nbits: 0 }
    }

    #[inline]
    async fn get_bits_as_num<T: funty::Integral, const N: usize>(
        &mut self,
        peek1: &InputPeeker<'_, '_, 2, 1>,
    ) -> T {
        debug_assert!(N <= 16);
        debug_assert!(N <= 8 * core::mem::size_of::<T>());
        debug_assert!(self.nbits <= 8);

        let v = self.bits.view_bits_mut::<Lsb0>();

        while N > self.nbits as usize {
            let nextbits = peek1.await[0];
            v[(self.nbits as usize)..(self.nbits as usize + 8)].store_le(nextbits);
            self.nbits += 8;
        }

        debug_assert!(N <= self.nbits as usize);
        let ret = v[..N].load_le::<T>();
        v.shift_left(N);
        self.nbits -= N as u8;
        ret
    }

    #[inline]
    async fn get_bits_as_num_dyn<T: funty::Integral>(
        &mut self,
        n: usize,
        peek1: &InputPeeker<'_, '_, 2, 1>,
    ) -> T {
        if n == 0 {
            return T::ZERO;
        }

        debug_assert!(n <= 16);
        debug_assert!(n <= 8 * core::mem::size_of::<T>());
        debug_assert!(self.nbits <= 8);

        let v = self.bits.view_bits_mut::<Lsb0>();

        while n > self.nbits as usize {
            let nextbits = peek1.await[0];
            v[(self.nbits as usize)..(self.nbits as usize + 8)].store_le(nextbits);
            self.nbits += 8;
        }

        debug_assert!(n <= self.nbits as usize);
        let ret = v[..n].load_le::<T>();
        v.shift_left(n);
        self.nbits -= n as u8;
        ret
    }
}

pub async fn decompress_impl<O>(
    outp: &mut O,
    peek1: InputPeeker<'_, '_, 2, 1>,
    peek2: InputPeeker<'_, '_, 2, 2>,
) -> Result<(), DecompressError>
where
    O: LZOutputBuf,
{
    let mut bitbuf = BitBuf::new();
    let mut coded_lengths_decoder = CanonicalHuffmanDecoder::<CodedLenSym, 19, 7, 8, 128>::new();
    let mut lit_decoder = CanonicalHuffmanDecoder::<LitSym, 288, 15, 16, { 1 << 15 }>::new();
    let mut dist_decoder = CanonicalHuffmanDecoder::<DistSym, 30, 15, 16, { 1 << 15 }>::new();

    loop {
        let bfinal = bitbuf.get_bits_as_num::<u8, 1>(&peek1).await;
        let btype = bitbuf.get_bits_as_num::<u8, 2>(&peek1).await;

        if btype == 3 {
            return Err(DecompressError::InvalidBlockType);
        } else if btype == 0 {
            // implicitly drop everything left in bitbuf

            let len = u16::from_le_bytes((&peek2).await);
            let nlen = u16::from_le_bytes((&peek2).await);
            if len != !nlen {
                return Err(DecompressError::BadNLen { len, nlen });
            }

            for _ in 0..len {
                let b = (&peek1).await;
                outp.add_lits(&b);
            }
        } else {
            lit_decoder.reset();
            dist_decoder.reset();

            if btype == 1 {
                let mut lit_lens = [0; 288];
                for i in 0..=143 {
                    lit_lens[i] = 8;
                }
                for i in 144..=255 {
                    lit_lens[i] = 9;
                }
                for i in 256..=279 {
                    lit_lens[i] = 7;
                }
                for i in 280..=287 {
                    lit_lens[i] = 8;
                }
                lit_decoder.init(&lit_lens).unwrap();
                dist_decoder.init(&[5; 30]).unwrap();
            } else {
                let hlit = bitbuf.get_bits_as_num::<u16, 5>(&peek1).await + 257;
                let hdist = bitbuf.get_bits_as_num::<u8, 5>(&peek1).await + 1;
                let hclen = bitbuf.get_bits_as_num::<u8, 4>(&peek1).await + 4;
                if hlit > 286 {
                    return Err(DecompressError::InvalidHLit);
                }
                if hdist > 30 {
                    return Err(DecompressError::InvalidHDist);
                }

                let mut code_len_lens = [0; CODE_LEN_ALPHABET_SIZE];
                for i in 0..hclen {
                    let l = bitbuf.get_bits_as_num::<u8, 3>(&peek1).await;
                    let idx = CODE_LEN_ORDER[i as usize];
                    code_len_lens[idx as usize] = l;
                }

                coded_lengths_decoder.init(&code_len_lens)?;

                let mut actual_huff_code_lens = [0u8; 286 + 32];
                let mut actual_huff_code_i = 0;
                while actual_huff_code_i < (hlit + hdist as u16) as usize {
                    let code_len_sym = coded_lengths_decoder
                        .read_sym_2(&mut bitbuf, &peek1)
                        .await?
                        .into();

                    match code_len_sym {
                        0..=15 => {
                            actual_huff_code_lens[actual_huff_code_i] = code_len_sym as u8;
                            actual_huff_code_i += 1;
                        }
                        16 => {
                            let rep = bitbuf.get_bits_as_num::<u8, 2>(&peek1).await + 3;
                            if actual_huff_code_i == 0 {
                                return Err(DecompressError::InvalidCodeLenRep);
                            }
                            let to_rep = actual_huff_code_lens[actual_huff_code_i - 1];
                            if actual_huff_code_i + rep as usize > (hlit + hdist as u16) as usize {
                                return Err(DecompressError::InvalidCodeLenRep);
                            }

                            for i in 0..rep as usize {
                                actual_huff_code_lens[actual_huff_code_i + i] = to_rep;
                            }
                            actual_huff_code_i += rep as usize;
                        }
                        17 => {
                            let rep = bitbuf.get_bits_as_num::<u8, 3>(&peek1).await + 3;
                            actual_huff_code_i += rep as usize;
                        }
                        18 => {
                            let rep = bitbuf.get_bits_as_num::<u8, 7>(&peek1).await + 11;
                            actual_huff_code_i += rep as usize;
                        }
                        _ => unreachable!(),
                    }
                }

                if actual_huff_code_i != (hlit + hdist as u16) as usize {
                    return Err(DecompressError::InvalidCodeLenRep);
                }

                let (literal_code_lens, dist_code_lens) =
                    actual_huff_code_lens.split_at(hlit as usize);
                let dist_code_lens = &dist_code_lens[..hdist as usize];

                lit_decoder.init(literal_code_lens)?;
                if !(hdist == 1 && dist_code_lens[0] == 0) {
                    dist_decoder.init(dist_code_lens)?;
                }
            }

            while !outp.is_at_limit() {
                let litsym = lit_decoder.read_sym_2(&mut bitbuf, &peek1).await?.into();
                match litsym {
                    0..=0xff => {
                        outp.add_lits(&[litsym as u8]);
                    }
                    257..=285 => {
                        let len_extra_nbits = LEN_EXTRA_BITS[litsym - 257];
                        let len_extra = bitbuf
                            .get_bits_as_num_dyn::<u16>(len_extra_nbits as usize, &peek1)
                            .await;
                        let len = LEN_FOR_SYM[litsym - 257] + len_extra;

                        let distsym: usize =
                            dist_decoder.read_sym_2(&mut bitbuf, &peek1).await?.into();
                        if distsym > 29 {
                            return Err(DecompressError::BadHuffSym);
                        }
                        let dist_extra_nbits = DIST_EXTRA_BITS[distsym];
                        let dist_extra = bitbuf
                            .get_bits_as_num_dyn::<u16>(dist_extra_nbits as usize, &peek1)
                            .await;
                        let dist = DIST_FOR_SYM[distsym] + dist_extra;

                        outp.add_match(dist as usize - 1, len as usize)
                            .map_err(|_| DecompressError::BadLookback {
                                disp: dist,
                                avail: outp.cur_pos() as u16,
                            })?;
                    }
                    256 => {
                        break;
                    }
                    _ => return Err(DecompressError::BadHuffSym),
                }
            }
        }

        if bfinal != 0 {
            break;
        }
    }

    Ok(())
}

pub type Decompress<'a, F> = StreamingDecompressState<'a, F, DecompressError, 2>;
pub type DecompressBuffer<O> = StreamingOutputBuf<O, LOOKBACK_SZ>;

#[inline]
fn get_bits_num<T: funty::Integral, const N: usize>(
    inp: &mut &BitSlice<u8>,
) -> Result<T, DecompressError> {
    if inp.len() < N {
        Err(DecompressError::Truncated)
    } else {
        let ret = inp[..N].load_le::<T>();
        *inp = &inp[N..];
        Ok(ret)
    }
}

#[inline]
fn get_bits_num_dyn<T: funty::Integral>(
    inp: &mut &BitSlice<u8>,
    n: usize,
) -> Result<T, DecompressError> {
    debug_assert!(n <= 8 * core::mem::size_of::<T>());
    if inp.len() < n {
        Err(DecompressError::Truncated)
    } else {
        let ret = inp[..n].load_le::<T>();
        *inp = &inp[n..];
        Ok(ret)
    }
}

#[inline]
fn get_bits_slice<'a>(
    inp: &mut &'a BitSlice<u8>,
    n: usize,
) -> Result<&'a BitSlice<u8>, DecompressError> {
    if inp.len() < n {
        Err(DecompressError::Truncated)
    } else {
        let ret = &inp[..n];
        *inp = &inp[n..];
        Ok(ret)
    }
}

#[derive(Clone)]
struct CanonicalHuffmanDecoder<
    SymTy: SymTyTrait,
    const ALPHABET_SZ: usize,
    const MAX_LEN: usize,
    const MAX_LEN_PLUS_ONE: usize,
    const TWO_POW_MAX_LEN: usize,
> {
    lookup: [SymTy; TWO_POW_MAX_LEN],
    min_code_len: usize,
}

impl<
        SymTy: SymTyTrait,
        const ALPHABET_SZ: usize,
        const MAX_LEN: usize,
        const MAX_LEN_PLUS_ONE: usize,
        const TWO_POW_MAX_LEN: usize,
    > CanonicalHuffmanDecoder<SymTy, ALPHABET_SZ, MAX_LEN, MAX_LEN_PLUS_ONE, TWO_POW_MAX_LEN>
{
    fn init(&mut self, code_lens: &[u8]) -> Result<(), DecompressError> {
        assert!(MAX_LEN < 16); // hard limit good enough for deflate
        assert!(1 << MAX_LEN == TWO_POW_MAX_LEN);
        assert!(MAX_LEN + 1 == MAX_LEN_PLUS_ONE);
        assert!(code_lens.len() <= ALPHABET_SZ);

        let mut codes_at_bit_count = [0u16; MAX_LEN_PLUS_ONE];
        let mut min_code_len = u8::MAX;
        for sym in 0..ALPHABET_SZ {
            let l = if sym < code_lens.len() {
                code_lens[sym]
            } else {
                0
            };
            if l > 0 {
                min_code_len = core::cmp::min(min_code_len, l);
                codes_at_bit_count[l as usize] += 1;
            }
        }
        if min_code_len == u8::MAX {
            return Err(DecompressError::InvalidHuffNoSyms);
        }
        self.min_code_len = min_code_len as usize;

        let mut next_code_at_bit_count = [0u16; MAX_LEN_PLUS_ONE];
        let mut code = 0;
        for i in 1..=MAX_LEN {
            code = (code + codes_at_bit_count[i - 1]) << 1;
            next_code_at_bit_count[i] = code;
        }

        for sym in 0..ALPHABET_SZ {
            let l = if sym < code_lens.len() {
                code_lens[sym] as usize
            } else {
                0
            };
            if l > 0 {
                let code = next_code_at_bit_count[l];
                debug_assert!(l <= MAX_LEN);
                let code = code << (MAX_LEN - l);
                debug_assert!(self.lookup[code as usize] == SymTy::default());
                self.lookup[code as usize] = (sym, l as u8).into();
                next_code_at_bit_count[l] += 1;
            }
        }

        Ok(())
    }

    fn read_sym(&self, inp: &mut &BitSlice<u8>) -> Result<SymTy, DecompressError> {
        let min_code = get_bits_slice(inp, self.min_code_len)?;
        let mut cur_code = 0u16;
        let cur_code_v = cur_code.view_bits_mut::<Msb0>();
        cur_code_v[(16 - MAX_LEN)..(16 - MAX_LEN + self.min_code_len)]
            .clone_from_bitslice(&min_code);
        let mut cur_code_len = self.min_code_len;

        while self.lookup[cur_code as usize] == SymTy::default()
            || self.lookup[cur_code as usize].nbits() as usize != cur_code_len
        {
            let nextbit: u8 = get_bits_num::<u8, 1>(inp)?;
            let cur_code_v = cur_code.view_bits_mut::<Msb0>();
            cur_code_v.set(16 - MAX_LEN + cur_code_len, nextbit != 0);
            cur_code_len += 1;
        }

        let sym = self.lookup[cur_code as usize];
        debug_assert!(sym != SymTy::default());
        Ok(sym)
    }

    async fn read_sym_2(
        &self,
        bb: &mut BitBuf,
        peek1: &InputPeeker<'_, '_, 2, 1>,
    ) -> Result<SymTy, DecompressError> {
        let min_code = bb
            .get_bits_as_num_dyn::<u16>(self.min_code_len, peek1)
            .await;
        let min_code = &min_code.view_bits::<Lsb0>()[..self.min_code_len];
        let mut cur_code = 0u16;
        let cur_code_v = cur_code.view_bits_mut::<Msb0>();
        cur_code_v[(16 - MAX_LEN)..(16 - MAX_LEN + self.min_code_len)]
            .clone_from_bitslice(&min_code);
        let mut cur_code_len = self.min_code_len;

        while self.lookup[cur_code as usize] == SymTy::default()
            || self.lookup[cur_code as usize].nbits() as usize != cur_code_len
        {
            let nextbit = bb.get_bits_as_num::<u8, 1>(peek1).await;
            let cur_code_v = cur_code.view_bits_mut::<Msb0>();
            cur_code_v.set(16 - MAX_LEN + cur_code_len, nextbit != 0);
            cur_code_len += 1;
        }

        let sym = self.lookup[cur_code as usize];
        debug_assert!(sym != SymTy::default());
        Ok(sym)
    }

    fn new() -> Self {
        Self {
            lookup: [SymTy::default(); TWO_POW_MAX_LEN],
            min_code_len: 0,
        }
    }
    unsafe fn initialize_at(p: *mut Self) {
        let p_lookup = core::ptr::addr_of_mut!((*p).lookup);
        for i in 0..TWO_POW_MAX_LEN {
            (*p_lookup)[i] = SymTy::default();
        }
        (*p).min_code_len = 0;
    }
    fn reset(&mut self) {
        for i in 0..TWO_POW_MAX_LEN {
            self.lookup[i] = SymTy::default();
        }
        self.min_code_len = 0;
    }
}

trait SymTyTrait: Default + Copy + core::fmt::Debug + From<(usize, u8)> + Into<usize> + Eq {
    fn nbits(&self) -> u8;
}

#[derive(Debug)]
struct CanonicalHuffmanEncoder<const ALPHABET_SZ: usize> {
    codes: [u16; ALPHABET_SZ],
    nbits: [u8; ALPHABET_SZ],
}

impl<const ALPHABET_SZ: usize> CanonicalHuffmanEncoder<ALPHABET_SZ> {
    fn new(code_lens: &[u8]) -> Self {
        assert!(code_lens.len() <= ALPHABET_SZ);

        let mut codes_at_bit_count = [0u16; 16];
        let mut min_code_len = u8::MAX;
        for sym in 0..ALPHABET_SZ {
            let l = if sym < code_lens.len() {
                code_lens[sym]
            } else {
                0
            };
            if l > 0 {
                min_code_len = core::cmp::min(min_code_len, l);
                codes_at_bit_count[l as usize] += 1;
            }
        }

        let mut codes = [0; ALPHABET_SZ];
        let mut nbits = [0; ALPHABET_SZ];
        if min_code_len == u8::MAX {
            // no codes, return dummy
            return Self { codes, nbits };
        }

        let mut next_code_at_bit_count = [0u16; 16];
        let mut code = 0;
        for i in 1..=15 {
            code = (code + codes_at_bit_count[i - 1]) << 1;
            next_code_at_bit_count[i] = code;
        }

        for sym in 0..ALPHABET_SZ {
            let l = if sym < code_lens.len() {
                code_lens[sym] as usize
            } else {
                0
            };
            if l > 0 {
                let mut code = next_code_at_bit_count[l];
                let code_v = code.view_bits_mut::<Lsb0>();
                code_v[..l].reverse();
                codes[sym] = code;
                nbits[sym] = l as u8;
                next_code_at_bit_count[l] += 1;
            }
        }

        Self { codes, nbits }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct CodedLenSym(u8);
impl Default for CodedLenSym {
    fn default() -> Self {
        Self(0xff)
    }
}
impl From<(usize, u8)> for CodedLenSym {
    fn from((value, nbits): (usize, u8)) -> Self {
        debug_assert!(value <= 18);
        debug_assert!(nbits <= 7);
        let mut ret = 0;
        let ret_v = ret.view_bits_mut::<Msb0>();
        ret_v[..3].store(nbits);
        ret_v[3..].store(value);
        Self(ret)
    }
}
impl Into<usize> for CodedLenSym {
    fn into(self) -> usize {
        self.0.view_bits::<Msb0>()[3..].load()
    }
}
impl SymTyTrait for CodedLenSym {
    fn nbits(&self) -> u8 {
        self.0.view_bits::<Msb0>()[..3].load()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DistSym {
    value: u8,
    nbits: u8,
}
impl Default for DistSym {
    fn default() -> Self {
        Self {
            value: 0xff,
            nbits: 0xff,
        }
    }
}
impl From<(usize, u8)> for DistSym {
    fn from((value, nbits): (usize, u8)) -> Self {
        debug_assert!(value <= 29); // XXX
        debug_assert!(nbits <= 15);
        Self {
            value: value as u8,
            nbits,
        }
    }
}
impl Into<usize> for DistSym {
    fn into(self) -> usize {
        self.value as usize
    }
}
impl SymTyTrait for DistSym {
    fn nbits(&self) -> u8 {
        self.nbits
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct LitSym(u16);
impl Default for LitSym {
    fn default() -> Self {
        Self(0xffff)
    }
}
impl From<(usize, u8)> for LitSym {
    fn from((value, nbits): (usize, u8)) -> Self {
        debug_assert!(value <= 287);
        debug_assert!(nbits <= 15);
        let mut ret = 0;
        let ret_v = ret.view_bits_mut::<Msb0>();
        ret_v[..4].store(nbits);
        ret_v[4..].store(value);
        Self(ret)
    }
}
impl Into<usize> for LitSym {
    fn into(self) -> usize {
        self.0.view_bits::<Msb0>()[4..].load()
    }
}
impl SymTyTrait for LitSym {
    fn nbits(&self) -> u8 {
        self.0.view_bits::<Msb0>()[..4].load()
    }
}

// sym, extra, #extra
const fn len_to_huff(len: u16) -> (u16, u8, u8) {
    match len {
        3 => (257, 0, 0),
        4 => (258, 0, 0),
        5 => (259, 0, 0),
        6 => (260, 0, 0),
        7 => (261, 0, 0),
        8 => (262, 0, 0),
        9 => (263, 0, 0),
        10 => (264, 0, 0),
        11..=12 => (265, (len - 11) as u8, 1),
        13..=14 => (266, (len - 13) as u8, 1),
        15..=16 => (267, (len - 15) as u8, 1),
        17..=18 => (268, (len - 17) as u8, 1),
        19..=22 => (269, (len - 19) as u8, 2),
        23..=26 => (270, (len - 23) as u8, 2),
        27..=30 => (271, (len - 27) as u8, 2),
        31..=34 => (272, (len - 31) as u8, 2),
        35..=42 => (273, (len - 35) as u8, 3),
        43..=50 => (274, (len - 43) as u8, 3),
        51..=58 => (275, (len - 51) as u8, 3),
        59..=66 => (276, (len - 59) as u8, 3),
        67..=82 => (277, (len - 67) as u8, 4),
        83..=98 => (278, (len - 83) as u8, 4),
        99..=114 => (279, (len - 99) as u8, 4),
        115..=130 => (280, (len - 115) as u8, 4),
        131..=162 => (281, (len - 131) as u8, 5),
        163..=194 => (282, (len - 163) as u8, 5),
        195..=226 => (283, (len - 195) as u8, 5),
        227..=257 => (284, (len - 227) as u8, 5),
        258 => (285, 0, 0),
        _ => unreachable!(),
    }
}

// sym, extra, #extra
const fn dist_to_huff(dist: u16) -> (u8, u16, u8) {
    match dist {
        1 => (0, 0, 0),
        2 => (1, 0, 0),
        3 => (2, 0, 0),
        4 => (3, 0, 0),
        5..=6 => (4, dist - 5, 1),
        7..=8 => (5, dist - 7, 1),
        9..=12 => (6, dist - 9, 2),
        13..=16 => (7, dist - 13, 2),
        17..=24 => (8, dist - 17, 3),
        25..=32 => (9, dist - 25, 3),
        33..=48 => (10, dist - 33, 4),
        49..=64 => (11, dist - 49, 4),
        65..=96 => (12, dist - 65, 5),
        97..=128 => (13, dist - 97, 5),
        129..=192 => (14, dist - 129, 6),
        193..=256 => (15, dist - 193, 6),
        257..=384 => (16, dist - 257, 7),
        385..=512 => (17, dist - 385, 7),
        513..=768 => (18, dist - 513, 8),
        769..=1024 => (19, dist - 769, 8),
        1025..=1536 => (20, dist - 1025, 9),
        1537..=2048 => (21, dist - 1537, 9),
        2049..=3072 => (22, dist - 2049, 10),
        3073..=4096 => (23, dist - 3073, 10),
        4097..=6144 => (24, dist - 4097, 11),
        6145..=8192 => (25, dist - 6145, 11),
        8193..=12288 => (26, dist - 8193, 12),
        12289..=16384 => (27, dist - 12289, 12),
        16385..=24576 => (28, dist - 16385, 13),
        24577..=32768 => (29, dist - 24577, 13),
        _ => unreachable!(),
    }
}

struct CompressState<const HUFF_BUF_SZ: usize> {
    huff_buf: [u16; HUFF_BUF_SZ],
    huff_buf_count: usize,
    pending_bits: u8,
    pending_bits_count: usize,
    len_lit_probabilities: [u16; 286],
    len_lit_denom: usize,
    dist_probabilities: [u16; 30],
    dist_denom: usize,
}

#[derive(Clone, Copy)]
enum PackMergeItemInner {
    Leaf { sym: u16 },
    Package { a: usize, b: usize },
}

impl Default for PackMergeItemInner {
    fn default() -> Self {
        Self::Leaf { sym: 0 }
    }
}

impl Debug for PackMergeItemInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Leaf { sym } => write!(f, "{}", sym),
            Self::Package { a, b } => write!(f, "{{{}, {}}}", a, b),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PackMergeItem {
    prob: u32,
    i: PackMergeItemInner,
}

impl Debug for PackMergeItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{{prob = {}, {:?}}}", self.prob, self.i)
    }
}

#[derive(Clone, Copy)]
struct PackMergeIteration<const NSYMS_TIMES_2: usize> {
    pkgs: [PackMergeItem; NSYMS_TIMES_2],
    npkgs: usize,
}

impl<const NSYMS_TIMES_2: usize> Default for PackMergeIteration<NSYMS_TIMES_2> {
    fn default() -> Self {
        Self {
            pkgs: [PackMergeItem::default(); NSYMS_TIMES_2],
            npkgs: 0,
        }
    }
}

impl<const NSYMS_TIMES_2: usize> Debug for PackMergeIteration<NSYMS_TIMES_2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write! {f, "{:?}", &self.pkgs[..self.npkgs]}
    }
}

impl<const NSYMS_TIMES_2: usize> PackMergeIteration<NSYMS_TIMES_2> {
    fn insert(&mut self, item: PackMergeItem) {
        let insert_idx = self.pkgs[..self.npkgs].partition_point(|x| x.prob < item.prob);
        self.pkgs[insert_idx..].rotate_right(1);
        self.pkgs[insert_idx] = item;
        self.npkgs += 1;
    }
}

struct PackMergeState<const NSYMS: usize, const NSYMS_TIMES_2: usize, const NBITS: usize> {
    iters: [PackMergeIteration<NSYMS_TIMES_2>; NBITS],
}

impl<const NSYMS: usize, const NSYMS_TIMES_2: usize, const NBITS: usize>
    PackMergeState<NSYMS, NSYMS_TIMES_2, NBITS>
{
    fn incr_nbits(&mut self, nbits: &mut [u8], item: &PackMergeItem, iter: usize) {
        debug_assert!(item.prob > 0);
        match item.i {
            PackMergeItemInner::Leaf { sym } => {
                nbits[sym as usize] += 1;
            }
            PackMergeItemInner::Package { a, b } => {
                let a_item = self.iters[iter + 1].pkgs[a];
                let b_item = self.iters[iter + 1].pkgs[b];
                self.incr_nbits(nbits, &a_item, iter + 1);
                self.incr_nbits(nbits, &b_item, iter + 1);
            }
        }
    }

    fn build<ProbType: Copy + Into<u32>>(sym_prob_table: &[ProbType]) -> [u8; NSYMS] {
        let mut ret = Self {
            iters: [PackMergeIteration::default(); NBITS],
        };

        assert!(NSYMS * 2 == NSYMS_TIMES_2);
        assert!(NSYMS_TIMES_2 % 2 == 0);
        assert!(sym_prob_table.len() * 2 <= NSYMS_TIMES_2);
        assert!(NBITS <= 16);
        assert!(NSYMS <= u16::MAX as usize);

        let mut num_nonzero_probs = 0;

        // base level -- all leaves
        let last_iter = &mut ret.iters[NBITS - 1];
        for i in 0..NSYMS {
            if sym_prob_table[i].into() > 0 {
                num_nonzero_probs += 1;
                last_iter.insert(PackMergeItem {
                    prob: sym_prob_table[i].into(),
                    i: PackMergeItemInner::Leaf { sym: i as u16 },
                });
            }
        }

        // newer levels
        for denom_iter in (1..NBITS).rev() {
            let (this_iter, last_iter) =
                &mut ret.iters[denom_iter - 1..=denom_iter].split_at_mut(1);
            let this_iter = &mut this_iter[0];
            let last_iter = &last_iter[0];
            // let this_iter = &mut ret.iters[denom_iter - 1];
            // leafs
            for i in 0..NSYMS {
                if sym_prob_table[i].into() > 0 {
                    this_iter.insert(PackMergeItem {
                        prob: sym_prob_table[i].into(),
                        i: PackMergeItemInner::Leaf { sym: i as u16 },
                    });
                }
            }

            for last_iter_i in (0..last_iter.npkgs).step_by(2) {
                let a = &last_iter.pkgs[last_iter_i];
                let b = &last_iter.pkgs[last_iter_i + 1];
                let newitem = PackMergeItem {
                    prob: a.prob + b.prob,
                    i: PackMergeItemInner::Package {
                        a: last_iter_i,
                        b: last_iter_i + 1,
                    },
                };
                this_iter.insert(newitem);
            }
        }

        debug_assert!(num_nonzero_probs > 0);

        let mut sym_nbits = [0; NSYMS];
        for i in 0..(2 * num_nonzero_probs - 2) {
            let item = ret.iters[0].pkgs[i];
            ret.incr_nbits(&mut sym_nbits, &item, 0);
        }

        sym_nbits
    }
}

impl<const HUFF_BUF_SZ: usize> CompressState<HUFF_BUF_SZ> {
    fn new() -> Self {
        assert!(HUFF_BUF_SZ <= u16::MAX as usize);
        Self {
            huff_buf: [0; HUFF_BUF_SZ],
            huff_buf_count: 0,
            pending_bits: 0,
            pending_bits_count: 0,
            len_lit_probabilities: [0; 286],
            len_lit_denom: 0,
            dist_probabilities: [0; 30],
            dist_denom: 0,
        }
    }
    unsafe fn initialize_at(p: *mut Self) {
        assert!(HUFF_BUF_SZ <= u16::MAX as usize);
        let p_huff_buf = core::ptr::addr_of_mut!((*p).huff_buf);
        for i in 0..HUFF_BUF_SZ {
            (*p_huff_buf)[i] = 0;
        }
        (*p).huff_buf_count = 0;
        (*p).pending_bits = 0;
        (*p).pending_bits_count = 0;
        let p_len_lit_probs = core::ptr::addr_of_mut!((*p).len_lit_probabilities);
        for i in 0..286 {
            (*p_len_lit_probs)[i] = 0;
        }
        (*p).len_lit_denom = 0;
        let p_dist_probs = core::ptr::addr_of_mut!((*p).dist_probabilities);
        for i in 0..30 {
            (*p_dist_probs)[i] = 0;
        }
        (*p).dist_denom = 0;
    }

    fn outbits<O>(&mut self, mut bits: u16, nbits: usize, mut outp: O)
    where
        O: FnMut(u8),
    {
        debug_assert!(nbits <= 16);
        bits = bits & ((1 << nbits) - 1);
        let mut work = self.pending_bits as u32;
        let work_v = work.view_bits_mut::<Lsb0>();
        work_v[self.pending_bits_count..self.pending_bits_count + nbits].store_le(bits);

        let mut totbits = self.pending_bits_count + nbits;
        while totbits >= 8 {
            outp((work & 0xff) as u8);
            work >>= 8;
            totbits -= 8;
        }
        self.pending_bits = (work & 0xff) as u8;
        self.pending_bits_count = totbits;
    }

    fn do_huff_buf<O>(&mut self, is_final: bool, mut outp: O)
    where
        O: FnMut(u8),
    {
        self.outbits(is_final as u16, 1, &mut outp);
        if self.huff_buf_count == 0 {
            self.outbits(0b00, 2, &mut outp);
            if self.pending_bits_count > 0 {
                outp(self.pending_bits);
            }
            // len, nlen
            outp(0x00);
            outp(0x00);
            outp(0xff);
            outp(0xff);
        } else {
            // end of block
            self.len_lit_probabilities[256] += 1;
            self.len_lit_denom += 1;

            // build huffman trees
            let lit_len_sym_nbits =
                PackMergeState::<286, { 286 * 2 }, 15>::build(&self.len_lit_probabilities);
            let mut hlit = 286
                - lit_len_sym_nbits
                    .iter()
                    .rev()
                    .position(|&x| x != 0)
                    .unwrap();
            if hlit < 257 {
                hlit = 257;
            }
            let (dist_sym_nbits, hdist) = if self.dist_denom != 0 {
                let dist_sym_nbits =
                    PackMergeState::<30, { 30 * 2 }, 15>::build(&self.dist_probabilities);
                let hdist = 30 - dist_sym_nbits.iter().rev().position(|&x| x != 0).unwrap();
                debug_assert!(hdist >= 1);
                (dist_sym_nbits, hdist)
            } else {
                ([0; 30], 1)
            };

            let mut huff_tree_probs = [0u32; 19];
            let mut huff_trees_codes = [0u8; { 286 + 30 }];
            let mut huff_trees_extra = [0u8; { 286 + 30 }];
            let mut huff_trees_ncodes = 0;
            let mut merged_codes = lit_len_sym_nbits[..hlit]
                .iter()
                .chain(dist_sym_nbits[..hdist].iter());
            let mut last_huff_code = 0xff;
            let mut last_huff_code_repeats = 0;

            macro_rules! dump_last_code {
                () => {
                    if last_huff_code != 0xff {
                        if (last_huff_code == 0 && last_huff_code_repeats <= 2)
                            || (last_huff_code != 0 && last_huff_code_repeats <= 3)
                        {
                            for _ in 0..last_huff_code_repeats {
                                huff_trees_codes[huff_trees_ncodes] = last_huff_code;
                                huff_tree_probs[last_huff_code as usize] += 1;
                                huff_trees_ncodes += 1;
                            }
                        } else if last_huff_code == 0 {
                            debug_assert!(last_huff_code_repeats <= 138);
                            if last_huff_code_repeats <= 10 {
                                huff_trees_codes[huff_trees_ncodes] = 17;
                                huff_trees_extra[huff_trees_ncodes] = last_huff_code_repeats - 3;
                                huff_tree_probs[17] += 1;
                                huff_trees_ncodes += 1;
                            } else {
                                huff_trees_codes[huff_trees_ncodes] = 18;
                                huff_trees_extra[huff_trees_ncodes] = last_huff_code_repeats - 11;
                                huff_tree_probs[18] += 1;
                                huff_trees_ncodes += 1;
                            }
                        } else {
                            debug_assert!(last_huff_code_repeats <= 7);
                            huff_trees_codes[huff_trees_ncodes] = last_huff_code;
                            huff_trees_codes[huff_trees_ncodes + 1] = 16;
                            huff_trees_extra[huff_trees_ncodes + 1] = last_huff_code_repeats - 4;
                            huff_tree_probs[last_huff_code as usize] += 1;
                            huff_tree_probs[16] += 1;
                            huff_trees_ncodes += 2;
                        }
                    }
                };
            }

            while let Some(&code) = merged_codes.next() {
                if code != last_huff_code {
                    dump_last_code!();

                    last_huff_code = code;
                    last_huff_code_repeats = 1;
                } else {
                    last_huff_code_repeats += 1;
                    if code != 0 && last_huff_code_repeats == 7 {
                        // full
                        huff_trees_codes[huff_trees_ncodes] = code;
                        huff_trees_codes[huff_trees_ncodes + 1] = 16;
                        huff_trees_extra[huff_trees_ncodes + 1] = 3;
                        huff_tree_probs[code as usize] += 1;
                        huff_tree_probs[16] += 1;
                        huff_trees_ncodes += 2;
                        last_huff_code = 0xff;
                        last_huff_code_repeats = 0;
                    }
                    if code == 0 && last_huff_code_repeats == 138 {
                        // full
                        huff_trees_codes[huff_trees_ncodes] = 18;
                        huff_trees_extra[huff_trees_ncodes] = 0x7f;
                        huff_tree_probs[18] += 1;
                        huff_trees_ncodes += 1;
                        last_huff_code = 0xff;
                        last_huff_code_repeats = 0;
                    }
                }
            }
            dump_last_code!();

            let huff_tree_sym_nbits = PackMergeState::<19, { 19 * 2 }, 7>::build(&huff_tree_probs);

            let mut huff_tree_sym_nbits_permuted = [0; 19];
            for i in 0..19 {
                huff_tree_sym_nbits_permuted[CODE_LEN_ORDER_REVERSE[i] as usize] =
                    huff_tree_sym_nbits[i];
            }
            let mut hclen = 19
                - huff_tree_sym_nbits_permuted
                    .iter()
                    .rev()
                    .position(|&x| x != 0)
                    .unwrap();
            if hclen < 4 {
                hclen = 4;
            }

            // TODO detect incompressible data
            self.outbits(0b10, 2, &mut outp);
            self.outbits((hlit - 257) as u16, 5, &mut outp);
            self.outbits((hdist - 1) as u16, 5, &mut outp);
            self.outbits((hclen - 4) as u16, 4, &mut outp);

            for i in 0..hclen {
                self.outbits(huff_tree_sym_nbits_permuted[i] as u16, 3, &mut outp);
            }

            let huff_trees_encoder = CanonicalHuffmanEncoder::<19>::new(&huff_tree_sym_nbits);
            for i in 0..huff_trees_ncodes {
                let code = huff_trees_codes[i];
                self.outbits(
                    huff_trees_encoder.codes[code as usize],
                    huff_trees_encoder.nbits[code as usize] as usize,
                    &mut outp,
                );
                if code == 16 {
                    self.outbits(huff_trees_extra[i] as u16, 2, &mut outp);
                } else if code == 17 {
                    self.outbits(huff_trees_extra[i] as u16, 3, &mut outp);
                } else if code == 18 {
                    self.outbits(huff_trees_extra[i] as u16, 7, &mut outp);
                }
            }

            let huff_lit_len_encoder = CanonicalHuffmanEncoder::<286>::new(&lit_len_sym_nbits);
            let huff_dist_encoder = CanonicalHuffmanEncoder::<30>::new(&dist_sym_nbits);

            let mut i = 0;
            while i < self.huff_buf_count {
                let thing = self.huff_buf[i];
                if thing & 0x8000 == 0 {
                    // lit
                    self.outbits(
                        huff_lit_len_encoder.codes[thing as usize],
                        huff_lit_len_encoder.nbits[thing as usize] as usize,
                        &mut outp,
                    );
                    i += 1;
                } else {
                    // ref
                    let len = thing & 0x7fff;
                    let disp = self.huff_buf[i + 1] + 1;
                    let (len_sym, len_extra, len_nextra) = len_to_huff(len);
                    self.outbits(
                        huff_lit_len_encoder.codes[len_sym as usize],
                        huff_lit_len_encoder.nbits[len_sym as usize] as usize,
                        &mut outp,
                    );
                    if len_nextra > 0 {
                        self.outbits(len_extra as u16, len_nextra as usize, &mut outp);
                    }
                    let (disp_sym, disp_extra, disp_nextra) = dist_to_huff(disp);
                    self.outbits(
                        huff_dist_encoder.codes[disp_sym as usize],
                        huff_dist_encoder.nbits[disp_sym as usize] as usize,
                        &mut outp,
                    );
                    if disp_nextra > 0 {
                        self.outbits(disp_extra as u16, disp_nextra as usize, &mut outp);
                    }
                    i += 2;
                }
            }

            self.outbits(
                huff_lit_len_encoder.codes[256],
                huff_lit_len_encoder.nbits[256] as usize,
                &mut outp,
            );

            self.huff_buf_count = 0;

            if is_final {
                if self.pending_bits_count > 0 {
                    outp(self.pending_bits);
                }
            }
        }
    }

    fn add_lit<O>(&mut self, lit: u8, outp: O)
    where
        O: FnMut(u8),
    {
        if self.huff_buf_count == HUFF_BUF_SZ {
            self.do_huff_buf(false, outp);
        }
        self.huff_buf[self.huff_buf_count] = lit as u16;
        self.huff_buf_count += 1;
        self.len_lit_probabilities[lit as usize] += 1;
        self.len_lit_denom += 1;
    }

    fn add_ref<O>(&mut self, disp: u16, len: u16, outp: O)
    where
        O: FnMut(u8),
    {
        if self.huff_buf_count >= HUFF_BUF_SZ - 1 {
            self.do_huff_buf(false, outp);
        }
        self.huff_buf[self.huff_buf_count] = (len as u16) | 0x8000;
        self.huff_buf[self.huff_buf_count + 1] = (disp - 1) as u16;
        self.huff_buf_count += 2;

        let (len_sym, _, _) = len_to_huff(len);
        self.len_lit_probabilities[len_sym as usize] += 1;
        self.len_lit_denom += 1;

        let (disp_sym, _, _) = dist_to_huff(disp);
        self.dist_probabilities[disp_sym as usize] += 1;
        self.dist_denom += 1;
    }
}

pub struct Compress<const HUFF_BUF_SZ: usize> {
    engine:
        LZEngine<LOOKBACK_SZ, 258, { LOOKBACK_SZ + 258 }, 3, 258, 15, { 1 << 15 }, 15, { 1 << 15 }>,
    state: CompressState<HUFF_BUF_SZ>,
}

impl<const HUFF_BUF_SZ: usize> Compress<HUFF_BUF_SZ> {
    pub fn new() -> Self {
        Self {
            engine: LZEngine::new(),
            state: CompressState::new(),
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            LZEngine::initialize_at(core::ptr::addr_of_mut!((*p).engine));
            CompressState::initialize_at(core::ptr::addr_of_mut!((*p).state));
            Box::from_raw(p)
        }
    }

    pub fn compress<O>(&mut self, inp: &[u8], end_of_stream: bool, mut outp: O)
    where
        O: FnMut(u8),
    {
        let settings = LZSettings {
            good_enough_search_len: 258,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: 4096, // copy zlib/gzip level 9
            defer_output_match: true,
            good_enough_defer_len: 258,
            search_faster_defer_len: 32, // copy zlib/gzip level 9
            min_disp: 1,
            eos_holdout_bytes: 0,
        };

        let s = &mut self.state;

        self.engine
            .compress::<_, ()>(&settings, inp, end_of_stream, |x| match x {
                LZOutput::Lit(lit) => {
                    s.add_lit(lit, &mut outp);
                    Ok(())
                }
                LZOutput::Ref { disp, len } => {
                    debug_assert!(disp >= 1);
                    debug_assert!(disp <= 32768);
                    debug_assert!(len >= 3);
                    debug_assert!(len <= 258);
                    s.add_ref(disp as u16, len as u16, &mut outp);
                    Ok(())
                }
            })
            .unwrap();

        if end_of_stream {
            s.do_huff_buf(true, &mut outp);
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
        vec::Vec,
    };

    use super::*;

    #[test]
    fn deflate_buffered_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate.bin");
        let ref_fn = d.join("deflatetest/tool.c");

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
    fn deflate_buffered_ref_decompress_stored() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate-stored.bin");
        let ref_fn = d.join("deflatetest/tool.c");

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
    fn deflate_buffered_ref_decompress_fixed() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate-fixed.bin");
        let ref_fn = d.join("deflatetest/tool.c");

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
    fn deflate_roundtrip_zerobytes() {
        let mut comp = Compress::<32768>::new_boxed();
        let mut compressed_out = Vec::new();
        comp.compress(&[], true, |x| compressed_out.push(x));

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
    fn deflate_roundtrip_bytewise() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/deflate.rs");
        let outp_fn = d.join("deflate-roundtrip-bytewise.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = Compress::<32768>::new_boxed();
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
        assert_eq!(decompress_ourselves, inp);

        let ref_fn = d.join("deflate-roundtrip-bytewise.out");
        Command::new("./deflatetest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }

    #[test]
    fn deflate_roundtrip_wholefile() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/deflate.rs");
        let outp_fn = d.join("deflate-roundtrip-wholefile.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut comp = Compress::<32768>::new_boxed();
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
        assert_eq!(decompress_ourselves, inp);

        let ref_fn = d.join("deflate-roundtrip-wholefile.out");
        Command::new("./deflatetest/tool")
            .arg("d")
            .arg(outp_fn.to_str().unwrap())
            .arg(ref_fn.to_str().unwrap())
            .status()
            .unwrap();
        let decompress_ref = std::fs::read(&ref_fn).unwrap();
        assert_eq!(inp, decompress_ref);
    }
}
