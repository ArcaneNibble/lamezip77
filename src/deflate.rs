use std::error::Error;

use bitvec::prelude::*;

use crate::util::*;

const CODE_LEN_ALPHABET_SIZE: usize = 19;
const CODE_LEN_ORDER: [u8; CODE_LEN_ALPHABET_SIZE] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
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

impl Error for DecompressError {}

#[inline]
pub fn get_bits_num<T: funty::Integral, const N: usize>(
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
pub fn get_bits_num_dyn<T: funty::Integral>(
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
pub fn get_bits_slice<'a>(
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
    pub fn init(&mut self, code_lens: &[u8]) -> Result<(), DecompressError> {
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

    pub fn read_sym(&self, inp: &mut &BitSlice<u8>) -> Result<SymTy, DecompressError> {
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

    pub fn new() -> Self {
        Self {
            lookup: [SymTy::default(); TWO_POW_MAX_LEN],
            min_code_len: 0,
        }
    }
    pub(crate) unsafe fn initialize_at(p: *mut Self) {
        let p_lookup = core::ptr::addr_of_mut!((*p).lookup);
        for i in 0..TWO_POW_MAX_LEN {
            (*p_lookup)[i] = SymTy::default();
        }
        (*p).min_code_len = 0;
    }
    pub fn reset(&mut self) {
        for i in 0..TWO_POW_MAX_LEN {
            self.lookup[i] = SymTy::default();
        }
        self.min_code_len = 0;
    }
}

trait SymTyTrait: Default + Copy + core::fmt::Debug + From<(usize, u8)> + Into<usize> + Eq {
    fn nbits(&self) -> u8;
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

#[derive(Clone)]
pub struct DecompressBuffered {
    coded_lengths_decoder: CanonicalHuffmanDecoder<CodedLenSym, 19, 7, 8, 128>,
    lit_decoder: CanonicalHuffmanDecoder<LitSym, 288, 15, 16, { 1 << 15 }>,
    dist_decoder: CanonicalHuffmanDecoder<DistSym, 30, 15, 16, { 1 << 15 }>,
}

impl DecompressBuffered {
    pub fn new() -> Self {
        Self {
            coded_lengths_decoder: CanonicalHuffmanDecoder::new(),
            lit_decoder: CanonicalHuffmanDecoder::new(),
            dist_decoder: CanonicalHuffmanDecoder::new(),
        }
    }
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = std::alloc::alloc(layout) as *mut Self;
            CanonicalHuffmanDecoder::initialize_at(core::ptr::addr_of_mut!(
                (*p).coded_lengths_decoder
            ));
            let p = std::alloc::alloc(layout) as *mut Self;
            CanonicalHuffmanDecoder::initialize_at(core::ptr::addr_of_mut!((*p).lit_decoder));
            let p = std::alloc::alloc(layout) as *mut Self;
            CanonicalHuffmanDecoder::initialize_at(core::ptr::addr_of_mut!((*p).dist_decoder));
            Box::from_raw(p)
        }
    }

    fn decompress<B>(&mut self, inp: &[u8], outp: &mut B) -> Result<(), DecompressError>
    where
        B: MaybeGrowableBuf,
    {
        let mut inp = inp.view_bits::<Lsb0>();

        loop {
            let bfinal = get_bits_num::<u8, 1>(&mut inp)?;
            let btype = get_bits_num::<u8, 2>(&mut inp)?;

            if btype == 3 {
                return Err(DecompressError::InvalidBlockType);
            } else if btype == 0 {
                let leftover_bits = inp.len() % 8;
                let _ = get_bits_slice(&mut inp, leftover_bits);

                // xxx efficiency???
                let len = get_bits_num::<u16, 16>(&mut inp)?;
                let nlen = get_bits_num::<u16, 16>(&mut inp)?;
                if len != !nlen {
                    return Err(DecompressError::BadNLen { len, nlen });
                }

                for _ in 0..len {
                    let b = get_bits_num::<u8, 8>(&mut inp)?;
                    outp.add_lit(b);
                }
            } else {
                self.lit_decoder.reset();
                self.dist_decoder.reset();

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
                    self.lit_decoder.init(&lit_lens).unwrap();
                    self.dist_decoder.init(&[5; 30]).unwrap();
                } else {
                    let hlit = get_bits_num::<u16, 5>(&mut inp)? + 257;
                    let hdist = get_bits_num::<u8, 5>(&mut inp)? + 1;
                    let hclen = get_bits_num::<u8, 4>(&mut inp)? + 4;
                    if hlit > 286 {
                        return Err(DecompressError::InvalidHLit);
                    }
                    if hdist > 30 {
                        return Err(DecompressError::InvalidHDist);
                    }

                    let mut code_len_lens = [0; CODE_LEN_ALPHABET_SIZE];
                    for i in 0..hclen {
                        let l = get_bits_num::<u8, 3>(&mut inp)?;
                        let idx = CODE_LEN_ORDER[i as usize];
                        code_len_lens[idx as usize] = l;
                    }

                    self.coded_lengths_decoder.init(&code_len_lens)?;

                    let mut actual_huff_code_lens = [0u8; 286 + 32];
                    let mut actual_huff_code_i = 0;
                    while actual_huff_code_i < (hlit + hdist as u16) as usize {
                        let code_len_sym = self.coded_lengths_decoder.read_sym(&mut inp)?.into();

                        match code_len_sym {
                            0..=15 => {
                                actual_huff_code_lens[actual_huff_code_i] = code_len_sym as u8;
                                actual_huff_code_i += 1;
                            }
                            16 => {
                                let rep = get_bits_num::<u8, 2>(&mut inp)? + 3;
                                if actual_huff_code_i == 0 {
                                    return Err(DecompressError::InvalidCodeLenRep);
                                }
                                let to_rep = actual_huff_code_lens[actual_huff_code_i - 1];
                                if actual_huff_code_i + rep as usize
                                    > (hlit + hdist as u16) as usize
                                {
                                    return Err(DecompressError::InvalidCodeLenRep);
                                }

                                for i in 0..rep as usize {
                                    actual_huff_code_lens[actual_huff_code_i + i] = to_rep;
                                }
                                actual_huff_code_i += rep as usize;
                            }
                            17 => {
                                let rep = get_bits_num::<u8, 3>(&mut inp)? + 3;
                                actual_huff_code_i += rep as usize;
                            }
                            18 => {
                                let rep = get_bits_num::<u8, 7>(&mut inp)? + 11;
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

                    self.lit_decoder.init(literal_code_lens)?;
                    if !(hdist == 1 && dist_code_lens[0] == 0) {
                        self.dist_decoder.init(dist_code_lens)?;
                    }
                }

                loop {
                    let litsym: usize = self.lit_decoder.read_sym(&mut inp)?.into();
                    match litsym {
                        0..=0xff => {
                            outp.add_lit(litsym as u8);
                        }
                        257..=285 => {
                            let len_extra_nbits = LEN_EXTRA_BITS[litsym - 257];
                            let len_extra = if len_extra_nbits > 0 {
                                get_bits_num_dyn::<u16>(&mut inp, len_extra_nbits as usize)?
                            } else {
                                0
                            };
                            let len = LEN_FOR_SYM[litsym - 257] + len_extra;

                            let distsym: usize = self.dist_decoder.read_sym(&mut inp)?.into();
                            if distsym > 29 {
                                return Err(DecompressError::BadHuffSym);
                            }
                            let dist_extra_nbits = DIST_EXTRA_BITS[distsym];
                            let dist_extra = if dist_extra_nbits > 0 {
                                get_bits_num_dyn::<u16>(&mut inp, dist_extra_nbits as usize)?
                            } else {
                                0
                            };
                            let dist = DIST_FOR_SYM[distsym] + dist_extra;

                            outp.add_match(dist as usize, len as usize).map_err(|_| {
                                DecompressError::BadLookback {
                                    disp: dist,
                                    avail: outp.cur_pos() as u16,
                                }
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

    pub fn decompress_into(&mut self, inp: &[u8], outp: &mut [u8]) -> Result<(), DecompressError> {
        self.decompress(inp, &mut FixedBuf::from(outp))
    }

    pub fn decompress_new(
        &mut self,
        inp: &[u8],
        max_sz: usize,
    ) -> Result<Vec<u8>, DecompressError> {
        let mut buf = VecBuf::new(0, max_sz);
        self.decompress(inp, &mut buf)?;
        Ok(buf.into())
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
    fn deflate_buffered_ref_decompress() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate.bin");
        let ref_fn = d.join("deflatetest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn deflate_buffered_ref_decompress_stored() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate-stored.bin");
        let ref_fn = d.join("deflatetest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }

    #[test]
    fn deflate_buffered_ref_decompress_fixed() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/deflate-fixed.bin");
        let ref_fn = d.join("deflatetest/tool.c");

        let inp = std::fs::read(inp_fn).unwrap();
        let ref_ = std::fs::read(ref_fn).unwrap();

        let mut dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, ref_);
    }
}
