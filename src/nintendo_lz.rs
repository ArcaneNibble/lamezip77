//! Handles LZ77 as used in the GBA/NDS BIOS.

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box};

use crate::decompress::StreamingOutputBuf;
use crate::{
    decompress::{InputPeeker, LZOutputBuf, StreamingDecompressState},
    LZEngine, LZOutput, LZSettings,
};

/// Possible decompression errors
#[derive(Debug, PartialEq, Eq)]
pub enum DecompressError {
    // Bad magic byte at beginning (not 0x10)
    BadMagic(u8),
    /// Tried to encode a match past the beginning of the buffer
    BadLookback {
        disp: u16,
        avail: u16,
    },
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
        }
    }
}

#[cfg(feature = "std")]
impl Error for DecompressError {}

const LOOKBACK_SZ: usize = 0x1000;

/// Construct a decompressor.
///
/// `decompress_make(ident, impl LZOutputBuf, optional crate path)`
pub use lamezip77_macros::nintendo_lz_decompress_make as decompress_make;

/// Internal decompression function. Most users should use the [decompress_make] macro instead.
pub async fn decompress_impl<O>(
    outp: &mut O,
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

        for i in 0..8 {
            if outp.cur_pos() == encoded_len || outp.is_at_limit() {
                break;
            }

            if flags & (1 << (7 - i)) == 0 {
                let b = (&peek1).await[0];
                outp.add_lits(&[b]);
            } else {
                let matchb = u16::from_be_bytes((&peek2).await);
                let disp = (matchb & 0xFFF) as usize;
                let len = (matchb >> 12) as usize + 3;

                let len = core::cmp::min(len, encoded_len - outp.cur_pos());

                outp.add_match(disp, len)
                    .map_err(|_| DecompressError::BadLookback {
                        disp: disp as u16,
                        avail: outp.cur_pos() as u16,
                    })?;
            }
        }
    }

    Ok(())
}

/// Type of a decompressor state.
pub type Decompress<'a, F> = StreamingDecompressState<'a, F, DecompressError, 4>;
/// [StreamingOutputBuf] with buffer size for this format.
pub type DecompressBuffer<O> = StreamingOutputBuf<O, LOOKBACK_SZ>;

/// Compressor for Nintendo LZ77
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
                            flags |= 1 << (7 - i);
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

                            let matchb = ((len << 12) as u16) | (disp as u16);
                            let matchb = matchb.to_be_bytes();
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

    use super::*;

    #[test]
    fn nin_all_lits() {
        let mut outvec = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| outvec.extend_from_slice(x), usize::MAX);
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10]);
            assert!(ret.is_ok() && ret.unwrap() != 0);

            let ret = dec.add_inp(&[10]);
            assert!(ret.is_ok() && ret.unwrap() != 0);
            let ret = dec.add_inp(&[0]);
            assert!(ret.is_ok() && ret.unwrap() != 0);
            let ret = dec.add_inp(&[0]);
            assert!(ret.is_ok() && ret.unwrap() != 0);

            let ret = dec.add_inp(&[0]);
            assert!(ret.is_ok() && ret.unwrap() != 0);

            for i in 0..8 {
                let ret = dec.add_inp(&[i]);
                assert!(ret.is_ok() && ret.unwrap() != 0);
            }

            let ret = dec.add_inp(&[0]);
            assert!(ret.is_ok() && ret.unwrap() != 0);
            let ret = dec.add_inp(&[8]);
            assert!(ret.is_ok() && ret.unwrap() != 0);
            let ret = dec.add_inp(&[9]);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }

        assert_eq!(outvec, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn nin_disp_repeating() {
        let mut outvec = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| outvec.extend_from_slice(x), usize::MAX);
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b0110_0000, 1]);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        assert_eq!(
            outvec,
            &[0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
        );
    }

    #[test]
    fn nin_disp_repeating_overlong() {
        let mut outvec = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| outvec.extend_from_slice(x), usize::MAX);
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b1111_0000, 1]);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        assert_eq!(
            outvec,
            &[0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
        );
    }

    #[test]
    fn nin_disp_non_repeating() {
        let mut outvec = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| outvec.extend_from_slice(x), usize::MAX);
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 2, 5, 6]);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        assert_eq!(outvec, &[1, 2, 3, 4, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn nin_disp_invalid() {
        let mut outvec = Vec::new();
        {
            let mut outp = DecompressBuffer::new(|x| outvec.extend_from_slice(x), usize::MAX);
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 4, 5, 6]);
            assert_eq!(ret, Err(DecompressError::BadLookback { disp: 4, avail: 4 }));
        }
        assert_eq!(outvec, &[1, 2, 3, 4]);
    }

    #[test]
    fn nin_buffered_short() {
        let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
        decompress_make!(dec, &mut outp, crate);

        let ret = dec.add_inp(&[0x10, 10]);
        assert!(ret.is_ok() && ret.unwrap() != 0);

        let ret = dec.add_inp(&[0x10, 10, 0, 0]);
        assert!(ret.is_ok() && ret.unwrap() != 0);

        let ret = dec.add_inp(&[0x10, 10, 0, 0, 0]);
        assert!(ret.is_ok() && ret.unwrap() != 0);
    }

    #[test]
    fn nin_buffered_lits() {
        let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
        {
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 10, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11]);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn nin_buffered_disp_repeating() {
        {
            let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
            {
                decompress_make!(dec, &mut outp, crate);

                let ret = dec.add_inp(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b0110_0000, 1]);
                assert!(ret.is_ok() && ret.unwrap() == 0);
            }
            let outvec: Vec<_> = outp.into();
            assert_eq!(
                outvec,
                vec![0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
            );
        }

        {
            let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
            {
                decompress_make!(dec, &mut outp, crate);

                let ret = dec.add_inp(&[0x10, 11, 0, 0, 0b00100000, 0xaa, 0xbb, 0b1111_0000, 1]);
                assert!(ret.is_ok() && ret.unwrap() == 0);
            }
            let outvec: Vec<_> = outp.into();
            assert_eq!(
                outvec,
                vec![0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb, 0xaa]
            );
        }
    }

    #[test]
    fn nin_buffered_disp_invalid() {
        let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
        {
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000, 4, 5, 6]);
            assert_eq!(ret, Err(DecompressError::BadLookback { disp: 4, avail: 4 }));
        }
        let outvec: Vec<_> = outp.into();
        assert_eq!(outvec, vec![1, 2, 3, 4]);
    }

    #[test]
    fn nin_buffered_disp_truncated() {
        let mut outp = crate::decompress::VecBuf::new(0, usize::MAX);
        {
            decompress_make!(dec, &mut outp, crate);

            let ret = dec.add_inp(&[0x10, 9, 0, 0, 0b00001000, 1, 2, 3, 4, 0b0000_0000]);
            assert!(ret.is_ok() && ret.unwrap() != 0);
        }
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

        let mut decompress = crate::decompress::VecBuf::new(0, usize::MAX);
        {
            decompress_make!(dec, &mut decompress, crate);
            let ret = dec.add_inp(&compressed_out);
            assert!(ret.is_ok() && ret.unwrap() == 0);
        }
        let decompress: Vec<_> = decompress.into();

        assert_eq!(inp, decompress);
    }
}
