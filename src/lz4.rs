use std::error::Error;

use crate::util::*;

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

            let mut nlits = (token >> 4) as usize;
            if nlits == 15 {
                loop {
                    let b = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                    nlits += b as usize;
                    if b != 0xff {
                        break;
                    }
                }
            }

            println!("nlits {}", nlits);
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

            let offset = get_inp::<2>(&mut inp).map_err(|_| DecompressError::Truncated)?;
            let offset = (offset[0] as usize) | (((offset[1]) as usize) << 8);
            println!("offset {}", offset);

            let mut matchlen = (token & 0xF) as usize + 4;
            if matchlen == 19 {
                loop {
                    let b = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                    matchlen += b as usize;
                    if b != 0xff {
                        break;
                    }
                }
            }
            println!("matchlen {}", matchlen);

            if offset == 0 {
                return Err(DecompressError::BadLookback {
                    disp: offset as u16,
                    avail: 0,
                });
            }

            outp.add_match(offset, matchlen)
                .map_err(|_| DecompressError::BadLookback {
                    disp: offset as u16,
                    avail: outp.cur_pos() as u16,
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
}
