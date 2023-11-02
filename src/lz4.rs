use std::error::Error;

use crate::util::*;

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
                    let nlit = b >> 4;
                    let matchlen = b & 0xF;

                    self.nlit = nlit as usize;
                    self.matchlen = matchlen as usize + 4;

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
}
