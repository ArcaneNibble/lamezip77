use std::error::Error;

use crate::util::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecompressState {
    Opcode0Start,
    Opcode0Lv1,
    OpcodeLv1MoreLen,
    OpcodeLv1MoreDisp,
    LiteralRunLv1(u8),
    Opcode0Lv2,
    OpcodeLv2MoreLen,
    OpcodeLv2MoreDisp0,
    OpcodeLv2MoreDisp1,
    OpcodeLv2MoreDisp2,
    LiteralRunLv2(u8),
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

impl Error for DecompressError {}

// max size for level 2 (level 1 is smaller)
const LOOKBACK_SZ: usize = 0xFFFF + 0x1FFF + 1;

// technically copy-able, but don't want to make it easy to accidentally do so
#[derive(Clone)]
pub struct DecompressStreaming {
    lookback: [u8; LOOKBACK_SZ],
    lookback_wptr: usize,
    lookback_avail: usize,
    state: DecompressState,
    matchlen: usize,
    matchdisp: usize,
}

impl DecompressStreaming {
    pub fn new() -> Self {
        Self {
            lookback: [0; LOOKBACK_SZ],
            lookback_wptr: 0,
            lookback_avail: 0,
            state: DecompressState::Opcode0Start,
            matchlen: 0,
            matchdisp: 0,
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
            (*p).state = DecompressState::Opcode0Start;
            (*p).matchlen = 0;
            (*p).matchdisp = 0;
            Box::from_raw(p)
        }
    }

    fn do_match<O>(&mut self, mut outp: O) -> Result<(), DecompressError>
    where
        O: FnMut(u8),
    {
        println!("match disp -{} len {}", self.matchdisp + 1, self.matchlen);

        if self.matchdisp + 1 > self.lookback_avail {
            return Err(DecompressError::BadLookback {
                disp: self.matchdisp as u32,
                avail: self.lookback_avail as u32,
            });
        }

        for _ in 0..self.matchlen {
            let idx = (self.lookback_wptr + LOOKBACK_SZ - self.matchdisp - 1) % LOOKBACK_SZ;
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
            println!("state {:?} byte {:08b}", self.state, b);

            match self.state {
                DecompressState::Opcode0Start => {
                    let level = b >> 5;
                    let nlit = (b & 0b11111) + 1;
                    println!("lits: {}", nlit);

                    if level == 0 {
                        self.state = DecompressState::LiteralRunLv1(nlit);
                    } else if level == 1 {
                        self.state = DecompressState::LiteralRunLv2(nlit);
                    } else {
                        return Err(DecompressError::BadCompressionLevel(level));
                    }
                }
                DecompressState::Opcode0Lv1 => {
                    let matchlen = (b >> 5) as usize + 2;
                    let matchdisp = ((b & 0b11111) as usize) << 8;

                    println!("matchlen {} matchdisp {}", matchlen, matchdisp);
                    self.matchlen = matchlen;
                    self.matchdisp = matchdisp;

                    if matchlen == 0b000 + 2 {
                        let nlit = (b & 0b11111) + 1;
                        println!("lits: {}", nlit);
                        self.state = DecompressState::LiteralRunLv1(nlit);
                    } else if matchlen == 0b111 + 2 {
                        self.state = DecompressState::OpcodeLv1MoreLen;
                    } else {
                        self.state = DecompressState::OpcodeLv1MoreDisp;
                    }
                }
                DecompressState::OpcodeLv1MoreLen => {
                    self.matchlen += b as usize;
                    println!("matchlen {}", self.matchlen);
                    self.state = DecompressState::OpcodeLv1MoreDisp;
                }
                DecompressState::OpcodeLv1MoreDisp => {
                    self.matchdisp |= b as usize;
                    println!("matchdisp {}", self.matchdisp);

                    self.do_match(&mut outp)?;
                    self.state = DecompressState::Opcode0Lv1;
                }
                DecompressState::Opcode0Lv2 => {
                    let matchlen = (b >> 5) as usize + 2;
                    let matchdisp = ((b & 0b11111) as usize) << 8;

                    println!("matchlen {} matchdisp {}", matchlen, matchdisp);
                    self.matchlen = matchlen;
                    self.matchdisp = matchdisp;

                    if matchlen == 0b000 + 2 {
                        let nlit = (b & 0b11111) + 1;
                        println!("lits: {}", nlit);
                        self.state = DecompressState::LiteralRunLv2(nlit);
                    } else if matchlen == 0b111 + 2 {
                        self.state = DecompressState::OpcodeLv2MoreLen;
                    } else {
                        self.state = DecompressState::OpcodeLv2MoreDisp0;
                    }
                }
                DecompressState::OpcodeLv2MoreLen => {
                    self.matchlen += b as usize;
                    println!("matchlen {}", self.matchlen);

                    if b != 0xff {
                        self.state = DecompressState::OpcodeLv2MoreDisp0;
                    }
                }
                DecompressState::OpcodeLv2MoreDisp0 => {
                    self.matchdisp |= b as usize;
                    println!("matchdisp {}", self.matchdisp);

                    if self.matchdisp == 0b11111_11111111 {
                        self.state = DecompressState::OpcodeLv2MoreDisp1;
                    } else {
                        self.do_match(&mut outp)?;
                        self.state = DecompressState::Opcode0Lv2;
                    }
                }
                DecompressState::OpcodeLv2MoreDisp1 => {
                    self.matchdisp += (b as usize) << 8;
                    println!("matchdisp {}", self.matchdisp);
                    self.state = DecompressState::OpcodeLv2MoreDisp2;
                }
                DecompressState::OpcodeLv2MoreDisp2 => {
                    self.matchdisp += b as usize;
                    println!("matchdisp {}", self.matchdisp);
                    self.do_match(&mut outp)?;
                    self.state = DecompressState::Opcode0Lv2;
                }
                DecompressState::LiteralRunLv1(nlit) | DecompressState::LiteralRunLv2(nlit) => {
                    outp(b);
                    self.lookback[self.lookback_wptr] = b;
                    self.lookback_wptr = (self.lookback_wptr + 1) % LOOKBACK_SZ;
                    if self.lookback_avail < LOOKBACK_SZ {
                        self.lookback_avail += 1;
                    }

                    match self.state {
                        DecompressState::LiteralRunLv1(_) => {
                            if nlit != 1 {
                                self.state = DecompressState::LiteralRunLv1(nlit - 1);
                            } else {
                                self.state = DecompressState::Opcode0Lv1;
                            }
                        }
                        DecompressState::LiteralRunLv2(_) => {
                            if nlit != 1 {
                                self.state = DecompressState::LiteralRunLv2(nlit - 1);
                            } else {
                                self.state = DecompressState::Opcode0Lv2;
                            }
                        }
                        _ => unreachable!(),
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
        println!("decompress lv1");

        let mut first_opc = true;

        while inp.len() > 0 {
            let opc0 = get_inp::<1>(&mut inp).unwrap()[0];
            println!("opc0 {:08b}", opc0);

            if first_opc || opc0 >> 5 == 0 {
                first_opc = false;
                // literal run
                let nlit = (opc0 & 0b11111) + 1;
                println!("lits: {}", nlit);
                for _ in 0..nlit {
                    // XXX efficiency?
                    if outp.cur_pos() < outp.limit() {
                        outp.add_lit(
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0],
                        );
                    }
                }
            } else {
                let matchlen = if (opc0 >> 5) != 0b111 {
                    ((opc0 >> 5) + 2) as usize
                } else {
                    let opc1 = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                    println!("opc1 {}", opc1);
                    (opc1 as usize) + 9
                };

                let opc12 = get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                println!("opc12 {}", opc12);

                let matchdisp = (((opc0 as usize) & 0b11111) << 8) | (opc12 as usize);

                println!("match disp -{} len {}", matchdisp + 1, matchlen);
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
        println!("decompress lv1");

        let mut first_opc = true;

        while inp.len() > 0 {
            let opc0 = get_inp::<1>(&mut inp).unwrap()[0];
            println!("opc0 {:08b}", opc0);

            if first_opc || opc0 >> 5 == 0 {
                first_opc = false;
                // literal run
                let nlit = (opc0 & 0b11111) + 1;
                println!("lits: {}", nlit);
                for _ in 0..nlit {
                    // XXX efficiency?
                    if outp.cur_pos() < outp.limit() {
                        outp.add_lit(
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0],
                        );
                    }
                }
            } else {
                let mut matchlen = ((opc0 >> 5) + 2) as usize;
                if matchlen == 0b111 + 2 {
                    loop {
                        let morelen =
                            get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                        println!("more len {}", morelen);
                        matchlen += morelen as usize;
                        if morelen != 0xff {
                            break;
                        }
                    }
                }

                let opc_dispnext =
                    get_inp::<1>(&mut inp).map_err(|_| DecompressError::Truncated)?[0];
                println!("opc_dispnext {}", opc_dispnext);

                let mut matchdisp = (((opc0 as usize) & 0b11111) << 8) | (opc_dispnext as usize);
                if matchdisp == 0b11111_11111111 {
                    let moredisp =
                        get_inp::<2>(&mut inp).map_err(|_| DecompressError::Truncated)?;
                    println!("more disp {:?}", moredisp);
                    let moredisp = ((moredisp[0] as usize) << 8) | (moredisp[1] as usize);
                    println!("more disp {:?}", moredisp);
                    matchdisp += moredisp;
                }

                println!("match disp -{} len {}", matchdisp + 1, matchlen);
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
}
