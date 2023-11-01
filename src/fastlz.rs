use std::error::Error;

use crate::util::*;

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
    use super::*;

    #[test]
    fn flz_ref_decompress_1() {
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
    fn flz_zeros_decompress_1() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv1.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, vec![0; 256 * 1024]);
    }

    #[test]
    fn flz_ref_decompress_2() {
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
    fn flz_zeros_decompress_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("tests/fastlz-zeros.lv2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let dec = DecompressBuffered::new();
        let out = dec.decompress_new(&inp, usize::MAX).unwrap();

        assert_eq!(out, vec![0; 256 * 1024]);
    }
}
