use std::error::Error;

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

impl Error for DecompressError {}

const LOOKBACK_SZ: usize = 0x1000;

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
            (*p).state = DecompressState::HdrMagic;
            (*p).remaining_bytes = 0;
            (*p).block_flags = 0;
            (*p).block_i = 0;
            (*p).match_b0 = 0;
            Box::from_raw(p)
        }
    }

    const fn need_more_bytes(&self) -> bool {
        self.state.is_header_state() || self.remaining_bytes > 0
    }

    pub fn decompress<O>(&mut self, mut inp: &[u8], mut outp: O) -> Result<bool, DecompressError>
    where
        O: FnMut(u8),
    {
        while inp.len() > 0 && self.need_more_bytes() {
            match self.state {
                DecompressState::HdrMagic => {
                    let magic = inp[0];
                    inp = &inp[1..];
                    if magic != 0x10 {
                        return Err(DecompressError::BadMagic(magic));
                    }
                    self.state = DecompressState::HdrLen0;
                }
                DecompressState::HdrLen0 => {
                    let b = inp[0];
                    inp = &inp[1..];
                    self.remaining_bytes = b as u32;
                    self.state = DecompressState::HdrLen1;
                }
                DecompressState::HdrLen1 => {
                    let b = inp[0];
                    inp = &inp[1..];
                    self.remaining_bytes = self.remaining_bytes | ((b as u32) << 8);
                    self.state = DecompressState::HdrLen2;
                }
                DecompressState::HdrLen2 => {
                    let b = inp[0];
                    inp = &inp[1..];
                    self.remaining_bytes = self.remaining_bytes | ((b as u32) << 16);
                    self.state = DecompressState::BlockFlags;
                }
                DecompressState::BlockFlags => {
                    let b = inp[0];
                    inp = &inp[1..];
                    self.block_flags = b;
                    self.block_i = 0;
                    self.state = DecompressState::BlockData0;
                }
                DecompressState::BlockData0 => {
                    if self.block_flags & (1 << (7 - self.block_i)) == 0 {
                        let lit = inp[0];
                        inp = &inp[1..];
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
                        let b = inp[0];
                        inp = &inp[1..];
                        self.match_b0 = b;
                        self.state = DecompressState::BlockData1;
                    }
                }
                DecompressState::BlockData1 => {
                    let b = inp[0];
                    inp = &inp[1..];

                    let disp = (b as usize) | (((self.match_b0 & 0xF) as usize) << 8);
                    let len = (self.match_b0 >> 4) as usize;

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

    pub fn decompress_into(&self, mut inp: &[u8], outp: &mut [u8]) -> Result<(), DecompressError> {
        if inp.len() < 4 {
            return Err(DecompressError::TooShort);
        }
        let encoded_len = (inp[1] as usize) | ((inp[2] as usize) << 8) | ((inp[3] as usize) << 16);
        let wanted_len = core::cmp::min(encoded_len, outp.len());
        let mut outpos = 0;

        inp = &inp[4..];

        while outpos < wanted_len {
            if inp.len() == 0 {
                return Err(DecompressError::TooShort);
            }

            let flags = inp[0];
            inp = &inp[1..];

            for i in (0..8).rev() {
                if outpos == wanted_len {
                    break;
                }

                if flags & (1 << i) == 0 {
                    if inp.len() == 0 {
                        return Err(DecompressError::TooShort);
                    }
                    let b = inp[0];
                    inp = &inp[1..];
                    outp[outpos] = b;
                    outpos += 1;
                } else {
                    if inp.len() < 2 {
                        return Err(DecompressError::TooShort);
                    }
                    let disp = (inp[1] as usize) | (((inp[0] & 0xF) as usize) << 8);
                    let len = (inp[0] >> 4) as usize;
                    inp = &inp[2..];

                    if disp + 1 > outpos {
                        return Err(DecompressError::BadLookback {
                            disp: disp as u16,
                            avail: outpos as u16,
                        });
                    }

                    let len = core::cmp::min(len + 3, wanted_len - outpos);
                    for j in 0..len {
                        outp[outpos + j] = outp[outpos - disp - 1 + j];
                    }
                    outpos += len;
                }
            }
        }

        Ok(())
    }

    pub fn decompress_new(&self, inp: &[u8]) -> Result<Vec<u8>, DecompressError> {
        if inp.len() < 4 {
            return Err(DecompressError::TooShort);
        }
        let encoded_len = (inp[1] as usize) | ((inp[2] as usize) << 8) | ((inp[3] as usize) << 16);
        let mut ret = vec![0; encoded_len];
        self.decompress_into(inp, &mut ret[..])?;
        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
