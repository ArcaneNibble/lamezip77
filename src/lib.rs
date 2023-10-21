struct SlidingWindowBuf<
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
    const MIN_MATCH: usize,
    const MAX_MATCH: usize,
    const HASH_BITS: u64,
> {
    buf: [u8; TOT_BUF_SZ],
    lookahead_valid_sz: usize,
}

impl<
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
        const MIN_MATCH: usize,
        const MAX_MATCH: usize,
        const HASH_BITS: u64,
    > SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ, MIN_MATCH, MAX_MATCH, HASH_BITS>
{
    fn new() -> Self {
        assert!(TOT_BUF_SZ == LOOKBACK_SZ + LOOKAHEAD_SZ);

        Self {
            buf: [0; TOT_BUF_SZ],
            lookahead_valid_sz: 0,
        }
    }

    fn add_inp<'a>(
        &'a self,
        inp: &'a [u8],
    ) -> SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ, MIN_MATCH, MAX_MATCH, HASH_BITS>
    {
        SlidingWindow {
            buf: self,
            inp: Some(inp),
        }
    }

    fn flush(
        &self,
    ) -> SlidingWindow<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ, MIN_MATCH, MAX_MATCH, HASH_BITS> {
        SlidingWindow {
            buf: self,
            inp: None,
        }
    }
}

struct SlidingWindow<
    'a,
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
    const MIN_MATCH: usize,
    const MAX_MATCH: usize,
    const HASH_BITS: u64,
> {
    buf: &'a SlidingWindowBuf<
        LOOKBACK_SZ,
        LOOKAHEAD_SZ,
        TOT_BUF_SZ,
        MIN_MATCH,
        MAX_MATCH,
        HASH_BITS,
    >,
    inp: Option<&'a [u8]>,
}

impl<
        'a,
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
        const MIN_MATCH: usize,
        const MAX_MATCH: usize,
        const HASH_BITS: u64,
    > SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ, MIN_MATCH, MAX_MATCH, HASH_BITS>
{
    fn get_hash(&self) -> u64 {
        let mut tmp_buf = [0; MIN_MATCH];
        let bytes_to_hash = if self.buf.lookahead_valid_sz >= MIN_MATCH {
            &self.buf.buf[LOOKBACK_SZ..LOOKBACK_SZ + MIN_MATCH]
        } else if self.buf.lookahead_valid_sz == 0 {
            &self.inp.unwrap()[..MIN_MATCH]
        } else {
            tmp_buf[..self.buf.lookahead_valid_sz].copy_from_slice(
                &self.buf.buf[LOOKBACK_SZ..LOOKBACK_SZ + self.buf.lookahead_valid_sz],
            );
            tmp_buf[self.buf.lookahead_valid_sz..]
                .copy_from_slice(&self.inp.unwrap()[..MIN_MATCH - self.buf.lookahead_valid_sz]);
            &tmp_buf
        };

        assert!(bytes_to_hash.len() == MIN_MATCH);
        let hash_shift: u64 = (HASH_BITS + (MIN_MATCH as u64) - 1) / (MIN_MATCH as u64);

        let mut hash = 0;
        for i in 0..MIN_MATCH {
            hash = (hash << hash_shift) ^ (bytes_to_hash[i] as u64);
        }

        hash & ((1 << HASH_BITS) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_all_buf() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
            SlidingWindowBuf::new();
        buf.buf[1024] = 0x12;
        buf.buf[1025] = 0x34;
        buf.buf[1026] = 0x56;
        buf.lookahead_valid_sz = 3;
        let hash = buf.flush().get_hash();
        assert_eq!(hash, ((0x12 << 10) ^ (0x34 << 5) ^ 0x56) & 0x7FFF);
    }

    #[test]
    fn hash_all_inp() {
        let buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> = SlidingWindowBuf::new();
        let hash = buf.add_inp(&[0x78, 0x9a, 0xbc]).get_hash();
        assert_eq!(hash, ((0x78 << 10) ^ (0x9a << 5) ^ 0xbc) & 0x7FFF);
    }

    #[test]
    fn hash_split() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
            SlidingWindowBuf::new();
        buf.buf[1024] = 0xde;
        buf.lookahead_valid_sz = 1;
        let hash = buf.add_inp(&[0xf0, 0x12]).get_hash();
        assert_eq!(hash, ((0xde << 10) ^ (0xf0 << 5) ^ 0x12) & 0x7FFF);
    }

    #[test]
    fn hash_longer_min_match() {
        let buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 4, 512, 15> = SlidingWindowBuf::new();
        let hash = buf.add_inp(&[0xde, 0xad, 0xfa, 0xce]).get_hash();
        assert_eq!(
            hash,
            ((0xde << 12) ^ (0xad << 8) ^ (0xfa << 4) ^ 0xce) & 0x7FFF
        );
    }
}
