struct SlidingWindowBuf<
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
    const MIN_MATCH: usize,
    const MAX_MATCH: usize,
    const HASH_BITS: u64,
> {
    buf: [u8; TOT_BUF_SZ],
    rpos: usize,
    wpos: usize,
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
            rpos: 0,
            wpos: 0,
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

    fn lookahead_valid_sz(&self) -> usize {
        if self.wpos >= self.rpos {
            self.wpos - self.rpos
        } else {
            self.wpos + TOT_BUF_SZ - self.rpos
        }
    }
}

// possible combinations:
// 1. all from inp, none from stored (1 span)
// 2. all from stored, no wraparound, none from inp (1 span)
// 3. all from stored, wraps around, none from inp (2 spans)
// 4. part from stored, no wraparound, part from inp (2 spans)
// 5. part from stored, wraps around, part from inp (3 spans)
struct SpanSet<'a>(&'a [u8], Option<&'a [u8]>, Option<&'a [u8]>);

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
    fn get_next_spans(&self, bytes: usize) -> SpanSet {
        let lookahead_valid_sz = self.buf.lookahead_valid_sz();
        if lookahead_valid_sz == 0 {
            // all from input
            SpanSet(&self.inp.unwrap()[..bytes], None, None)
        } else {
            let sz_from_internal_buf = if bytes <= lookahead_valid_sz {
                bytes
            } else {
                lookahead_valid_sz
            };
            let sz_from_external_buf = bytes - sz_from_internal_buf;
            let external_slice = if sz_from_external_buf != 0 {
                Some(&self.inp.unwrap()[..sz_from_external_buf])
            } else {
                None
            };

            if self.buf.rpos + sz_from_internal_buf <= TOT_BUF_SZ {
                // no wraparound
                SpanSet(
                    &self.buf.buf[self.buf.rpos..self.buf.rpos + sz_from_internal_buf],
                    external_slice,
                    None,
                )
            } else {
                // wraparound
                SpanSet(
                    &self.buf.buf[self.buf.rpos..TOT_BUF_SZ],
                    Some(&self.buf.buf[..self.buf.rpos + sz_from_internal_buf - TOT_BUF_SZ]),
                    external_slice,
                )
            }
        }
    }

    fn get_hash(&self) -> u64 {
        let mut tmp_buf = [0; MIN_MATCH];
        let lookahead_valid_sz = self.buf.lookahead_valid_sz();
        let bytes_to_hash = if lookahead_valid_sz >= MIN_MATCH {
            &self.buf.buf[self.buf.rpos..self.buf.rpos + MIN_MATCH]
        } else if lookahead_valid_sz == 0 {
            &self.inp.unwrap()[..MIN_MATCH]
        } else {
            tmp_buf[..lookahead_valid_sz]
                .copy_from_slice(&self.buf.buf[self.buf.rpos..self.buf.rpos + lookahead_valid_sz]);
            tmp_buf[lookahead_valid_sz..]
                .copy_from_slice(&self.inp.unwrap()[..MIN_MATCH - lookahead_valid_sz]);
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

    fn tot_ahead(&self) -> usize {
        self.buf.lookahead_valid_sz() + self.inp.map_or(0, |x| x.len())
    }

    fn roll_window(&mut self, bytes: usize) {
        let tot_ahead = self.tot_ahead();
        assert!(bytes <= tot_ahead);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_all_from_inp() {
        {
            let buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 1024 + 256 - 1;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
    }

    #[test]
    fn span_all_from_buf_no_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.wpos = 8;
            let win = buf.flush();
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[123..123 + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let win = buf.flush();
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.flush();
            let spans = win.get_next_spans(4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
    }

    #[test]
    fn span_all_from_buf_yes_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.flush();
            let spans = win.get_next_spans(8);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, Some(&[5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 1] = 1;
            buf.buf[..7].copy_from_slice(&[2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 7;
            let win = buf.flush();
            let spans = win.get_next_spans(8);
            assert_eq!(spans.0, [1]);
            assert_eq!(spans.1, Some(&[2, 3, 4, 5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 7..].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
            buf.buf[0] = 8;
            buf.rpos = 1024 + 256 - 7;
            buf.wpos = 1;
            let win = buf.flush();
            let spans = win.get_next_spans(8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(spans.1, Some(&[8][..]));
            assert_eq!(spans.2, None)
        }
    }

    #[test]
    fn span_straddle_no_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[0..3].copy_from_slice(&[1, 2, 3]);
            buf.wpos = 3;
            let win = buf.add_inp(&[4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(8);
            assert_eq!(spans.0, [1, 2, 3]);
            assert_eq!(spans.1, Some(&[4, 5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
                SlidingWindowBuf::new();
            buf.buf[123..123 + 5].copy_from_slice(&[1, 2, 3, 4, 5]);
            buf.rpos = 123;
            buf.wpos = 123 + 5;
            let win = buf.add_inp(&[6, 7, 8]);
            let spans = win.get_next_spans(8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5]);
            assert_eq!(spans.1, Some(&[6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
    }

    #[test]
    fn span_straddle_yes_wrap() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
            SlidingWindowBuf::new();
        buf.buf[1024 + 256 - 3..].copy_from_slice(&[1, 2, 3]);
        buf.buf[..2].copy_from_slice(&[4, 5]);
        buf.rpos = 1024 + 256 - 3;
        buf.wpos = 2;
        let win = buf.add_inp(&[6, 7, 8]);
        let spans = win.get_next_spans(8);
        assert_eq!(spans.0, [1, 2, 3]);
        assert_eq!(spans.1, Some(&[4, 5][..]));
        assert_eq!(spans.2, Some(&[6, 7, 8][..]));
    }

    #[test]
    fn hash_all_buf() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
            SlidingWindowBuf::new();
        buf.buf[0] = 0x12;
        buf.buf[1] = 0x34;
        buf.buf[2] = 0x56;
        buf.wpos = 3;
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
        buf.buf[0] = 0xde;
        buf.wpos = 1;
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

    #[test]
    fn hash_advanced_rpos() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }, 3, 512, 15> =
            SlidingWindowBuf::new();
        buf.buf[0] = 0x12;
        buf.buf[1] = 0x34;
        buf.buf[2] = 0x56;
        buf.wpos = 3;
        let hash = buf.flush().get_hash();
        assert_eq!(hash, ((0x12 << 10) ^ (0x34 << 5) ^ 0x56) & 0x7FFF);
    }
}
