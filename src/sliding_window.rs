// possible combinations:
// 1. all from inp, none from stored (1 span)
// 2. all from stored, no wraparound, none from inp (1 span)
// 3. all from stored, wraps around, none from inp (2 spans)
// 4. part from stored, no wraparound, part from inp (2 spans)
// 5. part from stored, wraps around, part from inp (3 spans)
pub struct SpanSet<'a>(&'a [u8], &'a [u8], &'a [u8]);

impl<'a> core::ops::Index<usize> for SpanSet<'a> {
    type Output = u8;

    fn index(&self, index: usize) -> &u8 {
        if index < self.0.len() {
            &self.0[index]
        } else {
            let index = index - self.0.len();
            let _1 = self.1;
            if index < _1.len() {
                &_1[index]
            } else {
                let index = index - _1.len();
                let _2 = self.2;
                &_2[index]
            }
        }
    }
}

impl<'a> SpanSet<'a> {
    pub fn len(&self) -> usize {
        self.0.len() + self.1.len() + self.2.len()
    }

    pub fn compare(&self, other: &Self) -> usize {
        // todo optimizations? or can the compiler optimize this?
        let mut len = 0;
        let mut a = self.0;
        let mut a_which = 0;
        let mut b = other.0;
        let mut b_which = 0;

        loop {
            while a.len() == 0 {
                match a_which {
                    0 => {
                        a = self.1;
                        a_which = 1;
                    }
                    1 => {
                        a = self.2;
                        a_which = 2;
                    }
                    _ => break,
                }
            }
            while b.len() == 0 {
                match b_which {
                    0 => {
                        b = other.1;
                        b_which = 1;
                    }
                    1 => {
                        b = other.2;
                        b_which = 2;
                    }
                    _ => break,
                }
            }

            let matching = a.iter().zip(b).take_while(|(x, y)| x == y).count();
            if matching == 0 {
                break;
            }
            len += matching;
            a = &a[matching..];
            b = &b[matching..];
        }

        len
    }
}

#[cfg(test)]
mod spanset_tests {
    use super::*;

    #[test]
    fn span_set_compare() {
        let a = SpanSet(&[1, 2, 3, 4, 5, 6, 7, 8], &[], &[]);

        let b = SpanSet(&[1, 2, 3, 4, 5, 6, 7], &[], &[]);
        assert_eq!(a.compare(&b), 7);

        let b = SpanSet(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[], &[]);
        assert_eq!(a.compare(&b), 8);

        let b = SpanSet(&[1, 2, 3, 4, 4, 6, 7], &[], &[]);
        assert_eq!(a.compare(&b), 4);

        let b = SpanSet(&[1, 2, 3], &[4, 5, 6, 7], &[]);
        assert_eq!(a.compare(&b), 7);

        let b = SpanSet(&[1, 2, 3], &[4, 5, 5, 7], &[]);
        assert_eq!(a.compare(&b), 5);

        let b = SpanSet(&[1, 2, 3], &[4], &[5, 6]);
        assert_eq!(a.compare(&b), 6);

        let a = SpanSet(&[1, 2], &[3, 4, 5, 6], &[7, 8]);
        let b = SpanSet(&[1, 2, 3], &[4, 5], &[6, 7, 8, 9]);
        assert_eq!(a.compare(&b), 8);
    }

    #[test]
    fn span_index() {
        let a = SpanSet(&[1, 2], &[3, 4, 5, 6], &[7, 8]);

        for i in 0..8 {
            assert_eq!(a[i], (i + 1) as u8);
        }
    }
}

pub struct SlidingWindowBuf<
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
> {
    buf: [u8; TOT_BUF_SZ],
    rpos: usize,
    wpos: usize,
    // this is the offset that the current read position
    // corresponds to in the entirety of the data stream.
    // it is used to map hashtable values to a position
    // in this buffer when matches are being compared
    rpos_real_offs: u64,
}

impl<const LOOKBACK_SZ: usize, const LOOKAHEAD_SZ: usize, const TOT_BUF_SZ: usize>
    SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>
{
    pub fn new() -> Self {
        // this is needed so that SlidingWindow can distinguish between
        // completely empty and completely full
        assert!(LOOKBACK_SZ > 0);
        assert!(TOT_BUF_SZ == LOOKBACK_SZ + LOOKAHEAD_SZ);

        Self {
            buf: [0; TOT_BUF_SZ],
            rpos: 0,
            wpos: 0,
            rpos_real_offs: 0,
        }
    }
    pub unsafe fn initialize_at(p: *mut Self) {
        assert!(LOOKBACK_SZ > 0);
        assert!(TOT_BUF_SZ == LOOKBACK_SZ + LOOKAHEAD_SZ);

        let p_buf = core::ptr::addr_of_mut!((*p).buf);
        for i in 0..TOT_BUF_SZ {
            (*p_buf)[i] = 0;
        }
        (*p).rpos = 0;
        (*p).wpos = 0;
        (*p).rpos_real_offs = 0;
    }

    pub fn add_inp<'a>(
        &'a mut self,
        inp: &'a [u8],
    ) -> SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ> {
        SlidingWindow { buf: self, inp }
    }

    pub fn lookahead_valid_sz(&self) -> usize {
        if self.wpos >= self.rpos {
            self.wpos - self.rpos
        } else {
            self.wpos + TOT_BUF_SZ - self.rpos
        }
    }
}

pub struct SlidingWindow<
    'a,
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
> {
    buf: &'a mut SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    inp: &'a [u8],
}

impl<'a, const LOOKBACK_SZ: usize, const LOOKAHEAD_SZ: usize, const TOT_BUF_SZ: usize>
    SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>
{
    pub fn get_next_spans(&self, from_pos: u64, bytes: usize) -> SpanSet {
        assert!(from_pos <= self.buf.rpos_real_offs);
        // assert!(from_pos >= self.buf.rpos_real_offs - LOOKBACK_SZ);
        assert!(from_pos + LOOKBACK_SZ as u64 >= self.buf.rpos_real_offs); // won't overflow when subtracting
        let dist_to_look_back = (self.buf.rpos_real_offs - from_pos) as usize; // <= LOOKBACK_SZ, >= 0
        let dist_to_look_forward = if bytes > dist_to_look_back {
            let tot_ahead_sz = self.tot_ahead_sz();
            let ideal_dist_to_look_forward = bytes - dist_to_look_back;
            // clamp to end of all inputs (instead of assert)
            if ideal_dist_to_look_forward <= tot_ahead_sz {
                ideal_dist_to_look_forward
            } else {
                tot_ahead_sz
            }
        } else {
            0
        };

        let lookahead_valid_sz = self.buf.lookahead_valid_sz();

        if dist_to_look_back == 0 && lookahead_valid_sz == 0 {
            // all from input
            SpanSet(&self.inp[..dist_to_look_forward], &[], &[])
        } else {
            let (sz_from_internal_buf, sz_from_external_buf) =
                if dist_to_look_forward <= lookahead_valid_sz {
                    (
                        // always from the internal buf, but how much?
                        // limited to bytes
                        core::cmp::min(dist_to_look_back + dist_to_look_forward, bytes),
                        0,
                    )
                } else {
                    (
                        dist_to_look_back + lookahead_valid_sz,
                        dist_to_look_forward - lookahead_valid_sz,
                    )
                };
            let external_slice = if sz_from_external_buf != 0 {
                &self.inp[..sz_from_external_buf]
            } else {
                &[]
            };

            let actual_rpos = (self.buf.rpos + TOT_BUF_SZ - dist_to_look_back) % TOT_BUF_SZ;
            if actual_rpos + sz_from_internal_buf <= TOT_BUF_SZ {
                // no wraparound
                SpanSet(
                    &self.buf.buf[actual_rpos..actual_rpos + sz_from_internal_buf],
                    external_slice,
                    &[],
                )
            } else {
                // wraparound
                SpanSet(
                    &self.buf.buf[actual_rpos..TOT_BUF_SZ],
                    &self.buf.buf[..actual_rpos + sz_from_internal_buf - TOT_BUF_SZ],
                    external_slice,
                )
            }
        }
    }

    pub fn peek_byte(&self, offs: usize) -> u8 {
        if offs >= self.tot_ahead_sz() {
            // return something invalid (used for calculating hash at the end of the stream)
            0
        } else {
            let lookahead_valid_sz = self.buf.lookahead_valid_sz();
            if offs < lookahead_valid_sz {
                self.buf.buf[(self.buf.rpos + offs) % TOT_BUF_SZ]
            } else {
                self.inp[offs - lookahead_valid_sz]
            }
        }
    }

    pub fn tot_ahead_sz(&self) -> usize {
        self.buf.lookahead_valid_sz() + self.inp.len()
    }
    pub fn cursor_pos(&self) -> u64 {
        self.buf.rpos_real_offs
    }

    pub fn roll_window(&mut self, bytes: usize) {
        let lookahead_valid_sz = self.buf.lookahead_valid_sz();
        let tot_ahead_sz = self.tot_ahead_sz();
        assert!(bytes <= tot_ahead_sz);

        // the offset just advances, no complexity at all here
        self.buf.rpos_real_offs += bytes as u64;

        // no matter how much we're trying to roll the window,
        // the read pointer "just" advances (and wraps around)
        // (i.e. if rpos ends up conceptually "inside inp",
        // this calculation still holds)
        // the only time it doesn't is if we roll more than the whole buffer at once
        // (where rpos will be completely rewritten)
        self.buf.rpos = (self.buf.rpos + bytes) % TOT_BUF_SZ;

        let cur_wpos = self.buf.wpos;
        // the amount we're rolling the window,
        // plus however much is needed to make the lookahead fill up
        let ideal_target_wsz = bytes + (LOOKAHEAD_SZ - lookahead_valid_sz);
        let target_wsz = core::cmp::min(ideal_target_wsz, self.inp.len());
        let target_wpos = (cur_wpos + target_wsz) % TOT_BUF_SZ;
        self.buf.wpos = target_wpos;

        // in the simple case, all we need to do is fill from
        // cur_wpos to target_wpos with inp
        // however, complexity arises if we manage to roll over
        // more than the entirety of the buffer (including lookback)
        if target_wsz < TOT_BUF_SZ {
            if target_wpos >= cur_wpos {
                // no wraparound
                self.buf.buf[cur_wpos..target_wpos].copy_from_slice(&self.inp[..target_wsz]);
            } else {
                // wraparound
                self.buf.buf[cur_wpos..].copy_from_slice(&self.inp[..TOT_BUF_SZ - cur_wpos]);
                self.buf.buf[..target_wpos]
                    .copy_from_slice(&self.inp[TOT_BUF_SZ - cur_wpos..target_wsz]);
            }
        } else {
            // we've rolled over more than a full buffer
            // can reset everything so that it starts at the beginning
            self.buf
                .buf
                .copy_from_slice(&self.inp[target_wsz - TOT_BUF_SZ..target_wsz]);
            let shortened_lookahead = ideal_target_wsz - target_wsz;
            debug_assert!(shortened_lookahead <= LOOKAHEAD_SZ);
            self.buf.rpos = (LOOKBACK_SZ + shortened_lookahead) % TOT_BUF_SZ;
            self.buf.wpos = 0;
        }
        self.inp = &self.inp[target_wsz..];
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn span_all_from_inp() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 1024 + 256 - 1;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
    }

    #[test]
    fn span_all_from_buf_no_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.wpos = 8;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, [])
        }
    }

    #[test]
    fn span_all_from_buf_yes_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, [5, 6, 7, 8]);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 1] = 1;
            buf.buf[..7].copy_from_slice(&[2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 7;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1]);
            assert_eq!(spans.1, [2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 7..].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
            buf.buf[0] = 8;
            buf.rpos = 1024 + 256 - 7;
            buf.wpos = 1;
            let win = buf.add_inp(&[]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(spans.1, [8]);
            assert_eq!(spans.2, [])
        }
    }

    #[test]
    fn span_straddle_no_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[0..3].copy_from_slice(&[1, 2, 3]);
            buf.wpos = 3;
            let win = buf.add_inp(&[4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3]);
            assert_eq!(spans.1, [4, 5, 6, 7, 8]);
            assert_eq!(spans.2, [])
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 5].copy_from_slice(&[1, 2, 3, 4, 5]);
            buf.rpos = 123;
            buf.wpos = 123 + 5;
            let win = buf.add_inp(&[6, 7, 8]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5]);
            assert_eq!(spans.1, [6, 7, 8]);
            assert_eq!(spans.2, [])
        }
    }

    #[test]
    fn span_straddle_yes_wrap() {
        let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
        buf.buf[1024 + 256 - 3..].copy_from_slice(&[1, 2, 3]);
        buf.buf[..2].copy_from_slice(&[4, 5]);
        buf.rpos = 1024 + 256 - 3;
        buf.wpos = 2;
        let win = buf.add_inp(&[6, 7, 8]);
        let spans = win.get_next_spans(0, 8);
        assert_eq!(spans.0, [1, 2, 3]);
        assert_eq!(spans.1, [4, 5]);
        assert_eq!(spans.2, [6, 7, 8]);
    }

    #[test]
    fn peek_from_inp() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(3), 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(5), 6);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 1024 + 256 - 1;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(7), 8);
        }
    }

    #[test]
    fn peek_from_buf() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.wpos = 8;
            let win = buf.add_inp(&[]);
            assert_eq!(win.peek_byte(0), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(8), 9);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let win = buf.add_inp(&[]);
            assert_eq!(win.peek_byte(0), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(9), 10);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.add_inp(&[]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(5), 6);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(5), 6);
            assert_eq!(win.peek_byte(10), 11);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 1] = 1;
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 0;
            let win = buf.add_inp(&[]);
            assert_eq!(win.peek_byte(0), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(0), 1);
            assert_eq!(win.peek_byte(1), 9);
        }
    }

    #[test]
    fn roll_window_no_refill() {
        // not passing in any input, so no refill possible
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.wpos = 8;
            let mut win = buf.add_inp(&[]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 5);
            assert_eq!(buf.wpos, 8);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let mut win = buf.add_inp(&[]);
            win.roll_window(3);
            assert_eq!(buf.rpos, 123 + 3);
            assert_eq!(buf.wpos, 123 + 8);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.add_inp(&[]);
            win.roll_window(3);
            assert_eq!(buf.rpos, 1024 + 256 - 1);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.add_inp(&[]);
            win.roll_window(4);
            assert_eq!(buf.rpos, 0);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.add_inp(&[]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 1);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123 + 256;
            let mut win = buf.add_inp(&[]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 123 + 5);
            assert_eq!(buf.wpos, 123 + 256);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 123;
            buf.wpos = 256 - 123;
            let mut win = buf.add_inp(&[]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 1024 + 256 - 123 + 5);
            assert_eq!(buf.wpos, 256 - 123);
        }
    }

    #[test]
    fn roll_window_refill_exact() {
        // refill with exactly as many bytes as consumed
        // (i.e. lookahead starts full)
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123 + 256;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 123 + 5);
            assert_eq!(buf.wpos, 123 + 256 + 5);
            assert_eq!(buf.buf[123 + 256..123 + 256 + 5], [1, 2, 3, 4, 5]);
            assert_eq!(buf.buf[123 + 256 + 5], 0);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 - 2;
            buf.wpos = 1024 + 256 - 2;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5]);
            win.roll_window(5);
            assert_eq!(buf.rpos, 1024 + 3);
            assert_eq!(buf.wpos, 3);
            assert_eq!(buf.buf[1024 + 256 - 2..], [1, 2]);
            assert_eq!(buf.buf[..3], [3, 4, 5]);
            assert_eq!(buf.buf[3], 0);
        }
    }

    #[test]
    fn roll_window_fill_up() {
        // refill with more bytes than consumed
        // (i.e. lookahead starts insufficiently full)
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5]);
            win.roll_window(0);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 123);
            assert_eq!(buf.wpos, 123 + 5);
            assert_eq!(buf.buf[123..123 + 5], [1, 2, 3, 4, 5]);
            assert_eq!(buf.buf[123 + 5], 0);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5]);
            win.roll_window(1);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 124);
            assert_eq!(buf.wpos, 123 + 5);
            assert_eq!(buf.buf[123..123 + 5], [1, 2, 3, 4, 5]);
            assert_eq!(buf.buf[123 + 5], 0);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[123; 500]);
            win.roll_window(0);
            assert_eq!(win.inp, &[123; 500 - 256][..]);
            assert_eq!(buf.rpos, 0);
            assert_eq!(buf.wpos, 256);
            assert_eq!(buf.buf[..256], [123; 256]);
            assert_eq!(buf.buf[256], 0);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[123; 500]);
            win.roll_window(111);
            assert_eq!(win.inp, &[123; 500 - 256 - 111][..]);
            assert_eq!(buf.rpos, 111);
            assert_eq!(buf.wpos, 111 + 256);
            assert_eq!(buf.buf[..111 + 256], [123; 111 + 256]);
            assert_eq!(buf.buf[111 + 256..], [0; 1024 + 256 - (111 + 256)]);
        }
        // small, very exhaustive tests
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(0);
            assert_eq!(win.inp, &[5, 6, 7, 8, 9, 10, 11, 12][..]);
            assert_eq!(buf.rpos, 0);
            assert_eq!(buf.wpos, 4);
            assert_eq!(buf.buf[..4], [1, 2, 3, 4]);
            assert_eq!(buf.buf[4..], [0; 8]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(7);
            assert_eq!(win.inp, &[12][..]);
            assert_eq!(buf.rpos, 7);
            assert_eq!(buf.wpos, 11);
            assert_eq!(buf.buf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            buf.wpos = 2;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(9);
            assert_eq!(win.inp, &[12][..]);
            assert_eq!(buf.rpos, 9);
            assert_eq!(buf.wpos, 1);
            assert_eq!(buf.buf, [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        }
    }

    #[test]
    fn roll_window_loop_around() {
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            buf.rpos = 7;
            buf.wpos = 7;
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(8);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 8);
            assert_eq!(buf.wpos, 0);
            assert_eq!(buf.buf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            win.roll_window(9);
            assert_eq!(win.inp, &[14, 15, 16][..]);
            assert_eq!(buf.rpos, 8);
            assert_eq!(buf.wpos, 0);
            assert_eq!(buf.buf, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(9);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 9);
            assert_eq!(buf.wpos, 0);
            assert_eq!(buf.buf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(11);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 11);
            assert_eq!(buf.wpos, 0);
            assert_eq!(buf.buf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        }
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            win.roll_window(12);
            assert_eq!(win.inp, &[][..]);
            assert_eq!(buf.rpos, 0);
            assert_eq!(buf.wpos, 0);
            assert_eq!(buf.buf, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        }
    }

    #[test]
    fn lookback_span_simple() {
        {
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            win.roll_window(4);
            // now we have 4 bytes in lookback, 4 bytes in lookahead, [8 9 10 11] not used yet
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [0, 1, 2, 3]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(1, 3);
            assert_eq!(spans.0, [1, 2, 3]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(3, 5);
            assert_eq!(spans.0, [3, 4, 5, 6, 7]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            // oversize, clamped
            let spans = win.get_next_spans(0, 1000);
            assert_eq!(spans.0, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(spans.1, [8, 9, 10, 11]);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(3, 1000);
            assert_eq!(spans.0, [3, 4, 5, 6, 7]);
            assert_eq!(spans.1, [8, 9, 10, 11]);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(4, 1000);
            assert_eq!(spans.0, [4, 5, 6, 7]);
            assert_eq!(spans.1, [8, 9, 10, 11]);
            assert_eq!(spans.2, []);
        }
        {
            // test having less than max lookbehind
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[0, 1, 2]);
            win.roll_window(3);
            let spans = win.get_next_spans(0, 1000);
            assert_eq!(spans.0, [0, 1, 2]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(1, 1000);
            assert_eq!(spans.0, [1, 2]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(2, 1000);
            assert_eq!(spans.0, [2]);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
            let spans = win.get_next_spans(3, 1000);
            assert_eq!(spans.0, []);
            assert_eq!(spans.1, []);
            assert_eq!(spans.2, []);
        }
    }

    #[test]
    fn lookback_span_multiple_adds() {
        let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
        let mut win = buf.add_inp(&[0, 1, 2]);
        win.roll_window(3);
        let mut win = buf.add_inp(&[3, 4, 5, 6]);
        win.roll_window(4);
        // seven bytes in lookback, zero in lookahead
        let spans = win.get_next_spans(1, 1000);
        assert_eq!(spans.0, [1, 2, 3, 4, 5, 6]);
        assert_eq!(spans.1, []);
        assert_eq!(spans.2, []);
        let spans = win.get_next_spans(5, 1000);
        assert_eq!(spans.0, [5, 6]);
        assert_eq!(spans.1, []);
        assert_eq!(spans.2, []);
        // full 8 bytes in lookback, 4 in lookahead, 3 in inp
        let mut win = buf.add_inp(&[7, 8, 9, 10, 11, 12, 13, 14]);
        win.roll_window(1);
        let spans = win.get_next_spans(1, 1000);
        assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, [12, 13, 14]);
        assert_eq!(spans.2, []);
        let spans = win.get_next_spans(5, 1000);
        assert_eq!(spans.0, [5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, [12, 13, 14]);
        assert_eq!(spans.2, []);
        let spans = win.get_next_spans(8, 1000);
        assert_eq!(spans.0, [8, 9, 10, 11]);
        assert_eq!(spans.1, [12, 13, 14]);
        assert_eq!(spans.2, []);
    }

    #[test]
    fn loopback_span_having_overwritten() {
        let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
        let mut win = buf.add_inp(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        win.roll_window(8);
        // 8 bytes in lookback, 4 in lookahead, 2 in inp
        let spans = win.get_next_spans(0, 1000);
        assert_eq!(spans.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, [12, 13]);
        assert_eq!(spans.2, []);
        win.roll_window(1);
        // lookback is still full with 8, 4 in lookahead, 1 in inp
        // min pos is now 1, and a wraparound happens
        let spans = win.get_next_spans(1, 1000);
        assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, [12]);
        assert_eq!(spans.2, [13]);
        win.roll_window(1);
        // lookback is still full with 8, 4 in lookahead, 0 in inp
        // min pos is now 2, and a wraparound happens
        let spans = win.get_next_spans(2, 1000);
        assert_eq!(spans.0, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, [12, 13]);
        assert_eq!(spans.2, []);
    }
}
