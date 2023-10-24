#[repr(C)] // not sure if we need this because of padding bytes / uninitialized memory?
struct SlidingWindowBuf<
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
    fn new() -> Self {
        assert!(TOT_BUF_SZ == LOOKBACK_SZ + LOOKAHEAD_SZ);

        Self {
            buf: [0; TOT_BUF_SZ],
            rpos: 0,
            wpos: 0,
            rpos_real_offs: 0,
        }
    }
    unsafe fn initialize_at(p: *mut Self) {
        assert!(TOT_BUF_SZ == LOOKBACK_SZ + LOOKAHEAD_SZ);

        let p_buf = core::ptr::addr_of_mut!((*p).buf);
        for i in 0..TOT_BUF_SZ {
            (*p_buf)[i] = 0;
        }
        (*p).rpos = 0;
        (*p).wpos = 0;
        (*p).rpos_real_offs = 0;
    }

    fn add_inp<'a>(
        &'a mut self,
        inp: &'a [u8],
    ) -> SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ> {
        SlidingWindow { buf: self, inp }
    }

    fn flush(&mut self) -> SlidingWindow<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ> {
        SlidingWindow {
            buf: self,
            inp: &[],
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
> {
    buf: &'a mut SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    inp: &'a [u8],
}

impl<'a, const LOOKBACK_SZ: usize, const LOOKAHEAD_SZ: usize, const TOT_BUF_SZ: usize>
    SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>
{
    fn get_next_spans(&self, from_pos: u64, bytes: usize) -> SpanSet {
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
            SpanSet(&self.inp[..dist_to_look_forward], None, None)
        } else {
            let (sz_from_internal_buf, sz_from_external_buf) =
                if dist_to_look_forward <= lookahead_valid_sz {
                    (dist_to_look_back + dist_to_look_forward, 0)
                } else {
                    (
                        dist_to_look_back + lookahead_valid_sz,
                        dist_to_look_forward - lookahead_valid_sz,
                    )
                };
            let external_slice = if sz_from_external_buf != 0 {
                Some(&self.inp[..sz_from_external_buf])
            } else {
                None
            };

            let actual_rpos = (self.buf.rpos + TOT_BUF_SZ - dist_to_look_back) % TOT_BUF_SZ;
            if actual_rpos + sz_from_internal_buf <= TOT_BUF_SZ {
                // no wraparound
                SpanSet(
                    &self.buf.buf[actual_rpos..actual_rpos + sz_from_internal_buf],
                    external_slice,
                    None,
                )
            } else {
                // wraparound
                SpanSet(
                    &self.buf.buf[actual_rpos..TOT_BUF_SZ],
                    Some(&self.buf.buf[..actual_rpos + sz_from_internal_buf - TOT_BUF_SZ]),
                    external_slice,
                )
            }
        }
    }

    fn peek_byte(&self) -> u8 {
        let lookahead_valid_sz = self.buf.lookahead_valid_sz();
        if lookahead_valid_sz == 0 {
            self.inp[0]
        } else {
            self.buf.buf[self.buf.rpos]
        }
    }

    fn tot_ahead_sz(&self) -> usize {
        self.buf.lookahead_valid_sz() + self.inp.len()
    }

    fn roll_window(&mut self, bytes: usize) {
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum LZOutput {
    Lit(u8),
    Ref { disp: u64, len: u64 },
}

#[repr(C)] // not sure if we need this because of padding bytes / uninitialized memory?
struct HashBits<
    const MIN_MATCH: u64,
    const HASH_BITS: usize,
    const HASH_SZ: usize,
    const DICT_BITS: usize,
    const DICT_SZ: usize,
> {
    htab: [u64; HASH_SZ],
    prev: [u64; DICT_SZ],
    initial_lits_done: u64,
    hash_of_head: u32,
}

impl<
        const MIN_MATCH: u64,
        const HASH_BITS: usize,
        const HASH_SZ: usize,
        const DICT_BITS: usize,
        const DICT_SZ: usize,
    > HashBits<MIN_MATCH, HASH_BITS, HASH_SZ, DICT_BITS, DICT_SZ>
{
    fn new() -> Self {
        Self {
            htab: [u64::MAX; HASH_SZ],
            prev: [u64::MAX; DICT_SZ],
            initial_lits_done: 0,
            hash_of_head: 0,
        }
    }
    unsafe fn initialize_at(p: *mut Self) {
        let p_htab = core::ptr::addr_of_mut!((*p).htab);
        for i in 0..HASH_SZ {
            // my understanding is that this is safe because u64 doesn't have Drop
            // and we don't construct a & anywhere
            (*p_htab)[i] = u64::MAX;
        }
        let p_prev = core::ptr::addr_of_mut!((*p).prev);
        for i in 0..DICT_SZ {
            (*p_prev)[i] = u64::MAX;
        }
        (*p).initial_lits_done = 0;
        (*p).hash_of_head = 0;
    }
    fn calc_new_hash(&self, old_hash: u32, b: u8) -> u32 {
        let hash_shift = ((HASH_BITS as u64) + MIN_MATCH - 1) / MIN_MATCH;
        let hash = (old_hash << hash_shift) ^ (b as u32);
        hash & ((1 << HASH_BITS) - 1)
    }
    // returns old hashtable entry, which is a chain to follow when compressing
    fn put_head_into_htab<
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
    >(
        &mut self,
        win: &SlidingWindow<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    ) -> u64 {
        let b = win.peek_byte();
        self.hash_of_head = self.calc_new_hash(self.hash_of_head, b);
        let old_hpos = self.htab[self.hash_of_head as usize];
        self.htab[self.hash_of_head as usize] = win.buf.rpos_real_offs;
        let prev_idx = win.buf.rpos_real_offs & ((1 << DICT_BITS) - 1);
        self.prev[prev_idx as usize] = old_hpos;

        old_hpos
    }
}

#[repr(C)] // not sure if we need this because of padding bytes / uninitialized memory?
struct LZEngine<
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
    const MIN_MATCH: u64,
    const MAX_MATCH: u64,
    const HASH_BITS: usize,
    const HASH_SZ: usize,
    const DICT_BITS: usize,
    const DICT_SZ: usize,
    const MIN_DISP: u64,
> {
    sbuf: SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    h: HashBits<MIN_MATCH, HASH_BITS, HASH_SZ, DICT_BITS, DICT_SZ>,
}

impl<
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
        const MIN_MATCH: u64,
        const MAX_MATCH: u64,
        const HASH_BITS: usize,
        const HASH_SZ: usize,
        const DICT_BITS: usize,
        const DICT_SZ: usize,
        const MIN_DISP: u64,
    >
    LZEngine<
        LOOKBACK_SZ,
        LOOKAHEAD_SZ,
        TOT_BUF_SZ,
        MIN_MATCH,
        MAX_MATCH,
        HASH_BITS,
        HASH_SZ,
        DICT_BITS,
        DICT_SZ,
        MIN_DISP,
    >
{
    fn new() -> Self {
        assert_eq!(HASH_SZ, 1 << HASH_BITS);
        assert_eq!(DICT_SZ, 1 << DICT_BITS);
        assert!(MIN_DISP >= 1);
        assert!(LOOKAHEAD_SZ > 0);
        assert!(HASH_BITS <= 32);

        Self {
            sbuf: SlidingWindowBuf::new(),
            h: HashBits::new(),
        }
    }
    fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = std::alloc::alloc(layout) as *mut Self;
            SlidingWindowBuf::initialize_at(core::ptr::addr_of_mut!((*p).sbuf));
            HashBits::initialize_at(core::ptr::addr_of_mut!((*p).h));
            Box::from_raw(p)
        }
    }

    fn compress<O>(&mut self, inp: &[u8], mut outp: O)
    where
        O: FnMut(LZOutput),
    {
        let mut win = self.sbuf.add_inp(inp);

        while self.h.initial_lits_done < MIN_MATCH {
            // the first MIN_MATCH bytes can never be compressed as a backreference,
            // and we need to prime the hash table
            if win.tot_ahead_sz() == 0 {
                return;
            }
            let b = win.peek_byte();
            outp(LZOutput::Lit(b));
            self.h.put_head_into_htab(&win);
            self.h.initial_lits_done += 1;
            win.roll_window(1);
        }

        // XXX is this the right condition?
        while win.tot_ahead_sz() >= LOOKAHEAD_SZ {
            let b = win.peek_byte();
            let old_hpos = self.h.put_head_into_htab(&win);

            if old_hpos == u64::MAX {
                // no match
                outp(LZOutput::Lit(b));
            } else {
                todo!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_all_from_inp() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 1024 + 256 - 1;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
    }

    #[test]
    fn span_all_from_buf_no_wrap() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.wpos = 8;
            let win = buf.flush();
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let win = buf.flush();
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.flush();
            let spans = win.get_next_spans(0, 4);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None)
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
            let win = buf.flush();
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4]);
            assert_eq!(spans.1, Some(&[5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 1] = 1;
            buf.buf[..7].copy_from_slice(&[2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 7;
            let win = buf.flush();
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1]);
            assert_eq!(spans.1, Some(&[2, 3, 4, 5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 7..].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
            buf.buf[0] = 8;
            buf.rpos = 1024 + 256 - 7;
            buf.wpos = 1;
            let win = buf.flush();
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(spans.1, Some(&[8][..]));
            assert_eq!(spans.2, None)
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
            assert_eq!(spans.1, Some(&[4, 5, 6, 7, 8][..]));
            assert_eq!(spans.2, None)
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 5].copy_from_slice(&[1, 2, 3, 4, 5]);
            buf.rpos = 123;
            buf.wpos = 123 + 5;
            let win = buf.add_inp(&[6, 7, 8]);
            let spans = win.get_next_spans(0, 8);
            assert_eq!(spans.0, [1, 2, 3, 4, 5]);
            assert_eq!(spans.1, Some(&[6, 7, 8][..]));
            assert_eq!(spans.2, None)
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
        assert_eq!(spans.1, Some(&[4, 5][..]));
        assert_eq!(spans.2, Some(&[6, 7, 8][..]));
    }

    #[test]
    fn peek_from_inp() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(), 1);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(), 1);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 1024 + 256 - 1;
            let win = buf.add_inp(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(win.peek_byte(), 1);
        }
    }

    #[test]
    fn peek_from_buf() {
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.wpos = 8;
            let win = buf.flush();
            assert_eq!(win.peek_byte(), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(), 1);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[123..123 + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let win = buf.flush();
            assert_eq!(win.peek_byte(), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(), 1);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 4..].copy_from_slice(&[1, 2, 3, 4]);
            buf.buf[..4].copy_from_slice(&[5, 6, 7, 8]);
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let win = buf.flush();
            assert_eq!(win.peek_byte(), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(), 1);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.buf[1024 + 256 - 1] = 1;
            buf.rpos = 1024 + 256 - 1;
            buf.wpos = 0;
            let win = buf.flush();
            assert_eq!(win.peek_byte(), 1);
            let win = buf.add_inp(&[9, 10, 11]);
            assert_eq!(win.peek_byte(), 1);
        }
    }

    #[test]
    fn roll_window_no_refill() {
        // not passing in any input, so no refill possible
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.wpos = 8;
            let mut win = buf.flush();
            win.roll_window(5);
            assert_eq!(buf.rpos, 5);
            assert_eq!(buf.wpos, 8);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123 + 8;
            let mut win = buf.flush();
            win.roll_window(3);
            assert_eq!(buf.rpos, 123 + 3);
            assert_eq!(buf.wpos, 123 + 8);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.flush();
            win.roll_window(3);
            assert_eq!(buf.rpos, 1024 + 256 - 1);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.flush();
            win.roll_window(4);
            assert_eq!(buf.rpos, 0);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 4;
            buf.wpos = 4;
            let mut win = buf.flush();
            win.roll_window(5);
            assert_eq!(buf.rpos, 1);
            assert_eq!(buf.wpos, 4);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 123;
            buf.wpos = 123 + 256;
            let mut win = buf.flush();
            win.roll_window(5);
            assert_eq!(buf.rpos, 123 + 5);
            assert_eq!(buf.wpos, 123 + 256);
        }
        {
            let mut buf: SlidingWindowBuf<1024, 256, { 1024 + 256 }> = SlidingWindowBuf::new();
            buf.rpos = 1024 + 256 - 123;
            buf.wpos = 256 - 123;
            let mut win = buf.flush();
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
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(1, 3);
            assert_eq!(spans.0, [1, 2, 3]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(3, 5);
            assert_eq!(spans.0, [3, 4, 5, 6, 7]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            // oversize, clamped
            let spans = win.get_next_spans(0, 1000);
            assert_eq!(spans.0, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(spans.1, Some(&[8, 9, 10, 11][..]));
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(3, 1000);
            assert_eq!(spans.0, [3, 4, 5, 6, 7]);
            assert_eq!(spans.1, Some(&[8, 9, 10, 11][..]));
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(4, 1000);
            assert_eq!(spans.0, [4, 5, 6, 7]);
            assert_eq!(spans.1, Some(&[8, 9, 10, 11][..]));
            assert_eq!(spans.2, None);
        }
        {
            // test having less than max lookbehind
            let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
            let mut win = buf.add_inp(&[0, 1, 2]);
            win.roll_window(3);
            let spans = win.get_next_spans(0, 1000);
            assert_eq!(spans.0, [0, 1, 2]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(1, 1000);
            assert_eq!(spans.0, [1, 2]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(2, 1000);
            assert_eq!(spans.0, [2]);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
            let spans = win.get_next_spans(3, 1000);
            assert_eq!(spans.0, []);
            assert_eq!(spans.1, None);
            assert_eq!(spans.2, None);
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
        assert_eq!(spans.1, None);
        assert_eq!(spans.2, None);
        let spans = win.get_next_spans(5, 1000);
        assert_eq!(spans.0, [5, 6]);
        assert_eq!(spans.1, None);
        assert_eq!(spans.2, None);
        // full 8 bytes in lookback, 4 in lookahead, 3 in inp
        let mut win = buf.add_inp(&[7, 8, 9, 10, 11, 12, 13, 14]);
        win.roll_window(1);
        let spans = win.get_next_spans(1, 1000);
        assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12, 13, 14][..]));
        assert_eq!(spans.2, None);
        let spans = win.get_next_spans(5, 1000);
        assert_eq!(spans.0, [5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12, 13, 14][..]));
        assert_eq!(spans.2, None);
        let spans = win.get_next_spans(8, 1000);
        assert_eq!(spans.0, [8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12, 13, 14][..]));
        assert_eq!(spans.2, None);
    }

    #[test]
    fn loopback_span_having_overwritten() {
        let mut buf: SlidingWindowBuf<8, 4, { 8 + 4 }> = SlidingWindowBuf::new();
        let mut win = buf.add_inp(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        win.roll_window(8);
        // 8 bytes in lookback, 4 in lookahead, 2 in inp
        let spans = win.get_next_spans(0, 1000);
        assert_eq!(spans.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12, 13][..]));
        assert_eq!(spans.2, None);
        win.roll_window(1);
        // lookback is still full with 8, 4 in lookahead, 1 in inp
        // min pos is now 1, and a wraparound happens
        let spans = win.get_next_spans(1, 1000);
        assert_eq!(spans.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12][..]));
        assert_eq!(spans.2, Some(&[13][..]));
        win.roll_window(1);
        // lookback is still full with 8, 4 in lookahead, 0 in inp
        // min pos is now 2, and a wraparound happens
        let spans = win.get_next_spans(2, 1000);
        assert_eq!(spans.0, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(spans.1, Some(&[12, 13][..]));
        assert_eq!(spans.2, None);
    }

    #[test]
    fn hash_3_15() {
        let hb: HashBits<3, 15, { 1 << 15 }, 15, { 1 << 15 }> = HashBits::new();
        let h = 0;
        let h = hb.calc_new_hash(h, 0xF2);
        assert_eq!(h, 0xF2);
        let h = hb.calc_new_hash(h, 0x34);
        assert_eq!(h, (0xF2 << 5) ^ 0x34);
        // note: tests bit truncation
        let h = hb.calc_new_hash(h, 0x56);
        assert_eq!(h, (0x12 << 10) ^ (0x34 << 5) ^ 0x56);
    }

    #[test]
    fn hash_4_15() {
        let hb: HashBits<4, 15, { 1 << 15 }, 15, { 1 << 15 }> = HashBits::new();
        let h = 0;
        let h = hb.calc_new_hash(h, 0xF2);
        assert_eq!(h, 0xF2);
        let h = hb.calc_new_hash(h, 0x34);
        assert_eq!(h, (0xF2 << 4) ^ 0x34);
        // // note: tests bit truncation
        let h = hb.calc_new_hash(h, 0x56);
        assert_eq!(h, (0x72 << 8) ^ (0x34 << 4) ^ 0x56);
        let h = hb.calc_new_hash(h, 0x78);
        assert_eq!(h, (0x2 << 12) ^ (0x34 << 8) ^ (0x56 << 4) ^ 0x78);
    }

    #[test]
    fn hashing_inp() {
        let mut lz: Box<LZEngine<1, 0, 1, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>> =
            LZEngine::new_boxed();
        let mut win = lz.sbuf.add_inp(&[0x12, 0x34, 0x56, 0x78]);

        lz.h.put_head_into_htab(&win);

        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);

        assert_eq!(lz.h.htab[0x12], 0);
        assert_eq!(lz.h.htab[(0x12 << 5) ^ 0x34], 1);
        assert_eq!(lz.h.htab[(0x12 << 10) ^ (0x34 << 5) ^ 0x56], 2);
        assert_eq!(lz.h.htab[(0x14 << 10) ^ (0x56 << 5) ^ 0x78], 3);
    }

    #[test]
    fn hashing_inp_with_chaining() {
        let mut lz: Box<LZEngine<1, 0, 1, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>> =
            LZEngine::new_boxed();
        let mut win = lz
            .sbuf
            .add_inp(&[0b000_00_001, 0b001_00_001, 0b001_00_010, 0b010_00_010]);

        lz.h.put_head_into_htab(&win);

        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);

        assert_eq!(lz.h.htab[1], 1);
        assert_eq!(lz.h.htab[2], 3);
        assert_eq!(lz.h.prev[1], 0);
        assert_eq!(lz.h.prev[0], u64::MAX);
        assert_eq!(lz.h.prev[3], 2);
        assert_eq!(lz.h.prev[2], u64::MAX);
    }

    #[test]
    fn lz_head() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&[0x12, 0x34, 0x56], |x| compressed_out.push(x));

        assert_eq!(compressed_out.len(), 3);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
    }

    #[test]
    fn lz_not_compressible() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc], |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 3);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
    }
}
