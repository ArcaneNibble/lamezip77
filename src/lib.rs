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
    unsafe fn initialize_at(p: *mut Self) {
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

    fn add_inp<'a>(
        &'a mut self,
        inp: &'a [u8],
    ) -> SlidingWindow<'a, LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ> {
        SlidingWindow { buf: self, inp }
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

impl<'a> core::ops::Index<usize> for SpanSet<'a> {
    type Output = u8;

    fn index(&self, index: usize) -> &u8 {
        if index < self.0.len() {
            &self.0[index]
        } else {
            let index = index - self.0.len();
            let _1 = self.1.unwrap();
            if index < _1.len() {
                &_1[index]
            } else {
                let index = index - _1.len();
                let _2 = self.2.unwrap();
                &_2[index]
            }
        }
    }
}

impl<'a> SpanSet<'a> {
    fn len(&self) -> usize {
        self.0.len() + self.1.map_or(0, |x| x.len()) + self.2.map_or(0, |x| x.len())
    }

    fn compare(&self, other: &Self) -> usize {
        // todo optimizations? or can the compiler optimize this?
        let mut len = 0;
        let mut a = self.0;
        let mut a_which = 0;
        let mut b = other.0;
        let mut b_which = 0;

        loop {
            if a.len() == 0 {
                match a_which {
                    0 => {
                        if self.1.is_none() {
                            break;
                        }
                        a = self.1.unwrap();
                        a_which = 1;
                    }
                    1 => {
                        if self.2.is_none() {
                            break;
                        }
                        a = self.2.unwrap();
                        a_which = 2;
                    }
                    _ => break,
                }
            }
            if b.len() == 0 {
                match b_which {
                    0 => {
                        if other.1.is_none() {
                            break;
                        }
                        b = other.1.unwrap();
                        b_which = 1;
                    }
                    1 => {
                        if other.2.is_none() {
                            break;
                        }
                        b = other.2.unwrap();
                        b_which = 2;
                    }
                    _ => break,
                }
            }

            if a[0] == b[0] {
                len += 1;
                a = &a[1..];
                b = &b[1..];
            } else {
                break;
            }
        }

        len
    }
}

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

    fn peek_byte(&self, offs: usize) -> u8 {
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
    const MIN_MATCH: usize,
    const HASH_BITS: usize,
    const HASH_SZ: usize,
    const DICT_BITS: usize,
    const DICT_SZ: usize,
> {
    htab: [u64; HASH_SZ],
    prev: [u64; DICT_SZ],
    hash_of_head: u32,
    redo_hash_at_cursor: bool,
    redo_hash_behind_cursor_num_missing: u8,
}

impl<
        const MIN_MATCH: usize,
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
            hash_of_head: 0,
            redo_hash_at_cursor: true,
            redo_hash_behind_cursor_num_missing: 0,
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
        (*p).hash_of_head = 0;
        (*p).redo_hash_at_cursor = true;
        (*p).redo_hash_behind_cursor_num_missing = 0;
    }
    fn calc_new_hash(&self, old_hash: u32, b: u8) -> u32 {
        let hash_shift = (HASH_BITS + MIN_MATCH - 1) / MIN_MATCH;
        let hash = (old_hash << hash_shift) ^ (b as u32);
        hash & ((1 << HASH_BITS) - 1)
    }
    // returns old hashtable entry, which is a chain to follow when compressing
    fn put_raw_into_htab(&mut self, hash: u32, offs: u64) -> u64 {
        let old_hpos = self.htab[hash as usize];
        self.htab[hash as usize] = offs;
        let prev_idx = offs & ((1 << DICT_BITS) - 1);
        self.prev[prev_idx as usize] = old_hpos;

        old_hpos
    }
    fn put_head_into_htab<
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
    >(
        &mut self,
        win: &SlidingWindow<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    ) -> u64 {
        let old_hpos = self.put_raw_into_htab(self.hash_of_head, win.buf.rpos_real_offs);

        let b = win.peek_byte(MIN_MATCH);
        self.hash_of_head = self.calc_new_hash(self.hash_of_head, b);

        old_hpos
    }
    fn put_span_into_htab(&mut self, span: &SpanSet, cur_offs: u64, len: usize) {
        debug_assert!(span.len() >= len);

        for i in 1..len {
            println!(
                "span -- hash @ {:08X} is {:04X}",
                cur_offs + i as u64,
                self.hash_of_head
            );

            if i + MIN_MATCH <= span.len() {
                self.put_raw_into_htab(self.hash_of_head, cur_offs + i as u64);
            } else {
                println!("span -- skipping htab update, out of range")
            }

            if i + MIN_MATCH < span.len() {
                let b = span[i + MIN_MATCH];
                self.hash_of_head = self.calc_new_hash(self.hash_of_head, b);
            } else {
                println!("span -- skipping hash update, out of range")
            }
        }
    }
}

#[repr(C)] // not sure if we need this because of padding bytes / uninitialized memory?
struct LZEngine<
    const LOOKBACK_SZ: usize,
    const LOOKAHEAD_SZ: usize,
    const TOT_BUF_SZ: usize,
    const MIN_MATCH: usize,
    const MAX_MATCH: usize,
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
        const MIN_MATCH: usize,
        const MAX_MATCH: usize,
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
        assert!(MIN_MATCH >= 1);
        assert!(MIN_MATCH <= u8::MAX as usize);
        // this condition is required so that we can actually calculate hash
        assert!(LOOKAHEAD_SZ >= MIN_MATCH);
        assert!(HASH_BITS <= 32);

        Self {
            sbuf: SlidingWindowBuf::new(),
            h: HashBits::new(),
        }
    }
    fn new_boxed() -> Box<Self> {
        assert_eq!(HASH_SZ, 1 << HASH_BITS);
        assert_eq!(DICT_SZ, 1 << DICT_BITS);
        assert!(MIN_DISP >= 1);
        assert!(MIN_MATCH >= 1);
        assert!(MIN_MATCH <= u8::MAX as usize);
        // this condition is required so that we can actually calculate hash
        assert!(LOOKAHEAD_SZ >= MIN_MATCH);
        assert!(HASH_BITS <= 32);

        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = std::alloc::alloc(layout) as *mut Self;
            SlidingWindowBuf::initialize_at(core::ptr::addr_of_mut!((*p).sbuf));
            HashBits::initialize_at(core::ptr::addr_of_mut!((*p).h));
            Box::from_raw(p)
        }
    }

    fn compress<O>(&mut self, inp: &[u8], end_of_stream: bool, mut outp: O)
    where
        O: FnMut(LZOutput),
    {
        let mut win = self.sbuf.add_inp(inp);

        // XXX is this the right condition?
        println!("tot ahead {}", win.tot_ahead_sz());
        while win.tot_ahead_sz() > if end_of_stream { 0 } else { LOOKAHEAD_SZ } {
            if self.h.redo_hash_behind_cursor_num_missing > 0 {
                println!(
                    "redo behind cursor @ {:08X} # {}",
                    win.buf.rpos_real_offs, self.h.redo_hash_behind_cursor_num_missing
                );

                // we know there is >= 1 byte always, and >= LOOKAHEAD_SZ + 1 (aka >= MIN_MATCH + 1) bytes if not EOS
                // FIXME change EOS to flush??

                let mut hash = self.h.hash_of_head;

                for i in (0..self.h.redo_hash_behind_cursor_num_missing).rev() {
                    println!("redo behind pos -{} old hash {:04X}", i, hash);
                    // when we are in this situation, there is always one more htab update than hash update
                    // so we start with a hash update
                    hash = self
                        .h
                        .calc_new_hash(hash, win.peek_byte(MIN_MATCH - 1 - i as usize));

                    if i != 0 {
                        let old_hpos = self
                            .h
                            .put_raw_into_htab(hash, win.buf.rpos_real_offs - i as u64);
                        println!(
                            "redo behind pos insert @ {:08X} is {:04X} old {:08X}",
                            win.buf.rpos_real_offs - i as u64,
                            hash,
                            old_hpos
                        );
                    }
                }

                self.h.redo_hash_behind_cursor_num_missing = 0;
                self.h.hash_of_head = hash;
            }

            if self.h.redo_hash_at_cursor {
                println!("redo at cursor @ {:08X}", win.buf.rpos_real_offs);
                // we need to prime the hash table by recomputing the hash for cursor
                // either at the start or after a span

                // we either have at least LOOKAHEAD_SZ + 1 bytes available
                // (which is at least MIN_MATCH),
                // *or* we don't but are at EOS already (very short input)
                let initial_bytes = win.get_next_spans(win.buf.rpos_real_offs, MIN_MATCH);
                let mut hash = 0;
                for i in 0..MIN_MATCH {
                    hash = self.h.calc_new_hash(hash, initial_bytes[i]);
                }
                self.h.hash_of_head = hash;
                self.h.redo_hash_at_cursor = false;
            }

            let b: u8 = win.peek_byte(0);
            println!(
                "working @ {:08X} with val {:02X} cur hash {:04X}",
                win.buf.rpos_real_offs, b, self.h.hash_of_head
            );

            let mut old_hpos = self.h.put_head_into_htab(&win);
            println!("hash pointer is {:08X}", old_hpos);

            // initialize to an invalid value
            let mut best_match_len = MIN_MATCH - 1;
            let mut best_match_pos = u64::MAX;

            // this is what we're matching against
            let max_match = MAX_MATCH.try_into().unwrap_or(usize::MAX);
            // extra in the hopes that we don't have to recompute hash later
            let cursor_spans =
                win.get_next_spans(win.buf.rpos_real_offs, max_match + MIN_MATCH + 1);

            // a match within range
            // (we can terminate immediately because the prev chain only ever goes
            // further and further backwards)
            while old_hpos != u64::MAX && old_hpos + (LOOKBACK_SZ as u64) >= win.buf.rpos_real_offs
            {
                let eval_hpos = old_hpos;
                println!("probing at {:08X}", eval_hpos);
                old_hpos = self.h.prev[(old_hpos & ((1 << DICT_BITS) - 1)) as usize];

                if !(eval_hpos + MIN_DISP <= win.buf.rpos_real_offs) {
                    // too close
                    continue;
                }

                let lookback_spans = win.get_next_spans(eval_hpos, max_match);

                // TODO many optimizations here
                let match_len = lookback_spans.compare(&cursor_spans);
                println!("match len {} span len {}", match_len, lookback_spans.len());
                debug_assert!(match_len <= MAX_MATCH);
                if match_len >= MIN_MATCH {
                    if match_len > best_match_len {
                        best_match_len = match_len;
                        best_match_pos = eval_hpos;
                    }
                }
            }

            if best_match_len < MIN_MATCH {
                // output a literal
                outp(LZOutput::Lit(b));
                win.roll_window(1);
            } else {
                // output a match
                outp(LZOutput::Ref {
                    disp: win.buf.rpos_real_offs - best_match_pos,
                    len: best_match_len as u64,
                });
                self.h
                    .put_span_into_htab(&cursor_spans, win.buf.rpos_real_offs, best_match_len);
                let avail_extra_bytes = cursor_spans.len() - best_match_len;
                println!("match -- {} extra bytes", avail_extra_bytes);
                if avail_extra_bytes < MIN_MATCH {
                    self.h.redo_hash_behind_cursor_num_missing =
                        (MIN_MATCH - avail_extra_bytes) as u8;
                }
                win.roll_window(best_match_len);
            }
        }

        if win.tot_ahead_sz() > 0 {
            debug_assert!(win.tot_ahead_sz() <= LOOKAHEAD_SZ);
            win.roll_window(0);
            debug_assert!(win.inp.len() == 0);
        }
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
            let win = buf.add_inp(&[]);
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
            let win = buf.add_inp(&[]);
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
            let win = buf.add_inp(&[]);
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
            let win = buf.add_inp(&[]);
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
            let win = buf.add_inp(&[]);
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
            let win = buf.add_inp(&[]);
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
        let mut lz: Box<LZEngine<1, 3, 4, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>> =
            LZEngine::new_boxed();
        let mut win = lz.sbuf.add_inp(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc]);

        lz.h.put_head_into_htab(&win);

        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);

        assert_eq!(lz.h.htab[0], 0); // XXX bogus
        assert_eq!(lz.h.htab[0x78], 1);
        assert_eq!(lz.h.htab[(0x78 << 5) ^ 0x9a], 2);
        assert_eq!(lz.h.htab[(0x18 << 10) ^ (0x9a << 5) ^ 0xbc], 3);
        assert_eq!(lz.h.htab[(0x1a << 10) ^ (0xbc << 5)], 4); // end
        assert_eq!(lz.h.htab[(0x1c << 10)], 5); // end
    }

    #[test]
    fn hashing_inp_with_chaining() {
        let mut lz: Box<LZEngine<1, 3, 4, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>> =
            LZEngine::new_boxed();
        let mut win = lz.sbuf.add_inp(&[
            0,
            0,
            0,
            0b000_00_001,
            0b001_00_001,
            0b001_00_010,
            0b010_00_010,
        ]);

        lz.h.put_head_into_htab(&win);

        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);
        win.roll_window(1);
        lz.h.put_head_into_htab(&win);

        assert_eq!(lz.h.htab[0], 0); // XXX bogus
        assert_eq!(lz.h.htab[1], 2);
        assert_eq!(lz.h.htab[2], 4);
        assert_eq!(lz.h.prev[2], 1);
        assert_eq!(lz.h.prev[1], u64::MAX);
        assert_eq!(lz.h.prev[4], 3);
        assert_eq!(lz.h.prev[3], u64::MAX);
    }

    #[test]
    fn span_set_compare() {
        let a = SpanSet(&[1, 2, 3, 4, 5, 6, 7, 8], None, None);

        let b = SpanSet(&[1, 2, 3, 4, 5, 6, 7], None, None);
        assert_eq!(a.compare(&b), 7);

        let b = SpanSet(&[1, 2, 3, 4, 5, 6, 7, 8, 9], None, None);
        assert_eq!(a.compare(&b), 8);

        let b = SpanSet(&[1, 2, 3, 4, 4, 6, 7], None, None);
        assert_eq!(a.compare(&b), 4);

        let b = SpanSet(&[1, 2, 3], Some(&[4, 5, 6, 7]), None);
        assert_eq!(a.compare(&b), 7);

        let b = SpanSet(&[1, 2, 3], Some(&[4, 5, 5, 7]), None);
        assert_eq!(a.compare(&b), 5);

        let b = SpanSet(&[1, 2, 3], Some(&[4]), Some(&[5, 6]));
        assert_eq!(a.compare(&b), 6);

        let a = SpanSet(&[1, 2], Some(&[3, 4, 5, 6]), Some(&[7, 8]));
        let b = SpanSet(&[1, 2, 3], Some(&[4, 5]), Some(&[6, 7, 8, 9]));
        assert_eq!(a.compare(&b), 8);
    }

    #[test]
    fn span_index() {
        let a = SpanSet(&[1, 2], Some(&[3, 4, 5, 6]), Some(&[7, 8]));

        for i in 0..8 {
            assert_eq!(a[i], (i + 1) as u8);
        }
    }

    #[test]
    fn lz_head_only() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&[0x12, 0x34, 0x56], true, |x| compressed_out.push(x));

        assert_eq!(compressed_out.len(), 3);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
    }

    #[test]
    fn lz_head_split() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&[0x12], false, |x| compressed_out.push(x));
        println!("compress 1");
        lz.compress(&[0x34, 0x56], true, |x| compressed_out.push(x));
        println!("compress 2");

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
        lz.compress(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc], true, |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 6);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Lit(0x78));
        assert_eq!(compressed_out[4], LZOutput::Lit(0x9a));
        assert_eq!(compressed_out[5], LZOutput::Lit(0xbc));
    }

    #[test]
    fn lz_simple_repeat() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&[0x12, 0x34, 0x56, 0x12, 0x34, 0x56], true, |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 3 });
    }

    #[test]
    fn lz_longer_than_disp_repeat() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| compressed_out.push(x),
        );

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 6 });
    }

    #[test]
    fn lz_longer_than_lookahead_repeat() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34,
                0x56, 0x12, 0x34, 0x56,
            ],
            true,
            |x| compressed_out.push(x),
        );

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 15 });
    }

    #[test]
    fn lz_split_repeat() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(&[0x12, 0x34, 0x56, 0x12, 0x34, 0x56], true, |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 5);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 9 });
        assert_eq!(compressed_out[4], LZOutput::Ref { disp: 3, len: 6 });
    }

    #[test]
    fn lz_detailed_backref_hashing() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(&[0x34, 0x56, 0x12, 0x34, 0x56], true, |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 5);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 10 });
        assert_eq!(compressed_out[4], LZOutput::Ref { disp: 3, len: 5 });

        assert_eq!(lz.h.htab[(0x12 << 10) ^ (0x34 << 5) ^ 0x56], 15);
        assert_eq!(lz.h.prev[15], 12);
        assert_eq!(lz.h.prev[12], 9);
        assert_eq!(lz.h.prev[9], 6);
        assert_eq!(lz.h.prev[6], 3);
        assert_eq!(lz.h.prev[3], 0);
        assert_eq!(lz.h.prev[0], u64::MAX);

        assert_eq!(lz.h.htab[(0x14 << 10) ^ (0x56 << 5) ^ 0x12], 13);
        assert_eq!(lz.h.prev[13], 10);
        assert_eq!(lz.h.prev[10], 7);
        assert_eq!(lz.h.prev[7], 4);
        assert_eq!(lz.h.prev[4], 1);
        assert_eq!(lz.h.prev[1], u64::MAX);

        assert_eq!(lz.h.htab[(0x16 << 10) ^ (0x12 << 5) ^ 0x34], 14);
        assert_eq!(lz.h.prev[14], 11);
        assert_eq!(lz.h.prev[11], 8);
        assert_eq!(lz.h.prev[8], 5);
        assert_eq!(lz.h.prev[5], 2);
        assert_eq!(lz.h.prev[2], u64::MAX);
    }

    #[test]
    fn lz_peek_ahead_after_span() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(&[0x34, 0x56, 0x12, 0x34, 0x56, 0xde], true, |x| {
            compressed_out.push(x)
        });

        assert_eq!(compressed_out.len(), 9);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 6 });
        assert_eq!(compressed_out[4], LZOutput::Lit(0x78));
        assert_eq!(compressed_out[5], LZOutput::Lit(0x9a));
        assert_eq!(compressed_out[6], LZOutput::Lit(0xbc));
        assert_eq!(compressed_out[7], LZOutput::Ref { disp: 8, len: 5 });
        assert_eq!(compressed_out[8], LZOutput::Lit(0xde));

        assert_eq!(lz.h.htab[(0x12 << 10) ^ (0x34 << 5) ^ 0x56], 14);
        assert_eq!(lz.h.prev[14], 6);
        assert_eq!(lz.h.prev[6], 3);
        assert_eq!(lz.h.prev[3], 0);
        assert_eq!(lz.h.prev[0], u64::MAX);

        assert_eq!(lz.h.htab[(0x14 << 10) ^ (0x56 << 5) ^ 0x12], 12);
        assert_eq!(lz.h.prev[12], 4);
        assert_eq!(lz.h.prev[4], 1);
        assert_eq!(lz.h.prev[1], u64::MAX);

        assert_eq!(lz.h.htab[(0x16 << 10) ^ (0x12 << 5) ^ 0x34], 13);
        assert_eq!(lz.h.prev[13], 5);
        assert_eq!(lz.h.prev[5], 2);
        assert_eq!(lz.h.prev[2], u64::MAX);

        assert_eq!(lz.h.htab[(0x14 << 10) ^ (0x56 << 5) ^ 0x78], 7);
        assert_eq!(lz.h.prev[7], u64::MAX);

        assert_eq!(lz.h.htab[(0x14 << 10) ^ (0x56 << 5) ^ 0xde], 15);
        assert_eq!(lz.h.prev[15], u64::MAX);

        assert_eq!(lz.h.htab[(0x16 << 10) ^ (0x78 << 5) ^ 0x9a], 8);
        assert_eq!(lz.h.prev[8], u64::MAX);

        assert_eq!(lz.h.htab[(0x18 << 10) ^ (0x9a << 5) ^ 0xbc], 9);
        assert_eq!(lz.h.prev[9], u64::MAX);

        assert_eq!(lz.h.htab[(0x1a << 10) ^ (0xbc << 5) ^ 0x34], 10);
        assert_eq!(lz.h.prev[10], u64::MAX);

        assert_eq!(lz.h.htab[(0x1c << 10) ^ (0x34 << 5) ^ 0x56], 11);
        assert_eq!(lz.h.prev[11], u64::MAX);
    }

    #[test]
    fn lz_big_file() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/lib.rs");
        let outp_fn = d.join("test.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&inp, true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(outp_fn).unwrap());
        for &lz_tok in &compressed_out {
            match lz_tok {
                LZOutput::Lit(lit) => {
                    outp_f.write(&[0]).unwrap();
                    outp_f.write(&[lit]).unwrap();
                }
                LZOutput::Ref { disp, len } => {
                    outp_f.write(&[0xff]).unwrap();
                    outp_f.write(&disp.to_le_bytes()).unwrap();
                    outp_f.write(&len.to_le_bytes()).unwrap();
                }
            }
        }

        let mut decompress = Vec::new();
        for &lz_tok in &compressed_out {
            match lz_tok {
                LZOutput::Lit(lit) => {
                    decompress.push(lit);
                }
                LZOutput::Ref { disp, len } => {
                    assert!(len > 0);
                    assert!(len <= 256);
                    assert!(disp >= 1);
                    assert!(disp <= 256);
                    for _ in 0..len {
                        decompress.push(decompress[decompress.len() - disp as usize]);
                    }
                }
            }
        }

        assert_eq!(inp, decompress);
    }

    #[test]
    fn lz_big_file_2() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/lib.rs");
        let outp_fn = d.join("test2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut lz: Box<
            LZEngine<32768, 256, { 32768 + 256 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            lz.compress(&[inp[i]], false, |x| compressed_out.push(x));
        }
        lz.compress(&[], true, |x| compressed_out.push(x));

        let mut outp_f = BufWriter::new(File::create(outp_fn).unwrap());
        for &lz_tok in &compressed_out {
            match lz_tok {
                LZOutput::Lit(lit) => {
                    outp_f.write(&[0]).unwrap();
                    outp_f.write(&[lit]).unwrap();
                }
                LZOutput::Ref { disp, len } => {
                    outp_f.write(&[0xff]).unwrap();
                    outp_f.write(&disp.to_le_bytes()).unwrap();
                    outp_f.write(&len.to_le_bytes()).unwrap();
                }
            }
        }

        let mut decompress = Vec::new();
        for &lz_tok in &compressed_out {
            match lz_tok {
                LZOutput::Lit(lit) => {
                    decompress.push(lit);
                }
                LZOutput::Ref { disp, len } => {
                    assert!(len > 0);
                    assert!(len <= 256);
                    assert!(disp >= 1);
                    assert!(disp <= 32768);
                    for _ in 0..len {
                        decompress.push(decompress[decompress.len() - disp as usize]);
                    }
                }
            }
        }

        assert_eq!(inp, decompress);
    }
}
