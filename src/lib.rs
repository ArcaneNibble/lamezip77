mod sliding_window;
use sliding_window::SlidingWindowBuf;

mod hashtables;
use hashtables::HashBits;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LZSettings {
    // if match >= this, use it immediately and stop searching
    pub good_enough_search_len: u64,
    // if len > this, don't bother inserting all sliding substrings
    // into hash table (only head)
    pub max_len_to_insert_all_substr: u64,
    // only follow the hash table at most this many times
    pub max_prev_chain_follows: u64,
    // try one position forward to see if it gets better matches
    pub defer_output_match: bool,
    // don't look for a potentially-longer match if there is already one this good
    pub good_enough_defer_len: u64,
    // if there is already a match this good, follow the chain fewer times
    pub search_faster_defer_len: u64,
}

impl Default for LZSettings {
    fn default() -> Self {
        Self {
            good_enough_search_len: u64::MAX,
            max_len_to_insert_all_substr: u64::MAX,
            max_prev_chain_follows: u64::MAX,
            defer_output_match: false,
            good_enough_defer_len: u64::MAX,
            search_faster_defer_len: u64::MAX,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum LZOutput {
    Lit(u8),
    Ref { disp: u64, len: u64 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DeferredMatch {
    first_byte: u8,
    best_match_pos: u64,
    best_match_len: u64,
}

pub struct LZEngine<
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
    redo_hash_at_cursor: bool,
    redo_hash_behind_cursor_num_missing: u8,
    deferred_match: Option<DeferredMatch>,
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
    pub fn new() -> Self {
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
            redo_hash_at_cursor: true,
            redo_hash_behind_cursor_num_missing: 0,
            deferred_match: None,
        }
    }
    pub fn new_boxed() -> Box<Self> {
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
            (*p).redo_hash_at_cursor = true;
            (*p).redo_hash_behind_cursor_num_missing = 0;
            (*p).deferred_match = None;
            Box::from_raw(p)
        }
    }

    pub fn compress<O>(
        &mut self,
        settings: &LZSettings,
        inp: &[u8],
        end_of_stream: bool,
        mut outp: O,
    ) where
        O: FnMut(LZOutput),
    {
        let mut win = self.sbuf.add_inp(inp);

        if !settings.defer_output_match {
            if let Some(deferred_match) = self.deferred_match {
                // XXX this is not tested
                println!("dump out deferred match");
                outp(LZOutput::Lit(deferred_match.first_byte));
                self.deferred_match = None;
            }
        }

        println!("tot ahead {}", win.tot_ahead_sz());
        while win.tot_ahead_sz() > if end_of_stream { 0 } else { LOOKAHEAD_SZ } {
            if self.redo_hash_behind_cursor_num_missing > 0 {
                println!(
                    "redo behind cursor @ {:08X} # {}",
                    win.cursor_pos(),
                    self.redo_hash_behind_cursor_num_missing
                );

                // we know there is >= 1 byte always, and >= LOOKAHEAD_SZ + 1 (aka >= MIN_MATCH + 1) bytes if not EOS
                // FIXME change EOS to flush??

                let mut hash = self.h.hash_of_head;

                for i in (0..self.redo_hash_behind_cursor_num_missing).rev() {
                    println!("redo behind pos -{} old hash {:04X}", i, hash);
                    // when we are in this situation, there is always one more htab update than hash update
                    // so we start with a hash update
                    hash = self
                        .h
                        .calc_new_hash(hash, win.peek_byte(MIN_MATCH - 1 - i as usize));

                    if i != 0 {
                        let old_hpos = self.h.put_raw_into_htab(hash, win.cursor_pos() - i as u64);
                        println!(
                            "redo behind pos insert @ {:08X} is {:04X} old {:08X}",
                            win.cursor_pos() - i as u64,
                            hash,
                            old_hpos
                        );
                    }
                }

                self.redo_hash_behind_cursor_num_missing = 0;
                self.h.hash_of_head = hash;
            }

            if self.redo_hash_at_cursor {
                println!("redo at cursor @ {:08X}", win.cursor_pos());
                // we need to prime the hash table by recomputing the hash for cursor
                // either at the start or after a span

                // we either have at least LOOKAHEAD_SZ + 1 bytes available
                // (which is at least MIN_MATCH),
                // *or* we don't but are at EOS already (very short input)
                let mut hash = 0;
                for i in 0..MIN_MATCH {
                    hash = self.h.calc_new_hash(hash, win.peek_byte(i));
                }
                self.h.hash_of_head = hash;
                self.redo_hash_at_cursor = false;
            }

            let b: u8 = win.peek_byte(0);
            println!(
                "working @ {:08X} with val {:02X} cur hash {:04X}",
                win.cursor_pos(),
                b,
                self.h.hash_of_head
            );

            let mut old_hpos = self.h.put_head_into_htab(&win);
            println!("hash pointer is {:08X}", old_hpos);

            // initialize to an invalid value
            let mut best_match_len = MIN_MATCH - 1;
            let mut best_match_pos = u64::MAX;

            // this is what we're matching against
            let max_match = MAX_MATCH.try_into().unwrap_or(usize::MAX);
            // extra few bytes in the hopes that we don't have to
            // do the redo_hash_behind_cursor_num_missing calculation
            let cursor_spans = win.get_next_spans(win.cursor_pos(), max_match + MIN_MATCH + 1);

            let skip_search = if let Some(deferred_match) = self.deferred_match {
                // should always be true because of the XXX check at the beginning
                debug_assert!(settings.defer_output_match);
                deferred_match.best_match_len >= settings.good_enough_defer_len
            } else {
                false
            };

            let mut prev_follow_limit = if let Some(deferred_match) = self.deferred_match {
                if deferred_match.best_match_len >= settings.search_faster_defer_len {
                    settings.max_prev_chain_follows / 4
                } else {
                    settings.max_prev_chain_follows
                }
            } else {
                settings.max_prev_chain_follows
            };

            if !skip_search {
                // a match within range
                // (we can terminate immediately because the prev chain only ever goes
                // further and further backwards)
                while prev_follow_limit > 0
                    && old_hpos != u64::MAX
                    && old_hpos + (LOOKBACK_SZ as u64) >= win.cursor_pos()
                {
                    prev_follow_limit -= 1;
                    let eval_hpos = old_hpos;
                    println!("probing at {:08X}", eval_hpos);
                    old_hpos = self.h.prev[(old_hpos & ((1 << DICT_BITS) - 1)) as usize];

                    if !(eval_hpos + MIN_DISP <= win.cursor_pos()) {
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
                            if match_len as u64 >= settings.good_enough_search_len {
                                break;
                            }
                        }
                    }
                }
            }

            if settings.defer_output_match {
                if let Some(deferred_match) = self.deferred_match {
                    if deferred_match.best_match_len >= best_match_len as u64 {
                        // if there is a deferred match, and it's better than this one,
                        // then output the deferred one now

                        println!(
                            "deferred match of {} @ {:08X} is better than this ({})",
                            deferred_match.best_match_len,
                            deferred_match.best_match_pos,
                            best_match_len,
                        );

                        outp(LZOutput::Ref {
                            // cursor has been advanced one position
                            disp: win.cursor_pos() - 1 - deferred_match.best_match_pos,
                            len: deferred_match.best_match_len,
                        });

                        let match_len_minus_1 = (deferred_match.best_match_len - 1) as usize;
                        if deferred_match.best_match_len <= settings.max_len_to_insert_all_substr {
                            // because of the advancing of one position, the hash starting at cursor_pos() is already inserted
                            // put_span_into_htab starts at +1, so this actually starts at cursor_pos()+1
                            let deferred_span =
                                win.get_next_spans(win.cursor_pos(), match_len_minus_1 + MIN_MATCH);
                            debug_assert!(deferred_span.len() >= match_len_minus_1);
                            self.h.put_span_into_htab(
                                &deferred_span,
                                win.cursor_pos(),
                                match_len_minus_1,
                            );
                            let avail_extra_bytes = deferred_span.len() - match_len_minus_1;
                            println!("match -- {} extra bytes", avail_extra_bytes);
                            if avail_extra_bytes < MIN_MATCH {
                                self.redo_hash_behind_cursor_num_missing =
                                    (MIN_MATCH - avail_extra_bytes) as u8;
                            }
                        } else {
                            self.redo_hash_at_cursor = true;
                        }
                        win.roll_window(match_len_minus_1);
                        self.deferred_match = None;
                    } else {
                        // there is a deferred match, but it's worse than this one
                        // output the first char, then save current as deferred

                        println!(
                            "deferred match of {} @ {:08X} = {:02X} is worse than this ({} @ {:08X} = {:02X})",
                            deferred_match.best_match_len,
                            deferred_match.best_match_pos,
                            deferred_match.first_byte,
                            best_match_len,
                            best_match_pos,
                            b,
                        );

                        outp(LZOutput::Lit(deferred_match.first_byte));
                        self.deferred_match = Some(DeferredMatch {
                            first_byte: b,
                            best_match_pos,
                            best_match_len: best_match_len as u64,
                        });
                        win.roll_window(1);
                    }
                } else {
                    // no deferred match
                    if best_match_len < MIN_MATCH {
                        // no match here either
                        outp(LZOutput::Lit(b));
                    } else {
                        // a match here, let's defer it

                        println!(
                            "deferring a match of {} @ {:08X} = {:02X}",
                            best_match_len, best_match_pos, b
                        );

                        self.deferred_match = Some(DeferredMatch {
                            first_byte: b,
                            best_match_pos,
                            best_match_len: best_match_len as u64,
                        });
                    }
                    // advance one position in any case
                    win.roll_window(1);
                }
            } else {
                if best_match_len < MIN_MATCH {
                    // output a literal
                    outp(LZOutput::Lit(b));
                    win.roll_window(1);
                } else {
                    // output a match
                    outp(LZOutput::Ref {
                        disp: win.cursor_pos() - best_match_pos,
                        len: best_match_len as u64,
                    });
                    if best_match_len as u64 <= settings.max_len_to_insert_all_substr {
                        self.h
                            .put_span_into_htab(&cursor_spans, win.cursor_pos(), best_match_len);
                        let avail_extra_bytes = cursor_spans.len() - best_match_len;
                        println!("match -- {} extra bytes", avail_extra_bytes);
                        if avail_extra_bytes < MIN_MATCH {
                            self.redo_hash_behind_cursor_num_missing =
                                (MIN_MATCH - avail_extra_bytes) as u8;
                        }
                    } else {
                        self.redo_hash_at_cursor = true;
                    }
                    win.roll_window(best_match_len);
                }
            }
        }

        if win.tot_ahead_sz() > 0 {
            debug_assert!(win.tot_ahead_sz() <= LOOKAHEAD_SZ);
            win.roll_window(0);
        }

        if end_of_stream {
            if let Some(deferred_match) = self.deferred_match {
                // output last deferred match
                // ignore window, EOS

                println!(
                    "last deferred match of {} @ {:08X} = {:02X}",
                    deferred_match.best_match_len,
                    deferred_match.best_match_pos,
                    deferred_match.first_byte,
                );

                outp(LZOutput::Ref {
                    // cursor has been advanced one position
                    disp: win.cursor_pos() - 1 - deferred_match.best_match_pos,
                    len: deferred_match.best_match_len,
                });
            }
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
    fn lz_head_only() {
        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(&LZSettings::default(), &[0x12, 0x34, 0x56], true, |x| {
            compressed_out.push(x)
        });

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
        lz.compress(&LZSettings::default(), &[0x12], false, |x| {
            compressed_out.push(x)
        });
        println!("compress 1");
        lz.compress(&LZSettings::default(), &[0x34, 0x56], true, |x| {
            compressed_out.push(x)
        });
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
        lz.compress(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc],
            true,
            |x| compressed_out.push(x),
        );

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
        lz.compress(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| compressed_out.push(x),
        );

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
            &LZSettings::default(),
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
            &LZSettings::default(),
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
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| compressed_out.push(x),
        );

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
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(
            &LZSettings::default(),
            &[0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| compressed_out.push(x),
        );

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
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
            ],
            false,
            |x| compressed_out.push(x),
        );
        lz.compress(
            &LZSettings::default(),
            &[0x34, 0x56, 0x12, 0x34, 0x56, 0xde],
            true,
            |x| compressed_out.push(x),
        );

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
        lz.compress(&LZSettings::default(), &inp, true, |x| {
            compressed_out.push(x)
        });

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
            lz.compress(&LZSettings::default(), &[inp[i]], false, |x| {
                compressed_out.push(x)
            });
        }
        lz.compress(&LZSettings::default(), &[], true, |x| {
            compressed_out.push(x)
        });

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

    #[test]
    fn lz_max_insert_len() {
        let mut settings = LZSettings::default();
        settings.max_len_to_insert_all_substr = 4;

        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &settings,
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56,
            ],
            true,
            |x| compressed_out.push(x),
        );

        assert_eq!(compressed_out.len(), 6);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 6 });
        assert_eq!(compressed_out[4], LZOutput::Lit(0x78));
        assert_eq!(compressed_out[5], LZOutput::Ref { disp: 7, len: 3 });
    }

    #[test]
    fn lz_deferred_insert() {
        let mut settings = LZSettings::default();
        settings.defer_output_match = true;

        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &settings,
            &[1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            true,
            |x| compressed_out.push(x),
        );

        assert_eq!(compressed_out.len(), 9);
        assert_eq!(compressed_out[0], LZOutput::Lit(1));
        assert_eq!(compressed_out[1], LZOutput::Lit(2));
        assert_eq!(compressed_out[2], LZOutput::Lit(3));
        assert_eq!(compressed_out[3], LZOutput::Lit(2));
        assert_eq!(compressed_out[4], LZOutput::Lit(3));
        assert_eq!(compressed_out[5], LZOutput::Lit(4));
        assert_eq!(compressed_out[6], LZOutput::Lit(5));
        assert_eq!(compressed_out[7], LZOutput::Lit(1));
        assert_eq!(compressed_out[8], LZOutput::Ref { disp: 5, len: 4 });
    }

    #[test]
    fn lz_deferred_insert_sike() {
        let mut settings = LZSettings::default();
        settings.defer_output_match = true;
        settings.good_enough_defer_len = 3;

        let mut lz: Box<
            LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress(
            &settings,
            &[1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            true,
            |x| compressed_out.push(x),
        );

        assert_eq!(compressed_out.len(), 10);
        assert_eq!(compressed_out[0], LZOutput::Lit(1));
        assert_eq!(compressed_out[1], LZOutput::Lit(2));
        assert_eq!(compressed_out[2], LZOutput::Lit(3));
        assert_eq!(compressed_out[3], LZOutput::Lit(2));
        assert_eq!(compressed_out[4], LZOutput::Lit(3));
        assert_eq!(compressed_out[5], LZOutput::Lit(4));
        assert_eq!(compressed_out[6], LZOutput::Lit(5));
        assert_eq!(compressed_out[7], LZOutput::Ref { disp: 7, len: 3 });
        // the defer doesn't kick in because this match is good enough
        assert_eq!(compressed_out[8], LZOutput::Lit(4));
        assert_eq!(compressed_out[9], LZOutput::Lit(5));
    }

    #[test]
    fn lz_big_file_defer() {
        let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let inp_fn = d.join("src/lib.rs");
        let outp_fn = d.join("test3.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut settings = LZSettings::default();
        settings.defer_output_match = true;
        let mut lz: Box<
            LZEngine<32768, 256, { 32768 + 256 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }, 1>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            lz.compress(&settings, &[inp[i]], false, |x| compressed_out.push(x));
        }
        lz.compress(&settings, &[], true, |x| compressed_out.push(x));

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
