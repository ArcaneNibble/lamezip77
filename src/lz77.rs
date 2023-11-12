use crate::hashtables::HashBits;
use crate::sliding_window::SlidingWindowBuf;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box};

/// Settings for tuning compression
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LZSettings {
    /// If a match has a length >= this value, use it immediately and stop searching.
    ///
    /// `nice_length` in the gzip implementation
    pub good_enough_search_len: u64,
    /// If a match has a length > this value, don't bother inserting all sliding substrings
    /// into hash table (only insert the head, and then skip everything until the next input after the match)
    ///
    /// `max_lazy` in the gzip implementation, only for compression levels <= 3
    pub max_len_to_insert_all_substr: u64,
    /// Only follow the hash table at most this many times.
    ///
    /// `max_chain` in the gzip implementation
    pub max_prev_chain_follows: u64,
    /// Don't output matches immediately. Instead, try one position forward to see if it produces a longer match
    ///
    /// In the gzip implementation, done for compression levels >= 4
    pub defer_output_match: bool,
    /// Don't look for a potentially-longer match if there is already one this good.
    /// Only used when [defer_output_match](Self::defer_output_match) is set.
    ///
    /// `max_lazy` in the gzip implementation, only for compression levels >= 4
    pub good_enough_defer_len: u64,
    /// If there is already a deferred match this long, follow hash table chain fewer times.
    /// Only used when [defer_output_match](Self::defer_output_match) is set.
    ///
    /// `good_length` in the gzip implementation.
    pub search_faster_defer_len: u64,
    /// Minimum distance needed for a match (i.e. larger than 1)
    ///
    /// This is used for Nintendo LZ77 in VRAM mode.
    pub min_disp: u64,
    /// Hold this number of literals at the end of stream, and output it as a literal run.
    ///
    /// This is used to meet LZ4's end-of-stream conditions in a simple way.
    pub eos_holdout_bytes: u64,
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
            min_disp: 1,
            eos_holdout_bytes: 0,
        }
    }
}

/// Something that the LZ77 engine can output.
/// Either a literal or a backreference.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum LZOutput {
    /// Literal byte
    Lit(u8),
    /// Backreference. `disp` is at least 1, where 1 is the last byte output.
    Ref { disp: u64, len: u64 },
}

impl Default for LZOutput {
    fn default() -> Self {
        Self::Lit(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DeferredMatch {
    first_byte: u8,
    best_match_pos: u64,
    best_match_len: u64,
}

/// Parameterized, streaming LZ77 compression engine
///
/// * `LOOKBACK_SZ` specifies the window size
/// * `LOOKAHEAD_SZ` specifies the number of bytes that will be buffered before attempting to compress any output.
/// * `TOT_BUF_SZ` must be the sum of `LOOKBACK_SZ` and `LOOKAHEAD_SZ` and is a const generics workaround
/// * `MIN_MATCH` specifies the minimum length of matches
/// * `MAX_MATCH` specifies the maximum length of matches. It may be usize::MAX to make this unlimited.
///    This may exceed `LOOKAHEAD_SZ`, but longer matches are only possible if a large chunk of data is passed to the
///    compression function in a single call.
/// * `HASH_BITS` specifies the number of bits to use in hashes.
/// * `HASH_SZ` must be `1 << HASH_BITS` and is a const generics workaround.
/// * `DICT_BITS` specifies the number of bits to use in the hash chain table.
/// * `DICT_SZ` must be `1 << DICT_BITS` and is a const generics workaround.
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
> {
    pub(crate) sbuf: SlidingWindowBuf<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    pub(crate) h: HashBits<MIN_MATCH, HASH_BITS, HASH_SZ, DICT_BITS, DICT_SZ>,
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
    >
{
    /// Construct a new object
    pub fn new() -> Self {
        assert_eq!(HASH_SZ, 1 << HASH_BITS);
        assert_eq!(DICT_SZ, 1 << DICT_BITS);
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
    /// Construct a new object in place into a heap-allocated box
    ///
    /// This is needed whenever the optimizer refuses to construct the object as desired
    /// without causing a stack overflow, as this object is large.
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            Self::initialize_at(p);
            Box::from_raw(p)
        }
    }
    /// Initialize this object in place at a given pointer
    ///
    /// This is needed whenever the optimizer refuses to construct the object as desired
    /// without causing a stack overflow, as this object is large.
    pub unsafe fn initialize_at(p: *mut Self) {
        assert_eq!(HASH_SZ, 1 << HASH_BITS);
        assert_eq!(DICT_SZ, 1 << DICT_BITS);
        assert!(MIN_MATCH >= 1);
        assert!(MIN_MATCH <= u8::MAX as usize);
        // this condition is required so that we can actually calculate hash
        assert!(LOOKAHEAD_SZ >= MIN_MATCH);
        assert!(HASH_BITS <= 32);

        SlidingWindowBuf::initialize_at(core::ptr::addr_of_mut!((*p).sbuf));
        HashBits::initialize_at(core::ptr::addr_of_mut!((*p).h));
        (*p).redo_hash_at_cursor = true;
        (*p).redo_hash_behind_cursor_num_missing = 0;
        (*p).deferred_match = None;
    }

    /// Compress some data
    ///
    /// * `settings` specifies compression settings. Settings can change between calls, although this is not well-tested.
    /// * `inp` contains data to compress
    /// * `end_of_stream` indicates whether there is no more input data, which causes final blocks to get flushed.
    /// * `outp` is a callback sink for compressed data.
    pub fn compress<O, E>(
        &mut self,
        settings: &LZSettings,
        inp: &[u8],
        end_of_stream: bool,
        mut outp: O,
    ) -> Result<(), E>
    where
        O: FnMut(LZOutput) -> Result<(), E>,
    {
        assert!(settings.min_disp >= 1);
        assert!(settings.eos_holdout_bytes <= LOOKAHEAD_SZ as u64);

        let mut win = self.sbuf.add_inp(inp);

        if !settings.defer_output_match {
            if let Some(deferred_match) = self.deferred_match {
                // XXX this is not tested
                outp(LZOutput::Lit(deferred_match.first_byte))?;
                self.deferred_match = None;
            }
        }

        let required_min_bytes_left = if end_of_stream {
            settings.eos_holdout_bytes as usize
        } else {
            LOOKAHEAD_SZ
        };

        while win.tot_ahead_sz() > required_min_bytes_left {
            if self.redo_hash_behind_cursor_num_missing > 0 {
                // we know there is >= 1 byte always, and >= LOOKAHEAD_SZ + 1 (aka >= MIN_MATCH + 1) bytes if not EOS
                // FIXME change EOS to flush??

                let mut hash = self.h.hash_of_head;

                for i in (0..self.redo_hash_behind_cursor_num_missing).rev() {
                    // when we are in this situation, there is always one more htab update than hash update
                    // so we start with a hash update
                    hash = self
                        .h
                        .calc_new_hash(hash, win.peek_byte(MIN_MATCH - 1 - i as usize));

                    if i != 0 {
                        let _old_hpos = self.h.put_raw_into_htab(hash, win.cursor_pos() - i as u64);
                    }
                }

                self.redo_hash_behind_cursor_num_missing = 0;
                self.h.hash_of_head = hash;
            }

            if self.redo_hash_at_cursor {
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

            let mut old_hpos = self.h.put_head_into_htab(&win);

            let mut best_match_endpeek = b;
            let mut best_match_len = 1;
            let mut best_match_pos = u64::MAX;

            // this is what we're matching against
            let max_match = if end_of_stream && settings.eos_holdout_bytes > 0 {
                win.tot_ahead_sz() - settings.eos_holdout_bytes as usize
            } else {
                // xxx this overflow prevention is broken
                MAX_MATCH.try_into().unwrap_or(usize::MAX - MIN_MATCH - 1)
            };
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
                    old_hpos = self.h.prev[(old_hpos & ((1 << DICT_BITS) - 1)) as usize];

                    if !(eval_hpos + settings.min_disp <= win.cursor_pos()) {
                        // too close
                        continue;
                    }

                    let lookback_spans = win.get_next_spans(eval_hpos, max_match);

                    // optimization because the match can't get better if it
                    // doesn't match the best one we've got up to the best length we've got
                    if lookback_spans[best_match_len - 1] != best_match_endpeek {
                        continue;
                    }

                    let match_len = lookback_spans.compare(&cursor_spans);
                    debug_assert!(match_len <= MAX_MATCH);
                    if match_len >= MIN_MATCH {
                        if match_len > best_match_len {
                            best_match_len = match_len;
                            best_match_pos = eval_hpos;
                            best_match_endpeek = lookback_spans[match_len - 1];
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

                        outp(LZOutput::Ref {
                            // cursor has been advanced one position
                            disp: win.cursor_pos() - 1 - deferred_match.best_match_pos,
                            len: deferred_match.best_match_len,
                        })?;

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

                        outp(LZOutput::Lit(deferred_match.first_byte))?;
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
                        outp(LZOutput::Lit(b))?;
                    } else {
                        // a match here, let's defer it

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
                    outp(LZOutput::Lit(b))?;
                    win.roll_window(1);
                } else {
                    // output a match
                    outp(LZOutput::Ref {
                        disp: win.cursor_pos() - best_match_pos,
                        len: best_match_len as u64,
                    })?;
                    if best_match_len as u64 <= settings.max_len_to_insert_all_substr {
                        self.h
                            .put_span_into_htab(&cursor_spans, win.cursor_pos(), best_match_len);
                        let avail_extra_bytes = cursor_spans.len() - best_match_len;
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
            // ensure everything left is pulled into the buffer before we let go of inp
            win.roll_window(0);
        }

        if end_of_stream {
            if let Some(deferred_match) = self.deferred_match {
                // output last deferred match
                // ignore window, EOS

                outp(LZOutput::Ref {
                    // cursor has been advanced one position
                    disp: win.cursor_pos() - 1 - deferred_match.best_match_pos,
                    len: deferred_match.best_match_len,
                })?;
            }

            if settings.eos_holdout_bytes > 0 {
                // dump these out as literals
                debug_assert!(win.tot_ahead_sz() as u64 <= settings.eos_holdout_bytes);
                while win.tot_ahead_sz() > 0 {
                    outp(LZOutput::Lit(win.peek_byte(0)))?;
                    win.roll_window(1);
                }
            }
            assert!(win.tot_ahead_sz() == 0);
        }

        Ok(())
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    extern crate std;
    use std::{
        boxed::Box,
        fs::File,
        io::{BufWriter, Write},
        println,
        vec::Vec,
    };

    use super::*;

    #[test]
    fn lz_head_only() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(&LZSettings::default(), &[0x12, 0x34, 0x56], true, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();

        assert_eq!(compressed_out.len(), 3);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
    }

    #[test]
    fn lz_head_split() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(&LZSettings::default(), &[0x12], false, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();
        println!("compress 1");
        lz.compress::<_, ()>(&LZSettings::default(), &[0x34, 0x56], true, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();
        println!("compress 2");

        assert_eq!(compressed_out.len(), 3);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
    }

    #[test]
    fn lz_not_compressible() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 3 });
    }

    #[test]
    fn lz_longer_than_disp_repeat() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 6 });
    }

    #[test]
    fn lz_longer_than_lookahead_repeat() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34,
                0x56, 0x12, 0x34, 0x56,
            ],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(compressed_out.len(), 4);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 15 });
    }

    #[test]
    fn lz_split_repeat() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56,
            ],
            false,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x12, 0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(compressed_out.len(), 5);
        assert_eq!(compressed_out[0], LZOutput::Lit(0x12));
        assert_eq!(compressed_out[1], LZOutput::Lit(0x34));
        assert_eq!(compressed_out[2], LZOutput::Lit(0x56));
        assert_eq!(compressed_out[3], LZOutput::Ref { disp: 3, len: 9 });
        assert_eq!(compressed_out[4], LZOutput::Ref { disp: 3, len: 6 });
    }

    #[test]
    fn lz_detailed_backref_hashing() {
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12,
            ],
            false,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x34, 0x56, 0x12, 0x34, 0x56],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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
        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
            ],
            false,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();
        lz.compress::<_, ()>(
            &LZSettings::default(),
            &[0x34, 0x56, 0x12, 0x34, 0x56, 0xde],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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

        let inp_fn = d.join("src/lz77.rs");
        let outp_fn = d.join("test.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(&LZSettings::default(), &inp, true, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();

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

        let inp_fn = d.join("src/lz77.rs");
        let outp_fn = d.join("test2.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut lz: Box<
            LZEngine<32768, 256, { 32768 + 256 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            lz.compress::<_, ()>(&LZSettings::default(), &[inp[i]], false, |x| {
                compressed_out.push(x);
                Ok(())
            })
            .unwrap();
        }
        lz.compress::<_, ()>(&LZSettings::default(), &[], true, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();

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

        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &settings,
            &[
                0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56,
            ],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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

        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &settings,
            &[1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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

        let mut lz: Box<LZEngine<256, 8, { 256 + 8 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>> =
            LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        lz.compress::<_, ()>(
            &settings,
            &[1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            true,
            |x| {
                compressed_out.push(x);
                Ok(())
            },
        )
        .unwrap();

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

        let inp_fn = d.join("src/lz77.rs");
        let outp_fn = d.join("test3.bin");

        let inp = std::fs::read(inp_fn).unwrap();

        let mut settings = LZSettings::default();
        settings.defer_output_match = true;
        let mut lz: Box<
            LZEngine<32768, 256, { 32768 + 256 }, 3, 256, 15, { 1 << 15 }, 16, { 1 << 16 }>,
        > = LZEngine::new_boxed();
        let mut compressed_out = Vec::new();
        for i in 0..inp.len() {
            lz.compress::<_, ()>(&settings, &[inp[i]], false, |x| {
                compressed_out.push(x);
                Ok(())
            })
            .unwrap();
        }
        lz.compress::<_, ()>(&settings, &[], true, |x| {
            compressed_out.push(x);
            Ok(())
        })
        .unwrap();

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
