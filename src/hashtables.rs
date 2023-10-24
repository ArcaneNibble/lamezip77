use crate::sliding_window::{SlidingWindow, SpanSet};

pub struct HashBits<
    const MIN_MATCH: usize,
    const HASH_BITS: usize,
    const HASH_SZ: usize,
    const DICT_BITS: usize,
    const DICT_SZ: usize,
> {
    pub htab: [u64; HASH_SZ],
    pub prev: [u64; DICT_SZ],
    pub hash_of_head: u32,
}

impl<
        const MIN_MATCH: usize,
        const HASH_BITS: usize,
        const HASH_SZ: usize,
        const DICT_BITS: usize,
        const DICT_SZ: usize,
    > HashBits<MIN_MATCH, HASH_BITS, HASH_SZ, DICT_BITS, DICT_SZ>
{
    pub fn new() -> Self {
        Self {
            htab: [u64::MAX; HASH_SZ],
            prev: [u64::MAX; DICT_SZ],
            hash_of_head: 0,
        }
    }
    pub unsafe fn initialize_at(p: *mut Self) {
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
    }
    pub fn calc_new_hash(&self, old_hash: u32, b: u8) -> u32 {
        let hash_shift = (HASH_BITS + MIN_MATCH - 1) / MIN_MATCH;
        let hash = (old_hash << hash_shift) ^ (b as u32);
        hash & ((1 << HASH_BITS) - 1)
    }
    // returns old hashtable entry, which is a chain to follow when compressing
    pub fn put_raw_into_htab(&mut self, hash: u32, offs: u64) -> u64 {
        let old_hpos = self.htab[hash as usize];
        self.htab[hash as usize] = offs;
        let prev_idx = offs & ((1 << DICT_BITS) - 1);
        self.prev[prev_idx as usize] = old_hpos;

        old_hpos
    }
    pub fn put_head_into_htab<
        const LOOKBACK_SZ: usize,
        const LOOKAHEAD_SZ: usize,
        const TOT_BUF_SZ: usize,
    >(
        &mut self,
        win: &SlidingWindow<LOOKBACK_SZ, LOOKAHEAD_SZ, TOT_BUF_SZ>,
    ) -> u64 {
        let old_hpos = self.put_raw_into_htab(self.hash_of_head, win.cursor_pos());

        let b = win.peek_byte(MIN_MATCH);
        self.hash_of_head = self.calc_new_hash(self.hash_of_head, b);

        old_hpos
    }
    pub fn put_span_into_htab(&mut self, span: &SpanSet, cur_offs: u64, len: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LZEngine;

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
        assert_eq!(lz.h.htab[0x1c << 10], 5); // end
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
}
