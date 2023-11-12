use core::ops::Index;
use core::slice::SliceIndex;

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::vec::Vec;

// we ideally want this trait to require Index
// but none of these methods here actually
// involve the index type. Adding the Index
// requirement breaks type inference
// causing unwanted "type annotations needed"
pub trait MaybeGrowableBuf {
    fn add_lit(&mut self, b: u8);
    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()>;
    fn cur_pos(&self) -> usize;
    fn limit(&self) -> usize;
}

pub struct FixedBuf<'a> {
    cur_pos: usize,
    buf: &'a mut [u8],
}

impl<'a> From<&'a mut [u8]> for FixedBuf<'a> {
    fn from(value: &'a mut [u8]) -> Self {
        Self {
            cur_pos: 0,
            buf: value,
        }
    }
}

impl<'a, I> Index<I> for FixedBuf<'a>
where
    I: SliceIndex<[u8]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.buf[index]
    }
}

impl<'a> MaybeGrowableBuf for FixedBuf<'a> {
    fn add_lit(&mut self, b: u8) {
        self.buf[self.cur_pos] = b;
        self.cur_pos += 1;
    }

    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()> {
        if disp < 1 {
            return Err(());
        }
        if disp > self.cur_pos {
            return Err(());
        }

        let len = core::cmp::min(len, self.limit() - self.cur_pos);

        for j in 0..len {
            self.buf[self.cur_pos + j] = self.buf[self.cur_pos - disp + j];
        }
        self.cur_pos += len;

        Ok(())
    }

    fn cur_pos(&self) -> usize {
        self.cur_pos
    }

    fn limit(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(feature = "alloc")]
pub struct VecBuf {
    limit: usize,
    buf: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl VecBuf {
    pub fn new(prealloc: usize, limit: usize) -> Self {
        let buf = if prealloc > 0 {
            Vec::with_capacity(prealloc)
        } else {
            Vec::new()
        };

        Self { limit, buf }
    }
}

#[cfg(feature = "alloc")]
impl Into<Vec<u8>> for VecBuf {
    fn into(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(feature = "alloc")]
impl<I> Index<I> for VecBuf
where
    I: SliceIndex<[u8]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.buf[index]
    }
}

#[cfg(feature = "alloc")]
impl MaybeGrowableBuf for VecBuf {
    fn add_lit(&mut self, b: u8) {
        if self.buf.len() == self.limit {
            return;
        }

        self.buf.push(b);
    }

    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()> {
        if disp < 1 {
            return Err(());
        }
        if disp > self.cur_pos() {
            return Err(());
        }

        let len = core::cmp::min(len, self.limit - self.cur_pos());

        for _ in 0..len {
            self.buf.push(self.buf[self.cur_pos() - disp]);
        }

        Ok(())
    }

    fn cur_pos(&self) -> usize {
        self.buf.len()
    }

    fn limit(&self) -> usize {
        self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn util_fixed_buf() {
        let mut buf_ = [0u8; 4];
        let mut buf = FixedBuf::from(&mut buf_[..]);

        assert_eq!(buf.limit(), 4);

        assert_eq!(buf.cur_pos(), 0);
        buf.add_lit(0x11);
        assert_eq!(buf.cur_pos(), 1);
        buf.add_lit(0x22);
        assert_eq!(buf.cur_pos(), 2);
        buf.add_lit(0x33);
        assert_eq!(buf.cur_pos(), 3);
        buf.add_lit(0x44);

        assert_eq!(buf[0], 0x11);
        assert_eq!(buf[1..3], [0x22, 0x33]);
    }

    #[test]
    fn util_fixed_buf_match() {
        let mut buf_ = [0u8; 8];
        let mut buf = FixedBuf::from(&mut buf_[..]);

        buf.add_lit(0x11);
        buf.add_lit(0x22);
        buf.add_lit(0x33);
        buf.add_match(2, 5).unwrap();

        assert_eq!(buf_, [0x11, 0x22, 0x33, 0x22, 0x33, 0x22, 0x33, 0x22]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn util_vec_buf() {
        let mut buf = VecBuf::new(0, 4);

        assert_eq!(buf.limit(), 4);

        assert_eq!(buf.cur_pos(), 0);
        buf.add_lit(0x11);
        assert_eq!(buf.cur_pos(), 1);
        buf.add_lit(0x22);
        assert_eq!(buf.cur_pos(), 2);
        buf.add_lit(0x33);
        assert_eq!(buf.cur_pos(), 3);
        buf.add_lit(0x44);

        assert_eq!(buf[0], 0x11);
        assert_eq!(buf[1..3], [0x22, 0x33]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn util_vec_buf_match() {
        let mut buf = VecBuf::new(0, 8);

        buf.add_lit(0x11);
        buf.add_lit(0x22);
        buf.add_lit(0x33);
        buf.add_match(2, 5).unwrap();

        let buf: Vec<u8> = buf.into();
        assert_eq!(buf, [0x11, 0x22, 0x33, 0x22, 0x33, 0x22, 0x33, 0x22]);
    }
}
