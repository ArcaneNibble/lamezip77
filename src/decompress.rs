use core::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

#[cfg(feature = "alloc")]
extern crate alloc as alloc_crate;
#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box, vec::Vec};

const DUMMY_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |_| RawWaker::new(core::ptr::null(), &DUMMY_WAKER_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);
const DUMMY_WAKER: RawWaker = RawWaker::new(core::ptr::null(), &DUMMY_WAKER_VTABLE);

struct InputWithBuf<const BUFSZ: usize> {
    buf: [u8; BUFSZ],
    nbuf: usize,
    inp: *const [u8],
}

impl<const BUFSZ: usize> InputWithBuf<BUFSZ> {
    fn new() -> Self {
        Self {
            buf: [0; BUFSZ],
            nbuf: 0,
            inp: core::ptr::slice_from_raw_parts(core::ptr::null(), 0),
        }
    }
}

pub struct InputPeeker<'a, 'c, const BUFSZ: usize, const PEEKSZ: usize> {
    buf: &'a RefCell<InputWithBuf<BUFSZ>>,
    nwaiting: &'c Cell<usize>,
}

impl<'a, 'c, const BUFSZ: usize, const PEEKSZ: usize> InputPeeker<'a, 'c, BUFSZ, PEEKSZ> {
    fn new(input_with_buf: &'a RefCell<InputWithBuf<BUFSZ>>, nwaiting: &'c Cell<usize>) -> Self {
        assert!(PEEKSZ <= BUFSZ);
        Self {
            buf: input_with_buf,
            nwaiting,
        }
    }
}

impl<'a, 'c, const BUFSZ: usize, const PEEKSZ: usize> Future
    for &InputPeeker<'a, 'c, BUFSZ, PEEKSZ>
{
    type Output = [u8; PEEKSZ];

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let s = self.get_mut();
        let mut input_with_buf = s.buf.borrow_mut();
        let mut data = [0; PEEKSZ];

        if PEEKSZ <= input_with_buf.nbuf {
            data.copy_from_slice(&input_with_buf.buf[..PEEKSZ]);
            input_with_buf.buf.copy_within(PEEKSZ.., 0);
            input_with_buf.nbuf -= PEEKSZ;
            s.nwaiting.set(0);
            Poll::Ready(data)
        } else {
            // SAFETY: StreamingDecompressState.add_inp is the only thing that
            // can set inp. It ensures that inp is valid before any calls to us (poll)
            // and then invalidates it after the async decompress function returns
            // so that the pointer doesn't dangle past its actual valid lifetime
            let inp = unsafe {
                if !input_with_buf.inp.is_null() {
                    &*input_with_buf.inp
                } else {
                    &[]
                }
            };
            let nbuf = input_with_buf.nbuf;
            let tot_avail = nbuf + inp.len();

            if PEEKSZ > tot_avail {
                input_with_buf.buf[nbuf..(nbuf + inp.len())].copy_from_slice(inp);
                input_with_buf.nbuf += inp.len();
                input_with_buf.inp = core::ptr::slice_from_raw_parts(core::ptr::null(), 0);
                s.nwaiting.set(PEEKSZ - tot_avail);
                Poll::Pending
            } else {
                if nbuf == 0 {
                    data.copy_from_slice(&inp[..PEEKSZ]);
                    input_with_buf.inp = &inp[PEEKSZ..];
                    s.nwaiting.set(0);
                    Poll::Ready(data)
                } else {
                    data[..nbuf].copy_from_slice(&input_with_buf.buf[..nbuf]);
                    data[nbuf..].copy_from_slice(&inp[..(PEEKSZ - nbuf)]);
                    input_with_buf.nbuf = 0;
                    input_with_buf.inp = &inp[(PEEKSZ - nbuf)..];
                    s.nwaiting.set(0);
                    Poll::Ready(data)
                }
            }
        }
    }
}

pub struct StreamingDecompressInnerState<const BUFSZ: usize> {
    inp: RefCell<InputWithBuf<BUFSZ>>,
    nwaiting: Cell<usize>,
}

impl<const BUFSZ: usize> StreamingDecompressInnerState<BUFSZ> {
    pub fn new() -> Self {
        Self {
            inp: RefCell::new(InputWithBuf::new()),
            nwaiting: Cell::new(0),
        }
    }

    pub fn get_peeker<const N: usize>(&self) -> InputPeeker<'_, '_, BUFSZ, N> {
        InputPeeker::new(&self.inp, &self.nwaiting)
    }
}

pub struct StreamingDecompressState<'a, F, E, const BUFSZ: usize>
where
    F: Future<Output = Result<(), E>>,
{
    s: &'a StreamingDecompressInnerState<BUFSZ>,
    f: Pin<&'a mut F>,
}

impl<'a, F, E, const BUFSZ: usize> StreamingDecompressState<'a, F, E, BUFSZ>
where
    F: Future<Output = Result<(), E>>,
{
    pub fn new(s: &'a StreamingDecompressInnerState<BUFSZ>, f: Pin<&'a mut F>) -> Self {
        Self { s, f }
    }

    pub fn add_inp(&mut self, inp: &[u8]) -> Result<usize, E> {
        {
            let mut input_with_buf = self.s.inp.borrow_mut();
            debug_assert!(input_with_buf.inp.is_null());
            input_with_buf.inp = inp;
        }

        let waker = unsafe { &Waker::from_raw(DUMMY_WAKER) };
        let mut cx = Context::from_waker(waker);
        let poll_result = self.f.as_mut().poll(&mut cx);

        let ret = match poll_result {
            Poll::Ready(result) => result.map(|_| 0),
            Poll::Pending => Ok(self.s.nwaiting.get()),
        };

        {
            let mut input_with_buf = self.s.inp.borrow_mut();
            if !input_with_buf.inp.is_null() {
                // if there is data left, we *must* be completed or errored
                // XXX this is not valid if the decompress function blocks on anything else other than
                // peeking, but that's not the intended use case
                if let Ok(nwaiting) = ret {
                    debug_assert_eq!(nwaiting, 0);
                }
            }
            input_with_buf.inp = core::ptr::slice_from_raw_parts(core::ptr::null(), 0);
        }

        ret
    }
}
pub trait LZOutputBuf {
    fn add_lits(&mut self, lits: &[u8]);
    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()>;
    fn cur_pos(&self) -> usize;
    fn is_at_limit(&self) -> bool;
}

pub struct PreallocatedBuf<'a> {
    cur_pos: usize,
    buf: &'a mut [u8],
}

impl<'a> From<&'a mut [u8]> for PreallocatedBuf<'a> {
    fn from(value: &'a mut [u8]) -> Self {
        Self {
            cur_pos: 0,
            buf: value,
        }
    }
}

impl<'a> LZOutputBuf for PreallocatedBuf<'a> {
    fn add_lits(&mut self, lits: &[u8]) {
        let len = core::cmp::min(lits.len(), self.buf.len() - self.cur_pos);
        self.buf[self.cur_pos..(self.cur_pos + len)].copy_from_slice(&lits[..len]);
        self.cur_pos += len;
    }

    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()> {
        if disp < 1 {
            return Err(());
        }
        if disp > self.cur_pos {
            return Err(());
        }

        let len = core::cmp::min(len, self.buf.len() - self.cur_pos);
        for j in 0..len {
            self.buf[self.cur_pos + j] = self.buf[self.cur_pos - disp + j];
        }
        self.cur_pos += len;

        Ok(())
    }

    fn cur_pos(&self) -> usize {
        self.cur_pos
    }

    fn is_at_limit(&self) -> bool {
        self.cur_pos == self.buf.len()
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
impl LZOutputBuf for VecBuf {
    fn add_lits(&mut self, lits: &[u8]) {
        let len = core::cmp::min(lits.len(), self.limit - self.buf.len());
        self.buf.extend_from_slice(&lits[..len]);
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

    fn is_at_limit(&self) -> bool {
        self.buf.len() == self.limit
    }
}

pub struct StreamingOutputBuf<O, const LOOKBACK_SZ: usize>
where
    O: FnMut(&[u8]),
{
    outp: O,
    buf: [u8; LOOKBACK_SZ],
    wptr: usize,
    cur_pos: usize,
}

impl<O, const LOOKBACK_SZ: usize> StreamingOutputBuf<O, LOOKBACK_SZ>
where
    O: FnMut(&[u8]),
{
    pub fn new(outp: O) -> Self {
        Self {
            outp,
            buf: [0; LOOKBACK_SZ],
            wptr: 0,
            cur_pos: 0,
        }
    }
    #[cfg(feature = "alloc")]
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let p = alloc::alloc(layout) as *mut Self;
            let p_buf = core::ptr::addr_of_mut!((*p).buf);
            for i in 0..LOOKBACK_SZ {
                (*p_buf)[i] = 0;
            }
            (*p).wptr = 0;
            (*p).cur_pos = 0;
            Box::from_raw(p)
        }
    }
}

impl<O, const LOOKBACK_SZ: usize> LZOutputBuf for StreamingOutputBuf<O, LOOKBACK_SZ>
where
    O: FnMut(&[u8]),
{
    fn add_lits(&mut self, lits: &[u8]) {
        (self.outp)(lits);

        if self.wptr + lits.len() > LOOKBACK_SZ {
            // wrapped
            if lits.len() < LOOKBACK_SZ {
                let len_until_end = LOOKBACK_SZ - self.wptr;
                self.buf[self.wptr..].copy_from_slice(&lits[..len_until_end]);
                self.buf[..(lits.len() - len_until_end)].copy_from_slice(&lits[len_until_end..]);
                self.wptr = (self.wptr + lits.len()) % LOOKBACK_SZ;
            } else {
                self.buf
                    .copy_from_slice(&lits[(lits.len() - LOOKBACK_SZ)..]);
                self.wptr = 0;
            }
        } else {
            // no wrap
            self.buf[self.wptr..(self.wptr + lits.len())].copy_from_slice(lits);
            self.wptr = (self.wptr + lits.len()) % LOOKBACK_SZ;
        }

        self.cur_pos += lits.len();
    }

    fn add_match(&mut self, disp: usize, len: usize) -> Result<(), ()> {
        if disp as usize > self.cur_pos {
            return Err(());
        }

        for _ in 0..len {
            let idx = (self.wptr + LOOKBACK_SZ - disp - 1) % LOOKBACK_SZ;
            let copy_b = self.buf[idx];
            (self.outp)(&[copy_b]);
            self.buf[self.wptr] = copy_b;
            self.wptr = (self.wptr + 1) % LOOKBACK_SZ;
            self.cur_pos += 1;
        }

        Ok(())
    }

    fn cur_pos(&self) -> usize {
        self.cur_pos
    }

    fn is_at_limit(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use core::pin::pin;

    use super::*;

    #[test]
    fn decompress_prealloc_buf() {
        let mut buf_ = [0u8; 4];
        let mut buf = PreallocatedBuf::from(&mut buf_[..]);

        assert_eq!(buf.cur_pos(), 0);
        buf.add_lits(&[0x11]);
        assert_eq!(buf.cur_pos(), 1);
        buf.add_lits(&[0x22]);
        assert_eq!(buf.cur_pos(), 2);
        buf.add_lits(&[0x33]);
        assert_eq!(buf.cur_pos(), 3);
        buf.add_lits(&[0x44, 0x55]);

        assert!(buf.is_at_limit());
        assert_eq!(buf_, [0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn decompress_prealloc_buf_match() {
        let mut buf_ = [0u8; 8];
        let mut buf = PreallocatedBuf::from(&mut buf_[..]);

        buf.add_lits(&[0x11]);
        buf.add_lits(&[0x22]);
        buf.add_lits(&[0x33]);
        buf.add_match(2, 7).unwrap();

        assert_eq!(buf_, [0x11, 0x22, 0x33, 0x22, 0x33, 0x22, 0x33, 0x22]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decompress_vec_buf() {
        let mut buf = VecBuf::new(0, 4);

        assert_eq!(buf.cur_pos(), 0);
        buf.add_lits(&[0x11]);
        assert_eq!(buf.cur_pos(), 1);
        buf.add_lits(&[0x22]);
        assert_eq!(buf.cur_pos(), 2);
        buf.add_lits(&[0x33]);
        assert_eq!(buf.cur_pos(), 3);
        buf.add_lits(&[0x44, 0x55]);

        assert!(buf.is_at_limit());
        let buf: Vec<_> = buf.into();
        assert_eq!(buf, [0x11, 0x22, 0x33, 0x44]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn decompress_vec_buf_match() {
        let mut buf = VecBuf::new(0, 8);

        buf.add_lits(&[0x11]);
        buf.add_lits(&[0x22]);
        buf.add_lits(&[0x33]);
        buf.add_match(2, 7).unwrap();

        let buf: Vec<_> = buf.into();
        assert_eq!(buf, [0x11, 0x22, 0x33, 0x22, 0x33, 0x22, 0x33, 0x22]);
    }

    #[test]
    fn decompress_stream_buf() {
        let mut xbuf = Vec::new();
        let mut buf = StreamingOutputBuf::<_, 4>::new(|x| xbuf.extend_from_slice(x));

        buf.add_lits(&[0x11]);
        assert_eq!(buf.buf, [0x11, 0, 0, 0]);
        assert_eq!(buf.wptr, 1);

        buf.add_lits(&[0x22, 0x33]);
        assert_eq!(buf.buf, [0x11, 0x22, 0x33, 0]);
        assert_eq!(buf.wptr, 3);

        buf.add_lits(&[0x44, 0x55, 0x66]);
        assert_eq!(buf.buf, [0x55, 0x66, 0x33, 0x44]);
        assert_eq!(buf.wptr, 2);

        buf.add_lits(&[0x77, 0x88, 0x99, 0xaa, 0xbb]);
        assert_eq!(buf.buf, [0x88, 0x99, 0xaa, 0xbb]);
        assert_eq!(buf.wptr, 0);

        assert_eq!(
            xbuf,
            [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]
        );
    }

    #[test]
    fn decompress_stream_buf_match() {
        let mut xbuf = Vec::new();
        let mut buf = StreamingOutputBuf::<_, 4>::new(|x| xbuf.extend_from_slice(x));

        buf.add_lits(&[0x11, 0x22, 0x33]);

        buf.add_match(0, 2).unwrap();
        assert_eq!(buf.buf, [0x33, 0x22, 0x33, 0x33]);
        assert_eq!(buf.wptr, 1);

        buf.add_match(3, 4).unwrap();
        assert_eq!(buf.buf, [0x33, 0x22, 0x33, 0x33]);
        assert_eq!(buf.wptr, 1);

        assert_eq!(xbuf, [0x11, 0x22, 0x33, 0x33, 0x33, 0x22, 0x33, 0x33, 0x33]);
    }

    async fn testfunc(
        peek1: InputPeeker<'_, '_, 8, 1>,
        peek2: InputPeeker<'_, '_, 8, 2>,
    ) -> Result<(), ()> {
        let peeked1 = (&peek1).await;
        assert_eq!(peeked1, [1]);
        let peeked2 = (&peek2).await;
        assert_eq!(peeked2, [2, 3]);

        let peeked1 = (&peek1).await;
        assert_eq!(peeked1, [4]);
        let peeked2 = (&peek2).await;
        assert_eq!(peeked2, [5, 6]);

        Ok(())
    }

    #[test]
    fn async_hax_test() {
        let state = StreamingDecompressInnerState::<8>::new();
        let peek1 = state.get_peeker::<1>();
        let peek2 = state.get_peeker::<2>();
        let x = pin!(testfunc(peek1, peek2));
        let mut state = StreamingDecompressState::new(&state, x);

        let a = state.add_inp(&[1, 2, 3, 4]);
        assert_eq!(a, Ok(2));

        let a = state.add_inp(&[5]);
        assert_eq!(a, Ok(1));

        let a = state.add_inp(&[6, 7, 8]);
        assert_eq!(a, Ok(0));
    }
}
