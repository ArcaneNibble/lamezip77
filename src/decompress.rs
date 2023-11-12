extern crate std;
use core::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};
use std::println;

const DUMMY_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |_| RawWaker::new(core::ptr::null(), &DUMMY_WAKER_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);
const DUMMY_WAKER: RawWaker = RawWaker::new(core::ptr::null(), &DUMMY_WAKER_VTABLE);

struct InputWithBuf<'a, const BUFSZ: usize> {
    buf: [u8; BUFSZ],
    nbuf: usize,
    inp: &'a [u8],
}

impl<'a, const BUFSZ: usize> InputWithBuf<'a, BUFSZ> {
    fn new() -> Self {
        Self {
            buf: [0; BUFSZ],
            nbuf: 0,
            inp: &[],
        }
    }

    fn add_inp<'inp: 'a>(&mut self, inp: &'inp [u8]) {
        debug_assert!(self.inp.len() == 0);
        self.inp = inp;
    }
}

struct InputPeeker<'a, 'b, 'c, const BUFSZ: usize, const PEEKSZ: usize> {
    buf: &'a RefCell<InputWithBuf<'b, BUFSZ>>,
    nwaiting: &'c Cell<usize>,
}

impl<'a, 'b, 'c, const BUFSZ: usize, const PEEKSZ: usize> InputPeeker<'a, 'b, 'c, BUFSZ, PEEKSZ> {
    fn new(
        input_with_buf: &'a RefCell<InputWithBuf<'b, BUFSZ>>,
        nwaiting: &'c Cell<usize>,
    ) -> Self {
        Self {
            buf: input_with_buf,
            nwaiting,
        }
    }
}

impl<'a, 'b, 'c, const BUFSZ: usize, const PEEKSZ: usize> Future
    for &InputPeeker<'a, 'b, 'c, BUFSZ, PEEKSZ>
{
    type Output = [u8; PEEKSZ];

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // FIXME optimize copying?
        let s = self.get_mut();
        let mut input_with_buf = s.buf.borrow_mut();
        let (mut buf, nbuf, inp) = (input_with_buf.buf, input_with_buf.nbuf, input_with_buf.inp);
        let n_missing = BUFSZ - nbuf;
        let n_to_copy = core::cmp::min(n_missing, inp.len());
        println!("missing {} avail to copy {}", n_missing, n_to_copy);
        buf[nbuf..(nbuf + n_to_copy)].copy_from_slice(&inp[..n_to_copy]);
        input_with_buf.buf = buf;
        input_with_buf.inp = &inp[n_to_copy..];
        input_with_buf.nbuf += n_to_copy;

        if PEEKSZ <= input_with_buf.nbuf {
            let mut data = [0; PEEKSZ];
            data.copy_from_slice(&input_with_buf.buf[..PEEKSZ]);
            input_with_buf.buf.copy_within(PEEKSZ.., 0);
            input_with_buf.nbuf -= PEEKSZ;
            s.nwaiting.set(0);
            Poll::Ready(data)
        } else {
            let nwaiting = PEEKSZ - input_with_buf.nbuf;
            s.nwaiting.set(nwaiting);
            Poll::Pending
        }
    }
}

async fn testfunc(
    peek1: InputPeeker<'_, '_, '_, 8, 1>,
    peek2: InputPeeker<'_, '_, '_, 8, 2>,
) -> Result<(), ()> {
    println!("in async");

    let peeked1 = (&peek1).await;
    println!("peek 1: {:?}", peeked1);
    let peeked2 = (&peek2).await;
    println!("peek 2: {:?}", peeked2);

    let peeked1 = (&peek1).await;
    println!("peek 1: {:?}", peeked1);
    let peeked2 = (&peek2).await;
    println!("peek 2: {:?}", peeked2);

    Ok(())
}

struct StreamingDecompressExecState<'a, const BUFSZ: usize> {
    inp: RefCell<InputWithBuf<'a, BUFSZ>>,
    nwaiting: Cell<usize>,
}

impl<'a, const BUFSZ: usize> StreamingDecompressExecState<'a, BUFSZ> {
    fn new() -> Self {
        Self {
            inp: RefCell::new(InputWithBuf::new()),
            nwaiting: Cell::new(0),
        }
    }

    fn get_peeker<const N: usize>(&self) -> InputPeeker<'_, 'a, '_, BUFSZ, N> {
        InputPeeker::new(&self.inp, &self.nwaiting)
    }

    fn add_inp(&self, inp: &'a [u8]) {
        let mut x = self.inp.borrow_mut();
        x.add_inp(inp);
    }
}

struct StreamingDecompressState<'a, 'b, F, E, const BUFSZ: usize>
where
    F: Future<Output = Result<(), E>>,
{
    s: &'a StreamingDecompressExecState<'b, BUFSZ>,
    f: Pin<&'a mut F>,
}

impl<'a, 'b, F, E, const BUFSZ: usize> StreamingDecompressState<'a, 'b, F, E, BUFSZ>
where
    F: Future<Output = Result<(), E>>,
{
    fn new(s: &'a StreamingDecompressExecState<'b, BUFSZ>, f: Pin<&'a mut F>) -> Self {
        Self { s, f }
    }

    fn add_inp(&mut self, inp: &'b [u8]) -> Result<usize, E> {
        self.s.add_inp(inp);

        let waker = unsafe { &Waker::from_raw(DUMMY_WAKER) };
        let mut cx = Context::from_waker(waker);
        let poll_result = self.f.as_mut().poll(&mut cx);

        match poll_result {
            Poll::Ready(result) => result.map(|_| 0),
            Poll::Pending => Ok(self.s.nwaiting.get()),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use core::pin::pin;
    use std::println;

    use super::*;

    #[test]
    fn async_hax() {
        let state = StreamingDecompressExecState::<'_, 8>::new();
        let peek1 = state.get_peeker::<1>();
        let peek2 = state.get_peeker::<2>();
        let x = pin!(testfunc(peek1, peek2));
        let mut state = StreamingDecompressState::new(&state, x);

        let a = state.add_inp(&[1, 2, 3, 4]);
        println!("{:?}", a);

        let a = state.add_inp(&[5]);
        println!("{:?}", a);

        let a = state.add_inp(&[6, 7, 8]);
        println!("{:?}", a);
    }
}
