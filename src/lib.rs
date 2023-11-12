#![no_std]

//! Univeral(-ish) LZ77 thing

mod decompress;
pub use decompress::{
    InputPeeker, LZOutputBuf, PreallocatedBuf, StreamingDecompressInnerState,
    StreamingDecompressState, StreamingOutputBuf,
};

#[cfg(feature = "alloc")]
pub use decompress::VecBuf;

mod sliding_window;

mod hashtables;

mod lz77;
pub use lz77::{LZEngine, LZOutput, LZSettings};

pub mod deflate;
pub mod fastlz;
pub mod lz4;
pub mod nintendo_lz;
