#![no_std]

mod decompress;
pub use decompress::{LZOutputBuf, PreallocatedBuf, StreamingDecompressInnerState, VecBuf};

mod sliding_window;

mod hashtables;

mod lz77;
pub use lz77::{LZEngine, LZOutput, LZSettings};

mod util;

pub mod deflate;
pub mod fastlz;
pub mod lz4;
pub mod nintendo_lz;
