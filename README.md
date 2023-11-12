# Univeral(-ish) LZ77 thing

Attempting to make a single library that does all of
* DEFLATE
* LZ4
* FastLZ
* Nintendo

This crate is specifically designed to work in a `no_std` environment, including not requiring any runtime allocations. However, as of this writing, it makes tradeoffs that consume extra memory such that it may or may not fit on microcontrollers.

This crate is also designed to work in a "streaming" fashion, where input/output data are supplied/generated small pieces at a time, with state being saved in a data structure between calls. As such, it also assumes a non-seekable output sink.

## Algorithm

This crate roughly implements the exact same algorithm used by gzip for LZ77 compression, involving a chained hash table with keys computed from the first few bytes of a string. However, it is not a direct port and does not generate bit-identical output. This algorithm is used for *every* output format (which makes it significantly slower than the reference implementation of e.g. FastLZ, but occasionally producing better compression ratios).

Unlike gzip, this crate uses the package-merge binary-coin-collector-problem algorithm to generate a length-limited Huffman code.

## Performance

This crate has not been optimized for performance, but it appears to run significantly slower.

## Notes on workarounds

This crate ended up encountering *numerous* Rust limitations. Some of the most notable hack workarounds are documented here.

### Const generics

Because the LZ77 engine is shared between all formats, it is heavily parameterized with const generics. However, `min_const_generics` does not allow *any* form of computation involving const generics, even when calculating e.g. array sizes.

The workaround is to pass in all of the required numbers down from the outermost non-generic level and write asserts in all relevant `new` functions to ensure that they indeed have the value that they must have (e.g. that `MAX_LEN_PLUS_ONE` indeed equals `MAX_LEN + 1`).

### Coroutines

In order to significantly simplify writing the decompression functions, it would be nice to write them in the straightforward "get input, process, write output, loop until done" fashion. However, this would not support the desired "streaming" use cases. Ideally, it would be nice to have the compiler automatically perform a "stackless coroutine transform" to convert between the two.

Currently, the coroutine API is not stable in Rust, *but async is*. In fact, async is built on top of the unstable API and desugars into it. Thus, by heavily abusing async functions, the compiler will perform the desired transformation for us.

To make this work, this crate includes just enough of a fake async executor to poll decompress function futures until they block needing more input (wakers are entirely ignored). This executor contains a back-channel between it and the `InputPeeker` future, so, when more input is supplied, it is made available for `InputPeeker`s to read.

For ergonomics reasons, this requires a tiny amount of unsafe code where we promise to the compiler that the passed-in data is no longer being retained by the time the decompression function has run out of input. Without this, it becomes impossible to e.g. supply input data inside a loop.

#### Creating a coroutine

Because this library is built for `no_std` use cases without a heap, decompress state is stored on the stack. However, the only way the construction of this state can be abstracted away from the user is to somehow create an unhygienic macro (otherwise, the needed objects will go out of scope and be dropped). This is done using a proc macro, as normal macros cannot do it.
