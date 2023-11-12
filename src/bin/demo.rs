use std::{
    env,
    error::Error,
    ffi::OsString,
    fs::File,
    io::{BufWriter, Write},
};

use lamezip77::{deflate, fastlz, lz4, nintendo_lz};

#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn Error>> {
    use lamezip77::VecBuf;

    let args: Vec<OsString> = env::args_os().collect();

    if args.len() < 5 {
        println!(
            "Usage: {} c|d format input output",
            args[0].to_string_lossy()
        );
        return Ok(());
    }

    let mode = &args[1];
    let format = &args[2];
    let inp_fn = &args[3];
    let outp_fn = &args[4];

    let inp = std::fs::read(inp_fn)?;
    let mut outp;

    match mode.to_str() {
        Some("c") => match format.to_str() {
            Some("nintendo") => {
                let mut cmp = nintendo_lz::Compress::new_boxed();
                outp = Vec::new();
                outp.extend_from_slice(&cmp.encode_header(inp.len() as u32));
                cmp.compress(false, &inp, true, |x| {
                    outp.push(x);
                });
            }
            Some("nintendo_vram") => {
                let mut cmp = nintendo_lz::Compress::new_boxed();
                outp = Vec::new();
                outp.extend_from_slice(&cmp.encode_header(inp.len() as u32));
                cmp.compress(true, &inp, true, |x| {
                    outp.push(x);
                });
            }
            Some("fastlz1") => {
                let mut cmp = fastlz::CompressLevel1::new_boxed();
                outp = Vec::new();
                cmp.compress(&inp, true, |x| {
                    outp.push(x);
                });
            }
            Some("fastlz2") => {
                let mut cmp = fastlz::CompressLevel2::new_boxed();
                outp = Vec::new();
                cmp.compress(&inp, true, |x| {
                    outp.push(x);
                });
            }
            Some("lz4") => {
                let mut cmp = lz4::Compress::<65536>::new_boxed();
                outp = Vec::new();
                cmp.compress(&inp, true, |x| {
                    outp.push(x);
                })
                .unwrap();
            }
            Some("deflate") => {
                let mut cmp = deflate::Compress::<32768>::new_boxed();
                outp = Vec::new();
                cmp.compress(&inp, true, |x| {
                    outp.push(x);
                });
            }
            _ => {
                println!("Invalid format {}", format.to_string_lossy());
                return Ok(());
            }
        },
        Some("d") => match format.to_str() {
            Some("nintendo") => {
                let mut outvec = VecBuf::new(0, usize::MAX);
                {
                    lamezip77::nintendo_lz::decompress_make!(dec, &mut outvec);
                    let ret = dec.add_inp(&inp)?;
                    assert_eq!(ret, 0);
                }
                outp = outvec.into();
            }
            Some("fastlz") => {
                let dec = fastlz::DecompressBuffered::new();
                outp = dec.decompress_new(&inp, usize::MAX)?;
            }
            Some("lz4") => {
                let dec = lz4::DecompressBuffered::new();
                outp = dec.decompress_new(&inp, usize::MAX)?;
            }
            Some("deflate") => {
                let mut dec = deflate::DecompressBuffered::new();
                outp = dec.decompress_new(&inp, usize::MAX)?;
            }
            _ => {
                println!("Invalid format {}", format.to_string_lossy());
                return Ok(());
            }
        },
        _ => {
            println!("Invalid mode {}", mode.to_string_lossy());
            return Ok(());
        }
    }

    let mut outp_f = BufWriter::new(File::create(&outp_fn).unwrap());
    outp_f.write(&outp).unwrap();

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    println!("Demo requires std feature");
}
