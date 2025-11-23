use std::{fs::File, io};

use anyhow::{bail, Context};
use escargot::CargoBuild;
use tempfile::NamedTempFile;

#[test]
fn test_mp4copy_example() {
    let mp4_copy_in_path = "./test-data/m4b/AliceInWonderland_librivox.m4b";
    let mp4_copy_actual_out = NamedTempFile::new().expect("error crating temp file");
    let mp4_copy_out_path = mp4_copy_actual_out.path().to_owned();

    let status = CargoBuild::new()
        .example("mp4copy")
        .release()
        .run()
        .expect("error building mp4copy example")
        .command()
        .arg(mp4_copy_in_path)
        .arg(&mp4_copy_out_path)
        .status()
        .expect("failed to run mp4copy example");
    assert!(status.success(), "failed to run mp4copy example");

    let mp4_copy_out_file = File::open(&mp4_copy_out_path).expect("error opening output file");
    match assert_file_hash(
        mp4_copy_out_file,
        "1ba22e284ee80e8cf25077bc8f10b81f0fadef6b8e61ac7964f059402e8ee1d3",
    ) {
        Ok(()) => {}
        Err(err) => {
            std::fs::copy(mp4_copy_out_path, mp4_copy_in_path)
                .inspect_err(|err| {
                    eprintln!("error copying output file for inspection: {err}");
                })
                .ok();
            panic!("{err}");
        }
    }
}

fn assert_file_hash(file: File, expected_hash: &str) -> anyhow::Result<()> {
    let file_hash = hash_file(file).context("error hashing file")?;

    if file_hash != expected_hash {
        bail!("left != right, expected {file_hash}");
    }

    Ok(())
}

fn hash_file(mut file: File) -> io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io;

    let mut hasher = Sha256::new();

    io::copy(&mut file, &mut hasher)?;

    Ok(format!("{:x}", hasher.finalize()))
}
