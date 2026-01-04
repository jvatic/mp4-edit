use std::{fs::File, io, path::PathBuf, process::Stdio};

use anyhow::{bail, Context};
use bon::Builder;
use escargot::CargoBuild;
use tempfile::NamedTempFile;

#[derive(Builder)]
struct ExampleTestCase {
    example: &'static str,
    input: &'static str,
    #[builder(default = false)]
    stdin: bool,
    #[builder(default = false)]
    stdout: bool,
    #[builder(default = Vec::new())]
    additional_args: Vec<&'static str>,
    expected_hash: &'static str,
}

macro_rules! test_example {
    (@example($example:literal), $(
        $name:ident {
            $( $field:ident: $value:expr ),+ $(,)?
        }
    )* => @sha256( $hash:literal ) ) => {
        $(
            test_example!(@inner $name {
                $( $field: $value ),+,
                example: $example
            } => $hash );
        )*
    };

    (@inner $name:ident {
        $( $field:ident: $value:expr ),+
    } => $hash:literal ) => {
        #[test]
        fn $name() {
            let test_case = ExampleTestCase::builder()
                .$( $field($value) ).+
                .expected_hash($hash)
                .build();

            test_example(test_case);
        }
    };
}

test_example!(
    @example("mp4copy"),

    mp4copy_file_to_file {
        input: "./test-data/m4b/AliceInWonderland_librivox_2_chapters.m4b",
    }
    mp4copy_stdin_to_file {
        input: "./test-data/m4b/AliceInWonderland_librivox_2_chapters.m4b",
        stdin: true,
    }
    mp4copy_file_to_stdout {
        input: "./test-data/m4b/AliceInWonderland_librivox_2_chapters.m4b",
        stdout: true,
    }
    mp4copy_stdin_to_stdout {
        input: "./test-data/m4b/AliceInWonderland_librivox_2_chapters.m4b",
        stdin: true,
        stdout: true,
    }

    => @sha256("db8470b4fbf813056aaf452b8ccb3e8ed4443a2f8cfeeeb795ccaae995a939c6")
);

fn test_example(test_case: ExampleTestCase) {
    let input_path = if test_case.stdin {
        "-"
    } else {
        test_case.input
    };

    let stdin = if test_case.stdin {
        // pipe input file through stdin
        Stdio::from(File::open(test_case.input).expect("error opening input file"))
    } else {
        Stdio::null()
    };

    let output_file = NamedTempFile::new().expect("error crating temp file");
    let output_path = if test_case.stdout {
        "-"
    } else {
        output_file.path().to_str().unwrap()
    };

    let stdout = if test_case.stdout {
        // pipe stdout into the temp file
        Stdio::from(
            File::options()
                .write(true)
                .truncate(true)
                .open(output_file.path())
                .expect("error opening output temp file"),
        )
    } else {
        Stdio::null()
    };

    let status = CargoBuild::new()
        .example(test_case.example)
        .run()
        .expect("error building example")
        .command()
        .arg(input_path)
        .arg(&output_path)
        .args(test_case.additional_args)
        .stdin(stdin)
        .stdout(stdout)
        .status()
        .expect("failed to run example");
    assert!(status.success(), "failed to run example");

    let output_path = output_file.path();
    let output_file = File::open(&output_path).expect("error opening output file");
    match assert_file_hash(output_file, test_case.expected_hash) {
        Ok(()) => {}
        Err(err) => {
            let mut inspect_path = PathBuf::from(test_case.input);
            let file_name = format!(
                "inspect_{}_{}",
                test_case.example,
                inspect_path
                    .file_name()
                    .expect("input file should have a file name")
                    .to_string_lossy()
                    .to_string()
            );
            inspect_path.set_file_name(file_name);
            std::fs::copy(output_path, inspect_path)
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
        bail!("left != right, expected {expected_hash}, got {file_hash}");
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
