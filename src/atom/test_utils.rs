//! Test utilities for atom round-trip testing

use crate::atom::FourCC;
use crate::parser::ParseAtom;
use crate::writer::SerializeAtom;
use anyhow::{Context, Result};
use futures_util::io::Cursor;
use std::fs;

pub mod test_file {
    use std::{
        fmt, fs,
        path::{Path, PathBuf},
    };

    use anyhow::{anyhow, Context};

    pub trait Matcher {
        fn dirname(&self) -> Option<&str> {
            None
        }

        fn match_file(&self, file_name: &str) -> bool;

        /// Given an input file name, return the output file name to look for.
        ///
        /// If `None`, it's assumed that the input file should be used for the expected output.
        ///
        /// NOTE: if the returned file name doesn't exist, the input file will be used same as if `None` had been returned.
        fn output_file(&self, file_name: &str) -> Option<PathBuf> {
            let output_file_name = PathBuf::from(file_name);
            match output_file_name.extension() {
                Some(ext) => {
                    let output_file_name =
                        output_file_name.with_extension("out").with_extension(ext);
                    Some(output_file_name)
                }
                None => None,
            }
        }
    }

    /// Matches paths following the pattern `{atom_type}{i:02}.bin`
    pub struct AtomMatcher<'a> {
        atom_type: &'a str,
    }

    impl<'a> AtomMatcher<'a> {
        pub fn new(atom_type: &'a str) -> Self {
            Self { atom_type }
        }
    }

    impl<'a> Matcher for AtomMatcher<'a> {
        fn match_file(&self, file_name: &str) -> bool {
            let atom_type = self.atom_type;
            if file_name.starts_with(atom_type) && file_name.ends_with(".bin") {
                // Check if it matches the pattern {atom_type}{i:02}.bin
                let expected_prefix = atom_type.to_string();
                if file_name.len() == expected_prefix.len() + 6 // {i:02}.bin
            && file_name[expected_prefix.len()..expected_prefix.len() + 2]
                .chars()
                .all(|c| c.is_ascii_digit())
                {
                    return true;
                }
            }
            false
        }
    }

    #[derive(Clone, PartialEq, Eq)]
    pub struct TestCase {
        input_file: PathBuf,
        expected_output_file: Option<PathBuf>,
    }

    impl fmt::Display for TestCase {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_fmt(format_args!("{}", self.input_file.display()))
        }
    }

    impl PartialOrd for TestCase {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.input_file.partial_cmp(&other.input_file)
        }
    }

    impl Ord for TestCase {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.input_file.cmp(&other.input_file)
        }
    }

    impl TestCase {
        fn new(input_file: PathBuf, expected_output_file: Option<PathBuf>) -> Self {
            Self {
                input_file,
                expected_output_file,
            }
        }

        pub fn input_file(&self) -> &PathBuf {
            &self.input_file
        }

        pub fn output_file(&self) -> Option<&PathBuf> {
            self.expected_output_file.as_ref()
        }
    }

    fn safe_join<P: AsRef<Path>>(left: PathBuf, right: P) -> anyhow::Result<PathBuf> {
        let right = right.as_ref();
        let res = left.join(right).canonicalize().context("invalid path")?;
        if !res.starts_with(left) {
            return Err(anyhow!("path escapes base dir: {}", right.display()));
        }
        Ok(res)
    }

    /// Discovers all test data files for a given [`Matcher`].
    pub fn discover(matcher: impl Matcher) -> Vec<TestCase> {
        let mut test_data_dir = PathBuf::new()
            .join("test-data")
            .canonicalize()
            .expect("invalid path");
        if let Some(path) = matcher.dirname() {
            test_data_dir = safe_join(test_data_dir, path).expect("invalid path");
        }

        assert!(test_data_dir.exists(), "test-data dir doesn't exist");

        let mut files = Vec::new();

        if let Ok(entries) = fs::read_dir(test_data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if matcher.match_file(file_name) {
                        let expected_output_file = match matcher.output_file(file_name) {
                            Some(p) => match fs::exists(&p) {
                                Ok(true) => Some(p),
                                Ok(false) | Err(_) => None,
                            },
                            None => None,
                        };
                        files.push(TestCase::new(path, expected_output_file));
                    }
                }
            }
        }

        files.sort();
        files
    }

    /// Discovers all test data files for a given atom type (e.g., `ilst`, `ftyp`, `dref`)
    pub fn discover_atom(atom_type: &str) -> Vec<TestCase> {
        discover(AtomMatcher::new(atom_type))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_discover_test_files() {
            let files = discover_atom("ilst")
                .into_iter()
                .map(|tc| tc.input_file().to_string_lossy().to_string())
                .collect::<Vec<_>>();

            // Should find at least ilst00.bin and ilst01.bin if they exist
            assert!(files.iter().any(|f| f.contains("ilst00.bin")) || files.is_empty());

            // Test files should be sorted
            let mut sorted_files = files.clone();
            sorted_files.sort();
            assert_eq!(files, sorted_files);
        }

        #[test]
        fn test_discover_nonexistent_files() {
            let files = discover_atom("nonexistent");
            assert!(files.is_empty());
        }
    }
}

/// Performs a round-trip test for a specific atom type using all available test data
///
/// # Arguments
/// * `atom_type_bytes` - The 4-byte atom type identifier
///
/// # Type Parameters
/// * `T` - The atom type that implements both Parse and SerializeAtom
///
/// # Example
/// ```
/// use mp4_edit::atom::test_utils::test_atom_roundtrip;
/// use mp4_edit::atom::ilst::{ItemListAtom, ILST};
///
/// test_atom_roundtrip::<ItemListAtom>(ILST).unwrap();
/// ```
pub async fn test_atom_roundtrip<T>(atom_type_bytes: &[u8; 4]) -> Result<()>
where
    T: ParseAtom + SerializeAtom + Send + Clone + std::fmt::Debug,
{
    let atom_type_str = std::str::from_utf8(atom_type_bytes)
        .unwrap_or_else(|_| panic!("Invalid atom type bytes: {:?}", atom_type_bytes));

    let test_cases = test_file::discover_atom(atom_type_str);

    if test_cases.is_empty() {
        println!("No test files found for atom type '{}'", atom_type_str);
        return Ok(());
    }

    println!(
        "Testing {} files for atom type '{}'",
        test_cases.len(),
        atom_type_str
    );

    for tc in test_cases {
        println!("Testing: {}", tc);
        test_atom_roundtrip_single::<T>(atom_type_bytes, tc).await?;
    }

    Ok(())
}

/// Tests a single file for round-trip consistency
async fn test_atom_roundtrip_single<T>(
    atom_type_bytes: &[u8; 4],
    tc: test_file::TestCase,
) -> Result<()>
where
    T: ParseAtom + SerializeAtom + Send + Clone + std::fmt::Debug,
{
    let input_data =
        fs::read(tc.input_file()).unwrap_or_else(|_| panic!("Failed to read test file: {}", tc));

    if input_data.len() < 8 {
        panic!("Test file too small (< 8 bytes): {}", tc);
    }

    let expected_output_data = match tc.output_file() {
        Some(output_path) => {
            let data = fs::read(output_path)
                .map(|data| data)
                .context(format!("failed to read test output file: {}", tc))
                .unwrap();
            if data.len() < 8 {
                panic!("test output file too small (< 8 bytes): {}", tc);
            }
            Some(data)
        }
        None => None,
    };

    // Skip the atom size (4 bytes) and fourcc (4 bytes) to get just the atom data
    let input_data = &input_data[8..];
    // If there's an output file, match against that, otherwise assert that output matches input
    let expected_output_data = match expected_output_data.as_ref() {
        Some(data) => &data[8..],
        None => input_data,
    };

    // Parse the atom data
    let fourcc = FourCC::from(*atom_type_bytes);
    let cursor = Cursor::new(input_data);
    let parsed_atom = T::parse(fourcc, cursor)
        .await
        .unwrap_or_else(|e| panic!("Failed to parse atom from {}: {:#?}", tc, e));

    // Serialize the atom back to bytes
    let re_encoded = parsed_atom.clone().into_body_bytes();

    let cursor = Cursor::new(re_encoded.clone());
    match T::parse(fourcc, cursor).await {
        Ok(_) => {}
        Err(_) => {
            println!("{parsed_atom:#?}");
        }
    }

    // Compare data in chunks for better error reporting
    const CHUNK_SIZE: usize = 200;
    for ((i, left), right) in re_encoded
        .chunks(CHUNK_SIZE)
        .enumerate()
        .zip(expected_output_data.chunks(CHUNK_SIZE))
    {
        let start_index = i * CHUNK_SIZE;

        if left != right {
            let mut local_mismatch_index = 0;
            for i in 0..left.len().min(right.len()) {
                if left[0..=i] != right[0..=i] {
                    local_mismatch_index = i;
                    break;
                }
            }

            let mut mismatch_len = left.len().max(right.len());
            if left.len() == right.len() {
                for i in (0..left.len()).into_iter().rev() {
                    if left[i..] != right[i..] {
                        mismatch_len = i + 1;
                        break;
                    }
                }
            }

            let mismatch_range_start = start_index + local_mismatch_index;
            let mismatch_range_end = start_index + mismatch_len;

            let re_encoded_len = re_encoded.len();
            let original_len = expected_output_data.len();
            let delta = re_encoded_len.max(original_len) - re_encoded_len.min(original_len);

            let matched_data_start = &right[0..local_mismatch_index];
            let mismatched_data_right = &right[local_mismatch_index..mismatch_len.min(right.len())];
            let mismatched_data_left = &left[local_mismatch_index..mismatch_len.min(left.len())];
            let matched_data_end = if mismatch_len > right.len() {
                &[0u8; 0]
            } else {
                &right[mismatch_len..]
            };

            println!(
                "Round-trip failed for {tc} at range [{mismatch_range_start}..{mismatch_range_end}] (left.len()={re_encoded_len}, right.len()={original_len}, delta={delta})\nOriginal:   {:02X?}{:02X?}{:02X?}\nRe-encoded: {:02X?}{:02X?}{:02X?}",
                matched_data_start,
                mismatched_data_right,
                matched_data_end,
                matched_data_start,
                mismatched_data_left,
                matched_data_end,
            );

            panic!("left != right");
        }
    }

    println!("✓ {} passed round-trip test", tc);
    Ok(())
}

// pub fn test_stsd_extension_roundtrip(typ: &[u8; 4]) {
//     use winnow::Parser;

//     use crate::atom::util::parser::stream;

//     let typ = FourCC(*typ);
//     let typ_string = typ.to_string();

//     let files = discover_test_files(&typ_string, Some("stsd"));
//     for file in files.iter() {
//         eprintln!("Testing {file}...");

//         let data = fs::read(file).expect(format!("error reading {file}").as_str());

//         let parsed = parse_stsd_extension(typ)
//             .parse(stream(&data))
//             .expect(format!("error parsing {file}").as_str());

//         let re_encoded = serialize_stsd_extension(parsed);

//         assert_bytes_equal(&re_encoded, &data);
//     }
// }

pub fn assert_bytes_equal(actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }

    // Format bytes as hex in groups of `size`
    fn format_hex_groups(size: usize, data: &[u8]) -> String {
        data.chunks(size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|b| format!("{:02X}", b))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn format_hex_groups_right_aligned(data: &[u8]) -> String {
        let mut data = data.to_vec();
        data.reverse();
        data.chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .rev()
                    .map(|b| format!("{:02X}", b))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .rev()
            .collect::<Vec<_>>()
            .join(" ")
    }

    let actual_hex = format_hex_groups(4, actual);
    let expected_hex = format_hex_groups(4, expected);

    let actual_hex_1 = format_hex_groups(1, actual);
    let expected_hex_1 = format_hex_groups(1, expected);

    let actual_right_aligned = format_hex_groups_right_aligned(actual);
    let expected_right_aligned = format_hex_groups_right_aligned(expected);

    panic!(
        "Bytes are not equal!\nActual length: {}\nExpected length: {}\nActual:   {}\nExpected: {}\n\nActual:   {:>w$}\nExpected: {:>w$}\n\nActual:   {}\nExpected: {}\n\nActual:   {:>w2$}\nExpected: {:>w2$}",
        actual.len(),
        expected.len(),
        actual_hex,
        expected_hex,
        actual_right_aligned,
        expected_right_aligned,
        actual_hex_1,
        expected_hex_1,
        actual_hex_1,
        expected_hex_1,
        w = actual_right_aligned.len().max(expected_right_aligned.len()),
        w2 = actual_hex_1.len().max(expected_hex_1.len()),
    )
}

/// Synchronous version of the round-trip test for use in `#[test]` functions
///
/// # Arguments
/// * `atom_type_bytes` - The 4-byte atom type identifier
///
/// # Type Parameters
/// * `T` - The atom type that implements both Parse and SerializeAtom
///
/// # Example
/// ```
/// use mp4_edit::atom::test_utils::test_atom_roundtrip_sync;
/// use mp4_edit::atom::ilst::{ItemListAtom, ILST};
///
/// fn test_ilst_roundtrip() {
///     test_atom_roundtrip_sync::<ItemListAtom>(ILST);
/// }
/// ```
pub fn test_atom_roundtrip_sync<T>(atom_type_bytes: &[u8; 4])
where
    T: ParseAtom + SerializeAtom + Send + Clone + std::fmt::Debug,
{
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async { test_atom_roundtrip::<T>(atom_type_bytes).await })
        .expect("Round-trip test failed");
}
