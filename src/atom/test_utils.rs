//! Test utilities for atom round-trip testing

use crate::atom::FourCC;
use crate::parser::ParseAtom;
use crate::writer::SerializeAtom;
use anyhow::Result;
use futures_util::io::Cursor;
use std::fs;
use std::path::PathBuf;

/// Discovers all test data files for a given filename_prefix and test-data/{path}
///
/// # Returns
/// A vector of file paths matching the pattern `{atom_type}{i:02}.bin`
pub fn discover_test_files(filename_prefix: &str, path: Option<&str>) -> Vec<String> {
    let mut test_data_dir = PathBuf::new();
    test_data_dir.push("test-data");
    if let Some(path) = path {
        test_data_dir = test_data_dir.join(path);
    }

    if !test_data_dir.exists() {
        return vec![];
    }

    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(test_data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with(filename_prefix) && file_name.ends_with(".bin") {
                    // Check if it matches the pattern {atom_type}{i:02}.bin
                    let expected_prefix = filename_prefix.to_string();
                    if file_name.len() == expected_prefix.len() + 6 // {i:02}.bin
                        && file_name[expected_prefix.len()..expected_prefix.len() + 2]
                            .chars()
                            .all(|c| c.is_ascii_digit())
                    {
                        files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    files.sort();
    files
}

/// Discovers all test data files for a given atom type
///
/// # Arguments
/// * `atom_type` - The 4-character atom type (e.g., "ilst", "ftyp", "dref")
///
/// # Returns
/// A vector of file paths matching the pattern `{atom_type}{i:02}.bin`
pub fn discover_test_files_default_dir(atom_type: &str) -> Vec<String> {
    discover_test_files(atom_type, None)
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

    let test_files = discover_test_files_default_dir(atom_type_str);

    if test_files.is_empty() {
        println!("No test files found for atom type '{}'", atom_type_str);
        return Ok(());
    }

    println!(
        "Testing {} files for atom type '{}'",
        test_files.len(),
        atom_type_str
    );

    for file_path in test_files {
        println!("Testing file: {}", file_path);
        test_single_file::<T>(atom_type_bytes, &file_path).await?;
    }

    Ok(())
}

/// Tests a single file for round-trip consistency
async fn test_single_file<T>(atom_type_bytes: &[u8; 4], file_path: &str) -> Result<()>
where
    T: ParseAtom + SerializeAtom + Send + Clone + std::fmt::Debug,
{
    let binary_data =
        fs::read(file_path).unwrap_or_else(|_| panic!("Failed to read test file: {}", file_path));

    if binary_data.len() < 8 {
        panic!("Test file too small (< 8 bytes): {}", file_path);
    }

    // Skip the atom size (4 bytes) and fourcc (4 bytes) to get just the atom data
    let atom_data = &binary_data[8..];

    // Parse the atom data
    let fourcc = FourCC::from(*atom_type_bytes);
    let cursor = Cursor::new(atom_data);
    let parsed_atom = T::parse(fourcc, cursor)
        .await
        .unwrap_or_else(|e| panic!("Failed to parse atom from {}: {:#?}", file_path, e));

    // Serialize the atom back to bytes
    let re_encoded = parsed_atom.clone().into_body_bytes();

    let cursor = Cursor::new(re_encoded.clone());
    match T::parse(fourcc, cursor).await {
        Ok(parsed_atom2) => {
            // println!("re-encoded parsed: {parsed_atom2:#?}");
        }
        Err(_) => {
            println!("{parsed_atom:#?}");
        }
    }

    // Compare data in chunks for better error reporting
    const CHUNK_SIZE: usize = 200;
    for ((i, left), right) in re_encoded
        .chunks(CHUNK_SIZE)
        .enumerate()
        .zip(atom_data.chunks(CHUNK_SIZE))
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
            let original_len = atom_data.len();
            let delta = re_encoded_len.max(original_len) - re_encoded_len.min(original_len);

            println!(
                "Round-trip failed for {file_path} at range [{mismatch_range_start}..{mismatch_range_end}] (left.len()={re_encoded_len}, right.len()={original_len}, delta={delta})\nOriginal:   {:02X?}{:02X?}{:02X?}\nRe-encoded: {:02X?}{:02X?}{:02X?}",
                &right[0..local_mismatch_index],
                &right[local_mismatch_index..mismatch_len],
                &right[mismatch_len..],
                &left[0..local_mismatch_index],
                &left[local_mismatch_index..mismatch_len],
                &left[mismatch_len..],
            );

            panic!("left != right");
        }
    }

    println!("✓ {} passed round-trip test", file_path);
    Ok(())
}

pub fn test_stsd_extension_roundtrip(typ: &[u8; 4]) {
    use winnow::Parser;

    use crate::atom::{
        stsd::extension::parser::parse_stsd_extension,
        util::{parser::stream, serializer::SizeU32},
    };

    let typ = FourCC(*typ);
    let typ_string = typ.to_string();

    let files = discover_test_files(&typ_string, Some("stsd"));
    for file in files.iter() {
        eprintln!("Testing {file}...");

        let data = fs::read(file).expect(format!("error reading {file}").as_str());

        let parsed = parse_stsd_extension(typ)
            .parse(stream(&data))
            .expect(format!("error parsing {file}").as_str());

        let re_encoded = parsed.to_bytes::<SizeU32>();

        assert_bytes_equal(&re_encoded, &data);
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_test_files() {
        let files = discover_test_files_default_dir("ilst");
        // Should find at least ilst00.bin and ilst01.bin if they exist
        assert!(files.iter().any(|f| f.contains("ilst00.bin")) || files.is_empty());

        // Test files should be sorted
        let mut sorted_files = files.clone();
        sorted_files.sort();
        assert_eq!(files, sorted_files);
    }

    #[test]
    fn test_discover_nonexistent_files() {
        let files = discover_test_files_default_dir("nonexistent");
        assert!(files.is_empty());
    }
}
