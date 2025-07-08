//! Test utilities for atom round-trip testing

use crate::atom::FourCC;
use crate::parser::Parse;
use crate::writer::SerializeAtom;
use anyhow::Result;
use futures_util::io::Cursor;
use std::fs;
use std::path::Path;

/// Discovers all test data files for a given atom type
///
/// # Arguments
/// * `atom_type` - The 4-character atom type (e.g., "ilst", "ftyp", "dref")
///
/// # Returns
/// A vector of file paths matching the pattern `{atom_type}{i:02}.bin`
pub fn discover_test_files(atom_type: &str) -> Vec<String> {
    let test_data_dir = Path::new("test-data");

    if !test_data_dir.exists() {
        return vec![];
    }

    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(test_data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with(atom_type) && file_name.ends_with(".bin") {
                    // Check if it matches the pattern {atom_type}{i:02}.bin
                    let expected_prefix = format!("{}", atom_type);
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
/// use mp4_parser::atom::test_utils::test_atom_roundtrip;
/// use mp4_parser::atom::ilst::{ItemListAtom, ILST};
///
/// test_atom_roundtrip::<ItemListAtom>(ILST).unwrap();
/// ```
pub async fn test_atom_roundtrip<T>(atom_type_bytes: &[u8; 4]) -> Result<()>
where
    T: Parse + SerializeAtom + Send,
{
    let atom_type_str = std::str::from_utf8(atom_type_bytes)
        .unwrap_or_else(|_| panic!("Invalid atom type bytes: {:?}", atom_type_bytes));

    let test_files = discover_test_files(atom_type_str);

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
    T: Parse + SerializeAtom + Send,
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
        .unwrap_or_else(|e| panic!("Failed to parse atom from {}: {}", file_path, e));

    // Serialize the atom back to bytes
    let re_encoded = parsed_atom.into_body_bytes();

    // Compare the original and re-encoded data
    if atom_data.len() != re_encoded.len() {
        panic!(
            "Round-trip length mismatch for {}: original={}, re-encoded={}",
            file_path,
            atom_data.len(),
            re_encoded.len()
        );
    }

    // Compare data in chunks for better error reporting
    const CHUNK_SIZE: usize = 200;
    for ((i, left), right) in re_encoded
        .chunks(CHUNK_SIZE)
        .enumerate()
        .zip(atom_data.chunks(CHUNK_SIZE))
    {
        if left != right {
            panic!(
                "Round-trip failed for {} at range [{}..{}] (left.len()={}, right.len()={})\nOriginal: {:02X?}\nRe-encoded: {:02X?}",
                file_path,
                i * CHUNK_SIZE,
                ((i + 1) * CHUNK_SIZE).min(atom_data.len()),
                re_encoded.len(),
                atom_data.len(),
                right,
                left
            );
        }
    }

    println!("✓ {} passed round-trip test", file_path);
    Ok(())
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
/// use mp4_parser::atom::test_utils::test_atom_roundtrip_sync;
/// use mp4_parser::atom::ilst::{ItemListAtom, ILST};
///
/// #[test]
/// fn test_ilst_roundtrip() {
///     test_atom_roundtrip_sync::<ItemListAtom>(ILST);
/// }
/// ```
pub fn test_atom_roundtrip_sync<T>(atom_type_bytes: &[u8; 4])
where
    T: Parse + SerializeAtom + Send,
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
        let files = discover_test_files("ilst");
        // Should find at least ilst00.bin and ilst01.bin if they exist
        assert!(files.iter().any(|f| f.contains("ilst00.bin")) || files.is_empty());

        // Test files should be sorted
        let mut sorted_files = files.clone();
        sorted_files.sort();
        assert_eq!(files, sorted_files);
    }

    #[test]
    fn test_discover_nonexistent_files() {
        let files = discover_test_files("nonexistent");
        assert!(files.is_empty());
    }
}
