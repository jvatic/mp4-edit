use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::{parse_fixed_size_atom, FourCC};

pub const SGPD: &[u8; 4] = b"sgpd";

/// Sample Group Description Atom (sgpd) - ISO/IEC 14496-12
/// This atom defines the characteristics of sample groups.
#[derive(Debug, Clone)]
pub struct SampleGroupDescriptionAtom {
    /// Version of the sgpd atom format (0, 1, or 2+)
    pub version: u8,
    /// Flags for the atom
    pub flags: [u8; 3],
    /// Grouping type - identifies the type of grouping
    pub grouping_type: FourCC,
    /// Default length of sample group description entries (version 1 only)
    pub default_length: Option<u32>,
    /// Default sample description index (version >= 2 only)
    pub default_sample_description_index: Option<u32>,
    /// Sample group description entries
    pub entries: Vec<SampleGroupDescriptionEntry>,
}

/// A single sample group description entry
#[derive(Debug, Clone)]
pub struct SampleGroupDescriptionEntry {
    /// Length of this entry's description data (if specified)
    pub description_length: Option<u32>,
    /// The actual sample group description data
    pub description_data: Vec<u8>,
}

impl SampleGroupDescriptionAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_sample_group_description_atom(reader)
    }
}

impl TryFrom<&[u8]> for SampleGroupDescriptionAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_sample_group_description_atom(reader)
    }
}

fn parse_sample_group_description_atom<R: Read>(
    reader: R,
) -> Result<SampleGroupDescriptionAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;

    // Verify this is an sgpd atom
    if atom_type != SGPD {
        return Err(anyhow!(
            "Invalid atom type: {} (expected 'sgpd')",
            atom_type
        ));
    }

    parse_sgpd_data(&data)
}

fn parse_sgpd_data(data: &[u8]) -> Result<SampleGroupDescriptionAtom, anyhow::Error> {
    let mut cursor = Cursor::new(data);
    let mut buffer = [0u8; 4];

    // Read version and flags (4 bytes total)
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read version and flags")?;
    let version = buffer[0];
    let flags = [buffer[1], buffer[2], buffer[3]];

    // Read grouping_type (4 bytes)
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read grouping_type")?;
    let grouping_type = FourCC(buffer);

    // Version-dependent fields
    let mut default_length = None;
    let mut default_sample_description_index = None;

    match version {
        0 => {
            // No additional fields
        }
        1 => {
            // Read default_length
            cursor
                .read_exact(&mut buffer)
                .context("Failed to read default_length")?;
            default_length = Some(u32::from_be_bytes(buffer));
        }
        _ => {
            // Version 2 and above: read default_sample_description_index
            cursor
                .read_exact(&mut buffer)
                .context("Failed to read default_sample_description_index")?;
            default_sample_description_index = Some(u32::from_be_bytes(buffer));
        }
    }

    // Read entry_count
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read entry_count")?;
    let entry_count = u32::from_be_bytes(buffer);

    // Read entries
    let mut entries = Vec::new();
    for i in 0..entry_count {
        let mut description_length = None;

        // For version 1, if default_length is 0, read description_length for each entry
        if version == 1 {
            if let Some(def_len) = default_length {
                if def_len == 0 {
                    cursor.read_exact(&mut buffer).with_context(|| {
                        format!("Failed to read description_length for entry {}", i)
                    })?;
                    description_length = Some(u32::from_be_bytes(buffer));
                }
            }
        }

        // Determine the size of the description data
        let data_size = match version {
            0 => {
                // For version 0, read remaining data for this entry
                // This is tricky without knowing the exact format, so we'll read all remaining data
                // In practice, this might need more sophisticated parsing based on grouping_type
                let remaining = data.len() - cursor.position() as usize;
                if entry_count == 1 {
                    remaining
                } else {
                    // This is a simplification - real implementation might need grouping_type-specific parsing
                    return Err(anyhow!(
                        "Version 0 with multiple entries requires grouping_type-specific parsing"
                    ));
                }
            }
            1 => {
                if let Some(desc_len) = description_length {
                    desc_len as usize
                } else if let Some(def_len) = default_length {
                    def_len as usize
                } else {
                    return Err(anyhow!("No length information available for entry {}", i));
                }
            }
            _ => {
                // Version 2+: need to determine size based on grouping_type or read remaining data
                let remaining = data.len() - cursor.position() as usize;
                if i == entry_count - 1 {
                    // Last entry gets remaining data
                    remaining
                } else {
                    // This would need grouping_type-specific parsing
                    return Err(anyhow!(
                        "Version {} requires grouping_type-specific parsing for entry size",
                        version
                    ));
                }
            }
        };

        // Read description data
        let mut description_data = vec![0u8; data_size];
        cursor
            .read_exact(&mut description_data)
            .with_context(|| format!("Failed to read description_data for entry {}", i))?;

        entries.push(SampleGroupDescriptionEntry {
            description_length,
            description_data,
        });
    }

    let atom = SampleGroupDescriptionAtom {
        version,
        flags,
        grouping_type,
        default_length,
        default_sample_description_index,
        entries,
    };

    Ok(atom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sgpd_data_v0(grouping_type: &[u8; 4], entry_data: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + 4 + 4 + 4 + entry_data.len(); // header + grouping_type + entry_count + data

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SGPD);

        // Version and flags
        data.push(0); // version 0
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Entry count
        data.extend_from_slice(&1u32.to_be_bytes());

        // Entry data
        data.extend_from_slice(entry_data);

        data
    }

    fn create_test_sgpd_data_v1(
        grouping_type: &[u8; 4],
        default_length: u32,
        entries: &[&[u8]],
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let entries_size: usize = entries.iter().map(|e| e.len()).sum();
        let total_size = 8 + 4 + 4 + 4 + 4 + entries_size; // header + grouping_type + default_length + entry_count + data

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SGPD);

        // Version and flags
        data.push(1); // version 1
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Default length
        data.extend_from_slice(&default_length.to_be_bytes());

        // Entry count
        data.extend_from_slice(&(entries.len() as u32).to_be_bytes());

        // Entries
        for entry in entries {
            if default_length == 0 {
                // Include length for each entry
                data.extend_from_slice(&(entry.len() as u32).to_be_bytes());
            }
            data.extend_from_slice(entry);
        }

        data
    }

    #[test]
    fn test_parse_sgpd_version_0() {
        let test_data = vec![1, 2, 3, 4, 5];
        let data = create_test_sgpd_data_v0(b"test", &test_data);
        let result = parse_sample_group_description_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 0);
        assert_eq!(result.grouping_type, FourCC(*b"test"));
        assert_eq!(result.default_length, None);
        assert_eq!(result.default_sample_description_index, None);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].description_data, test_data);
        assert_eq!(result.entries[0].description_length, None);
    }

    #[test]
    fn test_parse_sgpd_version_1_with_default_length() {
        let entries = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let entry_refs: Vec<&[u8]> = entries.iter().map(|e| e.as_slice()).collect();
        let data = create_test_sgpd_data_v1(b"grp1", 3, &entry_refs);
        let result = parse_sample_group_description_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 1);
        assert_eq!(result.grouping_type, FourCC(*b"grp1"));
        assert_eq!(result.default_length, Some(3));
        assert_eq!(result.default_sample_description_index, None);
        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries[0].description_data, vec![1, 2, 3]);
        assert_eq!(result.entries[1].description_data, vec![4, 5, 6]);
    }

    #[test]
    fn test_invalid_atom_type() {
        let data = create_test_sgpd_data_v0(b"test", &[1, 2, 3]);
        let mut modified_data = data;
        // Change atom type from 'sgpd' to 'badd'
        modified_data[4..8].copy_from_slice(b"badd");

        let result = parse_sample_group_description_atom(Cursor::new(&modified_data));
        assert!(result.is_err());
    }
}
