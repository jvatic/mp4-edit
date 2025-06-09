use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::{parse_fixed_size_atom, FourCC};

pub const SBGP: &[u8; 4] = b"sbgp";

/// Sample-to-Group Atom (sbgp) - ISO/IEC 14496-12
/// This atom maps samples to sample groups defined in the corresponding sgpd atom.
#[derive(Debug, Clone)]
pub struct SampleToGroupAtom {
    /// Version of the sbgp atom format (0 or 1+)
    pub version: u8,
    /// Flags for the atom
    pub flags: [u8; 3],
    /// Grouping type - identifies the type of grouping (must match corresponding sgpd)
    pub grouping_type: FourCC,
    /// Grouping type parameter (version >= 1 only)
    pub grouping_type_parameter: Option<u32>,
    /// Sample-to-group mapping entries
    pub entries: Vec<SampleToGroupEntry>,
}

/// A single sample-to-group mapping entry
#[derive(Debug, Clone)]
pub struct SampleToGroupEntry {
    /// Number of consecutive samples that belong to the same group
    pub sample_count: u32,
    /// Index into the sample group description table (1-based, 0 means no group assignment)
    pub group_description_index: u32,
}

impl SampleToGroupAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_sample_to_group_atom(reader)
    }
}

impl TryFrom<&[u8]> for SampleToGroupAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_sample_to_group_atom(reader)
    }
}

fn parse_sample_to_group_atom<R: Read>(reader: R) -> Result<SampleToGroupAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;

    // Verify this is an sbgp atom
    if atom_type != SBGP {
        return Err(anyhow!(
            "Invalid atom type: {} (expected 'sbgp')",
            atom_type
        ));
    }

    parse_sbgp_data(&data)
}

fn parse_sbgp_data(data: &[u8]) -> Result<SampleToGroupAtom, anyhow::Error> {
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
    let mut grouping_type_parameter = None;
    if version >= 1 {
        cursor
            .read_exact(&mut buffer)
            .context("Failed to read grouping_type_parameter")?;
        grouping_type_parameter = Some(u32::from_be_bytes(buffer));
    }

    // Read entry_count
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read entry_count")?;
    let entry_count = u32::from_be_bytes(buffer);

    // Read entries
    let mut entries = Vec::new();
    for i in 0..entry_count {
        // Read sample_count
        cursor
            .read_exact(&mut buffer)
            .with_context(|| format!("Failed to read sample_count for entry {}", i))?;
        let sample_count = u32::from_be_bytes(buffer);

        // Read group_description_index
        cursor
            .read_exact(&mut buffer)
            .with_context(|| format!("Failed to read group_description_index for entry {}", i))?;
        let group_description_index = u32::from_be_bytes(buffer);

        entries.push(SampleToGroupEntry {
            sample_count,
            group_description_index,
        });
    }

    let atom = SampleToGroupAtom {
        version,
        flags,
        grouping_type,
        grouping_type_parameter,
        entries,
    };

    Ok(atom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sbgp_data_v0(grouping_type: &[u8; 4], entries: &[(u32, u32)]) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + 4 + 4 + 4 + (entries.len() * 8); // header + grouping_type + entry_count + entries

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SBGP);

        // Version and flags
        data.push(0); // version 0
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Entry count
        data.extend_from_slice(&(entries.len() as u32).to_be_bytes());

        // Entries
        for (sample_count, group_description_index) in entries {
            data.extend_from_slice(&sample_count.to_be_bytes());
            data.extend_from_slice(&group_description_index.to_be_bytes());
        }

        data
    }

    fn create_test_sbgp_data_v1(
        grouping_type: &[u8; 4],
        grouping_type_parameter: u32,
        entries: &[(u32, u32)],
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + 4 + 4 + 4 + 4 + (entries.len() * 8); // header + grouping_type + parameter + entry_count + entries

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SBGP);

        // Version and flags
        data.push(1); // version 1
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Grouping type parameter
        data.extend_from_slice(&grouping_type_parameter.to_be_bytes());

        // Entry count
        data.extend_from_slice(&(entries.len() as u32).to_be_bytes());

        // Entries
        for (sample_count, group_description_index) in entries {
            data.extend_from_slice(&sample_count.to_be_bytes());
            data.extend_from_slice(&group_description_index.to_be_bytes());
        }

        data
    }

    #[test]
    fn test_parse_sbgp_version_0() {
        let entries = [(10, 1), (5, 2), (8, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let result = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 0);
        assert_eq!(result.grouping_type, FourCC(*b"test"));
        assert_eq!(result.grouping_type_parameter, None);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.entries[0].sample_count, 10);
        assert_eq!(result.entries[0].group_description_index, 1);
        assert_eq!(result.entries[1].sample_count, 5);
        assert_eq!(result.entries[1].group_description_index, 2);
        assert_eq!(result.entries[2].sample_count, 8);
        assert_eq!(result.entries[2].group_description_index, 1);
    }

    #[test]
    fn test_parse_sbgp_version_1() {
        let entries = [(20, 1), (15, 0)]; // Second entry is ungrouped
        let data = create_test_sbgp_data_v1(b"grp1", 0x12345678, &entries);
        let result = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 1);
        assert_eq!(result.grouping_type, FourCC(*b"grp1"));
        assert_eq!(result.grouping_type_parameter, Some(0x12345678));
        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries[0].sample_count, 20);
        assert_eq!(result.entries[0].group_description_index, 1);
        assert_eq!(result.entries[1].sample_count, 15);
        assert_eq!(result.entries[1].group_description_index, 0);
    }

    #[test]
    fn test_invalid_atom_type() {
        let data = create_test_sbgp_data_v0(b"test", &[(1, 1)]);
        let mut modified_data = data;
        // Change atom type from 'sbgp' to 'badd'
        modified_data[4..8].copy_from_slice(b"badd");

        let result = parse_sample_to_group_atom(Cursor::new(&modified_data));
        assert!(result.is_err());
    }
}
