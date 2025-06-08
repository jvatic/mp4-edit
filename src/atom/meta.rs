use anyhow::{anyhow, Context};
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const META: &[u8; 4] = b"meta";

#[derive(Clone)]
pub struct MetadataAtom {
    /// Version of the meta atom format (usually 0)
    pub version: u8,
    /// Flags for the meta atom (usually all zeros)
    pub flags: [u8; 3],
    /// Raw child atom data (contains various metadata atoms)
    pub child_data: Vec<u8>,
}

impl fmt::Debug for MetadataAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetadataAtom")
            .field("version", &self.version)
            .field("flags", &self.flags)
            .field("child_data", &DebugEllipsis(Some(self.child_data.len())))
            .finish()
    }
}

impl MetadataAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_meta_atom(reader)
    }

    /// Get summary information about the metadata
    pub fn get_summary(&self) -> MetadataSummary {
        MetadataSummary {
            version: self.version,
            flags: self.flags,
            child_data_size: self.child_data.len(),
        }
    }

    /// Validate the meta atom structure
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        // Version should typically be 0
        if self.version > 1 {
            return Err(anyhow!(
                "Unusual version: {} (expected 0 or 1)",
                self.version
            ));
        }

        Ok(())
    }

    /// Parse pitm (primary item) data
    #[allow(dead_code)]
    fn parse_pitm_data(&self, data: &[u8]) -> Result<u32, anyhow::Error> {
        if data.len() < 6 {
            return Err(anyhow!("pitm data too short"));
        }

        let version = data[0];
        // flags are data[1..4]

        match version {
            0 => {
                if data.len() < 6 {
                    return Err(anyhow!("pitm v0 data too short"));
                }
                Ok(u16::from_be_bytes([data[4], data[5]]) as u32)
            }
            1 => {
                if data.len() < 8 {
                    return Err(anyhow!("pitm v1 data too short"));
                }
                Ok(u32::from_be_bytes([data[4], data[5], data[6], data[7]]))
            }
            _ => Err(anyhow!("Unsupported pitm version: {}", version)),
        }
    }

    /// Create a new MetadataAtom with default values
    pub fn new() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            child_data: Vec::new(),
        }
    }

    /// Create a new MetadataAtom with a specific handler type
    pub fn new_with_handler(_handler_type: [u8; 4]) -> Self {
        // This would create a minimal meta atom with just an hdlr atom
        // Implementation would be more complex for actual use
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetadataSummary {
    pub version: u8,
    pub flags: [u8; 3],
    pub child_data_size: usize,
}

impl Default for MetadataAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<&[u8]> for MetadataAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_meta_atom(reader)
    }
}

fn parse_meta_atom<R: Read>(reader: R) -> Result<MetadataAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != META {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_meta_data(&mut cursor)
}

fn parse_meta_data<R: Read>(mut reader: R) -> Result<MetadataAtom, anyhow::Error> {
    // Read version and flags (4 bytes total)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Read remaining data (contains child atoms)
    let mut child_data = Vec::new();
    reader
        .read_to_end(&mut child_data)
        .context("read child data")?;

    Ok(MetadataAtom {
        version,
        flags,
        child_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_meta_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags
        data.push(0); // version
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Add raw child data (without parsing individual atoms)
        data.extend_from_slice(b"some child atom data");

        data
    }

    #[test]
    fn test_parse_meta_data() {
        let data = create_test_meta_data();
        let result = parse_meta_data(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 0);
        assert_eq!(result.flags, [0, 0, 0]);
        assert!(!result.child_data.is_empty());
    }

    #[test]
    fn test_default_meta() {
        let meta = MetadataAtom::default();
        assert_eq!(meta.version, 0);
        assert_eq!(meta.flags, [0, 0, 0]);
        assert!(meta.child_data.is_empty());
    }

    #[test]
    fn test_validation() {
        let data = create_test_meta_data();
        let meta = parse_meta_data(Cursor::new(&data)).unwrap();

        assert!(meta.validate().is_ok());

        // Test default meta
        let meta_default = MetadataAtom::new();
        assert!(meta_default.validate().is_ok());
    }

    #[test]
    fn test_summary() {
        let data = create_test_meta_data();
        let meta = parse_meta_data(Cursor::new(&data)).unwrap();

        let summary = meta.get_summary();
        assert_eq!(summary.version, 0);
        assert!(summary.child_data_size > 0);
    }

    #[test]
    fn test_pitm_parsing() {
        let meta = MetadataAtom::new();

        // Test pitm version 0 data
        let pitm_v0_data = vec![0, 0, 0, 0, 0x12, 0x34]; // version 0, flags, item_id=0x1234
        assert_eq!(meta.parse_pitm_data(&pitm_v0_data).unwrap(), 0x1234);

        // Test pitm version 1 data
        let pitm_v1_data = vec![1, 0, 0, 0, 0x12, 0x34, 0x56, 0x78]; // version 1, flags, item_id=0x12345678
        assert_eq!(meta.parse_pitm_data(&pitm_v1_data).unwrap(), 0x12345678);
    }
}
