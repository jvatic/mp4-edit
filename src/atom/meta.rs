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
}
