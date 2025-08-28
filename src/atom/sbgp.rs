use anyhow::Context;
use futures_io::AsyncRead;
use std::io::{Cursor, Read};

use crate::{
    atom::util::{async_to_sync_read, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

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

impl ParseAtom for SampleToGroupAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != SBGP {
            return Err(ParseError::new_unexpected_atom(atom_type, SBGP));
        }
        let cursor = async_to_sync_read(reader).await?;
        parse_sbgp_data(cursor.get_ref()).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for SampleToGroupAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*SBGP)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Grouping type (4 bytes)
        data.extend_from_slice(&self.grouping_type.0);

        // Grouping type parameter (4 bytes) - version >= 1 only
        if let Some(param) = self.grouping_type_parameter {
            data.extend_from_slice(&param.to_be_bytes());
        }

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(
            &(u32::try_from(self.entries.len()).expect("entries len should fit in u32"))
                .to_be_bytes(),
        );

        // Entries
        for entry in self.entries {
            // Sample count (4 bytes, big-endian)
            data.extend_from_slice(&entry.sample_count.to_be_bytes());
            // Group description index (4 bytes, big-endian)
            data.extend_from_slice(&entry.group_description_index.to_be_bytes());
        }

        data
    }
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
            .with_context(|| format!("Failed to read sample_count for entry {i}"))?;
        let sample_count = u32::from_be_bytes(buffer);

        // Read group_description_index
        cursor
            .read_exact(&mut buffer)
            .with_context(|| format!("Failed to read group_description_index for entry {i}"))?;
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
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available sbgp test data files
    #[test]
    fn test_sbgp_roundtrip() {
        test_atom_roundtrip_sync::<SampleToGroupAtom>(SBGP);
    }
}
