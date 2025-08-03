use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::util::{async_to_sync_read, FourCC},
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const DREF: &[u8; 4] = b"dref";

/// Data Reference Entry Types
pub mod entry_types {
    /// URL data reference
    pub const URL: &[u8; 4] = b"url ";
    /// URN data reference
    pub const URN: &[u8; 4] = b"urn ";
    /// Alias data reference (Mac OS)
    pub const ALIS: &[u8; 4] = b"alis";
}

/// Data Reference Entry Flags
pub mod flags {
    /// Media data is in the same file as the Movie Atom
    pub const SELF_CONTAINED: u32 = 0x000001;
}

#[derive(Debug, Clone)]
pub enum DataReferenceEntryInner {
    Url(String),
    Urn(String),
    Alias(Vec<u8>),
    Unknown(Vec<u8>),
}

impl DataReferenceEntryInner {
    fn new(entry_type: FourCC, data: Vec<u8>) -> Self {
        match &entry_type.0 {
            entry_types::URL => DataReferenceEntryInner::Url(String::from_utf8(data).unwrap()),
            entry_types::URN => DataReferenceEntryInner::Urn(String::from_utf8(data).unwrap()),
            entry_types::ALIS => DataReferenceEntryInner::Alias(data),
            _ => DataReferenceEntryInner::Unknown(data),
        }
    }
}

/// A single data reference entry
#[derive(Debug, Clone)]
pub struct DataReferenceEntry {
    /// Entry data/type (URL string, URN, alias data, etc.)
    pub inner: DataReferenceEntryInner,
    /// Version of the entry format
    pub version: u8,
    /// Entry flags
    pub flags: [u8; 3],
}

impl DataReferenceEntry {
    /// Check if this entry has the self-contained flag set
    pub fn is_self_contained(&self) -> bool {
        let flags_u32 = u32::from_be_bytes([0, self.flags[0], self.flags[1], self.flags[2]]);
        (flags_u32 & flags::SELF_CONTAINED) != 0
    }
}

/// Data Reference Atom (dref) - ISO/IEC 14496-12
/// Contains a table of data references that declare the location(s) of the media data
#[derive(Debug, Clone)]
pub struct DataReferenceAtom {
    /// Version of the dref atom format
    pub version: u8,
    /// Atom flags
    pub flags: [u8; 3],
    /// Number of entries in the table
    pub entry_count: u32,
    /// Data reference entries
    pub entries: Vec<DataReferenceEntry>,
}

impl Parse for DataReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != DREF {
            return Err(ParseError::new_unexpected_atom(atom_type, DREF));
        }
        parse_dref_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for DataReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*DREF)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&self.entry_count.to_be_bytes());

        // Entries
        for entry in self.entries {
            let mut entry_data = Vec::new();

            // Entry version and flags (4 bytes)
            entry_data.push(entry.version);
            entry_data.extend_from_slice(&entry.flags);

            // Entry data based on type
            let (entry_type, entry_payload) = match entry.inner {
                DataReferenceEntryInner::Url(url) => (entry_types::URL, url.into_bytes()),
                DataReferenceEntryInner::Urn(urn) => (entry_types::URN, urn.into_bytes()),
                DataReferenceEntryInner::Alias(alias_data) => (entry_types::ALIS, alias_data),
                DataReferenceEntryInner::Unknown(unknown_data) => {
                    // For unknown types, we can't determine the original type,
                    // so we'll use a generic approach
                    (b"unkn", unknown_data)
                }
            };

            entry_data.extend_from_slice(&entry_payload);

            // Calculate total entry size (4 + 4 + entry_data.len())
            let entry_size = 8 + entry_data.len();

            // Write entry size (4 bytes, big-endian)
            data.extend_from_slice(&(entry_size as u32).to_be_bytes());

            // Write entry type (4 bytes)
            data.extend_from_slice(entry_type);

            // Write entry data (version + flags + payload)
            data.extend_from_slice(&entry_data);
        }

        data
    }
}

fn parse_dref_data<R: Read>(mut reader: R) -> Result<DataReferenceAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Read entry count (4 bytes)
    let mut entry_count_bytes = [0u8; 4];
    reader
        .read_exact(&mut entry_count_bytes)
        .context("read entry count")?;
    let entry_count = u32::from_be_bytes(entry_count_bytes);

    // Read remaining data for entries
    let mut remaining_data = Vec::new();
    reader
        .read_to_end(&mut remaining_data)
        .context("read entries data")?;

    // Parse entries
    let mut entries = Vec::new();
    let mut offset = 0;

    for i in 0..entry_count {
        if offset + 8 > remaining_data.len() {
            return Err(anyhow!(
                "Incomplete entry header at entry {} (offset {})",
                i,
                offset
            ));
        }

        // Read entry size (4 bytes)
        let entry_size = u32::from_be_bytes([
            remaining_data[offset],
            remaining_data[offset + 1],
            remaining_data[offset + 2],
            remaining_data[offset + 3],
        ]) as usize;

        if entry_size < 8 {
            return Err(anyhow!("Invalid entry size: {} (minimum 8)", entry_size));
        }

        if offset + entry_size > remaining_data.len() {
            return Err(anyhow!(
                "Entry {} extends beyond data: offset={}, size={}, data_len={}",
                i,
                offset,
                entry_size,
                remaining_data.len()
            ));
        }

        // Read entry type (4 bytes)
        let entry_type = FourCC([
            remaining_data[offset + 4],
            remaining_data[offset + 5],
            remaining_data[offset + 6],
            remaining_data[offset + 7],
        ]);

        // Read entry version and flags (4 bytes)
        if offset + 12 > remaining_data.len() {
            return Err(anyhow!(
                "Incomplete entry version/flags at entry {} (offset {})",
                i,
                offset
            ));
        }

        let entry_version = remaining_data[offset + 8];
        let entry_flags = [
            remaining_data[offset + 9],
            remaining_data[offset + 10],
            remaining_data[offset + 11],
        ];

        // Read entry data (remaining bytes after 12-byte header)
        let data_start = offset + 12;
        let data_end = offset + entry_size;
        let entry_data = remaining_data[data_start..data_end].to_vec();

        entries.push(DataReferenceEntry {
            inner: DataReferenceEntryInner::new(entry_type, entry_data),
            version: entry_version,
            flags: entry_flags,
        });

        offset += entry_size;
    }

    // Verify we parsed the expected number of entries
    if entries.len() != entry_count as usize {
        return Err(anyhow!(
            "Entry count mismatch: expected {}, parsed {}",
            entry_count,
            entries.len()
        ));
    }

    Ok(DataReferenceAtom {
        version,
        flags,
        entry_count,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available dref test data files
    #[test]
    fn test_dref_roundtrip() {
        test_atom_roundtrip_sync::<DataReferenceAtom>(DREF);
    }
}
