use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::{Cursor, Read};

use crate::{
    atom::util::{async_to_sync_read, FourCC},
    parser::Parse,
};

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

impl Parse for SampleGroupDescriptionAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != SGPD {
            return Err(anyhow!(
                "Invalid atom type: {} (expected 'sgpd')",
                atom_type
            ));
        }
        let cursor = async_to_sync_read(reader).await?;
        parse_sgpd_data(cursor.get_ref())
    }
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
