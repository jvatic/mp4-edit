use anyhow::{anyhow, Context};
use derive_more::Deref;
use std::{
    fmt::{self},
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const STSZ: &[u8; 4] = b"stsz";

#[derive(Clone, Deref)]
pub struct SampleEntrySizes(Vec<u32>);

impl fmt::Debug for SampleEntrySizes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
    }
}

/// Sample Size Atom (stsz) - ISO/IEC 14496-12
/// This atom contains the sample count and a table giving the size in bytes of each sample.
/// Samples within the media may have different sizes, up to the limit of a 32-bit integer.
#[derive(Debug, Clone)]
pub struct SampleSizeAtom {
    /// Version of this atom (0 or 1)
    pub version: u8,
    /// Flags (24 bits)
    pub flags: u32,
    /// If this field is set to some value other than 0, then it gives the (constant) size
    /// of every sample in the track. If this field is set to 0, then the samples have
    /// different sizes, and those sizes are stored in the sample size table.
    pub sample_size: u32,
    /// Number of samples in the track
    pub sample_count: u32,
    /// If sample_size is 0, this contains the size of each sample, indexed by sample number.
    /// If sample_size is non-zero, this table is empty.
    pub entry_sizes: SampleEntrySizes,
}

impl SampleSizeAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_stsz_atom(reader)
    }
}

impl TryFrom<&[u8]> for SampleSizeAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_stsz_atom(reader)
    }
}

fn parse_stsz_atom<R: Read>(reader: R) -> Result<SampleSizeAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != STSZ {
        return Err(anyhow!(
            "Invalid atom type: expected stsz, got {}",
            atom_type
        ));
    }

    let mut cursor = Cursor::new(data);
    parse_stsz_data(&mut cursor)
}

fn parse_stsz_data<R: Read>(mut reader: R) -> Result<SampleSizeAtom, anyhow::Error> {
    // Read all data into buffer for easier parsing
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).context("reading stsz data")?;

    // Minimum size check: version/flags (4) + sample_size (4) + sample_count (4) = 12 bytes
    if buf.len() < 12 {
        return Err(anyhow!("stsz atom too small: {} bytes", buf.len()));
    }

    // Version and flags (4 bytes total)
    let version_flags = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let version = (version_flags >> 24) as u8;
    let flags = version_flags & 0x00FFFFFF;

    // Validate version
    if version != 0 {
        return Err(anyhow!("Unsupported stsz version: {}", version));
    }

    // Sample size (4 bytes)
    let sample_size = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);

    // Sample count (4 bytes)
    let sample_count = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

    let mut entry_sizes = Vec::new();

    if sample_size == 0 {
        // Variable sample sizes - read the table
        let expected_table_size = sample_count as usize * 4; // 4 bytes per entry
        let remaining_bytes = &buf[12..];

        if remaining_bytes.len() != expected_table_size {
            return Err(anyhow!(
                "Invalid stsz atom: expected {} bytes for sample size table, got {}",
                expected_table_size,
                remaining_bytes.len()
            ));
        }

        entry_sizes.reserve(sample_count as usize);
        for chunk in remaining_bytes.chunks_exact(4) {
            let size = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            entry_sizes.push(size);
        }
    } else {
        // Constant sample size - no table needed
        if buf.len() != 12 {
            return Err(anyhow!(
                "Invalid stsz atom: constant sample size specified but extra data present"
            ));
        }
    }

    Ok(SampleSizeAtom {
        version,
        flags,
        sample_size,
        sample_count,
        entry_sizes: SampleEntrySizes(entry_sizes),
    })
}

impl fmt::Display for SampleSizeAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SampleSize(count: {}, ", self.sample_count)?;

        if self.sample_size != 0 {
            write!(f, "constant_size: {})", self.sample_size)
        } else {
            write!(f, "variable_sizes: {} entries)", self.entry_sizes.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_size() {
        // Test with data too small
        let mut data = Vec::new();
        data.extend_from_slice(&16u32.to_be_bytes()); // 16 bytes total
        data.extend_from_slice(b"stsz");
        data.extend_from_slice(&0u32.to_be_bytes()); // Version/flags
        data.extend_from_slice(&1024u32.to_be_bytes()); // Sample size
                                                        // Missing sample count

        let result = SampleSizeAtom::parse(Cursor::new(data));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_table_size() {
        // Test with incorrect table size for variable samples
        let mut data = Vec::new();
        data.extend_from_slice(&24u32.to_be_bytes()); // 24 bytes total
        data.extend_from_slice(b"stsz");
        data.extend_from_slice(&0u32.to_be_bytes()); // Version/flags
        data.extend_from_slice(&0u32.to_be_bytes()); // Sample size (0 = variable)
        data.extend_from_slice(&2u32.to_be_bytes()); // Sample count (2)
        data.extend_from_slice(&512u32.to_be_bytes()); // Only one entry instead of two

        let result = SampleSizeAtom::parse(Cursor::new(data));
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_version() {
        // Test with unsupported version
        let mut data = Vec::new();
        data.extend_from_slice(&20u32.to_be_bytes());
        data.extend_from_slice(b"stsz");
        data.extend_from_slice(&0x01000000u32.to_be_bytes()); // Version 1 (unsupported)
        data.extend_from_slice(&1024u32.to_be_bytes());
        data.extend_from_slice(&100u32.to_be_bytes());

        let result = SampleSizeAtom::parse(Cursor::new(data));
        assert!(result.is_err());
    }
}
