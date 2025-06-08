use anyhow::{anyhow, Context};
use derive_more::Deref;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const STCO: &[u8; 4] = b"stco";
pub const CO64: &[u8; 4] = b"co64";

#[derive(Clone, Deref)]
pub struct ChunkOffsets(Vec<u64>);

impl fmt::Debug for ChunkOffsets {
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

/// Chunk Offset Atom - contains file offsets of chunks
#[derive(Debug, Clone)]
pub struct ChunkOffsetAtom {
    /// Version of the stco atom format (0)
    pub version: u8,
    /// Flags for the stco atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of chunk offsets
    pub chunk_offsets: ChunkOffsets,
    /// Whether this uses 64-bit offsets (co64) or 32-bit (stco)
    pub is_64bit: bool,
}

impl ChunkOffsetAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_chunk_offset_atom(reader)
    }

    /// Get the number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunk_offsets.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.chunk_offsets.is_empty()
    }

    /// Get the offset of a specific chunk by index
    pub fn get_chunk_offset(&self, index: usize) -> Option<u64> {
        self.chunk_offsets.get(index).copied()
    }

    /// Get the first chunk offset (most common case)
    pub fn first_chunk_offset(&self) -> Option<u64> {
        self.chunk_offsets.first().copied()
    }

    /// Get the last chunk offset
    pub fn last_chunk_offset(&self) -> Option<u64> {
        self.chunk_offsets.last().copied()
    }

    /// Get all chunk offsets as a slice
    pub fn offsets(&self) -> &[u64] {
        &self.chunk_offsets
    }

    /// Check if any chunk offset exceeds 32-bit range
    pub fn requires_64bit(&self) -> bool {
        self.chunk_offsets
            .iter()
            .any(|&offset| offset > u32::MAX as u64)
    }

    /// Get the range of file positions covered by chunks
    pub fn offset_range(&self) -> Option<(u64, u64)> {
        if self.chunk_offsets.is_empty() {
            return None;
        }

        let min = *self.chunk_offsets.iter().min()?;
        let max = *self.chunk_offsets.iter().max()?;
        Some((min, max))
    }

    /// Find the chunk index for a given file offset
    /// Returns the index of the chunk that contains or comes after the given offset
    pub fn find_chunk_for_offset(&self, target_offset: u64) -> Option<usize> {
        self.chunk_offsets
            .iter()
            .position(|&offset| offset >= target_offset)
    }

    /// Get chunk offsets within a specific range
    pub fn offsets_in_range(&self, start: u64, end: u64) -> Vec<(usize, u64)> {
        self.chunk_offsets
            .iter()
            .enumerate()
            .filter_map(|(i, &offset)| {
                if offset >= start && offset <= end {
                    Some((i, offset))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Validate that all offsets are in ascending order (which they should be)
    pub fn is_sorted(&self) -> bool {
        self.chunk_offsets.windows(2).all(|w| w[0] <= w[1])
    }

    /// Get statistics about the chunk distribution
    pub fn chunk_statistics(&self) -> ChunkStatistics {
        if self.chunk_offsets.is_empty() {
            return ChunkStatistics::default();
        }

        let min = *self.chunk_offsets.iter().min().unwrap();
        let max = *self.chunk_offsets.iter().max().unwrap();
        let total: u64 = self.chunk_offsets.iter().sum();
        let average = total / self.chunk_offsets.len() as u64;

        // Calculate gaps between consecutive chunks
        let mut gaps = Vec::new();
        for window in self.chunk_offsets.windows(2) {
            if window[1] > window[0] {
                gaps.push(window[1] - window[0]);
            }
        }

        let avg_gap = if gaps.is_empty() {
            0
        } else {
            gaps.iter().sum::<u64>() / gaps.len() as u64
        };

        ChunkStatistics {
            count: self.chunk_offsets.len(),
            min_offset: min,
            max_offset: max,
            average_offset: average,
            average_gap: avg_gap,
            is_sorted: self.is_sorted(),
        }
    }
}

/// Statistics about chunk offset distribution
#[derive(Debug, Clone, Default)]
pub struct ChunkStatistics {
    /// Total number of chunks
    pub count: usize,
    /// Minimum chunk offset
    pub min_offset: u64,
    /// Maximum chunk offset
    pub max_offset: u64,
    /// Average chunk offset
    pub average_offset: u64,
    /// Average gap between consecutive chunks
    pub average_gap: u64,
    /// Whether offsets are properly sorted
    pub is_sorted: bool,
}

impl TryFrom<&[u8]> for ChunkOffsetAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_chunk_offset_atom(reader)
    }
}

fn parse_chunk_offset_atom<R: Read>(reader: R) -> Result<ChunkOffsetAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;

    let is_64bit = if atom_type == STCO {
        false
    } else if atom_type == CO64 {
        true
    } else {
        return Err(anyhow!(
            "Invalid atom type: {} (expected stco or co64)",
            atom_type
        ));
    };

    let mut cursor = Cursor::new(data);
    parse_stco_data(&mut cursor, is_64bit)
}

fn parse_stco_data<R: Read>(
    mut reader: R,
    is_64bit: bool,
) -> Result<ChunkOffsetAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read entry count
    let mut count_buf = [0u8; 4];
    reader
        .read_exact(&mut count_buf)
        .context("read entry count")?;
    let entry_count = u32::from_be_bytes(count_buf);

    // Validate entry count (reasonable limit to prevent memory exhaustion)
    if entry_count > 1_000_000 {
        return Err(anyhow!("Too many chunk offsets: {}", entry_count));
    }

    let mut chunk_offsets = Vec::with_capacity(entry_count as usize);

    if is_64bit {
        // Read 64-bit offsets (co64)
        for i in 0..entry_count {
            let mut offset_buf = [0u8; 8];
            reader
                .read_exact(&mut offset_buf)
                .context(format!("read chunk offset {}", i))?;
            let offset = u64::from_be_bytes(offset_buf);
            chunk_offsets.push(offset);
        }
    } else {
        // Read 32-bit offsets (stco)
        for i in 0..entry_count {
            let mut offset_buf = [0u8; 4];
            reader
                .read_exact(&mut offset_buf)
                .context(format!("read chunk offset {}", i))?;
            let offset = u32::from_be_bytes(offset_buf) as u64;
            chunk_offsets.push(offset);
        }
    }

    Ok(ChunkOffsetAtom {
        version,
        flags,
        chunk_offsets: ChunkOffsets(chunk_offsets),
        is_64bit,
    })
}
