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
