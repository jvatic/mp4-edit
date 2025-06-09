use anyhow::{anyhow, Context};
use derive_more::Deref;
use futures_io::AsyncRead;
use std::{
    fmt,
    io::{BufRead, BufReader, Cursor, Read},
};

use crate::{
    atom::util::{parse_fixed_size_atom, DebugEllipsis},
    parser::Parse,
};

pub const CHPL: &[u8; 4] = b"chpl";

#[derive(Clone, Deref)]
pub struct ChapterEntries(Vec<ChapterEntry>);

impl fmt::Debug for ChapterEntries {
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

/// Chapter entry containing start time and title
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterEntry {
    /// Start time of the chapter in 100-nanosecond units
    pub start_time: u64,
    /// Chapter title as UTF-8 string
    pub title: String,
}

/// Chapter List Atom - contains chapter information for media
#[derive(Debug, Clone)]
pub struct ChapterListAtom {
    /// Version of the chpl atom format (1)
    pub version: u8,
    /// Flags for the chpl atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of chapter entries
    pub chapters: ChapterEntries,
}

impl Parse for ChapterListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(reader: R) -> Result<Self, anyhow::Error> {
        let (atom_type, data) = parse_fixed_size_atom(reader).await?;
        if atom_type != CHPL {
            return Err(anyhow!("Invalid atom type: {} (expected chpl)", atom_type));
        }

        // Parse the data using existing sync function
        let cursor = Cursor::new(data);
        parse_chpl_data(cursor)
    }
}

fn parse_chpl_data<R: Read>(mut reader: R) -> Result<ChapterListAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version (chpl typically uses version 1)
    if version != 1 {
        return Err(anyhow!("unsupported chpl version {}", version));
    }

    // Read reserved field (4 bytes) - should be zero
    let mut reserved = [0u8; 8];
    reader
        .read_exact(&mut reserved)
        .context("read reserved field")?;

    let mut chapters = Vec::new();

    let mut reader = BufReader::new(reader);

    // Read start time (8 bytes, big-endian)
    let mut start_time_buf = [0u8; 8];
    let mut i = -1;
    while let Ok(()) = reader.read_exact(&mut start_time_buf) {
        i += 1;
        let start_time = u64::from_be_bytes(start_time_buf);

        let mut title_buf = Vec::new();
        reader
            .read_until(0x00, &mut title_buf)
            .context(format!("read title for chapter {}", i))?;
        title_buf.pop(); // Remove trailing null byte

        let title = String::from_utf8(title_buf)
            .context(format!("invalid UTF-8 in chapter {} title", i))?;

        chapters.push(ChapterEntry { start_time, title });
    }

    Ok(ChapterListAtom {
        version,
        flags,
        chapters: ChapterEntries(chapters),
    })
}
