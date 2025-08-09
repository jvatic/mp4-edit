use anyhow::{anyhow, Context};
use bon::bon;
use derive_more::Deref;
use futures_io::AsyncRead;
use std::{
    fmt,
    io::{BufRead, BufReader, Read},
    time::Duration,
};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const CHPL: &[u8; 4] = b"chpl";

#[derive(Clone, Deref)]
pub struct ChapterEntries(Vec<ChapterEntry>);

impl ChapterEntries {
    pub fn into_vec(self) -> Vec<ChapterEntry> {
        self.0
    }
}

impl FromIterator<ChapterEntry> for ChapterEntries {
    fn from_iter<T: IntoIterator<Item = ChapterEntry>>(iter: T) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl From<Vec<ChapterEntry>> for ChapterEntries {
    fn from(entries: Vec<ChapterEntry>) -> Self {
        ChapterEntries(entries)
    }
}

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

#[bon]
impl ChapterEntry {
    #[builder]
    pub fn new(#[builder(into, finish_fn)] title: String, start_time: Duration) -> Self {
        // convert to 100-nanosecond units
        let start_time = (start_time.as_nanos() / 100).min(u64::MAX as u128) as u64;
        ChapterEntry { start_time, title }
    }
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

impl ChapterListAtom {
    pub fn new(chapters: impl Into<ChapterEntries>) -> Self {
        Self {
            version: 1,
            flags: [0u8; 3],
            chapters: chapters.into(),
        }
    }
}

impl Parse for ChapterListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != CHPL {
            return Err(ParseError::new_unexpected_atom(atom_type, CHPL));
        }
        parse_chpl_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for ChapterListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*CHPL)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Reserved field (8 bytes) - should be zero
        data.extend_from_slice(&[0u8; 8]);

        // Chapter entries
        for chapter in self.chapters.iter() {
            // Start time (8 bytes, big-endian)
            data.extend_from_slice(&chapter.start_time.to_be_bytes());

            // Title as UTF-8 string with null terminator
            data.extend_from_slice(chapter.title.as_bytes());
            data.push(0x00); // Null terminator
        }

        data
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available chpl test data files
    #[test]
    fn test_chpl_roundtrip() {
        test_atom_roundtrip_sync::<ChapterListAtom>(CHPL);
    }
}
