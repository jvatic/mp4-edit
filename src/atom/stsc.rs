use anyhow::{anyhow, Context};
use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use std::{fmt, io::Read, ops::Range};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const STSC: &[u8; 4] = b"stsc";

#[derive(Default, Clone, Deref, DerefMut)]
pub struct SampleToChunkEntries(Vec<SampleToChunkEntry>);

impl SampleToChunkEntries {
    pub fn inner(&self) -> &[SampleToChunkEntry] {
        &self.0
    }
}

impl From<Vec<SampleToChunkEntry>> for SampleToChunkEntries {
    fn from(inner: Vec<SampleToChunkEntry>) -> Self {
        Self(inner)
    }
}

impl fmt::Debug for SampleToChunkEntries {
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

/// Sample-to-Chunk entry - maps samples to chunks
#[derive(Debug, Clone, PartialEq, Eq, Builder)]
pub struct SampleToChunkEntry {
    /// First chunk number (1-based) that uses this entry
    pub first_chunk: u32,
    /// Number of samples in each chunk
    pub samples_per_chunk: u32,
    /// Sample description index (1-based, references stsd atom)
    pub sample_description_index: u32,
}

/// Sample-to-Chunk Atom - contains sample-to-chunk mapping table
#[derive(Default, Debug, Clone)]
pub struct SampleToChunkAtom {
    /// Version of the stsc atom format (0)
    pub version: u8,
    /// Flags for the stsc atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample-to-chunk entries
    pub entries: SampleToChunkEntries,
}

#[bon]
impl SampleToChunkAtom {
    #[builder]
    pub fn new(
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(with = FromIterator::from_iter)] entries: Vec<SampleToChunkEntry>,
    ) -> Self {
        Self {
            version,
            flags,
            entries: entries.into(),
        }
    }

    /// Removes sample indices from the sample-to-chunk mapping table,
    /// and returns the indices of any chunks which are now empty (and should be removed)
    pub fn remove_sample_indices(
        &mut self,
        sample_indices_to_remove: &[Range<usize>],
        total_chunks: usize,
    ) -> Vec<usize> {
        if sample_indices_to_remove.is_empty() {
            return Vec::new();
        }

        let mut new_entries: Vec<SampleToChunkEntry> = Vec::with_capacity(self.entries.len());
        let mut empty_chunks = Vec::new();
        let mut sample_index = 0usize;
        let mut chunk_index = 0usize;

        let mut entries_iter = self.entries.iter_mut().peekable();
        let mut new_next_first_chunk = None;
        while let Some(entry) = entries_iter.next() {
            // since we can't modify `next_entry`,
            // we need to carry forward any updates we would have done if we could
            if let Some(new_first_chunk) = new_next_first_chunk.take() {
                entry.first_chunk = new_first_chunk;
            }

            let next_entry = entries_iter.peek();

            let chunks_in_entry = match next_entry {
                Some(next_entry) => (next_entry.first_chunk - entry.first_chunk) as usize,
                _ => total_chunks - chunk_index,
            };

            for _ in 0..chunks_in_entry {
                let chunk_start_sample = sample_index;
                let chunk_end_sample = sample_index + entry.samples_per_chunk as usize;

                // Count samples remaining in this chunk
                let remaining_samples = (chunk_start_sample..chunk_end_sample)
                    .filter(|idx| {
                        for range in sample_indices_to_remove {
                            if range.contains(idx) {
                                return false;
                            }
                        }
                        true
                    })
                    .count() as u32;

                if remaining_samples == 0 {
                    empty_chunks.push(chunk_index);
                } else if remaining_samples != entry.samples_per_chunk {
                    // Chunk has different sample count, needs new entry
                    match new_entries.last() {
                        Some(prev_entry)
                            if prev_entry.samples_per_chunk == remaining_samples
                                && prev_entry.sample_description_index
                                    == entry.sample_description_index =>
                        {
                            // we can reuse the previous entry
                        }
                        _ => match next_entry {
                            Some(next_entry)
                                if next_entry.samples_per_chunk == remaining_samples
                                    && next_entry.sample_description_index
                                        == entry.sample_description_index =>
                            {
                                // we can reuse the next entry, just need to update it's first chunk
                                // this will have to happen in the next iteration of the loop though,
                                // since we can't mut `next_entry`
                                new_next_first_chunk = Some((chunk_index + 1) as u32);
                            }
                            _ => {
                                // we need to push a new entry
                                new_entries.push(SampleToChunkEntry {
                                    first_chunk: (chunk_index + 1) as u32,
                                    samples_per_chunk: remaining_samples,
                                    sample_description_index: entry.sample_description_index,
                                });
                            }
                        },
                    }
                } else {
                    // Chunk unchanged, preserve entry
                    let entry = SampleToChunkEntry {
                        first_chunk: (chunk_index + 1) as u32,
                        samples_per_chunk: entry.samples_per_chunk,
                        sample_description_index: entry.sample_description_index,
                    };
                    match new_entries.last() {
                        Some(prev_entry)
                            if prev_entry.samples_per_chunk == entry.samples_per_chunk
                                && prev_entry.sample_description_index
                                    == entry.sample_description_index =>
                        {
                            // we can reuse the existing entry
                        }
                        _ => {
                            new_entries.push(entry);
                        }
                    }
                }

                sample_index += entry.samples_per_chunk as usize;
                chunk_index += 1;
            }
        }

        self.entries = new_entries.into();

        empty_chunks
    }
}

impl<S: sample_to_chunk_atom_builder::State> SampleToChunkAtomBuilder<S> {
    pub fn entry(
        self,
        entry: impl Into<SampleToChunkEntry>,
    ) -> SampleToChunkAtomBuilder<sample_to_chunk_atom_builder::SetEntries<S>>
    where
        S::Entries: sample_to_chunk_atom_builder::IsUnset,
    {
        self.entries(vec![entry.into()])
    }
}

impl From<Vec<SampleToChunkEntry>> for SampleToChunkAtom {
    fn from(entries: Vec<SampleToChunkEntry>) -> Self {
        SampleToChunkAtom {
            version: 0,
            flags: [0u8; 3],
            entries: entries.into(),
        }
    }
}

impl Parse for SampleToChunkAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSC {
            return Err(ParseError::new_unexpected_atom(atom_type, STSC));
        }
        let mut cursor = async_to_sync_read(reader).await?;
        parse_stsc_data(&mut cursor).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for SampleToChunkAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSC)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Entries
        for entry in self.entries.iter() {
            // First chunk (4 bytes, big-endian)
            data.extend_from_slice(&entry.first_chunk.to_be_bytes());
            // Samples per chunk (4 bytes, big-endian)
            data.extend_from_slice(&entry.samples_per_chunk.to_be_bytes());
            // Sample description index (4 bytes, big-endian)
            data.extend_from_slice(&entry.sample_description_index.to_be_bytes());
        }

        data
    }
}

fn parse_stsc_data<R: Read>(mut reader: R) -> Result<SampleToChunkAtom, anyhow::Error> {
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
        return Err(anyhow!("Too many sample-to-chunk entries: {}", entry_count));
    }

    let mut entries = Vec::with_capacity(entry_count as usize);

    for i in 0..entry_count {
        // Read first chunk (4 bytes)
        let mut first_chunk_buf = [0u8; 4];
        reader
            .read_exact(&mut first_chunk_buf)
            .context(format!("read first_chunk for entry {i}"))?;
        let first_chunk = u32::from_be_bytes(first_chunk_buf);

        // Read samples per chunk (4 bytes)
        let mut samples_per_chunk_buf = [0u8; 4];
        reader
            .read_exact(&mut samples_per_chunk_buf)
            .context(format!("read samples_per_chunk for entry {i}"))?;
        let samples_per_chunk = u32::from_be_bytes(samples_per_chunk_buf);

        // Read sample description index (4 bytes)
        let mut sample_description_index_buf = [0u8; 4];
        reader
            .read_exact(&mut sample_description_index_buf)
            .context(format!("read sample_description_index for entry {i}"))?;
        let sample_description_index = u32::from_be_bytes(sample_description_index_buf);

        // Validate entry
        if first_chunk == 0 {
            return Err(anyhow!(
                "Entry {} has zero first_chunk (should be 1-based)",
                i
            ));
        }
        if samples_per_chunk == 0 {
            return Err(anyhow!("Entry {} has zero samples_per_chunk", i));
        }
        if sample_description_index == 0 {
            return Err(anyhow!(
                "Entry {} has zero sample_description_index (should be 1-based)",
                i
            ));
        }

        entries.push(SampleToChunkEntry {
            first_chunk,
            samples_per_chunk,
            sample_description_index,
        });
    }

    Ok(SampleToChunkAtom {
        version,
        flags,
        entries: SampleToChunkEntries(entries),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stco/co64 test data files
    #[test]
    fn test_stsc_roundtrip() {
        test_atom_roundtrip_sync::<SampleToChunkAtom>(STSC);
    }
}
