use anyhow::{anyhow, Context};
use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};
use either::Either;
use futures_io::AsyncRead;
use std::{fmt, io::Read, iter::Enumerate};

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
        #[builder(field = Vec::new())] entries: Vec<SampleToChunkEntry>,
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(setters(vis = ""), overwritable)]
        #[allow(unused)]
        entries_marker: bool,
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
        sample_indices_to_remove: &[usize],
        total_chunks: usize,
    ) -> Vec<usize> {
        let (next_entries, chunk_indices_to_remove) =
            SampleToChunkIter::new(self.entries.as_slice(), total_chunks)
                .remove_sample_indices(sample_indices_to_remove)
                .collapse();
        self.entries = next_entries;

        chunk_indices_to_remove
    }

    /// Calculates how many chunks need to be removed for the given number of samples
    /// and updates the chunk mapping accordingly
    ///
    /// `total_chunks` - the total number of chunks in the stco/co64 atom
    pub fn trim_samples_from_end(&mut self, samples_to_remove: u32, total_chunks: usize) -> u32 {
        let mut chunk_samples = self.expand_chunk_samples(total_chunks);
        let chunks_removed = self.trim_samples(samples_to_remove, chunk_samples.iter_mut().rev());

        // Remove empty chunks from our mapping
        chunk_samples.retain(|&samples| samples > 0);

        // Rebuild stsc entries from the remaining chunks
        self.entries = self.collapse_chunk_samples(&chunk_samples).into();

        chunks_removed
    }

    fn trim_samples<'a>(
        &mut self,
        samples_to_remove: u32,
        chunk_samples: impl Iterator<Item = &'a mut u32>,
    ) -> u32 {
        let mut remaining_samples_to_remove = samples_to_remove;
        let mut chunks_removed = 0u32;

        for samples_in_chunk in chunk_samples {
            if remaining_samples_to_remove == 0 {
                break;
            }

            if remaining_samples_to_remove >= *samples_in_chunk {
                // Remove entire chunk
                remaining_samples_to_remove -= *samples_in_chunk;
                *samples_in_chunk = 0; // Mark as removed
                chunks_removed += 1;
            } else {
                // Partial removal from this chunk
                *samples_in_chunk -= remaining_samples_to_remove;
                break;
            }
        }

        chunks_removed
    }

    fn expand_chunk_samples(&mut self, total_chunks: usize) -> Vec<u32> {
        // First, create a complete mapping of chunk -> samples_per_chunk
        let mut chunk_samples = vec![0u32; total_chunks];

        // Fill in the samples per chunk based on stsc entries
        for (i, entry) in self.entries.iter().enumerate() {
            let start_chunk = (entry.first_chunk - 1) as usize; // Convert to 0-based indexing
            let end_chunk = if i + 1 < self.entries.len() {
                (self.entries[i + 1].first_chunk - 1) as usize
            } else {
                total_chunks
            };

            // Fill the range with this entry's samples_per_chunk
            for chunk in chunk_samples.iter_mut().take(end_chunk).skip(start_chunk) {
                *chunk = entry.samples_per_chunk;
            }
        }

        chunk_samples
    }

    fn collapse_chunk_samples(&self, chunk_samples: &[u32]) -> Vec<SampleToChunkEntry> {
        if chunk_samples.is_empty() {
            return Vec::new();
        }

        let mut entries = Vec::new();
        let mut current_samples_per_chunk = chunk_samples[0];
        let mut current_first_chunk = 1u32;

        for (chunk_idx, &samples_per_chunk) in chunk_samples.iter().enumerate().skip(1) {
            if samples_per_chunk != current_samples_per_chunk {
                // Different sample count - need a new entry
                entries.push(SampleToChunkEntry {
                    first_chunk: current_first_chunk,
                    samples_per_chunk: current_samples_per_chunk,
                    sample_description_index: 1, // TODO: preserve this round-trip
                });

                current_first_chunk = (chunk_idx as u32) + 1;
                current_samples_per_chunk = samples_per_chunk;
            }
        }

        // Don't forget the last group
        if !chunk_samples.is_empty() {
            entries.push(SampleToChunkEntry {
                first_chunk: current_first_chunk,
                samples_per_chunk: current_samples_per_chunk,
                sample_description_index: 1, // TODO: preserve this round-trip
            });
        }

        entries
    }
}

impl<S: sample_to_chunk_atom_builder::State> SampleToChunkAtomBuilder<S> {
    fn push_entry(
        mut self,
        entry: SampleToChunkEntry,
    ) -> SampleToChunkAtomBuilder<sample_to_chunk_atom_builder::SetEntriesMarker<S>> {
        self.entries.push(entry);
        self.entries_marker(true)
    }

    pub fn entry(
        self,
        entry: impl Into<SampleToChunkEntry>,
    ) -> SampleToChunkAtomBuilder<sample_to_chunk_atom_builder::SetEntriesMarker<S>> {
        self.push_entry(entry.into())
    }

    pub fn entries(
        mut self,
        entries: impl Into<Vec<SampleToChunkEntry>>,
    ) -> SampleToChunkAtomBuilder<sample_to_chunk_atom_builder::SetEntriesMarker<S>>
    where
        S::EntriesMarker: sample_to_chunk_atom_builder::IsUnset,
    {
        self.entries = entries.into();
        self.entries_marker(true)
    }
}

struct SampleToChunkIterItem {
    pub num_samples: u32,
    pub sample_description_index: u32,
}

/// Iterates over each chunk's sample count.
struct SampleToChunkIter<'a> {
    entries: &'a [SampleToChunkEntry],
    /// index of current entry (before calling `next`)
    index: usize,
    /// index of current chunk relative to current entry
    chunk_index: usize,
    /// total number of chunks
    total_chunks: usize,
}

impl<'a> SampleToChunkIter<'a> {
    fn new(entries: impl Into<&'a [SampleToChunkEntry]>, total_chunks: usize) -> Self {
        Self {
            entries: entries.into(),
            index: 0,
            chunk_index: 0,
            total_chunks,
        }
    }

    fn remove_sample_indices<'b>(
        self,
        sample_indices: &'b [usize],
    ) -> SampleToChunkRemoveSamplesIter<'b, Self> {
        SampleToChunkRemoveSamplesIter::new(self, sample_indices)
    }
}

impl<'a> Iterator for SampleToChunkIter<'a> {
    type Item = SampleToChunkIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.entries.get(self.index)?;

        // calculate the end chunk index covered by this entry
        let end_chunk_index = match self.entries.get(self.index + 1) {
            Some(next_entry) => (next_entry.first_chunk - 1) as usize,
            None => self.total_chunks,
        };

        // current chunk is not contained in this entry
        if self.chunk_index > end_chunk_index {
            self.index += 1;
            return self.next();
        }

        self.chunk_index += 1;
        Some(SampleToChunkIterItem {
            num_samples: entry.samples_per_chunk,
            sample_description_index: entry.sample_description_index,
        })
    }
}

/// Iterates over each chunk's sample count, yielding either a chunk index to remove, or a modified sample count to keep
struct SampleToChunkRemoveSamplesIter<'a, I> {
    sample_indices: &'a [usize],
    inner: Enumerate<I>,
    /// index of sample each chunk entry starts from
    sample_offset: usize,
}

impl<'a, I> SampleToChunkRemoveSamplesIter<'a, I>
where
    I: Iterator<Item = SampleToChunkIterItem>,
{
    fn new(inner: I, sample_indices: &'a [usize]) -> Self {
        Self {
            sample_indices,
            inner: inner.enumerate(),
            sample_offset: 0,
        }
    }

    /// compresses chunk sample counts back into [`SampleToChunkEntry`]s and a list of removed chunk indices
    fn collapse(self) -> (SampleToChunkEntries, Vec<usize>) {
        self.enumerate().fold(
            (SampleToChunkEntries(Vec::new()), Vec::new()),
            |(mut entries, mut indices), (chunk_index, item)| {
                match item {
                    Either::Left(chunk) => match entries.last_mut() {
                        Some(entry)
                            if entry.samples_per_chunk == chunk.num_samples
                                && entry.sample_description_index
                                    == chunk.sample_description_index =>
                        {
                            // chunk is covered by the current entry
                        }
                        _ => {
                            entries.push(SampleToChunkEntry {
                                first_chunk: chunk_index as u32 + 1,
                                samples_per_chunk: chunk.num_samples,
                                sample_description_index: chunk.sample_description_index,
                            });
                        }
                    },
                    Either::Right(chunk_index) => {
                        indices.push(chunk_index);
                    }
                }
                (entries, indices)
            },
        )
    }
}

impl<'a, I> Iterator for SampleToChunkRemoveSamplesIter<'a, I>
where
    I: Iterator<Item = SampleToChunkIterItem>,
{
    type Item = Either<SampleToChunkIterItem, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk_index, mut chunk) = self.inner.next()?;

        let start_sample_index = self.sample_offset;
        let end_sample_index = start_sample_index + chunk.num_samples as usize - 1;
        self.sample_offset += chunk.num_samples as usize;

        let num_samples_removed = self
            .sample_indices
            .iter()
            .take_while(|&&sample_index| {
                start_sample_index >= sample_index && sample_index <= end_sample_index
            })
            .take(chunk.num_samples as usize)
            .count();

        chunk.num_samples -= num_samples_removed as u32;
        self.sample_indices = &self.sample_indices[num_samples_removed..];

        if chunk.num_samples == 0 {
            // all the samples have been removed from this chunk
            return Some(Either::Right(chunk_index));
        }
        Some(Either::Left(chunk))
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
